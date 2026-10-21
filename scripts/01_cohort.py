"""Stage 1 — cohort construction.

Four operationalisations of "first ICU admission at first hospitalization" are built side by side
(A-D, see results/decisions.md); the primary cohort is definition C. The subset DuckDB database is
built for the union of candidates so every definition can be evaluated for CBC completeness.

Outputs (patient-level, git-ignored): data/interim/cohort_candidates.parquet, data/processed/cohort.parquet
Outputs (aggregate): results/tables/cohort_flow.csv, results/tables/outcome_definitions.csv,
                     results/tables/cbc_missingness.csv, results/tables/dn_codes.csv
Only aggregate numbers are printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import connect, create_raw_views, dn_codes, load_config, path, sql_list
from src.subset_db import build_subset_db, open_subset

cfg = load_config()["cohort"]
PRIMARY = "C"
DEFS = {
    "A": "patient's first-ever hospitalization contains a DN code and an ICU stay; its first ICU stay",
    "B": "patient's first DN-coded hospitalization, required to contain an ICU stay; its first ICU stay",
    "C": "patient's first hospitalization that has both a DN code and an ICU stay; its first ICU stay",
    "D": "patient's first-ever ICU stay, whose hospitalization contains a DN code (old notebook)",
}
TABLES = path("tables")
TABLES.mkdir(parents=True, exist_ok=True)

con = connect()
create_raw_views(con, [("hosp", "patients"), ("hosp", "admissions"), ("icu", "icustays"),
                       ("hosp", "diagnoses_icd"), ("hosp", "d_icd_diagnoses")])

# ---------------------------------------------------------------- DN diagnoses
icd9, icd10 = sql_list(cfg["dn_icd9"]), sql_list(cfg["dn_icd10"])
con.execute(f"""
CREATE TEMP TABLE dx AS
SELECT subject_id, hadm_id, icd_version, upper(trim(replace(icd_code, '.', ''))) AS code
FROM raw.diagnoses_icd
WHERE upper(trim(replace(icd_code, '.', ''))) IN ({sql_list(dn_codes())})
""")
mismatch = con.execute(f"""
SELECT count(*) FROM dx WHERE NOT ((icd_version = 9 AND code IN ({icd9})) OR (icd_version = 10 AND code IN ({icd10})))
""").fetchone()[0]
con.execute(f"""
CREATE TEMP TABLE dn_adm AS
SELECT DISTINCT subject_id, hadm_id FROM dx
WHERE (icd_version = 9 AND code IN ({icd9})) OR (icd_version = 10 AND code IN ({icd10}))
""")
codes = con.execute("""
SELECT dx.code, dx.icd_version, any_value(d.long_title) AS long_title,
       count(DISTINCT dx.hadm_id) AS n_admissions, count(DISTINCT dx.subject_id) AS n_patients
FROM dx LEFT JOIN raw.d_icd_diagnoses d ON d.icd_code = dx.code AND d.icd_version = dx.icd_version
GROUP BY ALL ORDER BY icd_version, code
""").df()
codes.to_csv(TABLES / "dn_codes.csv", index=False)
print(f"DN code rows with version mismatch (excluded): {mismatch}")
print(codes[["code", "icd_version", "n_admissions", "n_patients"]].to_string(index=False))

# ---------------------------------------------------------------- episode definitions
con.execute("""
CREATE TEMP TABLE adm AS
SELECT subject_id, hadm_id, admittime, dischtime, deathtime, hospital_expire_flag,
       date_diff('second', admittime, dischtime) / 3600.0 AS hosp_los_h,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS adm_seq
FROM raw.admissions;
CREATE TEMP TABLE icu AS
SELECT subject_id, hadm_id, stay_id, intime, outtime, first_careunit,
       row_number() OVER (PARTITION BY hadm_id ORDER BY intime, stay_id) AS icu_seq_in_adm,
       row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) AS icu_seq_subject
FROM raw.icustays;
CREATE TEMP TABLE adm_flags AS
SELECT a.*, (d.hadm_id IS NOT NULL) AS has_dn,
       EXISTS (SELECT 1 FROM icu i WHERE i.hadm_id = a.hadm_id) AS has_icu
FROM adm a LEFT JOIN dn_adm d USING (subject_id, hadm_id);
""")

episode_sql = {
    "A": "SELECT subject_id, hadm_id FROM adm_flags WHERE adm_seq = 1 AND has_dn AND has_icu",
    "B": """SELECT subject_id, hadm_id FROM (
              SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) r
              FROM adm_flags WHERE has_dn) WHERE r = 1 AND has_icu""",
    "C": """SELECT subject_id, hadm_id FROM (
              SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) r
              FROM adm_flags WHERE has_dn AND has_icu) WHERE r = 1""",
    "D": """SELECT i.subject_id, i.hadm_id FROM icu i JOIN dn_adm d USING (subject_id, hadm_id)
            WHERE i.icu_seq_subject = 1""",
}
parts = []
for k, q in episode_sql.items():
    parts.append(con.execute(f"""
    SELECT '{k}' AS definition, e.subject_id, e.hadm_id, i.stay_id, i.intime, i.outtime, i.first_careunit,
           a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag, a.hosp_los_h, a.adm_seq,
           p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group, p.dod,
           p.anchor_age + year(a.admittime) - p.anchor_year AS age
    FROM ({q}) e
    JOIN icu i ON i.hadm_id = e.hadm_id AND i.icu_seq_in_adm = 1
    JOIN adm a ON a.hadm_id = e.hadm_id
    JOIN raw.patients p ON p.subject_id = e.subject_id
    """).df())
cand = pd.concat(parts, ignore_index=True)
cand["los_ok"] = cand["hosp_los_h"] > cfg["min_hosp_los_hours"]
cand["age_ok"] = cand["age"] > cfg["min_age_replication"]          # strict > 18 (paper, D1.4)
cand["intime_in_adm"] = (cand["intime"] >= cand["admittime"]) & (cand["intime"] <= cand["dischtime"])

# ---------------------------------------------------------------- outcomes
H = cfg["outcome_horizon_days"]
dod = pd.to_datetime(cand["dod"])
days_from_icu = (dod - cand["intime"].dt.normalize()).dt.days
days_from_adm = (dod - cand["admittime"].dt.normalize()).dt.days
inhosp = cand["hospital_expire_flag"].eq(1)
# primary (D8.10): death date in [date(intime), date(intime) + H], date granularity; index-admission deathtime for
# in-hospital deaths, otherwise dod (also when hospital_expire_flag = 1 but deathtime is missing)
d_icu = cand["intime"].dt.normalize()
dtime = pd.to_datetime(cand["deathtime"]).dt.normalize()
death_date = dtime.where(inhosp & dtime.notna(), dod.dt.normalize())
cand["death_1y_icu"] = (death_date - d_icu).dt.days.between(0, H).astype(int)          # primary
cand["death_1y_icu_old"] = (inhosp | (days_from_icu <= H)).astype(int)                 # D1.6, kept for the check
cand["death_1y_adm"] = (inhosp | (days_from_adm <= H)).astype(int)
cand["death_inhosp"] = inhosp.astype(int)
cand["death_any_dod"] = (inhosp | dod.notna()).astype(int)                 # hypothesis for the paper's 38%
cand["days_icu_to_dod"] = days_from_icu

# ---------------------------------------------------------------- subset DB + CBC completeness
pre_cbc = cand[cand["los_ok"] & cand["age_ok"]]
# clinical line (CLIN, D8.1): index admission as definition C, age >= 18, no LOS condition -> also in the subset DB
clin_cand = cand[(cand["definition"] == "C") & (cand["age"] >= cfg["min_age_clinical"])]
db_subjects = pd.concat([pre_cbc["subject_id"], clin_cand["subject_id"]]).drop_duplicates()
print(f"\nBuilding subset DB for {db_subjects.nunique()} candidate patients (union of A-D after LOS/age: "
      f"{pre_cbc['subject_id'].nunique()}; CLIN candidates: {clin_cand['subject_id'].nunique()})...", flush=True)
build_subset_db(db_subjects, log=lambda m: print("  " + m, flush=True))

sub = open_subset()
stays = pre_cbc[["subject_id", "stay_id", "intime"]].drop_duplicates("stay_id")
sub.register("stays_df", stays)
pre, post = cfg["lab_window_pre_hours"], cfg["lab_window_post_hours"]
win = f"x.charttime >= s.intime - INTERVAL {pre} HOUR AND x.charttime <= s.intime + INTERVAL {post} HOUR"
cbc = sub.execute(f"""
WITH c AS (
  SELECT s.stay_id, count(x.hemoglobin) hb, count(x.platelet) plt, count(x.rbc) rbc, count(x.rdw) rdw, count(x.wbc) wbc_c
  FROM stays_df s LEFT JOIN mimiciv_derived.complete_blood_count x ON x.subject_id = s.subject_id AND {win}
  GROUP BY s.stay_id),
d AS (
  SELECT s.stay_id, count(x.wbc) wbc_d, count(x.lymphocytes_abs) lym, count(x.neutrophils_abs) neu, count(x.monocytes_abs) mono
  FROM stays_df s LEFT JOIN mimiciv_derived.blood_differential x ON x.subject_id = s.subject_id AND {win}
  GROUP BY s.stay_id)
SELECT c.stay_id, hb > 0 AS has_hb, plt > 0 AS has_plt, rbc > 0 AS has_rbc, rdw > 0 AS has_rdw,
       (wbc_c + wbc_d) > 0 AS has_wbc, lym > 0 AS has_lym, neu > 0 AS has_neu, mono > 0 AS has_mono
FROM c JOIN d USING (stay_id)
""").df()
sub.close()
cbc_items = ["has_wbc", "has_hb", "has_plt", "has_rbc", "has_rdw", "has_lym", "has_neu", "has_mono"]
cbc["cbc_complete"] = cbc[cbc_items].all(axis=1)
cand = cand.merge(cbc, on="stay_id", how="left")
cand[cbc_items + ["cbc_complete"]] = cand[cbc_items + ["cbc_complete"]].fillna(False).astype(bool)

# ---------------------------------------------------------------- flow + outcome tables (aggregate)
flow_rows, out_rows, miss_rows = [], [], []
for k in DEFS:
    d = cand[cand["definition"] == k]
    _f = d[d.los_ok & d.age_ok & d.cbc_complete]
    assert (_f.death_1y_icu == _f.death_1y_icu_old).all(), f"definition {k}: new and old 1-year outcome differ (D8.10)"
    steps = [
        ("[2] first ICU at first hospitalization (definition)", d),
        ("[3] hospital LOS > 48h", d[d.los_ok]),
        ("[4] age > 18", d[d.los_ok & d.age_ok]),
        ("[5] complete blood count (8 items) in first-day window", d[d.los_ok & d.age_ok & d.cbc_complete]),
    ]
    for name, x in steps:
        flow_rows.append({"definition": k, "step": name, "n": len(x),
                          "deaths_1y_icu": int(x.death_1y_icu.sum()),
                          "mortality_1y_icu": round(x.death_1y_icu.mean(), 4) if len(x) else None})
    final = steps[-1][1]
    for o in ["death_1y_icu", "death_1y_adm", "death_inhosp", "death_any_dod"]:
        out_rows.append({"definition": k, "outcome": o, "n": len(final), "deaths": int(final[o].sum()),
                         "rate": round(final[o].mean(), 4)})
    base = steps[2][1]
    for it in cbc_items:
        miss_rows.append({"definition": k, "item": it.replace("has_", ""), "n_before_cbc_step": len(base),
                          "missing_pct": round(100 * (1 - base[it].mean()), 1)})

n_dn_pat = con.execute("SELECT count(DISTINCT subject_id) FROM dn_adm").fetchone()[0]
n_dn_adm = con.execute("SELECT count(*) FROM dn_adm").fetchone()[0]
n_dn_icu_pat = con.execute("SELECT count(DISTINCT subject_id) FROM adm_flags WHERE has_dn AND has_icu").fetchone()[0]
flow = pd.DataFrame(flow_rows)
head = pd.DataFrame([
    {"definition": "all", "step": "[1] patients with a DN code (any admission)", "n": n_dn_pat},
    {"definition": "all", "step": "[1a] DN-coded admissions", "n": n_dn_adm},
    {"definition": "all", "step": "[1b] patients with a DN-coded admission that has an ICU stay", "n": n_dn_icu_pat},
])
flow = pd.concat([head, flow], ignore_index=True)
flow.to_csv(TABLES / "cohort_flow.csv", index=False)
outcomes = pd.DataFrame(out_rows)
outcomes.to_csv(TABLES / "outcome_definitions.csv", index=False)
miss = pd.DataFrame(miss_rows)
miss.to_csv(TABLES / "cbc_missingness.csv", index=False)

print("\n=== cohort flow ===")
print(flow.to_string(index=False))
print("\n=== outcome definitions (final cohorts) ===")
print(outcomes.pivot(index="outcome", columns="definition", values="rate").to_string())
print("\n=== CBC item missing % before step [5] ===")
print(miss.pivot(index="item", columns="definition", values="missing_pct").to_string())

# ---------------------------------------------------------------- diagnostics for the primary cohort
prim_all = cand[cand["definition"] == PRIMARY]
prim = prim_all[prim_all.los_ok & prim_all.age_ok & prim_all.cbc_complete].copy()
print(f"\n=== primary cohort (definition {PRIMARY}) diagnostics ===")
print("ICU intime outside admission window:", int((~prim.intime_in_adm).sum()))
print("dod before admission date:", int((pd.to_datetime(prim.dod) < prim.admittime.dt.normalize()).sum()))
print("in-hospital death flag without dod:", int((prim.hospital_expire_flag.eq(1) & prim.dod.isna()).sum()))
print("index admission is patient's first admission (share):", round((prim.adm_seq == 1).mean(), 3))
alive_dis = prim[prim.hospital_expire_flag.eq(0) & prim.dod.notna()]
print("post-discharge dod: n =", len(alive_dis), "; days ICU->death quantiles:",
      alive_dis.days_icu_to_dod.quantile([0.25, 0.5, 0.75, 0.9]).round(0).tolist(),
      "; share > 365 d:", round((alive_dis.days_icu_to_dod > H).mean(), 3))
print("anchor_year_group distribution:")
print(prim.anchor_year_group.value_counts().sort_index().to_string())
print("age at admission describe:", prim.age.describe().round(1).to_dict())

# ---------------------------------------------------------------- save
out_interim = path("interim")
cand.to_parquet(out_interim / "cohort_candidates.parquet", index=False)
keep = ["subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "intime", "outtime", "first_careunit",
        "hosp_los_h", "age", "gender", "anchor_year_group", "hospital_expire_flag", "dod",
        "death_1y_icu", "death_1y_adm", "death_inhosp", "death_any_dod"]
prim = prim[keep].rename(columns={"death_1y_icu": "y"})
prim.to_parquet(path("processed") / "cohort.parquet", index=False)
print(f"\nsaved primary cohort: n={len(prim)}, deaths={int(prim.y.sum())}, mortality={prim.y.mean():.3f}")
