"""Stage 8d step 2 — comorbidity-timing sensitivity features for cohort C (decisions.md D8.2, D8.3).

Recomputes, for every cohort-C stay:
  cci_sens    : mimic-code Charlson index (age score kept) on  prior admissions (all codes)
                ∪ index admission minus the acute-event codes of D8.3
  sapsii_sens : sapsii − original comorbidity_score + comorbidity_score recomputed (same mimic-code AIDS /
                haematological malignancy / metastatic cancer code lists) on prior ∪ index admission (all codes;
                the three items are chronic, D8.2)
Prior admission = same subject, dischtime < index admittime. The unmodified mimic-code SQL (charlson.sql, the
`comorb` CTE of sapsii.sql) is re-run on a re-labelled diagnosis table (every code mapped to the index hadm_id);
re-running it on index-admission codes only must reproduce the original values exactly (asserted).

Checks first (aggregate counts only): share of cohort C with charlson diabetes_with_cc = 1; the subset database's
admissions / diagnoses_icd rows for the cohort-C subjects equal those of the raw files.
Outputs: data/processed/features_comorb_time.parquet (patient-level, git-ignored);
         results/replication/comorb_time_summary_C.csv, comorb_time_flags_C.csv (aggregates; counts 1–9 shown as "<10").

--track clinical (stage 8e-1, CLIN cohort, D8.2/D8.3): same rules on data/processed/cohort_clin.parquet; additionally the
6 model comorbidities (charlson MI / CHF / chronic pulmonary / renal; Quan-Elixhauser hypertension / obesity as in
02_features.py) on prior ∪ index-minus-acute codes, CCI WITHOUT the age score, SAPS-II with the recomputed chronic-disease
item, has_prior_admission, and the D8.3 586/N19 count (renal positive only through index-admission 586/N19).
Also the prior-admissions-only versions (*_prior: the 6 comorbidities, cci_prior without age score, sapsii_prior) for the
D8.3 sensitivity analysis (all 0 without a prior admission).
Outputs: data/processed/comorb_clin.parquet; results/clinical/comorb_clin_summary.csv, comorb_clin_flags.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import ROOT, path, read_csv_expr
from src.subset_db import open_subset

ap = argparse.ArgumentParser()
ap.add_argument("--track", choices=["replication", "clinical"], default="replication")
TRACK = ap.parse_args().track
CLIN = TRACK == "clinical"
LBL = "CLIN" if CLIN else "cohort-C"
OUT = ROOT / "results" / TRACK
OUT.mkdir(parents=True, exist_ok=True)
CONCEPTS = path("mimic_code") / "mimic-iv/concepts_duckdb"

# ---------------------------------------------------------------- D8.3: acute-event codes removed from the INDEX admission
# (codes in MIMIC carry no dots; prefix match within each ICD version)
ACUTE9 = (["410"] +                                                   # acute MI
          ["42821", "42831", "42841"] +                               # acute (not acute-on-chronic) heart failure
          ["586"] +                                                   # renal failure, unspecified (often AKI)
          [str(c) for c in range(430, 437)] + ["36234"] +             # acute stroke / TIA, retinal artery occlusion
          ["4410"] +                                                  # aortic dissection
          [f"{a}{d}" for a in range(531, 535) for d in range(0, 4)] +  # acute peptic ulcer (4th digit 0–3)
          ["570"])                                                    # acute / subacute liver necrosis
ACUTE10 = (["I21", "I22"] +
           ["I5021", "I5031", "I5041", "I50811"] +
           ["N19"] +
           [f"I6{d}" for d in range(0, 5)] + ["G45", "G46", "H340"] +
           ["I710"] +
           [f"K2{a}{d}" for a in range(5, 9) for d in range(0, 4)] +
           ["K762"])


def like_any(col: str, prefixes) -> str:
    return "(" + " OR ".join(f"{col} LIKE '{p}%'" for p in prefixes) + ")"


ACUTE_SQL = (f"((icd_version = 9 AND {like_any('code', ACUTE9)}) OR "
             f"(icd_version = 10 AND {like_any('code', ACUTE10)}))")


def small(n: int):
    return "<10" if 0 < n < 10 else int(n)


# ---------------------------------------------------------------- data
cohort = pd.read_parquet(path("processed") / ("cohort_clin.parquet" if CLIN else "cohort.parquet"))[
    ["subject_id", "hadm_id", "stay_id", "admittime"]]
assert cohort.stay_id.is_unique and cohort.hadm_id.is_unique
if not CLIN:
    feat = pd.read_parquet(path("processed") / "features.parquet")[["stay_id", "cci", "sapsii"]]
    assert len(cohort) == 1960

con = open_subset()
con.execute("SET threads=1")
con.register("idx_df", cohort)
con.execute("CREATE TEMP TABLE idx AS SELECT * FROM idx_df")

# ---- check 1: diabetes_with_cc share
dm = con.execute("""SELECT count(*) AS n, sum(c.diabetes_with_cc) AS n_dm_cc
                    FROM idx i LEFT JOIN mimiciv_derived.charlson c USING (hadm_id)""").fetchone()
# DN codes 249.6x (secondary diabetes with neurological manifestations) are not in charlson's diabetes lists
dm_249 = con.execute("""SELECT count(*) FROM idx i JOIN mimiciv_derived.charlson c USING (hadm_id)
                        WHERE c.diabetes_with_cc = 0 AND EXISTS (SELECT 1 FROM mimiciv_hosp.diagnoses_icd d
                          WHERE d.hadm_id = i.hadm_id AND d.icd_version = 9 AND d.icd_code LIKE '2496%')""").fetchone()[0]
n_not = dm[0] - dm[1]
print(f"[check] charlson diabetes_with_cc = 0: {small(n_not)} of {dm[0]}; of these with a 249.6x code: {small(dm_249)}"
      f" -> {'all explained by 249.6x' if dm_249 == n_not else 'NOT all explained'}")

# ---- check 2: the subset database holds every admission / diagnosis row of the cohort-C subjects
comp = {}
for t in ["admissions", "diagnoses_icd"]:
    sub = con.execute(f"SELECT count(*) FROM mimiciv_hosp.{t} WHERE subject_id IN (SELECT subject_id FROM idx)").fetchone()[0]
    raw = con.execute(f"SELECT count(*) FROM {read_csv_expr('hosp', t)} WHERE subject_id IN (SELECT subject_id FROM idx)").fetchone()[0]
    comp[t] = (sub, raw)
    print(f"[check] {t} rows for {LBL} subjects: subset {sub:,} | raw {raw:,}")
    assert sub == raw, f"subset database is missing {t} rows"

# ---- diagnosis sources, all re-labelled to the index hadm_id
con.execute("""
CREATE TEMP TABLE prior_adm AS
SELECT i.hadm_id AS index_hadm_id, a.hadm_id
FROM idx i JOIN mimiciv_hosp.admissions a ON a.subject_id = i.subject_id AND a.dischtime < i.admittime
""")
con.execute(f"""
CREATE TEMP TABLE dx_src AS
SELECT i.hadm_id AS index_hadm_id, 'index' AS src, d.icd_version, d.icd_code,
       upper(trim(replace(d.icd_code, '.', ''))) AS code
FROM idx i JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
UNION ALL
SELECT p.index_hadm_id, 'prior', d.icd_version, d.icd_code, upper(trim(replace(d.icd_code, '.', '')))
FROM prior_adm p JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = p.hadm_id
""")
con.execute(f"ALTER TABLE dx_src ADD COLUMN acute BOOLEAN")
con.execute(f"UPDATE dx_src SET acute = {ACUTE_SQL}")
views = {   # name -> rows fed to the mimic-code SQL
    "dx_index_all": "src = 'index'",                                  # reproduces the original concepts
    "dx_cci_sens": "src = 'prior' OR (src = 'index' AND NOT acute)",  # D8.3 rule
    "dx_index_chronic": "src = 'index' AND NOT acute",                # decomposition: acute exclusion only
    "dx_saps_sens": "TRUE",                                           # prior ∪ index, all codes
}
for v, w in views.items():
    con.execute(f"""CREATE TEMP TABLE {v} AS SELECT index_hadm_id AS hadm_id, icd_version, code AS icd_code
                    FROM dx_src WHERE {w}""")
con.execute("CREATE TEMP TABLE adm_idx AS SELECT a.* FROM mimiciv_hosp.admissions a SEMI JOIN idx USING (hadm_id)")

# ---- mimic-code charlson.sql on a given diagnosis table
charlson_sql = (CONCEPTS / "comorbidity/charlson.sql").read_text()
head = "DROP TABLE IF EXISTS mimiciv_derived.charlson; CREATE TABLE mimiciv_derived.charlson AS"
assert charlson_sql.count(head) == 1 and charlson_sql.count("FROM mimiciv_hosp.diagnoses_icd") == 1 \
    and charlson_sql.count("FROM mimiciv_hosp.admissions") == 2


def charlson_on(dx: str, name: str):
    sql = (charlson_sql.replace(head, f"CREATE TEMP TABLE {name} AS")
           .replace("FROM mimiciv_hosp.diagnoses_icd", f"FROM {dx}")
           .replace("FROM mimiciv_hosp.admissions", "FROM adm_idx"))
    con.execute(sql)
    return con.execute(f"SELECT * FROM {name}").df()


# ---- the `comorb` CTE of sapsii.sql on a given diagnosis table
saps_sql = (CONCEPTS / "score/sapsii.sql").read_text()
m = re.search(r"comorb AS \((.*?FROM mimiciv_hosp\.diagnoses_icd\s+GROUP BY\s+hadm_id\s*)\)", saps_sql, re.S)
assert m, "comorb CTE not found in sapsii.sql"
comorb_body = m.group(1)
assert "WHEN aids = 1 THEN 17 WHEN hem = 1 THEN 10 WHEN mets = 1 THEN 9 ELSE 0 END AS comorbidity_score" in saps_sql


def saps_comorb_on(dx: str) -> pd.DataFrame:
    body = comorb_body.replace("FROM mimiciv_hosp.diagnoses_icd", f"FROM {dx}")
    return con.execute(f"""
        WITH comorb AS ({body})
        SELECT i.hadm_id, coalesce(c.aids, 0) AS aids, coalesce(c.hem, 0) AS hem, coalesce(c.mets, 0) AS mets,
               CASE WHEN c.aids = 1 THEN 17 WHEN c.hem = 1 THEN 10 WHEN c.mets = 1 THEN 9 ELSE 0 END AS comorbidity_score
        FROM idx i LEFT JOIN comorb c USING (hadm_id)""").df()


ch_orig = con.execute("SELECT c.* FROM mimiciv_derived.charlson c SEMI JOIN idx USING (hadm_id)").df()
ch_idx = charlson_on("dx_index_all", "ch_index_all")
ch_chr = charlson_on("dx_index_chronic", "ch_index_chronic")
ch_sens = charlson_on("dx_cci_sens", "ch_sens")
sa_orig = con.execute("""SELECT s.stay_id, s.sapsii, s.comorbidity_score FROM mimiciv_derived.sapsii s
                         SEMI JOIN idx USING (stay_id)""").df()
sc_idx, sc_sens = saps_comorb_on("dx_index_all"), saps_comorb_on("dx_saps_sens")
n_prior = con.execute("SELECT count(DISTINCT index_hadm_id) FROM prior_adm").fetchone()[0]
n_prior_dx = con.execute("SELECT count(DISTINCT index_hadm_id) FROM dx_src WHERE src = 'prior'").fetchone()[0]
n_acute_idx = con.execute("SELECT count(DISTINCT index_hadm_id) FROM dx_src WHERE src = 'index' AND acute").fetchone()[0]
# 586 / N19 in the index admission vs other renal codes (why nobody loses renal_disease by the acute exclusion)
renal = con.execute("""
WITH r AS (
  SELECT index_hadm_id,
    max(CASE WHEN (icd_version = 9 AND code LIKE '586%') OR (icd_version = 10 AND code LIKE 'N19%') THEN 1 ELSE 0 END) AS u,
    max(CASE WHEN (icd_version = 9 AND code LIKE '585%') OR (icd_version = 10 AND code LIKE 'N18%') THEN 1 ELSE 0 END) AS ckd
  FROM dx_src WHERE src = 'index' GROUP BY index_hadm_id)
SELECT r.ckd, c.renal_disease AS other_renal, count(*) AS n
FROM r JOIN ch_index_chronic c ON c.hadm_id = r.index_hadm_id
WHERE r.u = 1 GROUP BY ALL""").df()
u_where = "(icd_version = 9 AND icd_code LIKE '586%') OR (icd_version = 10 AND icd_code LIKE 'N19%')"
n_u_rows_subset = con.execute(f"""SELECT count(*) FROM mimiciv_hosp.diagnoses_icd
    WHERE subject_id IN (SELECT subject_id FROM idx) AND ({u_where})""").fetchone()[0]
n_u_rows_raw_all = con.execute(f"SELECT count(*) FROM {read_csv_expr('hosp', 'diagnoses_icd')} WHERE {u_where}").fetchone()[0]
print(f"[check] 586/N19 diagnosis rows: cohort subjects, any admission {small(n_u_rows_subset)}; "
      f"whole MIMIC-IV raw diagnoses_icd {n_u_rows_raw_all:,}")
if CLIN:
    # hypertension / obesity: Quan-Elixhauser codes of 02_features.py, on prior ∪ index-minus-acute codes (D8.3: no
    # acute exclusion applies to them); ch_all: charlson on prior ∪ index with ALL codes (for the 586/N19 count)
    def icd_flag(p9, p10):
        return (f"max(CASE WHEN (icd_version = 9 AND {like_any('icd_code', p9)}) OR "
                f"(icd_version = 10 AND {like_any('icd_code', p10)}) THEN 1 ELSE 0 END)")
    def elix_on(dx: str) -> pd.DataFrame:
        return con.execute(f"""
        SELECT i.hadm_id,
          coalesce({icd_flag(('401', '402', '403', '404', '405'), ('I10', 'I11', 'I12', 'I13', 'I15', 'I16'))}, 0) AS hypertension,
          coalesce({icd_flag(('401',), ('I10',))}, 0) AS hypertension_uncomplicated,
          coalesce({icd_flag(('2780',), ('E66',))}, 0) AS obesity
        FROM idx i LEFT JOIN {dx} d USING (hadm_id) GROUP BY i.hadm_id""").df()
    elix = elix_on("dx_cci_sens")
    ch_all = charlson_on("dx_saps_sens", "ch_all")
    # D8.3 sensitivity analysis "prior admissions only" (+ has_prior_admission): all comorbidities 0 without a prior
    # admission (the charlson / comorb SQL left-joins the index admission, so missing codes give 0)
    con.execute("""CREATE TEMP TABLE dx_prior AS SELECT index_hadm_id AS hadm_id, icd_version, code AS icd_code
                   FROM dx_src WHERE src = 'prior'""")
    elix_p, ch_p, sc_p = elix_on("dx_prior"), charlson_on("dx_prior", "ch_prior"), saps_comorb_on("dx_prior")
    prior_set = set(con.execute("SELECT DISTINCT index_hadm_id FROM prior_adm").df().index_hadm_id)
con.close()
n_u = int(renal.n.sum())
n_u_other = int(renal.loc[renal.other_renal == 1, "n"].sum())
n_u_ckd = int(renal.loc[renal.ckd == 1, "n"].sum())

# ---------------------------------------------------------------- reproduce the original concepts (exact)
COMP = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease", "cerebrovascular_disease",
        "dementia", "chronic_pulmonary_disease", "rheumatic_disease", "peptic_ulcer_disease", "mild_liver_disease",
        "diabetes_without_cc", "diabetes_with_cc", "paraplegia", "renal_disease", "malignant_cancer",
        "severe_liver_disease", "metastatic_solid_tumor", "aids"]
key = lambda d: d.set_index("hadm_id").loc[cohort.hadm_id]
o, r = key(ch_orig), key(ch_idx)
assert (o[COMP + ["age_score", "charlson_comorbidity_index"]].fillna(-1).values
        == r[COMP + ["age_score", "charlson_comorbidity_index"]].fillna(-1).values).all(), "charlson not reproduced"
sa = sa_orig.set_index("stay_id").loc[cohort.stay_id]
assert (sa.comorbidity_score.values == key(sc_idx).comorbidity_score.values).all(), "SAPS-II comorbidity not reproduced"
if not CLIN:
    f = feat.set_index("stay_id").loc[cohort.stay_id]
    assert np.allclose(f.cci.values, o.charlson_comorbidity_index.values, equal_nan=True)
    assert np.allclose(f.sapsii.values, sa.sapsii.values, equal_nan=True)
print("[check] index-only re-run reproduces charlson (17 components, age score, CCI) and SAPS-II comorbidity_score: OK")

# ---------------------------------------------------------------- sensitivity values
s, c = key(ch_sens), key(ch_chr)
cci_sens = s.charlson_comorbidity_index.values.astype(float)
sapsii_sens = sa.sapsii.values - sa.comorbidity_score.values + key(sc_sens).comorbidity_score.values
if not CLIN:
    out = pd.DataFrame({"stay_id": cohort.stay_id.values, "cci_sens": cci_sens, "sapsii_sens": sapsii_sens.astype(float)})
    out.to_parquet(path("processed") / "features_comorb_time.parquet", index=False)
else:
    e, a_ = key(elix), key(ch_all)
    assert s.age_score.notna().all()
    has_prior = cohort.hadm_id.isin(prior_set).astype(int).values
    renal_only_586 = ((a_.renal_disease.values == 1) & (s.renal_disease.values == 0))
    out = pd.DataFrame({
        "stay_id": cohort.stay_id.values,
        **{k: s[k].values.astype(int) for k in ["myocardial_infarct", "congestive_heart_failure",
                                                 "chronic_pulmonary_disease", "renal_disease"]},
        **{k: e[k].values.astype(int) for k in ["hypertension", "hypertension_uncomplicated", "obesity"]},
        "cci": (s.charlson_comorbidity_index - s.age_score).values.astype(float),       # no age score (D8.2)
        "sapsii": sapsii_sens.astype(float),
        "has_prior_admission": has_prior,
        "diabetes_with_cc_index": o.diabetes_with_cc.values.astype(int),
    })
    sp, ep = key(ch_p), key(elix_p)
    assert sp.age_score.notna().all()
    for k in ["myocardial_infarct", "congestive_heart_failure", "chronic_pulmonary_disease", "renal_disease"]:
        out[f"{k}_prior"] = sp[k].values.astype(int)
    for k in ["hypertension", "obesity"]:
        out[f"{k}_prior"] = ep[k].values.astype(int)
    out["cci_prior"] = (sp.charlson_comorbidity_index - sp.age_score).values.astype(float)
    out["sapsii_prior"] = (sa.sapsii.values - sa.comorbidity_score.values + key(sc_p).comorbidity_score.values).astype(float)
    nop = out.has_prior_admission == 0
    flags_prior = [c for c in out.columns if c.endswith("_prior") and c not in ("cci_prior", "sapsii_prior")]
    assert (out.loc[nop, flags_prior] == 0).all().all() and (out.loc[nop, "cci_prior"] == 0).all()
    out.to_parquet(path("processed") / "comorb_clin.parquet", index=False)


def med_iqr(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.0f} ({x.quantile(.25):.0f}, {x.quantile(.75):.0f})"


cci0 = o.charlson_comorbidity_index.values.astype(float)
saps0 = sa.sapsii.values.astype(float)
if CLIN:      # the comparison CCI must also exclude the age score
    cci0, cci_sens = cci0 - o.age_score.values, out.cci.values
summary = [
    {"item": f"{'CLIN' if CLIN else 'cohort C'} stays", "value": len(cohort)},
    {"item": "charlson diabetes_with_cc = 0 (index admission)", "value": small(n_not)},
    {"item": "  of which explained by a 249.6x DN code (not in charlson diabetes lists)", "value": small(dm_249)},
    {"item": f"subset admissions rows = raw ({LBL} subjects)", "value": f"{comp['admissions'][0]} = {comp['admissions'][1]}"},
    {"item": f"subset diagnoses_icd rows = raw ({LBL} subjects)", "value": f"{comp['diagnoses_icd'][0]} = {comp['diagnoses_icd'][1]}"},
    {"item": "patients with >= 1 prior admission", "value": small(n_prior)},
    {"item": "patients with >= 1 prior diagnosis code", "value": small(n_prior_dx)},
    {"item": "patients with >= 1 D8.3 acute code in the index admission", "value": small(n_acute_idx)},
    {"item": "index admission with 586/N19", "value": small(n_u)},
    {"item": "  of which with another charlson renal code in the index admission", "value": small(n_u_other)},
    {"item": "  of which with 585.x/N18.x (CKD) in the index admission", "value": small(n_u_ckd)},
    {"item": f"586/N19 diagnosis rows, {LBL} subjects, any admission", "value": small(n_u_rows_subset)},
    {"item": "586/N19 diagnosis rows, whole MIMIC-IV v3.1 (raw diagnoses_icd)", "value": small(n_u_rows_raw_all)},
    {"item": "CCI original, median (IQR)", "value": med_iqr(cci0)},
    {"item": "CCI sensitivity, median (IQR)", "value": med_iqr(cci_sens)},
    {"item": "CCI changed (n)", "value": small(int((cci_sens != cci0).sum()))},
    {"item": "CCI increased (n)", "value": small(int((cci_sens > cci0).sum()))},
    {"item": "CCI decreased (n)", "value": small(int((cci_sens < cci0).sum()))},
    {"item": "CCI change, mean", "value": round(float(np.mean(cci_sens - cci0)), 3)},
    {"item": "SAPS-II original, median (IQR)", "value": med_iqr(saps0)},
    {"item": "SAPS-II sensitivity, median (IQR)", "value": med_iqr(sapsii_sens)},
    {"item": "SAPS-II changed (n)", "value": small(int((sapsii_sens != saps0).sum()))},
    {"item": "SAPS-II change, mean", "value": round(float(np.mean(sapsii_sens - saps0)), 3)},
]
if CLIN:
    nu = int(renal_only_586.sum())
    summary = [r_ for r_ in summary if not r_["item"].startswith(("CCI ", "SAPS-II "))] + [
        {"item": "CCI (no age score), index admission only, median (IQR)", "value": med_iqr(cci0)},
        {"item": "CCI (no age score), CLIN rule, median (IQR)", "value": med_iqr(cci_sens)},
        {"item": "CCI changed vs index-only (n)", "value": small(int((cci_sens != cci0).sum()))},
        {"item": "SAPS-II original, median (IQR)", "value": med_iqr(saps0)},
        {"item": "SAPS-II CLIN rule, median (IQR)", "value": med_iqr(sapsii_sens)},
        {"item": "SAPS-II changed (n)", "value": small(int((sapsii_sens != saps0).sum()))},
        {"item": "has_prior_admission = 1 (n)", "value": small(int(out.has_prior_admission.sum()))},
        {"item": "has_prior_admission = 1 (%)", "value": round(100 * float(out.has_prior_admission.mean()), 1)},
        {"item": "charlson diabetes_with_cc = 1, index admission (%)", "value": round(100 * float(o.diabetes_with_cc.mean()), 2)},
        {"item": "charlson diabetes_with_cc = 1, CLIN rule (%)", "value": round(100 * float(s.diabetes_with_cc.mean()), 2)},
        {"item": "D8.3: renal_disease only through index-admission 586/N19 (n)", "value": small(nu)},
        {"item": "D8.3: renal_disease only through index-admission 586/N19 (% of CLIN)", "value": round(100 * nu / len(cohort), 2)},
        {"item": "D8.3: >= 1% of CLIN -> 586/N19 add-back sensitivity analysis required", "value": bool(nu >= 0.01 * len(cohort))},
    ]
    for k in ["hypertension", "obesity"]:
        flags_k = (k, int(out[k].sum()))
        summary.append({"item": f"{k} = 1, CLIN rule (n)", "value": small(flags_k[1])})
flags = []
for comp_ in COMP:
    a0, ac, a1 = o[comp_].values, c[comp_].values, s[comp_].values
    flags.append({"component": comp_, "original_n": small(int(a0.sum())), "sensitivity_n": small(int(a1.sum())),
                  "lost_by_acute_exclusion": small(int(((a0 == 1) & (ac == 0)).sum())),
                  "lost_net": small(int(((a0 == 1) & (a1 == 0)).sum())),
                  "new_from_prior_admissions": small(int(((ac == 0) & (a1 == 1)).sum())),
                  "new_vs_original": small(int(((a0 == 0) & (a1 == 1)).sum()))})
si, ss = key(sc_idx), key(sc_sens)
for comp_ in ["aids", "hem", "mets"]:
    flags.append({"component": f"sapsii_{comp_}", "original_n": small(int(si[comp_].sum())),
                  "sensitivity_n": small(int(ss[comp_].sum())), "lost_by_acute_exclusion": 0, "lost_net": 0,
                  "new_from_prior_admissions": small(int(((si[comp_] == 0) & (ss[comp_] == 1)).sum())),
                  "new_vs_original": small(int(((si[comp_] == 0) & (ss[comp_] == 1)).sum()))})
stem = "comorb_clin" if CLIN else "comorb_time"
sfx = "" if CLIN else "_C"
pd.DataFrame(summary).to_csv(OUT / f"{stem}_summary{sfx}.csv", index=False)
pd.DataFrame(flags).to_csv(OUT / f"{stem}_flags{sfx}.csv", index=False)

pd.set_option("display.width", 220)
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(flags).to_string(index=False))
