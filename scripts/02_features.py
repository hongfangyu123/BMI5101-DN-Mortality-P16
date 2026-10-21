"""Stage 2 — feature extraction (55 predictors, paper 'Data extraction' section).

Sources (subset DuckDB database, mimic-code concepts):
  labs  min/max  : mimiciv_derived.first_day_lab           (window intime-6h .. intime+24h, by subject_id)
  labs  mean     : complete_blood_count / blood_differential / chemistry, same window
  vitals mean    : mimiciv_derived.first_day_vitalsign
  scores         : sapsii, apsiii, lods, oasis, first_day_gcs (gcs_min), charlson (index incl. age score)
  comorbidities  : charlson flags (MI, CHF, chronic pulmonary, renal); hypertension/obesity by ICD (Quan-Elixhauser)
Outputs: data/processed/features.parquet (patient-level, git-ignored) and aggregate tables in results/tables.

--track clinical (stage 8e-1, CLIN, D8.2–D8.4): same windows and definitions on data/processed/cohort_clin.parquet (no
CBC-completeness requirement); the 6 comorbidities, CCI (no age score) and SAPS-II are replaced by the CLIN versions
from scripts/02b_comorb_time.py --track clinical (data/processed/comorb_clin.parquet); adds wbc_diff_missing (all three
differential cell counts missing) and has_prior_admission. No filtering or imputation here (done inside the nested-CV
folds). Outputs: data/processed/features_clin.parquet; results/clinical/missingness_clin.csv, wbc_diff_patterns_clin.csv
(aggregate). The replication tables in results/tables are not written in this mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import load_config, path
from src.paper_ref import TABLE1_TOTAL
from src.subset_db import open_subset
from src.variables import BINARY, FEATURES, RATIOS

ap = argparse.ArgumentParser()
ap.add_argument("--track", choices=["replication", "clinical"], default="replication")
CLIN = ap.parse_args().track == "clinical"
cfg = load_config()["cohort"]
TABLES = path("tables")
PRE, POST = cfg["lab_window_pre_hours"], cfg["lab_window_post_hours"]
WIN = f"x.charttime >= s.intime - INTERVAL {PRE} HOUR AND x.charttime <= s.intime + INTERVAL {POST} HOUR"


def oavg(expr: str) -> str:
    """Row-order-independent mean: sort the non-null values, then average. avg() sums doubles in physical row order,
    which changes when the subset DB is rebuilt (stage 8e-1), so means differed in the last bits (~1e-14)."""
    return f"list_avg(list_sort(list({expr}) FILTER (WHERE ({expr}) IS NOT NULL)))"

LAB_ITEMS = [50868, 50882, 51006, 50893, 50902, 50912, 50931, 50983, 50971,          # chemistry
             51237, 51274, 51275,                                                    # coagulation
             51222, 51265, 51279, 51277, 51301, 51300, 51755,                        # CBC / WBC
             51133, 52769, 52075, 52074, 51253, 51244, 51245, 51256, 51254, 51146]   # differential
CHART_ITEMS = [220045, 220052, 220181, 225312, 220210, 224690, 223761, 223762, 220739, 223900, 223901]

# Hypertension / obesity: Quan et al. 2005 Elixhauser ICD-9-CM / ICD-10 definitions (+ I16, created FY2019)
HTN_ICD9 = ("401", "402", "403", "404", "405")
HTN_ICD10 = ("I10", "I11", "I12", "I13", "I15", "I16")
OBESITY_ICD9 = ("2780",)
OBESITY_ICD10 = ("E66",)

# FEATURES / BINARY / RATIOS: see src/variables.py

cohort = pd.read_parquet(path("processed") / ("cohort_clin.parquet" if CLIN else "cohort.parquet"))
con = open_subset()
# single-threaded: multi-threaded avg() sums doubles in a run-dependent order, so wbc_mean (and pwr) differed in the
# last bits between runs (~1e-14); one thread makes the feature table byte-reproducible (progress.md, stage 8c step 0)
con.execute("SET threads=1")
con.register("s_df", cohort[["subject_id", "hadm_id", "stay_id", "intime"]])
con.execute("CREATE TEMP TABLE s AS SELECT * FROM s_df")

# ---------------------------------------------------------------- 1. itemid audit (aggregate only)
lab_audit = con.execute(f"""
SELECT l.itemid, d.label, d.fluid, d.category, l.valueuom, count(*) AS n_rows, count(l.valuenum) AS n_numeric,
       round(quantile_cont(l.valuenum, 0.01), 2) AS p01, round(quantile_cont(l.valuenum, 0.5), 2) AS p50,
       round(quantile_cont(l.valuenum, 0.99), 2) AS p99
FROM mimiciv_hosp.labevents l JOIN mimiciv_hosp.d_labitems d USING (itemid)
WHERE l.itemid IN ({','.join(map(str, LAB_ITEMS))}) GROUP BY ALL ORDER BY l.itemid, n_rows DESC
""").df()
chart_audit = con.execute(f"""
SELECT c.itemid, d.label, d.category, d.unitname, count(*) AS n_rows,
       round(quantile_cont(c.valuenum, 0.01), 1) AS p01, round(quantile_cont(c.valuenum, 0.5), 1) AS p50,
       round(quantile_cont(c.valuenum, 0.99), 1) AS p99
FROM mimiciv_icu.chartevents c JOIN mimiciv_icu.d_items d USING (itemid)
WHERE c.itemid IN ({','.join(map(str, CHART_ITEMS))}) GROUP BY ALL ORDER BY c.itemid
""").df()
if not CLIN:
    lab_audit.to_csv(TABLES / "itemid_audit_lab.csv", index=False)
    chart_audit.to_csv(TABLES / "itemid_audit_chart.csv", index=False)
# hard checks on the unit assumptions the features rely on
units = lab_audit.groupby("itemid")["valueuom"].agg(lambda u: set(u.dropna()))
assert units[52075] == {"K/uL"} and units[51133] == {"K/uL"} and units[52074] == {"K/uL"}, "absolute count units changed"
assert units.get(52769, {"#/uL"}) <= {"#/uL"} and units.get(51253, {"#/uL"}) <= {"#/uL"}, "#/uL items changed"
assert units[51146] == {"%"}, "51146 expected to be Basophils %"

# ---------------------------------------------------------------- 2. first-day means (labs)
means = con.execute(f"""
WITH wbc_spec AS (   -- WBC per specimen from 51300/51301/51755 (all K/uL), de-duplicated across the two concepts
  SELECT subject_id, specimen_id, max(charttime) AS charttime, max(wbc) AS wbc FROM (
    SELECT subject_id, specimen_id, charttime, wbc FROM mimiciv_derived.complete_blood_count WHERE wbc IS NOT NULL
    UNION ALL
    SELECT subject_id, specimen_id, charttime, wbc FROM mimiciv_derived.blood_differential WHERE wbc > 0)
  GROUP BY subject_id, specimen_id)
SELECT s.stay_id,
  (SELECT {oavg('x.glucose')}          FROM mimiciv_derived.chemistry x            WHERE x.subject_id = s.subject_id AND {WIN}) AS glucose_mean,
  (SELECT {oavg('x.hemoglobin')}       FROM mimiciv_derived.complete_blood_count x WHERE x.subject_id = s.subject_id AND {WIN}) AS hemoglobin_mean,
  (SELECT {oavg('x.platelet')}         FROM mimiciv_derived.complete_blood_count x WHERE x.subject_id = s.subject_id AND {WIN}) AS platelets_mean,
  (SELECT {oavg('x.rbc')}              FROM mimiciv_derived.complete_blood_count x WHERE x.subject_id = s.subject_id AND {WIN}) AS rbc_mean,
  (SELECT {oavg('x.rdw')}              FROM mimiciv_derived.complete_blood_count x WHERE x.subject_id = s.subject_id AND {WIN}) AS rdw_mean,
  (SELECT {oavg('x.wbc')}              FROM wbc_spec x                             WHERE x.subject_id = s.subject_id AND {WIN}) AS wbc_mean,
  (SELECT {oavg('x.lymphocytes_abs')}  FROM mimiciv_derived.blood_differential x   WHERE x.subject_id = s.subject_id AND {WIN}) AS lymphocytes_mean,
  (SELECT {oavg('x.neutrophils_abs')}  FROM mimiciv_derived.blood_differential x   WHERE x.subject_id = s.subject_id AND {WIN}) AS neutrophils_mean,
  (SELECT {oavg('x.monocytes_abs')}    FROM mimiciv_derived.blood_differential x   WHERE x.subject_id = s.subject_id AND {WIN}) AS monocytes_mean
FROM s
""").df()

# ---------------------------------------------------------------- 3. min/max labs, vitals, scores
minmax_cols = [f"{v}_{f}" for v in ["aniongap", "bicarbonate", "creatinine", "bun", "calcium", "chloride",
                                    "sodium", "potassium", "ptt", "pt", "inr"] for f in ["min", "max"]]
other = con.execute(f"""
SELECT s.stay_id, {', '.join('l.' + c for c in minmax_cols)},
       v.heart_rate_mean, v.mbp_mean, v.resp_rate_mean, v.temperature_mean,
       sa.sapsii, ap.apsiii, lo.lods, oa.oasis, g.gcs_min AS gcs,
       ch.charlson_comorbidity_index AS cci, ch.myocardial_infarct, ch.congestive_heart_failure,
       ch.chronic_pulmonary_disease, ch.renal_disease, ch.age_score AS cci_age_score
FROM s
LEFT JOIN mimiciv_derived.first_day_lab l        USING (stay_id)
LEFT JOIN mimiciv_derived.first_day_vitalsign v  USING (stay_id)
LEFT JOIN mimiciv_derived.sapsii sa              USING (stay_id)
LEFT JOIN mimiciv_derived.apsiii ap              USING (stay_id)
LEFT JOIN mimiciv_derived.lods lo                USING (stay_id)
LEFT JOIN mimiciv_derived.oasis oa               USING (stay_id)
LEFT JOIN mimiciv_derived.first_day_gcs g        USING (stay_id)
LEFT JOIN mimiciv_derived.charlson ch            ON ch.hadm_id = s.hadm_id
""").df()

def icd_flag(prefix9, prefix10, name):
    c9 = " OR ".join(f"icd_code LIKE '{p}%'" for p in prefix9)
    c10 = " OR ".join(f"icd_code LIKE '{p}%'" for p in prefix10)
    return f"max(CASE WHEN (icd_version = 9 AND ({c9})) OR (icd_version = 10 AND ({c10})) THEN 1 ELSE 0 END) AS {name}"

dx = con.execute(f"""
SELECT s.stay_id,
  coalesce({icd_flag(HTN_ICD9, HTN_ICD10, 'x')}, 0) AS hypertension,
  coalesce({icd_flag(('401',), ('I10',), 'x')}, 0) AS hypertension_uncomplicated,
  coalesce({icd_flag(OBESITY_ICD9, OBESITY_ICD10, 'x')}, 0) AS obesity
FROM s LEFT JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = s.hadm_id
GROUP BY s.stay_id
""".replace(" AS x", "")).df()

# ---------------------------------------------------------------- 4. unit-mixing diagnostic (aggregate)
naive = None if CLIN else con.execute(f"""
SELECT s.stay_id,
  {oavg('CASE WHEN x.itemid IN (51133, 52769, 51244, 51245) THEN x.valuenum END')} AS lym_mixed_all,
  {oavg('CASE WHEN x.itemid IN (52075, 51256)               THEN x.valuenum END')} AS neu_mixed_all,
  {oavg('CASE WHEN x.itemid IN (52074, 51253, 51254)        THEN x.valuenum END')} AS mono_mixed_all,
  {oavg('CASE WHEN x.itemid IN (51133, 51244)               THEN x.valuenum END')} AS lym_oldcode,
  {oavg('CASE WHEN x.itemid IN (51146, 51254)               THEN x.valuenum END')} AS mono_oldcode
FROM s LEFT JOIN mimiciv_hosp.labevents x ON x.subject_id = s.subject_id AND {WIN}
     AND x.itemid IN (51133, 52769, 51244, 51245, 52075, 51256, 52074, 51253, 51254, 51146)
GROUP BY s.stay_id
""").df()
con.close()

# ---------------------------------------------------------------- 5. assemble
df = (cohort[["subject_id", "hadm_id", "stay_id", "y", "age", "gender", "anchor_year_group"]]
      .merge(means, on="stay_id", how="left").merge(other, on="stay_id", how="left")
      .merge(dx, on="stay_id", how="left"))
df["gender_male"] = (df["gender"] == "M").astype(int)
for c in ["myocardial_infarct", "congestive_heart_failure", "chronic_pulmonary_disease", "renal_disease"]:
    df[c] = df[c].fillna(0).astype(int)
num = [c for c in FEATURES if c not in BINARY + RATIOS]
df[num] = df[num].astype(float)

def ratio(a, b):
    b = b.where(b > 0)            # zero denominators -> missing (later imputed), never inf
    return a / b

df["nlr"] = ratio(df.neutrophils_mean, df.lymphocytes_mean)
df["plr"] = ratio(df.platelets_mean, df.lymphocytes_mean)
df["lmr"] = ratio(df.lymphocytes_mean, df.monocytes_mean)
df["pnr"] = ratio(df.platelets_mean, df.neutrophils_mean)
df["pwr"] = ratio(df.platelets_mean, df.wbc_mean)
df["sii"] = df.platelets_mean * df.neutrophils_mean / df.lymphocytes_mean.where(df.lymphocytes_mean > 0)
zero_den = {"lymphocytes_mean=0": int((df.lymphocytes_mean == 0).sum()), "monocytes_mean=0": int((df.monocytes_mean == 0).sum()),
            "neutrophils_mean=0": int((df.neutrophils_mean == 0).sum())}

assert len(FEATURES) == 55 and df.stay_id.is_unique and len(df) == len(cohort)
keep = ["subject_id", "hadm_id", "stay_id", "anchor_year_group", "y"] + FEATURES + ["hypertension_uncomplicated", "cci_age_score"]
if CLIN:
    # ---- CLIN: comorbidities / CCI (no age score) / SAPS-II from 02b --track clinical; wbc_diff_missing (D8.4)
    cm = pd.read_parquet(path("processed") / "comorb_clin.parquet")
    repl = ["myocardial_infarct", "congestive_heart_failure", "chronic_pulmonary_disease", "renal_disease",
            "hypertension", "hypertension_uncomplicated", "obesity", "cci", "sapsii"]
    df = df.drop(columns=repl + ["cci_age_score"]).merge(
        cm[["stay_id"] + repl + ["has_prior_admission"]], on="stay_id", how="left", validate="one_to_one")
    assert df.has_prior_admission.notna().all()
    for c in BINARY[1:] + ["hypertension_uncomplicated", "has_prior_admission"]:
        df[c] = df[c].astype(int)
    df[["cci", "sapsii"]] = df[["cci", "sapsii"]].astype(float)
    df["wbc_diff_missing"] = df[["lymphocytes_mean", "neutrophils_mean", "monocytes_mean"]].isna().all(axis=1).astype(int)
    keep = ["subject_id", "hadm_id", "stay_id", "anchor_year_group", "y"] + FEATURES + \
           ["wbc_diff_missing", "has_prior_admission", "hypertension_uncomplicated"]
    assert df.stay_id.is_unique and len(df) == len(cohort)
    df[keep].to_parquet(path("processed") / "features_clin.parquet", index=False)

    OUTC = path("tables").parent / "clinical"
    OUTC.mkdir(parents=True, exist_ok=True)
    small = lambda n: "<10" if 0 < n < 10 else int(n)
    cols = FEATURES + ["wbc_diff_missing", "has_prior_admission"]
    miss = pd.DataFrame({"feature": cols, "n_missing": [int(df[c].isna().sum()) for c in cols],
                         "missing_pct": [round(100 * df[c].isna().mean(), 2) for c in cols]})
    miss["over_20pct"] = miss.missing_pct > 20
    miss["n_missing"] = miss.n_missing.map(small)
    miss.to_csv(OUTC / "missingness_clin.csv", index=False)
    WBCD = ["lymphocytes_mean", "neutrophils_mean", "monocytes_mean", "nlr", "plr", "lmr", "pnr", "sii"]
    pat = df[WBCD].isna().apply(lambda r: "".join("M" if v else "." for v in r), axis=1)
    pats = pat.value_counts().rename_axis("pattern").reset_index(name="n")
    pats["n_missing_vars"] = pats.pattern.str.count("M")
    pats["n"] = pats.n.map(small)
    pats.to_csv(OUTC / "wbc_diff_patterns_clin.csv", index=False)
    partial = int(((df.wbc_diff_missing == 0) & df[WBCD].isna().any(axis=1)).sum())
    pd.set_option("display.width", 220)
    print(f"CLIN feature table: {len(df)} rows x {len(FEATURES)} predictors + wbc_diff_missing, has_prior_admission")
    print("zero denominators set to missing:", {k: small(v) for k, v in zero_den.items()})
    print("missing > 20%:", miss[miss.over_20pct].feature.tolist())
    print(miss.to_string(index=False))
    print("WBC differential missing patterns (order:", ", ".join(WBCD), "; M = missing):")
    print(pats.to_string(index=False))
    print(f"wbc_diff_missing = 1: {small(int(df.wbc_diff_missing.sum()))} ({100 * df.wbc_diff_missing.mean():.1f}%); "
          f"partial missing (wbc_diff_missing = 0 but >= 1 of the 8 missing): {small(partial)}")
    sys.exit(0)
df[keep].to_parquet(path("processed") / "features.parquet", index=False)

# ---------------------------------------------------------------- 6. aggregate reports
def summ(s: pd.Series, kind: str) -> str:
    s = s.dropna()
    if kind == "pct":
        return f"{100 * s.mean():.2f}%"
    if kind == "mean":
        return f"{s.mean():.2f} ± {s.std():.2f}"
    return f"{s.median():.2f} ({s.quantile(.25):.2f}, {s.quantile(.75):.2f})"

rows = []
for f in FEATURES:
    kind, v, a, b = TABLE1_TOTAL[f]
    paper = f"{v:.2f}%" if kind == "pct" else (f"{v:.2f} ± {a:.2f}" if kind == "mean" else f"{v:.2f} ({a:.2f}, {b:.2f})")
    ours_median = summ(df[f], "median") if kind != "pct" else ""
    rows.append({"feature": f, "missing_pct": round(100 * df[f].isna().mean(), 2), "paper_format": kind,
                 "ours": summ(df[f], kind), "ours_median_iqr": ours_median, "paper": paper})
comp = pd.DataFrame(rows)
comp.to_csv(TABLES / "feature_vs_paper.csv", index=False)

diag = []
for name, col in [("lymphocytes: abs K/uL (used)", df.lymphocytes_mean), ("lymphocytes: all items mixed", naive.lym_mixed_all),
                  ("lymphocytes: old code 51133+51244", naive.lym_oldcode), ("neutrophils: abs K/uL (used)", df.neutrophils_mean),
                  ("neutrophils: all items mixed", naive.neu_mixed_all), ("monocytes: abs K/uL (used)", df.monocytes_mean),
                  ("monocytes: all items mixed", naive.mono_mixed_all), ("monocytes: old code 51146+51254", naive.mono_oldcode)]:
    c = col.dropna()
    diag.append({"variant": name, "n": len(c), "median": round(c.median(), 2), "q1": round(c.quantile(.25), 2),
                 "q3": round(c.quantile(.75), 2)})
m = df[["stay_id", "platelets_mean", "wbc_mean"]].merge(naive, on="stay_id")
for name, val in [("NLR mixed", m.neu_mixed_all / m.lym_mixed_all), ("PLR mixed", m.platelets_mean / m.lym_mixed_all),
                  ("LMR mixed", m.lym_mixed_all / m.mono_mixed_all), ("PNR mixed", m.platelets_mean / m.neu_mixed_all),
                  ("SII mixed", m.platelets_mean * m.neu_mixed_all / m.lym_mixed_all)]:
    c = val.replace([np.inf, -np.inf], np.nan).dropna()
    diag.append({"variant": name, "n": len(c), "median": round(c.median(), 2), "q1": round(c.quantile(.25), 2),
                 "q3": round(c.quantile(.75), 2)})
pd.DataFrame(diag).to_csv(TABLES / "unit_mixing_diagnostic.csv", index=False)

pd.set_option("display.width", 220)
print(f"feature table: {df[keep].shape[0]} rows x {len(FEATURES)} predictors (+ ids/outcome)")
print("zero denominators set to missing:", zero_den)
print(comp.to_string(index=False))
print("\nunit-mixing diagnostic:")
print(pd.DataFrame(diag).to_string(index=False))
print("\nhypertension (Quan, uncomplicated+complicated) %:", round(100 * df.hypertension.mean(), 1),
      "| uncomplicated only (401/I10) %:", round(100 * df.hypertension_uncomplicated.mean(), 1))
print("CCI age-score component mean:", round(df.cci_age_score.mean(), 2))
