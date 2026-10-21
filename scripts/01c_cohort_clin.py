"""Stage 8e-1 step 1 — clinical-line cohort CLIN (decisions.md D8.1, D8.10).

Index admission = definition C (the patient's first admission with both a DN code and an ICU stay); its first ICU stay.
Prediction time = ICU admission + 24 h. Steps (each reports remaining n, excluded n, 1-year deaths among the remaining):
  1. patients with a DN-coded admission that has an ICU stay (= cohort_flow.csv step [1b])
  2. first ICU stay of the index admission
  3. age >= 18 (config cohort.min_age_clinical)
  4. exclude first ICU stay < 24 h when alive at 24 h (transferred out); outtime > intime + 24 h required
  5. exclude death before intime + 24 h (deathtime <= intime + 24 h, or dod < date(intime + 24 h)); a death within 24 h
     with an ICU stay < 24 h counts here, not in step 4
Outcome: D8.10 (death date in [date(intime), date(intime) + 365 d]; deathtime for in-hospital deaths, else dod).
Stratified 80/20 split, seed 42.
Outputs: data/processed/cohort_clin.parquet, split_clin.parquet (patient-level, git-ignored);
         results/clinical/cohort_flow_clin.csv, split_clin_summary.csv (aggregate; excluded counts 1–9 shown as "<10").
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import ROOT, load_config, path

cfg = load_config()
C = cfg["cohort"]
OUT = ROOT / "results" / "clinical"
OUT.mkdir(parents=True, exist_ok=True)


def small(n: int):
    return "<10" if 0 < n < 10 else int(n)


cand = pd.read_parquet(path("interim") / "cohort_candidates.parquet")
d = cand[cand.definition == "C"].copy()
assert d.subject_id.is_unique and d.stay_id.is_unique
n_1b = int(pd.read_csv(path("tables") / "cohort_flow.csv").set_index("step")
           .loc["[1b] patients with a DN-coded admission that has an ICU stay", "n"])
assert len(d) == n_1b, "definition-C episodes should equal step [1b] (one index admission per patient)"

t24 = d.intime + pd.Timedelta(hours=24)
alive24 = (d.deathtime.isna() | (d.deathtime > t24)) & (d.dod.isna() | (d.dod >= t24.dt.normalize()))
stay24 = d.outtime > t24
adult = d.age >= C["min_age_clinical"]

steps = [
    ("[1] patients with a DN-coded admission that has an ICU stay", pd.Series(True, index=d.index)),
    ("[2] first ICU stay of the index admission (definition C)", pd.Series(True, index=d.index)),
    (f"[3] age >= {C['min_age_clinical']}", adult),
    ("[4] exclude ICU stay < 24 h, alive at 24 h (transferred out)", adult & (stay24 | ~alive24)),
    ("[5] exclude death before ICU admission + 24 h", adult & alive24 & stay24),
]
rows, prev = [], len(d)
for name, keep in steps:
    x = d[keep]
    rows.append({"step": name, "n": len(x), "n_excluded": small(prev - len(x)),
                 "deaths_1y": int(x.death_1y_icu.sum()), "mortality_1y": round(float(x.death_1y_icu.mean()), 4)})
    prev = len(x)
flow = pd.DataFrame(rows)
flow.to_csv(OUT / "cohort_flow_clin.csv", index=False)

clin = d[steps[-1][1]].copy()
assert (clin.outtime > clin.intime + pd.Timedelta(hours=24)).all() and (clin.age >= C["min_age_clinical"]).all()
keep_cols = ["subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "intime", "outtime", "first_careunit",
             "hosp_los_h", "age", "gender", "anchor_year_group", "hospital_expire_flag", "deathtime", "dod",
             "death_1y_icu", "death_inhosp"]
clin = clin[keep_cols].rename(columns={"death_1y_icu": "y"}).reset_index(drop=True)
clin.to_parquet(path("processed") / "cohort_clin.parquet", index=False)

tr, te = train_test_split(clin.index, test_size=cfg["model"]["test_size"], stratify=clin.y, random_state=cfg["seed"])
split = clin[["stay_id", "y"]].copy()
split["split"] = "train"
split.loc[te, "split"] = "valid"
split[["stay_id", "split", "y"]].to_parquet(path("processed") / "split_clin.parquet", index=False)
ss = split.groupby("split").y.agg(n="size", deaths="sum", mortality="mean").reset_index()
ss.round(4).to_csv(OUT / "split_clin_summary.csv", index=False)

pd.set_option("display.width", 200)
print(flow.to_string(index=False))
print(ss.round(4).to_string(index=False))
print("LOS > 48 h would have excluded (n):", small(int((clin.hosp_los_h <= C["min_hosp_los_hours"]).sum())))
