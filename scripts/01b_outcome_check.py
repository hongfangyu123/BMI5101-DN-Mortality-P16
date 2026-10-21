"""Stage 8b — audit of the 1-year mortality outcome definition (aggregate counts only).

Old definition (D1.6):  hospital_expire_flag = 1  OR  dod <= date(intime) + 365 days
                         (no lower bound; in-hospital deaths have no 365-day upper bound)
New definition (8b):     death date in [date(intime), date(intime) + 365 days], compared at date granularity;
                         death date = date(admissions.deathtime) for in-hospital deaths of the index admission,
                         otherwise patients.dod (date only). If hospital_expire_flag = 1 but deathtime is missing,
                         dod is used (counted below).
Checks per cohort definition (A-D, final cohorts after LOS / age / CBC exclusions):
  dod before the ICU admission date; in-hospital death more than 365 days after ICU admission;
  hospital_expire_flag = 1 with missing dod or missing deathtime; dod differing from the deathtime date;
  number of patients classified differently by the old and new definitions (by direction).
Input: data/interim/cohort_candidates.parquet (stage 1). Output: results/tables/outcome_check.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import load_config, path

cfg = load_config()
H = cfg["cohort"]["outcome_horizon_days"]
T = path("tables")

cand = pd.read_parquet(path("interim") / "cohort_candidates.parquet")
final = cand[cand.los_ok & cand.age_ok & cand.cbc_complete].copy()

d_icu = final.intime.dt.normalize()
dod = pd.to_datetime(final.dod).dt.normalize()
dtime = pd.to_datetime(final.deathtime).dt.normalize()
inhosp = final.hospital_expire_flag.eq(1)

old = inhosp | ((dod - d_icu).dt.days <= H)
death_date = dtime.where(inhosp & dtime.notna(), dod)          # index-admission deathtime, else dod
days = (death_date - d_icu).dt.days
new = days.between(0, H)                                        # NaN (no death recorded) -> False

checks = {
    "n": lambda m: m.sum(),
    "deaths_old_definition": lambda m: (old & m).sum(),
    "deaths_new_definition": lambda m: (new & m).sum(),
    "dod_before_icu_date": lambda m: (m & ((dod - d_icu).dt.days < 0)).sum(),
    "deathtime_before_icu_date": lambda m: (m & inhosp & ((dtime - d_icu).dt.days < 0)).sum(),
    "inhosp_death_after_365d": lambda m: (m & inhosp & ((death_date - d_icu).dt.days > H)).sum(),
    "inhosp_flag_dod_missing": lambda m: (m & inhosp & dod.isna()).sum(),
    "inhosp_flag_deathtime_missing": lambda m: (m & inhosp & dtime.isna()).sum(),
    "deathtime_without_inhosp_flag": lambda m: (m & ~inhosp & dtime.notna()).sum(),
    "dod_ne_deathtime_date": lambda m: (m & inhosp & dod.notna() & dtime.notna() & (dod != dtime)).sum(),
    "old1_new0": lambda m: (m & old & ~new).sum(),
    "old0_new1": lambda m: (m & ~old & new).sum(),
}
rows = []
for k in ["A", "B", "C", "D"]:
    m = final.definition.eq(k)
    row = {"definition": k, **{c: int(f(m)) for c, f in checks.items()}}
    row["disagree_total"] = row["old1_new0"] + row["old0_new1"]
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(T / "outcome_check.csv", index=False)

pd.set_option("display.width", 220)
print(out.set_index("definition").T.to_string())
