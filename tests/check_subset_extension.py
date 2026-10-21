"""One-off check, kept for the record (stage 8e-1; run before 02 was made row-order independent): does extending the subset DB to the CLIN candidates change the replication features?

Builds a temporary subset DB for the ORIGINAL subject set (union of A-D after LOS/age, as before 8e-1), runs
scripts/02_features.py against it with all outputs redirected to a temporary directory under data/interim, and
compares the resulting feature table with data/processed/features.parquet (built from the extended DB).
Only aggregate differences are printed. The temporary DB and directory are deleted afterwards.
"""
from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import ROOT, load_config, path
from src.subset_db import build_subset_db

cand = pd.read_parquet(path("interim") / "cohort_candidates.parquet")
old_subjects = cand[cand.los_ok & cand.age_ok].subject_id.drop_duplicates()
print("original subject set:", len(old_subjects))

tmp = path("interim") / "subset_extension_check"
tmp.mkdir(exist_ok=True)
db = tmp / "mimic_subset_original.duckdb"
build_subset_db(old_subjects, db, log=lambda m: None)
(tmp / "processed").mkdir(exist_ok=True)
(tmp / "tables").mkdir(exist_ok=True)
shutil.copy(path("processed") / "cohort.parquet", tmp / "processed" / "cohort.parquet")

cfg = load_config()
saved = dict(cfg["paths"])
cfg["paths"].update({"subset_db": str(db), "processed": str(tmp / "processed"), "tables": str(tmp / "tables")})
try:
    runpy.run_path(str(ROOT / "scripts" / "02_features.py"), run_name="__main__")
finally:
    cfg["paths"].clear(); cfg["paths"].update(saved)

a = pd.read_parquet(tmp / "processed" / "features.parquet")   # original DB
b = pd.read_parquet(path("processed") / "features.parquet")    # extended DB
assert a.columns.tolist() == b.columns.tolist() and (a.stay_id.values == b.stay_id.values).all()
rows = []
for c in a.columns:
    if a[c].dtype.kind in "fi":
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both = ~np.isnan(x) & ~np.isnan(y)
        rows.append({"column": c, "nan_pattern_equal": bool((np.isnan(x) == np.isnan(y)).all()),
                     "n_different": int((x[both] != y[both]).sum()),
                     "max_abs_diff": float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0})
    else:
        rows.append({"column": c, "nan_pattern_equal": True, "n_different": int((a[c] != b[c]).sum()), "max_abs_diff": 0.0})
r = pd.DataFrame(rows)
print("byte-identical frame:", a.equals(b))
print(r[(r.n_different > 0) | ~r.nan_pattern_equal].to_string(index=False))
import hashlib
h = hashlib.sha256((tmp / 'processed' / 'features.parquet').read_bytes()).hexdigest()
print('original-DB features sha256:', h)
shutil.rmtree(tmp)
