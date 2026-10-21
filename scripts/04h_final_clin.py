"""Stage 8e-4 step 1 — fit the frozen CLIN final model on the CLIN training set (decisions.md D8.14, D8.15).

Follows results/clinical/frozen_model_clin.json exactly, on the 2,045 training patients only (the held-out set is never
loaded; only its stay_ids are read, to assert disjointness):
  20% missing rule (8 WBC-differential variables exempt, D8.4) -> imputation -> standardisation -> LASSO lambda.1se
  (inner 10-fold, seed 42) -> LR (L2): GridSearchCV over C (inner stratified 5-fold, seed 42, AUROC) on the
  standardised selected variables; comparator tuned RF: same GridSearchCV on the imputed selected variables.
Before fitting, the training stay_id hash, the candidate-matrix hash and the src/ code hashes are checked against the
frozen file. Refuses to overwrite an existing model bundle.

Outputs: data/interim/final_clin/final_clin.pkl (git-ignored: preprocessing parameters, LASSO result, both fitted
models) and results/clinical/final_model_clin.json (aggregate fit record incl. the bundle sha256, checked by 04i).
Prints only the selected variables, lambda.1se, the chosen C (and RF parameters), events and EPV.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.common import ROOT, load_config, path
from src.lasso import cv_lasso
from src.models import tuning_grids
from src.preprocess import impute_fit, missing_filter, scale_fit
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SPEC_FILE = ROOT / "results" / "clinical" / "frozen_model_clin.json"
REC_FILE = ROOT / "results" / "clinical" / "final_model_clin.json"
BUNDLE = path("interim") / "final_clin" / "final_clin.pkl"
spec = json.loads(SPEC_FILE.read_text())
spec_sha = hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest()
if BUNDLE.exists():
    raise SystemExit(f"{BUNDLE} already exists; the frozen model is fitted once (delete it only if nothing has used it)")

SEED = spec["seed"]
N_JOBS = int(load_config()["duckdb"]["threads"])
for f, h in spec["code_sha256"].items():
    assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h, f"{f} changed since the model was frozen"

# ---------------------------------------------------------------- training set only
P = path("processed")
hold_ids = pd.read_parquet(P / "split_clin.parquet", columns=["stay_id"], filters=[("split", "==", "valid")]).stay_id
split = pd.read_parquet(P / "split_clin.parquet", filters=[("split", "==", "train")])
feat = pd.read_parquet(P / "features_clin.parquet", filters=[("stay_id", "in", split.stay_id.tolist())])
assert (split.split == "train").all() and len(split) == spec["training_set"]["n"]
assert set(feat.stay_id).isdisjoint(set(hold_ids)) and len(hold_ids) == 512, "held-out CLIN rows were loaded"
assert len(feat) == len(split) and (feat.stay_id.values == split.stay_id.values).all() and (feat.y.values == split.y.values).all()
assert hashlib.sha256(np.sort(split.stay_id.values).tobytes()).hexdigest() == spec["training_set"]["stay_id_sha256"]
CAND = FEATURES + ["wbc_diff_missing"]
EXEMPT = spec["candidates"]["missing_filter_exempt"]
assert len(CAND) == spec["candidates"]["n"]
X = feat[CAND].reset_index(drop=True)
assert hashlib.sha256(np.ascontiguousarray(X.values.astype(float))).hexdigest() == spec["training_set"]["candidate_matrix_sha256"]
y = feat.y.values.astype(int)
assert int(y.sum()) == spec["training_set"]["events"]

# ---------------------------------------------------------------- steps 1-4
passed = set(missing_filter(X, X.index))
kept = [c for c in CAND if c in passed or c in EXEMPT]
fill = impute_fit(X[kept])
Xi = X[kept].fillna(fill)
mu, sd = scale_fit(Xi)
Xs = (Xi - mu) / sd
las = cv_lasso(Xs.values, y, kept, n_folds=10, seed=SEED, n_jobs=N_JOBS)
sel = las.selected("1se")
if not sel:
    raise SystemExit("LASSO selected no variable: no model is frozen (frozen_model_clin.json step 7)")

# ---------------------------------------------------------------- steps 5-6
def tune(m, a):
    base, grid = tuning_grids(len(sel), SEED)[m]
    return GridSearchCV(clone(base), grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc",
                        n_jobs=N_JOBS, refit=True).fit(a, y)

gs_lr = tune("LR", Xs[sel])
gs_rf = tune("RF", Xi[sel])
jsonable = lambda d: {k: ("inf" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}

events = int(y.sum())
bundle = {"spec_sha256": spec_sha, "train_stay_id_sha256": spec["training_set"]["stay_id_sha256"],
          "candidates": CAND, "kept": kept, "fill": fill, "mu": mu, "sd": sd,
          "lasso": {"lambda_1se": las.lambda_1se, "lambda_min": las.lambda_min, "selected": sel},
          "models": {"LR (L2)": gs_lr.best_estimator_, "RF (tuned)": gs_rf.best_estimator_},
          "params": {"LR (L2)": jsonable(gs_lr.best_params_), "RF (tuned)": jsonable(gs_rf.best_params_)},
          "inner_cv_auroc": {"LR (L2)": float(gs_lr.best_score_), "RF (tuned)": float(gs_rf.best_score_)}}
BUNDLE.parent.mkdir(parents=True, exist_ok=True)
tmp = BUNDLE.with_suffix(".tmp")
with open(tmp, "wb") as fh:
    pickle.dump(bundle, fh)
os.replace(tmp, BUNDLE)

rec = {"spec_file": str(SPEC_FILE.relative_to(ROOT)), "spec_sha256": spec_sha,
       "bundle": str(BUNDLE.relative_to(ROOT)), "bundle_sha256": hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
       "n_train": len(y), "events": events, "n_candidates": len(CAND),
       "n_dropped_missing": len(CAND) - len(kept), "dropped_missing": [c for c in CAND if c not in kept],
       "lambda_1se": las.lambda_1se, "lambda_min": las.lambda_min, "n_selected": len(sel), "selected": sel,
       "params": bundle["params"], "inner_cv_auroc": bundle["inner_cv_auroc"],
       "epv_selected": events / len(sel), "events_per_candidate": events / len(CAND)}
REC_FILE.write_text(json.dumps(rec, indent=1) + "\n")

print(f"selected ({len(sel)}): {', '.join(sel)}")
print(f"lambda.1se = {las.lambda_1se:.6g}   (dropped by the 20% rule: {rec['dropped_missing'] or 'none'})")
print(f"LR (L2) C = {bundle['params']['LR (L2)']['C']}   RF (tuned) = {bundle['params']['RF (tuned)']}")
print(f"events = {events} / {len(y)}; EPV = {rec['epv_selected']:.1f} (events / selected); "
      f"events / candidates = {rec['events_per_candidate']:.1f}")
