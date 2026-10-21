"""Stage 4d — nested cross-validation: LASSO variable selection repeated inside every CV fold.

Stage 4 runs LASSO once on the whole training set and then cross-validates the models on the 12 selected
variables, so its "10-fold CV AUC" does not account for the selection step (decisions.md D4.18).
Here, for each of the same 10 outer folds (StratifiedKFold, seed 42, no-leakage training set):
  1. imputation + standardisation fitted on the outer-training part only (src/preprocess.py)
  2. LASSO (glmnet-style, inner 10-fold CV on binomial deviance) -> variables at lambda.1se
  3. every model refitted on those variables, AUC on the held-out outer fold
Hyper-parameters are held at the stage-4 values (paper values / tuned_hyperparameters.json), so the
comparison isolates the optimism from variable selection; tuning itself is not nested.
The non-nested reference is recomputed on the same folds (stage-4 variables and preprocessing) and must
match the stage-4 CV AUC.
Outputs: results/tables/nested_cv_auc.csv, nested_cv_selection_freq.csv, nested_cv_folds.csv (aggregates only).
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.common import load_config, path
from src.lasso import cv_lasso
from src.models import SCALED_MODELS, paper_models, tuned_model
from src.preprocess import impute_scale
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

cfg = load_config()
SEED, K = cfg["seed"], cfg["model"]["cv_folds"]
P, T = path("processed"), path("tables")
N_JOBS = int(cfg["duckdb"]["threads"])
CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
VERSION = "noleak"

feat = pd.read_parquet(P / "features.parquet")
split = pd.read_parquet(P / "split.parquet")
assert (feat.stay_id.values == split.stay_id.values).all()
train = feat[(split.split == "train").values].reset_index(drop=True)
Xs_ref = pd.read_parquet(P / f"{VERSION}_scaled.parquet").query("split == 'train'").reset_index(drop=True)
Xr_ref = pd.read_parquet(P / f"{VERSION}_imputed.parquet").query("split == 'train'").reset_index(drop=True)
assert (train.stay_id.values == Xs_ref.stay_id.values).all()      # same row order -> same folds as stage 4
y = train.y.values
X_raw = train[FEATURES]

sel_ref = pd.read_csv(T / "lasso_selection.csv").set_index("version").loc[VERSION, "selected_1se"].split(", ")
tuned = json.load(open(T / "tuned_hyperparameters.json"))
perf = pd.read_csv(T / "model_performance.csv")
cv_stage4 = perf[(perf.version == VERSION) & (perf.split == "train")].set_index(["params", "model"]).cv_auc_train


def models(p: int) -> dict:
    out = {("paper", m): est for m, est in paper_models(SEED).items()}
    out.update({("tuned", m): tuned_model(m, tuned[f"{VERSION}/{m}"], p, SEED) for m in ["RF", "XGBoost", "SVM", "LR"]})
    return out


rows, sel_count, n_sel, fold_rows = [], pd.Series(0, index=FEATURES), [], []
for k, (itr, ite) in enumerate(CV.split(X_raw, y)):
    # ---- nested: preprocessing + LASSO inside the fold
    Xi, Xs = impute_scale(X_raw, itr)
    res = cv_lasso(Xs.iloc[itr].values, y[itr], FEATURES, n_folds=K, seed=SEED, n_jobs=N_JOBS)
    sel = res.selected("1se")
    sel_count[sel] += 1
    n_sel.append(len(sel))
    fold_rows.append({"fold": k, "lambda_1se": res.lambda_1se, "n_selected": len(sel),
                      "same_as_stage4": set(sel) == set(sel_ref)})
    for key, est in models(len(sel)).items():
        Xm = Xs if key[1] in SCALED_MODELS else Xi
        fit = clone(est).fit(Xm.iloc[itr][sel], y[itr])
        auc_n = roc_auc_score(y[ite], fit.predict_proba(Xm.iloc[ite][sel])[:, 1])
        # ---- non-nested reference: stage-4 variables and preprocessing, same fold
        Xm_ref = Xs_ref if key[1] in SCALED_MODELS else Xr_ref
        fit_r = clone(models(len(sel_ref))[key]).fit(Xm_ref.iloc[itr][sel_ref], y[itr])
        auc_r = roc_auc_score(y[ite], fit_r.predict_proba(Xm_ref.iloc[ite][sel_ref])[:, 1])
        rows.append({"fold": k, "params": key[0], "model": key[1], "auc_nested": auc_n, "auc_nonnested": auc_r})
    print(f"fold {k}: lambda.1se {res.lambda_1se:.4f}, {len(sel)} variables selected", flush=True)

f = pd.DataFrame(rows)
out = f.groupby(["params", "model"], sort=False).agg(
    cv_auc_nonnested=("auc_nonnested", "mean"), cv_auc_nested=("auc_nested", "mean"),
    sd_nested=("auc_nested", "std")).reset_index()
d = (f.auc_nested - f.auc_nonnested).groupby([f.params, f.model], sort=False)
out["optimism"] = out.cv_auc_nonnested - out.cv_auc_nested
out["optimism_fold_min"] = -d.max().values
out["optimism_fold_max"] = -d.min().values
out["cv_auc_stage4"] = [cv_stage4.loc[(p, m)] for p, m in zip(out.params, out.model)]
assert (out.cv_auc_nonnested - out.cv_auc_stage4).abs().max() < 1e-3, "non-nested reference does not match stage 4"
out.round(4).to_csv(T / "nested_cv_auc.csv", index=False)
freq = pd.DataFrame({"feature": FEATURES, "n_folds_selected": sel_count.values,
                     "in_stage4_selection": [x in sel_ref for x in FEATURES]}) \
    .query("n_folds_selected > 0 or in_stage4_selection").sort_values("n_folds_selected", ascending=False)
freq.to_csv(T / "nested_cv_selection_freq.csv", index=False)
pd.DataFrame(fold_rows).round(5).to_csv(T / "nested_cv_folds.csv", index=False)

pd.set_option("display.width", 200)
print(f"\nvariables selected per fold: min {min(n_sel)}, median {int(np.median(n_sel))}, max {max(n_sel)}")
print(freq.to_string(index=False))
print()
print(out.round(4).to_string(index=False))
