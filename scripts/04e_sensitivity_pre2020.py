"""Stage 4e — sensitivity analysis excluding anchor_year_group 2020 - 2022 (COVID-19 period, v3.x additions).

The whole modelling pipeline is re-run on the remaining patients (no-leakage preprocessing):
  cohort size / 1-year mortality -> new stratified 80/20 split (seed 42) -> >20%-missing filter -> imputation +
  standardisation fitted on the new training set -> LASSO (10-fold CV, lambda.1se) -> recommended model
  (unpenalised LR) and tuned RF (same grid as stage 4, 10-fold CV AUC) -> validation AUC with DeLong 95% CI,
  paired DeLong LR vs RF, Brier / calibration slope, SHAP top 5 (LinearExplainer / TreeExplainer).
Cohort selection is per patient (definition C), so dropping whole patients leaves every other patient's index
stay unchanged; filtering the final cohort is therefore equivalent to re-running stage 1 with this exclusion.
Output: results/tables/sensitivity_pre2020.csv (aggregates only).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from src.common import load_config, path
from src.lasso import cv_lasso
from src.metrics import cal_intercept_slope, delong_auc_ci, delong_roc_test
from src.models import paper_models, tuning_grids
from src.preprocess import impute_scale
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

cfg = load_config()
SEED, K = cfg["seed"], cfg["model"]["cv_folds"]
P, T = path("processed"), path("tables")
N_JOBS = int(cfg["duckdb"]["threads"])
CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
EXCLUDE = "2020 - 2022"

feat = pd.read_parquet(P / "features.parquet")
groups = feat.anchor_year_group.value_counts()
assert EXCLUDE in groups.index, f"anchor_year_group labels: {list(groups.index)}"
df = feat[feat.anchor_year_group != EXCLUDE].reset_index(drop=True)
print(f"excluded anchor_year_group {EXCLUDE}: {int(groups[EXCLUDE])} patients; remaining {len(df)}, "
      f"deaths {int(df.y.sum())} ({df.y.mean():.3f})")

# ---- new split and no-leakage preprocessing
tr_idx, va_idx = train_test_split(df.index, test_size=cfg["model"]["test_size"], stratify=df.y, random_state=SEED)
tr = df.index.isin(tr_idx)
va = ~tr
y = {"train": df.y.values[tr], "valid": df.y.values[va]}
kept = [f for f in FEATURES if df[f].isna().mean() <= 0.20]
Xi, Xs = impute_scale(df[kept], df.index[tr])
print(f"split: train {tr.sum()} (deaths {y['train'].sum()}), valid {va.sum()} (deaths {y['valid'].sum()}); "
      f"features kept {len(kept)}/{len(FEATURES)}")

# ---- LASSO
res = cv_lasso(Xs.loc[tr, kept].values, y["train"], kept, n_folds=K, seed=SEED, n_jobs=N_JOBS)
sel = res.selected("1se")
print(f"LASSO lambda.1se {res.lambda_1se:.5f}: {len(sel)} variables {sel}", flush=True)
Xtr_s, Xva_s, Xtr_r, Xva_r = Xs.loc[tr, sel], Xs.loc[va, sel], Xi.loc[tr, sel], Xi.loc[va, sel]

# ---- models: recommended unpenalised LR, tuned RF
lr = paper_models(SEED)["LR"]
lr_cv = cross_val_score(lr, Xtr_s, y["train"], cv=CV, scoring="roc_auc", n_jobs=N_JOBS).mean()
lr.fit(Xtr_s, y["train"])
est, grid = tuning_grids(len(sel), SEED)["RF"]
gs = GridSearchCV(est, grid, cv=CV, scoring="roc_auc", n_jobs=N_JOBS, refit=True).fit(Xtr_r, y["train"])
rf = gs.best_estimator_
print(f"tuned RF: CV AUC {gs.best_score_:.3f} params {gs.best_params_}", flush=True)
p_lr, p_rf = lr.predict_proba(Xva_s)[:, 1], rf.predict_proba(Xva_r)[:, 1]
test = delong_roc_test(y["valid"], p_lr, p_rf)

# ---- SHAP on the validation set
sv_lr = shap.LinearExplainer(lr, Xtr_s)(Xva_s).values
sv_rf = shap.TreeExplainer(rf)(Xva_r).values
sv_rf = sv_rf[:, :, 1] if sv_rf.ndim == 3 else sv_rf
top5 = {m: list(pd.Series(np.abs(v).mean(axis=0), index=sel).sort_values(ascending=False).index[:5])
        for m, v in [("LR (unpenalised)", sv_lr), ("RF (tuned)", sv_rf)]}

rows = []
for m, p_, cv_ in [("LR (unpenalised)", p_lr, lr_cv), ("RF (tuned)", p_rf, gs.best_score_)]:
    auc, lo, hi = delong_auc_ci(y["valid"], p_)
    citl, slope = cal_intercept_slope(y["valid"], p_)
    rows.append({"model": m, "n": len(df), "deaths": int(df.y.sum()), "mortality": df.y.mean(),
                 "n_excluded": int(groups[EXCLUDE]), "n_train": int(tr.sum()), "n_valid": int(va.sum()),
                 "deaths_valid": int(y["valid"].sum()), "lambda_1se": res.lambda_1se, "n_selected": len(sel),
                 "selected_1se": ", ".join(sel), "cv_auc_train": cv_, "auc_valid": auc, "auc_lo": lo, "auc_hi": hi,
                 "brier": brier_score_loss(y["valid"], p_), "cal_in_the_large": citl, "cal_slope": slope,
                 "shap_top5": ", ".join(top5[m]), "delong_diff_lr_minus_rf": test["diff"],
                 "delong_diff_lo": test["diff_lo"], "delong_diff_hi": test["diff_hi"], "delong_p": test["p_value"],
                 "rf_params": str(gs.best_params_) if m.startswith("RF") else ""})
out = pd.DataFrame(rows)
out.round(4).to_csv(T / "sensitivity_pre2020.csv", index=False)

pd.set_option("display.width", 220)
print(out.drop(columns=["selected_1se"]).round(3).T.to_string())
