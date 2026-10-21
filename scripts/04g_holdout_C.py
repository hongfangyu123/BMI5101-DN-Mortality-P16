"""Stage 8d step 1 — cohort C held-out validation set: secondary performance estimate (decisions.md D8.6, D8.7).

Models (stage-4 versions fitted on the whole 1,568-patient training set, no-leakage preprocessing, LASSO lambda.1se
selection of 12 variables): the recommended unpenalised LR (D8.11) and the pre-specified primary model, tuned RF (D4.7).
Both are rebuilt from src/models.py and checked against the saved stage-4 validation predictions.
Threshold = each model's own nested-CV pooled-OOF Youden cut-off (median over 5 repeats,
results/replication/nested_cv_full_oof_threshold_C.csv).

Metrics on the 392-patient validation set: AUROC (DeLong 95% CI); AUPRC (stratified bootstrap, 2,000 resamples,
seed 42, 95% percentile CI) with the prevalence baseline; Brier, scaled Brier; calibration intercept / slope;
accuracy / precision / recall / F1 / specificity / flagged % at the threshold; decision curve 0.05–0.60.
Final-model information: events, candidate variables, selected variables, EPV.
NB: this validation set was used in the post-hoc model choice of stages 6–7, so it is a secondary estimate only.
Outputs (aggregate only): results/replication/holdout_C.csv, holdout_dca_C.csv.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss

from src.common import ROOT, load_config, path
from src.metrics import (auprc, cal_intercept_slope_checked, decision_curve, delong_auc_ci, scaled_brier,
                         threshold_metrics)
from src.models import SCALED_MODELS, paper_models, tuned_model
from src.variables import FEATURES

cfg = load_config()
SEED = cfg["seed"]
P, T = path("processed"), path("tables")
OUT = ROOT / "results" / "replication"
N_BOOT = 2000

sel = pd.read_csv(T / "lasso_selection.csv").set_index("version").loc["noleak", "selected_1se"].split(", ")
tuned = json.load(open(T / "tuned_hyperparameters.json"))
Xs_all = pd.read_parquet(P / "noleak_scaled.parquet")
Xr_all = pd.read_parquet(P / "noleak_imputed.parquet")
assert (Xs_all.stay_id.values == Xr_all.stay_id.values).all()
tr, va = (Xs_all.split == "train").values, (Xs_all.split == "valid").values
ytr, yva = Xs_all.y.values[tr].astype(int), Xs_all.y.values[va].astype(int)
assert tr.sum() == 1568 and va.sum() == 392
candidates = [c for c in Xs_all.columns if c in FEATURES]

# pre-specified primary model (D4.7) = best 10-fold CV AUC among the no-leakage tuned models (first on ties, as in 04)
perf = pd.read_csv(T / "model_performance.csv")
cvt = perf[(perf.version == "noleak") & (perf.params == "tuned") & (perf.split == "train")].set_index("model").cv_auc_train
d47 = cvt.index[int(np.argmax(cvt.values))]
assert d47 == "RF", f"D4.7 model is {d47}, expected RF"

thr = pd.read_csv(OUT / "nested_cv_full_oof_threshold_C.csv").set_index(["params", "model"])["median"]
MODELS = {  # label -> (stage-4 params set, model name, estimator, nested-CV threshold key)
    "LR (unpenalised)": ("paper", "LR", paper_models(SEED)["LR"], ("paper", "LR (unpenalised)")),
    "RF (tuned)": ("tuned", "RF", tuned_model("RF", tuned["noleak/RF"], len(sel), SEED), ("tuned", "RF (tuned)")),
}


def boot_auprc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    """Stratified bootstrap: resample events and non-events separately (fixed counts), percentile 95% CI."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        vals[b] = auprc(y[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


rows, dca_rows = [], []
for label, (pset, m, est, tkey) in MODELS.items():
    a, b = (Xs_all.loc[tr, sel], Xs_all.loc[va, sel]) if m in SCALED_MODELS else (Xr_all.loc[tr, sel], Xr_all.loc[va, sel])
    fit = clone(est).fit(a, ytr)
    p = fit.predict_proba(b)[:, 1]
    saved = pd.read_parquet(P / f"preds_noleak_{pset}.parquet")
    saved = saved[saved.split == "valid"]
    assert (saved.stay_id.values == Xs_all.stay_id.values[va]).all()
    diff = np.abs(p - saved[m].values).max()
    assert diff < 1e-6, f"{label}: rebuilt model does not reproduce stage-4 validation predictions ({diff:.2e})"

    auc, lo, hi = delong_auc_ci(yva, p)
    ap_lo, ap_hi = boot_auprc_ci(yva, p)
    cal = cal_intercept_slope_checked(yva, p)
    t = float(thr.loc[tkey])
    events = int(ytr.sum())
    rows.append({
        "model": label, "role": "recommended (D8.11)" if pset == "paper" else "pre-specified primary (D4.7)",
        "n_valid": int(va.sum()), "events_valid": int(yva.sum()), "prevalence_valid": float(yva.mean()),
        "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
        "auprc": auprc(yva, p), "auprc_lo": ap_lo, "auprc_hi": ap_hi,
        "brier": brier_score_loss(yva, p), "scaled_brier": scaled_brier(yva, p),
        "cal_intercept": cal["cal_intercept"], "cal_slope": cal["cal_slope"],
        "cal_converged": cal["cal_intercept_ok"] and cal["cal_slope_ok"],
        "threshold_nested_oof_youden_median": t,
        **threshold_metrics(yva, p, t), "flagged_pct": 100 * float(np.mean(p >= t)),
        "n_train": int(tr.sum()), "events_train": events, "n_candidates": len(candidates), "n_selected": len(sel),
        "epv_selected": events / len(sel), "events_per_candidate": events / len(candidates),
    })
    d = decision_curve(yva, p, thresholds=np.round(np.arange(0.05, 0.6001, 0.01), 2))
    dca_rows.append(d.assign(model=label))

res = pd.DataFrame(rows)
res.round(6).to_csv(OUT / "holdout_C.csv", index=False)
dca = pd.concat(dca_rows)[["model", "threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none"]]
dca.round(6).to_csv(OUT / "holdout_dca_C.csv", index=False)

pd.set_option("display.width", 220)
print(res.round(4).T.to_string())
for m_, d in dca.groupby("model", sort=False):
    better = d[(d.net_benefit_model > d.net_benefit_all) & (d.net_benefit_model > 0)].threshold
    print(f"[DCA] {m_}: net benefit > treat-all and > 0 at {len(better)}/{len(d)} thresholds "
          f"({better.min():.2f}–{better.max():.2f})")
