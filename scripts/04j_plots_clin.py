"""Stage 8f — figures for the CLIN held-out evaluation (decisions.md D8.17). Plots only: nothing is refitted.

The frozen model bundle (data/interim/final_clin/final_clin.pkl) is checked against the committed fit record and frozen
spec (same checks as 04i), the held-out set (512 patients) is predicted again, and EVERY metric in
results/clinical/holdout_clin.csv and holdout_dca_clin.csv is recomputed with the same code and compared after the same
6-decimal rounding used when those files were written (|difference| < 1e-9, booleans / strings equal); any mismatch
aborts before plotting. Predictions are stored only under data/ (data/interim/final_clin/holdout_pred_clin.parquet).

Figures (results/figures/):
  figC1_roc_clin_holdout.png          ROC, held-out set, LR (L2) and tuned RF
  figC2_calibration_clin_holdout.png  calibration (deciles of predicted risk), held-out set
  figC3_dca_clin_holdout.png          decision curve 0.05–0.60, held-out set
  figC4_calibration_clin_nested_oof.png  calibration of the nested-CV pooled OOF predictions (training set, 5 repeats)
Also writes results/clinical/holdout_gap_diagnostics_clin.csv: the D8.15 post-hoc descriptive diagnostics of the gap
between nested-CV and held-out AUROC (apparent training AUROC, AUROC of held-out-sized resamples of the nested-CV OOF
predictions, standardised mean differences training vs held-out of the selected variables).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve

from src.common import ROOT, load_config, path
from src.metrics import (auprc, cal_intercept_slope_checked, decision_curve, delong_auc_ci, delong_roc_test,
                         scaled_brier, threshold_metrics)
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
CL, F, P = ROOT / "results" / "clinical", path("figures"), path("processed")
SPEC_FILE, REC_FILE = CL / "frozen_model_clin.json", CL / "final_model_clin.json"
PRED_OUT = path("interim") / "final_clin" / "holdout_pred_clin.parquet"
MODELS = ["LR (L2)", "RF (tuned)"]
COLORS = {"LR (L2)": "#eda100", "RF (tuned)": "#2a78d6"}
MARKERS = {"LR (L2)": "D", "RF (tuned)": "o"}
GRID, SURFACE = "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10})

# ---------------------------------------------------------------- frozen model (same checks as 04i guard 2)
spec, rec = json.loads(SPEC_FILE.read_text()), json.loads(REC_FILE.read_text())
BUNDLE = ROOT / rec["bundle"]
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == rec["bundle_sha256"], "model file differs from the fit record"
b = pickle.load(open(BUNDLE, "rb"))
assert b["spec_sha256"] == hashlib.sha256(SPEC_FILE.read_bytes()).hexdigest() == rec["spec_sha256"]
assert b["candidates"] == FEATURES + ["wbc_diff_missing"] and b["lasso"]["selected"] == rec["selected"]
sel, SEED = b["lasso"]["selected"], spec["seed"]
lr, rf = b["models"]["LR (L2)"], b["models"]["RF (tuned)"]
THR = {m: float(spec["thresholds"][m]) for m in MODELS}

split = pd.read_parquet(P / "split_clin.parquet")
tr_ids = split.loc[split.split == "train", "stay_id"].values
ev_ids = split.loc[split.split == "valid", "stay_id"].values
assert hashlib.sha256(np.sort(tr_ids).tobytes()).hexdigest() == spec["training_set"]["stay_id_sha256"]
assert len(ev_ids) == 512 and set(ev_ids).isdisjoint(set(tr_ids))
f = pd.read_parquet(P / "features_clin.parquet", filters=[("stay_id", "in", list(ev_ids))]).set_index("stay_id").loc[list(ev_ids)]
Xi = f[b["kept"]].fillna(b["fill"])
Xs = (Xi - b["mu"]) / b["sd"]
y = f.y.values.astype(int)
pred = {"LR (L2)": lr.predict_proba(Xs[sel])[:, 1], "RF (tuned)": rf.predict_proba(Xi[sel])[:, 1]}
PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"stay_id": ev_ids, "y": y, "p_lr_l2": pred["LR (L2)"], "p_rf_tuned": pred["RF (tuned)"]}).to_parquet(PRED_OUT, index=False)

# ---------------------------------------------------------------- recompute holdout_clin.csv / holdout_dca_clin.csv
def boot_auprc_ci(yy, p, n_boot=2000, seed=SEED):        # identical to 04i
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        vals[i] = auprc(yy[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


dl = delong_roc_test(y, pred["LR (L2)"], pred["RF (tuned)"])
ev = int(spec["training_set"]["events"]); n_tr = int(spec["training_set"]["n"]); n_cand = len(b["candidates"])
rows, dca_rows = [], []
for m in MODELS:
    p = pred[m]
    auc, lo, hi = delong_auc_ci(y, p)
    ap_lo, ap_hi = boot_auprc_ci(y, p)
    cal = cal_intercept_slope_checked(y, p)
    t = THR[m]
    rows.append({
        "model": m, "role": "recommended (D8.14)" if m == "LR (L2)" else "comparator (tuned RF)",
        "n_eval": len(y), "events_eval": int(y.sum()), "prevalence_eval": float(y.mean()),
        "auroc": auc, "auroc_lo": lo, "auroc_hi": hi, "auprc": auprc(y, p), "auprc_lo": ap_lo, "auprc_hi": ap_hi,
        "brier": brier_score_loss(y, p), "scaled_brier": scaled_brier(y, p),
        "cal_intercept": cal["cal_intercept"], "cal_slope": cal["cal_slope"],
        "cal_converged": cal["cal_intercept_ok"] and cal["cal_slope_ok"],
        "threshold_frozen": t, **threshold_metrics(y, p, t), "flagged_pct": 100 * float(np.mean(p >= t)),
        "delong_lr_minus_rf_diff": dl["diff"], "delong_lr_minus_rf_lo": dl["diff_lo"],
        "delong_lr_minus_rf_hi": dl["diff_hi"], "delong_lr_minus_rf_p": dl["p_value"],
        "n_train": n_tr, "events_train": ev, "n_candidates": n_cand, "n_selected": len(sel),
        "epv_selected": ev / len(sel), "events_per_candidate": ev / n_cand})
    dca_rows.append(decision_curve(y, p, thresholds=np.round(np.arange(0.05, 0.6001, 0.01), 2)).assign(model=m))
new = pd.DataFrame(rows).round(6)
new_dca = pd.concat(dca_rows)[["model", "threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none"]].round(6)


def compare(saved: pd.DataFrame, now: pd.DataFrame, name: str) -> float:
    assert list(saved.columns) == list(now.columns) and len(saved) == len(now), f"{name}: layout differs"
    worst = 0.0
    for c in saved.columns:
        a, z = saved[c].values, now[c].values
        if pd.api.types.is_numeric_dtype(saved[c]) and not pd.api.types.is_bool_dtype(saved[c]):
            d = float(np.max(np.abs(a.astype(float) - z.astype(float))))
            if not d < 1e-9:
                raise SystemExit(f"{name}: column {c} differs by {d:.3e}; not plotting")
            worst = max(worst, d)
        elif not (a.astype(str) == z.astype(str)).all():
            raise SystemExit(f"{name}: column {c} differs; not plotting")
    return worst


d1 = compare(pd.read_csv(CL / "holdout_clin.csv"), new, "holdout_clin.csv")
d2 = compare(pd.read_csv(CL / "holdout_dca_clin.csv"), new_dca.reset_index(drop=True), "holdout_dca_clin.csv")
print(f"recomputed metrics identical to the saved files: holdout_clin.csv ({new.shape[1]} columns, max |diff| {d1:.1e}), "
      f"holdout_dca_clin.csv ({len(new_dca)} rows, max |diff| {d2:.1e})")
R = new.set_index("model")

# ---------------------------------------------------------------- figures (held-out set)
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1)
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, pred[m])
    ax.plot(fpr, tpr, color=COLORS[m], lw=2.2 if m == "LR (L2)" else 1.5,
            label=f"{m}{' (recommended)' if m == 'LR (L2)' else ''}  AUROC {R.loc[m, 'auroc']:.3f} "
                  f"({R.loc[m, 'auroc_lo']:.3f}–{R.loc[m, 'auroc_hi']:.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
ax.set_title(f"Fig. C1 — ROC, CLIN held-out set (n = {len(y)}, {int(y.sum())} deaths)", loc="left", fontsize=11)
ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(F / "figC1_roc_clin_holdout.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1, label="Ideal")
for m in MODELS:
    obs, pr = calibration_curve(y, pred[m], n_bins=10, strategy="quantile")
    ax.plot(pr, obs, color=COLORS[m], marker=MARKERS[m], ms=5, lw=2.2 if m == "LR (L2)" else 1.5,
            label=f"{m}  Brier {R.loc[m, 'brier']:.3f}, intercept {R.loc[m, 'cal_intercept']:+.2f}, slope {R.loc[m, 'cal_slope']:.2f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Predicted probability (decile mean)"); ax.set_ylabel("Observed 1-year mortality")
ax.set_title("Fig. C2 — Calibration, CLIN held-out set", loc="left", fontsize=11)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(F / "figC2_calibration_clin_holdout.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 4.4))
d0 = new_dca[new_dca.model == MODELS[0]]
ax.plot(d0.threshold, d0.net_benefit_all, color="#8a8984", lw=1.2, ls="--", label="Treat all")
ax.axhline(0, color="#b9b8b3", lw=1, label="Treat none")
for m in MODELS:
    d = new_dca[new_dca.model == m]
    ax.plot(d.threshold, d.net_benefit_model, color=COLORS[m], lw=2.2 if m == "LR (L2)" else 1.5, label=m)
    ax.axvline(THR[m], color=COLORS[m], lw=0.8, ls=":")
ax.set_xlim(0.05, 0.60); ax.set_ylim(-0.05, max(0.30, float(new_dca.net_benefit_model.max()) + 0.02))
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_title("Fig. C3 — Decision curve, CLIN held-out set (dotted: frozen thresholds)", loc="left", fontsize=10.5)
ax.legend(fontsize=8)
fig.savefig(F / "figC3_dca_clin_holdout.png"); plt.close(fig)

# ---------------------------------------------------------------- nested-CV pooled OOF calibration (training set)
cache = path("interim") / "nested_cache" / "clinical_clin"
summ = pd.read_csv(CL / "nested_cv_full_summary_clin.csv").set_index("model")
split_tr = split[split.split == "train"].reset_index(drop=True)
ftr = pd.read_parquet(P / "features_clin.parquet", columns=["stay_id", "y"], filters=[("stay_id", "in", list(tr_ids))])
ftr = ftr.set_index("stay_id").loc[split_tr.stay_id.values]
ytr = ftr.y.values.astype(int)
n_rep = len(load_config()["nested_cv"]["seeds"])
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1, label="Ideal")
for m, slug in [("LR (L2)", "tuned_LR"), ("RF (tuned)", "tuned_RF")]:
    yy, pp = [], []
    for r in range(n_rep):
        pr = np.full(len(ytr), np.nan)
        for k in range(10):
            c = pickle.load(open(cache / f"r{r}_f{k}_{slug}.pkl", "rb"))
            pr[c["test_idx"]] = c["pred"]
        assert not np.isnan(pr).any()
        yy.append(ytr); pp.append(pr)
    obs, prd = calibration_curve(np.concatenate(yy), np.concatenate(pp), n_bins=10, strategy="quantile")
    ax.plot(prd, obs, color=COLORS[m], marker=MARKERS[m], ms=5, lw=2.2 if m == "LR (L2)" else 1.5,
            label=f"{m}  pooled-OOF slope {summ.loc[m, 'cal_slope_pooled_mean']:.2f} (mean of {n_rep} repeats)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Predicted probability (decile mean)"); ax.set_ylabel("Observed 1-year mortality")
ax.set_title(f"Fig. C4 — Calibration, CLIN nested-CV OOF (training set, {n_rep} × 10 folds)", loc="left", fontsize=10.5)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(F / "figC4_calibration_clin_nested_oof.png"); plt.close(fig)
print("figures written: figC1_roc_clin_holdout.png, figC2_calibration_clin_holdout.png, figC3_dca_clin_holdout.png, "
      "figC4_calibration_clin_nested_oof.png")

# ---------------------------------------------------------------- D8.15 post-hoc diagnostics of the CV-holdout gap
# (descriptive; same computations as recorded in D8.15, written to a table so that the report is generated from files)
from sklearn.metrics import roc_auc_score
ftr_all = pd.read_parquet(P / "features_clin.parquet", filters=[("stay_id", "in", list(tr_ids))]).set_index("stay_id").loc[split_tr.stay_id.values]
Xtr_s = ((ftr_all[b["kept"]].fillna(b["fill"]) - b["mu"]) / b["sd"])[sel]
apparent = roc_auc_score(ytr, lr.predict_proba(Xtr_s)[:, 1])
rng = np.random.default_rng(42)
pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
sub = []
for r in range(n_rep):
    pr = np.empty(len(ytr))
    for k in range(10):
        c = pickle.load(open(cache / f"r{r}_f{k}_tuned_LR.pkl", "rb"))
        pr[c["test_idx"]] = c["pred"]
    for _ in range(2000):
        i = np.concatenate([rng.choice(pos, n_pos, replace=True), rng.choice(neg, n_neg, replace=True)])
        sub.append(roc_auc_score(ytr[i], pr[i]))
sub = np.array(sub)
smd = pd.Series({c: (f[c].mean() - ftr_all[c].mean()) / np.sqrt((f[c].var() + ftr_all[c].var()) / 2) for c in sel})
diag = pd.DataFrame([
    {"item": "apparent_auroc_train_lr_l2", "value": apparent},
    {"item": "oof_subsample_auroc_mean", "value": sub.mean()},
    {"item": "oof_subsample_auroc_sd", "value": sub.std()},
    {"item": "oof_subsample_auroc_p2.5", "value": np.percentile(sub, 2.5)},
    {"item": "oof_subsample_p_le_holdout_auroc", "value": float((sub <= R.loc["LR (L2)", "auroc"]).mean())},
    {"item": "oof_subsample_n_draws", "value": len(sub)},
    {"item": "smd_max_abs", "value": float(smd.abs().max())},
    {"item": "smd_max_abs_feature", "value": smd.abs().idxmax()},
    {"item": "smd_n_abs_gt_0.1", "value": int((smd.abs() > 0.1).sum())},
    {"item": "smd_features_abs_gt_0.1", "value": ", ".join(smd[smd.abs() > 0.1].abs().sort_values(ascending=False).index)},
    {"item": "smd_n_features", "value": len(smd)}])
diag.to_csv(CL / "holdout_gap_diagnostics_clin.csv", index=False)
print(diag.to_string(index=False))
