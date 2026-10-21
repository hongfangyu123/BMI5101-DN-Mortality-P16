"""Stage 4b — calibration and overfitting checks from the saved stage-4 predictions.

Per configuration (version x hyper-parameter set) and model:
  - Brier score (train / validation), scaled Brier vs. the prevalence-only model
  - calibration-in-the-large (intercept with slope fixed at 1) and calibration slope (logistic recalibration)
  - reliability curve on the validation set (10 quantile bins) -> results/figures/fig7_calibration_<tag>.png
  - train-validation AUC gap (from model_performance.csv)
Outputs: results/tables/calibration.csv, results/tables/overfitting.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.common import path
from src.metrics import cal_intercept_slope

P, T, F = path("processed"), path("tables"), path("figures")
COLORS = {"RF": "#2a78d6", "XGBoost": "#eb6834", "SVM": "#1baf7a", "LR": "#eda100", "RF (R defaults)": "#e87ba4",
          "LR (unpenalised)": "#eda100", "LR (L2)": "#b9a36b"}
MARKERS = {"RF": "o", "XGBoost": "s", "SVM": "^", "LR": "D", "RF (R defaults)": "v", "LR (unpenalised)": "D", "LR (L2)": "P"}
UNPEN = "LR (unpenalised)"   # recommended model = paper-parameter LR (decisions.md D4.15)
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10})


split = pd.read_parquet(P / "split.parquet")[["stay_id", "y"]]
rows = []
for f in sorted(P.glob("preds_*.parquet")):
    tag = f.stem.replace("preds_", "")
    d = pd.read_parquet(f).merge(split, on="stay_id")
    models = [c for c in d.columns if c not in ("stay_id", "split", "y")]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1, label="Ideal")
    for m in models:
        for s in ["train", "valid"]:
            x = d[d.split == s]
            y, p = x.y.values, x[m].values
            prev = y.mean()
            brier = brier_score_loss(y, p)
            citl, slope = cal_intercept_slope(y, p)
            rows.append({"config": tag, "model": m, "split": s, "brier": brier,
                         "brier_null": prev * (1 - prev), "scaled_brier": 1 - brier / (prev * (1 - prev)),
                         "cal_in_the_large": citl, "cal_slope": slope, "mean_pred": p.mean(), "observed": prev})
            if s == "valid":
                obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
                ax.plot(pred, obs, color=COLORS[m], marker=MARKERS[m], ms=5, lw=1.6,
                        label=f"{m}  Brier {brier:.3f}, slope {slope:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability (decile mean)"); ax.set_ylabel("Observed 1-year mortality")
    ax.set_title(f"Fig. 7 — Calibration, validation cohort ({tag})", loc="left", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(F / f"fig7_calibration_{tag}.png"); plt.close(fig)

# ---- main calibration figure: recommended unpenalised LR + tuned models (validation cohort)
dp = pd.read_parquet(P / "preds_noleak_paper.parquet").merge(split, on="stay_id").query("split == 'valid'")
dt = pd.read_parquet(P / "preds_noleak_tuned.parquet").merge(split, on="stay_id").query("split == 'valid'")
assert (dp.stay_id.values == dt.stay_id.values).all()
main = {UNPEN: dp.LR.values, "RF": dt.RF.values, "XGBoost": dt.XGBoost.values, "SVM": dt.SVM.values, "LR (L2)": dt.LR.values}
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1, label="Ideal")
for m, p in main.items():
    citl, slope = cal_intercept_slope(dt.y.values, p)
    obs, pred = calibration_curve(dt.y.values, p, n_bins=10, strategy="quantile")
    ax.plot(pred, obs, color=COLORS[m], marker=MARKERS[m], ms=5, lw=2.4 if m == UNPEN else 1.4,
            zorder=3 if m == UNPEN else 2,
            label=f"{m}{' (recommended)' if m == UNPEN else ' (tuned)'}  Brier {brier_score_loss(dt.y.values, p):.3f}, slope {slope:.2f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Predicted probability (decile mean)"); ax.set_ylabel("Observed 1-year mortality")
ax.set_title("Fig. 7 — Calibration, validation cohort (no-leakage)", loc="left", fontsize=11)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(F / "fig7_calibration_noleak_main.png"); plt.close(fig)

cal = pd.DataFrame(rows)
cal.round(6).to_csv(T / "calibration.csv", index=False)

perf = pd.read_csv(T / "model_performance.csv")
w = perf.pivot_table(index=["version", "params", "model"], columns="split", values="auc").reset_index()
w = w.merge(perf[perf.split == "train"][["version", "params", "model", "cv_auc_train"]], on=["version", "params", "model"])
w["gap_train_minus_valid"] = w["train"] - w["valid"]
w["gap_train_minus_cv"] = w["train"] - w["cv_auc_train"]
w["flag_overfit"] = (w["gap_train_minus_valid"] > 0.10) | (w["train"] > 0.95)
w.round(4).to_csv(T / "overfitting.csv", index=False)

pd.set_option("display.width", 200)
print(cal[cal.split == "valid"].round(3).to_string(index=False))
print()
print(w.round(3).to_string(index=False))
