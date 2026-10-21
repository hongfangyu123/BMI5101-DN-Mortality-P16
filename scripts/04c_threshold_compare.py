"""Stage 4c — decision thresholds, pairwise model comparison, LR interpretation and hypertension definitions.

No-leakage preprocessing (primary), both hyper-parameter sets. Uses the stage-4 LASSO selection, tuned
hyper-parameters and saved predictions; models are rebuilt from src/models.py and checked against the saved
validation predictions before use.

  1. Youden cut-off per model from 10-fold out-of-fold (OOF) predictions on the training set (primary) and from
     the apparent (in-sample) training predictions (sensitivity); validation accuracy / precision / recall / F1 /
     specificity at 0.5 and at the Youden cut-offs            -> tables/threshold_metrics.csv, youden_thresholds.csv
  2. Pairwise paired DeLong tests of validation AUCs, Holm-adjusted within each family: paper-parameter models
     (excluding LR), tuned models, and the recommended model (unpenalised LR, i.e. the paper-parameter LR)
     against each of the other eight models              -> tables/delong_pairwise.csv
  3. Recommended model = unpenalised LR (decisions.md D4.15); tuned LR (L2, C=0.01) kept as comparison:
     OOF and validation discrimination / calibration of both LR variants -> tables/lr_variants.csv;
     odds ratios with 95% CI, VIF; SHAP ranking (LinearExplainer) of both LR variants on the validation cohort
     (the recommended LR's SHAP figures are drawn in stage 4 as fig6a/6b_*_primary_*)
                                                              -> tables/lr_coefficients.csv, shap_importance_lr.csv, fig8
  4. Decision curves of the tuned models and the unpenalised LR -> tables/dca_models_noleak.csv, fig5b
  5. Hypertension definitions (full Quan-Elixhauser vs uncomplicated 401/I10), aggregate counts by outcome
                                                              -> tables/hypertension_definitions.csv
Only aggregates are written or printed.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.common import load_config, path
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.metrics import (cal_intercept_slope, decision_curve, delong_roc_test, threshold_metrics, useful_range,
                         youden_threshold)
from src.models import SCALED_MODELS, paper_models, tuned_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

cfg = load_config()
SEED, K = cfg["seed"], cfg["model"]["cv_folds"]
P, T, F = path("processed"), path("tables"), path("figures")
N_JOBS = int(cfg["duckdb"]["threads"])
CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
VERSION = "noleak"

UNPEN = "LR (unpenalised)"   # = paper-parameter LR (C = inf, R glm); recommended model
COLORS = {"RF": "#2a78d6", "XGBoost": "#eb6834", "SVM": "#1baf7a", "LR": "#b9a36b", UNPEN: "#eda100",
          "RF (R defaults)": "#e87ba4"}
STYLES = {"RF": "-", "XGBoost": "--", "SVM": "-.", "LR": ":", UNPEN: "-", "RF (R defaults)": (0, (5, 1, 1, 1))}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

sel = pd.read_csv(T / "lasso_selection.csv").set_index("version").loc[VERSION, "selected_1se"].split(", ")
tuned = json.load(open(T / "tuned_hyperparameters.json"))
Xs_all = pd.read_parquet(P / f"{VERSION}_scaled.parquet")
Xr_all = pd.read_parquet(P / f"{VERSION}_imputed.parquet")
assert (Xs_all.stay_id.values == Xr_all.stay_id.values).all()
tr, va = (Xs_all.split == "train").values, (Xs_all.split == "valid").values
y = {"train": Xs_all.y.values[tr], "valid": Xs_all.y.values[va]}
Xtr_s, Xva_s = Xs_all.loc[tr, sel], Xs_all.loc[va, sel]
Xtr_r, Xva_r = Xr_all.loc[tr, sel], Xr_all.loc[va, sel]
def data(m):  return (Xtr_s, Xva_s) if m in SCALED_MODELS else (Xtr_r, Xva_r)

thr_rows, youden_rows, delong_rows, fitted, oofs, vals = [], [], [], {}, {}, {}
for pset in ["paper", "tuned"]:
    tag = f"{VERSION}_{pset}"
    saved = pd.read_parquet(P / f"preds_{tag}.parquet")
    assert (saved.stay_id.values == np.concatenate([Xs_all.stay_id.values[tr], Xs_all.stay_id.values[va]])).all()
    s_tr, s_va = saved[saved.split == "train"], saved[saved.split == "valid"]
    models = paper_models(SEED) if pset == "paper" else \
        {m: tuned_model(m, tuned[f"{VERSION}/{m}"], len(sel), SEED) for m in ["RF", "XGBoost", "SVM", "LR"]}
    for m, est in models.items():
        a, b = data(m)
        fit = clone(est).fit(a, y["train"])
        diff = np.abs(fit.predict_proba(b)[:, 1] - s_va[m].values).max()
        assert diff < 1e-6, f"{tag}/{m}: rebuilt model does not reproduce stage-4 predictions ({diff:.2e})"
        fitted[(pset, m)] = fit
        oof = cross_val_predict(clone(est), a, y["train"], cv=CV, method="predict_proba", n_jobs=N_JOBS)[:, 1]
        oofs[(pset, m)], vals[(pset, m)] = oof, s_va[m].values
        t_oof, j_oof = youden_threshold(y["train"], oof)
        t_app, j_app = youden_threshold(y["train"], s_tr[m].values)
        youden_rows.append({"params": pset, "model": m, "threshold_oof": t_oof, "youden_j_train_oof": j_oof,
                            "threshold_apparent": t_app, "youden_j_train_apparent": j_app})
        for rule, t in [("0.5", 0.5), ("youden_oof", t_oof), ("youden_apparent", t_app)]:
            pv = s_va[m].values
            thr_rows.append({"params": pset, "model": m, "rule": rule, "threshold": t,
                             **threshold_metrics(y["valid"], pv, t), "flagged_pct": float(np.mean(pv >= t))})
        print(f"[{tag}] {m}: Youden cut-off OOF {t_oof:.3f} (J {j_oof:.3f}) | apparent {t_app:.3f} (J {j_app:.3f})",
              flush=True)

    names = [m for m in models if not (pset == "paper" and m == "LR")]   # paper LR = recommended, own family below
    rows = []
    for m1, m2 in combinations(names, 2):
        rows.append({"params": pset, "model_1": m1, "model_2": m2,
                     **delong_roc_test(y["valid"], s_va[m1].values, s_va[m2].values)})
    rows = pd.DataFrame(rows)
    rows["p_holm"] = multipletests(rows.p_value, method="holm")[1]
    delong_rows.append(rows)

# recommended model (unpenalised LR = paper-parameter LR) against every other model, Holm-adjusted within this family
others = [("paper", m) for m in ["RF", "RF (R defaults)", "XGBoost", "SVM"]] + \
         [("tuned", m) for m in ["RF", "XGBoost", "SVM", "LR"]]
rows = pd.DataFrame([{"params": "recommended", "model_1": UNPEN,
                      "model_2": f"{'LR (L2)' if m == 'LR' else m} ({ps})",
                      **delong_roc_test(y["valid"], vals[("paper", "LR")], vals[(ps, m)])} for ps, m in others])
rows["p_holm"] = multipletests(rows.p_value, method="holm")[1]
delong_rows.append(rows)

thr = pd.DataFrame(thr_rows)
thr.round(4).to_csv(T / "threshold_metrics.csv", index=False)
pd.DataFrame(youden_rows).round(4).to_csv(T / "youden_thresholds.csv", index=False)
dl = pd.concat(delong_rows, ignore_index=True)
dl.round(4).to_csv(T / "delong_pairwise.csv", index=False)

# ---------------------------------------------------------------- LR variants: discrimination and calibration
vrows = []
for label, key in [(UNPEN, ("paper", "LR")), ("LR (L2, C=0.01)", ("tuned", "LR"))]:
    row = {"variant": label}
    for split_, yy_, pp_ in [("oof_train", y["train"], oofs[key]), ("valid", y["valid"], vals[key])]:
        citl, slope = cal_intercept_slope(yy_, pp_)
        row.update({f"auc_{split_}": roc_auc_score(yy_, pp_), f"brier_{split_}": brier_score_loss(yy_, pp_),
                    f"cal_in_the_large_{split_}": citl, f"cal_slope_{split_}": slope})
    vrows.append(row)
lrv = pd.DataFrame(vrows)
lrv.round(6).to_csv(T / "lr_variants.csv", index=False)

# ---------------------------------------------------------------- LR interpretation (recommended: unpenalised)
lr_unpen, lr_t = fitted[("paper", "LR")], fitted[("tuned", "LR")]
glm = sm.GLM(y["train"], sm.add_constant(Xtr_s), family=sm.families.Binomial()).fit()
ci = glm.conf_int()
# same model as lr_unpen; differences come from the lbfgs stopping tolerance (agree to 1e-7 with tol=1e-10)
assert np.allclose(lr_unpen.coef_[0], glm.params[sel].values, atol=5e-3)
Xc = sm.add_constant(Xtr_s).values
coef = pd.DataFrame({
    "feature": sel,
    "sd_original_units": Xtr_r.std(ddof=0).values,     # 1 SD of the standardised predictor, in original units
    "coef_glm_per_sd": glm.params[sel].values, "or_glm_per_sd": np.exp(glm.params[sel].values),
    "or_glm_lo": np.exp(ci.loc[sel, 0].values), "or_glm_hi": np.exp(ci.loc[sel, 1].values),
    "p_glm": glm.pvalues[sel].values,
    "coef_tuned_per_sd": lr_t.coef_[0], "or_tuned_per_sd": np.exp(lr_t.coef_[0]),
    "vif": [variance_inflation_factor(Xc, i + 1) for i in range(len(sel))],
}).sort_values("coef_glm_per_sd", key=abs, ascending=False)
coef.round(4).to_csv(T / "lr_coefficients.csv", index=False)
print(f"[LR] GLM intercept {glm.params['const']:.3f}; tuned (L2) intercept {lr_t.intercept_[0]:.3f}")

shap_rows = []
for label, mdl in [(UNPEN, lr_unpen), ("LR (L2, C=0.01)", lr_t)]:
    sv = shap.LinearExplainer(mdl, Xtr_s)(Xva_s)
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=sel).sort_values(ascending=False)
    shap_rows += [{"model": label, "rank": r, "feature": f_, "mean_abs_shap_logodds": v}
                  for r, (f_, v) in enumerate(imp.items(), 1)]
shp_lr = pd.DataFrame(shap_rows)
shp_lr.to_csv(T / "shap_importance_lr.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
c = coef.sort_values("coef_glm_per_sd")
yy = np.arange(len(c))
ax.errorbar(c.or_glm_per_sd, yy, xerr=[c.or_glm_per_sd - c.or_glm_lo, c.or_glm_hi - c.or_glm_per_sd], fmt="o",
            color="#2a78d6", ecolor="#8fb6e8", ms=5, capsize=2, label="Unpenalised LR, recommended (95% CI)")
ax.scatter(c.or_tuned_per_sd, yy + 0.22, marker="D", s=18, color="#eda100", label="Tuned LR (L2, C=0.01), comparison",
           zorder=3)
ax.axvline(1, color=INK2, lw=1)
ax.set_xscale("log"); ax.set_xticks([0.6, 0.8, 1, 1.25, 1.5, 2]); ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.5", "2"])
ax.minorticks_off(); ax.set_yticks(yy); ax.set_yticklabels(c.feature); ax.grid(axis="y", visible=False)
ax.set_xlabel("Odds ratio per 1 SD (log scale)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Fig. 8 — LR odds ratios (training cohort, standardised predictors)", loc="left", fontsize=11)
fig.savefig(F / "fig8_lr_odds_ratios_noleak.png"); plt.close(fig)

# ---------------------------------------------------------------- decision curves: tuned models + unpenalised LR
dca_models = {UNPEN: vals[("paper", "LR")], **{m: vals[("tuned", m)] for m in ["RF", "XGBoost", "SVM", "LR"]}}
dca = pd.concat([decision_curve(y["valid"], p_).assign(model=m) for m, p_ in dca_models.items()])
dca.round(5).to_csv(T / f"dca_models_{VERSION}.csv", index=False)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for m, d in dca.groupby("model", sort=False):
    ax.plot(d.threshold, d.net_benefit_model, color=COLORS[m], ls=STYLES[m], lw=2.4 if m == UNPEN else 1.6,
            label=m if m == UNPEN else f"{m} (tuned)", zorder=3 if m == UNPEN else 2)
d0 = dca[dca.model == UNPEN]
ax.plot(d0.threshold, d0.net_benefit_all, color="#b9b8b3", ls="--", lw=1.5, label="Treat all")
ax.plot(d0.threshold, d0.net_benefit_none, color=INK2, lw=1, label="Treat none")
ax.set_ylim(-0.05, 0.3); ax.set_xlim(0, 1); ax.legend(fontsize=8)
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_title("Fig. 5B — Decision curves (validation cohort)", loc="left", fontsize=11)
fig.savefig(F / f"fig5b_dca_models_{VERSION}.png"); plt.close(fig)
for m, d in dca.groupby("model", sort=False):
    lo, hi = useful_range(d)
    print(f"[DCA] {m}: net benefit > treat-all/none for thresholds {lo:.2f}–{hi:.2f}")

# ---------------------------------------------------------------- hypertension definitions (aggregate)
ft = pd.read_parquet(P / "features.parquet")[["hypertension", "hypertension_uncomplicated", "y"]]
groups = {
    "full Quan-Elixhauser (401–405 / I10–I13, I15, I16) — main analysis": ft.hypertension == 1,
    "uncomplicated only (401 / I10) — sensitivity, ≈ paper": ft.hypertension_uncomplicated == 1,
    "complicated codes without 401/I10 (coded 'no' by the uncomplicated definition)":
        (ft.hypertension == 1) & (ft.hypertension_uncomplicated == 0),
    "no hypertension code": ft.hypertension == 0,
}
hrows = []
for name, g in groups.items():
    tab = pd.crosstab(g, ft.y)
    assert tab.values.min() >= 10, "small cell"
    a_, b_, c_, d_ = tab.loc[True, 1], tab.loc[True, 0], tab.loc[False, 1], tab.loc[False, 0]
    or_ = (a_ * d_) / (b_ * c_); se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    hrows.append({"definition": name, "n": int(g.sum()), "pct_all": 100 * g.mean(),
                  "pct_survivors": 100 * g[ft.y == 0].mean(), "pct_deaths": 100 * g[ft.y == 1].mean(),
                  "mortality_pct": 100 * ft.y[g].mean(), "or_death": or_,
                  "or_lo": np.exp(np.log(or_) - 1.96 * se), "or_hi": np.exp(np.log(or_) + 1.96 * se),
                  "p_chi2": stats.chi2_contingency(tab.values, correction=False)[1]})
pd.DataFrame(hrows).round(4).to_csv(T / "hypertension_definitions.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== validation threshold metrics ===")
print(thr.round(3).to_string(index=False))
print("\n=== pairwise DeLong (validation) ===")
print(dl.round(4).to_string(index=False))
print("\n=== LR variants (OOF training / validation) ===")
print(lrv.round(3).to_string(index=False))
print("\n=== LR coefficients ===")
print(coef.round(3).to_string(index=False))
for m_, g_ in shp_lr.groupby("model", sort=False):
    print(f"SHAP top5 {m_}:", g_.sort_values("rank").feature.tolist()[:5])
print("\n=== hypertension definitions ===")
print(pd.DataFrame(hrows).round(3).to_string(index=False))
