"""Stage 4 — LASSO feature selection, four models (paper hyper-parameters and 10-fold-CV tuned),
evaluation (AUC + DeLong 95% CI, accuracy, precision, recall, F1; death = positive class),
DCA / clinical impact curve and SHAP for the selected model.

Run for both preprocessing versions (faithful, noleak). Primary analysis = noleak + tuned.
Patient-level predictions go to data/processed (git-ignored); only aggregates are printed.
"""
from __future__ import annotations

import json
import os
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
import shap
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.common import load_config, path
from src.lasso import cv_lasso
from src.metrics import clinical_impact, decision_curve, evaluate, useful_range
from src.models import SCALED_MODELS, paper_models, tuning_grids
from src.paper_ref import LASSO_SELECTED
from src.variables import FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

cfg = load_config()
SEED, K = cfg["seed"], cfg["model"]["cv_folds"]
P, T, F = path("processed"), path("tables"), path("figures")
F.mkdir(parents=True, exist_ok=True)
N_JOBS = int(cfg["duckdb"]["threads"])
CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

# ---- figure style (reference categorical palette, fixed order; colour follows the model)
COLORS = {"RF": "#2a78d6", "XGBoost": "#eb6834", "SVM": "#1baf7a", "LR": "#eda100", "RF (R defaults)": "#e87ba4",
          "LR (unpenalised)": "#eda100", "LR (L2)": "#b9a36b"}
STYLES = {"RF": "-", "XGBoost": "--", "SVM": "-.", "LR": ":", "RF (R defaults)": (0, (5, 1, 1, 1)),
          "LR (unpenalised)": "-", "LR (L2)": ":"}
UNPEN = "LR (unpenalised)"   # recommended model = paper-parameter LR (C = inf), decisions.md D4.15
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- figures
def fig_lasso(res, version):
    x = np.log(res.lambdas)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(x, res.cvm, yerr=res.cvsd, fmt="o", ms=3.5, color="#e34948", ecolor="#b9b8b3", elinewidth=1, capsize=2)
    for lam, lab, ha in [(res.lambda_min, "λ.min", "right"), (res.lambda_1se, "λ.1se", "left")]:
        ax.axvline(np.log(lam), color=INK2, ls="--", lw=1)
        pad = " " if ha == "left" else ""
        ax.text(np.log(lam), ax.get_ylim()[1], f"{pad}{lab} {pad}\n{pad}λ={lam:.4f} ", va="top", ha=ha,
                fontsize=8, color=INK2)
    top = ax.secondary_xaxis("top")
    idx = np.linspace(0, len(x) - 1, 12).astype(int)
    top.set_xticks(x[idx]); top.set_xticklabels(res.nzero[idx]); top.set_xlabel("non-zero coefficients", color=INK2)
    ax.set_xlabel("log(λ)"); ax.set_ylabel("Binomial deviance (10-fold CV)")
    ax.set_title(f"Fig. 2A — LASSO cross-validation ({version})", loc="left", fontsize=11)
    fig.savefig(F / f"fig2a_lasso_cv_{version}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sel = set(res.selected("1se"))
    for j, f in enumerate(res.features):
        ax.plot(x, res.coefs[:, j], lw=1.6 if f in sel else 0.8, color="#2a78d6" if f in sel else "#b9b8b3",
                zorder=3 if f in sel else 2)
    for lam in (res.lambda_min, res.lambda_1se):
        ax.axvline(np.log(lam), color=INK2, ls="--", lw=1)
    i = int(np.argmin(np.abs(res.lambdas - res.lambdas[-1])))
    order = np.argsort(-np.abs(res.coefs[i]))
    for j in [j for j in order if res.features[j] in sel][:12]:
        ax.text(x[-1], res.coefs[-1, j], " " + res.features[j], fontsize=6.5, va="center", color=INK2)
    ax.set_xlabel("log(λ)"); ax.set_ylabel("Coefficient (standardised)")
    ax.set_title(f"Fig. 2B — LASSO coefficient paths ({version}); blue = selected at λ.1se", loc="left", fontsize=11)
    fig.savefig(F / f"fig2b_lasso_path_{version}.png"); plt.close(fig)


def fig_roc(preds, y, tag, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, split in zip(axes, ["train", "valid"]):
        for m, pr in preds.items():
            e = evaluate(y[split], pr[split])
            fpr, tpr, _ = roc_curve(y[split], pr[split])
            ax.plot(fpr, tpr, color=COLORS[m], ls=STYLES[m], lw=2.6 if m == UNPEN else 1.8, zorder=3 if m == UNPEN else 2,
                    label=f"{m}  AUC {e['auc']:.3f} ({e['auc_lo']:.3f}–{e['auc_hi']:.3f})")
        ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1)
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.set_aspect("equal")
        ax.set_title(f"{'A' if split == 'train' else 'B'}  {'Training' if split == 'train' else 'Validation'} cohort",
                     loc="left", fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.savefig(F / f"fig3_roc_{tag}.png"); plt.close(fig)


def fig_metrics(perf, tag, title):
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = list(dict.fromkeys(perf.model))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    w = 0.8 / len(models)
    for ax, split in zip(axes, ["train", "valid"]):
        d = perf[perf.split == split].set_index("model")
        for k, m in enumerate(models):
            xs = np.arange(len(metrics)) + (k - (len(models) - 1) / 2) * w
            vals = d.loc[m, metrics].astype(float).values
            ax.bar(xs, vals, width=w - 0.02, color=COLORS[m], label=m, edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(np.arange(len(metrics))); ax.set_xticklabels([m.capitalize() if m != "f1" else "F1" for m in metrics])
        ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
        ax.set_title(f"{'A' if split == 'train' else 'B'}  {'Training' if split == 'train' else 'Validation'} cohort",
                     loc="left", fontsize=11)
    axes[0].set_ylabel("Value (threshold 0.5, death = positive)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.savefig(F / f"fig4_metrics_{tag}.png"); plt.close(fig)


def fig_dca_cic(y, p, tag, label):
    dca, cic = decision_curve(y, p), clinical_impact(y, p)
    dca.to_csv(T / f"dca_{tag}.csv", index=False); cic.to_csv(T / f"cic_{tag}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(dca.threshold, dca.net_benefit_model, color="#2a78d6", label=label)
    ax.plot(dca.threshold, dca.net_benefit_all, color="#b9b8b3", ls="--", lw=1.5, label="Treat all")
    ax.plot(dca.threshold, dca.net_benefit_none, color=INK2, lw=1, label="Treat none")
    ax.set_ylim(-0.05, max(0.05, dca.net_benefit_model.max() * 1.15)); ax.set_xlim(0, 1)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.legend(fontsize=8)
    ax.set_title("A  Decision curve (validation cohort)", loc="left", fontsize=11)
    ax = axes[1]
    ax.plot(cic.threshold, cic.n_high_risk, color="#e34948", label="Number high risk")
    ax.plot(cic.threshold, cic.n_high_risk_with_event, color="#2a78d6", ls="--", label="Number high risk with event")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Number per 1,000 patients"); ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("B  Clinical impact curve (validation cohort)", loc="left", fontsize=11)
    fig.savefig(F / f"fig5_dca_cic_{tag}.png"); plt.close(fig)
    return dca


def shap_analysis(model, name, Xtr, Xva, Xva_display, cases, tag, label=None):
    """SHAP on the validation cohort. Returns mean |SHAP| ranking (aggregate)."""
    if name.startswith("RF") or name == "XGBoost":
        expl = shap.TreeExplainer(model)
        sv = expl(Xva)
        if sv.values.ndim == 3:            # sklearn RF: (n, p, classes) -> death class
            sv = sv[:, :, 1]
        units = "probability" if name.startswith("RF") else "log-odds"
    elif name == "LR":
        expl = shap.LinearExplainer(model, Xtr)
        sv = expl(Xva); units = "log-odds"
    else:
        bg = shap.kmeans(Xtr.values, 30)
        f = lambda a: model.predict_proba(a)[:, 1]
        expl = shap.KernelExplainer(f, bg)
        vals = expl.shap_values(Xva.values, nsamples=300, silent=True)
        sv = shap.Explanation(vals, base_values=np.repeat(expl.expected_value, len(Xva)), data=Xva.values,
                              feature_names=list(Xva.columns))
        units = "probability"
    sv_disp = shap.Explanation(sv.values, base_values=sv.base_values, data=Xva_display.values,
                               feature_names=list(Xva.columns))
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=Xva.columns).sort_values(ascending=False)
    name = label or name

    plt.figure(); shap.plots.bar(sv_disp, max_display=len(imp), show=False)
    plt.title(f"Fig. 6A — mean |SHAP| ({name}, {units})", loc="left", fontsize=11)
    plt.savefig(F / f"fig6a_shap_bar_{tag}.png"); plt.close("all")
    plt.figure(); shap.plots.beeswarm(sv_disp, max_display=len(imp), show=False)
    plt.title(f"Fig. 6B — SHAP beeswarm ({name}, validation cohort)", loc="left", fontsize=11)
    plt.savefig(F / f"fig6b_shap_beeswarm_{tag}.png"); plt.close("all")
    base = float(np.ravel(sv.base_values)[0])
    for letter, (label, i) in zip("CDE", cases.items()):
        shap.plots.force(base, sv.values[i], np.round(Xva_display.values[i], 2), feature_names=list(Xva.columns),
                         matplotlib=True, show=False, text_rotation=15)
        plt.title(f"Fig. 6{letter} — force plot, validation case: {label}", loc="left", fontsize=10, y=1.35)
        plt.savefig(F / f"fig6{letter.lower()}_force_{label.replace(' ', '_')}_{tag}.png"); plt.close("all")
    return imp


# ---------------------------------------------------------------- main loop
cohort = pd.read_parquet(P / "cohort.parquet")[["stay_id", "death_inhosp"]]
perf_rows, lasso_rows, tuned_params, runs = [], [], {}, {}

for version in ["faithful", "noleak"]:
    Xs_all = pd.read_parquet(P / f"{version}_scaled.parquet")
    Xr_all = pd.read_parquet(P / f"{version}_imputed.parquet")
    assert (Xs_all.stay_id.values == Xr_all.stay_id.values).all()
    tr, va = (Xs_all.split == "train").values, (Xs_all.split == "valid").values
    y = {"train": Xs_all.y.values[tr], "valid": Xs_all.y.values[va]}

    # -------- LASSO (standardised training data)
    res = cv_lasso(Xs_all.loc[tr, FEATURES].values, y["train"], FEATURES, n_folds=K, seed=SEED)
    sel, sel_min = res.selected("1se"), res.selected("min")
    fig_lasso(res, version)
    lasso_rows.append({"version": version, "lambda_min": res.lambda_min, "lambda_1se": res.lambda_1se,
                       "n_min": len(sel_min), "n_1se": len(sel), "selected_1se": ", ".join(sel),
                       "selected_min": ", ".join(sel_min),
                       "overlap_with_paper": ", ".join(f for f in sel if f in LASSO_SELECTED),
                       "paper_not_selected": ", ".join(f for f in LASSO_SELECTED if f not in sel)})
    i1 = int(np.argmin(np.abs(res.lambdas - res.lambda_1se)))
    pd.DataFrame({"feature": FEATURES, "coef_1se_standardised": res.coefs[i1]}).query("coef_1se_standardised != 0") \
        .sort_values("coef_1se_standardised", key=abs, ascending=False).to_csv(T / f"lasso_coefs_{version}.csv", index=False)
    print(f"[{version}] LASSO λ.min={res.lambda_min:.5f} ({len(sel_min)} vars)  λ.1se={res.lambda_1se:.5f} ({len(sel)} vars)")
    print(f"[{version}] selected (1se): {sel}", flush=True)

    Xtr_s, Xva_s = Xs_all.loc[tr, sel], Xs_all.loc[va, sel]
    Xtr_r, Xva_r = Xr_all.loc[tr, sel], Xr_all.loc[va, sel]
    def data(m):  return (Xtr_s, Xva_s) if m in SCALED_MODELS else (Xtr_r, Xva_r)

    for pset in ["paper", "tuned"]:
        fitted, preds, cv_auc = {}, {}, {}
        if pset == "paper":
            for m, est in paper_models(SEED).items():
                a, b = data(m)
                cv_auc[m] = cross_val_score(clone(est), a, y["train"], cv=CV, scoring="roc_auc", n_jobs=N_JOBS).mean()
                fitted[m] = est.fit(a, y["train"])
        else:
            for m, (est, grid) in tuning_grids(len(sel), SEED).items():
                a, b = data(m)
                gs = GridSearchCV(est, grid, cv=CV, scoring="roc_auc", n_jobs=N_JOBS, refit=True).fit(a, y["train"])
                best = gs.best_estimator_
                if m == "SVM":   # refit with Platt scaling for probabilities
                    best = SVC(kernel="rbf", probability=True, random_state=SEED, **gs.best_params_).fit(a, y["train"])
                fitted[m], cv_auc[m] = best, gs.best_score_
                tuned_params[f"{version}/{m}"] = {k: (v if not isinstance(v, float) or np.isfinite(v) else "inf")
                                                  for k, v in gs.best_params_.items()}
                print(f"[{version}/tuned] {m}: CV AUC {gs.best_score_:.3f} params {tuned_params[f'{version}/{m}']}", flush=True)
        for m, est in fitted.items():
            a, b = data(m)
            preds[m] = {"train": est.predict_proba(a)[:, 1], "valid": est.predict_proba(b)[:, 1]}
            for split in ["train", "valid"]:
                perf_rows.append({"version": version, "params": pset, "model": m, "split": split,
                                  "cv_auc_train": cv_auc[m], **evaluate(y[split], preds[m][split])})
        tag = f"{version}_{pset}"
        perf = pd.DataFrame([r for r in perf_rows if r["version"] == version and r["params"] == pset])
        title = f"{'Faithful' if version == 'faithful' else 'No-leakage'} preprocessing, " \
                f"{'paper' if pset == 'paper' else '10-fold-CV tuned'} hyper-parameters"
        fig_roc(preds, y, tag, "Fig. 3 — ROC curves. " + title)
        fig_metrics(perf, tag, "Fig. 4 — Performance metrics. " + title)
        pd.DataFrame({"stay_id": np.concatenate([Xs_all.stay_id.values[tr], Xs_all.stay_id.values[va]]),
                      "split": ["train"] * tr.sum() + ["valid"] * va.sum(),
                      **{m: np.concatenate([p["train"], p["valid"]]) for m, p in preds.items()}}) \
            .to_parquet(P / f"preds_{tag}.parquet", index=False)
        runs[tag] = dict(fitted=fitted, preds=preds, cv_auc=cv_auc, y=y, sel=sel, version=version,
                         data=(Xtr_s, Xva_s, Xtr_r, Xva_r), va_ids=Xs_all.stay_id.values[va])

perf = pd.DataFrame(perf_rows)
perf.round(4).to_csv(T / "model_performance.csv", index=False)
pd.DataFrame(lasso_rows).to_csv(T / "lasso_selection.csv", index=False)
json.dump(tuned_params, open(T / "tuned_hyperparameters.json", "w"), indent=2, default=str)

# ---------------------------------------------------------------- DCA / CIC / SHAP for selected models
shap_rows = []
targets = {
    # primary = recommended model (D4.15): unpenalised LR, i.e. the paper-parameter LR, no-leakage preprocessing
    "primary": ("noleak_paper", "LR", "primary_noleak_recommended", UNPEN),
    # secondary = model chosen by the pre-specified rule D4.7 (best 10-fold CV AUC, no-leakage + tuned)
    "secondary": ("noleak_tuned", max(runs["noleak_tuned"]["cv_auc"], key=runs["noleak_tuned"]["cv_auc"].get),
                  "secondary_noleak_tuned", None),
    # the paper's own final configuration: faithful preprocessing, paper hyper-parameters, RF
    "paperconfig": ("faithful_paper", "RF", "paperconfig_faithful_paper", None),
}
for label, (tag, m, ftag, disp) in targets.items():
    r = runs[tag]
    Xtr_s, Xva_s, Xtr_r, Xva_r = r["data"]
    a, b = (Xtr_s, Xva_s) if m in SCALED_MODELS else (Xtr_r, Xva_r)
    yv, pv = r["y"]["valid"], r["preds"][m]["valid"]
    dca = fig_dca_cic(yv, pv, ftag, f"{disp or m} ({label})")
    lo, hi = useful_range(dca)
    va_info = pd.DataFrame({"stay_id": r["va_ids"], "y": yv, "p": pv}).merge(cohort, on="stay_id", how="left")
    cats = {"survivor": (va_info.y == 0), "in-hospital death": (va_info.death_inhosp == 1),
            "post-discharge death within 1 year": (va_info.y == 1) & (va_info.death_inhosp == 0)}
    cases = {}
    for k, mask in cats.items():   # representative case: prediction closest to the category median
        sub = va_info[mask]
        cases[k] = int((sub.p - sub.p.median()).abs().idxmin())
    imp = shap_analysis(r["fitted"][m], m, a, b, Xva_r, cases, ftag, label=disp)
    for rank, (f, v) in enumerate(imp.items(), 1):
        shap_rows.append({"analysis": label, "config": tag, "model": disp or m, "rank": rank, "feature": f,
                          "mean_abs_shap": v})
    print(f"[{label}] {tag}/{m}: DCA net benefit > treat-all/none for thresholds {lo:.2f}–{hi:.2f}; "
          f"SHAP top5 {list(imp.index[:5])}")
pd.DataFrame(shap_rows).to_csv(T / "shap_importance.csv", index=False)

# ---------------------------------------------------------------- main ROC figure: recommended LR + tuned models
main = {UNPEN: runs["noleak_paper"]["preds"]["LR"],
        **{("LR (L2)" if m == "LR" else m): p for m, p in runs["noleak_tuned"]["preds"].items()}}
fig_roc(main, runs["noleak_tuned"]["y"], "noleak_main",
        "Fig. 3 — ROC curves. No-leakage preprocessing: recommended unpenalised LR and 10-fold-CV tuned models")

# ---------------------------------------------------------------- sensitivity: hypertension definition (LASSO only)
feat = pd.read_parquet(P / "features.parquet")[["stay_id", "hypertension_uncomplicated"]]
Xs = pd.read_parquet(P / "noleak_scaled.parquet").merge(feat, on="stay_id")
trm = (Xs.split == "train").values
h = Xs["hypertension_uncomplicated"].astype(float)
Xs["hypertension"] = (h - h[trm].mean()) / h[trm].std(ddof=0)
res_h = cv_lasso(Xs.loc[trm, FEATURES].values, Xs.y.values[trm], FEATURES, n_folds=K, seed=SEED)
pd.DataFrame([{"analysis": "noleak, hypertension = uncomplicated only (401/I10)", "lambda_1se": res_h.lambda_1se,
               "selected_1se": ", ".join(res_h.selected("1se"))}]).to_csv(T / "lasso_sensitivity_hypertension.csv", index=False)
print("[sensitivity] hypertension uncomplicated -> LASSO 1se:", res_h.selected("1se"))

pd.set_option("display.width", 220)
print("\n=== performance (death = positive, threshold 0.5) ===")
print(perf.round(3).to_string(index=False))
