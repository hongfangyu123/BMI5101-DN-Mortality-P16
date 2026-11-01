"""Stage 5 — assemble results/report_en.md (template in scripts/templates/report_en.md) from the aggregate tables
(no patient-level data). Part IV and the extension summary are spliced in by hand between the <!-- BEGIN/END ... -->
markers and are lost when this script is rerun; re-insert them afterwards."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import ROOT, path
from src.metrics import useful_range
from src.variables import BINARY
from src.paper_ref import (AUC_TRAIN, AUC_VALID, COHORT_FLOW, LASSO_LAMBDA_1SE, LASSO_LAMBDA_MIN, LASSO_SELECTED,
                           N_DEATH, N_SURVIVAL, SHAP_TOP5)

T = path("tables")
R = ROOT / "results"


def md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            whole = d[c].dropna().apply(float.is_integer).all()
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else (f"{int(v)}" if whole else f"{v:.{floatfmt}f}"))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


flow = pd.read_csv(T / "cohort_flow.csv")
outc = pd.read_csv(T / "outcome_definitions.csv")
feat = pd.read_csv(T / "feature_vs_paper.csv")
mix = pd.read_csv(T / "unit_mixing_diagnostic.csv")
t1 = pd.read_csv(T / "table1.csv")
lasso = pd.read_csv(T / "lasso_selection.csv").set_index("version")
lcoef = pd.read_csv(T / "lasso_coefs_noleak.csv")
perf = pd.read_csv(T / "model_performance.csv")
cal = pd.read_csv(T / "calibration.csv")
ovf = pd.read_csv(T / "overfitting.csv")
shp = pd.read_csv(T / "shap_importance.csv")
tuned = json.load(open(T / "tuned_hyperparameters.json"))
lsens = pd.read_csv(T / "lasso_sensitivity_hypertension.csv")
thrm = pd.read_csv(T / "threshold_metrics.csv")
ydn = pd.read_csv(T / "youden_thresholds.csv")
dl = pd.read_csv(T / "delong_pairwise.csv")
lrc = pd.read_csv(T / "lr_coefficients.csv")
shp_lr = pd.read_csv(T / "shap_importance_lr.csv")
htn = pd.read_csv(T / "hypertension_definitions.csv")
dca_m = pd.read_csv(T / "dca_models_noleak.csv")
lrv = pd.read_csv(T / "lr_variants.csv").set_index("variant")
nest = pd.read_csv(T / "nested_cv_auc.csv")
nfreq = pd.read_csv(T / "nested_cv_selection_freq.csv")
nfold = pd.read_csv(T / "nested_cv_folds.csv")
sens = pd.read_csv(T / "sensitivity_pre2020.csv").set_index("model")

# ---------------------------------------------------------------- key numbers
C = flow[flow.definition == "C"].reset_index(drop=True)
n_final = int(C.n.iloc[-1]); d_final = int(C.deaths_1y_icu.iloc[-1])
o = outc[outc.definition == "C"].set_index("outcome")
prim_tag = "noleak_tuned"
cv = perf[(perf.version == "noleak") & (perf.params == "tuned") & (perf.split == "train")].set_index("model").cv_auc_train
best = cv.idxmax()
pv = perf[(perf.version == "noleak") & (perf.params == "tuned") & (perf.split == "valid")].set_index("model")
best_auc = pv.loc[best]
top5 = shp[(shp.analysis == "secondary")].sort_values("rank").feature.tolist()[:5]          # tuned RF (D4.7)
top5_paperconfig = shp[(shp.analysis == "paperconfig")].sort_values("rank").feature.tolist()[:5]
sel = lasso.loc["noleak", "selected_1se"].split(", ")
UNPEN, L2 = "LR (unpenalised)", "LR (L2, C=0.01)"
GROUP_ORDER = {"recommended": 0, "paper": 1, "tuned": 2}
def grp(d, col="params"):
    """The recommended unpenalised LR (= paper-parameter LR) gets its own group; tuned LR is labelled LR (L2)."""
    d = d.copy()
    rec_rows = (d[col] == "paper") & (d.model == "LR")
    d.loc[rec_rows, col] = "recommended"
    d.loc[rec_rows, "model"] = UNPEN
    d.loc[(d[col] == "tuned") & (d.model == "LR"), "model"] = "LR (L2)"
    return d.sort_values(col, key=lambda c: c.map(GROUP_ORDER), kind="stable")
top5_lr = shp_lr[shp_lr.model == UNPEN].sort_values("rank").feature.tolist()[:5]
assert shp[shp.analysis == "primary"].sort_values("rank").feature.tolist()[:5] == top5_lr   # stage-4 primary = recommended LR
top5_lr_l2 = shp_lr[shp_lr.model == L2].sort_values("rank").feature.tolist()[:5]
dlt = dl[dl.params == "tuned"]
dlr = dl[dl.params == "recommended"].set_index("model_2")
rf_lr = dlr.loc["RF (tuned)"]                       # recommended (unpenalised) LR vs tuned RF
rf_lr_l2 = dlt[(dlt.model_1 == "RF") & (dlt.model_2 == "LR")].iloc[0]
# the recommendation rule (D4.15): LR is recommended only if its AUC does not differ significantly from RF
assert rf_lr.p_value >= 0.05 and rf_lr_l2.p_value >= 0.05, "RF vs LR significant: the LR recommendation no longer applies"
assert dlt.p_holm.min() >= 0.05, "a tuned-model pair is significant after Holm: section 0 text no longer applies"
pv_paper = perf[(perf.version == "noleak") & (perf.params == "paper") & (perf.split == "valid")].set_index("model")
rec_auc = pv_paper.loc["LR"]                        # unpenalised LR = paper-parameter LR (C = inf)
cal_rec = cal[(cal.split == "valid") & (cal.config == "noleak_paper")].set_index("model").loc["LR"]
calt = cal[(cal.split == "valid") & (cal.config == "noleak_tuned")].set_index("model")
lr_thr = thrm[(thrm.params == "paper") & (thrm.model == "LR")].set_index("rule")       # recommended
l2_thr = thrm[(thrm.params == "tuned") & (thrm.model == "LR")].set_index("rule")       # comparison
cv_rec = perf[(perf.version == "noleak") & (perf.params == "paper") & (perf.model == "LR") & (perf.split == "train")].cv_auc_train.iloc[0]
_w = dca_m.pivot(index="threshold", columns="model", values="net_benefit_model").loc[0.1:0.4]
dca_spread = float((_w.max(axis=1) - _w.min(axis=1)).max())
assert abs(lrv.loc[UNPEN, "cal_slope_valid"] - cal_rec.cal_slope) < 1e-3
assert lrv.loc[UNPEN, "cal_slope_oof_train"] < lrv.loc[L2, "cal_slope_oof_train"]    # D4.15 argument holds on OOF too
assert lsens.selected_1se.iloc[0].split(", ") == sel
common_top5 = [f for f in top5_lr if f in top5]
nest_i = nest.set_index(["params", "model"])
nest_max = nest.optimism.abs().max()
_nm = nest[nest.model != "RF (R defaults)"]
nest_main_lo, nest_main_hi = _nm.optimism.min(), _nm.optimism.max()
nest_rfd = nest[nest.model == "RF (R defaults)"].optimism.iloc[0]
_nt = nest[nest.params == "tuned"]
nest_tuned_lo, nest_tuned_hi = _nt.cv_auc_nested.min(), _nt.cv_auc_nested.max()
n_rec = nest_i.loc[("paper", "LR")]; n_rf = nest_i.loc[("tuned", "RF")]
sL, sR = sens.loc["LR (unpenalised)"], sens.loc["RF (tuned)"]
sens_sel = sL.selected_1se.split(", ")
min_nsel, max_nsel = int(nfold.n_selected.min()), int(nfold.n_selected.max())
# robustness rule for the pre-2020 sensitivity analysis (stated in the text)
_top_ok = all(f in sL.shap_top5.split(", ")[:3] for f in ("rdw_mean", "cci"))
_robust = (abs(sL.auc_valid - rec_auc.auc) <= 0.05 and abs(sR.auc_valid - best_auc.auc) <= 0.05
           and sL.delong_p >= 0.05 and _top_ok)
def dca_range(d):
    lo, hi = useful_range(d)
    return f"{lo:.2f}–{hi:.2f}"
def useful_range_m(m):
    return dca_range(dca_m[dca_m.model == m])
def pfmt(v):
    return "<0.001" if v < 0.001 else f"{v:.3f}"
sel_min = lasso.loc["noleak", "selected_min"].split(", ")
overlap = [f for f in sel if f in LASSO_SELECTED]
overlap_min = [f for f in LASSO_SELECTED if f in sel_min]

# ---------------------------------------------------------------- tables
_flow_paper = [n for _, n in COHORT_FLOW]
_flow_ours = [int(flow.loc[flow.step.str.startswith("[1] "), "n"].iloc[0])] + C.n.astype(int).tolist()
_retained = lambda xs: [""] + [f"{b / a:.1%}" for a, b in zip(xs[:-1], xs[1:])]
flow_tab_en = pd.DataFrame({
    "Step": ["[1] DN patients", "[2] First ICU at first hospitalisation", "[3] Hospital stay > 48 h", "[4] Age > 18",
             "[5] Complete blood count"],
    "Paper": _flow_paper, "This study (definition C)": _flow_ours,
    "Paper retained": _retained(_flow_paper), "This study retained": _retained(_flow_ours)})
defs_tab_en = flow[flow.definition.isin(list("ABCD")) & flow.step.str.startswith("[5]")][["definition", "n", "deaths_1y_icu", "mortality_1y_icu"]].copy()
defs_tab_en.columns = ["Definition", "Final n", "1-year deaths", "1-year mortality"]
defs_tab_en.insert(1, "Description", ["First admission overall has a DN code and an ICU stay", "First DN admission (must have an ICU stay)",
                                      "First admission with both a DN code and an ICU stay (main analysis)",
                                      "Admission of the first-ever ICU stay has a DN code (old code)"])

outc_tab_en = pd.DataFrame({
    "Outcome definition": ["1-year all-cause death from ICU admission (main analysis)", "1-year all-cause death from hospital admission",
                           "In-hospital death", "In-hospital death or non-missing dod (any time)", "Paper"],
    "Deaths": [int(o.loc["death_1y_icu", "deaths"]), int(o.loc["death_1y_adm", "deaths"]), int(o.loc["death_inhosp", "deaths"]),
               int(o.loc["death_any_dod", "deaths"]), N_DEATH],
    "Mortality": [f"{o.loc[k, 'rate']:.1%}" for k in ["death_1y_icu", "death_1y_adm", "death_inhosp", "death_any_dod"]]
                 + [f"{N_DEATH / (N_DEATH + N_SURVIVAL):.1%}"],
})

feat_tab_en = feat[["feature", "missing_pct", "ours", "paper"]].rename(columns={"feature": "Variable", "missing_pct": "Missing %",
                                                                                "ours": "This study", "paper": "Paper"})

_nest = nest.set_index(["params", "model"])
auc_rows = [{"Group": "paper (as reported)", "Model": m, "Training": f"{AUC_TRAIN[m][0]:.3f}",
             "Validation (DeLong 95% CI)": f"{AUC_VALID[m][0]:.3f} ({AUC_VALID[m][1]:.3f}–{AUC_VALID[m][2]:.3f})",
             "10-fold CV": "—", "Nested CV": "—"} for m in ["RF", "XGBoost", "SVM", "LR"]]
_pf = grp(perf[perf.version == "noleak"])
for (g, m), q in _pf.groupby(["params", "model"], sort=False):
    q = q.set_index("split")
    nk = ("paper", "LR") if g == "recommended" else (g, "LR" if m == "LR (L2)" else m)
    auc_rows.append({"Group": g, "Model": m, "Training": f"{q.loc['train', 'auc']:.3f}",
                     "Validation (DeLong 95% CI)": f"{q.loc['valid', 'auc']:.3f} ({q.loc['valid', 'auc_lo']:.3f}–{q.loc['valid', 'auc_hi']:.3f})",
                     "10-fold CV": f"{q.loc['train', 'cv_auc_train']:.3f}",
                     "Nested CV": f"{_nest.loc[nk, 'cv_auc_nested']:.3f}" if nk in _nest.index else "—"})
auc_tab_en = pd.DataFrame(auc_rows)

def rank_str(d: dict) -> str:
    return " > ".join(f"{k} ({v:.3f})" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

paper_rank = rank_str({k: v[0] for k, v in AUC_VALID.items()})
q = perf[(perf.split == "valid") & (perf.version == "noleak")]
_q = grp(q)
ours_rank_paper = rank_str(_q[_q.params.isin(["paper", "recommended"])].set_index("model").auc.to_dict())
ours_rank_tuned = rank_str(_q[_q.params == "tuned"].set_index("model").auc.to_dict())

MET = ["accuracy", "precision", "recall", "f1", "specificity"]
thr05_tab = grp(thrm[thrm.rule == "0.5"][["params", "model", *MET, "flagged_pct"]])
thry_tab = grp(thrm[thrm.rule == "youden_oof"][["params", "model", "threshold", *MET, "flagged_pct"]])
ydn_tab = grp(ydn[["params", "model", "threshold_oof", "youden_j_train_oof", "threshold_apparent", "youden_j_train_apparent"]])
dl = dl.assign(model_1=[("LR (L2)" if (p_ == "tuned" and m == "LR") else m) for p_, m in zip(dl.params, dl.model_1)],
               model_2=[("LR (L2)" if (p_ == "tuned" and m == "LR") else m) for p_, m in zip(dl.params, dl.model_2)])
dl = dl.sort_values("params", key=lambda c: c.map(GROUP_ORDER), kind="stable")
dl_tab_en = dl.assign(**{"Comparison": dl.model_1 + " vs " + dl.model_2,
                         "ΔAUC (95% CI)": [f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})" for d, lo, hi in zip(dl["diff"], dl.diff_lo, dl.diff_hi)],
                         "p": dl.p_value.map(lambda v: f"{v:.3f}"), "p (Holm)": dl.p_holm.map(lambda v: f"{v:.3f}")})[
    ["params", "Comparison", "auc_1", "auc_2", "ΔAUC (95% CI)", "z", "p", "p (Holm)"]]
lr_tab_en = pd.DataFrame({
    "Variable": lrc.feature, "1 SD (original units)": lrc.sd_original_units.map(lambda v: f"{v:.3g}"),
    "Unpenalised LR β (per SD)": lrc.coef_glm_per_sd.map(lambda v: f"{v:+.3f}"),
    "Unpenalised LR OR (95% CI)": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(lrc.or_glm_per_sd, lrc.or_glm_lo, lrc.or_glm_hi)],
    "p": lrc.p_glm.map(pfmt),
    "Comparison: L2 LR OR": lrc.or_tuned_per_sd.map(lambda v: f"{v:.2f}"), "VIF": lrc.vif.map(lambda v: f"{v:.2f}")})
lrv_tab_en = pd.DataFrame({
    "LR variant": ["Unpenalised (recommended)", "L2, C = 0.01 (comparison)"],
    "Training OOF AUC": [f"{lrv.loc[k, 'auc_oof_train']:.3f}" for k in (UNPEN, L2)],
    "OOF Brier": [f"{lrv.loc[k, 'brier_oof_train']:.3f}" for k in (UNPEN, L2)],
    "OOF calibration slope": [f"{lrv.loc[k, 'cal_slope_oof_train']:.2f}" for k in (UNPEN, L2)],
    "Validation AUC": [f"{lrv.loc[k, 'auc_valid']:.3f}" for k in (UNPEN, L2)],
    "Validation Brier": [f"{lrv.loc[k, 'brier_valid']:.3f}" for k in (UNPEN, L2)],
    "Validation calibration intercept": [f"{lrv.loc[k, 'cal_in_the_large_valid']:+.3f}" for k in (UNPEN, L2)],
    "Validation calibration slope": [f"{lrv.loc[k, 'cal_slope_valid']:.2f}" for k in (UNPEN, L2)]})
nest = grp(nest)   # recommended LR as its own group (values unchanged)
nest_tab_en = pd.DataFrame({
    "params": grp(nest).params.values, "Model": grp(nest).model.values,
    "Original CV AUC (LASSO outside CV)": nest.cv_auc_nonnested.map(lambda v: f"{v:.3f}"),
    "Nested CV AUC (LASSO within each fold)": nest.cv_auc_nested.map(lambda v: f"{v:.3f}"),
    "Optimism (original − nested)": nest.optimism.map(lambda v: f"{v:+.3f}"),
    "Per-fold optimism range": [f"{a:+.3f} to {b:+.3f}" for a, b in zip(nest.optimism_fold_min, nest.optimism_fold_max)]})
main_sel = sel
sens_tab_en = pd.DataFrame({
    "Item": ["Cohort size (deaths, 1-year mortality)", "Training / validation (validation deaths)", "LASSO λ.1se (number of variables)",
             "Variables selected at λ.1se", "Recommended model (unpenalised LR) validation AUC (DeLong 95% CI)",
             "Tuned RF validation AUC (DeLong 95% CI)", "ΔAUC LR − RF (95% CI), p", "LR calibration slope / Brier",
             "RF calibration slope / Brier", "LR SHAP top 5", "RF SHAP top 5"],
    "Main analysis (all years)": [f"{n_final:,} ({d_final}, {d_final / n_final:.1%})", "1,568 / 392 (99)",
                                  f"{lasso.loc['noleak', 'lambda_1se']:.4f} ({len(sel)})", ", ".join(sel),
                                  f"{rec_auc.auc:.3f} ({rec_auc.auc_lo:.3f}–{rec_auc.auc_hi:.3f})",
                                  f"{best_auc.auc:.3f} ({best_auc.auc_lo:.3f}–{best_auc.auc_hi:.3f})",
                                  f"{rf_lr['diff']:+.3f} ({rf_lr.diff_lo:+.3f} to {rf_lr.diff_hi:+.3f}), p = {rf_lr.p_value:.2f}",
                                  f"{cal_rec.cal_slope:.2f} / {cal_rec.brier:.3f}", f"{calt.loc['RF', 'cal_slope']:.2f} / {calt.loc['RF', 'brier']:.3f}",
                                  ", ".join(top5_lr), ", ".join(top5)],
    "Sensitivity (excluding 2020–2022 group)": [f"{int(sL.n):,} ({int(sL.deaths)}, {sL.mortality:.1%})",
                                                f"{int(sL.n_train):,} / {int(sL.n_valid)} ({int(sL.deaths_valid)})",
                                                f"{sL.lambda_1se:.4f} ({int(sL.n_selected)})", sL.selected_1se,
                                                f"{sL.auc_valid:.3f} ({sL.auc_lo:.3f}–{sL.auc_hi:.3f})", f"{sR.auc_valid:.3f} ({sR.auc_lo:.3f}–{sR.auc_hi:.3f})",
                                                f"{sL.delong_diff_lr_minus_rf:+.3f} ({sL.delong_diff_lo:+.3f} to {sL.delong_diff_hi:+.3f}), p = {sL.delong_p:.2f}",
                                                f"{sL.cal_slope:.2f} / {sL.brier:.3f}", f"{sR.cal_slope:.2f} / {sR.brier:.3f}", sL.shap_top5, sR.shap_top5]})
_cic = pd.read_csv(T / "cic_primary_noleak_recommended.csv")
_c = _cic.iloc[(_cic.threshold - lr_thr.loc["youden_oof", "threshold"]).abs().argmin()]
cic_n, cic_e = _c.n_high_risk, _c.n_high_risk_with_event
sens_only = [f for f in sens_sel if f not in sel]; main_only = [f for f in sel if f not in sens_sel]
h_full, h_unc, h_cmp = htn.iloc[0], htn.iloc[1], htn.iloc[2]
htn_tab_en = pd.DataFrame({
    "Definition": ["Full: 401–405 / I10–I13, I15, I16 (main analysis)", "Uncomplicated only: 401 / I10 (sensitivity, ≈ paper)",
                   "Complicated codes only, no 401/I10 (coded 'no' under the uncomplicated definition)", "No hypertension code"],
    "n": htn.n, "Cohort %": htn.pct_all.map(lambda v: f"{v:.1f}"),
    "Survivors %": htn.pct_survivors.map(lambda v: f"{v:.1f}"), "Deaths %": htn.pct_deaths.map(lambda v: f"{v:.1f}"),
    "1-year mortality in group %": htn.mortality_pct.map(lambda v: f"{v:.1f}"),
    "Unadjusted OR (95% CI)": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(htn.or_death, htn.or_lo, htn.or_hi)],
    "p": htn.p_chi2.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")})
_yr = ydn[ydn.model != "RF (R defaults)"].threshold_oof
cal_tab = cal[(cal.split == "valid") & cal.config.isin(["noleak_paper", "noleak_tuned"])][
    ["config", "model", "brier", "scaled_brier", "cal_in_the_large", "cal_slope", "mean_pred", "observed"]]
cal_tab = grp(cal_tab.assign(config=cal_tab.config.str.replace("noleak_", "")), col="config").rename(columns={"config": "params"})
ovf_tab = grp(ovf[ovf.version == "noleak"][["params", "model", "train", "cv_auc_train", "valid", "gap_train_minus_valid", "flag_overfit"]])

shap_tab_en = pd.DataFrame({"Rank": range(1, 6), "Paper (RF)": SHAP_TOP5, "This study: recommended (unpenalised LR)": top5_lr,
                            "This study: comparison (L2 LR)": top5_lr_l2, "This study: secondary (tuned RF)": top5,
                            "This study: paper configuration (RF, literal parameters)": top5_paperconfig})

sig = t1.loc[t1.p_value < 0.05, "variable"].tolist()

sel_freq_str_en = ", ".join(f"{r.feature} {r.n_folds_selected}/10" for r in nfreq[nfreq.in_stage4_selection].itertuples())
other_freq_str_en = ", ".join(f"{r.feature} {r.n_folds_selected}/10" for r in nfreq[~nfreq.in_stage4_selection].itertuples()) or "none"
_crit_en = ("By criteria written down before the sensitivity analysis was run (set during this revision, not pre-specified at the start "
            "of the project: validation AUC of both models changes by ≤ 0.05, LR and RF still not significantly different, RDW and "
            "CCI still in the recommended LR's SHAP top 3), ")
sens_verdict_en = _crit_en + ("**the main conclusions are robust**; excluding the 2020–2022 group does not change them." if _robust
                              else "**the conclusions are not fully robust**; see the item-by-item notes below.")

# ---------------------------------------------------------------- stage 8 (D8.x): cohort C full nested CV, CLIN
RP, CLD = R / "replication", R / "clinical"
UL2 = ["LR (unpenalised)", "LR (L2)"]
SEEDS = [42, 43, 44, 45, 46]
f3 = lambda v: f"{v:.3f}"


def d88(summ: pd.DataFrame, pw: pd.DataFrame):
    """D8.8 selection rule recomputed from the nested-CV summary and pairwise tables: (table, M*, selected model)."""
    a = summ.auroc_fold_mean
    mstar = a.idxmax()
    rows = []
    for m in summ.index:
        if m == mstar:
            rows.append({"model": m, "delta": 0.0, "lo": np.nan, "hi": np.nan, "equivalent": True})
            continue
        r = pw[((pw.model_1 == mstar) & (pw.model_2 == m)) | ((pw.model_1 == m) & (pw.model_2 == mstar))].iloc[0]
        lo, hi = (r.lo, r.hi) if r.model_1 == mstar else (-r.hi, -r.lo)
        d = a[mstar] - a[m]
        assert abs(d - (r.mean_diff if r.model_1 == mstar else -r.mean_diff)) < 1e-5
        rows.append({"model": m, "delta": d, "lo": lo, "hi": hi, "equivalent": bool(d <= 0.01 or (lo < 0 < hi and d <= 0.02))})
    t = pd.DataFrame(rows).set_index("model")
    lrs = [m for m in UL2 if t.loc[m, "equivalent"]]
    if not lrs:
        chosen = mstar
    elif len(lrs) == 1:
        chosen = lrs[0]
    else:
        q = summ.loc[UL2, "abs_ln_cal_slope_pooled_mean_over_repeats"]
        chosen = "LR (L2)" if q["LR (L2)"] < q["LR (unpenalised)"] else "LR (unpenalised)"
    return t, mstar, chosen


def ncv_tab(summ, t88):
    s = summ.loc[summ.auroc_fold_mean.sort_values(ascending=False).index]
    yes, no = "yes", "no"
    d = pd.DataFrame({
        "m": s.index,
        "a": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(s.auroc_fold_mean, s.auroc_fold_sd)],
        "p": s.auroc_pooled_mean.map(f3), "ap": s.auprc_pooled_mean.map(f3),
        "b": [f"{b:.3f} / {sb:.3f}" for b, sb in zip(s.brier_pooled_mean, s.scaled_brier_pooled_mean)],
        "cs": s.cal_slope_pooled_mean.map(lambda v: f"{v:.3f}"),
        "d": ["— (M\\*)" if np.isnan(t88.loc[m, "lo"]) else f"{t88.loc[m, 'delta']:.4f} ({t88.loc[m, 'lo']:+.4f}, {t88.loc[m, 'hi']:+.4f})"
              for m in s.index],
        "e": [yes if t88.loc[m, "equivalent"] else no for m in s.index]})
    d.columns = ["Model", "AUROC (mean ± SD, 50 folds)", "Pooled OOF AUROC", "AUPRC (pooled OOF)", "Brier / scaled Brier (pooled OOF)",
                 "OOF calibration slope", "ΔAUROC vs M\\* (NB-corrected 95% CI)", "Equivalent to M\\*"]
    return d


def rep_slope_tab(summ):
    rows = []
    for m in UL2:
        r = {"m": m}
        for s_ in SEEDS:
            r[str(s_)] = f"{summ.loc[m, f'cal_slope_pooled_rep{s_}']:.3f} ({summ.loc[m, f'abs_ln_cal_slope_pooled_rep{s_}']:.3f})"
        r["mean"] = f"{summ.loc[m, 'abs_ln_cal_slope_pooled_mean_over_repeats']:.4f}"
        rows.append(r)
    d = pd.DataFrame(rows)
    d.columns = ["Model \\ seed"] + [str(s_) for s_ in SEEDS] + ["mean \\|ln slope\\|"]
    return d


def dca_better(d):
    b = d[(d.net_benefit_model > d.net_benefit_all) & (d.net_benefit_model > 0)].threshold
    return len(b), len(d), (b.min() if len(b) else np.nan), (b.max() if len(b) else np.nan)


def ho_tab(ho, dca, thr_col):
    ms = list(ho.index)
    def cell(m, k):
        r = ho.loc[m]
        if k == "auroc": return f"{r.auroc:.3f} ({r.auroc_lo:.3f}–{r.auroc_hi:.3f})"
        if k == "auprc": return f"{r.auprc:.3f} ({r.auprc_lo:.3f}–{r.auprc_hi:.3f})"
        if k == "brier": return f"{r.brier:.3f} / {r.scaled_brier:.3f}"
        if k == "cal": return f"{r.cal_intercept:+.3f} / {r.cal_slope:.2f}"
        if k == "thr": return f"{r[thr_col]:.4f}"
        if k == "cls": return " / ".join(f"{r[c]:.3f}" for c in ["accuracy", "precision", "recall", "f1", "specificity"])
        if k == "flag": return f"{r.flagged_pct:.1f}%"
        if k == "dca":
            n, N, lo, hi = dca_better(dca[dca.model == m])
            return f"{n}/{N} thresholds ({lo:.2f}–{hi:.2f})"
    keys = ["auroc", "auprc", "brier", "cal", "thr", "cls", "flag", "dca"]
    prev = ho[ho.columns[ho.columns.str.startswith("prevalence")][0]].iloc[0]
    lab_en = ["AUROC (DeLong 95% CI)", f"AUPRC (bootstrap 95% CI; prevalence baseline {prev:.3f})", "Brier / scaled Brier",
              "Calibration intercept / slope", "Threshold (nested-CV OOF Youden, median of 5 repeats)",
              "accuracy / precision / recall / F1 / specificity", "Share flagged high risk",
              "DCA (0.05–0.60): net benefit above both treat-all and 0"]
    d = pd.DataFrame({"k": lab_en, **{m: [cell(m, k) for k in keys] for m in ms}})
    d = d.rename(columns={"k": "Metric"})
    return d


def kv(path_):
    return pd.read_csv(path_).set_index("item").value


def nsel_stats(folds):
    g = folds.groupby(["seed", "fold"]).n_selected.first()
    return int(g.median()), int(g.min()), int(g.max())


# ---- cohort C (replication track)
sC = pd.read_csv(RP / "nested_cv_full_summary_C.csv").set_index("model")
pwC = pd.read_csv(RP / "nested_cv_full_pairwise_C.csv")
selC = pd.read_csv(RP / "nested_cv_full_selection_C.csv").set_index("feature").n_folds_selected
thrC = pd.read_csv(RP / "nested_cv_full_oof_threshold_C.csv").set_index("model")
dcaC = pd.read_csv(RP / "nested_cv_full_dca_C.csv")
foldC = pd.read_csv(RP / "nested_cv_full_folds_C.csv")
hoC = pd.read_csv(RP / "holdout_C.csv").set_index("model")
hodC = pd.read_csv(RP / "holdout_dca_C.csv")
ctS = kv(RP / "comorb_time_summary_C.csv")
ctF = pd.read_csv(RP / "comorb_time_flags_C.csv").set_index("component")
ctV = pd.read_csv(RP / "nested_cv_full_vs_main_C_comorb_time.csv").set_index("model")
ctSum = pd.read_csv(RP / "nested_cv_full_summary_C_comorb_time.csv").set_index("model")
ctSel = pd.read_csv(RP / "nested_cv_full_selection_C_comorb_time.csv").set_index("feature").n_folds_selected
t88C, mstarC, chosenC = d88(sC, pwC)
assert chosenC == "LR (unpenalised)", "D8.8 no longer selects the unpenalised LR for cohort C: report text must change"
eqC = [m for m in t88C.index if t88C.loc[m, "equivalent"]]
neqC = [m for m in t88C.index if not t88C.loc[m, "equivalent"]]
eqC_lo, eqC_hi = sC.loc[eqC].auroc_fold_mean.min(), sC.loc[eqC].auroc_fold_mean.max()
neqC_lo, neqC_hi = sC.loc[neqC].auroc_fold_mean.min(), sC.loc[neqC].auroc_fold_mean.max()
assert set(neqC) == {"RF (paper)", "RF (R defaults) (paper)", "SVM (paper)"}
nselC = nsel_stats(foldC)
_rC = foldC.groupby(["model", "seed"]).auroc.mean()
u42C, r42C = _rC.loc[("LR (unpenalised)", 42)], _rC.loc[("RF (tuned)", 42)]
u_rep_lo, u_rep_hi = _rC.loc["LR (unpenalised)"].min(), _rC.loc["LR (unpenalised)"].max()
sel50C = selC[selC == 50].index.tolist()
sel40C = selC[(selC >= 40) & (selC < 50)].index.tolist()
uC, rC = sC.loc["LR (unpenalised)"], sC.loc["RF (tuned)"]
l2C = sC.loc["LR (L2)"]
n_l2_closerC = sum(summ_v < 0 for summ_v in [sC.loc["LR (L2)", f"abs_ln_cal_slope_pooled_rep{s_}"] - sC.loc["LR (unpenalised)", f"abs_ln_cal_slope_pooled_rep{s_}"] for s_ in SEEDS])
hoCu, hoCr = hoC.loc["LR (unpenalised)"], hoC.loc["RF (tuned)"]
ho_gapC = max(abs(hoCu.auroc - sC.loc["LR (unpenalised)", "auroc_fold_mean"]), abs(hoCr.auroc - sC.loc["RF (tuned)", "auroc_fold_mean"]))
assert abs(hoCu.auroc - rec_auc.auc) < 1e-4 and abs(hoCr.auroc - best_auc.auc) < 1e-4   # 04g reproduces stage 4 (4-dp table)
dcaCu = dca_better(dcaC[dcaC.model == "LR (unpenalised)"]); dcaCr = dca_better(dcaC[dcaC.model == "RF (tuned)"])
mi = ctF.loc["myocardial_infarct"]
assert ctS["CCI original, median (IQR)"] == ctS["CCI sensitivity, median (IQR)"]   # text: "unchanged median"
ctu, ctr = ctV.loc["LR (unpenalised)"], ctV.loc["RF (tuned)"]
CT_COMP = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease", "cerebrovascular_disease",
           "chronic_pulmonary_disease", "renal_disease", "malignant_cancer", "mild_liver_disease", "peptic_ulcer_disease"]
ct_tab_en = ctF.loc[CT_COMP, ["original_n", "sensitivity_n", "lost_by_acute_exclusion", "new_from_prior_admissions"]].reset_index()
ct_tab_en.columns = ["Comorbidity (charlson component)", "Positive (index admission)", "Positive (recomputed)",
                     "Lost by removing acute codes", "Added from prior admissions"]
def ct_perf():
    rows = []
    for lab_en, f in [("AUROC (fold mean ± SD)", lambda s: f"{s.auroc_fold_mean:.3f} ± {s.auroc_fold_sd:.3f}"),
                      ("Pooled OOF AUROC", lambda s: f3(s.auroc_pooled_mean)),
                      ("AUPRC (pooled OOF)", lambda s: f3(s.auprc_pooled_mean)),
                      ("Brier (pooled OOF)", lambda s: f3(s.brier_pooled_mean)),
                      ("Calibration slope (pooled OOF)", lambda s: f3(s.cal_slope_pooled_mean))]:
        rows.append([lab_en] + [f(S.loc[m]) for m in ["LR (unpenalised)", "RF (tuned)"] for S in (sC, ctSum)])
    rows.append(["ΔAUROC (sensitivity − main, 95% CI)",
                 "—", f"{ctu.mean_diff:+.3f} ({ctu.lo:+.3f}, {ctu.hi:+.3f})", "—", f"{ctr.mean_diff:+.3f} ({ctr.lo:+.3f}, {ctr.hi:+.3f})"])
    rows.append(["Folds selecting CCI / SAPS-II",
                 f"{selC['cci']}/50, {selC['sapsii']}/50", f"{ctSel['cci']}/50, {ctSel['sapsii']}/50",
                 f"{selC['cci']}/50, {selC['sapsii']}/50", f"{ctSel['cci']}/50, {ctSel['sapsii']}/50"])
    return pd.DataFrame(rows, columns=["Metric", "Unpenalised LR: main", "Unpenalised LR: comorbidity timing", "Tuned RF: main",
                                       "Tuned RF: comorbidity timing"])
ncvC_tab_en = ncv_tab(sC, t88C)
slC_tab_en = rep_slope_tab(sC)
hoC_tab_en = ho_tab(hoC, hodC, "threshold_nested_oof_youden_median")
ctp_tab_en = ct_perf()

# ---- CLIN (clinical track)
flK = pd.read_csv(CLD / "cohort_flow_clin.csv")
spK = pd.read_csv(CLD / "split_clin_summary.csv").set_index("split")
cmK = kv(CLD / "comorb_clin_summary.csv")
wpK = pd.read_csv(CLD / "wbc_diff_patterns_clin.csv").set_index("pattern")
msK = pd.read_csv(CLD / "missingness_clin.csv")
sK = pd.read_csv(CLD / "nested_cv_full_summary_clin.csv").set_index("model")
pwK = pd.read_csv(CLD / "nested_cv_full_pairwise_clin.csv")
selK = pd.read_csv(CLD / "nested_cv_full_selection_clin.csv").set_index("feature").n_folds_selected
thrK = pd.read_csv(CLD / "nested_cv_full_oof_threshold_clin.csv").set_index("model")
foldK = pd.read_csv(CLD / "nested_cv_full_folds_clin.csv")
specK = json.load(open(CLD / "frozen_model_clin.json"))
recK = json.load(open(CLD / "final_model_clin.json"))
hoK = pd.read_csv(CLD / "holdout_clin.csv").set_index("model")
hodK = pd.read_csv(CLD / "holdout_dca_clin.csv")
coK = pd.read_csv(CLD / "coefs_clin.csv")
BIN_K = set(BINARY) | {"wbc_diff_missing", "has_prior_admission"}
_isb = coK.feature.isin(BIN_K)
for _c, _src in [("or_bin", "or_glm_per_sd"), ("or_bin_lo", "or_glm_lo"), ("or_bin_hi", "or_glm_hi")]:   # OR yes vs no
    coK[_c] = np.where(_isb, np.exp(np.log(coK[_src]) / coK.sd_original_units), np.nan)
shK = pd.read_csv(CLD / "shap_clin.csv").sort_values("rank")
gdK = kv(CLD / "holdout_gap_diagnostics_clin.csv")
t1K = pd.read_csv(CLD / "table1_clin.csv")
wmK = pd.read_csv(CLD / "wbc_diff_missing_mortality_clin.csv").set_index("wbc_diff_missing")
obK = pd.read_csv(CLD / "obesity_source_clin.csv").set_index("group")
sumP = pd.read_csv(CLD / "nested_cv_full_summary_clin_prior_only.csv").set_index("model")
sumW = pd.read_csv(CLD / "nested_cv_full_summary_clin_no_wbc_diff.csv").set_index("model")
vmP = pd.read_csv(CLD / "nested_cv_full_vs_main_clin_prior_only.csv").set_index("model")
vmW = pd.read_csv(CLD / "nested_cv_full_vs_main_clin_no_wbc_diff.csv").set_index("model")
selP = pd.read_csv(CLD / "nested_cv_full_selection_clin_prior_only.csv").set_index("feature").n_folds_selected
selW = pd.read_csv(CLD / "nested_cv_full_selection_clin_no_wbc_diff.csv").set_index("feature").n_folds_selected
nselK = nsel_stats(foldK)
nselP = nsel_stats(pd.read_csv(CLD / "nested_cv_full_folds_clin_prior_only.csv"))
_fW = pd.read_csv(CLD / "nested_cv_full_folds_clin_no_wbc_diff.csv")
nselW = nsel_stats(_fW)
n_drop_W = int(_fW.groupby(["seed", "fold"]).n_dropped_missing.first().min())
assert n_drop_W == int(_fW.n_dropped_missing.max()) == 8
t88K, mstarK, chosenK = d88(sK, pwK)
assert mstarK == "LR (L2)" and chosenK == "LR (L2)", "D8.8 selection for CLIN changed: report text must change"
eqK = [m for m in t88K.index if t88K.loc[m, "equivalent"]]
neqK = [m for m in t88K.index if not t88K.loc[m, "equivalent"]]
eqK_maxd = t88K.loc[[m for m in eqK if m != mstarK], "delta"].max()
uK, l2K, rK = sK.loc["LR (unpenalised)"], sK.loc["LR (L2)"], sK.loc["RF (tuned)"]
n_unpen_closerK = sum(sK.loc["LR (unpenalised)", f"abs_ln_cal_slope_pooled_rep{s_}"] < sK.loc["LR (L2)", f"abs_ln_cal_slope_pooled_rep{s_}"] for s_ in SEEDS)
top15K = selK.sort_values(ascending=False, kind="stable").head(15)
sel50K = selK[selK == 50].index.tolist()
hoKl, hoKr = hoK.loc["LR (L2)"], hoK.loc["RF (tuned)"]


def calci_text() -> str:
    """Post-hoc calibration CIs on the CLIN held-out set (04l; descriptive), empty if 04l has not been run."""
    f_ = CLD / "holdout_calibration_ci_clin.csv"
    if not f_.exists():
        return ""
    c = pd.read_csv(f_).set_index("model")
    for m in ("LR (L2)", "RF (tuned)"):                # same point estimates as holdout_clin.csv
        assert abs(c.loc[m, "cal_intercept"] - hoK.loc[m, "cal_intercept"]) < 1e-6
        assert abs(c.loc[m, "cal_slope"] - hoK.loc[m, "cal_slope"]) < 1e-6
    l_ = c.loc["LR (L2)"]
    iv = lambda r, a, b: f"({r[a]:+.3f}, {r[b]:+.3f})"
    sv = lambda r, a, b: f"{r[a]:.2f}–{r[b]:.2f}"
    nb = int(c.n_boot_ok.min())
    head = ("- **Post-hoc addition: 95% CIs for the calibration intercept and slope** (`clinical/holdout_calibration_ci_clin.csv`, "
            "`04l_calibration_ci_clin.py`; post-hoc descriptive analysis; the frozen model and the confirmatory results were not changed): "
            f"LR (L2) slope {l_.cal_slope:.2f} (Wald 95% CI {sv(l_, 'slope_wald_lo', 'slope_wald_hi')}; Wald test of slope = 1, "
            f"p = {l_.p_slope_eq_1:.2f}), intercept {l_.cal_intercept:+.3f} (Wald 95% CI {iv(l_, 'int_wald_lo', 'int_wald_hi')}; "
            f"Wald test of intercept = 0, p = {l_.p_int_eq_0:.2f}). With only {int(hoKl.events_eval)} deaths in the held-out set "
            "the intervals are wide: neither deviation is statistically distinguishable from perfect calibration, and the size "
            f"of the miscalibration is not precisely estimated. The stratified bootstrap (2,000 resamples, seed 42, predictions "
            f"only, {nb:,} converged) fixes the number of events and so omits outcome-rate sampling variability; its intercept "
            "interval is therefore narrower than the Wald interval, which is the one to use for the intercept.")
    cols = "| Model | Intercept (Wald 95% CI) | Intercept (bootstrap 95% CI) | Slope (Wald 95% CI) | Slope (bootstrap 95% CI) | p (slope = 1) | p (intercept = 0) |"
    rows_ = [f"| {m} | {r.cal_intercept:+.3f} {iv(r, 'int_wald_lo', 'int_wald_hi')} | {r.cal_intercept:+.3f} {iv(r, 'int_boot_lo', 'int_boot_hi')} "
             f"| {r.cal_slope:.2f} ({sv(r, 'slope_wald_lo', 'slope_wald_hi')}) | {r.cal_slope:.2f} ({sv(r, 'slope_boot_lo', 'slope_boot_hi')}) "
             f"| {r.p_slope_eq_1:.2f} | {r.p_int_eq_0:.2f} |" for m, r in c.iterrows()]
    return "\n".join([head, "", cols, "|---|---|---|---|---|---|---|", *rows_, "", ""])


calciK_en = calci_text()
assert abs(specK["thresholds"]["LR (L2)"] - thrK.loc["LR (L2)", "median"]) < 1e-9
assert abs(specK["thresholds"]["RF (tuned)"] - thrK.loc["RF (tuned)", "median"]) < 1e-9
assert recK["params"]["LR (L2)"]["C"] == 0.01
shap10K = shK.feature.tolist()[:10]
assert shap10K[:3] == top5_lr[:3], "CLIN SHAP top 3 no longer equal cohort C's: report text must change"
gd = {k: (float(v) if k not in ("smd_max_abs_feature", "smd_features_abs_gt_0.1") else v) for k, v in gdK.items()}
partial_wbc = int(cmK["CLIN stays"]) - int(wpK.loc["........", "n"]) - int(wpK.loc["MMMMMMMM", "n"])
n_wbc_missing = int(wpK.loc["MMMMMMMM", "n"])
wbc_glm = coK.set_index("feature").loc["wbc_diff_missing"]
ob_glm = coK.set_index("feature").loc["obesity"]
ob_ix, ob_pr, ob_no = (obK.loc["obesity = 1 only from the index admission"], obK.loc["obesity = 1 in prior admissions"],
                       obK.loc["obesity = 0 (CLIN rule)"])
pP, pW = vmP.loc["LR (L2)"], vmW.loc["LR (L2)"]
pPr, pWr = vmP.loc["RF (tuned)"], vmW.loc["RF (tuned)"]
over20K = msK[msK.over_20pct].feature.tolist()
max_other_missK = msK[~msK.over_20pct].sort_values("missing_pct").iloc[-1]
cfK = pd.read_csv(CLD / "comorb_clin_flags.csv").set_index("component")
dcaKl = dca_better(hodK[hodK.model == "LR (L2)"]); dcaKr = dca_better(hodK[hodK.model == "RF (tuned)"])

flK_tab_en = flK.assign(mortality_1y=flK.mortality_1y.map(lambda v: f"{v:.1%}"))
flK_tab_en.columns = ["Step", "Remaining", "Excluded at this step", "1-year deaths", "1-year mortality"]
ncvK_tab_en = ncv_tab(sK, t88K)
slK_tab_en = rep_slope_tab(sK)
hoK_tab_en = ho_tab(hoK, hodK, "threshold_frozen")
hoK_tab_en.loc[len(hoK_tab_en)] = ["Paired DeLong: LR (L2) − RF (tuned)",
                                   f"{hoKl.delong_lr_minus_rf_diff:+.3f} ({hoKl.delong_lr_minus_rf_lo:+.3f}, "
                                   f"{hoKl.delong_lr_minus_rf_hi:+.3f}), p = {hoKl.delong_lr_minus_rf_p:.2f}", "—"]
sel15_str_en = ", ".join(f"{f_} {n}" for f_, n in top15K.items())
_co = coK.copy()
coK_tab_en = pd.DataFrame({
    "Variable": _co.feature, "1 SD (original units)": _co.sd_original_units.map(lambda v: f"{v:.3g}"),
    "L2 LR OR (per SD, penalised)": _co.or_l2_per_sd.map(lambda v: f"{v:.2f}"),
    "Unpenalised LR OR (95% CI)": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(_co.or_glm_per_sd, _co.or_glm_lo, _co.or_glm_hi)],
    "Unpenalised LR OR (yes vs no, 95% CI)": ["—" if np.isnan(o) else f"{o:.2f} ({lo:.2f}–{hi:.2f})"
                                              for o, lo, hi in zip(_co.or_bin, _co.or_bin_lo, _co.or_bin_hi)],
    "p (unpenalised)": _co.p_glm.map(pfmt), "VIF": _co.vif.map(lambda v: f"{v:.2f}")})
sig_glmK = _co[_co.p_glm < 0.05].sort_values("coef_glm_per_sd", key=abs, ascending=False)
sig_glmK_str_en = ", ".join(f"{r.feature} {r.or_bin:.2f} ({r.or_bin_lo:.2f}–{r.or_bin_hi:.2f}, yes vs no)" if r.feature in BIN_K
                            else f"{r.feature} {r.or_glm_per_sd:.2f} ({r.or_glm_lo:.2f}–{r.or_glm_hi:.2f}, per SD)" for r in sig_glmK.itertuples())
vif_top = _co.sort_values("vif", ascending=False).head(3)
vif_str_en = ", ".join(f"{r.feature} {r.vif:.2f}" for r in vif_top.itertuples())

def sens_K():
    def c(S, m, f): return f(S.loc[m])
    M = ["LR (L2)", "RF (tuned)"]
    rows = []
    for le, f in [("AUROC (fold mean ± SD)", lambda s: f"{s.auroc_fold_mean:.3f} ± {s.auroc_fold_sd:.3f}"),
                  ("AUPRC (pooled OOF)", lambda s: f3(s.auprc_pooled_mean)),
                  ("Brier (pooled OOF)", lambda s: f3(s.brier_pooled_mean)),
                  ("Calibration slope (pooled OOF)", lambda s: f3(s.cal_slope_pooled_mean))]:
        rows.append([le] + [c(S, m, f) for m in M for S in (sK, sumP, sumW)])
    rows.insert(1, ["ΔAUROC (sensitivity − main, 95% CI)"] +
                [v for m in M for v in ("—", f"{vmP.loc[m, 'mean_diff']:+.3f} ({vmP.loc[m, 'lo']:+.3f}, {vmP.loc[m, 'hi']:+.4f})",
                                        f"{vmW.loc[m, 'mean_diff']:+.3f} ({vmW.loc[m, 'lo']:+.3f}, {vmW.loc[m, 'hi']:+.3f})")])
    ns = [f"{a} ({b}–{c_})" for a, b, c_ in (nselK, nselP, nselW)]
    rows.append(["Variables selected by LASSO (median, range)"] + ns + ns)
    cols = ["Metric", "LR (L2) main", "LR (L2) prior_only", "LR (L2) no_wbc_diff", "Tuned RF main", "Tuned RF prior_only",
            "Tuned RF no_wbc_diff"]
    return pd.DataFrame(rows, columns=cols)
sensK_tab_en = sens_K()

_t1 = t1K[t1K.variable != "n"]
t1K_tab_en = pd.DataFrame({"Variable": _t1.variable, "Missing %": _t1.missing_pct.map(lambda v: f"{v:.1f}"), "Overall": _t1.total,
                           "Survivors": _t1.survival, "Deaths": _t1.non_survival, "p": _t1.p_display})
n_sigK = int((t1K.p_value < 0.05).sum())
n_surv_K, n_death_K = int(t1K.survival.iloc[0]), int(t1K.non_survival.iloc[0])

# ---- appendix A: stage 4-7 single-validation-set tables (CSV links)
def csv_link(name, sub="tables"):
    return f"[`{sub}/{name}`]({sub}/{name})"

# ---------------------------------------------------------------- extension snapshot (Part IV of report_en.md)
EXT_ORDER = ["static_lr", "static_rf", "static_xgb", "static_stacking", "temporal_lr", "temporal_rf", "temporal_xgb",
             "burden_lr", "burden_rf", "burden_xgb", "embedding_lr", "embedding_rf", "embedding_xgb", "combined_stacking"]


def check_extension_snapshot() -> None:
    """Verify the archived extension snapshot cited in Part IV (file SHA-256 and run identity); aggregate files only."""
    latest = R / "extensions" / "latest.txt"
    if not latest.exists():
        return
    name = latest.read_text(encoding="utf-8").strip()
    if Path(name).name != name:
        raise ValueError("invalid extension snapshot name")
    folder = latest.parent / name
    prov = json.loads((folder / "source_provenance.json").read_text(encoding="utf-8"))
    if prov["source_signature"] != "27464ba98305edbc":
        raise ValueError("Part IV describes only the checked 27464ba98305edbc run; update the text before using a new snapshot")
    for f_, digest in prov["files_sha256"].items():
        if Path(f_).name != f_ or hashlib.sha256((folder / f_).read_bytes()).hexdigest() != digest:
            raise ValueError(f"extension aggregate file changed: {f_}; re-check the snapshot")
    summ = pd.read_csv(folder / "summary.csv").set_index("experiment")
    ident = json.loads((folder / "run_identity.json").read_text(encoding="utf-8"))
    if not set(EXT_ORDER) <= set(summ.index) or not summ.loc[EXT_ORDER, "n_outer_folds"].eq(50).all():
        raise ValueError("extension results do not contain all 14 experiments x 50 folds")
    if ident["holdout_labels_loaded"] or ident["synthetic"]:
        raise ValueError("Part IV accepts only real development-set runs that never loaded held-out labels")
    if ident["options"]["seeds"] != [42, 43, 44, 45, 46] or ident["options"]["outer_folds"] != 10:
        raise ValueError("Part IV assumes the full 5 x 10-fold design; update the text if the evaluation changes")


check_extension_snapshot()

# report: template in scripts/templates/report_en.md (an f-string body)
_tmpl = (Path(__file__).parent / "templates" / "report_en.md").read_text()
assert '"""' not in _tmpl
rep_en = eval('f"""' + _tmpl + '"""')
(R / "report_en.md").write_text(rep_en)
print(f"report written: results/report_en.md ({len(rep_en):,} chars)")
print(f"cohort C: M* {mstarC}, selected {chosenC} (nested CV {uC.auroc_fold_mean:.3f}); holdout {hoCu.auroc:.3f}")
print(f"CLIN: M* {mstarK}, selected {chosenK} (nested CV {l2K.auroc_fold_mean:.3f}); holdout {hoKl.auroc:.3f} "
      f"({hoKl.auroc_lo:.3f}-{hoKl.auroc_hi:.3f})")
