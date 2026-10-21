"""Stage 5 — assemble results/report.md (Chinese) and results/report_en.md (English, template in
scripts/templates/report_en.md) from the aggregate tables (no patient-level data)."""
from __future__ import annotations

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
sel_freq_str = "、".join(f"{r.feature} {r.n_folds_selected}/10" for r in nfreq[nfreq.in_stage4_selection].itertuples())
other_freq_str = "、".join(f"{r.feature} {r.n_folds_selected}/10" for r in nfreq[~nfreq.in_stage4_selection].itertuples()) or "无"
# robustness rule for the pre-2020 sensitivity analysis (stated in the text)
_top_ok = all(f in sL.shap_top5.split(", ")[:3] for f in ("rdw_mean", "cci"))
_robust = (abs(sL.auc_valid - rec_auc.auc) <= 0.05 and abs(sR.auc_valid - best_auc.auc) <= 0.05
           and sL.delong_p >= 0.05 and _top_ok)
sens_verdict = ("按在敏感性分析运行前写明的标准（在本次修订中制定，并非项目开始时预设；两个模型的验证集 AUC 变化均 ≤ 0.05、LR 与 RF 仍无显著差异、RDW 和 CCI 仍在推荐 LR 的 SHAP 前 3），"
                "**主要结论稳健**，排除 2020–2022 年组不改变结论。" if _robust else
                "按在敏感性分析运行前写明的标准（在本次修订中制定，并非项目开始时预设；两个模型的验证集 AUC 变化均 ≤ 0.05、LR 与 RF 仍无显著差异、RDW 和 CCI 仍在推荐 LR 的 SHAP 前 3），"
                "**结论不完全稳健**，见下文逐项说明。")
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
flow_tab = pd.DataFrame({
    "步骤": ["[1] DN 患者", "[2] 首次住院的首次 ICU", "[3] 住院 > 48 h", "[4] 年龄 > 18", "[5] 血常规完整"],
    "论文": [n for _, n in COHORT_FLOW],
    "本研究（口径 C）": [int(flow.loc[flow.step.str.startswith("[1] "), "n"].iloc[0])] + C.n.astype(int).tolist(),
})
flow_tab["论文 保留比例"] = [""] + [f"{b / a:.1%}" for a, b in zip(flow_tab["论文"][:-1], flow_tab["论文"][1:])]
flow_tab["本研究 保留比例"] = [""] + [f"{b / a:.1%}" for a, b in zip(flow_tab["本研究（口径 C）"][:-1], flow_tab["本研究（口径 C）"][1:])]
defs_tab = flow[flow.definition.isin(list("ABCD")) & flow.step.str.startswith("[5]")][["definition", "n", "deaths_1y_icu", "mortality_1y_icu"]]
defs_tab.columns = ["口径", "最终人数", "1 年死亡", "1 年死亡率"]
defs_tab.insert(1, "定义", ["全部住院中的首次住院含 DN 码且有 ICU", "首个 DN 住院（须有 ICU）",
                            "首个同时含 DN 码和 ICU 的住院（主分析）", "一生首次 ICU 所在住院含 DN 码（旧代码）"])

outc_tab = pd.DataFrame({
    "结局定义": ["1 年全因死亡，自 ICU 入科（主分析）", "1 年全因死亡，自入院", "院内死亡", "院内死亡或 dod 非空（任何时间）", "论文"],
    "死亡人数": [int(o.loc["death_1y_icu", "deaths"]), int(o.loc["death_1y_adm", "deaths"]), int(o.loc["death_inhosp", "deaths"]),
                 int(o.loc["death_any_dod", "deaths"]), N_DEATH],
    "死亡率": [f"{o.loc[k, 'rate']:.1%}" for k in ["death_1y_icu", "death_1y_adm", "death_inhosp", "death_any_dod"]]
             + [f"{N_DEATH / (N_DEATH + N_SURVIVAL):.1%}"],
})

feat_tab = feat[["feature", "missing_pct", "ours", "paper"]].rename(columns={"feature": "变量", "missing_pct": "缺失 %", "ours": "本研究", "paper": "论文"})

_nest = nest.set_index(["params", "model"])
auc_rows = [{"组": "论文（原文报告）", "模型": m, "训练": f"{AUC_TRAIN[m][0]:.3f}",
             "验证（DeLong 95% CI）": f"{AUC_VALID[m][0]:.3f} ({AUC_VALID[m][1]:.3f}–{AUC_VALID[m][2]:.3f})",
             "10 折 CV": "—", "嵌套 CV": "—"} for m in ["RF", "XGBoost", "SVM", "LR"]]
_pf = grp(perf[perf.version == "noleak"])
for (g, m), q in _pf.groupby(["params", "model"], sort=False):
    q = q.set_index("split")
    nk = ("paper", "LR") if g == "recommended" else (g, "LR" if m == "LR (L2)" else m)
    auc_rows.append({"组": g, "模型": m, "训练": f"{q.loc['train', 'auc']:.3f}",
                     "验证（DeLong 95% CI）": f"{q.loc['valid', 'auc']:.3f} ({q.loc['valid', 'auc_lo']:.3f}–{q.loc['valid', 'auc_hi']:.3f})",
                     "10 折 CV": f"{q.loc['train', 'cv_auc_train']:.3f}",
                     "嵌套 CV": f"{_nest.loc[nk, 'cv_auc_nested']:.3f}" if nk in _nest.index else "—"})
auc_tab = pd.DataFrame(auc_rows)

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
dl_tab = dl.assign(**{"比较": dl.model_1 + " vs " + dl.model_2,
                      "ΔAUC (95% CI)": [f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})" for d, lo, hi in zip(dl["diff"], dl.diff_lo, dl.diff_hi)],
                      "p": dl.p_value.map(lambda v: f"{v:.3f}"), "p (Holm)": dl.p_holm.map(lambda v: f"{v:.3f}")})[
    ["params", "比较", "auc_1", "auc_2", "ΔAUC (95% CI)", "z", "p", "p (Holm)"]]
lr_tab = pd.DataFrame({
    "变量": lrc.feature, "1 SD（原始单位）": lrc.sd_original_units.map(lambda v: f"{v:.3g}"),
    "无惩罚 LR β（每 SD）": lrc.coef_glm_per_sd.map(lambda v: f"{v:+.3f}"),
    "无惩罚 LR OR (95% CI)": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(lrc.or_glm_per_sd, lrc.or_glm_lo, lrc.or_glm_hi)],
    "p": lrc.p_glm.map(pfmt),
    "对照：L2 LR OR": lrc.or_tuned_per_sd.map(lambda v: f"{v:.2f}"), "VIF": lrc.vif.map(lambda v: f"{v:.2f}")})
lrv_tab = pd.DataFrame({
    "LR 版本": ["无惩罚（推荐）", "L2，C = 0.01（对照）"],
    "训练集 OOF AUC": [f"{lrv.loc[k, 'auc_oof_train']:.3f}" for k in (UNPEN, L2)],
    "OOF Brier": [f"{lrv.loc[k, 'brier_oof_train']:.3f}" for k in (UNPEN, L2)],
    "OOF 校准斜率": [f"{lrv.loc[k, 'cal_slope_oof_train']:.2f}" for k in (UNPEN, L2)],
    "验证集 AUC": [f"{lrv.loc[k, 'auc_valid']:.3f}" for k in (UNPEN, L2)],
    "验证集 Brier": [f"{lrv.loc[k, 'brier_valid']:.3f}" for k in (UNPEN, L2)],
    "验证集校准截距": [f"{lrv.loc[k, 'cal_in_the_large_valid']:+.3f}" for k in (UNPEN, L2)],
    "验证集校准斜率": [f"{lrv.loc[k, 'cal_slope_valid']:.2f}" for k in (UNPEN, L2)]})
nest = grp(nest)   # recommended LR as its own group (values unchanged)
nest_tab = pd.DataFrame({
    "params": grp(nest).params.values, "模型": grp(nest).model.values,
    "原 CV AUC（LASSO 在 CV 外）": nest.cv_auc_nonnested.map(lambda v: f"{v:.3f}"),
    "嵌套 CV AUC（LASSO 在每折内）": nest.cv_auc_nested.map(lambda v: f"{v:.3f}"),
    "乐观偏倚（原 − 嵌套）": nest.optimism.map(lambda v: f"{v:+.3f}"),
    "各折偏倚范围": [f"{a:+.3f} 至 {b:+.3f}" for a, b in zip(nest.optimism_fold_min, nest.optimism_fold_max)]})
main_sel = sel
sens_tab = pd.DataFrame({
    "项目": ["队列人数（死亡，1 年死亡率）", "训练集 / 验证集（验证集死亡）", "LASSO λ.1se（变量数）", "λ.1se 选出的变量",
            "推荐模型（无惩罚 LR）验证集 AUC（DeLong 95% CI）", "调参 RF 验证集 AUC（DeLong 95% CI）",
            "LR − RF 的 ΔAUC（95% CI），p", "LR 校准斜率 / Brier", "RF 校准斜率 / Brier",
            "LR SHAP 前 5", "RF SHAP 前 5"],
    "主分析（全部年份）": [f"{n_final:,}（{d_final}，{d_final / n_final:.1%}）", "1,568 / 392（99）",
                     f"{lasso.loc['noleak', 'lambda_1se']:.4f}（{len(sel)}）", ", ".join(sel),
                     f"{rec_auc.auc:.3f}（{rec_auc.auc_lo:.3f}–{rec_auc.auc_hi:.3f}）",
                     f"{best_auc.auc:.3f}（{best_auc.auc_lo:.3f}–{best_auc.auc_hi:.3f}）",
                     f"{rf_lr['diff']:+.3f}（{rf_lr.diff_lo:+.3f} 至 {rf_lr.diff_hi:+.3f}），p = {rf_lr.p_value:.2f}",
                     f"{cal_rec.cal_slope:.2f} / {cal_rec.brier:.3f}", f"{calt.loc['RF', 'cal_slope']:.2f} / {calt.loc['RF', 'brier']:.3f}",
                     ", ".join(top5_lr), ", ".join(top5)],
    "敏感性分析（排除 2020–2022 组）": [f"{int(sL.n):,}（{int(sL.deaths)}，{sL.mortality:.1%}）",
                     f"{int(sL.n_train):,} / {int(sL.n_valid)}（{int(sL.deaths_valid)}）",
                     f"{sL.lambda_1se:.4f}（{int(sL.n_selected)}）", sL.selected_1se,
                     f"{sL.auc_valid:.3f}（{sL.auc_lo:.3f}–{sL.auc_hi:.3f}）", f"{sR.auc_valid:.3f}（{sR.auc_lo:.3f}–{sR.auc_hi:.3f}）",
                     f"{sL.delong_diff_lr_minus_rf:+.3f}（{sL.delong_diff_lo:+.3f} 至 {sL.delong_diff_hi:+.3f}），p = {sL.delong_p:.2f}",
                     f"{sL.cal_slope:.2f} / {sL.brier:.3f}", f"{sR.cal_slope:.2f} / {sR.brier:.3f}", sL.shap_top5, sR.shap_top5]})
_cic = pd.read_csv(T / "cic_primary_noleak_recommended.csv")
_c = _cic.iloc[(_cic.threshold - lr_thr.loc["youden_oof", "threshold"]).abs().argmin()]
cic_n, cic_e = _c.n_high_risk, _c.n_high_risk_with_event
sens_only = [f for f in sens_sel if f not in sel]; main_only = [f for f in sel if f not in sens_sel]
htn_tab = pd.DataFrame({
    "定义": htn.definition, "n": htn.n, "全队列 %": htn.pct_all.map(lambda v: f"{v:.1f}"),
    "存活组 %": htn.pct_survivors.map(lambda v: f"{v:.1f}"), "死亡组 %": htn.pct_deaths.map(lambda v: f"{v:.1f}"),
    "该组 1 年死亡率 %": htn.mortality_pct.map(lambda v: f"{v:.1f}"),
    "未校正 OR (95% CI)": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(htn.or_death, htn.or_lo, htn.or_hi)],
    "p": htn.p_chi2.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")})
h_full, h_unc, h_cmp = htn.iloc[0], htn.iloc[1], htn.iloc[2]
htn_tab["定义"] = ["完整定义：401–405 / I10–I13、I15、I16（主分析）", "仅无并发症：401 / I10（敏感性分析，≈ 论文）",
                 "只有并发症编码、无 401/I10（在仅无并发症定义下记为\"无\"）", "无任何高血压编码"]
_yr = ydn[ydn.model != "RF (R defaults)"].threshold_oof
cal_tab = cal[(cal.split == "valid") & cal.config.isin(["noleak_paper", "noleak_tuned"])][
    ["config", "model", "brier", "scaled_brier", "cal_in_the_large", "cal_slope", "mean_pred", "observed"]]
cal_tab = grp(cal_tab.assign(config=cal_tab.config.str.replace("noleak_", "")), col="config").rename(columns={"config": "params"})
ovf_tab = grp(ovf[ovf.version == "noleak"][["params", "model", "train", "cv_auc_train", "valid", "gap_train_minus_valid", "flag_overfit"]])

shap_tab = pd.DataFrame({"排名": range(1, 6), "论文（RF）": SHAP_TOP5, "本研究 推荐模型（无惩罚 LR）": top5_lr,
                         "本研究 对照（L2 LR）": top5_lr_l2, "本研究 次要（调参 RF）": top5,
                         "本研究 论文配置（RF 字面参数）": top5_paperconfig})

sig = t1.loc[t1.p_value < 0.05, "variable"].tolist()

# ---------------------------------------------------------------- English tables (report_en.md)
flow_tab_en = pd.DataFrame({
    "Step": ["[1] DN patients", "[2] First ICU at first hospitalisation", "[3] Hospital stay > 48 h", "[4] Age > 18",
             "[5] Complete blood count"],
    "Paper": flow_tab["论文"].values, "This study (definition C)": flow_tab["本研究（口径 C）"].values,
    "Paper retained": flow_tab["论文 保留比例"].values, "This study retained": flow_tab["本研究 保留比例"].values})
defs_tab_en = defs_tab.copy()
defs_tab_en.columns = ["Definition", "Description", "Final n", "1-year deaths", "1-year mortality"]
defs_tab_en["Description"] = ["First admission overall has a DN code and an ICU stay", "First DN admission (must have an ICU stay)",
                              "First admission with both a DN code and an ICU stay (main analysis)",
                              "Admission of the first-ever ICU stay has a DN code (old code)"]
outc_tab_en = outc_tab.copy()
outc_tab_en.columns = ["Outcome definition", "Deaths", "Mortality"]
outc_tab_en["Outcome definition"] = ["1-year all-cause death from ICU admission (main analysis)", "1-year all-cause death from hospital admission",
                                     "In-hospital death", "In-hospital death or non-missing dod (any time)", "Paper"]
feat_tab_en = feat_tab.rename(columns={"变量": "Variable", "缺失 %": "Missing %", "本研究": "This study", "论文": "Paper"})
auc_tab_en = auc_tab.rename(columns={"组": "Group", "模型": "Model", "训练": "Training", "验证（DeLong 95% CI）": "Validation (DeLong 95% CI)",
                                     "10 折 CV": "10-fold CV", "嵌套 CV": "Nested CV"})
auc_tab_en["Group"] = auc_tab_en["Group"].replace({"论文（原文报告）": "paper (as reported)"})
dl_tab_en = dl_tab.rename(columns={"比较": "Comparison"})
lr_tab_en = lr_tab.rename(columns={"变量": "Variable", "1 SD（原始单位）": "1 SD (original units)",
                                   "无惩罚 LR β（每 SD）": "Unpenalised LR β (per SD)", "无惩罚 LR OR (95% CI)": "Unpenalised LR OR (95% CI)",
                                   "对照：L2 LR OR": "Comparison: L2 LR OR"})
lrv_tab_en = lrv_tab.rename(columns={"LR 版本": "LR variant", "训练集 OOF AUC": "Training OOF AUC", "OOF 校准斜率": "OOF calibration slope",
                                     "验证集 AUC": "Validation AUC", "验证集 Brier": "Validation Brier",
                                     "验证集校准截距": "Validation calibration intercept", "验证集校准斜率": "Validation calibration slope"})
lrv_tab_en["LR variant"] = ["Unpenalised (recommended)", "L2, C = 0.01 (comparison)"]
nest_tab_en = nest_tab.rename(columns={"模型": "Model", "原 CV AUC（LASSO 在 CV 外）": "Original CV AUC (LASSO outside CV)",
                                       "嵌套 CV AUC（LASSO 在每折内）": "Nested CV AUC (LASSO within each fold)",
                                       "乐观偏倚（原 − 嵌套）": "Optimism (original − nested)", "各折偏倚范围": "Per-fold optimism range"})
nest_tab_en["Per-fold optimism range"] = nest_tab_en["Per-fold optimism range"].str.replace(" 至 ", " to ")
htn_tab_en = htn_tab.rename(columns={"定义": "Definition", "全队列 %": "Cohort %", "存活组 %": "Survivors %", "死亡组 %": "Deaths %",
                                     "该组 1 年死亡率 %": "1-year mortality in group %", "未校正 OR (95% CI)": "Unadjusted OR (95% CI)"})
htn_tab_en["Definition"] = ["Full: 401–405 / I10–I13, I15, I16 (main analysis)", "Uncomplicated only: 401 / I10 (sensitivity, ≈ paper)",
                            "Complicated codes only, no 401/I10 (coded 'no' under the uncomplicated definition)", "No hypertension code"]
shap_tab_en = shap_tab.copy()
shap_tab_en.columns = ["Rank", "Paper (RF)", "This study: recommended (unpenalised LR)", "This study: comparison (L2 LR)",
                       "This study: secondary (tuned RF)", "This study: paper configuration (RF, literal parameters)"]
_en = lambda v: str(v).replace("（", " (").replace("）", ")").replace("，", ", ").replace(" 至 ", " to ")
sens_tab_en = pd.DataFrame({
    "Item": ["Cohort size (deaths, 1-year mortality)", "Training / validation (validation deaths)", "LASSO λ.1se (number of variables)",
             "Variables selected at λ.1se", "Recommended model (unpenalised LR) validation AUC (DeLong 95% CI)",
             "Tuned RF validation AUC (DeLong 95% CI)", "ΔAUC LR − RF (95% CI), p", "LR calibration slope / Brier",
             "RF calibration slope / Brier", "LR SHAP top 5", "RF SHAP top 5"],
    "Main analysis (all years)": [_en(v) for v in sens_tab["主分析（全部年份）"]],
    "Sensitivity (excluding 2020–2022 group)": [_en(v) for v in sens_tab["敏感性分析（排除 2020–2022 组）"]]})
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


def ncv_tab(summ, t88, zh=True):
    s = summ.loc[summ.auroc_fold_mean.sort_values(ascending=False).index]
    yes, no = ("是", "否") if zh else ("yes", "no")
    d = pd.DataFrame({
        "m": s.index,
        "a": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(s.auroc_fold_mean, s.auroc_fold_sd)],
        "p": s.auroc_pooled_mean.map(f3), "ap": s.auprc_pooled_mean.map(f3),
        "b": [f"{b:.3f} / {sb:.3f}" for b, sb in zip(s.brier_pooled_mean, s.scaled_brier_pooled_mean)],
        "cs": s.cal_slope_pooled_mean.map(lambda v: f"{v:.3f}"),
        "d": ["— (M\\*)" if np.isnan(t88.loc[m, "lo"]) else f"{t88.loc[m, 'delta']:.4f} ({t88.loc[m, 'lo']:+.4f}, {t88.loc[m, 'hi']:+.4f})"
              for m in s.index],
        "e": [yes if t88.loc[m, "equivalent"] else no for m in s.index]})
    d.columns = (["模型", "AUROC（50 折均值 ± SD）", "合并 OOF AUROC", "AUPRC（合并 OOF）", "Brier / scaled Brier（合并 OOF）",
                  "OOF 校准斜率", "ΔAUROC vs M\\*（NB 校正 95% CI）", "与 M\\* 等价"] if zh else
                 ["Model", "AUROC (mean ± SD, 50 folds)", "Pooled OOF AUROC", "AUPRC (pooled OOF)", "Brier / scaled Brier (pooled OOF)",
                  "OOF calibration slope", "ΔAUROC vs M\\* (NB-corrected 95% CI)", "Equivalent to M\\*"])
    return d


def rep_slope_tab(summ, zh=True):
    rows = []
    for m in UL2:
        r = {"m": m}
        for s_ in SEEDS:
            r[str(s_)] = (f"{summ.loc[m, f'cal_slope_pooled_rep{s_}']:.3f}（{summ.loc[m, f'abs_ln_cal_slope_pooled_rep{s_}']:.3f}）" if zh
                          else f"{summ.loc[m, f'cal_slope_pooled_rep{s_}']:.3f} ({summ.loc[m, f'abs_ln_cal_slope_pooled_rep{s_}']:.3f})")
        r["mean"] = f"{summ.loc[m, 'abs_ln_cal_slope_pooled_mean_over_repeats']:.4f}"
        rows.append(r)
    d = pd.DataFrame(rows)
    d.columns = (["模型 \\ 种子"] + [str(s_) for s_ in SEEDS] + ["平均 \\|ln 斜率\\|"]) if zh else \
                (["Model \\ seed"] + [str(s_) for s_ in SEEDS] + ["mean \\|ln slope\\|"])
    return d


def dca_better(d):
    b = d[(d.net_benefit_model > d.net_benefit_all) & (d.net_benefit_model > 0)].threshold
    return len(b), len(d), (b.min() if len(b) else np.nan), (b.max() if len(b) else np.nan)


def ho_tab(ho, dca, thr_col, zh=True):
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
            return (f"{n}/{N} 个阈值（{lo:.2f}–{hi:.2f}）" if zh else f"{n}/{N} thresholds ({lo:.2f}–{hi:.2f})")
    keys = ["auroc", "auprc", "brier", "cal", "thr", "cls", "flag", "dca"]
    prev = ho[ho.columns[ho.columns.str.startswith("prevalence")][0]].iloc[0]
    lab_zh = ["AUROC（DeLong 95% CI）", f"AUPRC（bootstrap 95% CI；阳性率基线 {prev:.3f}）", "Brier / scaled Brier", "校准截距 / 斜率",
              "阈值（嵌套 CV OOF Youden，5 次重复中位数）", "accuracy / precision / recall / F1 / specificity", "判为高风险的比例",
              "DCA（0.05–0.60）：净获益同时高于 treat-all 和 0"]
    lab_en = ["AUROC (DeLong 95% CI)", f"AUPRC (bootstrap 95% CI; prevalence baseline {prev:.3f})", "Brier / scaled Brier",
              "Calibration intercept / slope", "Threshold (nested-CV OOF Youden, median of 5 repeats)",
              "accuracy / precision / recall / F1 / specificity", "Share flagged high risk",
              "DCA (0.05–0.60): net benefit above both treat-all and 0"]
    d = pd.DataFrame({"k": lab_zh if zh else lab_en, **{m: [cell(m, k) for k in keys] for m in ms}})
    d = d.rename(columns={"k": "指标" if zh else "Metric"})
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
ct_tab = ctF.loc[CT_COMP, ["original_n", "sensitivity_n", "lost_by_acute_exclusion", "new_from_prior_admissions"]].reset_index()
ct_tab_en = ct_tab.copy()
ct_tab.columns = ["共病（charlson 分项）", "原阳性（索引住院）", "重算后阳性", "因剔除急性编码失去", "由既往住院新增"]
ct_tab_en.columns = ["Comorbidity (charlson component)", "Positive (index admission)", "Positive (recomputed)",
                     "Lost by removing acute codes", "Added from prior admissions"]
def ct_perf(zh=True):
    rows = []
    for lab_zh, lab_en, f in [("AUROC（折均值 ± SD）", "AUROC (fold mean ± SD)", lambda s: f"{s.auroc_fold_mean:.3f} ± {s.auroc_fold_sd:.3f}"),
                              ("合并 OOF AUROC", "Pooled OOF AUROC", lambda s: f3(s.auroc_pooled_mean)),
                              ("AUPRC（合并 OOF）", "AUPRC (pooled OOF)", lambda s: f3(s.auprc_pooled_mean)),
                              ("Brier（合并 OOF）", "Brier (pooled OOF)", lambda s: f3(s.brier_pooled_mean)),
                              ("校准斜率（合并 OOF）", "Calibration slope (pooled OOF)", lambda s: f3(s.cal_slope_pooled_mean))]:
        rows.append([lab_zh if zh else lab_en] + [f(S.loc[m]) for m in ["LR (unpenalised)", "RF (tuned)"] for S in (sC, ctSum)])
    rows.append(["ΔAUROC（敏感性 − 主分析，95% CI）" if zh else "ΔAUROC (sensitivity − main, 95% CI)",
                 "—", f"{ctu.mean_diff:+.3f} ({ctu.lo:+.3f}, {ctu.hi:+.3f})", "—", f"{ctr.mean_diff:+.3f} ({ctr.lo:+.3f}, {ctr.hi:+.3f})"])
    rows.append(["CCI / SAPS-II 入选折数" if zh else "Folds selecting CCI / SAPS-II",
                 f"{selC['cci']}/50, {selC['sapsii']}/50", f"{ctSel['cci']}/50, {ctSel['sapsii']}/50",
                 f"{selC['cci']}/50, {selC['sapsii']}/50", f"{ctSel['cci']}/50, {ctSel['sapsii']}/50"])
    return pd.DataFrame(rows, columns=(["指标", "无惩罚 LR：主分析", "无惩罚 LR：共病时间", "调参 RF：主分析", "调参 RF：共病时间"] if zh else
                                       ["Metric", "Unpenalised LR: main", "Unpenalised LR: comorbidity timing", "Tuned RF: main",
                                        "Tuned RF: comorbidity timing"]))
ncvC_tab, ncvC_tab_en = ncv_tab(sC, t88C), ncv_tab(sC, t88C, zh=False)
slC_tab, slC_tab_en = rep_slope_tab(sC), rep_slope_tab(sC, zh=False)
hoC_tab, hoC_tab_en = ho_tab(hoC, hodC, "threshold_nested_oof_youden_median"), ho_tab(hoC, hodC, "threshold_nested_oof_youden_median", zh=False)
ctp_tab, ctp_tab_en = ct_perf(), ct_perf(zh=False)

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

flK_tab = flK.assign(mortality_1y=flK.mortality_1y.map(lambda v: f"{v:.1%}"))
flK_tab_en = flK_tab.copy()
flK_tab_en.columns = ["Step", "Remaining", "Excluded at this step", "1-year deaths", "1-year mortality"]
flK_tab["step"] = ["[1] 有含 DN 编码且有 ICU 的住院的患者", "[2] 取索引住院（口径 C）的首次 ICU stay", "[3] 年龄 ≥ 18",
                   "[4] 排除 ICU stay < 24 h 且 24 h 时存活（转出）", "[5] 排除入科 24 h 前死亡"]
flK_tab.columns = ["步骤", "剩余", "本步排除", "1 年死亡", "1 年死亡率"]
ncvK_tab, ncvK_tab_en = ncv_tab(sK, t88K), ncv_tab(sK, t88K, zh=False)
slK_tab, slK_tab_en = rep_slope_tab(sK), rep_slope_tab(sK, zh=False)
hoK_tab, hoK_tab_en = ho_tab(hoK, hodK, "threshold_frozen"), ho_tab(hoK, hodK, "threshold_frozen", zh=False)
_dl_row = ["配对 DeLong：LR (L2) − RF (tuned)", f"{hoKl.delong_lr_minus_rf_diff:+.3f} ({hoKl.delong_lr_minus_rf_lo:+.3f}, "
           f"{hoKl.delong_lr_minus_rf_hi:+.3f}), p = {hoKl.delong_lr_minus_rf_p:.2f}", "—"]
hoK_tab.loc[len(hoK_tab)] = _dl_row
hoK_tab_en.loc[len(hoK_tab_en)] = ["Paired DeLong: LR (L2) − RF (tuned)"] + _dl_row[1:]
sel15_str = "、".join(f"{f_} {n}" for f_, n in top15K.items())
sel15_str_en = ", ".join(f"{f_} {n}" for f_, n in top15K.items())
_co = coK.copy()
coK_tab = pd.DataFrame({
    "变量": _co.feature, "1 SD（原始单位）": _co.sd_original_units.map(lambda v: f"{v:.3g}"),
    "L2 LR OR（每 SD，惩罚后）": _co.or_l2_per_sd.map(lambda v: f"{v:.2f}"),
    "无惩罚 LR OR（95% CI）": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(_co.or_glm_per_sd, _co.or_glm_lo, _co.or_glm_hi)],
    "无惩罚 LR OR（有 vs 无，95% CI）": ["—" if np.isnan(o) else f"{o:.2f} ({lo:.2f}–{hi:.2f})"
                                       for o, lo, hi in zip(_co.or_bin, _co.or_bin_lo, _co.or_bin_hi)],
    "p（无惩罚）": _co.p_glm.map(pfmt), "VIF": _co.vif.map(lambda v: f"{v:.2f}")})
coK_tab_en = coK_tab.rename(columns={"变量": "Variable", "1 SD（原始单位）": "1 SD (original units)",
                                     "L2 LR OR（每 SD，惩罚后）": "L2 LR OR (per SD, penalised)",
                                     "无惩罚 LR OR（95% CI）": "Unpenalised LR OR (95% CI)",
                                     "无惩罚 LR OR（有 vs 无，95% CI）": "Unpenalised LR OR (yes vs no, 95% CI)",
                                     "p（无惩罚）": "p (unpenalised)"})
sig_glmK = _co[_co.p_glm < 0.05].sort_values("coef_glm_per_sd", key=abs, ascending=False)
sig_glmK_str = "、".join(f"{r.feature} {r.or_bin:.2f}（{r.or_bin_lo:.2f}–{r.or_bin_hi:.2f}，有 vs 无）" if r.feature in BIN_K
                         else f"{r.feature} {r.or_glm_per_sd:.2f}（{r.or_glm_lo:.2f}–{r.or_glm_hi:.2f}，每 SD）" for r in sig_glmK.itertuples())
sig_glmK_str_en = ", ".join(f"{r.feature} {r.or_bin:.2f} ({r.or_bin_lo:.2f}–{r.or_bin_hi:.2f}, yes vs no)" if r.feature in BIN_K
                            else f"{r.feature} {r.or_glm_per_sd:.2f} ({r.or_glm_lo:.2f}–{r.or_glm_hi:.2f}, per SD)" for r in sig_glmK.itertuples())
vif_top = _co.sort_values("vif", ascending=False).head(3)
vif_str = "、".join(f"{r.feature} {r.vif:.2f}" for r in vif_top.itertuples())
vif_str_en = ", ".join(f"{r.feature} {r.vif:.2f}" for r in vif_top.itertuples())

def sens_K(zh=True):
    def c(S, m, f): return f(S.loc[m])
    M = ["LR (L2)", "RF (tuned)"]
    rows = []
    for lz, le, f in [("AUROC（折均值 ± SD）", "AUROC (fold mean ± SD)", lambda s: f"{s.auroc_fold_mean:.3f} ± {s.auroc_fold_sd:.3f}"),
                      ("AUPRC（合并 OOF）", "AUPRC (pooled OOF)", lambda s: f3(s.auprc_pooled_mean)),
                      ("Brier（合并 OOF）", "Brier (pooled OOF)", lambda s: f3(s.brier_pooled_mean)),
                      ("校准斜率（合并 OOF）", "Calibration slope (pooled OOF)", lambda s: f3(s.cal_slope_pooled_mean))]:
        rows.append([lz if zh else le] + [c(S, m, f) for m in M for S in (sK, sumP, sumW)])
    rows.insert(1, ["ΔAUROC（敏感性 − 主分析，95% CI）" if zh else "ΔAUROC (sensitivity − main, 95% CI)"] +
                [v for m in M for v in ("—", f"{vmP.loc[m, 'mean_diff']:+.3f} ({vmP.loc[m, 'lo']:+.3f}, {vmP.loc[m, 'hi']:+.4f})",
                                        f"{vmW.loc[m, 'mean_diff']:+.3f} ({vmW.loc[m, 'lo']:+.3f}, {vmW.loc[m, 'hi']:+.3f})")])
    ns = [f"{a} ({b}–{c_})" for a, b, c_ in (nselK, nselP, nselW)]
    rows.append(["LASSO 入选变量数（中位数，范围）" if zh else "Variables selected by LASSO (median, range)"] + ns + ns)
    cols = (["指标", "LR (L2) 主分析", "LR (L2) prior_only", "LR (L2) no_wbc_diff", "调参 RF 主分析", "调参 RF prior_only", "调参 RF no_wbc_diff"]
            if zh else ["Metric", "LR (L2) main", "LR (L2) prior_only", "LR (L2) no_wbc_diff", "Tuned RF main", "Tuned RF prior_only",
                        "Tuned RF no_wbc_diff"])
    return pd.DataFrame(rows, columns=cols)
sensK_tab, sensK_tab_en = sens_K(), sens_K(zh=False)

_t1 = t1K[t1K.variable != "n"]
t1K_tab = pd.DataFrame({"变量": _t1.variable, "缺失 %": _t1.missing_pct.map(lambda v: f"{v:.1f}"), "总体": _t1.total,
                        "存活": _t1.survival, "死亡": _t1.non_survival, "p": _t1.p_display})
t1K_tab_en = t1K_tab.rename(columns={"变量": "Variable", "缺失 %": "Missing %", "总体": "Overall", "存活": "Survivors", "死亡": "Deaths"})
n_sigK = int((t1K.p_value < 0.05).sum())
n_surv_K, n_death_K = int(t1K.survival.iloc[0]), int(t1K.non_survival.iloc[0])

# ---- appendix A: stage 4-7 single-validation-set tables (CSV links)
def csv_link(name, sub="tables"):
    return f"[`{sub}/{name}`]({sub}/{name})"

# ---------------------------------------------------------------- report
rep = f"""# 复现与临床改良报告：MIMIC-IV v3.1 糖尿病神经病变（DN）患者 1 年死亡风险机器学习预测

原文：Huang Y, Mo W, et al. *Machine learning models predict mortality risk in diabetic neuropathy patients using MIMIC-IV data.* Sci Rep 2025;15:38702.

本报告全部由 `scripts/05_report.py` 从聚合结果表生成，不含任何患者级数据。所有口径决策及理由见 [`decisions.md`](decisions.md)（下文用 D 编号引用），各阶段摘要见 [`progress.md`](progress.md)。报告分三部分：第一部分为论文复现（口径 C），第二部分为临床改良模型（CLIN），第三部分为与论文的比较、局限性和后续工作。

## 0. 结论速览

- **目标**：在 MIMIC-IV **v3.1** 上按论文方法完成研究，不追求人数、死亡率或指标吻合论文。论文使用更早的 v2.x（覆盖 2008–2019）。论文不明确之处选方法学上最合理的方案；论文中的错误不照搬。
- **论文复现（口径 C，第一部分）**：{n_final:,} 人，1 年死亡 {d_final}（{d_final / n_final:.1%}；论文 1,313 人、38.2%）。
  - **主要性能估计 = 训练集（1,568 人）上的完整嵌套 CV**（5 次重复 × 10 折，变量选择与调参都在折内完成，D8.11）。{len(eqC)} 个模型的外层 AUROC 在 {eqC_lo:.3f}–{eqC_hi:.3f} 之间，按预设规则彼此等价；论文参数的 RF（两版）和 SVM 为 {neqC_lo:.3f}–{neqC_hi:.3f}，不等价。
  - **推荐模型 = 无惩罚 LR**（按预设规则 D8.8 选出，外层 AUROC {uC.auroc_fold_mean:.3f}，在等价模型中最低，比最佳模型低 {t88C.loc['LR (unpenalised)', 'delta']:.3f}）。推荐理由是简约、可解释和 EPV，不是区分度更好；两种 LR 的校准相当。该规则写定时已看过口径 C 的阶段 4–7 结果，对口径 C 不是盲态的（6.2）。
  - 单次留出集（392 人）上无惩罚 LR 的 AUROC 为 {hoCu.auroc:.3f}（{hoCu.auroc_lo:.3f}–{hoCu.auroc_hi:.3f}），只作为次要估计：该验证集参与过阶段 6–7 的事后模型选择（6.3）。
  - 共病改用"既往住院 ∪ 索引住院慢性病编码"后，{int(ctS['CCI changed (n)'])} 人的 CCI 改变，心梗阳性中有一半（{int(mi.lost_by_acute_exclusion)}/{int(mi.original_n)}）只来自索引住院的急性编码，但 AUROC 变化 ≤ {max(abs(ctu.mean_diff), abs(ctr.mean_diff)):.3f}（9.1）。
- **临床改良模型（CLIN，第二部分）**：预测时点为 ICU 入科后 24 h，{int(flK.n.iloc[-1]):,} 人（1 年死亡 {flK.mortality_1y.iloc[-1]:.1%}）。按同一预设规则在训练集嵌套 CV 中选出 **L2 LR**（外层 AUROC {l2K.auroc_fold_mean:.3f}；与无惩罚 LR 近乎打平，由校准规则裁决）。模型冻结后在留出集（512 人）上只评估一次：**AUROC {hoKl.auroc:.3f}（{hoKl.auroc_lo:.3f}–{hoKl.auroc_hi:.3f}）**，比嵌套 CV 低约 {l2K.auroc_fold_mean - hoKl.auroc:.2f}，调参 RF 同步下降（{rK.auroc_fold_mean:.3f} → {hoKr.auroc:.3f}）；留出集校准斜率 {hoKl.cal_slope:.2f}，部署前需要再校准。只用既往住院诊断时 AUROC 下降 {-pP.mean_diff:.3f}（保守估计）；去掉白细胞分类变量没有损失。
- **最重要的预测因子**：两条线一致，RDW、CCI、年龄居前 3（口径 C 推荐 LR 的 SHAP 前 5：{", ".join(top5_lr)}；CLIN L2 LR 前 3：{", ".join(shap10K[:3])}）。与论文一致的是 RDW 第 1、CCI 前 3、低血氯提示风险升高。
- **与论文不一致**：中性粒细胞不再是重要预测因子（论文中该变量的分布提示可能受单位混杂影响，D2.11）；RF 并不优于 LR（论文 RF 训练集 AUC 0.999 与过拟合一致，6.5）。

## 1. 数据、工具与研究设计

- MIMIC-IV v3.1（hosp/icu 模块的 `.csv.gz`，只读），DuckDB 1.5.6 直接查询；列类型取自 mimic-code 官方 schema（D0.1）。
- 派生变量来自 MIT-LCP/mimic-code `concepts_duckdb`（commit 见 `external/MIMIC_CODE_COMMIT.txt`），在只含候选患者的子集库中运行（D0.2）。
- Python 3.11 / scikit-learn 1.9 / xgboost 3.4 / shap 0.51（`environment.yml`），全局随机种子 42；嵌套 CV 的 5 次重复用种子 42–46。
- **两条线**（D8.1）：
  - **论文复现（口径 C）**：首个同时含 DN 编码和 ICU 的住院及其首次 ICU stay（报告中命名为 "first eligible DN-ICU admission"），保留论文的 LOS > 48 h、年龄 > 18、CBC 8 项完整条件和索引住院共病。
  - **临床改良模型（CLIN）**：索引住院同上，预测时点为 ICU 入科后 24 h，只用该时点已知的信息，不用 LOS 条件，不要求 CBC 完整，共病取既往住院与索引住院慢性病编码。
- **结局**（D8.10）：死亡日期 ∈ [date(intime), date(intime) + 365 天]，院内死亡用 `deathtime`，院外死亡用 `dod`，在日期粒度比较。核查显示与旧定义在所有口径中逐人一致（不一致 0 人）。`dod` 覆盖至末次出院后 1 年，因此入科后 365 天内的死亡都在随访范围内。
- **分析计划的时间点**（decisions.md 第 8 节）：阶段 8 的模型选择规则（D8.8）、指标和敏感性分析在运行任何阶段 8 分析之前写定。对 **CLIN** 是盲态的（写定时没有任何 CLIN 结果）；对**口径 C 不是盲态的**，因为此前已看过口径 C 在阶段 4–7 的全部结果。执行中的修订都以"修订（日期、原因）"的形式记录在 decisions.md 中。

# 第一部分：论文复现（口径 C）

## 2. 队列构建（阶段 1）

### 2.1 流程与论文对比

{md(flow_tab)}

第 [1] 步定义为"任一住院含 DN 编码的患者"。我们的人数是论文的 2.2 倍，原因有二：v3.x 纳入了 2020–2022 年的住院（最终队列中 12.6% 来自 2020–2022 年组），论文第 [1] 步可能的口径也不同。**后续各步的保留比例与论文几乎相同**（LOS 95.5% vs 95.4%；年龄步骤均无人被排除；CBC 64.3% vs 64.2%），提示纳入/排除标准的实现与论文基本一致，人数差异可能主要来自数据版本。CBC 缺失几乎完全来自白细胞分类（缺失 35.6%，CBC 本体仅缺 1%），见 `tables/cbc_missingness.csv`。

### 2.2 "first ICU admission at first hospitalization" 的四种口径（D1.2）

{md(defs_tab)}

**本研究采用口径 C**：每位患者取第一次同时含 DN 编码且有 ICU 的住院及其首次 ICU stay。DN 是慢性病，更早的无 ICU 或无 DN 编码的住院不改变患者是否属于研究人群；口径 A、B、D 会因就诊史而排除合格患者，引入选择偏倚。论文没有说明具体的操作定义，上表只描述四种理解下的人数与死亡率。口径 A（论文字面口径）因选择偏倚和样本量小（837 人 / 189 例死亡）没有建模，该决定在口径 A 没有任何模型结果时做出（D8.1 修订）。

### 2.3 结局与死亡率差异（D1.6、D1.7、D8.10）

{md(outc_tab)}

- 主结局：自 ICU 入科起 1 年全因死亡（D8.10 的日期粒度定义，与旧定义逐人一致）。以入院为起点结果相同（入院与 ICU 之间没有死亡发生）。
- **死亡率差距（25.2% vs 38.2%）的来源**：MIMIC-IV v2.0+ 的 `dod` 覆盖末次出院后 1 年内的院外死亡。若把"`dod` 非空"都算作死亡，本队列死亡率为 {o.loc['death_any_dod', 'rate']:.1%}，与论文的 38.2% 同一量级。在院外有 `dod` 的患者中，49% 死于 ICU 入科 365 天之后。论文可能用了这种随访长度不一致的口径，但原文没有说明结局的具体定义，无法确认。

## 3. 特征提取（阶段 2）

55 个预测变量（论文说的"56 个"包括结局，正文实际列出 55 个）。全部 itemid 经 d_labitems / d_items 核对（`tables/itemid_audit_lab.csv`、`itemid_audit_chart.csv`），缺失率全部 ≤ 7.04%。

### 3.1 与论文 Table 1 总体列比较

{md(feat_tab, 2)}

绝大多数变量与论文几乎相同（中位数/IQR 相差通常 ≤ 1 个测量单位）。主要差异：
1. **高血压 84.6% vs 37.3%**：两种定义的差异见 3.3。
2. **淋巴/中性/单核细胞与 PLR、PNR**：见 3.2。
3. APS-III、LODS、CCI 略低，GCS 略高（中位数 15 vs 14）：病情略轻，可能与 v3.x 新增的 2020–2022 年患者、编码习惯的变化有关；方向和量级一致。
4. 钙：论文标注 mmol/L，但数值量级与 mg/dL 相符（MIMIC 的 50893 为 mg/dL），可能是单位标注有误；本研究使用 mg/dL。

### 3.2 细胞计数的单位混杂（D2.1、D2.2、D2.11）

- 旧代码错误：51146 是嗜碱性粒细胞 %，不是单核细胞；51133（淋巴细胞绝对值，K/uL）与 51244（淋巴细胞 %）被混合求均值。
- 论文 Table 1 的淋巴 3.69 (1.27–118.32)、中性 23.90 (8.63–831.30)、单核 2.00 (0.60–43.64)（单位标为 10⁹/L）在生理上不可能，这一分布与**跨患者的尺度混杂**（K/uL、%、#/uL）一致。支持这一解释的证据：用正确的绝对计数（mimic-code `blood_differential`，K/uL）计算，**NLR、LMR、SII、PWR 与论文几乎完全一致**（NLR 6.82 vs 6.80；SII 1336 vs 1342）。这些比值在同一患者三种细胞尺度相同时不受影响；而把血小板与淋巴/中性细胞组合的 **PLR（159 vs 36）和 PNR（22 vs 6.6）严重偏离**。
- 本研究的混合口径诊断（仅作说明，不用于建模）：

{md(mix, 2)}

- **对论文结论的影响**：如果上述解释成立，论文的 neutrophils_mean（SHAP 第 2 位）和 PLR/PNR 可能部分反映的是检测格式或年代，而不是生理量。本研究用正确单位后，中性粒细胞绝对值只在 λ.min 中被选入，不在 λ.1se 中，也不在 SHAP 前 5。

### 3.3 高血压的两种定义（D2.9、D4.10）

论文没有给出高血压的编码。两种可能的定义：

- **完整定义（主分析）**：Quan-Elixhauser 高血压，包括无并发症与有并发症两部分。ICD-9 为 401–405；ICD-10 为 I10–I13、I15、I16。
- **仅无并发症定义**：只看 401/I10 编码。这一定义下的患病率 {h_unc.pct_all:.1f}% 与论文 37.3% 接近，提示论文可能使用了该定义（原文未给出编码，无法确认）。

编码规则决定了两者的差异。ICD 规定，高血压合并 CKD 或心衰时应编码为 I12/I13（ICD-9 403/404），**不再**另编 I10/401。因此，DN 患者中最常见、病情最重的"高血压 + 糖尿病肾病/心衰"人群，在仅无并发症定义下会被记为"无高血压"。

{md(htn_tab)}

- 按完整定义，高血压与 1 年死亡无关：OR {h_full.or_death:.2f}（{h_full.or_lo:.2f}–{h_full.or_hi:.2f}），p = {h_full.p_chi2:.2f}。
- 按仅无并发症定义，高血压看起来是**保护因素**：OR {h_unc.or_death:.2f}（{h_unc.or_lo:.2f}–{h_unc.or_hi:.2f}）。在本队列中，这一现象主要可由错分解释：被记为"无高血压"的人群里有 {int(h_cmp.n):,} 名只有并发症编码的患者，其死亡率为 {h_cmp.mortality_pct:.1f}%（OR {h_cmp.or_death:.2f}）。
- **主分析采用完整定义**（标准可引用、语义正确、避免系统性错分）。两种定义下 LASSO λ.1se 都没有选入高血压，选出的变量完全相同（`tables/lasso_sensitivity_hypertension.csv`），模型不受影响。

## 4. 预处理与 Table 1（阶段 3）

- 固定分层 80/20 划分（seed 42）：训练集 1,568 人（死亡 25.19%）/ 验证集 392 人（25.26%）。阶段 8 的完整嵌套 CV 只用训练集，验证集不读入。
- 两个版本（D3.4）：**忠实版**（插补/标准化在全数据上拟合，同论文）和**无泄露版**（只在训练集上拟合）。两者的 LASSO 选择完全相同，模型指标差异 ≤ 0.001。以下以无泄露版为主。完整嵌套 CV 中，20% 缺失率筛选、插补和标准化都在每个外层训练折内完成（D8.4），各折均未剔除变量。
- Table 1（`tables/table1.csv`）使用**插补前的观测数据**。论文可能是在插补后统计的（D3.5）。样本量约 1,500 时 Shapiro-Wilk 对所有连续变量都拒绝正态，因此全部报中位数 (IQR) 和 Mann-Whitney U 检验（D3.6）。
- 组间 p < 0.05 的变量有 {len(sig)} 个（论文 42 个）。方向一致：死亡组年龄更大、心衰/肾病更多、肥胖更少、阴离子间隙/肌酐/BUN/RDW/PT/PTT/INR 更高、Hb/RBC/血氯/MBP 更低、各评分和 NLR/SII 更高、LMR 更低。不一致：慢性肺病、高血压、血糖、WBC、PWR 在本研究中不显著。

## 5. LASSO 特征选择（阶段 4，Fig. 2）

| | 论文 | 本研究（无泄露） | 本研究（忠实） |
|---|---|---|---|
| λ.min（变量数） | {LASSO_LAMBDA_MIN:.5f} (21) | {lasso.loc['noleak', 'lambda_min']:.5f} ({lasso.loc['noleak', 'n_min']}) | {lasso.loc['faithful', 'lambda_min']:.5f} ({lasso.loc['faithful', 'n_min']}) |
| λ.1se（变量数） | {LASSO_LAMBDA_1SE:.5f} (12) | {lasso.loc['noleak', 'lambda_1se']:.5f} ({lasso.loc['noleak', 'n_1se']}) | {lasso.loc['faithful', 'lambda_1se']:.5f} ({lasso.loc['faithful', 'n_1se']}) |

- **λ.1se 选出的变量**（完整训练集，用于最终模型）：{", ".join(sel)}
- **与论文 12 个变量重合 {len(overlap)}/12**：{", ".join(overlap)}。论文选了、我们在 λ.1se 下没选的：{lasso.loc['noleak', 'paper_not_selected']}；**λ.min 下与论文重合 {len(overlap_min)}/12**（只缺 pwr）。
- 我们多选的：lymphocytes_mean、sapsii、oasis、nlr、sii。如果论文的中性/淋巴细胞变量受单位混杂影响（3.2），就会改变炎症相关变量之间的竞争，以及它们与评分之间的共线结构。
- **完整嵌套 CV 中的入选频率**（50 个外层折各做一次 LASSO，`replication/nested_cv_full_selection_C.csv`）：每折入选 {nselC[1]}–{nselC[2]} 个变量（中位数 {nselC[0]}）；50/50 折入选：{", ".join(sel50C)}；≥ 40/50 折：{", ".join(sel40C)}。
- λ.1se 下的标准化系数（按绝对值排序）：

{md(lcoef, 3)}

![](figures/fig2a_lasso_cv_noleak.png)
![](figures/fig2b_lasso_path_noleak.png)

## 6. 模型表现

### 6.1 主要估计：完整嵌套交叉验证（D8.5、D8.6、D8.11）

**设计**：训练集 1,568 人，外层分层 10 折 × 5 次重复（种子 42–46）。每个外层训练折内依次完成 20% 缺失率筛选、插补、标准化、LASSO（内层 10 折，λ.1se）和调参（内层分层 5 折 `GridSearchCV`，AUROC），然后在外层验证折上评估。9 个模型均跑满 5 次重复；没有 LASSO 选出 0 个变量的折。指标：外层 50 折的 AUROC 均值 ± SD；每次重复把 10 个验证折合并为覆盖全部训练集的 OOF 预测，计算 AUPRC、Brier、校准斜率等，再对 5 次重复取平均。模型间比较用逐折配对差值和 Nadeau–Bengio 校正的 95% CI（r = 5，k = 10）。输出见 `replication/nested_cv_full_*_C.csv`。

{md(ncvC_tab)}

AUPRC 的基线（阳性率）为 {uC.prevalence_pooled_mean:.3f}。RF（R 默认）有 {int(sC.loc['RF (R defaults) (paper)', 'n_nonconv_cal_intercept_folds'])} 个外层折的校准截距不收敛，其校准只报告合并 OOF 值。

- **{len(eqC)} 个模型等价**：外层 AUROC {eqC_lo:.3f}–{eqC_hi:.3f}，与最佳模型的差值都 ≤ {t88C.loc[[m for m in eqC if m != mstarC], 'delta'].max():.3f}。
- **不等价的是论文参数的 RF（两版）和 SVM**（{neqC_lo:.3f}–{neqC_hi:.3f}）：RF（论文字面参数）Δ = {t88C.loc['RF (paper)', 'delta']:.3f}，CI 不跨 0；RF（R 默认）和 SVM（论文参数）Δ ≥ {min(t88C.loc['RF (R defaults) (paper)', 'delta'], t88C.loc['SVM (paper)', 'delta']):.3f}。论文参数中只有 XGBoost 与最佳模型持平。
- 与阶段 7 的"仅 LASSO 嵌套"CV（附录 A.5）相比，完整嵌套后无惩罚 LR 为 {n_rec.cv_auc_nested:.3f} → {uC.auroc_fold_mean:.3f}，调参模型为 {nest_tuned_lo:.3f}–{nest_tuned_hi:.3f} → {sC.loc[['RF (tuned)', 'XGBoost (tuned)', 'SVM (tuned)', 'LR (L2)']].auroc_fold_mean.min():.3f}–{sC.loc[['RF (tuned)', 'XGBoost (tuned)', 'SVM (tuned)', 'LR (L2)']].auroc_fold_mean.max():.3f}，结论不变。阶段 7 只有 1 次重复（种子 42），完整嵌套 CV 为 5 次重复的平均，差异主要来自重复间的变动：完整嵌套 CV 中种子 42 那一次重复的值为无惩罚 LR {u42C:.3f}、调参 RF {r42C:.3f}，无惩罚 LR 在 5 次重复间为 {u_rep_lo:.3f}–{u_rep_hi:.3f}。
- L2 LR 的折均值 AUROC（{l2C.auroc_fold_mean:.3f}）高于合并 OOF AUROC（{l2C.auroc_pooled_mean:.3f}）：内层选出的 C 很小，强收缩使各折预测尺度不同，合并后排序受损。

### 6.2 模型选择（预设规则 D8.8）

**规则**（在运行任何阶段 8 分析前写定）：(1) 外层 AUROC 均值最高者为 M\\*；(2) 与 M\\* 的差值 ≤ 0.01，或 Nadeau–Bengio 95% CI 跨 0 且差值 ≤ 0.02 的模型与 M\\* 等价；(3) 等价集中有 LR 则选 LR；(4) 两种 LR 都等价时，选合并 OOF 校准斜率的 |ln 斜率| 更小者（每次重复先取 |ln 斜率|，再对 5 次重复平均），相同时选无惩罚 LR；(5) 留出集不参与选择。**第 2、3 步是预设的简约性偏好**：它允许所选 LR 的 AUROC 比最佳模型最多低 0.02（CI 跨 0 时）或 0.01（无论 CI）。

**执行**：
1. M\\* = {mstarC}（外层 AUROC {sC.loc[mstarC, 'auroc_fold_mean']:.4f}）。
2. 等价集 = {", ".join(eqC)}。
3. 等价集中有 LR → 选 LR。
4. 两种 LR 都等价，比较每次重复的校准斜率（括号内为 |ln 斜率|）：

{md(slC_tab)}

   平均 |ln 斜率|：无惩罚 LR {uC.abs_ln_cal_slope_pooled_mean_over_repeats:.3f}，L2 LR {l2C.abs_ln_cal_slope_pooled_mean_over_repeats:.3f} → **无惩罚 LR**。差距很小，L2 LR 在 {n_l2_closerC}/5 次重复中更接近 1；这一步只是按规则分出先后，不构成"无惩罚 LR 校准更好"的证据。两者偏离 1 的程度相近、方向相反（无惩罚 LR 斜率 {uC.cal_slope_pooled_mean:.2f}，预测略偏极端；L2 LR {l2C.cal_slope_pooled_mean:.2f}，略偏向均值）。

**推荐模型 = 无惩罚 LR**。推荐理由只有三点：简约（{int(hoCu.n_selected)} 个 LASSO 变量上的无惩罚 logistic 模型）、可解释（系数/OR 直接可读，给出 95% CI 和 p 值，无需选择惩罚强度）、EPV 充足（{int(hoCu.events_train)} 例事件 / {int(hoCu.n_selected)} 个入选变量 = {hoCu.epv_selected:.1f}）。校准方面两种 LR 相当。**不声称区分度更好**：无惩罚 LR 的外层 AUROC（{uC.auroc_fold_mean:.3f}）在 {len(eqC)} 个等价模型中最低，比 M\\* 低 {t88C.loc['LR (unpenalised)', 'delta']:.3f}（95% CI {t88C.loc['LR (unpenalised)', 'lo']:+.3f} 至 {t88C.loc['LR (unpenalised)', 'hi']:+.3f}）。

**预设主分析模型（D4.7：调参 RF）照常报告**：外层 AUROC {rC.auroc_fold_mean:.3f}，与 M\\* 等价；OOF 校准斜率 {rC.cal_slope_pooled_mean:.2f}，预测向均值收缩。

**模型选择的历史**（D4.15，如实记录）：阶段 4 的预设规则（D4.7，训练集 10 折 CV AUC 最高）选出的是调参 RF（CV AUC {cv.loc['RF']:.3f}，与 XGBoost 并列；验证集 AUC {best_auc.auc:.3f}）。验证集配对 DeLong 检验中各调参模型两两差异经 Holm 校正均不显著，之后在看到验证集结果的情况下做了两次改动：先改为调参 L2 LR（C = 0.01），因其验证集校准斜率 {calt.loc['LR', 'cal_slope']:.2f} 再改为无惩罚 LR。这两次改动都不在当时的分析计划中。阶段 8 的 D8.8 规则选出的同样是无惩罚 LR，但该规则写定时已看过上述结果，因此对口径 C 而言**不是盲态的**；它的作用是把选择依据固定为嵌套 CV 上的预设标准，而不是证明事后选择正确。

### 6.3 次要估计：单次留出验证集（D8.12）

**注意：该验证集（392 人，{int(hoCu.events_valid)} 例死亡）参与过阶段 6–7 的事后模型选择，这里只是次要估计。**两个模型都是阶段 4 在完整训练集上拟合的版本（无泄露预处理 + λ.1se 选出的 {int(hoCu.n_selected)} 个变量）；阈值取各自嵌套 CV 合并 OOF 的 Youden 阈值（5 次重复中位数）。输出 `replication/holdout_C.csv`、`holdout_dca_C.csv`。

{md(hoC_tab)}

- 两个模型的留出集 AUROC 与嵌套 CV 估计（{uC.auroc_fold_mean:.3f}、{rC.auroc_fold_mean:.3f}）相差 ≤ {ho_gapC:.3f}（LR {hoCu.auroc:.3f} vs {uC.auroc_fold_mean:.3f}，RF {hoCr.auroc:.3f} vs {rC.auroc_fold_mean:.3f}），没有显示额外的过拟合。
- 调参 RF 留出集校准斜率 {hoCr.cal_slope:.2f}，与嵌套 CV OOF 的 {rC.cal_slope_pooled_mean:.2f} 方向一致（向均值收缩）；无惩罚 LR 为 {hoCu.cal_slope:.2f}。
- 留出集只有 {int(hoCu.events_valid)} 例死亡，两个模型 AUROC 的 CI 大幅重叠，不据此比较两者优劣。

### 6.4 阈值

主阈值改为**嵌套 CV 合并 OOF 预测的 Youden 最佳阈值（5 次重复中位数）**：无惩罚 LR {thrC.loc['LR (unpenalised)', 'median']:.3f}（范围 {thrC.loc['LR (unpenalised)', 'min']:.3f}–{thrC.loc['LR (unpenalised)', 'max']:.3f}），调参 RF {thrC.loc['RF (tuned)', 'median']:.3f}。阶段 4 在非嵌套 10 折 OOF 上得到的 {lr_thr.loc['youden_oof', 'threshold']:.3f}（D4.13）和阈值 0.5 的结果保留在附录 A.3。阈值 0.5 不适合本研究：1 年死亡率约 25%，阈值 0.5 时推荐 LR 只把 {lr_thr.loc['0.5', 'flagged_pct']:.1%} 的患者判为高风险，recall 只有 {lr_thr.loc['0.5', 'recall']:.2f}；它还隐含"漏判一例死亡与误判一例存活代价相同"的假设。Youden 指数对 sensitivity 与 specificity 赋予相同权重，这本身也是一种假设，实际阈值应结合 DCA 和使用场景选择。论文报告的 RF 分类指标只有在以存活为阳性类时才能自洽（D4.6），不能与本研究比较。

### 6.5 阶段 4–7 单次验证集结果（次要，历史）

阶段 4–7 在单次 80/20 划分的验证集上做了全部模型的比较。详细表格移到附录 A，这里只保留要点：
- **区分度**：验证集 AUC 全部在 0.76–0.80 之间，95% CI 宽约 ±0.05；调参模型两两配对 DeLong 检验经 Holm 校正后均不显著（最小校正 p = {dlt.p_holm.min():.2f}）。无惩罚 LR 与调参 RF 的 ΔAUC = {rf_lr['diff']:+.3f}（{rf_lr.diff_lo:+.3f} 至 {rf_lr.diff_hi:+.3f}），p = {rf_lr.p_value:.2f}（附录 A.1、A.2）。
- **校准**（附录 A.4）：无惩罚 LR 斜率 {cal_rec.cal_slope:.2f}，L2 LR {calt.loc['LR', 'cal_slope']:.2f}，调参 RF {calt.loc['RF', 'cal_slope']:.2f}，RF（R 默认）0.17。
- **过拟合与论文 RF 训练集 AUC 0.999**：

{md(ovf_tab)}

判定规则：训练集 − 验证集 AUC > 0.10 或训练集 AUC > 0.95（D4.12）。
  - **RF（R 默认：9 棵完全生长的树，mtry=2）**：训练集 AUC **0.999**、验证集 0.756，差 0.243，**与论文 RF 的 0.999 / 0.780 模式相近**。这与"论文的 RF 是 `randomForest(ntree=9, mtry=2)` 的完全生长树（nodesize=1），文中列出的 `max_depth=10, min_samples_split=30, min_samples_leaf=10` 未生效"的解释一致：按字面参数，我们的训练集 AUC 只有 0.887。但无法排除其他原因，例如论文在不同的数据上计算训练集 AUC，或使用了未报告的其他设置。树完全生长加上只有 9 棵树，会在训练集上记住每个样本；训练集 AUC 没有任何信息量，概率严重过度自信（校准斜率 0.17）。在完整嵌套 CV 中，RF（R 默认）的外层 AUROC 只有 {sC.loc['RF (R defaults) (paper)', 'auroc_fold_mean']:.3f}。
  - 如果论文把训练集 AUC 0.999 当作 RF 最优的依据之一，这是不恰当的；应该比较交叉验证或独立验证集的表现。

## 7. 临床效用：DCA 与 CIC

- **主要（嵌套 CV 合并 OOF，`replication/nested_cv_full_dca_C.csv`）**：阈值 0.05–0.60 上，5 次重复平均净获益同时高于 treat-all 和 0 的阈值数：无惩罚 LR {dcaCu[0]}/{dcaCu[1]}（{dcaCu[2]:.2f}–{dcaCu[3]:.2f}），调参 RF {dcaCr[0]}/{dcaCr[1]}（{dcaCr[2]:.2f}–{dcaCr[3]:.2f}）。
- **次要（留出验证集）**：见 6.3 的表，无惩罚 LR {dca_better(hodC[hodC.model == 'LR (unpenalised)'])[0]}/56，调参 RF {dca_better(hodC[hodC.model == 'RF (tuned)'])[0]}/56。
- 阶段 4 在验证集上的 DCA/CIC 图保留如下（论文为 0.1–0.7）：推荐 LR 的有用区间约 {useful_range_m(UNPEN)}；在 Youden 阈值附近，每 1,000 人中约 {cic_n:.0f} 人被判为高风险，其中约 {cic_e:.0f} 人实际死亡。阈值 0.1–0.4 内各模型净获益相差不超过 {dca_spread:.3f}。

![](figures/fig5_dca_cic_primary_noleak_recommended.png)
![](figures/fig5b_dca_models_noleak.png)

## 8. 模型解释（Fig. 6、Fig. 8）

### 8.1 推荐模型：无惩罚 LR 的系数与 SHAP

推荐的无惩罚 LR 在 12 个 LASSO 变量的标准化值上拟合（完整训练集），系数即每增加 1 个标准差（SD）的对数优势比，给出 Wald 95% CI 和 p 值（`tables/lr_coefficients.csv`）。最后一列附调参 L2 LR 的 OR 作对照。

{md(lr_tab)}

- **独立且显著的危险因素**（p < 0.05）：RDW、CCI、年龄、APS-III、OASIS。**保护因素**：chloride_max 较高（即低血氯提示风险升高，与论文方向一致）、淋巴细胞绝对值较高。
- **CCI 的解读**（D8.2）：mimic-code 的 CCI 含 0–4 分年龄分，而年龄已单独进入模型。CCI 的 OR 是在年龄固定时的条件效应，主要反映共病负担和年龄分组中未被线性年龄捕捉的部分，不能解读为纯共病效应。
- **共线性**：SAPS-II、SII、NLR 在多变量模型中不显著，VIF 约 3.7–4.3，提示其信息已被其他变量覆盖。单个系数不应理解为因果效应。
- **SHAP（无惩罚 LR，LinearExplainer，验证集，对数优势尺度）**：前 5 为 {", ".join(top5_lr)}。L2 LR 的前 5 为 {", ".join(top5_lr_l2)}。

![](figures/fig8_lr_odds_ratios_noleak.png)
![](figures/fig6a_shap_bar_primary_noleak_recommended.png)
![](figures/fig6b_shap_beeswarm_primary_noleak_recommended.png)

force plot（存活、院内死亡、出院后 1 年内死亡各一例代表性病例，D4.9）显示单个患者的特征值，只保留在本机（D5.1）。

### 8.2 SHAP 与论文比较

{md(shap_tab)}

- **与论文一致**：在所有分析中，RDW_mean 都是**第 1 位**的预测因子；CCI 都在前 3。chloride_max 在 RF 中排第 5（低血氯 → 风险升高，方向与论文一致），在 LR 中为显著的保护因素。
- **LR 与 RF 的一致性**：推荐 LR 与调参 RF 的前 5 中有 {len(common_top5)} 个相同（{", ".join(common_top5)}）。
- **不一致**：neutrophils_mean 不在我们的模型中，论文中它排第 2。我们推测这可能与单位混杂有关（3.2），但无法直接验证。

![](figures/fig6a_shap_bar_secondary_noleak_tuned.png)
![](figures/fig6b_shap_beeswarm_secondary_noleak_tuned.png)

## 9. 敏感性分析

### 9.1 共病时间（D8.2、D8.12）

论文条件下的共病来自索引住院的 `diagnoses_icd`，诊断编码没有时间戳，其中可能包含本次住院的急性事件。敏感性分析把 CCI（保留年龄分）和 SAPS-II 慢性病项改为按"既往住院的全部编码 ∪ 索引住院去掉 D8.3 急性编码后的编码"重算（直接复用 mimic-code SQL；只用索引住院编码重跑时与原概念表逐人一致），只替换这两列，对推荐 LR 和调参 RF 重跑完整嵌套 CV（外层划分与主分析相同）。

- {int(ctS['patients with >= 1 prior admission'])} 人（{int(ctS['patients with >= 1 prior admission']) / n_final:.0%}）有既往住院；{int(ctS['patients with >= 1 D8.3 acute code in the index admission'])} 人的索引住院含 D8.3 定义的急性编码。CCI 中位数（IQR）重算前后均为 {ctS['CCI original, median (IQR)']}，{int(ctS['CCI changed (n)'])} 人的 CCI 改变（升高 {int(ctS['CCI increased (n)'])}、降低 {int(ctS['CCI decreased (n)'])}）；SAPS-II {int(ctS['SAPS-II changed (n)'])} 人改变。

{md(ct_tab)}

- **最大的变化来自心梗**：{int(mi.original_n)} 个原阳性中，{int(mi.lost_by_acute_exclusion)} 人（约一半）的心梗只有索引住院的急性编码（410/I21/I22），重算后不再计为既往心梗。口径 C 索引住院中 586/N19 编码为 0 人，因此肾病的剔除规则不起作用。
- `diabetes_with_cc` 不是 100%：有 <10 人的 DN 编码为 249.6x（继发性糖尿病伴神经病变），不在 charlson 糖尿病编码表内（更正 D8.2 的原表述）。

{md(ctp_tab)}

- **结论**：尽管 {int(ctS['CCI changed (n)'])} 人的 CCI 改变、心梗约一半来自急性编码，两个模型的区分度、AUPRC、Brier 和 LR 的校准都基本不变（ΔAUROC CI 跨 0），CCI 仍在每折入选。"CCI 是稳定的重要预测因子"这一结论不依赖于索引住院编码的时间归属。

### 9.2 排除 2020–2022 年组患者（D4.19）

排除 `anchor_year_group` 为 "2020 - 2022" 的 {int(sL.n_excluded)} 名患者，在剩余人群中重跑建模流程（`scripts/04e_sensitivity_pre2020.py`，重新做单次分层 80/20 划分）。

{md(sens_tab)}

- **LASSO**：选出 {int(sL.n_selected)} 个变量，新增 {", ".join(sens_only) or "无"}，缺少 {", ".join(main_only) or "无"}。
- **结论**：{sens_verdict}这里的验证集是重新划分的单次划分，AUC 差异也包含随机划分带来的变动。`anchor_year_group` 是患者层面的 3 年锚定区间，不是每次住院的真实年份，只能近似"排除 2020 年以后的住院"。

# 第二部分：临床改良模型（CLIN）

## 10. 设计、队列与特征（D8.1–D8.4、D8.13）

**预测时点 = ICU 入科后 24 h**（landmark），只用该时点已知的信息：
- **纳入**：索引住院与口径 C 相同（首个同时含 DN 编码和 ICU 的住院），取其首次 ICU stay；年龄 ≥ 18；入科 24 h 时仍在 ICU 且存活。首次 ICU stay 不足 24 h 的患者直接排除，不改用同一住院中后来的 ICU stay。**不使用** LOS > 48 h（依赖预测时点之后的信息），**不要求** CBC 完整。
- **特征**：化验、生命体征和评分的时间窗同复现线（首日）。6 个共病变量取"既往住院全部编码 ∪ 索引住院去掉急性事件编码后的编码"（D8.3：剔除急性心梗、急性心衰、586/N19、急性卒中/TIA、主动脉夹层、急性消化性溃疡、急性肝坏死/休克肝等）；CCI 不含年龄分（避免与年龄重复计入）；SAPS-II 慢性病项在"既往 ∪ 索引住院"上重算。
- **白细胞分类**（D8.4 方案 2）：8 个白细胞分类变量（三种细胞计数及 NLR、PLR、LMR、PNR、SII）不受 20% 规则约束，增设 1 个指示变量 `wbc_diff_missing`（三种细胞计数全部缺失）作为 LASSO 候选变量；缺失值在折内用均值插补。

{md(flK_tab)}

- **划分**：分层 80/20（seed 42）：训练集 {int(spK.loc['train', 'n']):,} 人（死亡 {int(spK.loc['train', 'deaths'])}，{spK.loc['train', 'mortality']:.1%}），留出集 {int(spK.loc['valid', 'n'])} 人（死亡 {int(spK.loc['valid', 'deaths'])}，{spK.loc['valid', 'mortality']:.1%}）。
- **共病**：{int(cmK['has_prior_admission = 1 (n)']):,} 人（{float(cmK['has_prior_admission = 1 (%)']):.1f}%）有既往住院；不含年龄分的 CCI 中位数（IQR）只用索引住院编码为 {cmK['CCI (no age score), index admission only, median (IQR)']}，按 CLIN 规则为 {cmK['CCI (no age score), CLIN rule, median (IQR)']}（{int(cmK['CCI changed vs index-only (n)']):,} 人改变）。D8.3 的 586/N19 计数为 0（< 1%），按预设不做加回的敏感性分析。
- **白细胞分类**：`wbc_diff_missing = 1` 的有 {n_wbc_missing} 人（{n_wbc_missing / int(cmK['CLIN stays']):.1%}）；部分缺失 {partial_wbc} 人，均因比值分母为 0（D2.10）。缺失率 > 20% 的只有这 8 个白细胞分类变量（`clinical/missingness_clin.csv`），其余变量最高为 {max_other_missK.feature}（{max_other_missK.missing_pct:.1f}%）。

**Table 1（CLIN 全队列，按 1 年结局分组，插补前的观测值）**（`clinical/table1_clin.csv`；规则同 D3.5、D3.6；存活 {n_surv_K:,} 人、死亡 {n_death_K} 人；p < 0.05 的变量 {n_sigK} 个）：

{md(t1K_tab)}

## 11. 嵌套交叉验证与模型选择（D8.14）

训练集 {int(spK.loc['train', 'n']):,} 人，设计与 6.1 相同（5 × 10 折，r = 5，k = 10）。候选变量 {int(recK['n_candidates'])} 个（55 个 + `wbc_diff_missing`）；20% 规则在各折均未剔除变量；LASSO 每折入选 {nselK[1]}–{nselK[2]} 个变量（中位数 {nselK[0]}）。输出 `clinical/nested_cv_full_*_clin.csv`。

{md(ncvK_tab)}

AUPRC 的基线（阳性率）为 {l2K.prevalence_pooled_mean:.3f}。

**按 D8.8 执行**：
1. M\\* = **LR (L2)**（外层 AUROC {l2K.auroc_fold_mean:.4f}）。
2. 等价集 = {", ".join(eqK)}（与 M\\* 的差值都 ≤ {eqK_maxd:.4f}）。RF（论文字面参数）Δ = {t88K.loc['RF (paper)', 'delta']:.4f} > 0.02，不等价；SVM（论文参数）、RF（R 默认）差距更大。
3. 等价集中有 LR → 选 LR。
4. 两种 LR 都等价，比较每次重复的校准斜率（括号内为 |ln 斜率|）：

{md(slK_tab)}

   平均 |ln 斜率|：L2 LR {l2K.abs_ln_cal_slope_pooled_mean_over_repeats:.4f}，无惩罚 LR {uK.abs_ln_cal_slope_pooled_mean_over_repeats:.4f} → **L2 LR**。

**规则在近乎打平时做出裁决**：两种 LR 的 AUROC 相差 {t88K.loc['LR (unpenalised)', 'delta']:.4f}（CI 跨 0），平均 |ln 斜率| 只差 {uK.abs_ln_cal_slope_pooled_mean_over_repeats - l2K.abs_ln_cal_slope_pooled_mean_over_repeats:.3f}，而且无惩罚 LR 在 {n_unpen_closerK}/5 次重复中更接近 1。这一步只是按预设规则分出先后，不构成"L2 LR 校准更好"的证据；两者偏离 1 的程度相近、方向相反（L2 LR {l2K.cal_slope_pooled_mean:.2f}，无惩罚 LR {uK.cal_slope_pooled_mean:.2f}）。CLIN 所选模型与口径 C（无惩罚 LR）不同，原因只在这一步。

- **不声称区分度更好**：L2 LR 恰好是 M\\*，但与等价集中其余模型的差距都 ≤ {eqK_maxd:.3f} 且 CI 跨 0。推荐理由只写简约、可解释和"两种 LR 校准相当"。第 2、3 步是预设的简约性偏好。
- L2 LR 内层调参在 50/50 折都选 C = 0.01（网格 0.001–100 及 ∞，0.01 不在边界），属于较强收缩。
- **调参 RF**：外层 AUROC {rK.auroc_fold_mean:.3f}，与 M\\* 等价；合并 OOF 校准斜率 {rK.cal_slope_pooled_mean:.2f}，预测向均值收缩，与口径 C 方向一致。
- **LASSO 入选频率**（50 折中的次数）前 15：{sel15_str}；`wbc_diff_missing` {selK['wbc_diff_missing']}/50。6 个共病中只有 obesity 入选（{selK['obesity']}/50），心梗、心衰、慢性肺病、肾病、高血压均为 0/50。
- **阈值**（嵌套 CV 合并 OOF Youden，5 次重复中位数）：L2 LR {thrK.loc['LR (L2)', 'median']:.4f}（{thrK.loc['LR (L2)', 'min']:.3f}–{thrK.loc['LR (L2)', 'max']:.3f}），调参 RF {thrK.loc['RF (tuned)', 'median']:.4f}（{thrK.loc['RF (tuned)', 'min']:.3f}–{thrK.loc['RF (tuned)', 'max']:.3f}）。

![](figures/figC4_calibration_clin_nested_oof.png)

## 12. 冻结与留出集确认性评估（D8.14、D8.15、D8.17）

### 12.1 冻结流程

- **冻结规格** `clinical/frozen_model_clin.json` 在读取留出集之前写定并提交：在训练集上依次做 20% 筛选（白细胞分类豁免）→ 插补 → 标准化 → LASSO（内层 10 折，seed 42，λ.1se）→ LR (L2) 内层分层 5 折 `GridSearchCV` 选 C；对照调参 RF 同流程；阈值为嵌套 CV 合并 OOF Youden 阈值（LR {specK['thresholds']['LR (L2)']}，RF {specK['thresholds']['RF (tuned)']}）；同时记录训练集 stay_id 哈希、候选矩阵哈希、代码哈希和 git commit。
- `scripts/04h_final_clin.py` 按规格拟合，拟合记录 `clinical/final_model_clin.json` 含模型文件的 sha256。
- `scripts/04i_holdout_clin.py` 在读取留出集前检查三道防线：冻结文件已提交且无改动；模型文件与冻结规格一致；输出文件不存在（不允许覆盖）。运行前用训练集子样本做了冒烟测试（不读留出集）。正式评估**只运行了一次**。
- 两项敏感性分析（第 14 节）在留出集评估之后运行；它们只针对已选模型与调参 RF，不参与模型选择，也不改变冻结模型。
- 为绘图（Fig. C1–C3），`scripts/04j_plots_clin.py` 用冻结模型文件对留出集重新预测（不重新拟合任何东西），重算的全部指标与 `holdout_clin.csv`、`holdout_dca_clin.csv` 逐项一致（差 < 1e-9），未改变任何结论（D8.17）。

### 12.2 最终模型

- 训练集 {int(recK['n_train']):,} 人，事件 {int(recK['events'])} 例；候选变量 {int(recK['n_candidates'])} 个，20% 规则剔除 {int(recK['n_dropped_missing'])} 个。
- LASSO λ.1se = {recK['lambda_1se']:.4f}，入选 **{int(recK['n_selected'])} 个**变量（处于嵌套 CV 各折 {nselK[1]}–{nselK[2]} 的上端）：{", ".join(recK['selected'])}。
- LR (L2)：C = {recK['params']['LR (L2)']['C']}；调参 RF：{", ".join(f"{k} {v}" for k, v in recK['params']['RF (tuned)'].items())}。
- **EPV = {recK['epv_selected']:.1f}**（事件数 / 入选变量数）；事件数 / 候选变量数 = {recK['events_per_candidate']:.1f}。

### 12.3 留出集结果（确认性估计）

留出集 {int(hoKl.n_eval)} 人，死亡 {int(hoKl.events_eval)} 例。输出 `clinical/holdout_clin.csv`、`holdout_dca_clin.csv`。

{md(hoK_tab)}

![](figures/figC1_roc_clin_holdout.png)
![](figures/figC2_calibration_clin_holdout.png)
![](figures/figC3_dca_clin_holdout.png)

### 12.4 留出集 AUROC 低于嵌套 CV 估计

- **CLIN 的确认性估计是留出集 AUROC {hoKl.auroc:.3f}（{hoKl.auroc_lo:.3f}–{hoKl.auroc_hi:.3f}）**。它比嵌套 CV 估计低约 {l2K.auroc_fold_mean - hoKl.auroc:.2f}：LR (L2) {l2K.auroc_fold_mean:.3f} → {hoKl.auroc:.3f}，调参 RF {rK.auroc_fold_mean:.3f} → {hoKr.auroc:.3f}。**两个模型同步下降**，嵌套 CV 估计落在留出集 95% CI 的上限之外。
- 以下核查是看到结果后做的描述性分析，不改变冻结模型，也不改变确认性结果（`clinical/holdout_gap_diagnostics_clin.csv`）：
  - **不像是过拟合**：冻结 LR 在训练集上的表观 AUROC 为 {gd['apparent_auroc_train_lr_l2']:.3f}，只比嵌套 CV 高 {gd['apparent_auroc_train_lr_l2'] - l2K.auroc_fold_mean:.3f}；L2 强收缩（C = 0.01），嵌套 CV 本身已包含变量选择与调参。
  - **抽样变异的量级**：从训练集嵌套 CV 合并 OOF 预测中按留出集的构成（{int(hoKl.events_eval)} 例死亡 / {int(hoKl.n_eval - hoKl.events_eval)} 例存活）有放回抽样 {int(gd['oof_subsample_n_draws']):,} 次，AUROC 均值 {gd['oof_subsample_auroc_mean']:.3f}、SD {gd['oof_subsample_auroc_sd']:.3f}、2.5% 分位 {gd['oof_subsample_auroc_p2.5']:.3f}；≤ {hoKl.auroc:.3f} 的比例为 {gd['oof_subsample_p_le_holdout_auroc']:.1%}。该模拟未包含模型重拟合的变异，低估了总变异。
  - **分布差异**：{int(gd['smd_n_features'])} 个入选变量在训练集与留出集之间的标准化均值差最大 {gd['smd_max_abs']:.2f}（{gd['smd_max_abs_feature']}），{int(gd['smd_n_abs_gt_0.1'])} 个 > 0.1（{gd['smd_features_abs_gt_0.1']}）；划分是分层随机的，没有时间差异。
- **解读**：差距更可能来自 512 人留出集的抽样变异（处于抽样分布的下尾），而不是模型过拟合，但无法排除其他原因。按预设，**不据此更换模型或重新评估**。报告同时给出嵌套 CV 估计 {l2K.auroc_fold_mean:.3f} 和留出集估计 {hoKl.auroc:.3f}；两者合起来看，CLIN 的区分度在 0.75–0.80 之间。
- LR (L2) 与调参 RF 在留出集上没有差异（p = {hoKl.delong_lr_minus_rf_p:.2f}），AUPRC 和 Brier 相近。

### 12.5 校准：部署前需要再校准

- 留出集上 LR (L2) 的校准斜率为 **{hoKl.cal_slope:.2f}**（截距 {hoKl.cal_intercept:+.3f}），预测略偏极端；嵌套 CV OOF 的斜率为 {l2K.cal_slope_pooled_mean:.2f}（略偏向均值），**偏离方向在两处相反**。调参 RF 留出集斜率 {hoKr.cal_slope:.2f}（嵌套 CV {rK.cal_slope_pooled_mean:.2f}）。
- 校准在不同样本上不稳定，模型**部署前需要在目标人群中再校准**（至少重估截距与斜率），阈值也应在再校准后按使用场景重新确定。

## 13. 模型解释（`clinical/coefs_clin.csv`、`clinical/shap_clin.csv`）

LR (L2) 的惩罚后 OR（每 SD）用于描述冻结模型本身；推断用同一组 {int(recK['n_selected'])} 个变量上的无惩罚 LR（statsmodels GLM，训练集，同 D4.16），给出 Wald 95% CI、p 值和 VIF。

{md(coK_tab)}

- **无惩罚模型中 p < 0.05 的变量**（OR，95% CI；连续变量每 SD，二分类变量有 vs 无）：{sig_glmK_str}。
- **共线性**：VIF 最大为 {vif_str}，均 < 5；SAPS-II、LODS、NLR、SII 等在其余评分/比值固定时不显著。
- **SHAP**（LR (L2)，留出集，LinearExplainer，对数优势尺度，训练集为背景）前 10：{", ".join(shap10K)}。**前 3 名与口径 C 推荐模型相同**（{", ".join(shap10K[:3])}）。
- obesity 的 OR < 1（有 vs 无 {ob_glm.or_bin:.2f}，{ob_glm.or_bin_lo:.2f}–{ob_glm.or_bin_hi:.2f}）、chloride_max 的 OR < 1，方向与口径 C 的 Table 1 一致；obesity 的解读见 17.3。

## 14. 敏感性分析（D8.3、D8.4 修订，D8.16）

两项分析都只对 LR (L2) 与调参 RF 运行完整的 5 × 10 嵌套 CV，外层划分与主分析相同，ΔAUROC 为配对的 Nadeau–Bengio 95% CI。

{md(sensK_tab)}

- **prior_only**（6 个共病、不含年龄分的 CCI 和 SAPS-II 慢性病项都只用既往住院诊断；没有既往住院的患者共病记为 0；`has_prior_admission` 作为候选变量）：区分度小幅下降，CI 不跨 0（LR (L2) {pP.mean_diff:+.3f}，RF {pPr.mean_diff:+.3f}）。obesity 的入选从 {selK['obesity']}/50 降到 {selP['obesity']}/50，是最大的变化；cci {selK['cci']} → {selP['cci']}/50，myocardial_infarct {selK['myocardial_infarct']} → {selP['myocardial_infarct']}/50，`has_prior_admission` {selP['has_prior_admission']}/50。主分析使用索引住院的慢性病编码，依赖"慢性病编码代表入院前已存在的情况"这一假设；prior_only 不依赖该假设，因此**作为更保守的部署情景估计**（嵌套 CV AUROC 约 {sumP.loc['LR (L2)', 'auroc_fold_mean']:.3f}）。
- **no_wbc_diff**（8 个白细胞分类变量按 20% 规则在折内剔除，每折剔除 {n_drop_W} 个；不使用 `wbc_diff_missing`）：区分度、AUPRC、Brier、校准都基本不变（LR (L2) {pW.mean_diff:+.3f}，RF {pWr.mean_diff:+.3f}，CI 跨 0），入选变量数中位数 {nselK[0]} → {nselW[0]}。白细胞分类变量在 CLIN 中没有带来可检测的增量区分度，可以省略而不损失性能。冻结模型与主分析不变。

# 第三部分：与论文比较、局限性与后续工作

## 15. 与论文差异汇总及可能原因

| 项目 | 论文 | 本研究 | 主要原因 |
|---|---|---|---|
| 数据版本 | v2.x（2008–2019） | v3.1（2008–2022） | 新增 2020–2022 年患者 |
| DN 患者 / 最终人数 | 4,137 / 1,313 | {int(flow_tab.iloc[0, 2]):,} / {n_final:,}（口径 C） | 数据版本；后续各步保留比例一致 |
| 死亡率 | 38.2% | {d_final / n_final:.1%} | 论文可能用"dod 非空"（本队列按该口径为 {o.loc['death_any_dod', 'rate']:.1%}）；本研究用明确的 1 年时间窗 |
| 高血压患病率 | 37.3% | 84.6% | 论文可能只取了无并发症高血压（本队列 35.4%）；该定义会把高血压性 CKD/心衰错分为"无高血压"（3.3） |
| 淋巴/中性/单核细胞 | 分布异常宽 | 生理范围（K/uL） | 论文可能存在单位/尺度混杂 |
| PLR / PNR | 36 / 6.6 | 159 / 22 | 同上；NLR/LMR/SII/PWR 与论文一致，支持单位混杂的解释 |
| LASSO λ.min / λ.1se | 0.0106 / 0.0269 | {lasso.loc['noleak', 'lambda_min']:.4f} / {lasso.loc['noleak', 'lambda_1se']:.4f} | 几乎相同 |
| LASSO λ.1se 变量 | 12 | 12，重合 {len(overlap)} | 单位混杂变量、评分之间的共线性 |
| 性能估计方式 | 单次验证集 | 完整嵌套 CV（主要）+ 单次留出集（次要） | 嵌套 CV 把变量选择与调参都放在折内，估计更稳定 |
| 最佳模型 | RF（验证集 0.780） | 口径 C：{len(eqC)} 个模型等价（嵌套 CV {eqC_lo:.3f}–{eqC_hi:.3f}），按预设规则推荐无惩罚 LR；CLIN：L2 LR（嵌套 CV {l2K.auroc_fold_mean:.3f}，留出集 {hoKl.auroc:.3f}） | 论文 RF 训练集 AUC 0.999 与过拟合一致；论文参数 RF 在嵌套 CV 中不与最佳模型等价 |
| RF 训练集 AUC | 0.999 | 0.999（R 默认）/ 0.850（调参） | 与论文 RF 为完全生长树的解释一致 |
| 分类指标 | 以存活为阳性，阈值未说明 | 以死亡为阳性；嵌套 CV OOF Youden 阈值 | 论文指标可能以存活为阳性类（推算所得） |
| SHAP 前 5 | RDW, neutrophils, CCI, chloride_max, PT_min | 口径 C LR：{", ".join(top5_lr)}；CLIN LR (L2)：{", ".join(shap10K[:5])} | RDW 第 1、CCI 前 3 一致；中性粒细胞可能受单位混杂影响 |

**趋势与结论的一致性**：与论文一致的是，RDW、CCI、疾病严重度评分、年龄、低血氯是 DN 患者 1 年死亡的主要预测因子，模型区分度中等（约 0.75–0.80）。不一致的是"RF 最优"：在完整嵌套 CV 中，调参后的各类模型彼此等价，论文参数的 RF 反而显著较差。

## 16. 论文方法中的不明确之处及本研究的处理

| 不明确之处 | 本研究的处理 | 决策 |
|---|---|---|
| MIMIC-IV 版本未说明 | 使用 v3.1，数据覆盖 2008–2022 | — |
| "first ICU admission at first hospitalization" | 本研究采用口径 C（首个同时含 DN 编码与 ICU 的住院 + 其首次 ICU）；另列出 3 种替代口径的人数与死亡率 | D1.2、D8.1 |
| 死亡结局与随访时长未定义（"survival status"） | 自 ICU 入科 1 年全因死亡，日期粒度的明确时间窗 | D1.6、D8.10 |
| 首日时间窗未定义 | mimic-code：化验 [intime−6h, intime+24h]，生命体征 [intime, intime+24h] | D1.5、D2.3 |
| "complete blood counts were not missing" 所指项目 | 8 项（WBC、Hb、PLT、RBC、RDW + 三种白细胞绝对值） | D1.5 |
| 高血压、肥胖的编码未给出 | Quan-Elixhauser 定义；仅无并发症高血压作敏感性分析 | D2.9、D4.10 |
| 白细胞分类单位 | mimic-code 绝对计数（K/uL） | D2.2、D2.11 |
| 插补与 20% 缺失率筛选在全数据上进行（泄露） | 忠实版 + 无泄露版并行；嵌套 CV 中 20% 规则、插补、标准化都在折内完成 | D3.4、D8.4 |
| Table 1 在插补后统计 | 用观测数据；插补后版本另存 | D3.5 |
| LASSO 指标（正文写 MSE，图为 Binomial Deviance） | 二项偏差（glmnet 二分类默认） | D4.1 |
| RF 参数（R/sklearn 参数名混用） | 两版 RF：字面参数 / R 默认 | D4.4 |
| 调参网格未报告 | 网格搜索，在嵌套 CV 的内层完成 | D4.5、D8.5 |
| 分类指标的阳性类与阈值 | 死亡为阳性；嵌套 CV OOF Youden 阈值 | D4.6、D8.6 |
| 最佳模型的选择依据 | 预设规则 D8.8 + 完整嵌套 CV（留出集不参与选择）；阶段 4–7 的事后选择过程如实记录 | D8.8、D4.15 |
| 数据年份（v2.x 只到 2019 年） | 主分析用全部年份；敏感性分析排除 2020–2022 年组 | D4.19 |

## 17. 局限性

### 17.1 两条线共同的局限

1. **单中心、回顾性**：MIMIC-IV 只来自 BIDMC 一家医院，没有外部验证。口径 C 的留出集只有 {int(hoCu.events_valid)} 例死亡，CLIN 只有 {int(hoKl.events_eval)} 例，单次留出估计的 95% CI 宽约 ±0.05。
2. **数据版本不同**：v3.1 与论文的 v2.x 在年份覆盖上不同，与论文的逐项数值比较只能看趋势。
3. **DN 由 ICD 编码识别**：编码的敏感度和特异度未知，编码习惯随时间变化；DN 的纳入依据是索引住院编码，按"DN 为慢性病、入院前已存在"的假设处理；队列只含入住 ICU 的患者。
4. **诊断编码没有时间戳**：口径 C 按论文使用索引住院的全部编码，其中含本次住院的急性事件（心梗约一半来自急性编码，9.1）；CLIN 剔除了明确的急性编码，但保留的慢性病编码同样没有时间戳，偏瘫编码也可能来自本次住院的急性卒中。
5. **结局随访**：`dod` 覆盖至末次出院后 1 年，入科后 365 天内的死亡都在随访范围内（D8.10）；无法研究更长期的结局，无法区分死因；州登记依靠姓名、出生日期、社会安全号匹配，可能漏记外州死亡。
6. **首日特征**：只用 ICU 首日数据，不含治疗信息（如血管活性药物、机械通气）和时间动态；部分评分与原始变量有重叠和共线性。

### 17.2 论文复现线（口径 C）

7. **LOS > 48 h 依赖预测时点之后的信息**（D8.1）：该条件按论文保留，但住院时长在 ICU 首日结束时并不知道，排除了早期死亡或早期出院的患者，造成选择偏倚。CLIN 不使用该条件。
8. **CBC 完整性条件**：论文的纳入标准要求 8 项 CBC 完整，在口径 C 中排除了约 36% 的患者，主要因为没做白细胞分类。CLIN 保留了这些患者：`wbc_diff_missing = 1` 组的 1 年死亡率为 {wmK.loc[1, 'mortality']:.1%}（{int(wmK.loc[1, 'deaths'])}/{int(wmK.loc[1, 'n'])}），= 0 组为 {wmK.loc[0, 'mortality']:.1%}（{int(wmK.loc[0, 'deaths'])}/{int(wmK.loc[0, 'n'])}；χ² p = {wmK.loc[0, 'chi2_p']:.2f}）；在 CLIN 的无惩罚模型中（其余 21 个变量固定），`wbc_diff_missing` 的 OR（有 vs 无）为 {wbc_glm.or_bin:.2f}（{wbc_glm.or_bin_lo:.2f}–{wbc_glm.or_bin_hi:.2f}）。这**提示**是否送检白细胞分类与预后有关，完整病例的纳入条件可能带来选择偏倚；但该变量在嵌套 CV 中只在 {selK['wbc_diff_missing']}/50 折入选，粗死亡率差异也未达到显著，证据有限。另一方面，no_wbc_diff 敏感性分析显示白细胞分类变量本身没有增量区分度（第 14 节）。
9. **推荐模型的选择不是盲态的**：D8.8 规则写定时已看过口径 C 的阶段 4–7 结果，阶段 6–7 曾两次事后改选（6.2）。规则选出的无惩罚 LR 与事后选择一致，但这不能证明事后选择正确。
10. **口径 C 的留出集参与过事后选择**，只能作为次要估计（6.3）。

### 17.3 临床改良模型（CLIN）

11. **留出集 AUROC 低于嵌套 CV 约 {l2K.auroc_fold_mean - hoKl.auroc:.2f}**（12.4）：核查提示更可能是抽样变异，但无法排除其他原因；单次 512 人的留出集不足以精确估计性能。
12. **校准不稳定**：留出集斜率 {hoKl.cal_slope:.2f}，与嵌套 CV（{l2K.cal_slope_pooled_mean:.2f}）方向相反，部署前需要再校准（12.5）。
13. **索引住院慢性病编码的时间归属**：只用既往住院诊断时 AUROC 下降 {-pP.mean_diff:.3f}（第 14 节）。主分析的估计依赖"慢性病编码在入院时已存在"的假设；prior_only 的估计（约 {sumP.loc['LR (L2)', 'auroc_fold_mean']:.3f}）更保守。
14. **肥胖编码可能存在与病情相关的偏倚**：CLIN 中 obesity 的 OR < 1（无惩罚模型，有 vs 无 {ob_glm.or_bin:.2f}，{ob_glm.or_bin_lo:.2f}–{ob_glm.or_bin_hi:.2f}）。只用既往住院诊断时，obesity 的入选从 {selK['obesity']}/50 降到 {selP['obesity']}/50。描述性比较（`clinical/obesity_source_clin.csv`）：肥胖编码只出现在索引住院的患者 1 年死亡率为 {ob_ix.mortality:.1%}（n = {int(ob_ix.n)}），肥胖编码见于既往住院的为 {ob_pr.mortality:.1%}（n = {int(ob_pr.n)}），无肥胖编码的为 {ob_no.mortality:.1%}（n = {int(ob_no.n)}）。"保护"关联主要集中在只由索引住院编码的肥胖上。这**提示**肥胖编码可能受病情相关的记录习惯影响（例如病情较轻、住院较平稳的患者更可能被记录肥胖这类次要诊断），不一定反映肥胖本身的保护作用（"肥胖悖论"）；需要用 BMI 等客观测量验证。

## 18. 后续工作

1. **时间外推验证与外部数据**：在 MIMIC-IV 内按 `anchor_year_group` 做时间外推验证（例如较早年份组训练、较晚年份组验证，可在 `04e` 的基础上扩展），评估时间漂移；寻找有一年随访的外部数据做外部验证。不使用 eICU-CRD：它只有院内结局，无法验证 1 年死亡。
2. **再校准与阈值**：对推荐模型（尤其 CLIN 的 LR (L2)，留出集斜率 {hoKl.cal_slope:.2f}）在目标人群中做再校准（截距与斜率重估），报告再校准后的 DCA；按具体临床用途（如姑息治疗讨论、加强随访）确定阈值。
3. **肥胖编码偏倚**：从 `omr` 表提取 BMI，比较编码肥胖与 BMI 定义的肥胖，检验 17.3 第 14 条的提示。
4. **特征**：加入 HbA1c、治疗变量和纵向特征；比较不含复合评分（SAPS-II 等）的模型，减少共线性，提高可解释性；评估 RDW 与死亡的剂量-反应关系（限制性立方样条），以及在不同 DN 亚型（1 型/2 型）中的一致性。

# 附录

## 附录 A：阶段 4–7 单次验证集的详细结果（次要）

以下为阶段 4–7 在单次 80/20 划分验证集（392 人，99 例死亡）上的结果。主要估计已改为完整嵌套 CV（6.1），这些表格只作为过程记录。

### A.1 AUC（DeLong 95% CI）（{csv_link('model_performance.csv')}、{csv_link('nested_cv_auc.csv')}）

{md(auc_tab)}

`recommended` 组即无惩罚 LR（与论文参数组的 LR 是同一模型）；`tuned` 组为 10 折 CV 调参模型（LR 带 L2 惩罚）；"嵌套 CV"列为阶段 7 的"仅 LASSO 嵌套"CV（A.5）。调参得到的最优参数：`{json.dumps({k: v for k, v in tuned.items() if k.startswith('noleak')}, ensure_ascii=False)}`。

AUC 相对排序（验证集）：论文 {paper_rank}；本研究论文参数 {ours_rank_paper}；本研究调参 {ours_rank_tuned}。

![](figures/fig3_roc_noleak_main.png)

### A.2 模型两两比较（配对 DeLong 检验，Holm 校正）（{csv_link('delong_pairwise.csv')}）

{md(dl_tab)}

### A.3 阈值与分类指标（{csv_link('youden_thresholds.csv')}、{csv_link('threshold_metrics.csv')}）

阈值（阶段 4，非嵌套 10 折 OOF 的 Youden 阈值，D4.13）：

{md(ydn_tab)}

Youden 阈值下的验证集指标：

{md(thry_tab)}

阈值 0.5 下的验证集指标（对照）：

{md(thr05_tab)}

![](figures/fig4_metrics_noleak_tuned.png)

### A.4 校准（{csv_link('calibration.csv')}、{csv_link('lr_variants.csv')}）

{md(cal_tab)}

{md(lrv_tab)}

![](figures/fig7_calibration_noleak_main.png)

### A.5 "仅 LASSO 嵌套"的 CV（阶段 7，D4.18）（{csv_link('nested_cv_auc.csv')}、{csv_link('nested_cv_folds.csv')}）

外层用与阶段 4 相同的 10 折（seed 42），每折内重做插补、标准化和 LASSO，超参数固定为阶段 4 的值，只隔离变量选择的偏倚。已被完整嵌套 CV（6.1）取代。

{md(nest_tab)}

## 附录 B：文件清单

- **口径 C 队列与特征**：`tables/cohort_flow.csv`、`outcome_definitions.csv`、`outcome_check.csv`、`cbc_missingness.csv`、`dn_codes.csv`、`itemid_audit_*.csv`、`feature_vs_paper.csv`、`unit_mixing_diagnostic.csv`、`hypertension_definitions.csv`、`table1.csv`、`table1_after_imputation.csv`、`missingness.csv`
- **口径 C 阶段 4–7**：`tables/lasso_*.csv`、`model_performance.csv`、`tuned_hyperparameters.json`、`calibration.csv`、`overfitting.csv`、`youden_thresholds.csv`、`threshold_metrics.csv`、`delong_pairwise.csv`、`lr_variants.csv`、`lr_coefficients.csv`、`shap_importance*.csv`、`dca_*.csv`、`cic_*.csv`、`nested_cv_*.csv`、`sensitivity_pre2020.csv`
- **口径 C 阶段 8**（`replication/`）：`nested_cv_full_*_C.csv`（完整嵌套 CV）、`holdout_C.csv`、`holdout_dca_C.csv`（留出集次要估计）、`comorb_time_*_C.csv`、`nested_cv_full_*_C_comorb_time.csv`（共病时间敏感性分析）
- **CLIN**（`clinical/`）：`cohort_flow_clin.csv`、`split_clin_summary.csv`、`comorb_clin_*.csv`、`wbc_diff_patterns_clin.csv`、`missingness_clin.csv`、`table1_clin.csv`、`wbc_diff_missing_mortality_clin.csv`、`obesity_source_clin.csv`、`nested_cv_full_*_clin.csv`、`frozen_model_clin.json`、`final_model_clin.json`、`holdout_clin.csv`、`holdout_dca_clin.csv`、`holdout_gap_diagnostics_clin.csv`、`coefs_clin.csv`、`shap_clin.csv`、`nested_cv_full_*_clin_{{prior_only,no_wbc_diff}}.csv`
- **图**：`figures/fig2a–fig8_*.png`（口径 C；名称带 `primary` 的以无惩罚 LR 为主模型，带 `secondary` 的为调参 RF）；`figures/figC1–C4_*.png`（CLIN）。单患者 force plot `fig6c–e` 只保留在本机
"""
# 已归档扩展只读聚合表；重新生成中文报告时保留第四部分，不触发训练。
from report_extensions import integrate_extensions_report
rep = integrate_extensions_report(rep, R)
(R / "report.md").write_text(rep)
print(f"report written: results/report.md ({len(rep):,} chars)")

# English version: same numbers and tables, template in scripts/templates/report_en.md (an f-string body)
_tmpl = (Path(__file__).parent / "templates" / "report_en.md").read_text()
assert '"""' not in _tmpl
rep_en = eval('f"""' + _tmpl + '"""')
(R / "report_en.md").write_text(rep_en)
print(f"report written: results/report_en.md ({len(rep_en):,} chars)")
print(f"cohort C: M* {mstarC}, selected {chosenC} (nested CV {uC.auroc_fold_mean:.3f}); holdout {hoCu.auroc:.3f}")
print(f"CLIN: M* {mstarK}, selected {chosenK} (nested CV {l2K.auroc_fold_mean:.3f}); holdout {hoKl.auroc:.3f} "
      f"({hoKl.auroc_lo:.3f}-{hoKl.auroc_hi:.3f})")
