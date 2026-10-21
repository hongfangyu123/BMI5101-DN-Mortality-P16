# Replication and clinical-model report: machine-learning prediction of 1-year mortality in patients with diabetic neuropathy (DN), MIMIC-IV v3.1

Original paper: Huang Y, Mo W, et al. *Machine learning models predict mortality risk in diabetic neuropathy patients using MIMIC-IV data.* Sci Rep 2025;15:38702.

This report is generated entirely by `scripts/05_report.py` from aggregate result tables and contains no patient-level data. It is the English version of [`report.md`](report.md) and carries the same numbers. All definitions and their rationale are recorded in [`decisions.md`](decisions.md) (cited below by D number; written in Chinese), and stage summaries in [`progress.md`](progress.md). The report has three parts: Part I, replication of the paper (definition C); Part II, the clinical model (CLIN); Part III, comparison with the paper, limitations and future work.

## 0. Summary

- **Aim**: to carry out the paper's study on MIMIC-IV **v3.1** following its methods, without trying to match its sample size, mortality or metrics. The paper used an earlier v2.x release (covering 2008–2019). Where the paper is unclear, we chose the most defensible method; errors in the paper were not copied.
- **Replication (definition C, Part I)**: {n_final:,} patients, {d_final} deaths within 1 year ({d_final / n_final:.1%}; paper 1,313 patients, 38.2%).
  - **Main performance estimate = full nested CV on the training set (1,568 patients)** (5 repeats × 10 folds, with variable selection and tuning inside the folds; D8.11). {len(eqC)} models have outer AUROC between {eqC_lo:.3f} and {eqC_hi:.3f} and are equivalent under the pre-specified rule; the paper-parameter RF (both versions) and SVM reach {neqC_lo:.3f}–{neqC_hi:.3f} and are not equivalent.
  - **Recommended model = unpenalised LR** (selected by the pre-specified rule D8.8; outer AUROC {uC.auroc_fold_mean:.3f}, the lowest among the equivalent models, {t88C.loc['LR (unpenalised)', 'delta']:.3f} below the best). The reasons are simplicity, interpretability and EPV, not better discrimination; the two LRs are comparably calibrated. The rule was written after the stage 4–7 results of definition C had been seen, so for definition C it is not blind (Section 6.2).
  - On the single held-out set (392 patients) the unpenalised LR has AUROC {hoCu.auroc:.3f} ({hoCu.auroc_lo:.3f}–{hoCu.auroc_hi:.3f}), a secondary estimate only: this validation set was used in the post-hoc model choice of stages 6–7 (Section 6.3).
  - Recomputing the comorbidities from "prior admissions ∪ chronic codes of the index admission" changes CCI for {int(ctS['CCI changed (n)'])} patients, and half of the myocardial-infarction positives ({int(mi.lost_by_acute_exclusion)}/{int(mi.original_n)}) rest only on acute codes of the index admission, yet AUROC changes by ≤ {max(abs(ctu.mean_diff), abs(ctr.mean_diff)):.3f} (Section 9.1).
- **Clinical model (CLIN, Part II)**: prediction 24 h after ICU admission, {int(flK.n.iloc[-1]):,} patients (1-year mortality {flK.mortality_1y.iloc[-1]:.1%}). The same pre-specified rule selects the **L2 LR** in nested CV on the training set (outer AUROC {l2K.auroc_fold_mean:.3f}; a near tie with the unpenalised LR, decided by the calibration rule). After freezing, the model was evaluated once on the held-out set (512 patients): **AUROC {hoKl.auroc:.3f} ({hoKl.auroc_lo:.3f}–{hoKl.auroc_hi:.3f})**, about {l2K.auroc_fold_mean - hoKl.auroc:.2f} below nested CV, with the tuned RF dropping in parallel ({rK.auroc_fold_mean:.3f} → {hoKr.auroc:.3f}); the held-out calibration slope is {hoKl.cal_slope:.2f}, so recalibration is needed before deployment. Using prior-admission diagnoses only lowers AUROC by {-pP.mean_diff:.3f} (conservative estimate); dropping the white-cell differential costs nothing.
- **Most important predictors**: consistent across both tracks, with RDW, CCI and age in the top 3 (SHAP top 5 of the recommended LR in definition C: {", ".join(top5_lr)}; top 3 of the CLIN L2 LR: {", ".join(shap10K[:3])}). Consistent with the paper: RDW first, CCI in the top 3, low chloride indicating higher risk.
- **Disagreements with the paper**: neutrophils are no longer an important predictor (the distribution of this variable in the paper suggests mixed units, D2.11); RF is not better than LR (the paper's RF training AUC of 0.999 is consistent with overfitting, Section 6.5).

## 1. Data, tools and study design

- MIMIC-IV v3.1 (`.csv.gz` files of the hosp/icu modules, read-only), queried directly with DuckDB 1.5.6; column types taken from the official mimic-code schema (D0.1).
- Derived variables come from MIT-LCP/mimic-code `concepts_duckdb` (commit in `external/MIMIC_CODE_COMMIT.txt`), run in a subset database containing only candidate patients (D0.2).
- Python 3.11 / scikit-learn 1.9 / xgboost 3.4 / shap 0.51 (`environment.yml`); global random seed 42; seeds 42–46 for the 5 nested-CV repeats.
- **Two tracks** (D8.1):
  - **Replication (definition C)**: the first admission with both a DN code and an ICU stay and its first ICU stay (named "first eligible DN-ICU admission" in this report), keeping the paper's conditions LOS > 48 h, age > 18 and complete 8-item CBC, and comorbidities from the index admission.
  - **Clinical model (CLIN)**: the same index admission, with prediction 24 h after ICU admission, using only information available at that time, no LOS condition, no complete-CBC requirement, and comorbidities from prior admissions plus chronic codes of the index admission.
- **Outcome** (D8.10): date of death ∈ [date(intime), date(intime) + 365 days], using `deathtime` for in-hospital and `dod` for out-of-hospital deaths, compared at date granularity. A check showed patient-by-patient agreement with the old definition for every cohort definition (0 disagreements). `dod` covers up to 1 year after the last discharge, so all deaths within 365 days of ICU admission fall within follow-up.
- **Timing of the analysis plan** (decisions.md Section 8): the stage-8 model-selection rule (D8.8), metrics and sensitivity analyses were written down before any stage-8 analysis was run. For **CLIN** they are blind (no CLIN result existed); for **definition C they are not blind**, because all stage 4–7 results of definition C had been seen. Amendments made during execution are recorded in decisions.md as "revision (date, reason)".

# Part I: Replication of the paper (definition C)

## 2. Cohort construction (stage 1)

### 2.1 Flow compared with the paper

{md(flow_tab_en)}

Step [1] is defined as "patients with a DN code in any admission". Our count is 2.2 times the paper's, for two reasons: v3.x adds admissions from 2020–2022 (12.6% of the final cohort comes from the 2020–2022 group), and the paper's step [1] may have been defined differently. **The retention at each later step is almost identical to the paper** (LOS 95.5% vs 95.4%; nobody excluded at the age step in either; CBC 64.3% vs 64.2%), suggesting that the inclusion/exclusion criteria were implemented much as in the paper and that the difference in numbers may come mainly from the data release. Missing CBC is almost entirely due to the white-cell differential (35.6% missing; the CBC itself only 1%), see `tables/cbc_missingness.csv`.

### 2.2 Four readings of "first ICU admission at first hospitalization" (D1.2)

{md(defs_tab_en)}

**This study uses definition C**: for each patient, the first admission with both a DN code and an ICU stay, and the first ICU stay of that admission. DN is a chronic disease, so earlier admissions without an ICU stay or without a DN code do not change whether a patient belongs to the study population; definitions A, B and D exclude eligible patients because of their care history and introduce selection bias. The paper does not state its operational definition, and the table only describes the headcount and mortality under four readings. Definition A (the paper's literal wording) was not modelled because of its selection bias and small size (837 patients / 189 deaths); this decision was made before any model result existed for definition A (D8.1 revision).

### 2.3 Outcome and the mortality gap (D1.6, D1.7, D8.10)

{md(outc_tab_en)}

- Primary outcome: 1-year all-cause death from ICU admission (the date-granularity definition of D8.10, identical patient by patient to the old definition). Starting from hospital admission gives the same result (no deaths occurred between admission and ICU).
- **Source of the mortality gap (25.2% vs 38.2%)**: from v2.0, MIMIC-IV `dod` covers out-of-hospital deaths up to 1 year after the last discharge. If every non-missing `dod` were counted as a death, mortality in our cohort would be {o.loc['death_any_dod', 'rate']:.1%}, the same order as the paper's 38.2%. Among patients with an out-of-hospital `dod`, 49% died more than 365 days after ICU admission. The paper may have used this definition with unequal follow-up, but it does not state its outcome definition, so this cannot be confirmed.

## 3. Feature extraction (stage 2)

55 predictors (the paper's "56" includes the outcome; its text lists 55). All itemids were checked against d_labitems / d_items (`tables/itemid_audit_lab.csv`, `itemid_audit_chart.csv`); all missing rates are ≤ 7.04%.

### 3.1 Comparison with the paper's Table 1 (overall column)

{md(feat_tab_en, 2)}

Most variables are nearly identical to the paper (medians/IQRs usually differ by ≤ 1 unit). Main differences:
1. **Hypertension 84.6% vs 37.3%**: see Section 3.3 for the two definitions.
2. **Lymphocytes / neutrophils / monocytes and PLR, PNR**: see Section 3.2.
3. APS-III, LODS and CCI slightly lower, GCS slightly higher (median 15 vs 14): slightly less severe illness, possibly related to the 2020–2022 patients added in v3.x and to changes in coding practice; direction and magnitude are consistent.
4. Calcium: the paper labels it mmol/L, but the values match mg/dL (MIMIC itemid 50893 is in mg/dL), so the unit label may be wrong; we use mg/dL.

### 3.2 Mixed units in the cell counts (D2.1, D2.2, D2.11)

- Errors in the old code: 51146 is basophils %, not monocytes; 51133 (absolute lymphocytes, K/uL) and 51244 (lymphocytes %) were averaged together.
- The paper's Table 1 values for lymphocytes 3.69 (1.27–118.32), neutrophils 23.90 (8.63–831.30) and monocytes 2.00 (0.60–43.64) (labelled 10⁹/L) are physiologically impossible; this distribution is consistent with **mixing of scales across patients** (K/uL, %, #/uL). Evidence for this explanation: computed from correct absolute counts (mimic-code `blood_differential`, K/uL), **NLR, LMR, SII and PWR are almost identical to the paper** (NLR 6.82 vs 6.80; SII 1336 vs 1342). These ratios are unaffected when all three cell counts of a patient share the same scale, whereas **PLR (159 vs 36) and PNR (22 vs 6.6), which combine platelets with lymphocytes/neutrophils, deviate strongly**.
- Our mixed-unit diagnostic (illustration only, not used for modelling):

{md(mix, 2)}

- **Implication for the paper's conclusions**: if this explanation holds, the paper's neutrophils_mean (2nd in SHAP) and PLR/PNR may partly reflect test format or period rather than physiology. With correct units, absolute neutrophils are selected only at λ.min in our analysis, not at λ.1se, and are not in the SHAP top 5.

### 3.3 Two definitions of hypertension (D2.9, D4.10)

The paper does not give its hypertension codes. Two possible definitions:

- **Full definition (main analysis)**: Quan-Elixhauser hypertension, uncomplicated and complicated. ICD-9 401–405; ICD-10 I10–I13, I15, I16.
- **Uncomplicated only**: only 401/I10 codes. Prevalence under this definition is {h_unc.pct_all:.1f}%, close to the paper's 37.3%, suggesting the paper may have used it (the paper gives no codes, so this cannot be confirmed).

The coding rules drive the difference. When hypertension coexists with CKD or heart failure, ICD requires I12/I13 (ICD-9 403/404) and **no separate** I10/401. So the most common and sickest DN group, "hypertension plus diabetic kidney disease or heart failure", is recorded as "no hypertension" under the uncomplicated definition.

{md(htn_tab_en)}

- Under the full definition, hypertension is unrelated to 1-year death: OR {h_full.or_death:.2f} ({h_full.or_lo:.2f}–{h_full.or_hi:.2f}), p = {h_full.p_chi2:.2f}.
- Under the uncomplicated definition, hypertension looks **protective**: OR {h_unc.or_death:.2f} ({h_unc.or_lo:.2f}–{h_unc.or_hi:.2f}). In our cohort this is mainly explained by misclassification: the "no hypertension" group contains {int(h_cmp.n):,} patients with only complicated codes, whose mortality is {h_cmp.mortality_pct:.1f}% (OR {h_cmp.or_death:.2f}).
- **The main analysis uses the full definition** (standard and citable, correct meaning, avoids systematic misclassification). Under both definitions LASSO λ.1se does not select hypertension and the selected variables are identical (`tables/lasso_sensitivity_hypertension.csv`), so the model is unaffected.

## 4. Preprocessing and Table 1 (stage 3)

- Fixed stratified 80/20 split (seed 42): training 1,568 patients (25.19% deaths) / validation 392 (25.26%). The stage-8 full nested CV uses the training set only; the validation set is not read.
- Two versions (D3.4): **faithful** (imputation/standardisation fitted on all data, as in the paper) and **no-leakage** (fitted on the training set only). Both give identical LASSO selections and model metrics that differ by ≤ 0.001. The no-leakage version is used from here on. In the full nested CV, the 20% missing-rate rule, imputation and standardisation are all done inside each outer training fold (D8.4); no variable was dropped in any fold.
- Table 1 (`tables/table1.csv`) uses **observed data before imputation**. The paper may have computed its Table 1 after imputation (D3.5). With about 1,500 patients, Shapiro-Wilk rejects normality for every continuous variable, so all are reported as median (IQR) with Mann-Whitney U tests (D3.6).
- {len(sig)} variables differ between groups at p < 0.05 (paper: 42). Same direction as the paper: deaths are older, have more heart failure/renal disease and less obesity, higher anion gap/creatinine/BUN/RDW/PT/PTT/INR, lower Hb/RBC/chloride/MBP, higher scores and NLR/SII, and lower LMR. Different: chronic pulmonary disease, hypertension, glucose, WBC and PWR are not significant in our study.

## 5. LASSO feature selection (stage 4, Fig. 2)

| | Paper | This study (no-leakage) | This study (faithful) |
|---|---|---|---|
| λ.min (number of variables) | {LASSO_LAMBDA_MIN:.5f} (21) | {lasso.loc['noleak', 'lambda_min']:.5f} ({lasso.loc['noleak', 'n_min']}) | {lasso.loc['faithful', 'lambda_min']:.5f} ({lasso.loc['faithful', 'n_min']}) |
| λ.1se (number of variables) | {LASSO_LAMBDA_1SE:.5f} (12) | {lasso.loc['noleak', 'lambda_1se']:.5f} ({lasso.loc['noleak', 'n_1se']}) | {lasso.loc['faithful', 'lambda_1se']:.5f} ({lasso.loc['faithful', 'n_1se']}) |

- **Variables selected at λ.1se** (whole training set, used for the final model): {", ".join(sel)}
- **Overlap with the paper's 12 variables: {len(overlap)}/12**: {", ".join(overlap)}. Selected by the paper but not by us at λ.1se: {lasso.loc['noleak', 'paper_not_selected']}; **at λ.min the overlap with the paper is {len(overlap_min)}/12** (only pwr missing).
- Selected by us but not by the paper: lymphocytes_mean, sapsii, oasis, nlr, sii. If the paper's neutrophil/lymphocyte variables were affected by mixed units (Section 3.2), the competition among inflammation variables and their collinearity with the scores would change.
- **Selection frequency in the full nested CV** (LASSO run once in each of the 50 outer folds, `replication/nested_cv_full_selection_C.csv`): {nselC[1]}–{nselC[2]} variables per fold (median {nselC[0]}); selected in 50/50 folds: {", ".join(sel50C)}; in ≥ 40/50 folds: {", ".join(sel40C)}.
- Standardised coefficients at λ.1se (sorted by absolute value):

{md(lcoef, 3)}

![](figures/fig2a_lasso_cv_noleak.png)
![](figures/fig2b_lasso_path_noleak.png)

## 6. Model performance

### 6.1 Main estimate: full nested cross-validation (D8.5, D8.6, D8.11)

**Design**: training set of 1,568 patients, outer stratified 10 folds × 5 repeats (seeds 42–46). Inside every outer training fold, the 20% missing-rate rule, imputation, standardisation, LASSO (inner 10 folds, λ.1se) and tuning (inner stratified 5-fold `GridSearchCV`, AUROC) are done in turn, and the models are evaluated on the outer validation fold. All 9 models were run for all 5 repeats; no fold had LASSO selecting 0 variables. Metrics: mean ± SD of AUROC over the 50 outer folds; in each repeat the 10 validation folds are pooled into OOF predictions covering the whole training set, from which AUPRC, Brier, calibration slope etc. are computed and averaged over the 5 repeats. Models are compared with fold-wise paired differences and Nadeau–Bengio-corrected 95% CIs (r = 5, k = 10). Outputs in `replication/nested_cv_full_*_C.csv`.

{md(ncvC_tab_en)}

The AUPRC baseline (prevalence) is {uC.prevalence_pooled_mean:.3f}. RF (R defaults) has {int(sC.loc['RF (R defaults) (paper)', 'n_nonconv_cal_intercept_folds'])} outer folds with a non-converged calibration intercept; its calibration is reported from the pooled OOF predictions only.

- **{len(eqC)} models are equivalent**: outer AUROC {eqC_lo:.3f}–{eqC_hi:.3f}, each within {t88C.loc[[m for m in eqC if m != mstarC], 'delta'].max():.3f} of the best model.
- **Not equivalent: the paper-parameter RF (both versions) and SVM** ({neqC_lo:.3f}–{neqC_hi:.3f}): RF (literal parameters) Δ = {t88C.loc['RF (paper)', 'delta']:.3f} with a CI excluding 0; RF (R defaults) and SVM (paper parameters) Δ ≥ {min(t88C.loc['RF (R defaults) (paper)', 'delta'], t88C.loc['SVM (paper)', 'delta']):.3f}. Among the paper-parameter models only XGBoost matches the best model.
- Compared with the "LASSO-only nested" CV of stage 7 (Appendix A.5), full nesting gives {n_rec.cv_auc_nested:.3f} → {uC.auroc_fold_mean:.3f} for the unpenalised LR and {nest_tuned_lo:.3f}–{nest_tuned_hi:.3f} → {sC.loc[['RF (tuned)', 'XGBoost (tuned)', 'SVM (tuned)', 'LR (L2)']].auroc_fold_mean.min():.3f}–{sC.loc[['RF (tuned)', 'XGBoost (tuned)', 'SVM (tuned)', 'LR (L2)']].auroc_fold_mean.max():.3f} for the tuned models; the conclusions do not change. Stage 7 had a single repeat (seed 42), whereas the full nested CV averages 5 repeats, so the difference comes mainly from variation between repeats: in the seed-42 repeat of the full nested CV the values are {u42C:.3f} for the unpenalised LR and {r42C:.3f} for the tuned RF, and the unpenalised LR ranges over {u_rep_lo:.3f}–{u_rep_hi:.3f} across the 5 repeats.
- The L2 LR's fold-mean AUROC ({l2C.auroc_fold_mean:.3f}) is higher than its pooled OOF AUROC ({l2C.auroc_pooled_mean:.3f}): the inner loop chose a very small C, and strong shrinkage puts each fold's predictions on a different scale, which harms the ranking once they are pooled.

### 6.2 Model selection (pre-specified rule D8.8)

**Rule** (written before any stage-8 analysis was run): (1) the model with the highest mean outer AUROC is M\\*; (2) a model is equivalent to M\\* if its difference is ≤ 0.01, or if the Nadeau–Bengio 95% CI crosses 0 and the difference is ≤ 0.02; (3) if the equivalent set contains an LR, choose the LR; (4) if both LRs are equivalent, choose the one whose pooled-OOF calibration slope has the smaller |ln slope| (|ln slope| taken in each repeat, then averaged over the 5 repeats), the unpenalised LR on a tie; (5) the held-out set takes no part in the choice. **Steps 2 and 3 are a pre-specified preference for parsimony**: they allow the selected LR's AUROC to be up to 0.02 below the best model (when the CI crosses 0) or 0.01 (regardless of the CI).

**Execution**:
1. M\\* = {mstarC} (outer AUROC {sC.loc[mstarC, 'auroc_fold_mean']:.4f}).
2. Equivalent set = {", ".join(eqC)}.
3. The equivalent set contains an LR → choose an LR.
4. Both LRs are equivalent; calibration slope in each repeat (|ln slope| in brackets):

{md(slC_tab_en)}

   Mean |ln slope|: unpenalised LR {uC.abs_ln_cal_slope_pooled_mean_over_repeats:.3f}, L2 LR {l2C.abs_ln_cal_slope_pooled_mean_over_repeats:.3f} → **unpenalised LR**. The gap is small and the L2 LR is closer to 1 in {n_l2_closerC}/5 repeats; this step only breaks the tie by rule and is not evidence that the unpenalised LR is better calibrated. The two deviate from 1 by similar amounts in opposite directions (unpenalised LR slope {uC.cal_slope_pooled_mean:.2f}, predictions slightly too extreme; L2 LR {l2C.cal_slope_pooled_mean:.2f}, slightly towards the mean).

**Recommended model = unpenalised LR**. The reasons are only three: simplicity (an unpenalised logistic model on {int(hoCu.n_selected)} LASSO variables), interpretability (coefficients/ORs read directly, with 95% CIs and p values, no penalty strength to choose) and adequate EPV ({int(hoCu.events_train)} events / {int(hoCu.n_selected)} selected variables = {hoCu.epv_selected:.1f}). On calibration the two LRs are comparable. **No claim of better discrimination is made**: the unpenalised LR's outer AUROC ({uC.auroc_fold_mean:.3f}) is the lowest of the {len(eqC)} equivalent models, {t88C.loc['LR (unpenalised)', 'delta']:.3f} below M\\* (95% CI {t88C.loc['LR (unpenalised)', 'lo']:+.3f} to {t88C.loc['LR (unpenalised)', 'hi']:+.3f}).

**The pre-specified primary model (D4.7: tuned RF) is still reported**: outer AUROC {rC.auroc_fold_mean:.3f}, equivalent to M\\*; OOF calibration slope {rC.cal_slope_pooled_mean:.2f}, with predictions shrunk towards the mean.

**History of the model choice** (D4.15, recorded as it happened): the stage-4 pre-specified rule (D4.7, highest 10-fold CV AUC on the training set) selected the tuned RF (CV AUC {cv.loc['RF']:.3f}, tied with XGBoost; validation AUC {best_auc.auc:.3f}). In paired DeLong tests on the validation set, no pair of tuned models differed significantly after Holm correction; two changes were then made after seeing the validation results: first to the tuned L2 LR (C = 0.01), and then, because of its validation calibration slope of {calt.loc['LR', 'cal_slope']:.2f}, to the unpenalised LR. Neither change was in the analysis plan at the time. The stage-8 rule D8.8 also selects the unpenalised LR, but it was written after those results had been seen, so for definition C it is **not blind**; its role is to fix the basis of the choice as a pre-specified criterion on nested CV, not to prove that the post-hoc choice was right.

### 6.3 Secondary estimate: the single held-out validation set (D8.12)

**Note: this validation set (392 patients, {int(hoCu.events_valid)} deaths) was used in the post-hoc model choice of stages 6–7, so this is a secondary estimate only.** Both models are the stage-4 versions fitted on the whole training set (no-leakage preprocessing + the {int(hoCu.n_selected)} variables selected at λ.1se); each threshold is the model's own nested-CV pooled-OOF Youden threshold (median of 5 repeats). Outputs `replication/holdout_C.csv`, `holdout_dca_C.csv`.

{md(hoC_tab_en)}

- The held-out AUROCs of both models differ from the nested-CV estimates by ≤ {ho_gapC:.3f} (LR {hoCu.auroc:.3f} vs {uC.auroc_fold_mean:.3f}, RF {hoCr.auroc:.3f} vs {rC.auroc_fold_mean:.3f}), with no sign of additional overfitting.
- The tuned RF's held-out calibration slope is {hoCr.cal_slope:.2f}, in the same direction as its nested-CV OOF slope of {rC.cal_slope_pooled_mean:.2f} (shrinkage towards the mean); the unpenalised LR's is {hoCu.cal_slope:.2f}.
- With only {int(hoCu.events_valid)} deaths, the two AUROC CIs overlap heavily, and the two models are not ranked on this basis.

### 6.4 Thresholds

The main threshold is now **the Youden-optimal threshold of the nested-CV pooled OOF predictions (median of 5 repeats)**: unpenalised LR {thrC.loc['LR (unpenalised)', 'median']:.3f} (range {thrC.loc['LR (unpenalised)', 'min']:.3f}–{thrC.loc['LR (unpenalised)', 'max']:.3f}), tuned RF {thrC.loc['RF (tuned)', 'median']:.3f}. The stage-4 value of {lr_thr.loc['youden_oof', 'threshold']:.3f} from non-nested 10-fold OOF predictions (D4.13) and the results at 0.5 are kept in Appendix A.3. A threshold of 0.5 is unsuitable here: 1-year mortality is about 25%, and at 0.5 the recommended LR flags only {lr_thr.loc['0.5', 'flagged_pct']:.1%} of patients as high risk, with recall {lr_thr.loc['0.5', 'recall']:.2f}; it also assumes that missing a death costs the same as wrongly flagging a survivor. The Youden index weights sensitivity and specificity equally, which is itself an assumption; in practice the threshold should be chosen with DCA and the intended use. The RF classification metrics reported in the paper are internally consistent only if survival is the positive class (D4.6), so they are not comparable with ours.

### 6.5 Stage 4–7 single-validation-set results (secondary, history)

Stages 4–7 compared all models on the validation set of a single 80/20 split. The detailed tables have moved to Appendix A; only the key points are kept here:
- **Discrimination**: all validation AUCs lie between 0.76 and 0.80, with 95% CIs of about ±0.05; no pair of tuned models differs significantly in paired DeLong tests after Holm correction (smallest adjusted p = {dlt.p_holm.min():.2f}). ΔAUC between the unpenalised LR and the tuned RF = {rf_lr['diff']:+.3f} ({rf_lr.diff_lo:+.3f} to {rf_lr.diff_hi:+.3f}), p = {rf_lr.p_value:.2f} (Appendix A.1, A.2).
- **Calibration** (Appendix A.4): slope {cal_rec.cal_slope:.2f} for the unpenalised LR, {calt.loc['LR', 'cal_slope']:.2f} for the L2 LR, {calt.loc['RF', 'cal_slope']:.2f} for the tuned RF, 0.17 for RF (R defaults).
- **Overfitting and the paper's RF training AUC of 0.999**:

{md(ovf_tab)}

Rule: training − validation AUC > 0.10, or training AUC > 0.95 (D4.12).
  - **RF (R defaults: 9 fully grown trees, mtry=2)**: training AUC **0.999**, validation 0.756, a gap of 0.243, **similar to the paper's RF pattern of 0.999 / 0.780**. This is consistent with the explanation that "the paper's RF is `randomForest(ntree=9, mtry=2)` with fully grown trees (nodesize=1), and the listed `max_depth=10, min_samples_split=30, min_samples_leaf=10` had no effect": with the literal parameters our training AUC is only 0.887. Other causes cannot be ruled out, for example the paper computing training AUC on different data or using other unreported settings. Fully grown trees with only 9 trees memorise every training sample; the training AUC carries no information, and the probabilities are severely overconfident (calibration slope 0.17). In the full nested CV, the outer AUROC of RF (R defaults) is only {sC.loc['RF (R defaults) (paper)', 'auroc_fold_mean']:.3f}.
  - If the paper used the training AUC of 0.999 as one reason for RF being best, that would be inappropriate; cross-validated or independent validation performance should be compared instead.

## 7. Clinical utility: DCA and CIC

- **Main (nested-CV pooled OOF, `replication/nested_cv_full_dca_C.csv`)**: over thresholds 0.05–0.60, the number of thresholds at which the net benefit (mean of 5 repeats) exceeds both treat-all and 0: unpenalised LR {dcaCu[0]}/{dcaCu[1]} ({dcaCu[2]:.2f}–{dcaCu[3]:.2f}), tuned RF {dcaCr[0]}/{dcaCr[1]} ({dcaCr[2]:.2f}–{dcaCr[3]:.2f}).
- **Secondary (held-out validation set)**: see the table in Section 6.3; unpenalised LR {dca_better(hodC[hodC.model == 'LR (unpenalised)'])[0]}/56, tuned RF {dca_better(hodC[hodC.model == 'RF (tuned)'])[0]}/56.
- The stage-4 DCA/CIC figures on the validation set are kept below (paper: 0.1–0.7): the recommended LR's useful range is about {useful_range_m(UNPEN)}; around its Youden threshold, about {cic_n:.0f} per 1,000 patients are classified as high risk, of whom about {cic_e:.0f} actually die. Over thresholds 0.1–0.4 the models' net benefit differs by at most {dca_spread:.3f}.

![](figures/fig5_dca_cic_primary_noleak_recommended.png)
![](figures/fig5b_dca_models_noleak.png)

## 8. Model interpretation (Fig. 6, Fig. 8)

### 8.1 Recommended model: coefficients and SHAP of the unpenalised LR

The recommended unpenalised LR is fitted on the standardised values of the 12 LASSO variables (whole training set), so each coefficient is the log odds ratio per 1 standard deviation (SD) increase, with Wald 95% CIs and p values (`tables/lr_coefficients.csv`). The last column gives the tuned L2 LR's OR for comparison.

{md(lr_tab_en)}

- **Independent, significant risk factors** (p < 0.05): RDW, CCI, age, APS-III, OASIS. **Protective factors**: higher chloride_max (i.e. low chloride indicates higher risk, the same direction as the paper) and higher absolute lymphocytes.
- **Interpreting CCI** (D8.2): the mimic-code CCI includes an age score of 0–4 points, while age enters the model separately. The OR of CCI is a conditional effect at fixed age, reflecting mainly comorbidity burden and the part of the age bands not captured by linear age; it cannot be read as a pure comorbidity effect.
- **Collinearity**: SAPS-II, SII and NLR are not significant in the multivariable model, with VIF of about 3.7–4.3, suggesting their information is already covered by other variables. Individual coefficients should not be read as causal effects.
- **SHAP (unpenalised LR, LinearExplainer, validation set, log-odds scale)**: top 5 are {", ".join(top5_lr)}. The L2 LR's top 5 are {", ".join(top5_lr_l2)}.

![](figures/fig8_lr_odds_ratios_noleak.png)
![](figures/fig6a_shap_bar_primary_noleak_recommended.png)
![](figures/fig6b_shap_beeswarm_primary_noleak_recommended.png)

The force plots (one representative case each for a survivor, an in-hospital death and a death within 1 year after discharge, D4.9) show individual patients' feature values and are kept on the local machine only (D5.1).

### 8.2 SHAP compared with the paper

{md(shap_tab_en)}

- **Consistent with the paper**: RDW_mean is the **top** predictor in every analysis, and CCI is always in the top 3. chloride_max ranks 5th in the RF (low chloride → higher risk, the same direction as the paper) and is a significant protective factor in the LR.
- **Agreement between LR and RF**: the recommended LR and the tuned RF share {len(common_top5)} of their top 5 ({", ".join(common_top5)}).
- **Disagreement**: neutrophils_mean is not in our models, whereas it ranks 2nd in the paper. We suspect this may be related to mixed units (Section 3.2), but cannot verify it directly.

![](figures/fig6a_shap_bar_secondary_noleak_tuned.png)
![](figures/fig6b_shap_beeswarm_secondary_noleak_tuned.png)

## 9. Sensitivity analyses

### 9.1 Comorbidity timing (D8.2, D8.12)

Under the paper's conditions, comorbidities come from the index admission's `diagnoses_icd`; diagnosis codes have no timestamps and may include acute events of the current admission. This sensitivity analysis recomputes CCI (keeping the age score) and the SAPS-II chronic item from "all codes of prior admissions ∪ codes of the index admission after removing the D8.3 acute codes" (re-using the mimic-code SQL directly; re-running with index-admission codes only reproduces the original concept tables patient by patient), replaces only these two columns, and re-runs the full nested CV for the recommended LR and the tuned RF (same outer splits as the main analysis).

- {int(ctS['patients with >= 1 prior admission'])} patients ({int(ctS['patients with >= 1 prior admission']) / n_final:.0%}) have a prior admission; {int(ctS['patients with >= 1 D8.3 acute code in the index admission'])} have a D8.3 acute code in the index admission. The median (IQR) CCI is {ctS['CCI original, median (IQR)']} both before and after recomputation; CCI changes for {int(ctS['CCI changed (n)'])} patients ({int(ctS['CCI increased (n)'])} up, {int(ctS['CCI decreased (n)'])} down), and SAPS-II for {int(ctS['SAPS-II changed (n)'])}.

{md(ct_tab_en)}

- **The largest change is myocardial infarction**: of {int(mi.original_n)} original positives, {int(mi.lost_by_acute_exclusion)} (about half) have only acute index-admission codes (410/I21/I22) and no longer count as prior MI. No index admission in definition C carries a 586/N19 code, so the renal exclusion rule has no effect.
- `diabetes_with_cc` is not 100%: <10 patients have a 249.6x DN code (secondary diabetes with neuropathy), which is not in the charlson diabetes code list (correcting the original wording of D8.2).

{md(ctp_tab_en)}

- **Conclusion**: although CCI changes for {int(ctS['CCI changed (n)'])} patients and about half of the MI positives come from acute codes, discrimination, AUPRC, Brier and the LR's calibration are essentially unchanged for both models (ΔAUROC CIs cross 0), and CCI is still selected in every fold. The conclusion that "CCI is a stable, important predictor" does not depend on when the index-admission codes arose.

### 9.2 Excluding the 2020–2022 group (D4.19)

The {int(sL.n_excluded)} patients with `anchor_year_group` "2020 - 2022" are excluded and the modelling pipeline is re-run on the rest (`scripts/04e_sensitivity_pre2020.py`, with a new single stratified 80/20 split).

{md(sens_tab_en)}

- **LASSO**: {int(sL.n_selected)} variables selected; adds {", ".join(sens_only) or "none"} and drops {", ".join(main_only) or "none"}.
- **Conclusion**: {sens_verdict_en} The validation set here is a new single split, so AUC differences also include variation from the random split. `anchor_year_group` is a patient-level 3-year anchor band, not the actual year of each admission, so this only approximates "excluding admissions after 2020".

# Part II: Clinical model (CLIN)

## 10. Design, cohort and features (D8.1–D8.4, D8.13)

**Prediction time = 24 h after ICU admission** (landmark), using only information available at that time:
- **Inclusion**: the same index admission as definition C (the first admission with both a DN code and an ICU stay) and its first ICU stay; age ≥ 18; still in the ICU and alive 24 h after ICU admission. Patients whose first ICU stay is shorter than 24 h are excluded outright; a later ICU stay of the same admission is not used instead. LOS > 48 h is **not used** (it depends on information after the prediction time), and a complete CBC is **not required**.
- **Features**: labs, vital signs and scores use the same (first-day) windows as the replication track. The 6 comorbidity variables come from "all codes of prior admissions ∪ codes of the index admission after removing acute-event codes" (D8.3: acute MI, acute heart failure, 586/N19, acute stroke/TIA, aortic dissection, acute peptic ulcer, acute hepatic necrosis/shock liver, etc. are removed); CCI excludes the age score (to avoid counting age twice); the SAPS-II chronic item is recomputed over "prior ∪ index admissions".
- **White-cell differential** (D8.4 method 2): the 8 white-cell-differential variables (the three cell counts and NLR, PLR, LMR, PNR, SII) are exempt from the 20% rule, and one indicator `wbc_diff_missing` (all three cell counts missing) is added as a LASSO candidate; missing values are mean-imputed inside the folds.

{md(flK_tab_en)}

- **Split**: stratified 80/20 (seed 42): training {int(spK.loc['train', 'n']):,} patients ({int(spK.loc['train', 'deaths'])} deaths, {spK.loc['train', 'mortality']:.1%}), held-out {int(spK.loc['valid', 'n'])} ({int(spK.loc['valid', 'deaths'])} deaths, {spK.loc['valid', 'mortality']:.1%}).
- **Comorbidities**: {int(cmK['has_prior_admission = 1 (n)']):,} patients ({float(cmK['has_prior_admission = 1 (%)']):.1f}%) have a prior admission; the median (IQR) CCI without the age score is {cmK['CCI (no age score), index admission only, median (IQR)']} from index-admission codes only and {cmK['CCI (no age score), CLIN rule, median (IQR)']} under the CLIN rule ({int(cmK['CCI changed vs index-only (n)']):,} patients change). The D8.3 count for 586/N19 is 0 (< 1%), so as pre-specified the add-back sensitivity analysis is not run.
- **White-cell differential**: `wbc_diff_missing = 1` for {n_wbc_missing} patients ({n_wbc_missing / int(cmK['CLIN stays']):.1%}); {partial_wbc} patients have partial missingness, all because a ratio's denominator was 0 (D2.10). Only these 8 white-cell-differential variables exceed 20% missingness (`clinical/missingness_clin.csv`); the highest among the rest is {max_other_missK.feature} ({max_other_missK.missing_pct:.1f}%).

**Table 1 (whole CLIN cohort by 1-year outcome, observed values before imputation)** (`clinical/table1_clin.csv`; rules as in D3.5, D3.6; {n_surv_K:,} survivors, {n_death_K} deaths; {n_sigK} variables with p < 0.05):

{md(t1K_tab_en)}

## 11. Nested cross-validation and model selection (D8.14)

Training set of {int(spK.loc['train', 'n']):,} patients, same design as Section 6.1 (5 × 10 folds, r = 5, k = 10). {int(recK['n_candidates'])} candidate variables (55 + `wbc_diff_missing`); the 20% rule dropped no variable in any fold; LASSO selected {nselK[1]}–{nselK[2]} variables per fold (median {nselK[0]}). Outputs in `clinical/nested_cv_full_*_clin.csv`.

{md(ncvK_tab_en)}

The AUPRC baseline (prevalence) is {l2K.prevalence_pooled_mean:.3f}.

**Applying D8.8**:
1. M\\* = **LR (L2)** (outer AUROC {l2K.auroc_fold_mean:.4f}).
2. Equivalent set = {", ".join(eqK)} (each within {eqK_maxd:.4f} of M\\*). RF (literal parameters) Δ = {t88K.loc['RF (paper)', 'delta']:.4f} > 0.02, not equivalent; SVM (paper parameters) and RF (R defaults) are further behind.
3. The equivalent set contains an LR → choose an LR.
4. Both LRs are equivalent; calibration slope in each repeat (|ln slope| in brackets):

{md(slK_tab_en)}

   Mean |ln slope|: L2 LR {l2K.abs_ln_cal_slope_pooled_mean_over_repeats:.4f}, unpenalised LR {uK.abs_ln_cal_slope_pooled_mean_over_repeats:.4f} → **L2 LR**.

**The rule decided a near tie**: the two LRs' AUROCs differ by {t88K.loc['LR (unpenalised)', 'delta']:.4f} (CI crossing 0), their mean |ln slope| by only {uK.abs_ln_cal_slope_pooled_mean_over_repeats - l2K.abs_ln_cal_slope_pooled_mean_over_repeats:.3f}, and the unpenalised LR is closer to 1 in {n_unpen_closerK}/5 repeats. This step only breaks the tie by the pre-specified rule and is not evidence that the L2 LR is better calibrated; the two deviate from 1 by similar amounts in opposite directions (L2 LR {l2K.cal_slope_pooled_mean:.2f}, unpenalised LR {uK.cal_slope_pooled_mean:.2f}). The CLIN choice differs from definition C (unpenalised LR) only because of this step.

- **No claim of better discrimination**: the L2 LR happens to be M\\*, but it is within {eqK_maxd:.3f} of every other equivalent model with CIs crossing 0. The stated reasons are only simplicity, interpretability and "the two LRs are comparably calibrated". Steps 2 and 3 are a pre-specified preference for parsimony.
- Inner tuning of the L2 LR chose C = 0.01 in 50/50 folds (grid 0.001–100 and ∞; 0.01 is not on the boundary), i.e. fairly strong shrinkage.
- **Tuned RF**: outer AUROC {rK.auroc_fold_mean:.3f}, equivalent to M\\*; pooled-OOF calibration slope {rK.cal_slope_pooled_mean:.2f}, with predictions shrunk towards the mean, in the same direction as in definition C.
- **LASSO selection frequency** (folds out of 50), top 15: {sel15_str_en}; `wbc_diff_missing` {selK['wbc_diff_missing']}/50. Of the 6 comorbidities only obesity is selected ({selK['obesity']}/50); MI, heart failure, chronic pulmonary disease, renal disease and hypertension are all 0/50.
- **Thresholds** (nested-CV pooled-OOF Youden, median of 5 repeats): L2 LR {thrK.loc['LR (L2)', 'median']:.4f} ({thrK.loc['LR (L2)', 'min']:.3f}–{thrK.loc['LR (L2)', 'max']:.3f}), tuned RF {thrK.loc['RF (tuned)', 'median']:.4f} ({thrK.loc['RF (tuned)', 'min']:.3f}–{thrK.loc['RF (tuned)', 'max']:.3f}).

![](figures/figC4_calibration_clin_nested_oof.png)

## 12. Freezing and the confirmatory held-out evaluation (D8.14, D8.15, D8.17)

### 12.1 Freezing procedure

- The **frozen specification** `clinical/frozen_model_clin.json` was written and committed before the held-out set was read: on the training set, the 20% rule (white-cell differential exempt) → imputation → standardisation → LASSO (inner 10 folds, seed 42, λ.1se) → LR (L2) with C chosen by inner stratified 5-fold `GridSearchCV`; the tuned RF comparator follows the same procedure; thresholds are the nested-CV pooled-OOF Youden thresholds (LR {specK['thresholds']['LR (L2)']}, RF {specK['thresholds']['RF (tuned)']}); the training stay_id hash, candidate-matrix hash, code hashes and git commit are recorded.
- `scripts/04h_final_clin.py` fits the model as specified; the fit record `clinical/final_model_clin.json` includes the model file's sha256.
- Before reading the held-out set, `scripts/04i_holdout_clin.py` checks three guards: the frozen files are committed and unchanged; the model file matches the frozen specification; no output file exists (no overwriting). A smoke test on a training subsample (not reading the held-out set) was run first. The confirmatory evaluation was **run exactly once**.
- The two sensitivity analyses (Section 14) were run after the held-out evaluation; they concern only the selected model and the tuned RF, take no part in model selection and do not change the frozen model.
- For the figures (Fig. C1–C3), `scripts/04j_plots_clin.py` predicts the held-out set again with the frozen model file (nothing is refitted); every recomputed metric is identical to `holdout_clin.csv` and `holdout_dca_clin.csv` (difference < 1e-9), and no conclusion changed (D8.17).

### 12.2 Final model

- Training set of {int(recK['n_train']):,} patients with {int(recK['events'])} events; {int(recK['n_candidates'])} candidate variables, {int(recK['n_dropped_missing'])} dropped by the 20% rule.
- LASSO λ.1se = {recK['lambda_1se']:.4f}, selecting **{int(recK['n_selected'])}** variables (at the upper end of the {nselK[1]}–{nselK[2]} seen across nested-CV folds): {", ".join(recK['selected'])}.
- LR (L2): C = {recK['params']['LR (L2)']['C']}; tuned RF: {", ".join(f"{k} {v}" for k, v in recK['params']['RF (tuned)'].items())}.
- **EPV = {recK['epv_selected']:.1f}** (events / selected variables); events / candidate variables = {recK['events_per_candidate']:.1f}.

### 12.3 Held-out results (confirmatory estimate)

Held-out set of {int(hoKl.n_eval)} patients with {int(hoKl.events_eval)} deaths. Outputs `clinical/holdout_clin.csv`, `holdout_dca_clin.csv`.

{md(hoK_tab_en)}

![](figures/figC1_roc_clin_holdout.png)
![](figures/figC2_calibration_clin_holdout.png)
![](figures/figC3_dca_clin_holdout.png)

### 12.4 Held-out AUROC below the nested-CV estimate

- **The confirmatory estimate for CLIN is the held-out AUROC of {hoKl.auroc:.3f} ({hoKl.auroc_lo:.3f}–{hoKl.auroc_hi:.3f})**, about {l2K.auroc_fold_mean - hoKl.auroc:.2f} below the nested-CV estimate: LR (L2) {l2K.auroc_fold_mean:.3f} → {hoKl.auroc:.3f}, tuned RF {rK.auroc_fold_mean:.3f} → {hoKr.auroc:.3f}. **Both models drop in parallel**, and the nested-CV estimate lies above the upper limit of the held-out 95% CI.
- The following checks are descriptive analyses done after seeing the result; they change neither the frozen model nor the confirmatory result (`clinical/holdout_gap_diagnostics_clin.csv`):
  - **Overfitting looks unlikely**: the frozen LR's apparent AUROC on the training set is {gd['apparent_auroc_train_lr_l2']:.3f}, only {gd['apparent_auroc_train_lr_l2'] - l2K.auroc_fold_mean:.3f} above nested CV; the L2 shrinkage is strong (C = 0.01), and nested CV already includes variable selection and tuning.
  - **Size of sampling variation**: resampling the training set's nested-CV pooled OOF predictions with replacement to the held-out composition ({int(hoKl.events_eval)} deaths / {int(hoKl.n_eval - hoKl.events_eval)} survivors) {int(gd['oof_subsample_n_draws']):,} times gives a mean AUROC of {gd['oof_subsample_auroc_mean']:.3f}, SD {gd['oof_subsample_auroc_sd']:.3f}, 2.5th percentile {gd['oof_subsample_auroc_p2.5']:.3f}; {gd['oof_subsample_p_le_holdout_auroc']:.1%} of resamples are ≤ {hoKl.auroc:.3f}. This simulation omits the variation from refitting the model and so understates the total variation.
  - **Distribution differences**: across the {int(gd['smd_n_features'])} selected variables, the largest standardised mean difference between training and held-out sets is {gd['smd_max_abs']:.2f} ({gd['smd_max_abs_feature']}), with {int(gd['smd_n_abs_gt_0.1'])} above 0.1 ({gd['smd_features_abs_gt_0.1']}); the split is stratified random, with no time difference.
- **Interpretation**: the gap more likely reflects sampling variation in a 512-patient held-out set (the result lies in the lower tail of the resampling distribution) than overfitting, but other causes cannot be excluded. As pre-specified, **the model is not changed and the evaluation is not repeated**. The report gives both the nested-CV estimate {l2K.auroc_fold_mean:.3f} and the held-out estimate {hoKl.auroc:.3f}; taken together, CLIN's discrimination lies between 0.75 and 0.80.
- LR (L2) and the tuned RF do not differ on the held-out set (p = {hoKl.delong_lr_minus_rf_p:.2f}), and their AUPRC and Brier are similar.

### 12.5 Calibration: recalibration needed before deployment

- On the held-out set the L2 LR's calibration slope is **{hoKl.cal_slope:.2f}** (intercept {hoKl.cal_intercept:+.3f}), predictions slightly too extreme; its nested-CV OOF slope is {l2K.cal_slope_pooled_mean:.2f} (slightly towards the mean), so **the deviation points in opposite directions in the two estimates**. The tuned RF's held-out slope is {hoKr.cal_slope:.2f} (nested CV {rK.cal_slope_pooled_mean:.2f}).
- Calibration is unstable across samples, so the model **must be recalibrated in the target population before deployment** (at least re-estimating intercept and slope), and the threshold should be re-set for the intended use after recalibration.

## 13. Model interpretation (`clinical/coefs_clin.csv`, `clinical/shap_clin.csv`)

The L2 LR's penalised ORs (per SD) describe the frozen model itself; inference uses the unpenalised LR on the same {int(recK['n_selected'])} variables (statsmodels GLM, training set, as in D4.16), with Wald 95% CIs, p values and VIF.

{md(coK_tab_en)}

- **Variables with p < 0.05 in the unpenalised model** (OR, 95% CI; per SD for continuous variables, yes vs no for binary variables): {sig_glmK_str_en}.
- **Collinearity**: the largest VIFs are {vif_str_en}, all < 5; SAPS-II, LODS, NLR, SII etc. are not significant with the other scores/ratios held fixed.
- **SHAP** (LR (L2), held-out set, LinearExplainer, log-odds scale, training set as background), top 10: {", ".join(shap10K)}. **The top 3 are the same as for the recommended model in definition C** ({", ".join(shap10K[:3])}).
- obesity has OR < 1 (yes vs no {ob_glm.or_bin:.2f}, {ob_glm.or_bin_lo:.2f}–{ob_glm.or_bin_hi:.2f}), as does chloride_max, in the same direction as definition C's Table 1; for the interpretation of obesity see Section 17.3.

## 14. Sensitivity analyses (D8.3, D8.4 revisions, D8.16)

Both analyses run the full 5 × 10 nested CV for the L2 LR and the tuned RF only, on the same outer splits as the main analysis; ΔAUROC is the paired Nadeau–Bengio 95% CI.

{md(sensK_tab_en)}

- **prior_only** (the 6 comorbidities, CCI without age score and the SAPS-II chronic item from prior-admission diagnoses only; comorbidities set to 0 for patients without a prior admission; `has_prior_admission` as a candidate): discrimination falls slightly with CIs excluding 0 (LR (L2) {pP.mean_diff:+.3f}, RF {pPr.mean_diff:+.3f}). The largest change is obesity, whose selection falls from {selK['obesity']}/50 to {selP['obesity']}/50; cci {selK['cci']} → {selP['cci']}/50, myocardial_infarct {selK['myocardial_infarct']} → {selP['myocardial_infarct']}/50, `has_prior_admission` {selP['has_prior_admission']}/50. The main analysis uses chronic codes of the index admission and relies on the assumption that they represent conditions present before admission; prior_only does not rely on that assumption and is therefore **reported as the more conservative deployment estimate** (nested-CV AUROC about {sumP.loc['LR (L2)', 'auroc_fold_mean']:.3f}).
- **no_wbc_diff** (the 8 white-cell-differential variables go through the 20% rule inside the folds and are dropped in every fold, {n_drop_W} per fold; no `wbc_diff_missing`): discrimination, AUPRC, Brier and calibration are essentially unchanged (LR (L2) {pW.mean_diff:+.3f}, RF {pWr.mean_diff:+.3f}, CIs crossing 0), and the median number of selected variables goes from {nselK[0]} to {nselW[0]}. The white-cell differential adds no detectable discrimination in CLIN and can be omitted without loss. The frozen model and the main analysis are unchanged.

# Part III: Comparison with the paper, limitations and future work

## 15. Summary of differences from the paper and possible reasons

| Item | Paper | This study | Main reason |
|---|---|---|---|
| Data release | v2.x (2008–2019) | v3.1 (2008–2022) | Patients from 2020–2022 added |
| DN patients / final n | 4,137 / 1,313 | {int(flow_tab.iloc[0, 2]):,} / {n_final:,} (definition C) | Data release; retention at later steps is consistent |
| Mortality | 38.2% | {d_final / n_final:.1%} | The paper may have counted any non-missing dod (our cohort gives {o.loc['death_any_dod', 'rate']:.1%} under that definition); we use an explicit 1-year window |
| Hypertension prevalence | 37.3% | 84.6% | The paper may have counted only uncomplicated hypertension (35.4% in our cohort); that definition misclassifies hypertensive CKD/heart failure as "no hypertension" (Section 3.3) |
| Lymphocytes / neutrophils / monocytes | Implausibly wide distributions | Physiological range (K/uL) | The paper may have mixed units/scales |
| PLR / PNR | 36 / 6.6 | 159 / 22 | Same as above; NLR/LMR/SII/PWR agree with the paper, supporting the mixed-units explanation |
| LASSO λ.min / λ.1se | 0.0106 / 0.0269 | {lasso.loc['noleak', 'lambda_min']:.4f} / {lasso.loc['noleak', 'lambda_1se']:.4f} | Almost identical |
| LASSO λ.1se variables | 12 | 12, {len(overlap)} shared | Mixed-unit variables; collinearity among scores |
| Performance estimate | Single validation set | Full nested CV (main) + single held-out set (secondary) | Nested CV keeps variable selection and tuning inside the folds and gives a more stable estimate |
| Best model | RF (validation 0.780) | Definition C: {len(eqC)} equivalent models (nested CV {eqC_lo:.3f}–{eqC_hi:.3f}), unpenalised LR recommended by the pre-specified rule; CLIN: L2 LR (nested CV {l2K.auroc_fold_mean:.3f}, held-out {hoKl.auroc:.3f}) | The paper's RF training AUC of 0.999 is consistent with overfitting; the paper-parameter RF is not equivalent to the best model in nested CV |
| RF training AUC | 0.999 | 0.999 (R defaults) / 0.850 (tuned) | Consistent with the explanation that the paper's RF used fully grown trees |
| Classification metrics | Survival as positive class, threshold not stated | Death as positive class; nested-CV OOF Youden threshold | The paper's metrics may have used survival as the positive class (inferred) |
| SHAP top 5 | RDW, neutrophils, CCI, chloride_max, PT_min | Definition C LR: {", ".join(top5_lr)}; CLIN LR (L2): {", ".join(shap10K[:5])} | RDW first and CCI in the top 3 agree; neutrophils may be affected by mixed units |

**Agreement in trends and conclusions**: consistent with the paper, RDW, CCI, illness-severity scores, age and low chloride are the main predictors of 1-year death in DN patients, and discrimination is moderate (about 0.75–0.80). The disagreement is "RF is best": in full nested CV the tuned models of all types are equivalent, and the paper-parameter RF is in fact significantly worse.

## 16. Unclear points in the paper's methods and how we handled them

| Unclear point | Our handling | Decision |
|---|---|---|
| MIMIC-IV release not stated | v3.1, covering 2008–2022 | — |
| "first ICU admission at first hospitalization" | This study uses definition C (first admission with both a DN code and an ICU stay + its first ICU stay); headcount and mortality of 3 alternative definitions also listed | D1.2, D8.1 |
| Death outcome and follow-up not defined ("survival status") | 1-year all-cause death from ICU admission, an explicit window at date granularity | D1.6, D8.10 |
| First-day window not defined | mimic-code: labs [intime−6h, intime+24h], vital signs [intime, intime+24h] | D1.5, D2.3 |
| Items meant by "complete blood counts were not missing" | 8 items (WBC, Hb, PLT, RBC, RDW + three absolute white-cell counts) | D1.5 |
| Codes for hypertension and obesity not given | Quan-Elixhauser definitions; uncomplicated hypertension only as sensitivity analysis | D2.9, D4.10 |
| Units of the white-cell differential | mimic-code absolute counts (K/uL) | D2.2, D2.11 |
| Imputation and the 20% missing-rate rule applied to all data (leakage) | Faithful and no-leakage versions in parallel; in nested CV the 20% rule, imputation and standardisation are all done inside the folds | D3.4, D8.4 |
| Table 1 computed after imputation | Observed data; post-imputation version saved separately | D3.5 |
| LASSO criterion (text says MSE, figure shows binomial deviance) | Binomial deviance (glmnet default for binary outcomes) | D4.1 |
| RF parameters (R and sklearn names mixed) | Two RF versions: literal parameters / R defaults | D4.4 |
| Tuning grid not reported | Grid search, done in the inner loop of nested CV | D4.5, D8.5 |
| Positive class and threshold of the classification metrics | Death as positive class; nested-CV OOF Youden threshold | D4.6, D8.6 |
| Basis for choosing the best model | Pre-specified rule D8.8 + full nested CV (held-out set not used for the choice); the post-hoc choice of stages 4–7 recorded as it happened | D8.8, D4.15 |
| Data years (v2.x ends in 2019) | Main analysis uses all years; sensitivity analysis excludes the 2020–2022 group | D4.19 |

## 17. Limitations

### 17.1 Limitations shared by both tracks

1. **Single centre, retrospective**: MIMIC-IV comes from one hospital (BIDMC) and there is no external validation. The held-out set has only {int(hoCu.events_valid)} deaths in definition C and {int(hoKl.events_eval)} in CLIN, so single held-out estimates have 95% CIs of about ±0.05.
2. **Different data release**: v3.1 and the paper's v2.x cover different years, so item-by-item comparisons with the paper can only show trends.
3. **DN identified by ICD codes**: the sensitivity and specificity of the codes are unknown and coding practice changed over time; inclusion rests on index-admission codes under the assumption that DN is a chronic condition present before admission; the cohort contains only ICU patients.
4. **Diagnosis codes have no timestamps**: definition C uses all index-admission codes, as in the paper, including acute events of the current admission (about half of MI comes from acute codes, Section 9.1); CLIN removes explicit acute codes, but the chronic codes it keeps are equally untimed, and hemiplegia codes may come from an acute stroke during the current admission.
5. **Outcome follow-up**: `dod` covers up to 1 year after the last discharge, so all deaths within 365 days of ICU admission fall within follow-up (D8.10); longer-term outcomes cannot be studied and causes of death cannot be distinguished; the state linkage relies on name, date of birth and social security number, so deaths out of state may be missed.
6. **First-day features**: only first ICU-day data are used, without treatment information (e.g. vasopressors, mechanical ventilation) or time dynamics; some scores overlap and are collinear with the raw variables.

### 17.2 Replication track (definition C)

7. **LOS > 48 h depends on information after the prediction time** (D8.1): the condition is kept as in the paper, but length of stay is not known at the end of the first ICU day; it excludes patients who die or are discharged early and causes selection bias. CLIN does not use it.
8. **CBC-completeness criterion**: the paper's inclusion criteria require all 8 CBC items, which excludes about 36% of patients in definition C, mainly because no white-cell differential was done. CLIN keeps these patients: 1-year mortality is {wmK.loc[1, 'mortality']:.1%} ({int(wmK.loc[1, 'deaths'])}/{int(wmK.loc[1, 'n'])}) with `wbc_diff_missing = 1` and {wmK.loc[0, 'mortality']:.1%} ({int(wmK.loc[0, 'deaths'])}/{int(wmK.loc[0, 'n'])}) with `wbc_diff_missing = 0` (χ² p = {wmK.loc[0, 'chi2_p']:.2f}); in CLIN's unpenalised model (other 21 variables held fixed), the OR (yes vs no) of `wbc_diff_missing` is {wbc_glm.or_bin:.2f} ({wbc_glm.or_bin_lo:.2f}–{wbc_glm.or_bin_hi:.2f}). This **suggests** that whether a white-cell differential is ordered is related to prognosis, so a complete-case inclusion criterion may introduce selection bias; but the variable was selected in only {selK['wbc_diff_missing']}/50 nested-CV folds and the crude mortality difference is not significant, so the evidence is limited. Separately, the no_wbc_diff sensitivity analysis shows that the white-cell-differential variables themselves add no discrimination (Section 14).
9. **The choice of recommended model is not blind**: rule D8.8 was written after the stage 4–7 results of definition C had been seen, and stages 6–7 changed the choice twice post hoc (Section 6.2). The rule selects the same unpenalised LR as the post-hoc choice, but this does not prove that the post-hoc choice was right.
10. **Definition C's held-out set was used in the post-hoc choice** and serves only as a secondary estimate (Section 6.3).

### 17.3 Clinical model (CLIN)

11. **Held-out AUROC about {l2K.auroc_fold_mean - hoKl.auroc:.2f} below nested CV** (Section 12.4): the checks suggest sampling variation is more likely, but other causes cannot be excluded; a single 512-patient held-out set cannot estimate performance precisely.
12. **Unstable calibration**: held-out slope {hoKl.cal_slope:.2f}, opposite in direction to nested CV ({l2K.cal_slope_pooled_mean:.2f}); recalibration is needed before deployment (Section 12.5).
13. **Timing of the index-admission chronic codes**: using prior-admission diagnoses only lowers AUROC by {-pP.mean_diff:.3f} (Section 14). The main estimate relies on the assumption that the chronic codes were present at admission; the prior_only estimate (about {sumP.loc['LR (L2)', 'auroc_fold_mean']:.3f}) is more conservative.
14. **Obesity coding may carry an illness-related bias**: in CLIN, obesity has OR < 1 (unpenalised model, yes vs no {ob_glm.or_bin:.2f}, {ob_glm.or_bin_lo:.2f}–{ob_glm.or_bin_hi:.2f}). With prior-admission diagnoses only, obesity's selection falls from {selK['obesity']}/50 to {selP['obesity']}/50. Descriptively (`clinical/obesity_source_clin.csv`), 1-year mortality is {ob_ix.mortality:.1%} (n = {int(ob_ix.n)}) for patients whose obesity code appears only in the index admission, {ob_pr.mortality:.1%} (n = {int(ob_pr.n)}) for those with an obesity code in a prior admission, and {ob_no.mortality:.1%} (n = {int(ob_no.n)}) for those without an obesity code. The "protective" association is concentrated in obesity coded only during the index admission. This **suggests** that obesity coding may be influenced by illness-related documentation practice (for example, patients who are less severely ill and have a more stable stay may be more likely to have secondary diagnoses such as obesity recorded), rather than reflecting a protective effect of obesity itself (the "obesity paradox"); objective measures such as BMI are needed to verify it.

## 18. Future work

1. **Temporal extrapolation and external data**: within MIMIC-IV, temporal extrapolation validation by `anchor_year_group` (e.g. train on earlier year groups and validate on later ones, extending `04e`) to assess temporal drift; and a search for external data with one-year follow-up for external validation. eICU-CRD is not used: it has only in-hospital outcomes and cannot validate 1-year death.
2. **Recalibration and thresholds**: recalibrate the recommended models in the target population (especially the CLIN L2 LR, held-out slope {hoKl.cal_slope:.2f}; re-estimating intercept and slope), report DCA after recalibration, and set the threshold according to the specific clinical use (e.g. palliative-care discussions, closer follow-up).
3. **Obesity coding bias**: extract BMI from the `omr` table and compare coded obesity with BMI-defined obesity to test the suggestion in item 14 of Section 17.3.
4. **Features**: add HbA1c, treatment variables and longitudinal features; compare models without composite scores (SAPS-II etc.) to reduce collinearity and improve interpretability; assess the dose-response relation between RDW and death (restricted cubic splines) and its consistency across DN subtypes (type 1/type 2).

# Appendices

## Appendix A: detailed stage 4–7 results on the single validation set (secondary)

These are the stage 4–7 results on the validation set of a single 80/20 split (392 patients, 99 deaths). The main estimate is now the full nested CV (Section 6.1); these tables are kept as a record of the process.

### A.1 AUC (DeLong 95% CI) ({csv_link('model_performance.csv')}, {csv_link('nested_cv_auc.csv')})

{md(auc_tab_en)}

The `recommended` group is the unpenalised LR (the same model as the LR of the paper-parameter group); the `tuned` group holds the 10-fold-CV-tuned models (LR with an L2 penalty); the "Nested CV" column is the stage-7 "LASSO-only nested" CV (A.5). Tuned optimal parameters: `{json.dumps({k: v for k, v in tuned.items() if k.startswith('noleak')}, ensure_ascii=False)}`.

Relative ranking of AUC (validation set): paper {paper_rank}; this study, paper parameters {ours_rank_paper}; this study, tuned {ours_rank_tuned}.

![](figures/fig3_roc_noleak_main.png)

### A.2 Pairwise model comparison (paired DeLong test, Holm correction) ({csv_link('delong_pairwise.csv')})

{md(dl_tab_en)}

### A.3 Thresholds and classification metrics ({csv_link('youden_thresholds.csv')}, {csv_link('threshold_metrics.csv')})

Thresholds (stage 4, Youden thresholds of non-nested 10-fold OOF predictions, D4.13):

{md(ydn_tab)}

Validation metrics at the Youden threshold:

{md(thry_tab)}

Validation metrics at threshold 0.5 (comparison):

{md(thr05_tab)}

![](figures/fig4_metrics_noleak_tuned.png)

### A.4 Calibration ({csv_link('calibration.csv')}, {csv_link('lr_variants.csv')})

{md(cal_tab)}

{md(lrv_tab_en)}

![](figures/fig7_calibration_noleak_main.png)

### A.5 "LASSO-only nested" CV (stage 7, D4.18) ({csv_link('nested_cv_auc.csv')}, {csv_link('nested_cv_folds.csv')})

The outer loop uses the same 10 folds as stage 4 (seed 42); imputation, standardisation and LASSO are redone inside each fold, with hyperparameters fixed at the stage-4 values, isolating only the variable-selection bias. Superseded by the full nested CV (Section 6.1).

{md(nest_tab_en)}

## Appendix B: file list

- **Definition C cohort and features**: `tables/cohort_flow.csv`, `outcome_definitions.csv`, `outcome_check.csv`, `cbc_missingness.csv`, `dn_codes.csv`, `itemid_audit_*.csv`, `feature_vs_paper.csv`, `unit_mixing_diagnostic.csv`, `hypertension_definitions.csv`, `table1.csv`, `table1_after_imputation.csv`, `missingness.csv`
- **Definition C, stages 4–7**: `tables/lasso_*.csv`, `model_performance.csv`, `tuned_hyperparameters.json`, `calibration.csv`, `overfitting.csv`, `youden_thresholds.csv`, `threshold_metrics.csv`, `delong_pairwise.csv`, `lr_variants.csv`, `lr_coefficients.csv`, `shap_importance*.csv`, `dca_*.csv`, `cic_*.csv`, `nested_cv_*.csv`, `sensitivity_pre2020.csv`
- **Definition C, stage 8** (`replication/`): `nested_cv_full_*_C.csv` (full nested CV), `holdout_C.csv`, `holdout_dca_C.csv` (secondary held-out estimate), `comorb_time_*_C.csv`, `nested_cv_full_*_C_comorb_time.csv` (comorbidity-timing sensitivity analysis)
- **CLIN** (`clinical/`): `cohort_flow_clin.csv`, `split_clin_summary.csv`, `comorb_clin_*.csv`, `wbc_diff_patterns_clin.csv`, `missingness_clin.csv`, `table1_clin.csv`, `wbc_diff_missing_mortality_clin.csv`, `obesity_source_clin.csv`, `nested_cv_full_*_clin.csv`, `frozen_model_clin.json`, `final_model_clin.json`, `holdout_clin.csv`, `holdout_dca_clin.csv`, `holdout_gap_diagnostics_clin.csv`, `coefs_clin.csv`, `shap_clin.csv`, `nested_cv_full_*_clin_{{prior_only,no_wbc_diff}}.csv`
- **Figures**: `figures/fig2a–fig8_*.png` (definition C; names with `primary` use the unpenalised LR as the main model, names with `secondary` the tuned RF); `figures/figC1–C4_*.png` (CLIN). Single-patient force plots `fig6c–e` are kept on the local machine only
