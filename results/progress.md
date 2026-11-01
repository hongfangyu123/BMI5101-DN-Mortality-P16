# Progress log (progress.md)

## Stage 0: project setup (2026-09-30)
- Directories, conda environment `dn_repro` (Python 3.11.16; duckdb 1.5.6, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.2, shap 0.51.0), `environment.yml`, `config/config.yaml`, `.gitignore`, `CLAUDE.md`, `.claude/settings.json` (data-access deny rules).
- mimic-code cloned into `external/mimic-code`, commit 303d26c6 (with the official `concepts_duckdb`).
- The paper PDF and two old notebooks (outputs cleared) copied into `docs/`.

## Stage 1: cohort construction
- Subset database: 3,046 candidate patients (union of definitions A–D, after LOS/age), 2,962 × 10⁴ (29.62 million) chartevents rows; all 24 mimic-code concept tables generated successfully (no dependency chain needed simplifying).
- DN codes: 8,992 patients / 24,834 hospitalizations (no conflicts in ICD-9/ICD-10 version matching); 3,189 of them have a hospitalization with both a DN code and an ICU stay.
- Final counts by definition (paper 1,313): A 837, B 1,228, **C 1,960 (main analysis)**, D 1,685.
- Definition C flow: 3,189 → LOS>48h 3,046 → age>18 3,046 → complete CBC 1,960. The proportion kept at each step is almost identical to the paper (LOS 95.5% vs 95.4%; CBC 64.3% vs 64.2%); neither excluded anyone at the age step. CBC missingness comes almost entirely from the white-cell differential (35.6%); the CBC itself is only 1% missing.
- Outcome (definition C, n=1,960): one-year death (from ICU admission) 494 (25.2%); the same counting from hospital admission (25.2%); in-hospital death 7.1%; the "dod not null" definition 42.8%.
- Gap to the paper's 38.2%: the paper most likely counted any non-null `dod` as death (this definition gives 42.8% in our cohort, and v3.1 includes patients from 2020–2022, with `dod` follow-up extending with the last discharge). Of the post-discharge deaths with a `dod`, 49% occur > 365 days after the ICU stay and do not belong to the one-year outcome. The old code's 24.9% agrees with this result (25.2%).
- 247 patients (12.6%) in the cohort come from the 2020–2022 group (new in v3.x), one reason the count is higher than the paper's; another reason is the number of DN patients itself (8,992 vs 4,137).
- Decisions: D1.1–D1.7 (see decisions.md).

## Stage 2: feature extraction
- Feature table: 1,960 rows × 55 predictors (2 demographic + 6 comorbidity + 31 lab + 4 vital sign + 6 score + 6 composite indices) + id/outcome, saved as `data/processed/features.parquet`.
- Missing rates: all ≤ 7.04% (Calcium, PTT 7.04%; PT/INR 6.79%; temperature 2.19%; others < 1%); no variable exceeds the 20% exclusion threshold.
- Compared with the overall column of the paper's Table 1 (`feature_vs_paper.csv`): demographics, MI/heart failure/chronic pulmonary disease/renal disease/obesity, all electrolytes/renal function/coagulation/CBC/vital signs, SAPS-II and OASIS are almost the same as the paper. APS-III (45 vs 47), LODS (4 vs 5), GCS (15 vs 14) and CCI (7.11 vs 7.80) are slightly lower.
- Unit mixing confirmed: 51146 is basophils %; the paper's white-cell differential mixes scales across patients (D2.11). With the correct definition, NLR/LMR/SII/PWR agree with the paper and PLR/PNR do not.
- Hypertension: complete definition 84.6%, whereas uncomplicated hypertension 35.4% ≈ paper 37.3% (D2.9).
- Decisions: D2.1–D2.11.

## Stage 3: preprocessing and Table 1
- Split: training set 1,568 patients (395 deaths, 25.19%) / validation set 392 patients (99 deaths, 25.26%).
- No variable has > 20% missing, so all 55 variables are kept; two sets of imputed matrices, faithful / noleak (original scale + standardised), are produced.
- Table 1 (`results/tables/table1.csv`, observed data): no continuous variable passes Shapiro-Wilk, so all are reported as median (IQR). 39 variables differ between groups with p < 0.05 (42 in the paper). Directions consistent with the paper: age, heart failure, renal disease, obesity (higher in survivors), anion gap, creatinine/BUN, Hb/RBC, RDW, chloride (lower in deaths), PT/PTT/INR, MBP, respiratory rate, all scores, NLR/LMR/SII.
- Inconsistent with the paper: chronic pulmonary disease, hypertension, glucose, WBC and PWR are not significant in our data; bicarbonate_min and potassium_max are significant but not in the paper. Hypertension is not significant because the complete definition is used (D2.9).
- Decisions: D3.1–D3.6.

## Stage 4: modelling and evaluation (including calibration)
- LASSO (binomial deviance, 10-fold CV): faithful version λ.min = 0.01065 (20 variables), λ.1se = 0.02700 (12); leakage-free version λ.min = 0.01052, λ.1se = 0.02667, selecting the same 12 variables. The paper has 0.01056 / 0.02687 (21 / 12). Variables at λ.1se: age, aniongap_min, rdw_mean, lymphocytes_mean, chloride_max, pt_min, sapsii, apsiii, oasis, cci, nlr, sii, overlapping 7 of the paper's 12 (age, aniongap_min, rdw_mean, chloride_max, pt_min, apsiii, cci); the paper's obesity, neutrophils_mean, ptt_min and lmr appear in our λ.min set (11/12 overlap at λ.min, only pwr missing). The selection is unchanged in the hypertension sensitivity analysis.
- Validation AUC (leakage-free + tuned): LR 0.803, SVM 0.802, RF 0.793, XGBoost 0.782 (95% CIs about ±0.05 wide, overlapping). Paper parameters: LR 0.803, XGB 0.787, RF (literal parameters) 0.780 (the same as the paper's 0.780), SVM 0.764, RF (R default) 0.756.
- Overfitting: RF (R default, 9 fully grown trees) training AUC 0.999 / validation 0.756 (difference 0.243), exactly the same pattern as the paper's RF 0.999 / 0.780; RF (literal parameters) difference 0.106; after tuning every model has a difference ≤ 0.057.
- Calibration (validation set): LR Brier 0.145 (slope 1.21), SVM 0.146 (0.99), XGB 0.148 (0.88), tuned RF 0.153 (slope 1.76, probabilities compressed toward the prevalence, recall only 0.13 at threshold 0.5); RF (R default) slope 0.17, severely overconfident.
- Main-analysis model (chosen by CV AUC): tuned RF (tied with XGBoost). DCA net benefit is above treat-all/none for thresholds 0.07–0.59. SHAP top 5: rdw_mean, cci, sapsii, apsiii, chloride_max (paper: rdw_mean, neutrophils_mean, cci, chloride_max, pt_min); top 5 of the paper configuration (faithful version + paper parameters + RF): rdw_mean, sapsii, cci, age, apsiii.
- The faithful and leakage-free versions give almost the same results (the leakage from mean imputation is negligible when missingness is ≤ 7%).
- Decisions: D4.1–D4.12.

## Stage 5: replication report
- `results/report.md` is generated automatically by `scripts/05_report.py` from aggregate tables and covers: cohort flow and the four definitions, outcome and mortality differences, comparison of the 55 features with the paper's Table 1, unit-mixing analysis, Table 1, LASSO, AUC table and relative ranking, threshold metrics, calibration, overfitting check, DCA/CIC, SHAP, summary of differences, unclear points and how they were handled, limitations and future work.
- Compliance check: `results/tables` contains no ID columns; the single-patient SHAP force plots (fig6c–e) are in .gitignore and kept only on the local machine.
- Run order: 01 → 02 → 03 → 04 → 04b → 05 (each script can be rerun on its own).

## Stage 6: thresholds, model comparison and recommended model (2026-09-30, `04c_threshold_compare.py`)
- Thresholds: for each model the Youden optimal threshold is taken on the training-set 10-fold OOF predictions (0.23–0.26 for the tuned models, close to the prevalence), and accuracy/precision/recall/F1/specificity are reported on the validation set. Recommended LR threshold 0.228: recall 0.77, specificity 0.67, precision 0.44, F1 0.56 (recall 0.37 at 0.5). Threshold 0.5 is kept as a comparison, and the report explains why it is unsuitable (D4.13).
- Model comparison: validation-set paired DeLong + Holm correction; no pair of models is significant. Tuned RF vs LR: ΔAUC −0.010 (−0.025 to +0.006), p = 0.21 (D4.14).
- The recommended model changed to the tuned LR (AUC 0.803, lowest Brier among the tuned models, interpretable), with RF as a secondary result (D4.15; a post-hoc rule, described in the report's limitations). LR interpretation: ORs (95% CI) of unpenalised LR, VIF, SHAP. Significant factors: RDW, CCI, age, APS-III, OASIS (risk), chloride_max, lymphocytes (protective); 4 of the SHAP top 5 are the same for LR and RF (D4.16).
- Hypertension: new report Section 3.3. The uncomplicated-only definition (35.4%, ≈ paper) records 964 patients with only complicated codes as "no", producing a spurious protective effect (OR 0.49); complete definition OR 1.07 (p = 0.65). The main analysis keeps the complete definition, and the LASSO selection is unaffected.
- DCA useful range changed to the D4.17 definition; new DCA comparison figure of the tuned models (Fig. 5B) and LR OR figure (Fig. 8).
- Code: model definitions moved to `src/models.py` (shared by 04 and 04c; the models rebuilt in 04c agree with the predictions saved in stage 4, error < 1e-6); `src/metrics.py` gets specificity, Youden threshold, paired DeLong and DCA useful range.

## Stage 7: changes after review comments (2026-09-30)
- **Task 1 (check + nested CV, `04d_nested_cv.py`, D4.18)**: the check concluded that the stage-4 LASSO was done once on the whole training set and CV was run on the fixed 12 variables, so the CV AUC contains variable-selection bias. The nested version redoes imputation/standardisation and LASSO (λ.1se) inside each fold, with fixed hyperparameters. The bias is +0.008 to +0.015 (RF R default +0.030): recommended LR 0.793 → 0.781, tuned RF 0.800 → 0.785. Each fold selects 9–15 variables, and 6 of the stage-4 12 are selected in 10/10 folds. The bias is small and the main conclusions do not change; the non-nested recomputation agrees with stage 4 (difference < 1e-3).
- **Task 2 (recommended model changed to unpenalised LR, `04c`, D4.15/D4.16 revision)**:
  - Validation set: AUC 0.803 (0.752–0.853), Brier 0.145 (0.1445, previously written as 0.144 because of double rounding), calibration slope 0.96 (1.21 for L2 LR).
  - Training-set OOF calibration slope: 0.93 vs 1.20, showing that the difference is not only on the validation set.
  - OOF Youden threshold 0.259, validation recall/specificity/precision/F1 = 0.74/0.74/0.49/0.59 (L2 LR 0.228, 0.77/0.67/0.44/0.56).
  - DCA useful range 0.01–0.71, SHAP top 5: rdw_mean, cci, age, apsiii, oasis.
  - DeLong vs tuned RF: +0.010 (−0.008 to +0.028), p = 0.26.
  - New `lr_variants.csv`; the DeLong table gets a new `recommended` group.
- **Task 3**: report Section 0 describes the whole process as "pre-specified rule selects RF → DeLong shows no difference → post hoc changed to L2 LR → changed again to unpenalised LR", and states that external validation is needed.
- **Task 4**: inferences about the paper's methods are now worded cautiously, including the hypertension definition, the fully grown RF trees (adding "other causes cannot be ruled out"), the outcome definition, unit mixing, statistics after imputation, the positive class, calcium units, the spurious neutrophil signal, etc.; D2.9 and D4.4 in decisions.md were changed to match. Factual comparisons (NLR/SII agree, PLR/PNR deviate) are kept.
- **Task 5 (`04e_sensitivity_pre2020.py`, D4.19)**:
  - Cohort: excluding the 247 patients in anchor_year_group 2020–2022 leaves 1,713 patients (mortality 25.0%), with a new 80/20 split (seed 42).
  - LASSO λ.1se 0.0264, 11 variables selected (ptt_min and plr added; aniongap_min, oasis and nlr missing).
  - Validation AUC: recommended LR 0.799 (0.747–0.852), tuned RF 0.802 (0.750–0.855), DeLong p = 0.77.
  - Calibration slope: LR 0.93, RF 1.65. The SHAP top 2 are still rdw_mean and cci.
  - By the pre-specified criteria the conclusions are judged robust. The report gets a new Section 9, the former Sections 9–11 become 10–12, and the former item 6 of "Future work" is deleted.
- Code: new `src/preprocess.py` (shared by 03/04d/04e); `cal_intercept_slope` moved into `src/metrics.py` (shared by 04b/04c/04e). After the refactoring 03 and 04b were rerun, and all result tables were byte-for-byte unchanged.

## Stage 7b: wrap-up
- **Main figures now centred on the recommended unpenalised LR** (D5.2):
  - New `fig3_roc_noleak_main.png` and `fig7_calibration_noleak_main.png`, containing unpenalised LR and the four tuned models.
  - In `04_model.py`, primary is now unpenalised LR (`*_primary_noleak_recommended.*`: fig5 DCA/CIC, fig6a–e SHAP), and the tuned RF is renamed `*_secondary_noleak_tuned.*`; the old `*_primary_noleak_tuned.*` were deleted.
  - `04c` no longer draws the LR SHAP figures a second time.
  - The text and figure paths in report Sections 7 and 8 were updated, with a new description of the CIC of the recommended LR.
- **Table grouping**:
  - Unpenalised LR is listed in every table as a separate `recommended` group and the tuned LR is labelled `LR (L2)`; the AUC table in Section 6.1 becomes a long table (with a nested-CV column) and explains that recommended and the LR in the paper-parameter group are the same model.
  - DeLong comparisons are split into three groups, paper (without LR, 6 pairs), tuned (6 pairs) and recommended (unpenalised LR vs the other 8 models), each with its own Holm correction; the smallest corrected p in the recommended group is 0.19, so the conclusion is unchanged.
- **Wording**:
  - The "RF training AUC" row in Section 10 now reads "consistent with the explanation that the paper's RF used fully grown trees".
  - In Sections 0 and 9, "by criteria written down in advance" becomes "by criteria written down before the sensitivity analysis was run (set in this revision, not pre-specified at the start of the project)"; D4.19 updated to match.
- **Precision**: `calibration.csv` and `lr_variants.csv` now store 6 decimals, to avoid double rounding making the report and the figure legends disagree (the Brier of unpenalised LR is 0.1445, now shown consistently as 0.145).
- **Re-check**: reran 04 → 04b → 04c → 05. Model performance, thresholds, LASSO and hyperparameter tables are byte-for-byte unchanged; calibration has only precision changes; delong_pairwise changed because of the regrouping.

## Stage 8: splitting into a replication track and a clinical track (from 2026-10-01)
- **8a analysis plan**: written into Section 8 of decisions.md (D8.1–D8.9). The user proposed 5 changes when confirming, which have been written in:
  - Only 1 indicator variable for the white-cell differential; ratio variables are imputed directly;
  - The equivalence rule becomes "difference ≤ 0.01, or (CI crosses 0 and difference ≤ 0.02)";
  - Count 586/N19 first, and do the add-back sensitivity analysis if ≥ 1%;
  - When reduced to 3 repeats, paired comparisons use only the common splits;
  - The CLIN flow table lists the number excluded at each step.
- **8b one-year mortality outcome check** (`scripts/01b_outcome_check.py`, D8.10):
  - The new definition is "date of death ∈ [date(intime), date(intime) + 365 days]", with deathtime for in-hospital deaths and dod for out-of-hospital deaths.
  - Under definitions A/B/C/D the number of disagreements between the old and new definitions is 0 (deaths 189/283/494/400 unchanged), so nothing is rerun.
  - The numbers with dod before ICU admission, in-hospital death more than 365 days later, or disagreeing dod and deathtime dates are all 0; 1 patient has hospital_expire_flag = 1 but a null deathtime, and dod is used instead by the rule.
  - Checked against the PhysioNet v3.1 documentation: dod is censored 1 year after the last discharge, so deaths within 365 days of ICU admission are fully observable. Report Section 12.5 was corrected accordingly (the only change).
- **English report** (2026-10-01): `scripts/05_report.py` now generates both `results/report.md` (Chinese) and `results/report_en.md` (English).
  - The English template is in `scripts/templates/report_en.md`; it shares the same numbers and tables as the Chinese version, with English headers and text.
  - The Chinese report is byte-for-byte unchanged; the English report has no leftover Chinese, no unrendered placeholders, and all image links work.
  - When the report is reorganised in stage 8f, both templates need to be changed together.
- **8c step 0: small fixes and determinism** (2026-10-02 to 10-03):
  - **Changes**:
    - The primary outcome in `01_cohort.py` is changed to the D8.10 new definition, keeping the old definition and asserting that the final cohorts of definitions A–D agree patient by patient (passed);
    - In config, `min_age` is split into `min_age_replication` / `min_age_clinical`, and a new `nested_cv` section is added;
    - The first "Stage 8: wrap-up" heading in progress.md is renamed "Stage 7b: wrap-up".
  - **First rerun 01 → 05**: 36 of 38 CSVs are byte-for-byte identical and both reports are byte-for-byte identical; in `table1.csv` and `table1_after_imputation.csv` the p-values of wbc_mean and pwr change beyond the 7th significant digit, so work stopped and this was reported, as required.
  - **Diagnosis**: running the unmodified 02 twice in a row, only wbc_mean (maximum difference 7e-15, 220 patients) and pwr (1.4e-14, 205 patients) differ. The cause is that the floating-point summation order of DuckDB multithreaded `avg()` varies between runs.
  - **Determinism**: the 02 connection is made single-threaded (`SET threads=1`); running 02 twice in a row gives identical feature tables. Compared with the old multithreaded results, the hemoglobin/wbc/rbc/rdw means and pwr differ by ≤ 3e-14.
  - **Split check**: training 1,568 / validation 392, stay_id sets identical to the original split.parquet.
  - **After rerunning 03 → 05, compared with the previous commit (847a84d)**:
    - 32 of 38 CSVs are byte-for-byte identical;
    - 4 have only last-digit floating-point differences: `lasso_coefs_faithful` 1.3e-15, `lasso_coefs_noleak` 6.1e-16, `shap_importance` 2.8e-16, `shap_importance_lr` 2.8e-16;
    - `table1.csv`, `table1_after_imputation.csv`: the Mann-Whitney p-values of hemoglobin/wbc/rbc/rdw_mean and pwr change, maximum absolute difference 0.0011 (wbc_mean: 0.7461 → 0.7472, displayed 0.746 → 0.747). The cause is that last-digit differences broke or created ties, changing the ranks and the tie correction. Other columns are unchanged.
    - `report.md` and `report_en.md` are byte-for-byte identical; no number shown in the reports changed.
- **8c steps 1–2: full nested CV framework** (`scripts/04f_nested_full.py`, 2026-10-03):
  - **Seeds** (one per repeat, config `nested_cv.seeds = [42, 43, 44, 45, 46]`; all the places below use the seed of that repeat):
    - Outer split: `StratifiedKFold(10, shuffle=True, random_state=seed)`;
    - LASSO inner 10 folds: `cv_lasso(..., n_folds=10, seed=seed)`, internally `StratifiedKFold(random_state=seed)`, and liblinear also uses that `random_state`;
    - Tuning inner 5 folds: `GridSearchCV(cv=StratifiedKFold(5, shuffle=True, random_state=seed))` (D8.5 revision 2026-10-02);
    - The models' own `random_state`: `paper_models(seed)`, `tuning_grids(p, seed)` (RF, XGBoost, SVC); LR uses lbfgs and has no randomness;
    - Internal CV of SVC Platt scaling: `SVC(probability=True, random_state=seed)` when refitting after tuning; the paper-parameter SVM is likewise set by `paper_models(seed)`.
  - **Leakage prevention**: the held-out validation set is not read (parquet row filter, asserting n = 1,568). Before missingness screening, imputation/standardisation, LASSO and every model fit, the script asserts that the fitting rows do not overlap the outer validation fold.
  - **Nadeau–Bengio unit check** (`tests/check_nadeau_bengio.py`, simulated data):
    - Formula check passed: σ² is the sample variance of all k·r paired differences (ddof = 1), the t distribution has k·r − 1 degrees of freedom, correction factor 1/(k·r) + n_test/n_train;
    - Null simulation (200 data sets, 10 × 5 CV): rejection rate 0.040 for the corrected test, 0.380 for the uncorrected t-test.
  - **Consistency self-check** (`--selfcheck`: seed 42, inner tuning switched off, stage-4 hyperparameters): the CV AUCs of the 9 models agree with `nested_cv_auc.csv` (04d), maximum difference 4.1e-5 < 1e-3 (from the 4-decimal rounding of that table).
  - **Run-time estimate** (`--timing`: repeat 0, fold 0, all 9 models run, `results/replication/nested_timing_C.csv`):
    - Time per fold: LASSO (shared by all models) 35 s; tuned RF 60 s; tuned XGBoost 10 s; tuned SVM 1.9 s; tuned LR 0.1 s; the 5 paper-parameter models < 0.4 s in total;
    - Extrapolated to 5 × 10 = 50 folds: LASSO 0.49 h + tuned RF 0.83 h + tuned XGBoost 0.14 h + the rest < 0.05 h, about **1.5 h** in total;
    - Far below 48 h, so the D8.5 fallback is not triggered and all 9 models run the full 5 repeats;
    - The fold-0 results have been written to the cache and are reused directly in the full run. The number of selected variables differs between folds, so the actual time may differ.
- **8c step 3: full nested CV run and model selection for definition C** (2026-10-03, D8.11):
  - All 9 models × 5 repeats × 10 folds completed, no zero-variable fold, no fallback; LASSO selected 8–16 variables per fold (median 12).
  - Outer AUROC (fold mean): XGBoost paper parameters 0.791 (M\*), XGBoost tuned 0.790, LR L2 0.788, RF tuned 0.788, SVM tuned 0.788, LR unpenalised 0.786, RF paper literal 0.767, RF R default 0.724, SVM paper parameters 0.723 (SDs all about 0.035–0.044).
  - By D8.8: equivalence set = XGBoost×2, LR×2, RF tuned, SVM tuned (all Δ ≤ 0.005); both LRs are equivalent, and by the OOF calibration slope **unpenalised LR** is chosen (0.854 vs L2 1.189). The recommended model agrees with stages 6–7, so the report's recommendation is unchanged; the AUROC of unpenalised LR is the lowest in the equivalence set (0.005 below M\*), which the report states.
  - ~~Next step: full nested CV for definition A (04f `--cohort A`), then the A/C consistency judgement (D8.7) and the secondary held-out estimate for definition C.~~ (void after the 2026-10-03 revision, see below)
- **8c step 3 addendum: exact recomputation of the LR calibration slopes and handling of non-converged calibration** (2026-10-03, D8.11):
  - The pooled OOF calibration slope of each repeat was recomputed from the cached per-fold predictions, as "|ln slope| per repeat, then averaged": unpenalised LR 0.158, L2 LR 0.170, so **unpenalised LR** is still chosen and the conclusion is unchanged; L2 is closer to 1 in 3/5 repeats and the gap comes mainly from seed 44, so the report describes calibration as "the two LRs are similar" and bases the recommendation mainly on parsimony and interpretability.
  - Convergence checks were added to the 04f calibration fits (non-convergence or |value| > 1e3 recorded as NaN), and the summary gets columns for the number of non-converged folds and the slope of each repeat. Only RF (R default) has 5 folds whose intercept does not converge, and only its pooled OOF calibration is reported; the values of the other models are unchanged.
  - 04f rerun from the cache (no model refitted): dca, pairwise, params, selection and oof_threshold are byte-for-byte identical to before; folds and summary only gain columns, and the fold-level calibration values of RF (R default) become NaN. Log `results/logs/04f_nested_full_C_recal.log`.
- **Plan revision: modelling of definition A cancelled** (2026-10-03, D8.1 and D8.7 revisions):
  - Definition A no longer gets full nested CV, and no A/C consistency judgement is made. The decision was made when there were no model results for definition A.
  - In the report, definitions A, B and D keep only the cohort flow and mortality; Section 2.2 states only the counts and mortality of the four definitions; the wording is "this study uses definition C".
- **Remaining work** (after the revision):
  1. Secondary estimate on the held-out validation set for definition C (D8.6, D8.7): unpenalised LR, Youden threshold 0.229;
  2. 8d: check of the `diabetes_with_cc` proportion; comorbidity-timing sensitivity analysis for definition C (recommended LR + tuned RF, D8.2);
  3. 8e: CLIN cohort, features and full nested CV, model selection by D8.8, confirmatory held-out evaluation (D8.1–D8.4);
  4. 8f: report reorganisation, with the Chinese and English templates kept in step. Written in the D8.9 structure, implementing this revision: definitions A/B/D keep only the flow and mortality, Section 2.2 is rewritten, and the wording "this study uses definition C" is used.
- **8d step 1: secondary held-out estimate for definition C** (2026-10-03, D8.12, `scripts/04g_holdout_C.py`):
  - Unpenalised LR: AUROC 0.803 (0.753–0.853), AUPRC 0.581 (0.500–0.683, baseline 0.253), Brier 0.145, calibration slope 0.96.
  - Tuned RF: AUROC 0.793 (0.741–0.845), AUPRC 0.605 (0.521–0.693), Brier 0.153, calibration slope 1.76.
  - Thresholds are the median Youden thresholds of the nested-CV OOF predictions (LR 0.229, RF 0.247): LR recall 0.78, specificity 0.70; RF recall 0.77, specificity 0.66.
  - EPV 32.9 (395 events / 12 selected variables). This validation set took part in post-hoc model selection and serves only as a secondary estimate.
- **8d step 2: comorbidity-timing sensitivity analysis** (D8.12, `scripts/02b_comorb_time.py` + 04f `--variant comorb_time`):
  - The diagnosis data in the subset database are complete: row counts agree with the raw files.
  - `diabetes_with_cc` is 0 for <10 patients because of 249.6x codes; the statement in D8.2 needs correcting.
  - After recomputation the CCI of 843 patients and the SAPS-II of 21 patients change; for 304 of them the MI comes only from acute codes of the index hospitalization.
  - The smoke test passed; the full run is extrapolated to about 1.2 h. **Full run pending**; results go into D8.12 after the run.
  - **Full run completed** (2026-10-03): compared with the main analysis, ΔAUROC is +0.001 (−0.006, 0.008) for LR and +0.002 (−0.006, 0.010) for RF. AUPRC, Brier and the calibration slope of LR are essentially unchanged (0.854 → 0.852); the calibration slope of RF goes 1.48 → 1.59. CCI is still selected in 50/50 folds. The conclusion is not affected by the timing attribution of comorbidity codes.
  - Why renal disease has "lost = 0": the number of patients with 586/N19 in the index hospitalizations of definition C is 0. The whole of MIMIC-IV v3.1 has only 119 rows of 586/N19; in these data AKI is coded as 584/N17.
- **Remaining work** (updated):
  1. 8e: clinically improved model (CLIN);
  2. 8f: report reorganisation.
- **8e-1: CLIN cohort and features; feature means made independent of row order** (2026-10-03, D8.13):
  - **Subset database**: extended to 3,189 patients (replication-track candidates ∪ CLIN candidates). After the extension, 5 columns of `features.parquet` show differences of about 1e-14, because the row order is not fixed when the database is built and `avg()` accumulates in row order.
  - **Fix in 02**: means are now computed by sorting first and then averaging, independent of row order. Rerun 02 → 05:
    - LASSO variables, λ, tuned hyperparameters and all model metrics are unchanged, and the Chinese and English reports are byte-for-byte identical.
    - The only displayed value that changes is the median of rbc_mean in Table 1, 3.44 → 3.43, which is not in the report.
    - 04d, 04f and 04g keep the old features.
  - **CLIN cohort**: 3,189 → 2,557 patients (594 transferred out and 38 died before 24 h excluded), 680 one-year deaths (26.6%). Training set 2,045 / held-out set 512, both with mortality 26.6%.
  - **CLIN features**:
    - `wbc_diff_missing = 1` for 873 patients (34.1%); partial missingness in 21.
    - Only the 8 white-cell differential variables have > 20% missing.
    - 1,452 patients (56.8%) have a prior hospitalization.
    - 586/N19 affects 0 patients, so no add-back sensitivity analysis is done.
    - `diabetes_with_cc` is 99.9%.
  - **New scripts**: `01c_cohort_clin.py`; 02 and 02b get `--track clinical`; the one-off check script moved to `tests/check_subset_extension.py`.
- **Remaining work** (updated):
  1. 8e-2: CLIN full nested CV (04f `--track clinical`), model selection by D8.8, confirmatory held-out evaluation;
  2. 8f: report reorganisation.
- **8e-3: CLIN model selection** (2026-10-03, D8.14, **selection only, not yet frozen**; held-out set not read):
  - 04f `--track clinical` full run: 9 models × 5 × 10 folds completed, no fallback, no zero-variable fold; LASSO selected 13–23 variables per fold (median 17); only RF (R default) has 1 fold whose calibration intercept does not converge.
  - Outer AUROC: LR L2 0.801 (M\*), LR unpenalised 0.799, SVM tuned 0.798, RF tuned 0.797, XGBoost tuned 0.795, XGBoost paper 0.794, RF paper literal 0.781, SVM paper 0.756, RF R default 0.740.
  - By D8.8: equivalence set = both LRs, tuned SVM/RF/XGBoost, XGBoost paper parameters (Δ ≤ 0.008); |ln slope| of the two LRs (per repeat, then averaged) L2 0.102 vs unpenalised 0.109 → **LR (L2)** chosen. The gap is very small and unpenalised LR is closer to 1 in 3/5 repeats, so the report says "the two LRs are similarly calibrated".
  - L2 LR: pooled OOF calibration slope 1.108, Youden threshold 0.258; tuned RF: 1.465, 0.241.
  - `wbc_diff_missing` selected in 13/50 folds; 9 variables selected in 50/50 (age, aniongap_min, cci, oasis, apsiii, resp_rate_mean, rdw_mean, chloride_max, obesity).
  - Next step: the two pre-specified sensitivity analyses for L2 LR + tuned RF (prior_only, no_wbc_diff), then freeze and do the confirmatory held-out evaluation.
- **8e-4 step 1: CLIN model frozen** (2026-10-04): `frozen_model_clin.json` + `scripts/04h_final_clin.py`. On the training set LASSO λ.1se 0.0168, 22 variables selected; LR (L2) C = 0.01; tuned RF max_depth 10 / max_features 2 / min_samples_leaf 20; EPV 24.7. The held-out set was not read when freezing. The sensitivity analyses were moved to after the held-out evaluation (they do not affect the frozen model).
- **8e-4 step 2: confirmatory evaluation on the CLIN held-out set** (2026-10-04, D8.15, `scripts/04i_holdout_clin.py`, run only once):
  - Run formally once after the three safeguards and the training-set smoke test passed.
  - LR (L2): AUROC 0.750 (0.702–0.797), AUPRC 0.533 (baseline 0.266), Brier 0.165, calibration slope 0.82; tuned RF: AUROC 0.752 (0.704–0.799), calibration slope 1.14; paired DeLong p = 0.83.
  - The held-out AUROC is about 0.05 lower than nested CV (0.801), for both models alike. The apparent training AUROC is 0.817, which does not look like overfitting; sampling from the OOF predictions gives a probability of about 1.4% for ≤ 0.750, i.e. the lower tail of sampling. Reported as is; the model is not changed.
  - SHAP top 3: rdw_mean, cci, age (the same as definition C).
  - Remaining: the two sensitivity analyses; 8f report reorganisation.
- **8e sensitivity analyses** (2026-10-04, D8.16; LR (L2) + tuned RF, 5 × 10 nested CV, same splits as the main analysis):
  - prior_only: LR ΔAUROC −0.014 (−0.023, −0.005), RF −0.008 (−0.016, −0.0001). obesity selection 50 → 4/50. Chronic-disease codes of the index hospitalization add something, so this is reported as a conservative deployment scenario (AUROC about 0.787).
  - no_wbc_diff: LR +0.001 (−0.005, 0.007), RF −0.002 (−0.010, 0.005), median number of selected variables 17 → 13. The white-cell differential variables add nothing.
  - The frozen model and the main analysis are unchanged. Remaining: 8f report reorganisation.
- **8f: report reorganisation** (2026-10-04, D8.17):
  - New `scripts/04j_plots_clin.py`: uses the frozen model to predict the CLIN held-out set again and draws ROC, calibration and DCA (plus a nested-CV OOF calibration figure). The recomputed metrics agree item by item with `holdout_clin.csv` and `holdout_dca_clin.csv` (difference 0), and the post-hoc checks of D8.15 are also written to `holdout_gap_diagnostics_clin.csv`.
  - New `scripts/03c_table1_clin.py`: CLIN Table 1 (2,557 patients, grouped by outcome), mortality in the two `wbc_diff_missing` groups (28.6% vs 25.5%, p = 0.092), and the table of obesity code sources.
  - `05_report.py` and the English template rewritten in the three-part structure (definition C / CLIN / comparison and limitations), Chinese and English in step; the detailed stage 4–7 tables moved to Appendix A. Per the D8.1/D8.7 revisions, definitions A/B/D list only the flow and mortality; "definitions A and C side by side" in `plans/stage8_prompt.md` is no longer carried out.
  - Correction: the CI upper limit of the CLIN held-out LR (L2) AUROC is 0.797 (previously miswritten as 0.798), and the CI upper limit of the paired DeLong difference is 0.016 (previously miswritten as 0.017).
  - Added after review: ORs of binary variables are now reported as "yes vs no" (obesity 0.63, wbc_diff_missing 1.33, gender_male 1.31); 6.1 adds the seed-42 comparison (0.781, consistent with stage 7); the title becomes "Replication and clinical-improvement report".
  - New `scripts/04k_shap_clin.py` (the paper's Figure 4a): SHAP plots of the frozen LR (L2) on the CLIN held-out set (512 patients), without refitting; the model file, fit record and frozen spec hashes are checked as in 04j. Outputs `results/figures/figC5_shap_bar_clin_holdout.{png,pdf}`, `figC6_shap_beeswarm_clin_holdout.{png,pdf}` (22 variables, 300 dpi). Consistency check passed: mean |SHAP| agrees with `clinical/shap_clin.csv` with a maximum difference of 0 after rounding to 6 decimals, and the ranking agrees.
- 2026-10-22 merge of the extensions and project tidy-up (branch `merge-extensions`, backup tag `pre-merge-2026-10-22`): the classmate's extension code was placed in `extensions/` (balanced is the final version; of the strict version only the feature-preparation chain is kept, and its experiment code was moved to the local `archive/`), with the extension result snapshot in `results/extensions/balanced_27464ba98305edbc/`; Part IV (§19–25, Appendix C, opening summary) was spliced into `report.md` / `report_en.md`, with Parts I–III and all numbers unchanged; `scripts/report_extensions.py` and the 3-line interface in `05_report.py` were merged, and the local versions of 04h/04j kept; new `paper/`, `DATA_ACCESS.md`, `tools/check_public_release.py` (checks only the files git will publish), README rewritten, REPRODUCE updated (Section 7, extensions); `CLAUDE.md`, `.claude/`, `results/logs/` and `docs/topic16.pdf` are no longer tracked (still on the local machine). Checks: the release check passes and catches counter-examples; the hashes of the frozen model, fit record, frozen spec, held-out results and `src/` match those at the tag; the synthetic-data smoke tests (Nadeau–Bengio, balanced test_tree_ablations, smoke_checks) pass.
- 2026-10-23 post-hoc addition (D8.18): new `scripts/04l_calibration_ci_clin.py`, adding Wald and stratified bootstrap 95% CIs for the calibration intercept/slope on the CLIN held-out set (smoke passed; point estimates agree with `holdout_clin.csv`; bootstrap 2,000/2,000 converged). LR (L2) slope 0.82 (0.62–1.03), intercept +0.049 (−0.171, +0.270). Report Section 12.5 updated in both Chinese and English; the frozen model and confirmatory results are unchanged.
- 2026-11-01 repository translated to English (tag `pre-english-2026-11-01`): decisions.md and progress.md translated line for line (same line numbers, numbers unchanged); the Chinese `results/report.md` and `scripts/report_extensions.py` removed; `05_report.py` now builds only `report_en.md` (its regenerated output is identical apart from the wording on line 5) and runs the extension-snapshot SHA-256 check itself; Part IV re-spliced by hand; extension code, notebooks (cell sources edited in place, not regenerated) and READMEs translated. The hash-protected snapshot file `results/extensions/balanced_27464ba98305edbc/cache_reuse.json` was left byte-identical and is exempted in `tools/find_cjk.py`. No modelling script, held-out evaluation or extension run was rerun.
