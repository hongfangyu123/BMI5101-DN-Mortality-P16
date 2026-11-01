# Decision log (decisions.md)

Overall principle: carry out the study on MIMIC-IV v3.1 following the paper's methods; where the paper is unclear, choose the methodologically most sound option; do not copy the paper's errors.
The numbering makes entries easy to cite in the report.

## 0. Infrastructure

| # | Decision | Rationale |
|---|---|---|
| D0.1 | Query the `.csv.gz` files directly with DuckDB; column types come from the official mimic-code schema (`buildmimic/postgres/create.sql`, patched the way `buildmimic/duckdb/build_mimic.sh` does it, with `FLOAT` additionally changed to `DOUBLE`) | Avoids type-inference errors; FLOAT is 4 bytes in DuckDB but 8 bytes in Postgres |
| D0.2 | Build a subset database `data/interim/mimic_subset.duckdb` containing only candidate patients (hosp: patients, admissions, diagnoses_icd, services, labevents, d_labitems; icu: icustays, chartevents, outputevents, d_items), and compute the required derived tables with the unmodified mimic-code `concepts_duckdb` SQL | The mimic-code concepts are the closest to the paper's SQL extraction; the subset database avoids repeatedly decompressing the 3.5 GB chartevents. Approved by the user (git-ignored, never leaves the local machine) |
| D0.3 | mimic-code commit: see `external/MIMIC_CODE_COMMIT.txt` (303d26c6…, 2026-09-01) | Reproducibility |
| D0.4 | `icu/` contains both `icustays.csv` and `icustays.csv.gz`; use the official `.csv.gz` (the two have identical stay_id sets, 94,458 rows) | The official file matching SHA256SUMS |

## 1. Cohort (stage 1)

| # | Decision | Rationale |
|---|---|---|
| D1.1 | DN codes strictly follow the paper's 14 codes; matched by `icd_version` (ICD-9 codes matched only in version 9, ICD-10 codes only in version 10); codes are stripped of dots and spaces and upper-cased | Avoids cross-version mismatches; the observed number of non-matching rows is 0 |
| D1.2 | **"first ICU admission at first hospitalization" uses definition C**: for each patient take the **first hospitalization that has both a DN code and an ICU stay**, and the first ICU stay of that hospitalization; then apply the LOS, age and blood-count exclusions (in the paper's order; the exclusion criteria do not take part in choosing the index). The counts under definitions A/B/D are listed in `cohort_flow.csv` as a sensitivity comparison | The usual meaning of this phrase is "keep only one hospitalization and one ICU stay per patient, to avoid repeated measurements". DN is a chronic disease, and whether a patient earlier had non-ICU hospitalizations with a DN code does not change whether they belong to the study population; A (the absolute first hospitalization) would exclude many eligible patients because the DN code or the ICU stay appears in a later hospitalization, introducing selection bias related to disease course and care history; B likewise (excluded if the first DN hospitalization had no ICU stay); D (first ICU stay in life) would exclude patients who were not coded with DN at their first ICU stay but were coded at a later ICU hospitalization. C keeps the earliest eligible ICU event for each eligible patient, the standard "first eligible event" approach |
| D1.3 | Length of stay = `dischtime − admittime` > 48 h | Paper's text |
| D1.4 | Age = `anchor_age + year(admittime) − anchor_year`, include **> 18** (strictly greater, literally as in the paper) | MIMIC-IV age must be converted to the admission year; the paper explicitly says > 18 |
| D1.5 | Blood-count completeness: within 6 h before to 24 h after ICU admission (the mimic-code `first_day_lab` window, joined by subject_id), all 8 items WBC, Hb, PLT, RBC, RDW, absolute lymphocyte count, absolute neutrophil count and absolute monocyte count have at least 1 value; the three absolute white-cell counts come from mimic-code `blood_differential` (direct absolute values, or percentage × WBC) | The user chose (a), the 8 items; the window matches the later first-day features; using absolute counts avoids unit mixing (see stage 2) |
| D1.6 | **Primary outcome: all-cause death within 1 year of ICU admission (intime)** = `hospital_expire_flag = 1` or `dod ≤ intime date + 365 days`. The version starting from admittime is reported as a sensitivity result | The model uses first-day ICU data and predicts at the end of the first ICU day, so time zero should be ICU admission; the paper's SHAP case descriptions include "in-hospital death" and "death within one year after discharge". In MIMIC-IV v2.0+, `dod` covers out-of-hospital deaths up to 1 year after the last discharge, so deaths within 365 days of the index ICU stay are fully captured |
| D1.7 | Do not use "`dod` not null means death" | The follow-up length of `dod` differs between patients (up to 1 year after the last discharge), so deaths many years later would count as the outcome and the outcome definition would vary with the number of later visits; methodologically invalid. This definition is computed only as a comparison to explain the paper's 38.2% mortality |
| D1.8 | In definition C, 5 ICU stays have an intime outside [admittime, dischtime] (usually ICU admission during the emergency-department phase); kept | The proportion is tiny (0.3%) and is a known timing discrepancy in MIMIC; there is no methodological basis for exclusion |

## 2. Features (stage 2)

| # | Decision | Rationale |
|---|---|---|
| D2.1 | All itemids were checked against d_labitems / d_items for label, fluid and unit (`results/tables/itemid_audit_lab.csv`, `itemid_audit_chart.csv`), and the scripts hard-assert the units they rely on | 51146 = Basophils (%), not monocytes; 51133 absolute lymphocytes (K/uL) and 51244 lymphocyte percentage (%) are different quantities, which the old code averaged together; 52075/52074/51133 are all K/uL, 52769/51253 are #/uL and are divided by 1000 by mimic-code |
| D2.2 | Lymphocytes, neutrophils and monocytes = the **absolute counts (K/uL = 10⁹/L)** from mimic-code `blood_differential`: use the directly measured absolute value when available, otherwise percentage × WBC / 100 (mimic-code rule); take the mean within the first-day window | The only definition with a single unit that matches the unit stated in Table 1 (10⁹/L) |
| D2.3 | Labs summarised by the mean (glucose, Hb, PLT, WBC, RBC, RDW, the three white-cell types) are averaged over the [intime−6h, intime+24h] window from mimic-code chemistry / complete_blood_count / blood_differential; labs summarised by min/max are taken directly from mimic-code `first_day_lab` | The window matches the mimic-code first-day definition; first_day_lab itself does not provide means |
| D2.4 | Glucose_mean uses laboratory serum glucose (chemistry, 50931), not bedside fingerstick glucose (chartevents) | The paper lists glucose among the "first-day laboratory results" |
| D2.5 | WBC: 51300/51301/51755 (all K/uL), averaged after de-duplicating by specimen | The three itemids are different codes for the same test |
| D2.6 | Vital signs: heart_rate_mean, mbp_mean (invasive + non-invasive), resp_rate_mean and temperature_mean (°F converted to °C, with range filtering) from mimic-code `first_day_vitalsign` | Official definition |
| D2.7 | GCS = mimic-code `first_day_gcs.gcs_min` (lowest on the first day) | The standard way to take a first-day score (worst value) |
| D2.8 | CCI = mimic-code `charlson.charlson_comorbidity_index` (including the age points, from the diagnoses of the index hospitalization); myocardial infarction, heart failure, chronic pulmonary disease and renal disease come from the same charlson flags | Specified by the user; the old code left out the age points |
| D2.9 | **Hypertension = Quan-Elixhauser hypertension (uncomplicated + complicated)**: ICD-9 401–405, ICD-10 I10–I13, I15, I16 (I16 added in FY2019); obesity = ICD-9 278.0x, ICD-10 E66.x (Quan-Elixhauser); only the diagnoses of the index hospitalization are used | A standard, citable coding definition. Note: hypertension defined this way has a prevalence of 84.6%, whereas uncomplicated hypertension alone (401/I10) gives 35.4%, close to the paper's 37.3%, suggesting that the paper may have used "uncomplicated hypertension" (the paper gives no codes, so this cannot be confirmed). The latter would record patients with hypertensive CKD/heart failure (I12/I13) as "no hypertension", a misclassification, so the main analysis uses the complete definition; "uncomplicated only" is a stage-4 sensitivity analysis. Addendum (stage 4c, `hypertension_definitions.csv`): under ICD rules, I12/I13 is coded instead of I10 when CKD/heart failure is present, so under the uncomplicated-only definition the "no hypertension" group contains 964 patients with only complicated codes (mortality 31.3%, OR 1.91), making "hypertension" look like a spurious protective factor (OR 0.49); under the complete definition OR 1.07 (p = 0.65). The LASSO λ.1se selection is the same under both definitions, so the model is unaffected |
| D2.10 | Composite inflammation indices: NLR = N/L, PLR = PLT/L, LMR = L/M, PNR = PLT/N, PWR = PLT/WBC, SII = PLT × N / L, all from first-day means; when the denominator is 0 the value is set to missing (6 for lymphocytes, 18 for monocytes, 1 for neutrophils) and imputed later | Avoids infinite values |
| D2.11 | The distributions of lymphocytes, neutrophils and monocytes in the paper's Table 1 are abnormal because the white-cell differentials of different patients are on different scales (K/uL, %, #/uL): with our "absolute count" definition, NLR, LMR, SII and PWR are almost identical to the paper's (median NLR 6.82 vs 6.80, SII 1336 vs 1342). These ratios are unaffected when the three cell types of one patient are on the same scale, whereas PLR (159 vs 36) and PNR (22 vs 6.6), which involve platelets, deviate strongly. The mixed-definition diagnostic is in `unit_mixing_diagnostic.csv` | The paper's neutrophils_mean (SHAP rank 2) therefore partly reflects test format or period rather than a physiological quantity; PLR/PNR are distorted in the same way. This study does not copy it |

## 3. Preprocessing and Table 1 (stage 3)

| # | Decision | Rationale |
|---|---|---|
| D3.1 | First fix a stratified 80/20 split (random_state=42, stratified by outcome), saved as `split.parquet` and shared by both preprocessing versions and stage 4 | Keeps the two versions comparable |
| D3.2 | Drop variables with > 20% missing, judged on the whole cohort (as in the paper); in practice no variable was dropped (maximum 7.04%), so both versions have the same variable set | Paper's rule |
| D3.3 | Imputation: mean for continuous variables; mode for binary variables and ordinal count scores (GCS, CCI) (the paper's "counting data"). The other scores (SAPS-II, APS-III, LODS, OASIS) have no missing values | Paper's rule |
| D3.4 | **Faithful version** `faithful_*`: imputation and standardisation (z-score) fitted on all data; **leakage-free version** `noleak_*`: fitted on the training set only and then applied to the validation set. Each version saves the imputed original-scale matrix (for tree models and SHAP) and the standardised matrix (for LASSO, SVM, LR) | The user asked to keep both versions |
| D3.5 | **Table 1 uses the non-imputed observed data** (with the missing rate reported for each variable); an imputed version `table1_after_imputation.csv` is also produced for comparison with the paper | The paper computed its statistics after imputation (repeated non-integer values such as Calcium 8.17/8.69 and PTT_max 43.78 are traces of mean imputation), which compresses variance and distorts medians and IQRs |
| D3.6 | Normality: Shapiro-Wilk in each group separately; normal only if both p > 0.05 (mean ± SD, Welch t-test), otherwise median (IQR) and Mann-Whitney U (equivalent to Kruskal-Wallis for two groups); categorical variables use Pearson chi-square (no continuity correction, matching SPSS "Pearson Chi-Square"). With about 1,500 patients Shapiro-Wilk rejects normality for every continuous variable, so all are reported as non-normal; the CSV also stores mean ± SD columns | Strictly follows the specified methods; the t-test uses the Welch version, which does not assume equal variances and is more robust |

## 4. Modelling and evaluation (stage 4)

| # | Decision | Rationale |
|---|---|---|
| D4.1 | LASSO: L1 logistic regression on the standardised training data (glmnet objective −loglik/n + λ‖β‖₁, intercept not penalised; sklearn `C = 1/(nλ)`; solver liblinear (newGLMNET coordinate descent, like glmnet), with `intercept_scaling=1e3` so the intercept penalty is negligible; coefficients differ from the saga solution with an unpenalised intercept by < 3e-5 with the same non-zero pattern; saga converges too slowly on the highly collinear min/max variables; note that sklearn 1.9 requires `l1_ratio=1`). The λ grid follows the glmnet default: 100 log-spaced values from λ_max to λ_max × 1e-4; 10-fold stratified CV (seed 42) with binomial deviance as the metric; λ.min = the minimum CV deviance, λ.1se = the largest λ whose deviance does not exceed "minimum + 1 SE"; λ.1se is used in the end | Consistent with R glmnet; binomial deviance specified by the user (the y-axis of the paper's Fig. 2A is Binomial Deviance) |
| D4.2 | A given λ means the same thing in R and Python (both on the glmnet scale), so λ values can be compared directly with the paper (0.01056 / 0.02687) | — |
| D4.3 | The models use only the LASSO-selected variables; RF/XGBoost use the imputed original scale (tree models are insensitive to monotone transformations, and SHAP values are then interpreted in original units), SVM/LR use the standardised data | e1071 `svm()` defaults to `scale=TRUE`, i.e. trains on standardised data |
| D4.4 | **R → Python mapping of the paper's hyperparameters**: RF `mtry=2` → `max_features=2`, `ntree=9` → `n_estimators=9`. The paper also gives sklearn-style `max_depth=10, min_samples_split=30, min_samples_leaf=10`, plus `max_features='sqrt'`, which conflicts with mtry=2 (sqrt is 3 with 12 variables); mtry=2 takes precedence. Two versions are run: **RF (literal parameters)** uses all the parameters above; **RF (R default)** mimics `randomForest(ntree=9, mtry=2)`, i.e. nodesize=1 with fully grown trees. SVM: e1071 `radial, gamma=0.1, cost=1` → `SVC(rbf, gamma=0.1, C=1)` (both libsvm, same kernel definition, both use Platt scaling for probabilities). XGBoost: `nrounds=50` → `n_estimators=50`, `eta=0.3` → `learning_rate=0.3`, other parameters have the same names. LR: R `glm` → unpenalised `LogisticRegression(C=inf)` | The user asked for both RF versions. With min_samples_leaf=10, the 9 trees reach a training AUC of only 0.887 on these data, whereas the paper reports 0.999, which is consistent with the R default (fully grown trees), though other causes cannot be ruled out |
| D4.5 | Tuning: 10-fold stratified CV on the training set, grid search with AUC as the metric. RF: 500 trees × max_features × min_samples_leaf × max_depth; XGBoost: n_estimators × max_depth × learning_rate × colsample × subsample × min_child_weight; SVM: C × gamma; LR: L2 penalty C (including C=∞). Grids are in `04_model.py` and the best parameters in `tuned_hyperparameters.json`; SVM is refitted at the best parameters (with Platt probabilities enabled) | The paper mentions tuning with caret and 10-fold CV but does not report the grids |
| D4.6 | Metrics: AUC with DeLong 95% CI (Sun & Xu fast algorithm, verified by bootstrap); **death is the positive class**; accuracy, precision, recall, F1 (and specificity) originally used a threshold of 0.5, which is now kept for comparison, and the main results use the Youden threshold instead (D4.13) | Death as the positive class was specified by the user. The paper's RF validation metrics (0.721/0.735/0.858/0.792) are internally consistent only when "survival" is the positive class (with 163 survivors / 100 deaths in the validation set, TP=140, FP=50, TN=50 → acc 0.722), i.e. R caret's default of treating the first factor level as the positive class; this study does not copy it |
| D4.7 | The "best model" is chosen by **training-set 10-fold CV AUC** (main analysis: leakage-free version + tuning); the paper's final configuration (faithful version + paper hyperparameters + RF) serves as a comparison and also gets DCA/CIC/SHAP. **The recommended model was changed post hoc by D4.15 to unpenalised LR**; the tuned RF chosen by this rule is kept as a secondary result | Choosing a model by validation AUC and then reporting the AUC on the same validation set gives optimistic bias |
| D4.8 | DCA: net benefit = TP/n − FP/n × pt/(1−pt), with treat-all and treat-none also drawn; CIC: per 1,000 people, the number classified as high risk and the number of true deaths among them (rmda definition, point estimates); both computed on the validation set | Standard definitions |
| D4.9 | SHAP is computed on the validation set: TreeExplainer for tree models (probability scale of the death class for RF, log-odds for XGBoost), LinearExplainer for LR, KernelExplainer for SVM (30 k-means background points). The 3 force-plot cases are taken one from each class: survivor, in-hospital death, death within 1 year after discharge; in each class the case whose predicted probability is closest to the class median is chosen (representative case, deterministic choice; the paper's #156/#994/#1144 cannot be matched) | The paper's case numbers depend on its row order and cannot be reproduced |
| D4.10 | Sensitivity analysis: replace hypertension with "uncomplicated hypertension only (401/I10)" and redo LASSO (leakage-free version) to see whether it is selected | Corresponds to D2.9 |
| D4.11 | In the main analysis, the training-set CV AUCs of RF and XGBoost are tied (0.7998 vs 0.7998, difference < 1e-4); by the pre-specified rule (first maximum in lexicographic order) RF is chosen; the report states that the two are equivalent and that the CV AUCs of the 4 tuned models differ by ≤ 0.007 | Recorded as is; the model choice is not changed post hoc |
| D4.12 | Calibration (`04b_calibration.py`, based on the saved predicted probabilities): Brier score, scaled Brier, calibration-in-the-large (intercept with the slope fixed at 1), calibration slope (logistic recalibration), and a calibration curve with the validation set split into 10 groups by predicted-probability decile. Overfitting criterion: training AUC − validation AUC > 0.10 or training AUC > 0.95 is recorded as overfitting | Requested by the user; standard reporting content for clinical prediction models (TRIPOD) |

## 5. Report (stage 5)

| # | Decision | Rationale |
|---|---|---|
| D5.1 | Single-patient SHAP force plots (fig6c–e) show individual feature values, so they are kept only on the local machine and are in .gitignore | Data-compliance red line |
| D5.2 | **Primary/secondary naming of figures (revised 2026-09-30)**: figures with `primary` in the name (fig5 DCA/CIC, fig6a–e SHAP) use the recommended unpenalised LR as the main model; the tuned RF chosen by D4.7 is renamed `secondary` and kept; Fig. 3 (ROC) and Fig. 7 (calibration) get new main figures `*_noleak_main.png` containing the unpenalised LR and the four tuned models. In the report tables, unpenalised LR is listed as a separate `recommended` group (it is the same model as the LR in the paper-parameter group); DeLong comparisons are split into three groups, paper (without LR), tuned and recommended (unpenalised LR vs the other 8 models), each with its own Holm correction | Consistent with the recommended model of D4.15, so readers do not take the recommended model for a member of the paper-parameter group |

## 4c. Thresholds, model comparison and recommended model (`04c_threshold_compare.py`)

| # | Decision | Rationale |
|---|---|---|
| D4.13 | **Classification threshold = training-set Youden optimal threshold**: for each model, 10-fold stratified CV on the training set (the same folds as tuning, seed 42) gives out-of-fold (OOF) predicted probabilities, and the threshold that maximises sensitivity + specificity − 1 is taken (prediction ≥ threshold is classified as death); with that threshold fixed, accuracy, precision, recall, F1, specificity and the proportion classified positive are reported on the validation set. The Youden threshold from in-sample (apparent) training predictions is a comparison; the results at threshold 0.5 are also kept | The user asked for the threshold to be set on the training set; OOF rather than in-sample predictions are used because in-sample predictions place no constraint on overfitted models (RF R default: in-sample J = 0.98, threshold 0.556, OOF threshold 0.333). 0.5 does not match the 25% prevalence, implicitly assumes equal costs of missed and false classifications, and mistakes calibration differences (compressed RF probabilities) for differences in discrimination |
| D4.14 | **Model comparison = paired DeLong test** (Sun & Xu structural components; the standard error of ΔAUC was checked against 2,000 bootstrap replicates on simulated data, 0.0229 vs 0.0230), on the validation set, comparing all models pairwise within each parameter set, with Holm correction for multiple comparisons (6 pairs for tuned, 10 for paper parameters); uncorrected p and ΔAUC 95% CI are also reported | AUCs on the same patients are correlated, so independent-sample comparisons overestimate the variance; pairwise comparisons need multiplicity correction |
| D4.15 | **Recommended model (two post-hoc adjustments, revised 2026-09-30)**: the pre-specified rule D4.7 selected the tuned RF. (1) First adjustment: in the validation-set paired DeLong test the tuned RF and tuned LR did not differ significantly (ΔAUC(RF − LR) = −0.010, −0.025 to +0.006, p = 0.21; no pair of tuned models was significant after Holm correction), so LR was recommended instead (at that time the tuned L2 LR, C = 0.01). (2) Second adjustment: the recommended model was changed to **unpenalised LR** (the same as R `glm` and the paper-parameter LR); the tuned L2 LR is kept as a comparison and the tuned RF is a secondary result. ΔAUC of unpenalised LR vs tuned RF = +0.010 (−0.008 to +0.028), p = 0.26. `05_report.py` asserts that "the differences between RF and both LRs are not significant, and no tuned model differs after Holm correction"; report generation fails if this does not hold | When discrimination cannot be distinguished statistically, prefer the model that is more parsimonious, better calibrated, and directly interpretable and implementable through its coefficients. Between the two LRs: same validation AUC (0.803 vs 0.803, DeLong p = 0.96); the calibration slope of unpenalised LR is 0.96 versus 1.21 for L2 LR (the L2 penalty shrinks coefficients toward 0, making predictions conservative); the same difference is present in the training-set 10-fold OOF predictions (0.93 vs 1.20, `lr_variants.csv`), so it is not a chance result of the validation set. Unpenalised LR also gives 95% CIs and p-values for the ORs; with 12 variables and 395 events (about 33 events per variable), no penalty is needed to prevent overfitting. **Note**: both adjustments were proposed by the user after seeing the validation results and are not part of a pre-specified analysis plan. They did not pick the model with "the highest validation AUC", so the optimistic bias is limited, but external validation is still needed. Section 0 of the report describes this process as it happened |
| D4.16 | **LR interpretation** (revised 2026-09-30: after the recommended model changed to unpenalised LR, SHAP is computed on unpenalised LR, and the coefficients and SHAP ranking of L2 LR are kept as a comparison): (1) coefficients of the tuned LR (L2, C = 0.01) (β per SD and OR); (2) ORs, Wald 95% CIs and p-values of unpenalised LR on the same training set (statsmodels GLM, the same as R `glm` and the paper-parameter LR; coefficient differences < 5e-3, from the lbfgs convergence tolerance), for inference; (3) VIF to assess collinearity; (4) SHAP of the tuned LR on the validation set (LinearExplainer, log-odds scale, training set as background). No single-patient force plot is made for LR | Penalised regression coefficients have no standard confidence intervals, so inference uses the unpenalised model; both have the same validation AUC (0.803). For a linear model SHAP equals β × (x − mean), which corroborates the coefficients |
| D4.17 | Reporting of the **DCA useful range**: from the threshold at which the model's net benefit first exceeds both treat-all and treat-none, up to the last threshold before the net benefit first falls to ≤ 0 (`src/metrics.useful_range`). Tiny fluctuations (±0.003) where the model coincides with the treat-all curve at low thresholds do not cut the range | The original "min–max" wording counted sporadic positive values in the noise at high thresholds (e.g. LR 0.03–0.93, when it actually falls below 0 at about 0.73); strictly requiring continuity would cut the range at tiny fluctuations at low thresholds (XGBoost would be left with only 0.02–0.02). The ranges of RF and the paper configuration are the same under both wordings |

## 4d/4e. Nested cross-validation and sensitivity analysis

| # | Decision | Rationale |
|---|---|---|
| D4.18 | **Relationship between LASSO and CV (check and handling)**. Original implementation: `04_model.py` runs LASSO once on the whole training set (1,568 patients) (inner 10-fold CV to choose λ.1se), giving 12 variables; then 10-fold CV is run on these 12 variables (`cross_val_score` for the paper-parameter version, `GridSearchCV.best_score_` for the tuned version). The validation part of every fold took part in variable selection, so the "10-fold CV AUC" contains optimistic bias from variable selection. Handling: `04d_nested_cv.py` implements a nested version. The outer loop uses the same 10 folds as stage 4; in each fold the training part refits imputation, standardisation and LASSO (λ.1se), then fits the model and evaluates it on the validation part of that fold. Hyperparameters are fixed at the stage-4 values, isolating only the bias from variable selection. The non-nested CV is also recomputed on the same folds and checked against the stage-4 values (difference < 1e-3). The size of the bias is in `nested_cv_auc.csv` and report 6.6 | Variable selection must be repeated inside every CV fold for the CV estimate to be unbiased. The validation-set AUC is unaffected because the validation set never took part in LASSO. Tuning was not nested as well: the RF/XGBoost grids (100/288 combinations × 10 × 10 folds) are too expensive on an 8 GB machine; the remaining optimistic bias from this is described in the report's limitations |
| D4.19 | **2020–2022 sensitivity analysis**: exclude patients with `anchor_year_group = "2020 - 2022"` and rerun the whole modelling pipeline in the remaining population. Steps: new stratified 80/20 split (seed 42), > 20% missingness screening, imputation/standardisation fitted on the new training set, LASSO λ.1se, unpenalised LR, tuned RF (same grid and 10-fold CV as stage 4), then AUC (DeLong 95% CI), paired DeLong, Brier/calibration slope and SHAP top 5 on the new validation set (`04e_sensitivity_pre2020.py`). Robustness criteria (written down before the sensitivity analysis was run; set in this revision, not pre-specified at the start of the project): the validation AUC of both models changes by ≤ 0.05, LR and RF still do not differ significantly, and RDW and CCI are still in the SHAP top 3 of LR | The v2.x used by the paper runs only to 2019, and 2020–2022 was affected by the COVID-19 pandemic. Definition C chooses the index hospitalization per patient, and excluding whole patients does not change the index of the other patients, so filtering the final cohort directly is equivalent to rerunning stage 1. `anchor_year_group` is a patient-level anchor-year interval, not the true year of the index hospitalization, so it only approximates "excluding hospitalizations from 2020 onward" |

## 8. Stage 8 analysis plan (pre-specified)

**Status**: written on 2026-10-01, when none of the new stage-8 analyses (8b–8e) had been run. Execution started only after the user confirmed. If the plan needs to change during execution, the change is appended after the relevant entry as "Revision (date, reason)", keeping the original text; rules are not changed because of results that have been seen.

**Honest statement of what was already known**:
- **Definition C**: when this plan was written, all definition C results from stages 4–7 had been seen, including the single validation-set AUC, the "LASSO-only nested" CV (recommended LR 0.781, tuned models 0.783–0.789) and the 2020–2022 sensitivity analysis. For definition C, therefore, the rules below are not blinded.
- **Definition A**: only the counts had been seen (837 patients, 189 deaths), no model results at all.
- **CLIN**: no results at all.

Numbered D8.x.

**Changes at confirmation (2026-10-01, proposed by the user when confirming the plan, before any stage-8 analysis was run)**:
1. Only 1 indicator variable for the white-cell differential; ratio variables are imputed directly (D8.4);
2. The equivalence rule becomes "difference ≤ 0.01, or (corrected 95% CI crosses 0 and difference ≤ 0.02)" (D8.8);
3. Handling and counting of 586/N19 (D8.3);
4. When downgraded to 3 repeats, paired comparisons use only the 3 common splits (D8.5, D8.6);
5. The CLIN cohort flow table lists the number excluded at each step (D8.1).

### D8.1 Cohorts and naming

| Track | Cohort | Definition | Known counts | Role |
|---|---|---|---|---|
| Replication track | **Definition A** | The first of all the patient's hospitalizations has both a DN code and an ICU stay; take the first ICU stay of that hospitalization (definition A of D1.2), then apply LOS > 48 h, age > 18 and complete 8-item CBC in the paper's order | 837 / 189 deaths (old outcome definition) | The cohort under the paper's literal definition. The plans called it the "strict replication cohort", but as required by 8f the report never uses the term "strict replication" and writes "definition A (the paper's literal definition)" instead |
| Replication track | **Definition C** | The first hospitalization with both a DN code and an ICU stay, and its first ICU stay (D1.2); exclusions as above | 1,960 / 494 deaths (old outcome definition) | Extended cohort, named "first eligible DN-ICU admission" in the report |
| Clinical track | **CLIN** | Same index hospitalization as definition C (the first hospitalization with both a DN code and an ICU stay), taking the first ICU stay of that hospitalization; prediction time = 24 h after ICU admission | unknown | Clinically improved model |

- **CLIN inclusion criteria** (using only information known at the prediction time):
  - Age ≥ 18 (adults).
  - Still in the ICU at 24 h after admission: `outtime` of the first ICU stay > `intime + 24 h`.
  - Alive at 24 h after admission: `admissions.deathtime` is null or > `intime + 24 h`; `dod` is null or ≥ date(intime + 24 h).
  - Patients whose first ICU stay is shorter than 24 h are excluded outright; a later ICU stay in the same hospitalization is not used instead.
  - LOS > 48 h is not used and a complete CBC is not required.
  - **Cohort flow table** (`results/clinical/cohort_flow_clin.csv`, listed in the report) gives, step by step, the number remaining and the number excluded at this step:
    1. Patients with a hospitalization that has a DN code and an ICU stay (the same as step [1b] of definition C);
    2. Take the first ICU stay of the index hospitalization;
    3. Age ≥ 18;
    4. Exclude patients whose first ICU stay is < 24 h but who did not die before 24 h (transferred out of the ICU);
    5. Exclude patients who died before 24 h after admission.

    Steps 4 and 5 are counted separately: if a patient died within 24 h and the ICU stay was also < 24 h, they are counted in step 5, "died before 24 h". Each step also reports the number of one-year deaths among those remaining.
- The **replication track** keeps the paper's LOS > 48 h, complete CBC and index-hospitalization comorbidities. Two points are described in the limitations: LOS > 48 h depends on information after the prediction time, causing selection bias; diagnosis codes have no timestamps.
- **Outcome**: all three tracks use the new 8b definition, i.e. date of death ∈ [date(intime), date(intime) + 365 days], with `deathtime` for in-hospital deaths and `dod` for out-of-hospital deaths, compared at date granularity. If 8b changes the counts or deaths, the table above is updated to the 8b results.

- **Revision (2026-10-03): modelling of definition A cancelled**. Original text kept.
  - **Decision**: the replication track models only definition C. Definition A no longer gets full nested CV, a final model or SHAP, and no A/C consistency judgement is made (see the D8.7 revision).
  - **Rationale**:
    1. Definition A uses the "first hospitalization in life", so the hospitalization with both a DN code and an ICU stay must be the patient's first hospitalization. A patient is excluded if they had any earlier hospitalization, even if a later hospitalization fully meets the criteria. Whether a patient is excluded depends on care history, which introduces selection bias (D1.2).
    2. The sample is small (837 patients / 189 deaths), so comparisons between models lack statistical power.
    3. Review comment 3 can be answered simply by naming definition C honestly and describing how it differs from the paper's literal definition.
  - **Timing**: the decision was made when there were no model results for definition A (only the counts and deaths had been seen).
  - **Definitions A, B and D in the report**: only the cohort flow and mortality are kept, as a description of different readings of the paper's index definition. Section 2.2 drops "the choice of definition does not change the conclusions" and states only the counts and mortality of the four definitions.
  - **Wording**: the report says "this study uses definition C", not "the paper used definition C", because the paper does not give a specific operational definition. The "Role" column for definition A in the table above no longer applies.

### D8.2 Feature time windows and comorbidity sources

| Item | Replication track (definitions A, C) | Clinical track CLIN |
|---|---|---|
| Labs | mimic-code first day: [intime − 6 h, intime + 24 h] (D1.5, D2.3) | same as left |
| Vital signs | [intime, intime + 24 h] (D2.6) | same as left |
| Scores (SAPS-II, APS-III, LODS, OASIS, GCS) | mimic-code first-day concepts, unchanged | same as left; the chronic-disease items of SAPS-II are recomputed as below |
| 6 comorbidity variables (MI, heart failure, chronic pulmonary disease, renal disease, hypertension, obesity) | index-hospitalization `diagnoses_icd` (paper's condition, D2.8, D2.9) | prior-hospitalization diagnoses ∪ chronic-disease codes in the index hospitalization (D8.3) |
| CCI | mimic-code `charlson_comorbidity_index`, **including age points** (D2.8 unchanged) | comorbidity points only, no age points (recomputed from the D8.3 sources) |
| SAPS-II chronic-disease items | original mimic-code values (index hospitalization) | recomputed from prior ∪ index hospitalization |

**Age points in CCI (verified from the code, no data run)**: in `external/mimic-code/mimic-iv/concepts_duckdb/comorbidity/charlson.sql`, `charlson_comorbidity_index = age_score + the comorbidity points`. `age_score` gives 0–4 points for age < 50, 50–59, 60–69, 70–79 and ≥ 80 respectively.
- **Replication track**: keeps the original definition, comparable with stages 2–7 and the paper. The interpretation of the CCI OR in report Section 8.1 must change to: CCI includes 0–4 age points and age already enters the model separately; the CCI OR is a conditional effect at fixed age, mainly reflecting comorbidity burden and the part of the age grouping not captured by linear age, and cannot be read as a pure comorbidity effect.
- **CLIN**: uses a CCI with comorbidity points only, to avoid counting age twice.

**Diabetes item (verified from the code)**:
- All 14 of the paper's DN codes (ICD-9 2506x; ICD-10 E104x, E114x) fall within the charlson `diabetes_with_cc` code list (2504–2507; E102–E105, E107, E112–E115, E117 …).
- So for every patient in all three tracks the diabetes item is 2 points (a constant), which only shifts CCI and does not affect the models.
- 8d only needs to confirm with an aggregate count that the proportion with `diabetes_with_cc = 1` is 100%.

**SAPS-II chronic-disease items (verified from the code)**:
- The `comorb` subquery in `score/sapsii.sql` checks three items in `mimiciv_hosp.diagnoses_icd` by `hadm_id` (i.e. the index hospitalization): AIDS (17 points), haematological malignancy (10 points) and metastatic cancer (9 points), taking the highest of the three.
- **Replication track**: unchanged.
- **CLIN**: recompute `comorbidity_score` with the same code list on "prior hospitalizations ∪ index hospitalization", then `sapsii_clin = sapsii − original comorbidity_score + recomputed comorbidity_score`. All three items are chronic diseases and all codes in the index hospitalization are kept, so in practice this only adds cases supplied by prior hospitalizations.

**Comorbidity-timing sensitivity analysis for definition C (8d)**:
- CCI is recomputed from "prior hospitalizations ∪ chronic-disease codes of the index hospitalization" (the D8.3 rule), keeping the age points, for comparison with the replication-track main analysis.
- The SAPS-II chronic-disease items are recomputed in the same way as above.
- Full nested CV (D8.5) is rerun only for the recommended LR and the tuned RF, and compared with the main analysis.

### D8.3 CLIN comorbidity rules and acute codes removed from the index hospitalization

- **Prior hospitalizations**: all hospitalizations of the same patient with `dischtime` < the index hospitalization's `admittime`. All their diagnosis codes count, acute or chronic. An acute MI, stroke and the like in a prior hospitalization is itself medical history.
- **Index hospitalization**: the acute-event codes in the table below are removed; the remaining codes count.
- **Code matching**: codes are stripped of dots and matched by prefix, distinguishing ICD-9 and ICD-10 by `icd_version` (as in D1.1).
- **Code lists**: the code lists used to define comorbidities are exactly those of mimic-code charlson (and the Quan-Elixhauser lists for hypertension and obesity, D2.9). The table below lists only what is additionally removed from the index hospitalization.

| Comorbidity | Removed from the index hospitalization (acute events) | Kept (examples) | Rationale |
|---|---|---|---|
| Myocardial infarction | ICD-9 410.x; ICD-10 I21.x, I22.x | 412, I25.2 (old MI) | 410/I21/I22 are acute events of this hospitalization |
| Heart failure | ICD-9 428.21, 428.31, 428.41; ICD-10 I50.21, I50.31, I50.41, I50.811 | chronic and acute-on-chronic (428.22/23, I50.22/23 etc.), unspecified (428.0, 428.9, I50.9) | Only codes explicitly labelled "acute" with no "chronic" component are removed; unspecified codes cannot be distinguished and are kept |
| Chronic pulmonary disease | none | all | All codes in the list are chronic; exacerbation codes (e.g. 491.21, J44.1) themselves imply COPD |
| Renal disease | ICD-9 586; ICD-10 N19 (unspecified renal failure) | N18.x, 585.x, dialysis status (V45.1, Z99.2) etc. | 586/N19 are in the Quan-Charlson renal disease definition but are often used to code AKI in the index hospitalization, so the main analysis removes them from the index hospitalization. The AKI codes 584.x/N17.x are not in the charlson list and do not count from any source. See the count and sensitivity analysis below |
| Hypertension | none | all (including I16 hypertensive crisis) | A crisis implies existing hypertension |
| Obesity | none | all | — |
| Cerebrovascular disease | ICD-9 430–436, 362.34; ICD-10 I60–I64, G45, G46, H34.0 | 437.x, 438.x (late effects), I65–I69 | Acute stroke/TIA is the current event; late-effect codes represent past history |
| Peripheral vascular disease | ICD-9 441.0x; ICD-10 I71.0 (aortic dissection) | 440.x, the rest of 441, I70, the rest of I71 etc. | Dissection is an acute event |
| Peptic ulcer | ICD-9 531–534 with 4th digit .0–.3 (acute); ICD-10 K25–K28 .0–.3 (acute) | 4th digit .4–.7 (chronic), .9 (unspecified) | The ICD 4th digit distinguishes acute from chronic |
| Mild liver disease | ICD-9 570; ICD-10 K76.2 | 571.x, B18, K70.0–K70.3, K73, K74 etc. | 570 is acute/subacute liver necrosis, K76.2 is shock liver |
| Dementia, rheumatic disease, severe liver disease, diabetes, hemiplegia/paraplegia, malignancy, metastatic cancer, AIDS | none | all | Chronic-disease codes. Limitation stated: hemiplegia codes may come from an acute stroke in this hospitalization |

- **Sensitivity analysis**: use only prior-hospitalization diagnoses, and add an indicator "has a prior hospitalization" (as a LASSO candidate). For patients with no prior hospitalization all comorbidities are set to 0.
- **Count and sensitivity analysis for 586/N19**:
  - Before modelling, count in CLIN the patients who "would be recorded as having renal disease if index-hospitalization 586/N19 counted, but not after removal"; only the count is output.
  - If this number is ≥ 1% of the CLIN cohort it is considered non-negligible: add a sensitivity analysis that puts index-hospitalization 586/N19 back into the renal disease definition (CCI updated accordingly), rerun full nested CV for the recommended model and compare with the main analysis.
  - If < 1%, only record the count in decisions.md and the report.
- **Limitation**: DN inclusion is based on index-hospitalization codes and is handled under the assumption that "DN is a chronic disease that existed before admission"; chronic-disease codes of the index hospitalization likewise have no timestamps.
- **Revision (2026-10-03, no CLIN model results yet)**: original text kept.
  - **Sensitivity analysis "prior-hospitalization diagnoses only + has-prior-admission indicator"**: full 5 × 10 nested CV is run only for the model selected by D8.8 in the CLIN main analysis and the tuned RF, in the same way as the comorbidity-timing sensitivity analysis for definition C (D8.12). Specific rules:
    - The 6 comorbidities, the CCI without age points and the SAPS-II chronic-disease items are all computed from prior-hospitalization diagnoses only;
    - For patients with no prior hospitalization all comorbidities are set to 0;
    - `has_prior_admission` is a LASSO candidate.
  - **The 586/N19 add-back analysis is not done**: the D8.13 count is 0 (0%, < 1%), so under the rule above only the count is recorded.

### D8.4 Missingness screening and white-cell differential variables

- **20% missingness rule**: all tracks now compute it within each outer training fold (answering review comment 7), no longer on the whole cohort. Definitions A and C of the replication track follow D3.2.
- **White-cell differential variables in CLIN** (lymphocytes, neutrophils, monocytes and NLR, PLR, LMR, PNR, SII) are expected to have more than 20% missing (about 36% before the CBC exclusion in the replication track):
  - **Main analysis = option 2**: these variables are not subject to the 20% rule. They are handled as follows:
    - **Only 1 indicator variable** `wbc_diff_missing` ("white-cell differential not measured"), as a LASSO candidate, not forced in. Not one per variable: 8 indicators would be perfectly collinear and unpenalised LR could not be estimated.
    - **Definition of the indicator**: before modelling, confirm with aggregate counts whether the 8 variables are missing together, outputting only the number of patients in each missingness pattern. `wbc_diff_missing = 1` if and only if all three cell counts are missing (i.e. no differential was done). If there is partial missingness (only 1–2 of the cell counts missing, or a ratio missing because the denominator is 0, D2.10), these patients have `wbc_diff_missing = 0`, the missing values are imputed as in the next item, and the number with partial missingness is given in the report.
    - **Imputation**: mean imputation within the training fold (consistent with D3.3). Ratio variables (NLR, PLR, LMR, PNR, SII) are imputed directly with their own training-fold means, not recomputed from the imputed cell counts.
    - **Rationale**: at deployment the missingness mechanism is the same (whether a differential was ordered), so the indicator can carry information about illness; the paper's key variables are kept; multiple imputation inside nested CV + LASSO is complex to implement within folds, and the final model would be hard to deploy.
  - **Sensitivity analysis = option 1**: drop these variables under the 20% rule.
- Other CLIN variables with more than 20% missing are dropped by the rule as usual.
- **Revision (2026-10-03, no CLIN model results yet)**: original text kept. Sensitivity analysis option 1 (the 8 white-cell differential variables dropped within folds under the 20% rule, without `wbc_diff_missing`) runs full 5 × 10 nested CV only for the model selected by D8.8 in the CLIN main analysis and the tuned RF, in the same way as the comorbidity-timing sensitivity analysis for definition C (D8.12).

### D8.5 Design of the full nested CV (shared by all three tracks)

- **Outer loop**: stratified 10-fold, repeated 5 times. Each repeat uses a different seed, written into `config/config.yaml` as `nested_cv.seeds: [42, 43, 44, 45, 46]`.
- **All steps are done in order within each outer training fold**; the outer validation fold is used only for evaluation:
  1. 20% missingness screening (D8.4);
  2. Imputation (D3.3; CLIN additionally adds the missingness indicator);
  3. Standardisation;
  4. LASSO: glmnet style (D4.1), inner 10-fold CV, λ.1se; the seed is the seed of that repeat;
  5. For each model: tuned models use inner stratified 5-fold `GridSearchCV` (AUROC, same grids as stage 4 `src/models.tuning_grids`; the `max_features` grid adapts to the number of selected variables p); paper-parameter models have fixed parameters;
  6. Fit on the whole outer training fold and predict the outer validation fold. When SVM is refitted after tuning, Platt probabilities are enabled (as in stage 4).
- **Revision (2026-10-02): seed of inner tuning**. The original plan specified only the seed of the LASSO inner CV (the seed of that repeat), not the seed of the inner stratified 5-fold `GridSearchCV` used for tuning. Added now: the inner tuning `StratifiedKFold(n_splits=5, shuffle=True)` also uses the seed of that repeat (42–46). This fills in an implementation detail the original plan did not specify; when written, no 8c analysis had been run and no results had been seen.
- **Edge case**: if LASSO selects 0 variables in a fold, every model in that fold predicts the training-fold mortality (intercept only), and the number of occurrences is recorded.
- **Models**: the same 9 as in stage 4.
  - Paper-parameter group: RF (literal parameters), RF (R default), XGBoost, SVM, LR (unpenalised);
  - Tuned group: RF, XGBoost, SVM, LR (L2).
- **Samples**:
  - Definition C: within the 1,568-patient training set (the held-out validation set does not take part);
  - Definition A: all 837 patients;
  - CLIN: the new 80% training set.
- **Running**: local machine with 8 GB RAM, n_jobs controlled, run in batches by model, each fold's result cached in `data/interim/nested_cache/` (git-ignored), with resume support. Before starting, estimate the total run time and record it in progress.md.
- **Pre-specified fallback**: if the estimated total time exceeds 48 h, reduce the number of repeats to 3 only for the tuned RF and tuned XGBoost (other models keep 5), and record this in progress.md. The grids are not reduced.
  - When reduced to 3, keep the splits for seeds 42, 43 and 44.
  - All paired comparisons involving RF/XGBoost (including the D8.8 equivalence judgement) use only these 3 common splits (3 × 10 = 30 folds). The Nadeau–Bengio correction factor then uses r = 3.
  - Comparisons not involving these two models still use all 5 splits.
  - Each model's own performance summary uses all its repeats as usual, with the number of repeats stated.
- **Outputs**: `nested_cv_full_*.csv` under `results/replication/` and `results/clinical/`. The original 04d results are kept and labelled "LASSO-only nested".

### D8.6 List of metrics

- **For each outer fold** (5 × 10 = 50 folds) the following metrics are computed and reported as mean ± SD:
  - AUROC;
  - AUPRC (average precision), with the positive rate given as the baseline;
  - Brier, scaled Brier (relative to the positive rate of that fold);
  - Calibration intercept (calibration-in-the-large) and calibration slope.
- **Pooled OOF predictions per repeat**: the same metrics are computed and then averaged over the 5 repeats. DCA is also done: thresholds 0.05–0.60 in steps of 0.01, with treat-all / treat-none; the net benefit at each threshold is the average over the 5 repeats.
- **Paired model comparisons**: using the paired per-fold differences in the outer loop, the 95% CI of the mean difference is computed with the Nadeau–Bengio corrected repeated-CV t statistic. Variance correction factor = 1/(k·r) + n_test/n_train, where k = 10 and r is the number of repeats common to the two models (usually 5; 3 under the D8.5 fallback). ΔAUROC in D8.8 is also computed only on the common splits.
- **Held-out set** (secondary estimate for definition C; confirmatory estimate for CLIN):
  - AUROC (DeLong 95% CI);
  - AUPRC (stratified bootstrap with 2,000 replicates, seed 42, 95% percentile CI);
  - Brier, calibration intercept/slope, DCA;
  - Threshold metrics (accuracy, precision, recall, F1, specificity). The threshold is the Youden optimal threshold of the selected model's pooled OOF predictions in the training-set nested CV, taking the median over the 5 repeats (following the idea of D4.13).
- **Final model**: report the number of events, the number of candidate variables, the number of selected variables, and EPV (events / selected variables, and also events / candidate variables).

### D8.7 Main performance estimate for each track

| Track | Main estimate | Secondary estimate |
|---|---|---|
| Definition C | full nested CV within the 1,568-patient training set | single estimate on the held-out validation set (+ AUPRC and bootstrap CI). Stated: this validation set already took part in the post-hoc model selection of stages 4–7 |
| Definition A | full nested CV on all 837 patients (no held-out set: after 80/20 the validation set would have only about 38 deaths, giving an AUROC 95% CI of about ±0.09) | none. SHAP and coefficients are computed in the final model fitted on the whole cohort; the final model is fitted with the same procedure as D8.5 (LASSO + inner 5-fold tuning) |
| CLIN | after the model is frozen, the held-out set is evaluated only once (confirmatory estimate) | full nested CV on the training set (also used for model selection) |

**Consistency judgement between definitions A and C** (pre-specified): meeting all three conditions below counts as "consistent", two of them as "partly consistent", otherwise "inconsistent".
1. The nested-CV AUROCs of the models selected by D8.8 under the two definitions differ by ≤ 0.05;
2. The two definitions select the same model, or models within each other's equivalence range as defined in D8.8;
3. RDW and CCI are both in the SHAP top 5 of the model selected under each definition.

The statement "the choice of definition does not change the conclusions" in Section 2.2 is removed and replaced by the judgement above.

- **Revision (2026-10-03): A/C consistency judgement cancelled**. Original text kept.
  - **Decision**: definition A is no longer modelled (see the D8.1 revision), so the definition A row of the table above and the consistency judgement above are no longer carried out.
  - **Rationale**: as in the D8.1 revision, i.e. the selection bias of definition A (D1.2), the small sample (837 / 189) lacking power for comparisons, and review comment 3 can be answered by naming definition C honestly. The decision was made when there were no model results for definition A.
  - **Section 2.2**: the previous sentence "replaced by the judgement above" no longer applies. "The choice of definition does not change the conclusions" is still removed; the section now only states the counts and mortality of the four definitions and does not compare model conclusions between definitions.

### D8.8 Model selection rule (written before any new results were seen; applied separately to each track)

1. **Main criterion**: mean outer AUROC of the full nested CV (50 folds). The highest is called M\*.
2. **Equivalence set**: for each candidate model m, compute ΔAUROC = AUROC(M\*) − AUROC(m) and its Nadeau–Bengio corrected 95% CI (D8.6). m is equivalent to M\* if either condition holds:
   - ΔAUROC ≤ 0.01;
   - The Nadeau–Bengio corrected 95% CI crosses 0 and ΔAUROC ≤ 0.02.

   M\* itself belongs to the equivalence set. A model with a difference above 0.02 is not equivalent even if the CI crosses 0.
3. **Parsimony first**: if the equivalence set contains an LR (unpenalised or L2), choose LR; otherwise choose M\*.
4. **Choosing between LRs**: if both LRs are in the equivalence set, choose the one whose pooled OOF calibration slope in nested CV is closer to 1 (smaller |ln slope|, averaged over the 5 repeats); if tied, choose unpenalised LR. Calibration is only an auxiliary criterion between LRs and is not used to choose between LR and other models.
5. **The held-out set does not take part in selection**. The model chosen by this rule is the recommended model of the track, and **the result is binding**:
   - If the rule does not select unpenalised LR for definition C, the recommended model in the report changes accordingly, and the difference from the post-hoc choice of stages 6–7 is explained.
   - The pre-specified main-analysis model for definition C (D4.7: tuned RF) is reported as usual.
6. **How the recommendation is justified**: write only about parsimony, calibration (with the nested-CV OOF calibration slope as evidence) and interpretability; do not claim that the recommended model discriminates better. If the selected LR has a lower AUROC than other models in nested CV, say so.
7. **The report must state**: steps 2 and 3 are a **pre-specified preference for parsimony**. It allows the selected LR to have an AUROC up to 0.02 lower than the best model (when the CI crosses 0) or 0.01 lower (regardless of the CI).

### D8.9 Other pre-specifications

- **Seeds**: the outer `nested_cv.seeds` are written into config; the CLIN 80/20 split and bootstrap use seed 42.
- **Implementation**: the existing scripts take `--track {replication,clinical}` and `--cohort {A,C}` arguments and write their outputs to `results/replication/` and `results/clinical/` respectively, without copying the whole set of scripts.
- **Report structure** (8f):
  1. Replication of the paper: definitions A and C side by side;
  2. Clinically improved model (CLIN);
  3. Comparison with the paper, limitations, future work.
- **Future work**:
  - Drop external validation in eICU-CRD (eICU has only in-hospital outcomes and cannot validate one-year mortality), replacing it with temporal validation by anchor_year_group and a search for external data with one-year follow-up;
  - "Repeated nested CV" was completed in this stage and is removed from future work.
- **Data compliance**: all checks (8b outcome check, consistency of white-cell differential missingness, the `diabetes_with_cc` proportion, etc.) output only counts and summary statistics; cells with n < 10 do not show individually identifiable values.

### D8.10 8b results: one-year mortality outcome check (2026-10-01, run)

**New definition**: date of death ∈ [date(intime), date(intime) + 365 days], compared at date granularity. The date of death is the index hospitalization's `admissions.deathtime` (in-hospital death); otherwise `patients.dod`. If `hospital_expire_flag = 1` but `deathtime` is null, `dod` is likewise used.

**Aggregate checks** (`scripts/01b_outcome_check.py` → `results/tables/outcome_check.csv`, counts only; final cohorts, definitions A / B / C / D):

| Check | A | B | C | D |
|---|---|---|---|---|
| Patients | 837 | 1,228 | 1,960 | 1,685 |
| Deaths: old definition / new definition | 189 / 189 | 283 / 283 | 494 / 494 | 400 / 400 |
| dod before the ICU admission date | 0 | 0 | 0 | 0 |
| deathtime before the ICU admission date | 0 | 0 | 0 | 0 |
| In-hospital death more than 365 days after ICU admission | 0 | 0 | 0 | 0 |
| hospital_expire_flag = 1 but dod null | 0 | 0 | 0 | 0 |
| hospital_expire_flag = 1 but deathtime null | 0 | 0 | 1 | 1 |
| deathtime present but hospital_expire_flag = 0 | 0 | 0 | 0 | 0 |
| dod and deathtime dates disagree | 0 | 0 | 0 | 0 |
| Old and new definitions disagree (old 1 new 0 / old 0 new 1) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- **Conclusion**: under every definition the number of disagreements between the old and new definitions is 0, so **nothing is rerun**. The lower bound missing from the old definition and the upper bound on in-hospital deaths caused no misclassification in these data.
- **The new definition is used from now on**: all three tracks (D8.1) use the new definition, as a preventive correction. The CLIN inclusion criteria of 8e introduce a check for death within 24 h after ICU admission, to which the date-granularity rule of the new definition also applies.
- One patient (the same patient in definitions C/D) has `hospital_expire_flag = 1` but a null `deathtime`; by the rule `dod` is used instead, and the patient counts as a death under both definitions.

**Completeness of follow-up** (verified against the PhysioNet MIMIC-IV v3.1 documentation, https://physionet.org/content/mimiciv/3.1/):
- Documentation text: "Date of death is derived from hospital records and state records. If both exist, hospital records take precedence."
- Documentation text: "Dates of death occurring more than one year after hospital discharge are censored as a part of the deidentification process. As a result, the maximum time of follow up for each patient is exactly one year after their last hospital discharge."
- The documentation also states that state and hospital death records were collected two years after the last patient in MIMIC-IV was discharged, and that from v2.0 `dod` includes out-of-hospital deaths.
- **Inference**: `dod` is censored one year after the patient's last discharge. The last discharge is no earlier than the discharge of the index hospitalization, which is later than the index ICU admission, so every patient's follow-up covers at least 365 days after ICU admission. Deaths within 365 days of ICU admission are fully observable in the data, and there is no right-censoring due to insufficient follow-up.
- **Limitation**: state registry linkage relies on rule-based matching of name, date of birth and social security number, so deaths in other states or failed matches are missed. This is outcome measurement error, not a follow-up length problem.
- **Report change**: in Section 12.5, "covers only 1 year after discharge" becomes "covers up to 1 year after the last discharge", and the inference and limitation above are added.

### D8.11 8c results: full nested CV for definition C and model selection by D8.8 (2026-10-03, run)

**Run**: `scripts/04f_nested_full.py --track replication --cohort C`, training set of 1,568 patients (the held-out validation set was not read), outer 10 folds × 5 repeats (seeds 42–46), all 9 models ran the full 5 repeats (the D8.5 fallback was not triggered), paired comparisons with r = 5, k = 10. Outputs `results/replication/nested_cv_full_{summary,folds,pairwise,dca,oof_threshold,params,selection}_C.csv`.
- Across the 50 outer folds LASSO selected 8–16 variables (median 12); no zero-variable fold (`n_constant_folds = 0`); the 20% missingness screening dropped no variable in any fold. Variables selected in every fold: age, rdw_mean, cci, apsiii, sapsii; in ≥ 40/50 folds: chloride_max, sii, pt_min, aniongap_min, oasis.

**Outer AUROC (mean ± SD over 50 folds) and OOF calibration slope (pooled OOF per repeat, mean over 5 repeats)**:

| Model | AUROC (fold mean ± SD) | Pooled OOF AUROC | ΔAUROC vs M\* (NB-corrected 95% CI) | Equivalent | OOF calibration slope |
|---|---|---|---|---|---|
| XGBoost (paper parameters) = **M\*** | 0.7907 ± 0.0348 | 0.7897 | — | yes (M\*) | 0.957 |
| XGBoost (tuned) | 0.7904 ± 0.0350 | 0.7899 | 0.0003 (−0.0050, 0.0056) | yes | 0.960 |
| LR (L2) | 0.7883 ± 0.0356 | 0.7817 | 0.0024 (−0.0088, 0.0136) | yes | 1.189 |
| RF (tuned) | 0.7882 ± 0.0362 | 0.7855 | 0.0024 (−0.0061, 0.0110) | yes | 1.483 |
| SVM (tuned) | 0.7877 ± 0.0356 | 0.7853 | 0.0029 (−0.0078, 0.0137) | yes | 0.900 |
| LR (unpenalised) | 0.7859 ± 0.0355 | 0.7849 | 0.0048 (−0.0067, 0.0163) | yes | 0.854 |
| RF (paper literal parameters) | 0.7669 ± 0.0378 | 0.7668 | 0.0237 (0.0062, 0.0412) | no | 0.939 |
| RF (R default) | 0.7243 ± 0.0385 | 0.7245 | 0.0664 (0.0428, 0.0899) | no | 0.154 |
| SVM (paper parameters) | 0.7230 ± 0.0443 | 0.7207 | 0.0676 (0.0421, 0.0932) | no | 0.961 |

**Applying D8.8**:
1. M\* = XGBoost (paper parameters), outer AUROC 0.7907. It differs from the tuned XGBoost by only 0.0003, and which one is M\* does not affect the later steps (the two have the same equivalence set).
2. Equivalence set = {XGBoost paper parameters, XGBoost tuned, LR L2, RF tuned, SVM tuned, LR unpenalised}: these 5 candidates all have ΔAUROC ≤ 0.01 (maximum 0.0048) and are equivalent by the first condition. RF (paper literal parameters) has Δ = 0.024 > 0.02 with a CI not crossing 0; RF (R default) and SVM (paper parameters) have Δ > 0.06; none is equivalent.
3. The equivalence set contains LR → choose LR.
4. Both LRs are in the equivalence set → compare |ln slope| of the pooled OOF calibration slope (mean over 5 repeats): unpenalised LR **0.158**, L2 LR **0.170** → **unpenalised LR**.
   - **Computation (exact recomputation on 2026-10-03)**: for each repeat (seeds 42–46), the predictions of the 10 outer validation folds are pooled into OOF predictions covering all 1,568 patients, logistic recalibration (logit(y) ~ a + b·logit(p), p clipped to [1e-6, 1 − 1e-6]) on the pooled OOF gives that repeat's slope b_r, and then (1/5)·Σ|ln b_r| is taken. That is, |ln| is taken per repeat and then averaged, not |ln| of the mean of the 5 slopes. The data are the per-fold predictions cached in `data/interim/nested_cache/`; no model was refitted. The slope and |ln slope| of each repeat are stored in the columns `cal_slope_pooled_rep{42..46}`, `abs_ln_cal_slope_pooled_rep{42..46}` and `abs_ln_cal_slope_pooled_mean_over_repeats` of `nested_cv_full_summary_C.csv`.
   - Slope of each repeat (|ln slope|):

     | Seed | 42 | 43 | 44 | 45 | 46 | Mean |
     |---|---|---|---|---|---|---|
     | Unpenalised LR | 0.830 (0.186) | 0.859 (0.153) | 0.866 (0.144) | 0.866 (0.143) | 0.848 (0.166) | 0.158 |
     | L2 LR | 1.107 (0.101) | 1.179 (0.165) | 1.369 (0.314) | 1.146 (0.137) | 1.145 (0.136) | 0.170 |

   - The conclusion is the same as the initial judgement (|ln| of the mean slope: 0.158 vs 0.173), so the selection does not change. But the gap is small (0.012), and repeat by repeat L2 LR is closer to 1 in 3/5 repeats (42, 45, 46); the mean is pulled apart mainly by seed 44 (L2 slope 1.369). This step therefore only orders the two mechanically by the rule and is not evidence that "unpenalised LR is better calibrated".
5. **Recommended model for definition C = unpenalised LR**, consistent with the post-hoc choice of stages 6–7, so the recommended model in the report does not change. The pre-specified main-analysis model (D4.7: tuned RF) is reported as usual: outer AUROC 0.788, equivalent to M\*, but OOF calibration slope 1.48 (predictions too conservative / shrunk toward the mean).

**Justification of the recommendation (worded as in D8.8 items 6 and 7)**: under full nested CV the OOF calibration slope is 0.854 for unpenalised LR and 1.189 for L2 LR; the two deviate from 1 by similar amounts in opposite directions (the former slightly too extreme, the latter slightly toward the mean). The recommendation rests mainly on parsimony (an unpenalised logistic model with ≈12 LASSO-selected variables) and interpretability (coefficients/ORs directly readable, no penalty strength to choose); on calibration the report says only that "the two LRs are similar", not that unpenalised LR is better calibrated. **No claim of better discrimination**: the outer AUROC of unpenalised LR (0.786) is the lowest of the 6 equivalent models, 0.005 below M\* (95% CI −0.007 to 0.016). The report must state that steps 2 and 3 are a pre-specified preference for parsimony, allowing the selected LR to have an AUROC up to 0.02 lower than the best model (when the CI crosses 0) or 0.01 lower (regardless of the CI); the actual difference here is 0.005.

**Other observations (do not affect the selection)**:
- Compared with 04d ("LASSO-only nested"), after full nesting unpenalised LR goes 0.781 → 0.786 and the tuned models 0.783–0.789 → 0.788–0.790; the direction of the conclusion is unchanged: the gap between the tuned models and LR is within 0.005.
- The fold-mean AUROC of LR (L2) (0.788) is higher than its pooled OOF AUROC (0.782): the C chosen in the inner loop is very small (0.01: 39 folds; 0.001: 11 folds), and the strong shrinkage puts each fold's predictions on a different scale, which damages the ranking after pooling. Its fold-mean calibration slope of 1.53 ± 0.85 is also unstable.
- The mean fold-level calibration intercept of RF (R default) was once a numerically divergent value (−1.6e14): the model overfits the training set, some folds' predictions are close to 0/1, and the logistic recalibration does not converge.
  - **Handling (2026-10-03)**: 04f now uses `src/metrics.cal_intercept_slope_checked`, which records NaN when the GLM fit raises an error, does not converge, shows complete separation/convergence warnings, or gives |value| > 1e3. The summary gets new columns `n_nonconv_cal_intercept_folds`, `n_nonconv_cal_slope_folds`, `n_nonconv_cal_pooled_repeats`; the folds table gets `cal_intercept_ok`, `cal_slope_ok`.
  - Result: only RF (R default) has 5 outer folds whose calibration intercept does not converge (all slopes converge); the other 8 models and all pooled OOF fits converge, with values identical digit for digit to before.
  - For RF (R default) only the pooled OOF calibration is reported (intercept −0.028, slope 0.154), and its fold-level calibration mean/SD in the summary are set to NaN (`cal_fold_level_reported = False`).
  - The old `cal_intercept_slope` is unchanged, so the results of 04b/04c/04e are unaffected.
- The outer corrected t-test shows that RF (paper literal parameters) is significantly worse than M\* (Δ 0.024), and RF (R default) and SVM (paper parameters) are about 0.066 worse: among the paper parameters only XGBoost matches the best model under full nesting.
- To do: after definition A has run, make the A/C consistency judgement per D8.7; the secondary estimate on the held-out validation set (D8.6) uses unpenalised LR with Youden threshold 0.229 (pooled OOF, median over 5 repeats, range 0.220–0.312).

### D8.12 8d results: secondary held-out estimate for definition C and comorbidity-timing sensitivity analysis (2026-10-03)

#### Step 1: secondary estimate on the held-out validation set (D8.6, D8.7, run)

**Note**: this validation set (392 patients, 99 deaths) took part in the post-hoc model selection of stages 6–7 (D4.15), so this is only a **secondary estimate**. The main estimate for definition C is the full nested CV of D8.11.

- **Script**: `scripts/04g_holdout_C.py`, outputs `results/replication/holdout_C.csv`, `holdout_dca_C.csv`.
- **Models**: both models are the versions fitted in stage 4 on the full training set (1,568 patients), i.e. leakage-free preprocessing + the 12 variables selected by LASSO λ.1se. After rebuilding, they were compared one by one with the validation predictions saved in stage 4, with differences < 1e-6.
  - Recommended model: unpenalised LR (D8.11).
  - Pre-specified main-analysis model: tuned RF (D4.7), parameters max_depth 4, max_features 1, min_samples_leaf 10. The script re-applies the D4.7 rule (highest 10-fold CV AUC) and asserts that the model selected is RF.
- **Thresholds**: each model uses the Youden threshold of its pooled OOF predictions in nested CV (median over 5 repeats): LR 0.229, RF 0.247.
- **CI of AUPRC**: stratified bootstrap, resampling deaths and survivors separately with replacement with fixed group sizes, 2,000 replicates, seed 42, 95% percentile interval.

| Metric | Unpenalised LR (recommended) | Tuned RF (D4.7) |
|---|---|---|
| AUROC (DeLong 95% CI) | 0.803 (0.753–0.853) | 0.793 (0.741–0.845) |
| AUPRC (bootstrap 95% CI); positive-rate baseline 0.253 | 0.581 (0.500–0.683) | 0.605 (0.521–0.693) |
| Brier / scaled Brier | 0.145 / 0.234 | 0.153 / 0.192 |
| Calibration intercept / slope | −0.054 / 0.96 | −0.013 / 1.76 |
| Threshold | 0.229 | 0.247 |
| accuracy / precision / recall / F1 / specificity | 0.722 / 0.470 / 0.778 / 0.586 / 0.703 | 0.686 / 0.432 / 0.768 / 0.553 / 0.659 |
| Proportion classified high risk | 41.8% | 44.9% |
| DCA (0.05–0.60): net benefit above both treat-all and 0 | 56/56 thresholds | 53/56 thresholds (0.07–0.59) |

- **Final model information** (the same for both models):
  - Training set of 1,568 patients, 395 events;
  - 55 candidate variables (the 20% missingness screening dropped none), 12 selected by LASSO;
  - EPV (events / selected variables) = 32.9; events / candidate variables = 7.2.
- **Interpretation**:
  - The AUROCs of both models are in the same 0.79–0.80 range as the nested-CV estimates (LR 0.786, RF 0.788); the held-out set shows no additional overfitting.
  - The calibration slope of the tuned RF is 1.76 on the held-out set and 1.48 in the nested-CV OOF, in the same direction: RF predictions are shrunk toward the mean. Unpenalised LR has a slope of 0.96 on the held-out set.
  - The held-out set has only 99 deaths and the AUROC CIs of the two models overlap widely, so the two are not ranked on this basis.

#### Step 2: comorbidity-timing sensitivity analysis (D8.2, D8.3, run)

- **Script**: `scripts/02b_comorb_time.py`, outputs:
  - `data/processed/features_comorb_time.parquet` (patient-level, git-ignored);
  - `results/replication/comorb_time_summary_C.csv`, `comorb_time_flags_C.csv` (aggregate; counts of 1–9 shown as "<10").
- **Implementation**: reuses the mimic-code SQL directly, without rewriting the code lists.
  - Diagnosis codes are all attached to the index hadm_id to build a relabelled diagnosis table, and then the `comorb` CTEs in `charlson.sql` and `sapsii.sql` are run unchanged.
  - Code source for CCI = all codes of prior hospitalizations ∪ codes of the index hospitalization after removing the D8.3 acute codes; age points kept.
  - Code source for the SAPS-II chronic-disease items = all codes of prior hospitalizations ∪ index hospitalization; `sapsii_sens = sapsii − original comorbidity_score + recomputed value`.
  - Prior hospitalizations = hospitalizations of the same patient with `dischtime` < index `admittime`.
  - **Correctness check**: when rerun with index-hospitalization codes only, the 17 charlson components, age points, CCI and the SAPS-II comorbidity_score match the original concept tables exactly for every patient (assertion passed).
- **Completeness of the subset database**: for definition C patients, the subset database and the raw files have the same row counts, 13,936 rows of `admissions` and 266,948 rows of `diagnoses_icd` in both. The subset database is extracted by subject_id, so it contains all hospitalizations of these patients.
- **`diabetes_with_cc` is not 100%**:
  - In definition C, <10 patients have charlson `diabetes_with_cc = 0` in the index hospitalization, all explained by 249.6x codes. 249.6x is secondary diabetes with neuropathy, which is among the paper's 14 DN codes but not in the charlson diabetes code list.
  - **The statement in D8.2 needs correcting**: the original "all 14 DN codes fall within the diabetes_with_cc code list" is inaccurate. The ICD-9 part is actually 2506x plus 24960/24961, and the latter are not in the list.
  - Impact: for these <10 patients the CCI diabetes item is not a constant 2 points. The replication-track main analysis keeps the original definition unchanged.
  - In the sensitivity analysis `diabetes_with_cc` is 1,960/1,960, filled in by prior hospitalizations.
- **Recomputed results** (definition C, 1,960 patients):
  - 1,123 patients (57%) have a prior hospitalization; the index hospitalization of 550 patients contains acute codes as defined in D8.3.
  - CCI median (IQR): 7 (5, 9) before recomputation, 7 (5, 9) after. The CCI of 843 patients changes, 528 up and 315 down, mean change +0.37.
  - SAPS-II median (IQR): 36 (28, 44) before, 36 (29, 45) after. 21 patients change, mean change +0.13.
  - **Change in each comorbidity**:

    | Comorbidity | Originally positive | Positive after recomputation | Lost by removing acute codes | Added from prior hospitalizations |
    |---|---|---|---|---|
    | Myocardial infarction | 611 | 436 | 304 | 129 |
    | Heart failure | 852 | 907 | 20 | 75 |
    | Peripheral vascular disease | 302 | 473 | <10 | 174 |
    | Cerebrovascular disease | 260 | 266 | 108 | 114 |
    | Chronic pulmonary disease | 517 | 634 | 0 | 117 |
    | Renal disease | 909 | 981 | 0 | 72 |
    | Malignancy | 190 | 252 | 0 | 62 |
    | Mild liver disease | 213 | 265 | 11 | 63 |
    | Peptic ulcer | 62 | 106 | <10 | 45 |

    - "Added from prior hospitalizations" means: not positive in the index hospitalization after removing acute codes, positive after adding prior hospitalizations.
    - The other components (dementia, rheumatic disease, hemiplegia, severe liver disease, metastatic cancer, AIDS etc.) only increase, see `comorb_time_flags_C.csv`.
    - Numbers added from prior hospitalizations for the three SAPS-II chronic-disease items: AIDS <10, haematological malignancy <10, metastatic cancer 14.
  - **The largest change comes from myocardial infarction**: for 304 patients the only MI codes are acute codes of the index hospitalization (410/I21/I22), so after recomputation they no longer count as prior MI.
  - **Why renal disease has "lost by removing acute codes = 0"**: in the index hospitalizations of definition C the number of patients with a 586/N19 code is **0**, so the removal rule has no effect on renal disease; it is not because these patients also have other renal codes such as N18/585.
    - 586/N19 are almost never used in MIMIC-IV: the whole MIMIC-IV v3.1 `diagnoses_icd` has only 119 such rows; across all hospitalizations of definition C patients (including prior ones) there are <10 rows.
    - In these data AKI is coded as 584.x/N17.x (not in the charlson list) and chronic kidney disease as 585.x/N18.x.
    - Implication for 8e: the pre-specified D8.3 count of 586/N19 in CLIN is also expected to be close to 0 (< 1%), in which case only the count will be recorded, per the rule.
- **Nested CV**:
  - **Implementation**: 04f gets a new `--variant comorb_time`.
    - Only the cci and sapsii columns are replaced; all other features are the same as the main analysis. Only the recommended LR and tuned RF are run, 5 × 10 folds, with the same seeds as the main analysis, so the outer splits are also the same.
    - The cache is in `data/interim/nested_cache/replication_C_comorb_time/`, and the outputs are `nested_cv_full_*_C_comorb_time.csv`.
    - An additional output `nested_cv_full_vs_main_C_comorb_time.csv` compares ΔAUROC (sensitivity − main analysis) on the same outer splits with the Nadeau–Bengio correction.
    - After the main analysis (without `--variant`) was rerun from the cache, its 7 outputs were byte-for-byte unchanged.
  - **Smoke test** (repeat 0, fold 0): LASSO 32 s, tuned RF 57 s, LR < 0.1 s. Extrapolated full run about **1.2 h**.
  - **Full results** (2026-10-03, 5 × 10 folds, log `results/logs/04f_nested_full_C_comorb_time.log`):
    - Run status: no zero-variable fold; all calibration fits converged; LASSO selected 8–16 variables per fold (median 12).
    - Same-model comparison with the main analysis (metric definitions as in D8.6; ΔAUROC computed pairwise on the same 50 outer splits, Nadeau–Bengio corrected):

      | Metric | Unpenalised LR: main analysis | Unpenalised LR: comorbidity timing | Tuned RF: main analysis | Tuned RF: comorbidity timing |
      |---|---|---|---|---|
      | AUROC (fold mean ± SD) | 0.786 ± 0.036 | 0.787 ± 0.036 | 0.788 ± 0.036 | 0.790 ± 0.036 |
      | ΔAUROC (sensitivity − main analysis, 95% CI) | — | +0.001 (−0.006, 0.008) | — | +0.002 (−0.006, 0.010) |
      | Pooled OOF AUROC | 0.785 | 0.786 | 0.786 | 0.787 |
      | AUPRC (pooled OOF) | 0.550 | 0.548 | 0.548 | 0.545 |
      | Brier (pooled OOF) | 0.151 | 0.151 | 0.154 | 0.155 |
      | Calibration slope (pooled OOF, mean over 5 repeats) | 0.854 | 0.852 | 1.483 | 1.591 |
      | Folds in which LASSO selected CCI | 50/50 | 50/50 | 50/50 | 50/50 |
      | Folds in which LASSO selected SAPS-II | 50/50 | 50/50 | 50/50 | 50/50 |

      (LASSO is run only once per fold and shared by all models, so the selection frequencies are the same for the two models.)
    - Youden threshold (pooled OOF, median over 5 repeats): LR 0.254, RF 0.258 (0.229 and 0.247 in the main analysis).
    - The selection frequencies of the other variables change little, e.g. nlr 36 → 42, oasis 41 → 46, aniongap_min 43 → 48, sii 47 → 44.
    - **Conclusion**: after changing CCI and the SAPS-II chronic-disease items to "prior hospitalizations ∪ chronic-disease codes of the index hospitalization", the discrimination, AUPRC and Brier of both models and the calibration of LR are essentially unchanged (|ΔAUROC| ≤ 0.002, CI crosses 0), and CCI is still selected in every fold.
      - Although the CCI of 843 patients changes, the main-analysis conclusion that "CCI is a stable important predictor" does not depend on the timing attribution of index-hospitalization codes.
      - The calibration slope of the tuned RF rises from 1.48 to 1.59, with more pronounced shrinkage toward the mean, but in the same direction as the main analysis.

### D8.13 8e-1 results: CLIN cohort and features; feature means made independent of row order (2026-10-03)

#### Extension of the subset database
- **Extension**: `01_cohort.py` now builds the subset database for "replication-track candidates (definitions A–D, after LOS/age) ∪ CLIN candidates (index hospitalization of definition C, age ≥ 18, no LOS condition)", 3,189 patients in total (previously 3,046). All of the original 3,046 patients are included.
- **Completeness**: for CLIN patients, `admissions` (17,022 rows) and `diagnoses_icd` (323,916 rows) in the subset database match the raw files.
- **Checks after the extension**:
  - `cohort.parquet`, split, `features_comorb_time.parquet` and all files under `results/replication/` are byte-for-byte unchanged.
  - `itemid_audit_*.csv` counts the whole subset database and changes as the number of patients increases; this is expected.
  - In `features.parquet`, 5 columns (hemoglobin_mean, wbc_mean, rbc_mean, rdw_mean, pwr) differ in the last floating-point digits, maximum absolute difference 1.4e-14, with the missingness pattern unchanged.
- **Cause**: the physical row order is not fixed when DuckDB builds the database (`preserve_insertion_order = false`, parallel inserts), and `avg()` accumulates floating-point numbers in physical order.
  - `tests/check_subset_extension.py` rebuilt the database once with the original 3,046 patients and also could not reproduce the hash of the original `features.parquet`. So the old file itself depended on the row order of one particular build and is not byte-for-byte reproducible.
  - The multithreaded `avg()` difference found in stage 8c is the same kind of problem.
- **Fix**: all means in `02_features.py` (the 9 first-day lab means and the 5 means in the unit-mixing diagnostic) now use `list_avg(list_sort(list(x) FILTER (WHERE x IS NOT NULL)))`, i.e. sort first and then average, so the result no longer depends on row order and thread count.
  - Residual risk: the vital-sign means come from the mimic-code `first_day_vitalsign` concept table (computed with `avg()` when the database is built) and in theory also depend on row order. After this extension these columns are unchanged digit for digit; following the principle of "not modifying mimic-code concepts", they are not changed.
- **Rerun**: 02 (C and clinical) → 03 → 04 → 04b → 04c → 04e → 05. Compared with the previous commit (4cbbf99):
  - **Completely unchanged**: the variables selected by LASSO (12 in both the faithful and leakage-free versions), λ.min, λ.1se (maximum difference 0), `tuned_hyperparameters.json`, `model_performance.csv` and all other model metric tables. `report.md` and `report_en.md` are byte-for-byte identical.
  - **Only last-digit floating-point differences**:
    - `lasso_coefs_faithful` 1.3e-15, `lasso_coefs_noleak` 9.2e-16;
    - `shap_importance`, `shap_importance_lr` 3.9e-16.
  - **`table1.csv`, `table1_after_imputation.csv`**:
    - The Mann-Whitney p-values of hemoglobin/wbc/rbc/rdw_mean and pwr change, maximum absolute difference 7.0e-4 (last-digit differences changed ties, as in 8c); Shapiro p-values differ by ≤ 4e-19.
    - The only displayed value that changes is the median of rbc_mean, 3.44 → 3.43 (overall column, IQR unchanged). This value does not appear in the report.
  - **`feature_vs_paper.csv`**: likewise only the median of rbc_mean, 3.44 → 3.43.
  - **`itemid_audit_lab.csv`, `itemid_audit_chart.csv`**: count the whole subset database, so the counts change after the number of patients increases; this is expected.
- **Results that keep the old features**: 04d, 04f (main analysis and comorb_time sensitivity analysis) and 04g are not rerun and keep the earlier features (which differ from the new ones by about 1e-14). This difference is far smaller than any display precision and model randomness and does not affect any conclusion. The 04f cache meta does not include a feature hash, so a future rerun would reuse these fold results directly.

#### CLIN cohort (D8.1, D8.10; `scripts/01c_cohort_clin.py` → `results/clinical/cohort_flow_clin.csv`)

| Step | Remaining | Excluded at this step | One-year deaths | Mortality |
|---|---|---|---|---|
| [1] Patients with a hospitalization that has a DN code and an ICU stay | 3,189 | 0 | 857 | 26.9% |
| [2] Take the first ICU stay of the index hospitalization (definition C) | 3,189 | 0 | 857 | 26.9% |
| [3] Age ≥ 18 | 3,189 | 0 | 857 | 26.9% |
| [4] Exclude ICU stay < 24 h and alive at 24 h (transferred out) | 2,595 | 594 | 718 | 27.7% |
| [5] Exclude death before 24 h after ICU admission | 2,557 | 38 | 680 | 26.6% |

- **Criteria for steps 4 and 5**:
  - Alive at 24 h after ICU admission = `deathtime` null or > intime + 24 h, and `dod` null or ≥ date(intime + 24 h).
  - Patients who died within 24 h and whose ICU stay was also < 24 h are counted in step 5.
  - The 3,189 patients of step [1] agree with [1b] of the replication-track flow table (assertion passed).
- **Split**: stratified 80/20 (seed 42), training set 2,045 patients (544 deaths, 26.6%), held-out set 512 patients (136 deaths, 26.6%). Saved as `data/processed/split_clin.parquet`.
- Applying the LOS > 48 h condition to CLIN as well would exclude 56 patients (for reference only; CLIN does not use this condition).

#### CLIN features (D8.2–D8.4; `02b_comorb_time.py --track clinical`, `02_features.py --track clinical`)
- **Labs, vital signs, scores**: time windows and definitions as in the replication track; a complete CBC is not required.
- **Comorbidities**:
  - The 6 comorbidity variables come from "prior hospitalizations ∪ index hospitalization without the D8.3 acute codes": MI, heart failure, chronic pulmonary disease and renal disease use charlson; hypertension and obesity use Quan-Elixhauser.
  - CCI has no age points.
  - The SAPS-II chronic-disease items are recomputed on "prior hospitalizations ∪ index hospitalization".
  - Outputs `data/processed/comorb_clin.parquet`, `results/clinical/comorb_clin_summary.csv`, `comorb_clin_flags.csv`.
- **Distribution of CCI and SAPS-II before and after recomputation** (2,557 patients):
  - CCI (without age points) median (IQR): 5 (3, 6) with index-hospitalization codes only, 5 (4, 7) under the CLIN rule; 1,075 patients change.
  - SAPS-II median 36 (29, 45) unchanged; 26 patients change.
- **Counts**:
  - `diabetes_with_cc = 1`: 99.8% looking only at the index hospitalization, 99.9% under the CLIN rule. The gap comes from 249.6x codes, as in D8.12.
  - D8.3 586/N19: 0 patients count as having renal disease only because of index-hospitalization 586/N19 (0%, < 1%). The number with a 586/N19 code in the index hospitalization is also 0. The add-back sensitivity analysis is **not done**; only the count is recorded.
  - `has_prior_admission = 1`: 1,452 patients (56.8%).
- **White-cell differential** (D8.4):
  - New `wbc_diff_missing` (all three cell counts missing), equal to 1 for 873 patients (34.1%).
  - Missingness patterns of the 8 related variables (`wbc_diff_patterns_clin.csv`): all present 1,663 patients; all missing 873; only lmr missing 15; two other patterns <10 each.
  - Partial missingness (`wbc_diff_missing = 0` but at least one missing) in 21 patients in total, all because the denominator of a ratio is 0 (D2.10). Per D8.4, these values are mean-imputed within folds.
- **Missing rates** (`missingness_clin.csv`, computed on the whole cohort, for flagging only):
  - Above 20% are lymphocytes/neutrophils/monocytes_mean (34.1%) and nlr, plr, lmr, pnr, sii (34.2–34.7%), i.e. the white-cell differential variables expected in D8.4.
  - The highest among the other variables is ptt (10.3%).
  - Nothing is dropped or imputed on the whole cohort; all of this is done within the 04f folds.
- **Output**: `data/processed/features_clin.parquet`, containing the 55 variables, `wbc_diff_missing`, `has_prior_admission` and `hypertension_uncomplicated`.

### D8.14 8e-3 results: CLIN model selection by D8.8 (2026-10-03, **not yet frozen**)

**Status**: selection only, **not yet frozen**. The two sensitivity analyses pre-specified in the D8.3 and D8.4 revisions (prior-hospitalization diagnoses only; white-cell differential under option 1) have not been run; freezing and the confirmatory held-out evaluation are done separately. This step did not read or compute anything on the CLIN held-out set (512 patients).

**Data source**: `scripts/04f_nested_full.py --track clinical` (ce548d3), outputs `results/clinical/nested_cv_full_{summary,folds,pairwise,dca,oof_threshold,params,selection}_clin.csv`, log `results/logs/04f_nested_full_clin.log`. Training set of 2,045 patients (544 deaths, 26.6%), outer 10 folds × 5 repeats (seeds 42–46), all 9 models ran the full 5 repeats (no fallback), paired comparisons with r = 5, k = 10 (outer training folds 1,840–1,841 patients, validation folds 204–205).
- 56 candidate variables (55 + `wbc_diff_missing`). The 20% missingness screening dropped no variable in any fold (the 8 white-cell differential variables are exempt under D8.4 option 2, and all other variables have < 20% missing). LASSO selected 13–23 variables per fold (median 17); no zero-variable fold.
- Calibration fits: only RF (R default) has 1 outer fold whose intercept does not converge (recorded as NaN; handled as in D8.11, reporting only its pooled OOF calibration); all fold-level and pooled OOF calibration fits of the other models converge, with no NaN slopes.
- The ΔAUROC and Nadeau–Bengio CIs below were recomputed independently from the folds table and agree with the pairwise table.

| Model | AUROC (mean ± SD over 50 folds) | AUPRC (pooled OOF, mean over 5 repeats; baseline 0.266) | ΔAUROC vs M\* (NB-corrected 95% CI) | Equivalent | OOF calibration slope (mean over 5 repeats) |
|---|---|---|---|---|---|
| LR (L2) = **M\*** | 0.8013 ± 0.0261 | 0.594 | — | yes (M\*) | 1.108 |
| LR (unpenalised) | 0.7985 ± 0.0259 | 0.592 | 0.0028 (−0.0010, 0.0065) | yes | 0.898 |
| SVM (tuned) | 0.7976 ± 0.0281 | 0.584 | 0.0036 (−0.0022, 0.0094) | yes | 0.920 |
| RF (tuned) | 0.7970 ± 0.0282 | 0.591 | 0.0042 (−0.0052, 0.0137) | yes | 1.465 |
| XGBoost (tuned) | 0.7953 ± 0.0280 | 0.596 | 0.0059 (−0.0035, 0.0154) | yes | 0.989 |
| XGBoost (paper parameters) | 0.7936 ± 0.0289 | 0.597 | 0.0077 (−0.0018, 0.0173) | yes | 1.019 |
| RF (paper literal parameters) | 0.7809 ± 0.0289 | 0.560 | 0.0203 (0.0081, 0.0325) | no | 1.088 |
| SVM (paper parameters) | 0.7557 ± 0.0350 | 0.550 | 0.0456 (0.0265, 0.0646) | no | 0.980 |
| RF (R default) | 0.7400 ± 0.0373 | 0.488 | 0.0613 (0.0391, 0.0835) | no | 0.180 |

**Applying D8.8** (same procedure as D8.11 for definition C):
1. M\* = **LR (L2)**, outer AUROC 0.8013.
2. Equivalence set = {LR L2, LR unpenalised, SVM tuned, RF tuned, XGBoost tuned, XGBoost paper parameters}: these 5 candidates all have ΔAUROC ≤ 0.01 (maximum 0.0077) and are equivalent by the first condition. RF (paper literal parameters) has Δ = 0.0203 > 0.02 with a CI not crossing 0; SVM (paper parameters) and RF (R default) have Δ > 0.04; none is equivalent.
3. The equivalence set contains LR → choose LR.
4. Both LRs are in the equivalence set → compare |ln slope| of the pooled OOF calibration slope, computed as in D8.11: for each repeat the 10 outer validation folds are pooled into OOF predictions covering all 2,045 patients, logistic recalibration (p clipped to [1e-6, 1 − 1e-6]) gives slope b_r, and then (1/5)·Σ|ln b_r| is taken; non-convergence is recorded as NaN (neither LR had any non-convergence here).
   - Slope of each repeat (|ln slope|):

     | Seed | 42 | 43 | 44 | 45 | 46 | Mean |
     |---|---|---|---|---|---|---|
     | Unpenalised LR | 0.908 (0.097) | 0.885 (0.122) | 0.864 (0.146) | 0.904 (0.101) | 0.927 (0.076) | 0.1085 |
     | L2 LR | 1.115 (0.109) | 1.098 (0.093) | 1.084 (0.081) | 1.112 (0.106) | 1.132 (0.124) | 0.1024 |

   - 0.1024 < 0.1085 → **L2 LR**.
   - The gap is very small (0.006), and repeat by repeat unpenalised LR is closer to 1 in 3/5 repeats (42, 45, 46); the average advantage of L2 LR comes from seeds 43 and 44. As in D8.11, this step only orders the two mechanically by the rule and is not evidence that "L2 LR is better calibrated"; the two deviate from 1 by similar amounts in opposite directions (unpenalised LR slightly too extreme, L2 LR slightly toward the mean).
5. **Selected model for CLIN = LR (L2)** (not yet frozen). It differs from the recommended model of definition C (unpenalised LR) only because of the calibration tie-break in step 4; the AUROCs of the two LRs differ by 0.003 (CI crosses 0).
   - Inner tuning chose C = 0.01 in 50/50 folds (grid 0.001–100 and ∞; 0.01 is not at the boundary), which is fairly strong shrinkage. The final model is re-tuned on the full training set with the same procedure as D8.5.
   - The pooled OOF AUROC of 0.8006 is close to the fold mean of 0.8013, with no fold-mean/pooled gap like that of L2 LR in definition C (D8.11).
- **How the recommendation is justified (D8.8 items 6 and 7)**: L2 LR happens to be M\*, but its gap to each of the other 5 models in the equivalence set is ≤ 0.008 with CIs crossing 0, so **no claim of better discrimination** is made. The justification covers only parsimony (a logistic model on the LASSO-selected variables), interpretability and "the two LRs are similarly calibrated". The report must state that steps 2 and 3 are a pre-specified preference for parsimony.

**Tuned RF (corresponding to the D4.7 pre-specified main-analysis model of definition C, reported as usual)**: AUROC 0.7970 ± 0.0282, equivalent to M\*; pooled OOF calibration slope 1.465 (1.34–1.55 across repeats), predictions shrunk toward the mean, in the same direction as definition C (1.48).

**Youden thresholds** (pooled OOF, median over 5 repeats): L2 LR 0.258 (range 0.239–0.277); tuned RF 0.241 (0.233–0.266); unpenalised LR 0.273 (0.252–0.299).

**LASSO selection frequencies** (50 folds; LASSO is run once per fold and shared by all models; `nested_cv_full_selection_clin.csv`):
- 50/50: age, aniongap_min, cci, oasis, apsiii, resp_rate_mean, rdw_mean, chloride_max, obesity;
- 49: bun_min; 47: sapsii; 46: pt_min; 45: nlr; 35: ptt_min; 33: lmr; 31: plr; 29: lods; 28: sii; 22: monocytes_mean; 15: gender_male, calcium_min;
- `wbc_diff_missing`: 13/50.
- Of the 6 comorbidities only obesity is selected (50/50); MI, heart failure, chronic pulmonary disease, renal disease and hypertension are all 0/50 (CCI 50/50).

**Comparison with definition C** (different cohorts, prediction times and candidate variables; unpaired, descriptive only): the nested-CV AUROC of the L2 LR selected for CLIN is 0.801, and that of the unpenalised LR recommended for definition C is 0.786 (D8.11).

**Next step**: run the two sensitivity analyses pre-specified in the D8.3 and D8.4 revisions for L2 LR and the tuned RF (04f `--variant prior_only`, `--variant no_wbc_diff`), then freeze the model and do one confirmatory evaluation on the held-out set.

**Freezing (2026-10-04, 8e-4 step 1, held-out set not yet read)**:
- Frozen specification `results/clinical/frozen_model_clin.json`: fully specifies the fitting procedure of the final model (on the 2,045-patient training set: 20% screening (white-cell differential exempt) → imputation → standardisation → LASSO inner 10-fold seed 42, λ.1se → LR (L2) inner stratified 5-fold GridSearchCV to choose C; the comparator tuned RF uses the same procedure), thresholds LR 0.2578 and RF 0.2414 (pooled OOF Youden in nested CV, median over 5 repeats), and records the hash of the training-set stay_ids, the hash of the candidate matrix (matching the 04f cache meta), the hashes of the src code and the base commit ce548d3.
- `scripts/04h_final_clin.py` fits the model according to the specification; the model is saved in `data/interim/final_clin/final_clin.pkl` (git-ignored), and the fit record in `results/clinical/final_model_clin.json` (including the sha256 of the model file, for 04i to check).
- Result: LASSO λ.1se = 0.0168, the 20% rule dropped no variable, 22 selected (at the upper end of the 13–23 range across nested-CV folds): age, gender_male, obesity, aniongap_min, bun_min, rdw_mean, monocytes_mean, calcium_min, chloride_max, ptt_min, pt_min, resp_rate_mean, sapsii, apsiii, lods, oasis, cci, nlr, plr, lmr, sii, wbc_diff_missing. LR (L2) C = 0.01 (consistent with the 50/50 nested-CV folds); tuned RF: max_depth 10, max_features 2, min_samples_leaf 20. 544 events, EPV = 24.7 (events / selected variables), events / candidate variables = 9.7.
- **Change of order (user decision)**: the "Next step" above said to run the two sensitivity analyses first and then freeze. In practice the model was frozen first and the confirmatory held-out evaluation done, and then the sensitivity analyses were run. Per the D8.3 and D8.4 revisions, the two sensitivity analyses concern only the selected model and the tuned RF, do not take part in model selection and do not change the frozen model, so the order does not affect the confirmatory evaluation.

### D8.15 8e-4 results: confirmatory evaluation on the CLIN held-out set (2026-10-04, run only once)

**Procedure**:
- After the freeze commit 0de268b (`frozen_model_clin.json`, `final_model_clin.json`, `04h_final_clin.py`), `scripts/04i_holdout_clin.py` evaluated the model once on the held-out set (512 patients, 136 deaths, 26.6%). Log `results/logs/04i_holdout_clin.log`. Outputs `results/clinical/holdout_clin.csv`, `holdout_dca_clin.csv`, `coefs_clin.csv`, `shap_clin.csv`.
- **Three safeguards** (checked before reading the held-out set):
  1. The frozen specification and the fit record are both committed and do not differ from HEAD;
  2. The sha256 of the model file, the specification hash, the training-set stay_id hash, the src code hashes, the selected variables, the hyperparameters and the estimator configuration all match the frozen files, and refitting LR (L2) with the preprocessing parameters in the model bundle reproduces its coefficients;
  3. The script refuses to run if any output file already exists (checked again before writing).
- **Pre-run test**: `--smoke` mode follows exactly the same code path, but the evaluation set is replaced by 512 patients drawn stratified from the training set; the held-out set is not read, outputs go to a temporary directory and no metrics are printed. The smoke test passed. Safeguard 1 was also verified in smoke mode (it refused to run after the frozen file was modified, which was then restored with git). The formal run was done only once, without errors.

| Metric | LR (L2, recommended) | Tuned RF (comparator) |
|---|---|---|
| AUROC (DeLong 95% CI) | 0.750 (0.702–0.797) | 0.752 (0.704–0.799) |
| Paired DeLong: LR − RF | −0.002 (−0.021, 0.016), p = 0.83 | — |
| AUPRC (bootstrap 95% CI); positive-rate baseline 0.266 | 0.533 (0.459–0.611) | 0.530 (0.460–0.615) |
| Brier / scaled Brier | 0.165 / 0.153 | 0.164 / 0.161 |
| Calibration intercept / slope | 0.049 / 0.82 | 0.013 / 1.14 |
| Threshold (frozen) | 0.2578 | 0.2414 |
| accuracy / precision / recall / F1 / specificity | 0.693 / 0.449 / 0.684 / 0.542 / 0.697 | 0.645 / 0.411 / 0.779 / 0.538 / 0.596 |
| Proportion classified high risk | 40.4% | 50.4% |
| DCA (0.05–0.60): net benefit above both treat-all and 0 | 53/56 thresholds (0.08–0.60) | 55/56 thresholds (0.06–0.60) |

- Final model: training set of 2,045 patients, 544 events; 56 candidates, 22 selected; EPV 24.7, events / candidate variables 9.7 (same as the freeze record).

**Interpretation (as in D4.16; `coefs_clin.csv`)**:
- Penalised ORs (per SD) of LR (L2, C = 0.01), ordered by |coefficient|: rdw_mean 1.47, cci 1.34, age 1.31, apsiii 1.22, chloride_max 0.85, obesity 0.85, oasis 1.16, resp_rate_mean 1.15, sapsii 1.14, pt_min 1.13, etc.
- Unpenalised LR on the same 22 variables (statsmodels GLM, training set): rdw_mean OR 1.67 (1.47–1.89), age 1.50 (1.30–1.75), cci 1.44 (1.27–1.63), apsiii 1.38 (1.14–1.68), obesity 0.80 (0.71–0.91), chloride_max 0.83 (0.73–0.94), oasis 1.23 (1.04–1.46), resp_rate_mean 1.20 (1.06–1.34), calcium_min 1.16 (1.02–1.32), wbc_diff_missing 1.15 (1.02–1.29), gender_male 1.14 (1.01–1.28), all p < 0.05. sapsii (OR 1.02, p = 0.90), lods, nlr, sii etc. are not significant when the other scores/ratios are held fixed. The largest VIFs are sapsii 4.74, sii 4.00 and nlr 3.56, all < 5.
- The OR of obesity is < 1 (the "obesity paradox" direction) and the OR of chloride_max is < 1; whether these directions agree with definition C is compared in the 8f report.
- SHAP top 10 (LR (L2), held-out set, LinearExplainer, log-odds scale, training set as background): rdw_mean, cci, age, apsiii, obesity, chloride_max, oasis, resp_rate_mean, sapsii, wbc_diff_missing. The top 3 (RDW, CCI, age) are the same as the SHAP top 3 of the definition C recommended model (rdw_mean, cci, age).

**Held-out AUROC lower than the nested-CV estimate** (recorded as is; the checks below are all post-hoc descriptive analyses that change neither the frozen model nor the confirmatory result):
- The held-out AUROC is about 0.05 lower than nested CV: LR (L2) 0.801 → 0.750, tuned RF 0.797 → 0.752. Both models drop together, and the nested-CV estimate lies above the upper end of the held-out 95% CI (LR upper limit 0.797).
- **Does not look like overfitting**: the apparent AUROC of the frozen LR on the training set is 0.817, only 0.016 higher than nested CV; L2 shrinkage is strong (C = 0.01). The nested CV itself already includes variable selection and tuning.
- **Magnitude of sampling variation**: sampling with replacement from the training-set nested-CV pooled OOF predictions (LR (L2), 5 repeats) with the composition of the held-out set (136 deaths / 376 survivors), 2,000 × 5 times, gives a mean AUROC of 0.800, SD 0.022 and 2.5th percentile 0.756; the proportion ≤ 0.750 is 1.4%. This simulation does not include the variation from refitting the model and therefore underestimates the total variation. The held-out result lies in the lower tail of the sampling distribution but is not impossible.
- **Distribution differences**: the largest standardised mean difference of the 22 selected variables between the training and held-out sets is 0.11 (calcium_min), with 3 > 0.1 (calcium_min, lods, rdw_mean); no clear distribution shift. The split is stratified random, so there is no temporal difference.
- **How the conclusion is written**: the confirmatory estimate for CLIN is the held-out AUROC of 0.750 (0.702–0.797). The report also gives the nested CV value of 0.801 and explains the gap, the checks above, and that the gap more likely comes from sampling variation in the 512-patient held-out set (lower tail, about 1–2%) than from model overfitting, though other causes cannot be ruled out. The model is not changed or re-evaluated on this basis. LR (L2) and the tuned RF do not differ on the held-out set (p = 0.83), with similar AUPRC and Brier; the calibration slope is 0.82 for LR (predictions slightly too extreme) and 1.14 for RF.
- **To do**: the two pre-specified sensitivity analyses (prior_only, no_wbc_diff; L2 LR + tuned RF, nested CV); 8f report reorganisation.

### D8.16 8e sensitivity analysis results: CLIN prior_only and no_wbc_diff (2026-10-04)

**Run**: `04f_nested_full.py --track clinical --variant {prior_only, no_wbc_diff} --models tuned:LR,tuned:RF`, i.e. the LR (L2) selected in D8.14 and the tuned RF. 5 × 10 folds, seeds 42–46, outer splits the same as the main analysis. Logs `results/logs/04f_nested_full_clin_{prior_only,no_wbc_diff}.log`, outputs `results/clinical/nested_cv_full_*_clin_{prior_only,no_wbc_diff}.csv`. Both analyses were run after the held-out evaluation (D8.15), use only the training set and do not affect the frozen model (for the change of order see the D8.14 freeze record).
- Neither analysis had a zero-variable fold, and all calibration fits converged.
- prior_only: the 20% rule dropped no variable. no_wbc_diff: every fold drops the 8 white-cell differential variables (about 34% missing).
- ΔAUROC = sensitivity − main analysis, computed pairwise on the same 50 outer splits with the Nadeau–Bengio correction (r = 5, k = 10), from `nested_cv_full_vs_main_clin_*.csv`. AUPRC, Brier and calibration slope are the pooled OOF values per repeat, averaged over the 5 repeats.

| Metric | LR (L2) main analysis | LR (L2) prior_only | LR (L2) no_wbc_diff | Tuned RF main analysis | Tuned RF prior_only | Tuned RF no_wbc_diff |
|---|---|---|---|---|---|---|
| AUROC (fold mean ± SD) | 0.801 ± 0.026 | 0.787 ± 0.029 | 0.802 ± 0.027 | 0.797 ± 0.028 | 0.789 ± 0.030 | 0.795 ± 0.029 |
| ΔAUROC (95% CI) | — | −0.014 (−0.023, −0.005) | +0.001 (−0.005, 0.007) | — | −0.008 (−0.016, −0.0001) | −0.002 (−0.010, 0.005) |
| AUPRC (pooled OOF) | 0.594 | 0.574 | 0.603 | 0.591 | 0.582 | 0.589 |
| Brier (pooled OOF) | 0.151 | 0.155 | 0.150 | 0.155 | 0.156 | 0.155 |
| Calibration slope (pooled OOF) | 1.108 | 1.076 | 1.113 | 1.465 | 1.372 | 1.431 |
| Number of LASSO-selected variables (median, range) | 17 (13–23) | 16 (10–22) | 13 (12–18) | same as left | same as left | same as left |

(LASSO is run only once per fold and shared by all models, so the number of selected variables is the same for the two models.)

**prior_only** (the comorbidities, the CCI without age points and the SAPS-II chronic-disease items all use prior-hospitalization diagnoses only; patients with no prior hospitalization have all comorbidities set to 0; `has_prior_admission` is a candidate variable):
- The discrimination of both models drops slightly, with CIs not crossing 0: LR (L2) −0.014, RF −0.008 (upper bound close to 0). AUPRC also drops (LR 0.594 → 0.574).
- **Changes in selection**:
  - obesity drops from 50/50 to 4/50, the largest change. The obesity information in the CLIN main analysis comes mainly from index-hospitalization codes.
  - cci drops from 50/50 to 46/50; myocardial_infarct rises from 0 to 16/50; `has_prior_admission` is 0/50.
- **Interpretation**: chronic-disease codes in the index hospitalization (after removing the D8.3 acute codes) contribute somewhat to prediction, and using only prior hospitalizations loses about 0.01 AUROC. Whether the main-analysis approach (prior ∪ chronic-disease codes of the index hospitalization) is fully available at the prediction time depends on the assumption that "chronic-disease codes represent conditions that existed before admission" (D8.3 limitation). The report should therefore present prior_only as a more conservative deployment scenario: AUROC about 0.787, still comparable with the definition C recommended model (0.786).
- Under both approaches RDW, age and APS-III are selected in every fold, so the conclusion about the main predictors is unchanged.

**no_wbc_diff** (the 8 white-cell differential variables are dropped within folds under the 20% rule and `wbc_diff_missing` is not used, i.e. D8.4 option 1):
- Discrimination, AUPRC, Brier and calibration are essentially unchanged: LR (L2) Δ +0.001, RF Δ −0.002, both CIs cross 0. The number of selected variables drops from 17 to 13.
- **Interpretation**: the white-cell differential variables (NLR, PLR, LMR, SII etc.) and their missingness indicator bring no detectable incremental discrimination in CLIN. The main value of option 2 is that it keeps the paper's key variables; the more parsimonious option 1 performs the same. As pre-specified, the main analysis and the frozen model do not change, and the report states this as "the white-cell differential can be omitted without loss of performance".

**Youden thresholds** (pooled OOF, median over 5 repeats): prior_only LR 0.253, RF 0.256; no_wbc_diff LR 0.240, RF 0.249 (0.258 and 0.241 in the main analysis).

### D8.17 8f: report reorganisation, CLIN figures and Table 1 (2026-10-04)

**The held-out set was read again for plotting; the results agree item by item and no conclusion changed.**
- `scripts/04j_plots_clin.py` uses the frozen model file (`data/interim/final_clin/final_clin.pkl`, first checking the model-file sha256, specification hash and selected variables as in 04i safeguard 2) to predict the CLIN held-out set (512 patients) again, without refitting anything. The predictions are saved only in `data/interim/final_clin/holdout_pred_clin.parquet`.
- With the same code as 04i, all 33 columns of `holdout_clin.csv` and the 112 rows of `holdout_dca_clin.csv` are recomputed and compared after rounding to the same 6 decimals as when written: all differences are 0 (the rule is < 1e-9, otherwise the script exits with an error and draws nothing).
- Figures: `figC1_roc_clin_holdout.png` (ROC), `figC2_calibration_clin_holdout.png` (calibration curve, predicted-risk deciles), `figC3_dca_clin_holdout.png` (DCA 0.05–0.60), `figC4_calibration_clin_nested_oof.png` (training-set nested-CV pooled OOF calibration, 5 repeats, from the cache, as a comparison).
- The same script also outputs `results/clinical/holdout_gap_diagnostics_clin.csv`: the three post-hoc descriptive checks of D8.15 (apparent training AUROC 0.817; 10,000 samples with replacement from the OOF predictions with the held-out composition, mean 0.800, SD 0.021, 2.5th percentile 0.755, proportion ≤ 0.750 1.4%; maximum |SMD| of the 22 selected variables 0.11 (calcium_min), 3 > 0.1), with the same code and seeds as D8.15 and values consistent with D8.15. This way these numbers in the report are also generated from result tables.
- **Correction**: in D8.15 the upper limit of the 95% CI of the held-out LR (L2) AUROC was originally written as 0.798; the actual value is 0.797467, so it should be **0.797**; the upper limit of the CI of the paired DeLong difference was originally written as 0.017; the actual value is 0.016467, so it should be **0.016** (both original values came from double rounding of a 4-decimal printout). D8.15 and progress.md have been corrected; all other values were checked item by item and are correct.

**CLIN Table 1** (`scripts/03c_table1_clin.py`, descriptive):
- The whole cohort of 2,557 patients (1,877 survivors, 680 deaths) grouped by one-year outcome, using observed values before imputation, with the same rules as D3.5 and D3.6; variables are the 55 CLIN predictors, `wbc_diff_missing` and `has_prior_admission`. Output `results/clinical/table1_clin.csv`; 42 variables have p < 0.05.
- `wbc_diff_missing` and one-year mortality (`wbc_diff_missing_mortality_clin.csv`): 28.6% (250/873) in the = 1 group, 25.5% (430/1,684) in the = 0 group, χ² p = 0.092.
- Source of obesity codes (`obesity_source_clin.csv`): the 416 patients with obesity coded only in the index hospitalization have a mortality of 18.8%; the 443 with an obesity code in a prior hospitalization 26.4%; the 1,698 with no obesity code 28.6%. Used for the discussion in report 17.3 item 14 that "obesity coding may be subject to illness-related bias" (worded as "suggests").

**Report reorganisation** (D8.9 and its revisions, outline confirmed by the user on 2026-10-04):
- Structure: Sections 0–1 (summary of conclusions; data, tools and study design); Part I replication of the paper (definition C, Sections 2–9); Part II CLIN (Sections 10–14); Part III comparison with the paper, limitations, future work (Sections 15–18); Appendix A (detailed tables from the single validation set of stages 4–7, with CSV links), Appendix B (file list).
- The main performance of definition C becomes the full nested CV (D8.11); 0.803 is only the secondary held-out estimate, noted as having taken part in post-hoc selection (D8.12); the justification of the recommendation covers only parsimony, interpretability and EPV, and calibration is described as "the two LRs are similar"; that D8.8 is not blinded for definition C is stated in Sections 1, 6.2 and 17.2.
- Per the D8.1/D8.7 revisions, definitions A/B/D list only the flow and mortality, "the choice of definition does not change the conclusions" is removed, and the report says "this study uses definition C". The sentence "definitions A and C side by side" in 8f of `plans/stage8_prompt.md` is no longer carried out, per that revision.
- Stage 4–7 content kept in the main text: the post-hoc selection process (D4.15, 6.2), the comparison table of the two hypertension definitions (3.3), the unit-mixing diagnostic table (3.2), and the overfitting explanation of the RF training AUC of 0.999 with the overfitting table (6.5); the other detailed tables (AUC, paired DeLong, thresholds, calibration, "LASSO-only nested" CV) move to Appendix A.
- The CBC item (17.2 item 8) is rewritten as requested by the user: no inference from penalised ORs; instead a descriptive comparison of one-year mortality between the two `wbc_diff_missing` groups, citing the unpenalised-model OR of 1.15 (1.02–1.29) from `coefs_clin.csv`, noting that the variable is selected in only 13/50 folds, worded as "suggests".
- Future work: external validation in eICU is removed and replaced by temporal validation by anchor_year_group and a search for external data with one-year follow-up; "repeated nested CV" is done and removed; the original "CBC missingness sensitivity analysis" item is covered by CLIN and removed.
- `05_report.py` reads the results from `results/replication/` and `results/clinical/`, re-applies the D8.8 model selection when generating the report and asserts the result (unpenalised LR for definition C, L2 LR for CLIN); report generation fails if the condition does not hold. All stage-8 numbers in the text are computed from result tables.
- **Further changes (2026-10-04, proposed by the user after reviewing the report)**:
  - The coefficient table in Section 13 gets a new column "unpenalised LR OR (yes vs no, 95% CI)": for binary variables OR = exp(β_per_SD / SD), with the CI converted the same way; "—" for continuous variables. The binary ORs cited in the text now use "yes vs no": obesity 0.63 (0.49–0.81), wbc_diff_missing 1.33 (1.04–1.71), gender_male 1.31 (1.03–1.66).
  - Section 6.3 becomes "differs from the nested-CV estimate by ≤ 0.017 (LR 0.803 vs 0.786, RF 0.793 vs 0.788)".
  - Section 6.1 explains: stage 7 had only 1 repeat (seed 42); in the full nested CV the seed-42 repeat gives unpenalised LR 0.781 and tuned RF 0.786, consistent with stage 7; unpenalised LR ranges 0.781–0.792 across the 5 repeats, so the 0.781 → 0.786 difference comes mainly from variation between repeats.
  - The report title becomes "Replication and clinical-improvement report: ... machine-learning prediction of one-year mortality risk" (English version updated to match).

### D8.18 Post-hoc addition: 95% CIs for the calibration intercept and slope on the CLIN held-out set (2026-10-23, reviewer request)

- Post-hoc descriptive analysis (`scripts/04l_calibration_ci_clin.py` → `results/clinical/holdout_calibration_ci_clin.csv`). The frozen model and the confirmatory results such as `holdout_clin.csv` are unchanged and 04i is not rerun; the code that loads the model and rebuilds the held-out predictions is the same as 04i (guards 1/2, `prep()`, the LR refit consistency check), with guard 3 ("refuse if outputs exist") removed. Point estimates agree with `holdout_clin.csv` (tolerance 1e-6).
- Intervals: Wald (slope: GLM of y on logit(p); intercept: constant-only GLM with offset logit(p)) and stratified bootstrap (2,000 replicates, seed 42, events/non-events resampled separately, only the predictions resampled, non-converged replicates skipped and counted). Tests: two-sided Wald tests of slope = 1 and intercept = 0.
- For the intercept the Wald interval is the reference: the stratified bootstrap fixes the number of events, so it excludes sampling variation in the outcome rate and underestimates the uncertainty of the intercept (calibration-in-the-large); it is kept for consistency with the bootstrap method used for AUPRC.
- Results: LR (L2) slope 0.82 (Wald 0.62–1.03, p = 0.09), intercept +0.049 (Wald −0.171 to +0.270, p = 0.66); RF slope 1.14 (0.86–1.42, p = 0.32). Neither deviation differs statistically from perfect calibration; the recommendation to "recalibrate before deployment" is unchanged (the reason is instability across samples, not a single point estimate).
- Report: `05_report.py` and the English template get `calci_text()` (inserted into Section 12.5 when the file exists); because the English Part IV is spliced by hand, 05_report was not rerun this time; instead the text rendered by the same function was inserted into `report.md` / `report_en.md`.
