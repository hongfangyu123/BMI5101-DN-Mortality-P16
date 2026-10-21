# Number ledger for paper_v2_en.md

Every number in `docs/paper_v2_en.md` with its location and source. "report §x" = `results/report_en.md` section x. CSV/JSON paths are relative to `results/`. "PDF" = the benchmark paper `docs/topic16.pdf`. Implementation constants in Methods that report_en.md does not state are traced to the code (file:line) that the Methods describe.

Not listed (not data): citation numbers in brackets, section/table/figure/item labels (e.g. "Section 3.3", "I5", "Table 1", "S9"), years and page numbers in the reference list, and the publication year of the benchmark (2025).

## Abstract, Introduction

| Number | Section | Source |
|---|---|---|
| 12 (predictors selected by LASSO in the benchmark) | Abstract; 1.2 | report §5 table (paper λ.1se, 12 variables) |
| 55 (candidate predictors) | Abstract; 1.2 | report §3 ("55 predictors") |
| 0.780 (benchmark RF validation AUC) | Abstract; 1.2; 1.3 Table 1 (I6) | report §15 ("RF (validation 0.780)"); report App. A.1 |
| 263 (benchmark validation set) | Abstract; 1.3 Table 1 (I5) | PDF, Abstract and Methods ("validation sets (263, 20%)"; "internal validation cohort (n = 263)") |
| 24 h (landmark) | Abstract; 1.3; 2.2; 2.3 | report §10 ("Prediction time = 24 h after ICU admission") |
| 365 days | Abstract; 1.3 (I4); 2.3 | report §1 Outcome (D8.10) |
| 5 × 10 (repeats × outer folds) | Abstract; 2.5 | report §6.1 Design |
| 0.999 (RF training AUC) | Abstract; 1.3 (I6) | report §6.5 overfitting table (RF (R defaults) train 0.999); report App. A.1 (paper as reported 0.999) |
| 0.724–0.767 (benchmark RF configurations, nested CV) | Abstract | report §6.1 table: RF (R defaults) (paper) 0.724, RF (paper) 0.767 |
| 0.791 (best model, reference cohort) | Abstract | report §6.1 table: XGBoost (paper) 0.791 |
| 2,557 (landmark cohort) | Abstract; 2.2 Fig. 1 | report §10 flow table step [5]; clinical/cohort_flow_clin.csv |
| 26.6% (landmark one-year mortality) | Abstract; 2.2 Fig. 1 | report §10 flow table step [5] |
| six (equivalent configurations) | Abstract | report §11 "Applying D8.8" step 2 (six models listed) |
| 0.801 (L2 LR nested-CV AUROC) | Abstract | report §11 table LR (L2) 0.801 ± 0.026 |
| 512 (held-out set) | Abstract; 2.2 Fig. 1 | report §10 Split; §12.3 |
| 0.750 (0.702–0.797) held-out AUROC | Abstract | report §12.3 table; clinical/holdout_clin.csv |
| 0.82 (held-out calibration slope) | Abstract | report §12.3 table; §12.5 |
| 95% CI | Abstract; 2.5; 2.7 | definitional (confidence level) |
| eight (complete-blood-count items) | 1.3 (I2); 2.2 | report §16 ("8 items (WBC, Hb, PLT, RBC, RDW + three absolute white-cell counts)") |
| about one third (excluded by CBC criterion) | 1.3 (I2) | report §2.1 (CBC retention 64.3%; differential missing 35.6%) |
| 48 h | 1.3 (I1); 2.2 | report §1, §2.1 step [3] |
| seven issues | 1.3 | count of rows I1–I7 in Table 1 (outline_v2) |
| three contributions | 1.4 | count (outline_v2 §1.4) |
| ICD-9 401–405; ICD-10 I10–I13, I15, I16 | 1.3 (I4, implicit); 2.3 | report §3.3 |

## Methods

| Number | Section | Source |
|---|---|---|
| version 3.1; 2008–2022 | 2.1 | report §15 table (Data release) |
| v2.x; 2008–2019 | 2.1 | report §15 table; PDF Methods ("covers the period from 2008 to 2019") |
| Python 3.11, DuckDB 1.5.6, scikit-learn 1.9, xgboost 3.4, shap 0.51 | 2.1 | report §1 |
| seed 42 | S9 | report §1 |
| four model families | 2.1 | report §6 (RF, XGBoost, SVM, LR) |
| 18 years (landmark age criterion, "18 years or older") | 2.2; Fig. 1 | report §10 Inclusion ("age ≥ 18") |
| 80% / 20% split | 2.2; 2.7 | report §10 Split ("stratified 80/20"); §4; §9.2 |
| age over 18 years (reference cohort) | 2.2; Fig. 1 | report §2.1 step [4] |
| 8,992 | 2.2 Fig. 1 | report §2.1 step [1] |
| 3,189 | 2.2 Fig. 1 | report §2.1 step [2]; §10 steps [1]–[3] |
| 3,046 (twice) | 2.2 Fig. 1 | report §2.1 steps [3], [4] |
| 1,960; 494; 25.2% | 2.2 Fig. 1 | report §2.1 step [5]; §2.2 definition C; §2.3 |
| 1,568 / 392 | 2.2 Fig. 1 | report §4 (training 1,568 / validation 392) |
| 594 → 2,595 | 2.2 Fig. 1 | report §10 flow table step [4] |
| 38 → 2,557; 680 deaths | 2.2 Fig. 1 | report §10 flow table step [5] |
| 2,045 / 512 | 2.2 Fig. 1 | report §10 Split |
| six comorbidities | 2.3 | report §3.1 table (6 comorbidity rows) |
| 31 laboratory summaries | 2.3 | report §3.1 table: count of laboratory rows glucose_mean … inr_max = 31 |
| four vital signs | 2.3 | report §3.1 table (heart_rate, mbp, resp_rate, temperature) |
| six severity scores | 2.3 | report §3.1 table (sapsii, apsiii, lods, oasis, gcs, cci) |
| six inflammatory ratios | 2.3 | report §3.1 table (nlr, plr, lmr, pnr, pwr, sii) |
| 6 h before to 24 h after ICU admission (labs); first 24 h (vitals) | 2.3 | report §16 ("labs [intime−6h, intime+24h], vital signs [intime, intime+24h]") |
| 10⁹/L | 2.3 | report §3.2 (K/uL = 10⁹/L) |
| three absolute cell counts; eight differential-derived variables | 2.3; 2.4 | report §10 White-cell differential |
| 56 candidate predictors | 2.3 | report §11 ("56 candidate variables (55 + wbc_diff_missing)") |
| 20% missingness rule | 2.4; 2.5 | report §4; §10; §6.1 |
| 1,000 (intercept_scaling) | S9 | code: src/lasso.py:50 (`intercept_scaling=1e3`) |
| C = 1/(nλ) | 2.4 | code: src/lasso.py:3–4, 49 |
| 100 λ values; 10⁻⁴ λ_max | S9 | code: src/lasso.py:6, 63–66 (`n_lambda=100`, `min_ratio=1e-4`) |
| 10-fold LASSO CV; SE = SD/√10 | 2.4; 2.5 | report §6.1 (inner 10 folds); code: src/lasso.py:73 |
| C grid {0.001, 0.01, 0.1, 1, 10, 100, ∞} | 2.4 | report §11 ("grid 0.001–100 and ∞"); code: src/models.py:39 |
| λ₂ = 1/(nC) (L2 conversion) | 2.4 | derived from scikit-learn objective (code: src/models.py:38, `LogisticRegression(l1_ratio=0.0)`) |
| B = 500 (tuned RF trees) | 2.4 | code: src/models.py:31 |
| γ = 0; λ = 1 (XGBoost) | 2.4 | code: src/models.py:20 (paper `gamma=0`); xgboost default `reg_lambda=1` (not tuned, src/models.py:33–35) |
| nine configurations | 2.4 | report §6.1 ("All 9 models") |
| nine fully grown trees (RF, R defaults) | 2.4 | report §6.5 ("9 fully grown trees, mtry=2") |
| inner stratified 5-fold tuning | 2.4; 2.5 | report §6.1 |
| 10 outer folds, 5 repeats, J = 50 | 2.5 | report §6.1 ("outer stratified 10 folds × 5 repeats"; "50 outer folds") |
| 49 degrees of freedom (J − 1) | 2.5 | code: src/metrics.py:98 (`df = k * r - 1`) |
| 0.975 (t quantile) | 2.5 | definitional (two-sided 95%) |
| 200 null datasets; 4.0% (corrected) vs 38.0% (uncorrected) rejection rates; 10-fold, 5-repeat design | 2.5 | results/progress.md line 129 ("200 个数据集，10 × 5 CV"; 0.040 vs 0.380); not in report_en.md, source approved by the authors |
| 0.01; 0.02 (equivalence margins) | 2.5 | report §6.2 Rule |
| 5 repeats (thresholds, rule 4) | 2.5; 2.6 | report §6.2; §11 |
| 2,000 bootstrap resamples | 2.5 | code: scripts/04i_holdout_clin.py:13, 153 (report §12.3 says "bootstrap 95% CI" without the count) |
| a = 0, b = 1 (perfect calibration) | 2.6 | definitional |
| 56 thresholds; 0.05–0.60; step 0.01 | 2.6 | report §7, §12.3 ("0.05–0.60", "/56 thresholds"); code: scripts/04f_nested_full.py (`np.arange(0.05, 0.6001, 0.01)`) |
| 1.96 (Wald) | 2.7 | definitional |
| 2020–2022 | 2.7 | report §9.2 |
| 80/20 (new split) | 2.7 | report §9.2 |

## Results

| Number | Section | Source |
|---|---|---|
| 8,992; 3,189 | 3.1 | report §2.1 steps [1], [2] |
| 2,557; 680; 26.6% | 3.1; 4.x | report §10 flow step [5] |
| 594; 24 h; 38 | 3.1 | report §10 flow steps [4], [5] |
| 1,960; 494; 25.2% | 3.1; 3.2; 4.2 | report §2.1 step [5]; §2.3 |
| 95.5% vs 95.4% (LOS retention) | 3.1 | report §2.1 |
| 64.3% vs 64.2% (CBC retention) | 3.1 | report §2.1 |
| 12.6% admitted 2020–2022 | 3.1 | report §2.1 |
| 35.6% (missing differential); 1% (core CBC missing) | 3.1 | report §2.1 |
| 873 (34.1%) | 3.1; 4.2 | report §10 White-cell differential; clinical/table1_clin.csv |
| 28.6% (250/873) vs 25.5% (430/1,684); χ² p = 0.09 | 3.1; 4.2 | report §17.2 item 8; clinical/wbc_diff_missing_mortality_clin.csv |
| 1,452 (56.8%) prior admission | 3.1 | report §10 Comorbidities |
| Table 2: all values (n 2,557 / 1,877 / 680; medians, IQRs, counts, %, p) | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), rows age, gender_male, congestive_heart_failure, renal_disease, obesity, cci, rdw_mean, chloride_max, aniongap_min, bun_min, lymphocytes_mean, resp_rate_mean, sapsii, apsiii, oasis, wbc_diff_missing, has_prior_admission; trailing ".00" dropped |
| 34.1% missing (lymphocytes footnote) | 3.1 Table 2 | report §10 Table 1, lymphocytes_mean missing 34.1 |
| 25.2% vs 38.2% (mortality) | 3.2; 4.3 | report §2.3 table |
| 7.1% in-hospital | 3.2 | report §2.3 table |
| 42.8% (any dod) | 3.2; 4.2 | report §2.3 table |
| 49%; 365 days | 3.2 | report §2.3 bullet |
| 3.69 (1.27–118.32); 23.90 (8.63–831.30); 2.00 (0.60–43.64) | 3.2 | report §3.2 |
| 1.24 (0.77–1.91); 8.59 (5.99–12.57); 0.55 (0.35–0.84) | 3.2 | report §3.1 table (lymphocytes/neutrophils/monocytes_mean) |
| three cell counts | 3.2 | report §3.2 |
| NLR 6.82 vs 6.80; SII 1,336 vs 1,342 | 3.2 | report §3.2 |
| LMR 2.31 vs 2.42 | 3.2 | report §3.1 table (lmr) |
| PLR 159 vs 36; PNR 22 vs 6.6 | 3.2 | report §3.2 |
| 24.00; 3.67 (mixed PLR, PNR) | 3.2 | report §3.2 diagnostic table (PLR mixed, PNR mixed medians) |
| second (neutrophils rank in benchmark); five most important | 3.2 | report §8.2 table |
| 84.6% vs 37.3%; 35.4% | 3.2 | report §3.3; §3.1 table |
| OR 1.07 (0.80–1.42) | 3.2; 4.2 | report §3.3 table |
| OR 0.49 (0.39–0.62) | 3.2; 4.2 | report §3.3 table |
| 964; 31.3%; OR 1.91 (1.55–2.35) | 3.2 | report §3.3 table |
| 12 variables; λ = 0.02667 vs 0.02687 | 3.3 | report §5 table |
| seven of 12 overlap; 11 of 12 at λ.min | 3.3 | report §5 |
| six equivalent; 0.786–0.791 | 3.3 | report §6.1 bullets |
| Table 3: all values | 3.3 Table 3 | report §6.1 table; replication/nested_cv_full_summary_C.csv, replication/nested_cv_full_pairwise_C.csv (ΔAUROC vs XGBoost (paper), rounded to 3 dp); benchmark validation AUCs 0.732/0.773/0.780/0.679 from report App. A.1 "paper (as reported)" |
| 1,568 | 3.3 Table 3 | report §4 |
| 0.252 (AUPRC baseline) | 3.3 | report §6.1 |
| 392; 0.803 (0.753–0.853); slope 0.96 | 3.3 | report §6.3 table; replication/holdout_C.csv |
| nine fully grown trees; 0.999; 0.756; 0.724; 0.15 | 3.3 | report §6.5 overfitting table and bullet; §6.1 table (0.724, slope 0.154) |
| 0.887 (literal RF training AUC) | 3.3 | report §6.5 overfitting table |
| 0.011 (0.793 vs 0.781); 0.008–0.030 | 3.3; 4.2 | report App. A.5 table (optimism column: min 0.008 XGBoost paper, max 0.030 RF R defaults) |
| at most 0.001 (preprocessing leakage) | 3.3; 4.2 | report §4 ("differ by ≤ 0.001") |
| 7.04% (max missingness) | 3.3 | report §3 ("all missing rates are ≤ 7.04%") |
| smallest adjusted p = 0.13 | 3.3 | report §6.5 (Holm, smallest adjusted p = 0.13) |
| 13–23 of 56; median 17 | 3.4; 3.6 | report §11 |
| 50 folds (nine variables selected in all 50) | 3.4 | report §11 selection frequency (age, aniongap_min, cci, oasis, apsiii, resp_rate_mean, rdw_mean, chloride_max, obesity: 50) |
| 13 of 50 (wbc_diff_missing) | 3.4; 4.2 | report §11 |
| six comorbidities; only obesity | 3.4 | report §11 ("Of the 6 comorbidities only obesity is selected") |
| 0.794–0.801 | 3.4 | report §11 table (range of equivalent models) |
| Table 4: all values | 3.4 Table 4 | report §11 table; clinical/nested_cv_full_summary_clin.csv; clinical/nested_cv_full_pairwise_clin.csv (ΔAUROC vs LR (L2), sign flipped, rounded to 3 dp: LR unpenalised 0.00275 (−0.00098, 0.00648) → 0.003 (−0.001, 0.006); RF R defaults hi 0.083497 → 0.083; RF tuned slope 1.464927 → 1.46) |
| 2,045 | 3.4 Table 4 | report §10 Split |
| 0.266 (AUPRC baseline) | 3.4; Table 5 | report §11; §12.3 |
| mean |ln slope| 0.102 vs 0.109 | 3.4 | report §11 (0.1024 vs 0.1085) |
| 0.003 (AUROC difference between LRs) | 3.4 | report §11 (0.0028) |
| 1.11 and 0.90 | 3.4; 3.5; 4.2; 4.5 | report §11 table (1.108, 0.897) |
| three of five repeats | 3.4 | report §11 ("closer to 1 in 3/5 repeats") |
| C = 0.01; all 50 folds | 3.4 | report §11; §12.2 |
| 544 events; 22 variables; λ.1se = 0.0168; 24.7 | 3.4 | report §12.2 |
| 512; 136 deaths | 3.4; 4.5 | report §12.3 |
| Table 5: all values | 3.4 Table 5 | report §12.3 table; clinical/holdout_clin.csv (thresholds 0.2578 → 0.258, 0.2414 → 0.241; flagged 40.43% → 40.4%, 50.39% → 50.4%) |
| −0.002 (−0.021, 0.016), p = 0.83 | 3.4 | report §12.3 table |
| about 0.05 below | 3.4; 4.2 | report §12.4 |
| 0.797 to 0.752 | 3.4 | report §12.4 |
| 0.817; 0.016 | 3.4 | report §12.4 |
| 10,000 resamples; 136 / 376; mean 0.800; SD 0.021; 2.5th percentile 0.755; 1.4%; 0.750 | 3.4 | report §12.4 |
| 0.11 (max SMD); 22 selected variables | 3.4 | report §12.4 |
| 0.75–0.80 | 3.4; 4.3; 5 | report §12.4 ("between 0.75 and 0.80"); §15 |
| 1.11; 0.90 (OOF slopes) | 3.5 | report §11 table |
| 0.82; +0.049 | 3.5 | report §12.3; §12.5 |
| 1.46 (landmark tuned RF OOF slope); 1.48 (reference) | 3.5 | report §11 bullet ("1.46"); §6.1 table (1.483) |
| 1.14 (RF held-out slope) | 3.5 | report §12.3 |
| 0.18; 0.15 (R-default RF slopes) | 3.5; 4.2 | report §11 table (0.180); §6.1 table (0.154) |
| 0.165 / 0.153; 0.164 / 0.161 | 3.5 | report §12.3 |
| 53 of 56 (0.08–0.60); 55 of 56 (0.06–0.60) | 3.5; 4.2 | report §12.3 table |
| 56 of 56 (reference, nested CV and held-out) | 3.5 | report §7; §6.3 table |
| 0.258; 40.4%; 44.9% (= precision 0.449); 0.684; 0.697 | 3.5 | report §12.3 table |
| 0.5; 16.3%; 0.44 | 3.5 | report §6.4 |
| precision 0.735; recall 0.858 (benchmark RF) | 3.5 | PDF Results p. 6 ("accuracy, precision, recall, and F1 scores of 0.721, 0.735, 0.858, and 0.792") |
| Table 6: all values | 3.6 Table 6 | report §9.1 table; §14 table; replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_prior_only.csv (LR hi −0.004507 → −0.005; RF hi −0.00007 → −0.0001 as in report); clinical/nested_cv_full_vs_main_clin_no_wbc_diff.csv |
| 843; 304 of 611 | 3.6; 4.2 | report §9.1 |
| 0.014 | 3.6; 4.2 | report §14 |
| 50 to 4 of 50 (obesity) | 3.6; 4.2 | report §14 prior_only bullet |
| about 0.787 | 3.6; 4.5 | report §14; §17.3 item 13 |
| eight differential variables; 17 to 13 | 3.6 | report §14 table and no_wbc_diff bullet |
| 247; 2020–2022; 80/20 | 3.6 | report §9.2 |
| 0.799; 0.802; 0.803; 0.793 | 3.6 | report §9.2 table |
| two most important (RDW, CCI) | 3.6 | report §9.2 table (LR SHAP top 5) |
| OR 1.67 (1.47–1.89); 1.50 (1.30–1.75); 1.44 (1.27–1.63); 1.38 (1.14–1.67); 1.23 (1.04–1.46); 1.20 (1.06–1.34); 1.16 (1.02–1.32) | 3.7 | report §13 table and bullet |
| 0.82 (0.73–0.94); 0.63 (0.49–0.81) | 3.7 | report §13 |
| 1.31 (1.03–1.66); 1.33 (1.04–1.71) | 3.7; 4.2 | report §13 bullet |
| VIF < 5; 4.74 | 3.7 | report §13 Collinearity |
| Table 7: ranks 1–5 | 3.7 Table 7 | report §8.2 table (paper RF, unpenalised LR, tuned RF); §13 SHAP (LR (L2) top 5); §15 |

## Discussion and Conclusion

| Number | Section | Source |
|---|---|---|
| seven issues | 4.1 | Table 1 |
| 24 h | 4.1 | report §10 |
| third of patients (34.1%) | 4.1 | report §10 |
| 0.801; 0.750 | 4.1; 4.2; 4.3 | report §11; §12.3 |
| 0.999; nine fully grown trees | 4.1 | report §6.5 |
| Table 8 values: 0.801; 0.786–0.791; 873 (34.1%); 28.6% vs 25.5%; p = 0.09; OR 1.33 (1.04–1.71); 13 of 50; +0.001 (−0.005, 0.007); 843; 304 of 611; +0.001; 0.014; 50 to 4 of 50; 25.2%; 42.8%; OR 0.49 vs 1.07; 0.008–0.030; 0.011; 0.001; 0.05; 0.724; 0.767; 0.791; 0.788; 0.15; 1.11; 0.82; 53 of 56 | 4.2 Table 8 | as listed for the same numbers in Results above (report §§2.3, 3.3, 6.1, 6.5, 9.1, 10, 11, 12.3, 12.4, 13, 14, 17.2; App. A.5) |
| 0.005 (reference); 0.008 (landmark) | 4.3 | report §6.1 ("each within 0.005 of the best model"); §11 ("each within 0.0077 of M*") |
| 263 | 4.3 | PDF Abstract/Methods |
| ±0.05 | 4.3 | report §6.5 ("95% CIs of about ±0.05"); §17.1 |
| 0.723–0.837 | 4.3 | report App. A.1 (paper as reported, RF validation CI) |
| 0.750 vs 0.780 | 4.3 | report §12.3; App. A.1 |
| four reasons | 4.3 | count |
| 38.2% | 4.3 | report §2.3 |
| 0.702–0.797 | 4.3 | report §12.3 |
| 0.786–0.791; 0.801 | 4.3 | report §6.1; §11 |
| 0.75–0.80 | 4.3; 5 | report §12.4; §15 |
| 18.8% (n = 416); 26.4% (n = 443); 28.6% (n = 1,698) | 4.4 | report §17.3 item 14 (clinical/obesity_source_clin.csv) |
| 136 events | 4.5 | report §12.3; §17.1 |
| 0.787 | 4.5 | report §17.3 item 13 |
| 1.11; 0.82 | 4.5 | report §12.5 |
| 24 h (first-24-h trajectory features) | 4.6 | outline_v2 §1 (4.6) / draft v0.1 §2.9 (descriptive, not a result) |
| 24 h; AUROC 0.75–0.80 | 5 | report §10; §12.4 |

## Supplementary list

| Number | Section | Source |
|---|---|---|
| four cohort definitions | S1 | report §2.2 (definitions A–D) |
| nine configurations | S2 | report §6.1 |
| seed 42 | S9 | report §1 |
| 100 values; 10⁻⁴ λ_max | S9 | code: src/lasso.py:6, 63–66 |
| 1,000 (intercept feature) | S9 | code: src/lasso.py:50 |

## Table cells (explicit, row by row)

| Number(s) | Section | Source |
|---|---|---|
| 67; 58; 75; 65; 57; 73; 71; 63; 79; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Age, years" |
| 1,520; 59.44; 1,097; 58.44; 423; 62.21; 0.087 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Male sex" |
| 1,178; 46.07; 775; 41.29; 403; 59.26; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Congestive heart failure" |
| 1,299; 50.80; 877; 46.72; 422; 62.06; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Renal disease" |
| 859; 33.59; 664; 35.38; 195; 28.68; 0.002 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Obesity (coded)" |
| 5; 4; 7; 5; 3; 6; 6; 5; 8; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "CCI (without age)" |
| 14.65; 13.60; 16.20; 14.30; 13.40; 15.65; 15.81; 14.49; 17.63; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "RDW, %" |
| 105; 100; 109; 106; 101; 109; 103; 98; 108; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Chloride (max), mmol/L" |
| 13; 10; 15; 12; 10; 14; 14; 11; 17; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Anion gap (min), mmol/L" |
| 23; 15; 39; 21; 14; 34; 32; 20; 53; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Urea nitrogen (min), mg/dL" |
| 1.27; 0.79; 1.95; 1.39; 0.87; 2.08; 0.95; 0.60; 1.47; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Lymphocytes, 10⁹/L †" |
| 18.28; 16.43; 20.62; 18.08; 16.32; 20.20; 19.12; 16.99; 21.75; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Respiratory rate, /min" |
| 36; 29; 45; 34; 27; 42; 42; 35; 52.25; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "SAPS-II" |
| 45; 35; 57; 42; 34; 53; 54; 43; 67; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "APS-III" |
| 30; 25; 36; 30; 24; 35; 33.5; 27; 40; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "OASIS" |
| 873; 34.14; 623; 33.19; 250; 36.76; 0.092 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "No white-cell differential" |
| 1,452; 56.79; 1,028; 54.77; 424; 62.35; 0.001 | 3.1 Table 2 | report §10 Table 1 (clinical/table1_clin.csv), row "Prior admission" |
| 0.732; 0.791; 0.035; 0.559; 0.150; 0.96 | 3.3 Table 3 | report §6.1 table, row "XGBoost (benchmark parameters)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.790; 0.035; 0.558; 0.150; 0.96; 0.000; 0.005; 0.006 | 3.3 Table 3 | report §6.1 table, row "XGBoost (tuned)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.788; 0.036; 0.536; 0.154; 1.19; 0.002; 0.009; 0.014 | 3.3 Table 3 | report §6.1 table, row "LR (L2, tuned)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.788; 0.036; 0.548; 0.154; 1.48; 0.002; 0.006; 0.011 | 3.3 Table 3 | report §6.1 table, row "RF (tuned)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.788; 0.036; 0.541; 0.153; 0.90; 0.003; 0.008; 0.014 | 3.3 Table 3 | report §6.1 table, row "SVM (tuned)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.773; 0.786; 0.035; 0.550; 0.151; 0.85; 0.005; 0.007; 0.016 | 3.3 Table 3 | report §6.1 table, row "LR (unpenalised)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.780; 0.767; 0.038; 0.519; 0.156; 0.94; 0.024; 0.006; 0.041 | 3.3 Table 3 | report §6.1 table, row "RF (benchmark, literal parameters)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.780; 0.724; 0.039; 0.440; 0.171; 0.15; 0.066; 0.043; 0.090 | 3.3 Table 3 | report §6.1 table, row "RF (benchmark, R defaults)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.679; 0.723; 0.044; 0.497; 0.163; 0.96; 0.068; 0.042; 0.093 | 3.3 Table 3 | report §6.1 table, row "SVM (benchmark parameters)" (replication/nested_cv_full_summary_C.csv; ΔAUROC from replication/nested_cv_full_pairwise_C.csv); benchmark AUC from report App. A.1 |
| 0.801; 0.026; 0.594; 0.151; 1.11 | 3.4 Table 4 | report §11 table, row "LR (L2, tuned)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.799; 0.026; 0.592; 0.151; 0.90; 0.003; 0.001; 0.006 | 3.4 Table 4 | report §11 table, row "LR (unpenalised)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.798; 0.028; 0.584; 0.153; 0.92; 0.004; 0.002; 0.009 | 3.4 Table 4 | report §11 table, row "SVM (tuned)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.797; 0.028; 0.591; 0.155; 1.46; 0.004; 0.005; 0.014 | 3.4 Table 4 | report §11 table, row "RF (tuned)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.795; 0.028; 0.596; 0.152; 0.99; 0.006; 0.004; 0.015 | 3.4 Table 4 | report §11 table, row "XGBoost (tuned)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.794; 0.029; 0.597; 0.152; 1.02; 0.008; 0.002; 0.017 | 3.4 Table 4 | report §11 table, row "XGBoost (benchmark parameters)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.781; 0.029; 0.560; 0.157; 1.09; 0.020; 0.008; 0.033 | 3.4 Table 4 | report §11 table, row "RF (benchmark, literal parameters)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.756; 0.035; 0.550; 0.161; 0.98; 0.046; 0.027; 0.065 | 3.4 Table 4 | report §11 table, row "SVM (benchmark parameters)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.740; 0.037; 0.488; 0.170; 0.18; 0.061; 0.039; 0.083 | 3.4 Table 4 | report §11 table, row "RF (benchmark, R defaults)" (clinical/nested_cv_full_summary_clin.csv; ΔAUROC from clinical/nested_cv_full_pairwise_clin.csv) |
| 0.750; 0.702; 0.797; 0.752; 0.704; 0.799 | 3.4 Table 5 | report §12.3 table, row "AUROC (95% CI)" (clinical/holdout_clin.csv) |
| 0.533; 0.459; 0.611; 0.529; 0.460; 0.615 | 3.4 Table 5 | report §12.3 table, row "AUPRC (95% CI; baseline 0.266)" (clinical/holdout_clin.csv) |
| 0.165; 0.153; 0.164; 0.161 | 3.4 Table 5 | report §12.3 table, row "Brier / scaled Brier" (clinical/holdout_clin.csv) |
| 0.049; 0.82; 0.013; 1.14 | 3.4 Table 5 | report §12.3 table, row "Calibration intercept / slope" (clinical/holdout_clin.csv) |
| 0.258; 0.241 | 3.4 Table 5 | report §12.3 table, row "Threshold (OOF Youden)" (clinical/holdout_clin.csv) |
| 0.684; 0.697; 0.779; 0.596 | 3.4 Table 5 | report §12.3 table, row "Sensitivity / specificity" (clinical/holdout_clin.csv) |
| 0.449; 0.542; 0.411; 0.538 | 3.4 Table 5 | report §12.3 table, row "Precision / F1" (clinical/holdout_clin.csv) |
| 40.4; 50.4 | 3.4 Table 5 | report §12.3 table, row "Flagged as high risk" (clinical/holdout_clin.csv) |
| 0.787; 0.001; 0.006; 0.008 | 3.6 Table 6 | report §9.1 / §14 tables, row "Reference / Comorbidity timing / LR (unpenalised)" (replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_*.csv) |
| 0.790; 0.002; 0.006; 0.010 | 3.6 Table 6 | report §9.1 / §14 tables, row "Reference / Comorbidity timing / RF (tuned)" (replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_*.csv) |
| 0.787; 0.014; 0.023; 0.005 | 3.6 Table 6 | report §9.1 / §14 tables, row "Landmark / Prior-admission comorbidities only / LR (L2)" (replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_*.csv) |
| 0.789; 0.008; 0.016; 0.0001 | 3.6 Table 6 | report §9.1 / §14 tables, row "Landmark / Prior-admission comorbidities only / RF (tuned)" (replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_*.csv) |
| 0.802; 0.001; 0.005; 0.007 | 3.6 Table 6 | report §9.1 / §14 tables, row "Landmark / No white-cell differential / LR (L2)" (replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_*.csv) |
| 0.795; 0.002; 0.010; 0.005 | 3.6 Table 6 | report §9.1 / §14 tables, row "Landmark / No white-cell differential / RF (tuned)" (replication/nested_cv_full_vs_main_C_comorb_time.csv; clinical/nested_cv_full_vs_main_clin_*.csv) |

## Exploratory extensions (added 2026-10-21; source: classmate's report_en.md Part IV §19–25, copy at /mnt/project-files/code/dn_extensions_clean/reports/report_en.md; CSVs in reports/extensions/balanced_27464ba98305edbc/)

| Number | Where in paper | Source |
|---|---|---|
| 2,045 patients, 544 deaths (development set) | 2.8 | §19.1 |
| 56 static candidates; 25 series × 5 statistics = 125 trajectory candidates | 2.8 | §19.1, §20.1 |
| 4 burden variables; TF-IDF min_df 10; 32 SVD dimensions; 3-fold stacking | 2.8 | §19.2, §21.1–21.2, §22.1 |
| C = 1; RF 200 trees depth 10; XGBoost 100 trees depth 2; LASSO 5 folds, 30 λ | 2.8 | §19.2 table; balanced/config.yaml |
| 15 comparisons per metric (Holm) | 2.8 | §19.3 |
| 0.805 vs 0.799; Δ 0.005 (−0.001, 0.011); p 0.077; Holm 1.00 | Abstract, 3.8 | §22.2; paired_comparisons.csv row static_stacking/static_lr AUROC (0.00539, −0.00060, 0.01138, p 0.0765) |
| 41 of 50 folds; all 5 repeats | 3.8 | §22.2 |
| OOF calibration slope 1.05 vs 0.90 | 3.8 | §24.1 (1.0502, 0.8983) |
| 31 of 56 thresholds | 3.8 | §24.2 dca_comparison |
| median 4.5 temporal per fold, range 1–13; SpO2 std 48/50 | 3.8 | §20.2 |
| Trajectory ΔAUROC −0.004, −0.001, −0.002 (LR, RF, XGBoost) | 3.8 | §20.3 (−0.0036, −0.0006, −0.0022) |
| 1,183 (57.8%) with prior diagnosis document | 3.8 | §21.2 |
| Embedding RF +0.002 vs static RF; Brier 0.1518 → 0.1527; slope 1.35 | 3.8 | §21.3, §24.1 (0.00167; 1.3522) |
| Combined stacking Δ < 0.001 | 3.8 | §22.3 (+0.000086) |
| Table 8 all values | 3.8 Table 8 | §23.1 table and summary.csv (fold means/SD rounded to 3 dp); ΔAUROC and CIs from §23.2 / paired_comparisons.csv |
| 14 models, 45 comparisons | S12 | §23.1, §23.2 |
