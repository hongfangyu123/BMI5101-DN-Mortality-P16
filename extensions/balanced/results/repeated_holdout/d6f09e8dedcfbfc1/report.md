# Five repeated holdouts: split-sensitivity analysis

All five prespecified splits completed on the original CLIN cohort: 2,557 patients, 680 deaths. Each stratified 80/20 split has 2,045 training patients (544 deaths) and 512 test patients (136 deaths). Split seeds are 101-105; model seed is held at 42. The three recipes share each split. Hyperparameters and training rules are fixed, while preprocessing, selection, embeddings, coefficients/trees and stacking weights are refitted in every training subset. Stacking keeps three-fold internal cross-fitting.

| Recipe | Mean AUROC +/- SD | AUROC range | Mean AUPRC | Mean Brier |
|---|---:|---:|---:|---:|
| Static LR | 0.7949 +/- 0.0166 | 0.7737-0.8156 | 0.5937 | 0.1511 |
| Static stacking | 0.7983 +/- 0.0130 | 0.7810-0.8133 | 0.6051 | 0.1500 |
| Combined stacking | 0.8005 +/- 0.0142 | 0.7851-0.8174 | 0.6063 | 0.1499 |

SD and range describe split variation, not a 95% confidence interval. The five splits share patients and are not independent cohorts. Do not select the best seed, pool repeated predictions as independent patient observations, or apply an ordinary independent/paired t-test to the five scores.

| Split seed | Static LR AUROC | Static stacking AUROC | Combined stacking AUROC |
|---|---:|---:|---:|
| 101 | 0.7906 | 0.7956 | 0.8016 |
| 102 | 0.7737 | 0.7810 | 0.7874 |
| 103 | 0.7876 | 0.7928 | 0.7851 |
| 104 | 0.8070 | 0.8089 | 0.8174 |
| 105 | 0.8156 | 0.8133 | 0.8112 |

## Paired findings

- Static stacking minus LR: mean AUROC difference +0.00341 (4/5 higher); AUPRC +0.01140 (5/5 higher), Brier -0.00106 (5/5 lower). Improvements are small but directionally more consistent than the additional-feature increment.
- Combined minus static stacking: mean AUROC +0.00222 (3/5 higher, 2/5 lower); AUPRC +0.00113 (2/5 higher), Brier -0.00015 (2/5 lower). Added trajectories/diagnosis representations do not show stable incremental benefit.
- Combined minus LR: mean AUROC +0.00563, AUPRC +0.01253 and Brier -0.00121. Their approximate corrected difference intervals all include zero; see `paired_corrected_ci.md`.

## Approximate corrected intervals (post-hoc supplement)

| Comparison | Mean AUROC difference | Approximate corrected 95% CI | Raw p |
|---|---:|---:|---:|
| Static stacking - static LR | +0.00341 | [-0.00356, +0.01039] | 0.2454 |
| Combined stacking - static stacking | +0.00222 | [-0.01065, +0.01509] | 0.6573 |
| Combined stacking - static LR | +0.00563 | [-0.01014, +0.02140] | 0.3775 |

The Nadeau-Bengio corrected resampled-t approximation uses five paired differences, test/train ratio 512/2045 and df=4. All nine AUROC/AUPRC/Brier intervals include zero. Applying this approximation to AUROC/AUPRC does not guarantee exact nominal coverage, and five repetitions give little ability to check distributional assumptions. CIs are marginal, not multiplicity adjusted; the CSV additionally gives Holm p-values across the three comparisons per metric. This supplement does not correct prior model-selection bias or prove equivalence.

## Actual temporal input selection

| Split seed | Temporal variables after missingness/validity filtering | Temporal variables after LASSO |
|---|---:|---:|
| 101 | 57 | 1 |
| 102 | 57 | 14 |
| 103 | 57 | 3 |
| 104 | 57 | 4 |
| 105 | 48 | 3 |

The most frequently selected temporal inputs in the 50 development outer refits were SpO2 SD (48/50), temperature measurement-time count (42/50), chloride last-minus-first (24/50), calcium count (21/50), SpO2 slope (21/50) and chloride slope (18/50). Median number selected was 4.5 (range 1-13). The numeric selection rule is shared by temporal LR/RF/XGBoost and combined stacking: identical frequencies are not independent replications. These metadata describe final outer refits, not every stacking subfold.

In the five full-training refits, selected temporal counts were 1, 14, 3, 4 and 3; each combined model also received 32 diagnosis dimensions. Selected names are in `temporal_selected_features.csv`. Time variables therefore did enter the model inputs. Selection does not establish importance, causal direction or use in an actual tree split. A count is measurement frequency and an SD is variability, not an upward/downward biological trajectory.

## Interpretation and limits

Other splits can produce AUROC near 0.80, suggesting that the original partition may contribute to the discrepancy with the original holdout near 0.756. Five additional splits do not prove the original result was an outlier, that the discrepancy is entirely random, or that CV was leakage-free. This analysis uses the merged cohort, unlike CV restricted to the original 2,045 development patients.

Report all three views: development CV (combined AUROC 0.8048), original inspected holdout (0.7556), and exploratory repeated holdout (0.8005 +/- 0.0142). Do not replace the original holdout with a higher sensitivity estimate or claim superiority to the paper's 0.780 across different cohorts/designs. No new untouched holdout remains after merging. Static index-admission diagnoses retain their original timing assumption; split repetition does not resolve it. Calibration-slope means were 0.949, 1.088 and 1.091, without interval-based calibration comparisons; DCA is descriptive at 0.05-0.60 without CIs.
