# Supplementary evaluation on the original V3 holdout

Each recipe was refitted on all 2,045 development patients (544 deaths), then evaluated on the same original 512 held-out patients (136 deaths), with model seed 42. Base-model hyperparameters were fixed; imputation, selection, embeddings, model coefficients/trees and stacking weights were learned again. The holdout had already been inspected in the original V3 study, so this is supplementary internal evaluation, not a new blind or external validation.

| Recipe | AUROC (95% DeLong CI) | AUPRC | Brier |
|---|---:|---:|---:|
| Static LR | 0.7533 (0.7056-0.8010) | 0.5391 | 0.1650 |
| Static stacking | 0.7559 (0.7085-0.8033) | 0.5452 | 0.1635 |
| Combined stacking | 0.7556 (0.7079-0.8032) | 0.5387 | 0.1624 |

Static stacking minus LR: AUROC difference +0.00256, paired bootstrap 95% CI [-0.00882, +0.01404]. Combined minus static stacking: -0.00035 [-0.01267, +0.01148]. Neither establishes a reliable improvement. AUPRC/Brier difference intervals also include zero. CIs use 2,000 stratified patient bootstrap resamples with the same sampled patients across models; prevalence is fixed. They condition on the fitted models and do not include retraining uncertainty. Difference intervals are not multiplicity adjusted.

Calibration slopes were 0.731, 0.825 and 0.830, respectively. These are descriptive estimates; no holdout recalibration was applied. DCA uses prespecified thresholds 0.05-0.60 without confidence intervals.

The final combined model selected 14 numeric features, including one temporal input (`trajectory_spo2_std`), plus four prior-diagnosis burden variables and 32 diagnosis embedding dimensions. LR, RF and XGBoost base models received the selected inputs. Selection confirms input availability, not large importance, causal actionability, a risk direction, or that every tree uses that variable.

Keep the original V3 L2-LR held-out AUROC near 0.750 separate: the balanced models here use a different fixed-parameter recipe. Development CV near 0.805 must not be presented as held-out performance. The archived signature names the original evaluated run; relocated/translated reruns receive different signatures.

See `holdout_summary.csv`, `holdout_paired_differences.csv`, `holdout_dca.csv` and `source_provenance.json`. No patient-level data or model objects are included.
