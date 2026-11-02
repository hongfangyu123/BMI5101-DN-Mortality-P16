# Approximate corrected paired confidence intervals

This post-hoc supplement reuses the five completed, overlapping holdouts. It does not retrain models. For paired differences d, use SE = sqrt((1/R + n_test/n_train)*s_d^2), with R=5, n_test=512, n_train=2045, df=4, and a two-sided t critical value for marginal 95% intervals.

| Comparison (first minus reference) | Metric | Mean difference | Approx. corrected 95% CI | Raw p |
|---|---|---:|---:|---:|
| Static stacking - Static LR | AUROC | +0.00341 | [-0.00356, +0.01039] | 0.2454 |
| Static stacking - Static LR | AUPRC | +0.01140 | [-0.00142, +0.02423] | 0.0691 |
| Static stacking - Static LR | Brier | -0.00106 | [-0.00280, +0.00068] | 0.1670 |
| Combined stacking - Static stacking | AUROC | +0.00222 | [-0.01065, +0.01508] | 0.6573 |
| Combined stacking - Static stacking | AUPRC | +0.00113 | [-0.01426, +0.01652] | 0.8486 |
| Combined stacking - Static stacking | Brier | -0.00015 | [-0.00266, +0.00236] | 0.8759 |
| Combined stacking - Static LR | AUROC | +0.00563 | [-0.01014, +0.02140] | 0.3775 |
| Combined stacking - Static LR | AUPRC | +0.01253 | [-0.00941, +0.03448] | 0.1880 |
| Combined stacking - Static LR | Brier | -0.00121 | [-0.00388, +0.00146] | 0.2773 |

All nine intervals include zero; reliable improvement is not established. This does not prove equivalence or absence of benefit. Positive AUROC/AUPRC and negative Brier differences are favorable. Intervals are not multiplicity adjusted; `p_holm_three_comparisons_per_metric` in the CSV controls the three comparisons separately for each metric.

This is the Nadeau-Bengio corrected resampled-t approximation, not an exact coverage guarantee for non-additive AUROC/AUPRC. Five resamples give little ability to assess assumptions. The correction addresses dependence across overlapping samples, not adaptive model selection, upstream timing/leakage problems or external validity. These intervals differ from the original single-holdout patient-bootstrap intervals.

Reference: [Nadeau and Bengio, Inference for the Generalization Error (2003)](https://doi.org/10.1023/A:1024068626366).
