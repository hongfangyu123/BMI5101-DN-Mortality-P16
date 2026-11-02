# Five repeated holdouts and paired intervals

Run from the repository root in WSL/Linux. The existing wrapper defaults to the `dn_repro` Conda environment; set `BMI5101_CONDA_ENV=ml` if using the existing WSL ml environment, and `BMI5101_CONDA_BASE` if needed. The wrapper locates the repository root automatically. No real training runs by default.

```bash
bash extensions/balanced/run_extensions.sh repeated-holdout --check-only
bash extensions/balanced/run_extensions.sh repeated-holdout --run
# Aggregate-only CI calculation, no MIMIC data or retraining needed:
bash extensions/balanced/run_extensions.sh paired-ci
```

Use all 2,557 original CLIN patients (680 deaths); stratified 80/20 splits with prespecified seeds 101-105. Each split compares static LR, static stacking and combined stacking on identical patients. Model seed remains 42. Hyperparameters are fixed; coefficients, trees, imputation, LASSO selection, embedding and stacking weights are relearned for every training subset. Retain inner LASSO selection and three-fold stacking cross-fitting. Fifteen outer recipe fits include additional internal base-model fits; they are not fifteen classifiers in total.

Only unfitted raw trajectories/history are shared across splits. All labels are available for stratification and scoring, but each fitting call receives training labels only. Each split is patient-disjoint, and one cohort record per patient is required. This is exploratory internal split sensitivity after inspecting the old holdout, not a new untouched or external test. Preserve the original held-out results. SD/range describe split variation, not confidence intervals; repeated test predictions are not pooled as independent patients. Static diagnosis timing assumptions remain unchanged.

Progress is saved after every complete split under `../results/repeated_holdout/<new-signature>/`, including all split scores, means/SD/ranges, paired differences, DCA and a report/results notebook. Repeat the same command to resume validated completed recipe predictions. Only the unfinished recipe is retrained. Do not run the identical task concurrently: a lock guards its output. After abnormal power loss, verify no task is running before deleting a stale `.running.lock`. Raw features, identifiers and predictions stay under private `data/`.

The original evaluated archive is `../results/repeated_holdout/d6f09e8dedcfbfc1/`; its numeric CSVs are byte-identical to the reviewed local results. `active.txt` identifies a running rerun and `latest.txt` a completed rerun. Notebook 08 uses the archive when no rerun marker exists.

`paired-ci` prints approximate Nadeau-Bengio corrected resampled-t intervals using completed local metadata or the hash-checked published aggregate archive. It does not fit models. To also write a new supplement/report/notebook cell, explicitly add `--write`. All three paired comparisons and AUROC/AUPRC/Brier are retained; Holm p-values adjust three comparisons per metric. Marginal intervals are approximate, not exact or multiplicity-adjusted, and do not correct selection bias. See the archived CI note for the formula and limitations.

Synthetic checks:

```bash
BMI5101_V3_ROOT=$(pwd) conda run -n dn_repro python -B extensions/balanced/repeated_holdout/test_repeated_holdout.py
```

The original single run took about 8 minutes for all recipes, but full temporal fitting varied substantially across splits; do not assume identical runtime.
