> **This repository:** run from the repository root with `bash extensions/balanced/run_extensions.sh ...` (environment `dn_repro`, `BMI5101_V3_ROOT` set automatically). Shared features are in `extensions/data/processed/` (local only, git-ignored). Paths such as `/mnt/d/...` below refer to the original author's machine.
>
> Comments and messages were translated to English on 2026-11-01 after the runs, so the code_sha256 recorded in run_identity.json refers to the pre-translation source; the logic is unchanged. For the same reason `--reuse-from ebf3f368bb8beb5c` now refuses the old local cache (its training fingerprint covers these files); fresh runs are unaffected. The archived snapshot `results/extensions/balanced_27464ba98305edbc/` was not edited because its files are SHA-256 checked, so the `note` in its `cache_reuse.json` remains in Chinese.

# V3 extension: balanced, simplified-compute version

2026-10-14: the original strict version had completed the first 6 outer folds of the first repeat. The original code and cache are kept; this subdirectory uses its own method and cache and does not mix in old predictions.

## Switching directly

First press **Ctrl+C** in the original training terminal. Completed folds are not lost; a fold that was being fitted and has not saved complete predictions must be recomputed.

Start the simplified version from the original extension directory:

```bash
cd /mnt/d/NUS_Course/BMI5101/dn_mortality_repro_V3_extensions
bash balanced/run_extensions.sh full --experiments static_rf,static_xgb,static_stacking
```

Later runs can still be done item by item:

```bash
bash balanced/run_extensions.sh full --experiments temporal_lr
bash balanced/run_extensions.sh full --experiments burden_lr,embedding_lr
bash balanced/run_extensions.sh full --experiments combined_stacking
```

`full` is still the 5×10 outer validation with seeds 42–46. To check only the wiring and run time, use `pilot`; a pilot is not a formal conclusion. This time only a smoke test on synthetic data was run; no real patient training of the new version has been started.

## What was reduced

| Item | Original strict version | balanced full |
|---|---|---|
| Outer evaluation | 5×10 folds | 5×10 folds kept |
| Stacking cross-fitting | 5 folds | 3 folds |
| Base-model tuning | another 5-fold GridSearchCV in every stacking sub-fold | fixed parameters, no GridSearchCV |
| LASSO selection | 10 folds × 100 λ | 5 folds × 30 λ |
| Preprocessing cache | representations fitted separately for LR and trees | shared common representation cache, LR gets its own lightweight scaling |

The fixed-parameter scheme was written into the configuration before this version of the experiments started; the "best parameters" were not chosen from the 6 folds that were running:

| Model | Fixed settings |
|---|---|
| LR | L2, C=1 |
| RF | 200 trees, max_depth=10, min_samples_leaf=10, max_features=sqrt |
| XGBoost | 100 trees, max_depth=2, learning_rate=0.1, subsample=0.8, colsample_bytree=0.6 |
| Meta-model | L2 LR, C=1 |

These parameters are an explicit moderate-compute recipe, not a claim that they are optimal. The static baseline and all extensions use the same recipe. Changes in the number of RF trees, the λ grid and the hyperparameter strategy may change performance and the selected variables, so fold results from the old and new versions cannot be stitched together.

## Validity constraints kept

- The same V3 CLIN cohort, outcome, development-set split and outer folds.
- The labels or scores of the fixed held-out set are not read.
- Missing-rate screening, imputation, scaling, LASSO, TF-IDF and SVD are fitted on the corresponding training subsets.
- The stacking meta-model uses out-of-fold predictions; every sub-fold retrains the Pipeline, and features are never screened in advance on the whole development set.
- LASSO λ still uses the 1-SE rule, and preprocessing is also redone inside every inner fold.
- Paired comparisons on the same outer folds, reporting AUROC, AUPRC, Brier, calibration and DCA.

Only the nested parameter search of the base models was dropped; the independent outer evaluation remains valid. The project as a whole has already explored several schemes, so the new results are still exploratory development-set evidence, not an independent confirmatory conclusion.

## Data and result locations

- Shared read-only features: `../data/processed/development_features.parquet`; no need to rerun prepare.
- New cache: `balanced/data/cache/`, separate from the original strict-version cache.
- New results: `balanced/results/full/<signature>/`.
- New notebooks: `balanced/notebooks/01–05`, conclusions updated automatically after the experiments finish.
- The original strict version continues to use the parent directory; its first 6 folds and old code are not deleted or overwritten.

Running `bash run_extensions.sh full ...` from the original directory still runs the old strict version; the simplified version must use the entry point with `balanced/`.

Fewer fits do not guarantee the same AUC, and real run times cannot be promised on the basis of synthetic tests alone. Watch the first real outer fold, then estimate the time for the whole round.


## RF / XGBoost ablations with the new features (2026-10-15)

Experiment names were added to the existing entry point and Pipeline, without changing the parameters or the default experiment list in config.yaml.
Run from the `dn_mortality_repro_V3_extensions` directory:

```bash
# Run the four requested ablations in one go; the same-model static/diagnosis-burden comparators are added automatically.
bash balanced/run_extensions.sh full --experiments temporal_rf,temporal_xgb,embedding_rf,embedding_xgb --reuse-from ebf3f368bb8beb5c
```

The same command can also be run twice, with `--experiments` set to
`temporal_rf,temporal_xgb` and `embedding_rf,embedding_xgb` respectively.
Add `--check-only` at the end of the command to validate only, without training or writing results.
The script still runs through `conda run -n ml` in WSL.

| Experiment | Features | Main comparator |
|---|---|---|
| temporal_rf / temporal_xgb | static + 0–24h temporal features | same-model static |
| burden_rf / burden_xgb | static + 4 prior-diagnosis burden variables | same-model static |
| embedding_rf / embedding_xgb | static + diagnosis burden + ICD TF-IDF/SVD | same-model burden |

The diagnosis-burden groups are added automatically to separate the contribution of the number of diagnoses from that of the embedding; for embedding the total gain relative to the static model is also reported.
For each model the parameters, seeds 42–46, 5×10 folds, 20% variable missing-rate rule, imputation and LASSO 1-SE are unchanged.
TF-IDF (min_df=10) and SVD (at most 32 dimensions) are fitted within each training fold. Validation data are only transformed.
Tree models still use the same LASSO feature selection as LR, to keep the comparison fair; this is not a search over all tree-specific selection schemes.

Changing the code produces a new result signature. `--reuse-from` accepts only the version reviewed before the change:
after checking the complete data, configuration, software versions, training functions and the indices of every fold, the previously completed predictions are copied into the new cache.
The original result directory is not overwritten; the reuse source is recorded in `cache_reuse.json` of the new results.
The preprocessing joblib cache for identical inputs can also be reused; old stacking predictions only take part in summarising existing results and never trigger stacking retraining.
If parameters, training algorithm, data or environment do not match, the program raises an error and does not relax the checks automatically.

After the run finishes, the results are written to a new `results/full/<signature>/`, and the results and conclusions of notebooks 03, 04 and 06 are updated automatically.
Notebook 06 holds the method explanation and the run switch for this run; by default it does not start training.
Note: the updated Holm p-values refer to the current comparison family; the manual records in the old notebooks are the historical results of that time.
