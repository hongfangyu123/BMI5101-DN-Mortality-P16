> **This repository:** the extensions live in `extensions/` inside the V3 project; the run scripts set `BMI5101_V3_ROOT` to the repository root and use the `dn_repro` environment. Only feature preparation (`check`, `prepare`) is kept here; the incomplete strict experiment branch is not included. The experiments reported in the paper and in `results/report_en.md` Part IV are in [`balanced/`](balanced/README.md). See the root `README.md` for current commands.

# V3: stacking, temporal trajectories and comorbidity embedding

This directory is a standalone extension of V3 that reads the development set and the subset DuckDB from `../dn_mortality_repro_V3/dn_mortality_repro`. The original V3 files, frozen model, held-out results and old caches are not modified.

## Implemented and not yet run

- Three kinds of extension, a common static baseline, combined ablations, per-fold caching, AUROC/AUPRC/Brier/calibration/DCA and paired comparisons.
- Notebooks 01–05 (originally in Chinese), which automatically record the results and conclusions after every real development-set experiment.
- Only the lightweight preflight check and a smoke test on 120 rows of synthetic data have been run; no real patient feature extraction or formal cross-validation has been run. There is currently no conclusion of any improvement.

## Running in WSL

```bash
conda activate ml
cd /mnt/d/NUS_Course/BMI5101/dn_mortality_repro_V3_extensions

# Check inputs and structure, no training.
bash run_extensions.sh check

# First run: only extract development-set trajectories and prior diagnoses; may take a few minutes, no CV.
bash run_extensions.sh prepare

# Optional: first check the real-data pipeline and run time with 1×3 outer folds.
bash run_extensions.sh pilot --experiments static_stacking

# Formal experiments can be run one at a time; completed folds reuse the cache automatically.
bash run_extensions.sh full --experiments static_rf,static_xgb,static_stacking
bash run_extensions.sh full --experiments temporal_lr
bash run_extensions.sh full --experiments burden_lr,embedding_lr
bash run_extensions.sh full --experiments combined_stacking
```

Without `--experiments` all eight experiment groups in the configuration are run. To make troubleshooting easier, run them one at a time in the order above. Different profiles use separate caches; pilot results are never mixed into full. After an interruption, continue with the same command; changing code, data or configuration produces a new signature, which prevents wrongly reusing an old cache. Do not start two training processes in the same directory at the same time.

`jobs: 2` in `config.yaml` controls the internal tuning parallelism; inner models are fixed to 1 thread and stacking runs sequentially inside, to avoid recursive parallelism filling the memory. `--jobs 1` can be added. The formal strict stacking can still be very slow, and its run time cannot be extrapolated from the old fixed-parameter version.

Select the notebook kernel **Python (ml - WSL)**; the run switches default to False. The most convenient way is to run in the terminal and then open the corresponding notebook to read the automatically written conclusions.

## Methods and comparisons

- Reuses the V3 CLIN sample, one-year mortality label and development/held-out split; 5×10 outer folds with seeds 42–46, in the same order as V3.
- The labels of the fixed held-out set are not read and its performance is not evaluated. The V3 holdout has already been looked at and cannot be used to select extensions repeatedly.
- Missingness screening inherits the V3 20% variable rule and the exemption of the 8 static WBC differential variables; no patient is dropped because of missing variables. Trajectory variables are not exempted.
- Static features keep the V3 −6 to 24 h window; trajectories use only 0 to 24 h, within the same hadm_id/stay_id, to avoid stitching across admissions. Labs without hadm_id are not used for trajectories.
- LASSO reuses the V3 glmnet-style path and the λ.1se rule; this implementation redoes preprocessing even inside the LASSO inner folds. Variables with no variation or all missing are removed automatically; if no variable is selected, a constant column is used and the model degenerates to predicting the base rate.
- Model tuning uses small fixed grids: 4 settings for LR, 8 for RF and 8 for XGBoost. This is smaller than the exhaustive V3 search, but every inner tuning fold retrains the complete Pipeline. The new static baseline is compared pairwise with the new extensions; old V3 tables are not treated as results that can be subtracted directly.
- The stacking base models are themselves GridSearchCV(Pipeline), with tuning and representation learning redone inside their OOF sub-folds; the meta-model uses OOF probabilities, and the outer validation patients are completely independent.
- Prior documents come only from `previous dischtime < index admittime`. The TF-IDF vocabulary, IDF, SVD and scaling are all fitted on the training folds. The embedding has at most 32 dimensions and is not a clinical-text or causal comorbidity representation.
- The three ICD ablations separate the number of prior diagnoses from the embedding gain; the combined model is only an exploratory comparison.

## Outputs

`results/full/<signature>/` or `results/pilot/<signature>/`:

- `summary.csv`: fold means and the mean of the pooled OOF metrics per repeat.
- `paired_comparisons.csv`: Nadeau–Bengio corrected 95% CI and p-value for extension − reference; Holm correction per metric over the comparisons currently available.
- `fold_metrics.csv`, `pooled_oof_metrics.csv`, `pooled_dca.csv`.
- `fit_metadata.json`: selected variables, embedding dimensions and parameters for each fold.
- `run_identity.json`: data, code and configuration identity, and run mode.
- `report.md`: results and limits of interpretation; also written back to the corresponding notebook.

`data/` is kept only on the local machine and ignored by Git: development-set features, ICD documents, out-of-fold probabilities and Pipeline caches. The aggregate results contain no patient IDs. Do not publish patient-level features, documents or model caches.

For AUROC/AUPRC a difference > 0 is better; for Brier a difference < 0 is better. A pilot cannot support formal conclusions, and smoke is entirely synthetic data. The formal experiments are also exploratory development-set evidence; they should not promise to beat the paper's 0.780 or claim independent validation.

`bash run_extensions.sh smoke` can be used to check the code wiring again; its results are written only to `results/smoke/` and never as real notebook conclusions.


## 2026-10-14: moderate-compute version

The original strict version is kept in full. The simplified version is in [balanced/README.md](balanced/README.md).

First press Ctrl+C in the original training terminal, then run:

```bash
bash balanced/run_extensions.sh full --experiments static_rf,static_xgb,static_stacking
```

The new method keeps the 5×10 outer evaluation, preprocessing within training folds and out-of-fold stacking predictions; it drops the repeated grid search of the base models, uses 5 folds/30 λ for LASSO and 3 folds for stacking. The two versions use separate caches, and the old predictions of the first 6 folds were not stitched into the new results. No need to rerun prepare.
