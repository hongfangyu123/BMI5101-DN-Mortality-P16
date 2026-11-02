# Supplementary original-holdout evaluation

Run from the repository root in WSL/Linux. The existing wrapper defaults to the `dn_repro` Conda environment; set `BMI5101_CONDA_ENV=ml` if using the existing WSL ml environment, and `BMI5101_CONDA_BASE` if needed. The wrapper locates the repository root automatically. No real training runs by default.

```bash
bash extensions/balanced/run_extensions.sh holdout --check-only
bash extensions/balanced/run_extensions.sh holdout --run
```

Recipes: static LR, static stacking and combined stacking. Refit each recipe on all 2,045 development patients, model seed 42; predict the original 512 held-out patients. This is one final fit per recipe, not five-model averaging. No 50-fold outer CV is rerun. LASSO selection, imputation, TF-IDF/SVD and three-fold stacking cross-fitting remain inside the relevant training data. Raw feature windows and history boundaries are unchanged.

`--check-only` checks splits, settings, source compatibility, package versions and schemas without reading holdout labels or fitting. `--run` extracts raw holdout features, freezes a run protocol, fits/predicts all three recipes and only then reads outcomes. No holdout-based model/threshold selection or recalibration. Previously completed prediction checkpoints can be resumed with the same command; an interrupted unfinished recipe must be retrained.

The repository's English translation changed source bytes, not the reviewed training algorithm. `source_compatibility.json` pins the exact reviewed repository fingerprint and seven source/config file hashes. Original development data, recipe settings and package versions must still match; any unreviewed source/data/settings/environment change is rejected. Do not disable these checks. Direct Python use additionally requires `BMI5101_V3_ROOT` set to the repository root.

Archived aggregate results: `../results/holdout/6f7b96b948977755/`. New reruns write a new signature directory under `../results/holdout/`; `latest.txt` points to a completed rerun. Generated outputs include summary, paired stratified-bootstrap intervals, calibration, DCA, a report and a results notebook. Patient-level predictions and checkpoints remain under private `data/holdout/`. The original inspected holdout is supplementary, not new external confirmation.

Synthetic checks (no patient fitting):

```bash
BMI5101_V3_ROOT=$(pwd) conda run -n dn_repro python -B extensions/balanced/holdout/test_holdout.py
```
