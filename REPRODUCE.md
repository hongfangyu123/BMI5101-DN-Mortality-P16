# Reproduction guide

This guide explains how to re-run the whole project from the raw data on another computer and obtain the same aggregate results as in the repository (`results/`). All commands are run in a terminal from the project root (the unpacked `dn_mortality_repro/`).

## 1. Prerequisites

- **PhysioNet credentialing**: you need your own PhysioNet account, completed CITI training, and a signed MIMIC-IV data use agreement (DUA).
- **Download MIMIC-IV v3.1** (https://physionet.org/content/mimiciv/3.1/), for example:
  ```bash
  wget -r -N -c -np --user <your PhysioNet username> --ask-password https://physionet.org/files/mimiciv/3.1/
  ```
  The downloaded directory (called `MIMIC_ROOT` below) must directly contain the two subdirectories `hosp/` and `icu/`. The project reads 10 tables from it: patients, admissions, diagnoses_icd, services, labevents and d_labitems from `hosp/`, and icustays, chartevents, outputevents and d_items from `icu/` (`.csv.gz`, no need to decompress).
- **The data must not be shared with anyone**: the raw data and any patient-level file derived from it (the DuckDB database, parquet files, model files and predictions under `data/`, and the single-patient SHAP force plots `fig6c–e_*`) must stay on your own computer. This repository and the archive contain only code and aggregate results.
- **Hardware**: a Mac of the same model as the original author's (Apple silicon, 8 GB RAM). 8 GB of RAM is enough (the DuckDB memory limit is set to 4 GB); close other memory-hungry programs while running. Keep at least 10 GB of free disk space.

## 2. Installation

```bash
# 1) conda environment (name dn_repro, Python 3.11; environment.yml pins the core package versions and works across platforms)
conda env create -f environment.yml
#    All dependencies exactly as on the original author's Mac: conda env create -f environment-macos-full.yml
#    Explicit Linux lock file: conda create -n dn_repro --file environment-linux-64.lock.txt

# 2) mimic-code: clone into external/mimic-code and check out the pinned commit (see external/MIMIC_CODE_COMMIT.txt)
git clone https://github.com/MIT-LCP/mimic-code.git external/mimic-code
git -C external/mimic-code checkout 303d26c623dcc9c49cc0f204468d4acc2f063797

# 3) point to your MIMIC-IV v3.1 directory (the level that contains hosp/ and icu/)
export MIMIC_ROOT=/path/to/mimic-iv-3.1
#    To make it permanent, add it to ~/.zshrc:
#    echo 'export MIMIC_ROOT=/path/to/mimic-iv-3.1' >> ~/.zshrc
```

- `external/mimic-code/` is in `.gitignore` and is not in the repository or the archive; you must clone it yourself with the commands above. Only the commit hash is recorded.
- If `MIMIC_ROOT` is not set, the programs use `paths.mimic_root` from `config/config.yaml` (default `~/Desktop/mimic-iv-3.1`). If the directory does not exist, the programs stop with an error that explains how to set it.
- Do not change anything else in `src/` or `config/config.yaml`: the stage-8 frozen model (`results/clinical/frozen_model_clin.json`) records the hashes of `src/models.py`, `src/lasso.py`, `src/preprocess.py`, `src/variables.py` and `src/metrics.py`, and `04h`/`04j` refuse to run after any change.

## 3. Running

### 3.1 Environment self-check

```bash
conda run -n dn_repro python scripts/00_check_env.py
```

Prints OK / FAIL item by item: whether Python and the key package versions match `environment.yml`; whether `MIMIC_ROOT` exists and all 10 raw tables are present (only checks that the files exist, does not read them); whether `external/mimic-code` is at the right commit; memory and disk space. "available memory now" is only a hint (WARN) and never causes a failure. `run_all.sh` and `run_stage8.sh` run this step automatically at the start and stop on any FAIL.

### 3.2 Stages 0–7: `run_all.sh`

| Script | Content | Approximate run time (observed on an 8 GB Mac) |
|---|---|---|
| `01_cohort.py` | cohort construction + subset DuckDB database + mimic-code concept tables (scans chartevents/labevents; the most memory-intensive step) | 5–20 min |
| `01b_outcome_check.py` | one-year mortality outcome check | < 1 min |
| `02_features.py` | 55 features | 1–2 min |
| `03_preprocess.py` | split, imputation, Table 1 | < 1 min |
| `04_model.py` | LASSO, 9 models, tuning, DCA, SHAP | 20–30 min |
| `04b`, `04c` | calibration, thresholds, DeLong, LR interpretation | a few minutes |
| `04d_nested_cv.py` | "LASSO-only nested" CV | about 10 min |
| `04e_sensitivity_pre2020.py` | sensitivity analysis excluding 2020–2022 | about 10 min |
| `05_report.py` | report generation | seconds |

About 1–1.5 hours in total.

### 3.3 Stage 8: `run_stage8.sh` (must run after `run_all.sh` has finished)

Runs in order: `01c` (CLIN cohort) → `02b` (definition C comorbidity timing) → `02b --track clinical` (CLIN comorbidities) → `02_features --track clinical` (CLIN features) → `03c` (CLIN Table 1) → `04f` five full nested CV runs (definition C main analysis about 1.5 h, definition C comorb_time about 1.2 h, CLIN main analysis about 2.2 h, CLIN prior_only about 1.8 h, CLIN no_wbc_diff about 1.5 h) → `04g` (definition C held-out set) → `04h` (fit the frozen CLIN model) → `04j` (CLIN figures and checks) → `05` (report). About **9–10 hours** in total (the original author ran it on an 8 GB Mac). Each step writes its log to `results/logs/<step name>.log`.

`04f` caches the result of every completed fold in `data/interim/nested_cache/`; if the run is interrupted, re-running `run_stage8.sh` reuses the completed folds directly.

### 3.4 Running long jobs in the background

Use `nohup` so that closing the terminal does not stop the run, and `caffeinate -i` to keep the Mac from sleeping:

```bash
cd /path/to/dn_mortality_repro
export MIMIC_ROOT=/path/to/mimic-iv-3.1

# stages 0–7 and stage 8 one after the other (about 11 hours)
nohup caffeinate -i sh -c "bash run_all.sh && bash run_stage8.sh" > results/logs/run_everything.out 2>&1 &

# or run them separately:
nohup caffeinate -i bash run_all.sh   > results/logs/run_all.out    2>&1 &
nohup caffeinate -i bash run_stage8.sh > results/logs/run_stage8.out 2>&1 &   # start only after the previous one has finished
```

Checking progress:

```bash
tail -f results/logs/run_everything.out          # which step is running (Ctrl-C stops watching, not the run)
ls -lt results/logs | head                       # most recently updated logs
tail -n 5 results/logs/04f_nested_full_clin.log  # latest progress of one step (04f writes one line per fold, 5 × 10 folds)
pgrep -fl "scripts/"                             # scripts currently running; no output means the run has finished
```

## 4. About 04i (confirmatory evaluation on the CLIN held-out set)

- The CLIN held-out set (512 patients) is, as pre-specified, **evaluated only once** (decisions.md D8.15). `04i_holdout_clin.py` has already been run once on the original author's computer and is **not run when reproducing**; `run_stage8.sh` does not include it. The files `results/clinical/holdout_clin.csv`, `holdout_dca_clin.csv`, `coefs_clin.csv` and `shap_clin.csv` in the repository are the official results.
- `04i` has its own safeguard: it refuses to run if the output files already exist, so it cannot overwrite the results in the repository.
- **Reproduction check**: `04j_plots_clin.py` uses your refitted frozen model (`04h`) to predict the held-out set again, recomputes all 33 columns of `holdout_clin.csv` and the 112 rows of `holdout_dca_clin.csv`, and compares them with the values in the repository (differences must be < 1e-9); it exits with an error if they disagree. A successful `04j` run means that your frozen model and held-out evaluation agree with the official results.
- `04h` rewrites `results/clinical/final_model_clin.json`. The model-file hash `bundle_sha256` in it depends on the serialised bytes and may differ from the repository value on another computer; all other fields (selected variables, λ, C, EPV) should be identical.

## 5. How to check that the reproduction succeeded

After everything has run:

```bash
git status --short results/                        # only the logs should have changed (if the unpacked directory is under git)
git diff --stat -- results/ ':!results/logs'       # aggregate results: there should be no differences
```

- If you received an archive rather than a git repository, first run `git init && git add -A && git commit -qm original` in the unpacked directory, then run the pipeline, and finally compare with the commands above.
- **Expected**: on the same Mac model with the same conda environment, the CSV/JSON files in `results/tables/`, `results/replication/` and `results/clinical/` and `results/report_en.md` should be **byte-for-byte identical** (feature computation was made independent of row order and thread count, D8.13). Exceptions: `bundle_sha256` in `final_model_clin.json` may differ (see Section 4); `results/logs/` contains run times and local paths and will always differ; PNG images may differ in metadata under different matplotlib builds.
- On other machine models or operating systems the order of floating-point operations differs, and differences of the order of 1e-10 can appear (for example in the last digit of a few p-values in Table 1); such differences do not affect any displayed number or conclusion. If larger differences appear, check the output of `00_check_env.py`, the mimic-code commit and the MIMIC-IV version (must be v3.1).

## 6. Data compliance

- `data/` (subset DuckDB database, parquet files, model files, held-out predictions, nested-CV cache) and the single-patient force plots (`results/figures/fig6c_*`, `fig6d_*`, `fig6e_*`) are patient-level data and **must stay on the local machine**: do not upload them, send them (including through instant messaging and cloud storage) or commit them to git. They are already in `.gitignore`.
- Do not inspect the raw data or files under `data/` with `head()`, `print(df)`, `cat`/`zcat` and the like; when checking data, look only at aggregate information such as row counts, missing rates and summary statistics. See `DATA_ACCESS.md` for details.
- Before committing or publishing, run `python tools/check_public_release.py`: it checks only the files tracked and staged by git, and reports an error when it finds `data/` paths, suffixes such as `.parquet`/`.duckdb`/`.pkl`, `fig6c–e`, CSV/JSON headers with patient-level fields, or secrets.

## 7. Extension experiments (stacking, first-day temporal trajectories, prior-diagnosis embedding; exploratory)

The extension code is in `extensions/`; the version finally adopted is `extensions/balanced/` (fixed parameters, 5 × 10 outer folds, using only the CLIN development set of 2,045 patients and never reading held-out labels during that original development-CV run). The result snapshot is in `results/extensions/balanced_27464ba98305edbc/`, and the report is Part IV of `results/report_en.md` (§19–25, Appendix C). The strict version (re-tuning within every stacking inner fold) completed only 6 folds and was not finished; its experiment code is not in the repository and its results are not in the report.

- **Dependencies**: the extensions read this project's `src/` (variables, LASSO, metric functions) and the stage-8 outputs `data/processed/{cohort_clin,features_clin,split_clin}.parquet`; feature preparation also queries `data/interim/mimic_subset.duckdb`. Section 3 must therefore be completed first (at least up to the CLIN features and split).
- **Locating the V3 project**: `extensions/run_extensions.sh` and `extensions/balanced/run_extensions.sh` automatically set the environment variable `BMI5101_V3_ROOT` to the repository root and use the `dn_repro` environment (can be overridden with `BMI5101_CONDA_ENV` and `BMI5101_CONDA_BASE`). When running extension scripts directly with `python`, first `export BMI5101_V3_ROOT=$(pwd)` (repository root). Keep the original value of `source_project` in `config.yaml` and do not change it: it is part of the result signature, and changing it makes the signature disagree with the published `27464ba98305edbc`.

```bash
bash extensions/run_extensions.sh check            # check the V3 inputs and concept tables, no training
bash extensions/run_extensions.sh prepare          # prepare extension features -> extensions/data/processed/ (patient-level, git-ignored)
bash extensions/balanced/run_extensions.sh full --check-only
bash extensions/balanced/run_extensions.sh full --experiments static_lr,static_rf,static_xgb,static_stacking,temporal_lr,temporal_rf,temporal_xgb,burden_lr,burden_rf,burden_xgb,embedding_lr,embedding_rf,embedding_xgb,combined_stacking
```

- Without `--experiments` only the 8 groups in the configuration are run; the full design in the report is the 14 groups above and takes several hours. Do not use `--reuse-from` in a new environment (the per-fold cache is not in the repository).
- New runs are written to `extensions/balanced/results/full/<signature>/`. A different machine, operating system or software version may give a different signature and different last digits; the report cites the hash-checked snapshot `results/extensions/balanced_27464ba98305edbc/`, which is not replaced automatically by new runs.
- Wiring check with synthetic data only: `BMI5101_V3_ROOT=$(pwd) conda run -n dn_repro python extensions/balanced/tests/test_tree_ablations.py`.


## Supplementary validation added after development CV

The original Part IV development analysis did not read holdout labels. The following later evaluations are separate; retain all three views rather than substituting the highest score:

| Evaluation | Static LR AUROC | Static stacking AUROC | Combined stacking AUROC |
|---|---:|---:|---:|
| Development 5x10-fold CV, fixed-parameter recipe | 0.7993 | 0.8047 | 0.8048 |
| Previously inspected original holdout, 512 patients | 0.7533 | 0.7559 | 0.7556 |
| Five exploratory repeated 80/20 holdouts, mean | 0.7949 | 0.7983 | 0.8005 |

Repeated-split SDs are 0.0166, 0.0130 and 0.0142, not CIs. Static stacking has small directional gains, but all approximate corrected paired AUROC/AUPRC/Brier difference intervals include zero. Added trajectories/diagnosis representation show no stable incremental gain. The original holdout has already been inspected; the merged repeated-split analysis is exploratory, not new blind/external confirmation.

English supplementary reports and provenance: [original holdout](extensions/balanced/results/holdout/6f7b96b948977755/report.md), [five splits](extensions/balanced/results/repeated_holdout/d6f09e8dedcfbfc1/report.md), [paired intervals](extensions/balanced/results/repeated_holdout/d6f09e8dedcfbfc1/paired_corrected_ci.md). Interactive readers: [Notebook 07](extensions/balanced/notebooks/07_holdout_evaluation.ipynb), [Notebook 08](extensions/balanced/notebooks/08_repeated_holdout.ipynb). These additions do not alter the existing manuscript or baseline report generator.

```bash
bash extensions/balanced/run_extensions.sh holdout --check-only
bash extensions/balanced/run_extensions.sh holdout --run
bash extensions/balanced/run_extensions.sh repeated-holdout --check-only
bash extensions/balanced/run_extensions.sh repeated-holdout --run
bash extensions/balanced/run_extensions.sh paired-ci
```

Real patient fitting requires your own authorized MIMIC features/database; it never starts by default. `paired-ci` and synthetic tests need no patient data. Wrappers preserve the existing `dn_repro` environment default and `BMI5101_CONDA_ENV` override. Existing recipe settings and baseline training code are unchanged. New result archives contain only aggregates; historical run signatures are preserved, while translated/relocated reruns create new signatures. See the workflow READMEs for exact source-compatibility checks and resume behavior.

For the new data-free checks, additionally run the two holdout test scripts with `BMI5101_V3_ROOT=$(pwd)` as shown in their READMEs. Do not rerun old CV to reproduce the archived paired CI table.
