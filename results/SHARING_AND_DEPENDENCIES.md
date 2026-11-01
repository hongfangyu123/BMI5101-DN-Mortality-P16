> **Note (this repository):** sections 3–4 below describe the original two-folder layout used when the extensions were shared. In this repository the extensions live in `extensions/` inside the V3 project and find it automatically; see the root `README.md` for the current commands.

# Files shared with classmates and dependencies on the original V3

Compiled on 2026-10-21. The analysis is as of 2026-10-15; the final extension signature is `27464ba98305edbc`.

## 1. Which report to read first

- Full Chinese report (`report_zh.md`): copied unchanged from the latest Chinese report of the original V3, with all four parts: replication of the original paper, CLIN, comparison and limitations, and extension experiments. It is not included in this repository; use [`report_en.md`](report_en.md), which has the same content and numbers.
- [Full English report](report_en.md): keeps the first three parts of the original English report and adds the latest summary, the updates to Sections 17–18, Part IV Sections 19–25 and Appendix C; it is not just a translation of the extension summary.
- [Report folder description and check record](README.md): describes the accompanying files and where each version came from.

`reports/` already includes the figures, aggregate result tables, decision log and progress log cited by the reports. Copying the whole `reports/` folder on its own is enough to read the reports and their supporting results, without the original V3, the database or Python. `decisions.md` and `progress.md` are the original records (since translated to English); the data-table column names and experiment names cited in the English report keep their original names.

## 2. Why running the extensions still needs the original V3

The extensions are an experiment module attached to the end of the original V3 CLIN data pipeline; they do not re-implement the original cohort construction or all the shared functions.

| Dependency | Path in the original V3 project | Use |
|---|---|---|
| Variable definitions | `src/variables.py` | imports `FEATURES` and `MODE_VARS` to fix the 55 static candidates and the mode-imputed variables; the extensions add `wbc_diff_missing` |
| Shared LASSO functions | `src/lasso.py` | imports `_deviance`, `_path` and `lambda_max` directly, reusing the penalty path and the binomial deviance; the within-fold procedure of balanced is implemented by the extensions themselves |
| Shared evaluation functions | `src/metrics.py` | imports the calibration, DCA and Nadeau–Bengio correction functions directly |
| CLIN cohort | `data/processed/cohort_clin.parquet` | admission/ICU identifiers and times of the development patients, used to check same-admission and 24 h window rules |
| CLIN static features | `data/processed/features_clin.parquet` | reuses the completed static predictors and the development-set outcome |
| Original development/held-out split | `data/processed/split_clin.parquet` | keeps the original split and development-set row order fixed; for the held-out set only `stay_id` is read to check disjointness, never the held-out labels |
| Subset database | `data/interim/mimic_subset.duckdb` | queried for raw measurements, prior admissions and diagnoses when preparing the new features and during the preflight check |

The database must contain `chemistry`, `complete_blood_count`, `blood_differential`, `coagulation` and `vitalsign` under `mimiciv_derived`, plus `mimiciv_hosp.admissions` and `mimiciv_hosp.diagnoses_icd`. Not every DuckDB file that contains only the raw tables can be used as a substitute.

**Having the raw MIMIC data is not the same as having the V3 intermediate files the extensions need.** If the parquet files and concept tables in the table above are missing, first complete the corresponding cohort, concept-table, CLIN comorbidity/feature and split steps following `REPRODUCE.md` of the original V3. If these inputs are all present, there is no need to rerun all the original V3 models just to run the extensions.

## 3. Folder layout

By default the two projects must be sibling directories, keeping the inner directory of the unpacked V3:

```text
any working directory/
├── dn_mortality_repro_V3/
│   └── dn_mortality_repro/
│       ├── src/
│       └── data/
│           ├── processed/
│           │   ├── cohort_clin.parquet
│           │   ├── features_clin.parquet
│           │   └── split_clin.parquet
│           └── interim/mimic_subset.duckdb
└── dn_mortality_repro_V3_extensions/
    ├── reports/
    ├── data/processed/development_features.parquet
    └── balanced/
```

In the strict version `config.yaml`, `source_project` is `../dn_mortality_repro_V3/dn_mortality_repro`; in the balanced configuration it is `../../dn_mortality_repro_V3/dn_mortality_repro`. Each is resolved relative to its own code root and both point to the same V3 project, independent of the directory the terminal started in.

If a classmate's original V3 is somewhere else, set the environment variable below in the same terminal; both versions use it and no configuration file needs to change. The path must point to the level that directly contains `src/` and `data/`:

```bash
export BMI5101_V3_ROOT="/actual/path/dn_mortality_repro"
```

## 4. Running balanced from a classmate's existing V3 inputs

The commands below use Bash/WSL and are run from the `dn_mortality_repro_V3_extensions` directory. They are instructions for classmates; no training was run when this report was compiled.

```bash
cd /actual/path/dn_mortality_repro_V3_extensions

# Only check the original V3 inputs and concept tables, no training.
bash run_extensions.sh check

# On the first run, or after the original V3 inputs change, prepare the new features; no model is trained here.
bash run_extensions.sh prepare

# Only check the balanced inputs and the experiment list; no training, no experiment results written.
bash balanced/run_extensions.sh full --check-only

# Optional: check the pipeline with a few folds first. A pilot cannot be used for formal conclusions.
bash balanced/run_extensions.sh pilot --experiments static_stacking

# Run all 14 groups behind the report from scratch; the old cache on my computer is not needed.
bash balanced/run_extensions.sh full --experiments static_lr,static_rf,static_xgb,static_stacking,temporal_lr,temporal_rf,temporal_xgb,burden_lr,burden_rf,burden_xgb,embedding_lr,embedding_rf,embedding_xgb,combined_stacking
```

The shell entry points above use the Conda environment named `ml` in `$HOME/miniconda3` by default. If Miniconda is installed elsewhere, set `BMI5101_CONDA_BASE`. If a classmate uses the original V3 `dn_repro` environment or another environment with the dependencies installed, they can activate their own environment and call Python directly, without renaming the environment to `ml`:

```bash
conda activate dn_repro
python scripts/00_preflight.py
python scripts/01_prepare_features.py
python balanced/scripts/02_run_experiments.py --profile full --check-only

# For the formal training run, drop --check-only from the previous command and add the --experiments list above.
```

The dependencies are NumPy, pandas, SciPy, scikit-learn, XGBoost, DuckDB, joblib, PyYAML, statsmodels, nbformat and a parquet read/write engine. The key versions of the original experiment are in [run_identity.json](extensions/balanced_27464ba98305edbc/run_identity.json). Watch API compatibility when reusing the V3 source; for example, the current `src/lasso.py` uses `LogisticRegression(l1_ratio=...)`, so it cannot be assumed to run with any older scikit-learn.

**balanced shares the features of the parent extension project**: `data/processed/development_features.parquet`, not `balanced/data/processed/`. `balanced/run_extensions.sh prepare` does not extract features; use `prepare` of the parent entry point. That parquet comes from the classmate's local V3 inputs and contains the static features, 125 trajectory candidates, 4 diagnosis-burden variables and the prior ICD documents; the vocabulary, SVD, imputation and selection are fitted only later, inside the training folds.

Even when the extension features are ready, formal training still reads the three original V3 parquet files to verify that the development set and the static columns have not changed, so keeping only the extension feature file is not enough. DuckDB is used in the `prepare` and `check` stages; formal training with existing features does not query it directly.

Without `--experiments`, the balanced configuration has only 8 groups by default, so it should not be assumed to rerun all 14 groups in the report automatically. `--reuse-from ebf3f368bb8beb5c` applies only when you hold the old local prediction cache and the compatibility check passes; it is not needed when you only have the aggregate results. Changes in machine, software or inputs may also produce a different result signature, which should not be forcibly renamed to the original directory name.

## 5. What to send to classmates

- To read the reports only: send the whole `reports/`, or open the two reports from the extension project README.
- To read the code and run it themselves: send the extension code, configuration, notebooks, aggregate results and reports; classmates prepare the inputs above in their own original V3 project.
- The newly added `reports/` contains no patient-level parquet, DuckDB, per-patient predictions, training cache, serialised models or single-patient force plots. The existing `data/` and `balanced/data/` of the extension project are still local run data; they are not needed to read the reports.

English: the complete bilingual reports and their aggregate supporting assets are self-contained in `reports/`. Running the extensions additionally requires the original V3 source helpers, CLIN cohort/features/split parquet files, and the subset DuckDB during preparation/checking. Set `BMI5101_V3_ROOT` if your original V3 project is elsewhere. A fresh run does not require the author's caches; explicitly select all 14 experiments to reproduce the full extension design.
