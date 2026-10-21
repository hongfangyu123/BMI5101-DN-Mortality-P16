# Data access and repository contents

MIMIC-IV v3.1 is a credentialed dataset. Obtain it with your own authorized PhysioNet account, after completing the required training and signing the data use agreement: https://physionet.org/content/mimiciv/3.1/ . This repository does not redistribute it.

Set `MIMIC_ROOT` to the directory that directly contains `hosp/` and `icu/` (if it is not set, `paths.mimic_root` in `config/config.yaml` is used). The pipeline reads `patients`, `admissions`, `diagnoses_icd`, `services`, `labevents` and `d_labitems` from `hosp/`, and `icustays`, `chartevents`, `outputevents` and `d_items` from `icu/`. Compressed `.csv.gz` files are read directly. The raw files are treated as read-only.

## Kept local and excluded from this repository

- Raw MIMIC files and every `data/` directory (`data/`, `extensions/data/`), including the subset DuckDB database, parquet files, the frozen model bundle, held-out predictions and the extension feature file `development_features.parquet`.
- Subject, admission and stay identifiers, cohort rows and prior-ICD documents.
- Nested-CV and pipeline caches, logs, credentials and machine backups.
- Single-patient SHAP force plots (`fig6c–e`).

## What the repository contains

Source code, configuration, method and decision documents, output-cleared notebooks, the frozen model **specification** (`results/clinical/frozen_model_clin.json`, which records hashes but no patient identifiers), aggregate tables and figures, the reports and the paper. It contains no fitted model bundles. The archived result snapshots are reference aggregates, not substitutes for the restricted data.

## Working with the data

- Inspect patient-level data only through aggregate summaries (row counts, missingness, descriptive statistics). Do not print or share individual rows.
- Run `python tools/check_public_release.py` before every commit that will be published. It checks the files git would publish for forbidden paths and file types, notebook outputs, credential patterns and patient-level field names in CSV and JSON files. It complements, and does not replace, a deliberate review of any new output: renamed patient-level data must never be uploaded.
