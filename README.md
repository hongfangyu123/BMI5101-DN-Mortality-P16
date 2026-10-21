# One-year mortality in ICU patients with diabetic neuropathy: a leakage-free, landmark-based model on MIMIC-IV v3.1

BMI5101 Group Project (P16). We rebuild the framework of Huang et al., *Machine learning models predict mortality risk in diabetic neuropathy patients using MIMIC-IV data*, Sci Rep 2025;15:38702 ([doi:10.1038/s41598-025-22363-x](https://doi.org/10.1038/s41598-025-22363-x)) on MIMIC-IV v3.1, correct its methodological problems, and develop a landmark model that predicts one-year mortality 24 h after ICU admission. Exploratory extensions (stacking, first-day temporal trajectories and prior-diagnosis embeddings) are evaluated on the same development set.

**Paper:** [`paper/main.pdf`](paper/main.pdf) (LaTeX sources in `paper/`).
**Reports:** [`results/report_en.md`](results/report_en.md) (English) and [`results/report.md`](results/report.md) (Chinese), Parts I–IV.

Authors: [team members]

> 中文简介：本项目在 MIMIC-IV v3.1 上按 Huang et al. (2025) 的框架重做糖尿病神经病变 ICU 患者 1 年死亡预测，修正原文的方法问题（预测时点、完整血常规纳入条件、共病时间、结局与单位定义、单次验证集选模等），建立入 ICU 24 h 的 landmark 模型（CLIN），并在同一开发集上探索 stacking、首日时间轨迹和既往诊断 embedding 扩展。仓库中没有任何患者级数据。

## Main results

All numbers are taken from `results/report_en.md`.

| Analysis | Result |
|---|---|
| Landmark (CLIN) cohort | 2,557 patients, one-year mortality 26.6%; training 2,045, held-out 512 |
| Selected model | L2-penalised logistic regression, chosen by a pre-specified rule in 5 × 10 repeated nested cross-validation |
| Nested-CV AUROC (training set) | 0.801 |
| Confirmatory held-out AUROC (frozen model, evaluated once) | **0.750 (95% CI 0.702–0.797)**; calibration slope 0.82, so local recalibration is needed |
| Benchmark pipeline under nested CV (replication cohort, 1,960 patients) | six configurations equivalent (AUROC 0.786–0.791); the benchmark's random-forest settings were significantly worse (0.724–0.767) |
| Leading predictors | RDW, Charlson comorbidity index, age |

**Extensions (exploratory, Part IV of the reports).** On the CLIN development set (2,045 patients, 544 deaths; 14 experiments × 50 outer folds, fixed-parameter "balanced" recipe), static stacking reached AUROC 0.8047 against 0.7993 for logistic regression under the same recipe: a difference of +0.0054, 95% CI of the difference [−0.0006, +0.0114], final Holm p = 1.0. The direction is consistent but the gain is not statistically confirmed. Temporal trajectories showed no clear gain, the diagnosis embedding gave only the random forest a small numerical improvement, and combined stacking (0.8048) was almost identical to static stacking. The held-out set was not used for the extensions, so these development-set figures do not replace the held-out AUROC of about 0.750.

## Repository layout

```text
README.md, DATA_ACCESS.md, REPRODUCE.md   overview, data rules, full reproduction guide (Chinese)
environment.yml                           core pinned packages (conda env "dn_repro")
environment-macos-full.yml                complete export of the original macOS environment
environment-linux-64.lock.txt             explicit Linux lock file
run_all.sh, run_stage8.sh                 main pipeline, stages 0–7 and stage 8
config/                                   data path (MIMIC_ROOT), seeds, DuckDB memory limit
src/                                      shared code: DuckDB access, variables, LASSO, models, preprocessing, metrics
scripts/                                  numbered pipeline scripts 00–05 (see REPRODUCE.md); report_extensions.py
tests/                                    Nadeau–Bengio unit check (synthetic data); one-off subset-database check (needs local data)
extensions/                               extension feature preparation (check / prepare)
  balanced/                               extension experiments used in the reports (code, notebooks, results)
results/                                  aggregate results only
  tables/, replication/                   replication cohort (definition C)
  clinical/                               landmark model: nested CV, frozen specification, held-out results
  extensions/balanced_27464ba98305edbc/   hash-checked extension snapshot cited by the reports
  figures/                                figures (single-patient force plots are excluded)
  report_en.md, report.md, decisions.md, progress.md
paper/                                    paper LaTeX sources, figures and main.pdf
tools/check_public_release.py             pre-publication check of the files git would publish
external/MIMIC_CODE_COMMIT.txt            pinned MIT-LCP/mimic-code commit (the clone itself is not included)
```

`data/`, `extensions/data/` and `external/mimic-code/` exist only on the analyst's machine.

## Reproducing the analysis

The full guide, with run times and checks, is in [`REPRODUCE.md`](REPRODUCE.md). In short:

1. **Data access.** Obtain your own credentialed access to MIMIC-IV v3.1 on PhysioNet (see [`DATA_ACCESS.md`](DATA_ACCESS.md)) and point `MIMIC_ROOT` to the directory that contains `hosp/` and `icu/`.
2. **Environment and concepts.**
   ```bash
   conda env create -f environment.yml        # environment "dn_repro", Python 3.11
   git clone https://github.com/MIT-LCP/mimic-code.git external/mimic-code
   git -C external/mimic-code checkout 303d26c623dcc9c49cc0f204468d4acc2f063797
   export MIMIC_ROOT=/path/to/mimic-iv-3.1
   conda run -n dn_repro python scripts/00_check_env.py
   ```
3. **Main pipeline.** `bash run_all.sh` (stages 0–7, about 1–1.5 h), then `bash run_stage8.sh` (landmark model, nested CV and checks, about 9–10 h on an 8 GB Mac). The single confirmatory held-out evaluation (`scripts/04i_holdout_clin.py`) is deliberately not re-run: its results are in `results/clinical/`, and `scripts/04j_plots_clin.py` checks a refitted frozen model against them.
4. **Extensions** (after stage 8):
   ```bash
   bash extensions/run_extensions.sh check
   bash extensions/run_extensions.sh prepare
   bash extensions/balanced/run_extensions.sh full --experiments static_lr,static_rf,static_xgb,static_stacking,temporal_lr,temporal_rf,temporal_xgb,burden_lr,burden_rf,burden_xgb,embedding_lr,embedding_rf,embedding_xgb,combined_stacking
   ```
   The scripts locate this project automatically (`BMI5101_V3_ROOT`). The full 14-experiment run takes several hours. A different machine or library build may change the last digits and the result signature; the reports cite the archived snapshot.

Data-free checks: `python tools/check_public_release.py`, `conda run -n dn_repro python tests/check_nadeau_bengio.py`, and `BMI5101_V3_ROOT=$(pwd) conda run -n dn_repro python extensions/balanced/tests/test_tree_ablations.py`.

## Data compliance

This repository contains **no patient-level data**: no raw MIMIC files, identifiers, cohort rows, parquet or DuckDB files, fitted models, predictions, caches, logs or single-patient plots. Everything derived from MIMIC-IV at the patient level stays in git-ignored `data/` directories on the analyst's machine, as required by the PhysioNet data use agreement. Re-running the analysis requires your own authorized access. Run `python tools/check_public_release.py` before publishing any change.

## Acknowledgements

The benchmark study by Huang et al. (2025), MIMIC-IV (Johnson et al., Sci Data 2023) and MIT-LCP/mimic-code. The data licence is separate from the code, and this repository grants no data access.
