#!/usr/bin/env bash
# Run stages 0-7 in order (conda env: dn_repro). Logs go to results/logs/. Stage 8: run_stage8.sh. See REPRODUCE.md.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs data/interim data/processed   # data/ is git-ignored (patient-level), so a fresh copy has none
echo "== 00_check_env"
conda run --no-capture-output -n dn_repro python -u scripts/00_check_env.py 2>&1 | tee results/logs/00_check_env.log
for s in 01_cohort 01b_outcome_check 02_features 03_preprocess 04_model 04b_calibration 04c_threshold_compare 04d_nested_cv 04e_sensitivity_pre2020 05_report; do
  echo "== $s"
  conda run --no-capture-output -n dn_repro python -u "scripts/$s.py" > "results/logs/$s.log" 2>&1
done
echo "done: see results/report.md"
