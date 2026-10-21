#!/usr/bin/env bash
# Run stage 8 in order (conda env: dn_repro), after run_all.sh. Logs go to results/logs/. See REPRODUCE.md.
# Total about 9-10 h on an 8 GB Apple-silicon Mac (04f is the long part). 04i (the single confirmatory evaluation of the
# CLIN held-out set) is deliberately NOT run: its result is results/clinical/holdout_clin.csv in the repository, and 04j
# re-predicts the held-out set with the frozen model and asserts that every metric equals that file.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs data/interim data/processed   # data/ is git-ignored (patient-level), so a fresh copy has none

step() {  # step <log name> <script> [args...]
  local log="$1" script="$2"; shift 2
  echo "== $(date '+%H:%M:%S') $script $* -> results/logs/$log.log"
  conda run --no-capture-output -n dn_repro python -u "scripts/$script.py" "$@" > "results/logs/$log.log" 2>&1
}

echo "== 00_check_env"
conda run --no-capture-output -n dn_repro python -u scripts/00_check_env.py 2>&1 | tee results/logs/00_check_env.log

step 01c_cohort_clin            01c_cohort_clin                                    # CLIN cohort and split (D8.13)
step 02b_comorb_time            02b_comorb_time                                    # cohort C comorbidity timing (D8.12)
step 02b_comorb_clin            02b_comorb_time --track clinical                   # CLIN comorbidities (D8.3)
step 02_features_clin           02_features --track clinical                       # CLIN features (D8.4)
step 03c_table1_clin            03c_table1_clin                                    # CLIN Table 1 (D8.17)
step 04f_nested_full_C          04f_nested_full --track replication --cohort C    # ~1.5 h (D8.11)
step 04f_nested_full_C_comorb_time 04f_nested_full --track replication --cohort C --variant comorb_time   # ~1.2 h
step 04f_nested_full_clin       04f_nested_full --track clinical                   # ~2.2 h (D8.14)
step 04f_nested_full_clin_prior_only 04f_nested_full --track clinical --variant prior_only --models tuned:LR,tuned:RF  # ~1.8 h
step 04f_nested_full_clin_no_wbc_diff 04f_nested_full --track clinical --variant no_wbc_diff --models tuned:LR,tuned:RF  # ~1.5 h
step 04g_holdout_C              04g_holdout_C                                      # cohort C secondary held-out estimate
if [ -f data/interim/final_clin/final_clin.pkl ]; then
  echo "== 04h_final_clin skipped: data/interim/final_clin/final_clin.pkl already exists (the frozen model is fitted once)"
else
  step 04h_final_clin           04h_final_clin                                     # fit the frozen CLIN model (D8.14)
fi
step 04j_plots_clin             04j_plots_clin                                     # re-predict held-out set, assert = holdout_clin.csv
step 05_report                  05_report
echo "done: see results/report.md; check reproduction with: git status --short results/ && git diff --stat results/"
