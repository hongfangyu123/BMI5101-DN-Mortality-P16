#!/usr/bin/env bash
# Default is check only. prepare/pilot/full must be requested explicitly by the user.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export BMI5101_V3_ROOT="${BMI5101_V3_ROOT:-$(cd ../.. && pwd)}"   # repository root = the V3 project
CONDA_BASE="${BMI5101_CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
ENV="${BMI5101_CONDA_ENV:-dn_repro}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
mode="${1:-check}"
if (($#)); then shift; fi
case "$mode" in
  check) conda run --no-capture-output -n "$ENV" python -u scripts/00_preflight.py "$@" ;;
  prepare) printf "Features are shared with the parent directory; run: bash extensions/run_extensions.sh prepare\n" >&2; exit 2 ;;
  holdout) conda run --no-capture-output -n "$ENV" python -B -u holdout/run_holdout.py "$@" ;;
  repeated-holdout) conda run --no-capture-output -n "$ENV" python -B -u repeated_holdout/run_repeated_holdout.py "$@" ;;
  paired-ci) conda run --no-capture-output -n "$ENV" python -B -u repeated_holdout/paired_confidence_intervals.py "$@" ;;
  pilot|full) conda run --no-capture-output -n "$ENV" python -u scripts/02_run_experiments.py --profile "$mode" "$@" ;;
  smoke) conda run --no-capture-output -n "$ENV" python -u tests/smoke_checks.py ;;
  *) printf 'Usage: bash run_extensions.sh [check|prepare|pilot|full|holdout|repeated-holdout|paired-ci|smoke] [arguments]\n' >&2; exit 2 ;;
esac
