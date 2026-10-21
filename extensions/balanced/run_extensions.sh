#!/usr/bin/env bash
# 默认只是检查。prepare/pilot/full 都需要用户显式指定。
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
  pilot|full) conda run --no-capture-output -n "$ENV" python -u scripts/02_run_experiments.py --profile "$mode" "$@" ;;
  smoke) conda run --no-capture-output -n "$ENV" python -u tests/smoke_checks.py ;;
  *) printf 'Usage: bash run_extensions.sh [check|prepare|pilot|full|smoke] [arguments]\n' >&2; exit 2 ;;
esac
