#!/usr/bin/env bash
# Feature preparation for the extensions (shared by balanced/). Default: check only; prepare must be requested.
# The strict experiment branch was not completed and its experiment code is not part of this repository.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export BMI5101_V3_ROOT="${BMI5101_V3_ROOT:-$(cd .. && pwd)}"   # repository root = the V3 project
CONDA_BASE="${BMI5101_CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
ENV="${BMI5101_CONDA_ENV:-dn_repro}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
mode="${1:-check}"
if (($#)); then shift; fi
case "$mode" in
  check) conda run --no-capture-output -n "$ENV" python -u scripts/00_preflight.py "$@" ;;
  prepare) conda run --no-capture-output -n "$ENV" python -u scripts/01_prepare_features.py "$@" ;;
  *) printf 'Usage: bash run_extensions.sh [check|prepare] [arguments]\nExperiments: bash balanced/run_extensions.sh full ...\n' >&2; exit 2 ;;
esac
