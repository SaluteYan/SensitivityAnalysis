#!/usr/bin/env bash
set -euo pipefail

# Balanced paper-oriented Problem 21 experiments for remote Linux servers.
# Override any setting from the command line, for example:
#   WORKERS=24 ANCHOR_REPEATS=10 PARETO_GRID=5 bash scripts/run_problem21_paper_experiments.sh

ENV_NAME="${ENV_NAME:-algorithm_py_env}"
OUTPUT_DIR="${OUTPUT_DIR:-results/problem21_opmwade_sensitivity_balanced}"
INIT_FILE="${INIT_FILE:-init_data/PrG21InitData-target_1_05-none.npz}"
DAMPING_MODE="${DAMPING_MODE:-none}"
TARGET_ANGLE="${TARGET_ANGLE:-1.05}"

# Recommended balanced defaults:
# - 6400 NFEs follows the original problem-21 iteration setting.
# - More repeats are assigned to the normalization anchors because they define
#   all subsequent min-max scales.
# - A 4x4 epsilon grid gives a useful Pareto check without exploding runtime.
ANCHOR_MAX_NFES="${ANCHOR_MAX_NFES:-6400}"
SENSITIVITY_MAX_NFES="${SENSITIVITY_MAX_NFES:-6400}"
PARETO_MAX_NFES="${PARETO_MAX_NFES:-6400}"
ANCHOR_REPEATS="${ANCHOR_REPEATS:-8}"
SENSITIVITY_REPEATS="${SENSITIVITY_REPEATS:-6}"
PARETO_REPEATS="${PARETO_REPEATS:-6}"
PARETO_GRID="${PARETO_GRID:-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-800}"

INITIAL_NP_FACTOR="${INITIAL_NP_FACTOR:-18.0}"
MIN_NP_FACTOR="${MIN_NP_FACTOR:-5.0}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"
if [[ -z "${WORKERS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    CORES="$(nproc)"
  else
    CORES="$(python -c 'import os; print(os.cpu_count() or 1)')"
  fi
  if [[ "${CORES}" -ge 8 ]]; then
    WORKERS="$((CORES - 2))"
  else
    WORKERS="$((CORES > 1 ? CORES - 1 : 1))"
  fi
fi

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_WORKER}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

python -m pip install -r requirements.txt
mkdir -p logs

LOG_FILE="logs/problem21_paper_$(date +%Y%m%d_%H%M%S).log"
echo "Writing log to ${LOG_FILE}"
echo "workers=${WORKERS}, threads_per_worker=${THREADS_PER_WORKER}, progress_interval=${PROGRESS_INTERVAL}"

python experiments/problem21_opmwade_sensitivity.py \
  --output-dir "${OUTPUT_DIR}" \
  --init-file "${INIT_FILE}" \
  --damping-mode "${DAMPING_MODE}" \
  --target-angle "${TARGET_ANGLE}" \
  --anchor-max-nfes "${ANCHOR_MAX_NFES}" \
  --sensitivity-max-nfes "${SENSITIVITY_MAX_NFES}" \
  --pareto-max-nfes "${PARETO_MAX_NFES}" \
  --anchor-repeats "${ANCHOR_REPEATS}" \
  --sensitivity-repeats "${SENSITIVITY_REPEATS}" \
  --pareto-repeats "${PARETO_REPEATS}" \
  --workers "${WORKERS}" \
  --initial-np-factor "${INITIAL_NP_FACTOR}" \
  --min-np-factor "${MIN_NP_FACTOR}" \
  --pareto-grid "${PARETO_GRID}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  --enable-late-enhancements \
  "$@" 2>&1 | tee "${LOG_FILE}"
