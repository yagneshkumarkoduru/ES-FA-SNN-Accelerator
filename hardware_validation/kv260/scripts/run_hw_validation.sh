#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
MODEL_ID="${MODEL_ID:-baseline_paper1}"
SCHEDULER_MODE="${SCHEDULER_MODE:-both}"
CLOCK_MHZ="${CLOCK_MHZ:-100.0}"
SKIP_VIVADO="${SKIP_VIVADO:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${PROJECT_ROOT}/scripts/toolchain/bootstrap_xilinx.sh"

cd "${PROJECT_ROOT}"
ARGS=(
  "hardware_validation/kv260/scripts/run_hw_validation.py"
  "--model-id" "${MODEL_ID}"
  "--scheduler-mode" "${SCHEDULER_MODE}"
  "--clock-mhz" "${CLOCK_MHZ}"
)
if [[ "${SKIP_VIVADO}" == "1" ]]; then
  ARGS+=("--skip-vivado")
fi

"${PYTHON_EXE}" "${ARGS[@]}"

