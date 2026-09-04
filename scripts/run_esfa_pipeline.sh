#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
MODE="${MODE:-smoke}"
CLOCK_MHZ="${CLOCK_MHZ:-100.0}"
SKIP_VIVADO="${SKIP_VIVADO:-0}"
SKIP_HARDWARE="${SKIP_HARDWARE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PROJECT_ROOT}/scripts/toolchain/bootstrap_xilinx.sh"

cd "${PROJECT_ROOT}"
ARGS=(
  "scripts/run_esfa_pipeline.py"
  "--mode" "${MODE}"
  "--clock-mhz" "${CLOCK_MHZ}"
)
if [[ "${SKIP_VIVADO}" == "1" ]]; then
  ARGS+=("--skip-vivado")
fi
if [[ "${SKIP_HARDWARE}" == "1" ]]; then
  ARGS+=("--skip-hardware")
fi

"${PYTHON_EXE}" "${ARGS[@]}"
