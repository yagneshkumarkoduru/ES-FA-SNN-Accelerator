#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TIME_STEPS="${TIME_STEPS:-16}"
LAYER_TO_MAP="${LAYER_TO_MAP:-fc1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACTS_DIR="${PROJECT_ROOT}/software/artifacts"
EXPORT_DIR="${ARTIFACTS_DIR}/export_int8"
MAP_DIR="${ARTIFACTS_DIR}/fpga_mem"
BEST_CKPT="${ARTIFACTS_DIR}/checkpoints/best_model.pt"

cd "${PROJECT_ROOT}"

echo "Running SNN training..."
"${PYTHON_EXE}" software/train_snn.py \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --time-steps "${TIME_STEPS}" \
  --output-dir "${ARTIFACTS_DIR}"

echo "Exporting INT8 weights..."
"${PYTHON_EXE}" software/export_weights.py \
  --checkpoint "${BEST_CKPT}" \
  --output-dir "${EXPORT_DIR}"

echo "Mapping exported weights to banked FPGA memory files..."
"${PYTHON_EXE}" software/map_weights_to_banks.py \
  --export-dir "${EXPORT_DIR}" \
  --layer "${LAYER_TO_MAP}" \
  --num-banks 2 \
  --bank-depth 64 \
  --out-dir "${MAP_DIR}"

echo "Software pipeline complete."
echo "Artifacts: ${ARTIFACTS_DIR}"
