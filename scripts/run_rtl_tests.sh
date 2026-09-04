#!/usr/bin/env bash
set -euo pipefail

IVERILOG_EXE="${IVERILOG_EXE:-iverilog}"
VVP_EXE="${VVP_EXE:-vvp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARDWARE_ROOT="${PROJECT_ROOT}/hardware"
OUT_DIR="${OUT_DIR:-${HARDWARE_ROOT}/sim_out}"

mkdir -p "${OUT_DIR}"
cd "${HARDWARE_ROOT}"

RTL_SOURCES=(
  "memory/neuron_bram.v"
  "memory/weight_bram_bank.v"
  "compute/lif_neuron_pe.v"
  "routing/spike_router.v"
  "scheduler/basic_scheduler.v"
  "scheduler/event_queue.v"
  "scheduler/advanced_scheduler.v"
  "top/snn_top.v"
)

TB_LIST=(
  "tb_neuron_bram"
  "tb_weight_bram_bank"
  "tb_lif_neuron_pe"
  "tb_spike_router"
  "tb_basic_scheduler"
  "tb_event_queue"
  "tb_advanced_scheduler"
  "tb_top"
)

for tb in "${TB_LIST[@]}"; do
  tb_file="tb/${tb}.v"
  sim_out="${OUT_DIR}/${tb}.out"
  echo "Compiling ${tb_file}"
  "${IVERILOG_EXE}" -g2012 -o "${sim_out}" "${RTL_SOURCES[@]}" "${tb_file}"
  echo "Running ${tb}"
  "${VVP_EXE}" "${sim_out}"
done

echo "All RTL tests completed."
