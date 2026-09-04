#!/usr/bin/env bash
set -euo pipefail

add_if_exists() {
  local p="$1"
  if [[ -d "$p" && ":$PATH:" != *":$p:"* ]]; then
    export PATH="$p:$PATH"
  fi
}

for k in XILINX_VIVADO XILINX_VITIS XILINX_HOME XILINX_INSTALL VIVADO_HOME VITIS_HOME; do
  v="${!k:-}"
  if [[ -n "$v" ]]; then
    add_if_exists "$v/bin"
    add_if_exists "$v"
  fi
done

for root in /tools/Xilinx "$HOME/Xilinx" /opt/Xilinx /mnt/c/AMDDesignTools /c/AMDDesignTools; do
  if [[ -d "$root" ]]; then
    while IFS= read -r d; do
      add_if_exists "$d"
    done < <(find "$root" -maxdepth 3 -type d \( -path "*/Vivado/bin" -o -path "*/Vitis/bin" -o -path "*/Vitis_HLS/bin" \) 2>/dev/null || true)
    while IFS= read -r d; do
      add_if_exists "$d"
    done < <(find "$root" -type d -name bin 2>/dev/null || true)
  fi
done

echo "Xilinx PATH bootstrap complete."
command -v vivado || true
command -v xvlog || true
