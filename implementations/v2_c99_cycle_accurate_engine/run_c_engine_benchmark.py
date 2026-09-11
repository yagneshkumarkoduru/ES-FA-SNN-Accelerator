#!/usr/bin/env python3
"""
=============================================================================
Run the C99 Cycle-Accurate Engine Benchmarks
Project: ES-FA Neuromorphic Accelerator (C99 Cycle-Accurate Engine)
Author: Yagnesh Kumar Koduru (Esthien Labs)

Builds and runs BOTH C99 binaries when a GCC toolchain is available:
  1. snn_simulator.exe     - multi-core cycle-accurate LIF/STDP simulation,
                             writes c_benchmark_results.json
  2. spike_attn_bench.exe  - Spike-Driven FlashAttention operation/energy
                             benchmark, prints its telemetry to stdout

If GCC is unavailable, the pre-built spike_attn_bench.exe (if present) is run
and its stdout is reported; the simulator JSON is skipped with a clear notice
rather than reporting numbers from a binary that was not run.
=============================================================================
"""

import os
import shutil
import subprocess
import sys
import time

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_logged(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print("[stderr] " + result.stderr.strip())
    return result.returncode == 0


def main():
    print("=" * 70)
    print("  ES-FA C99 CYCLE-ACCURATE ENGINE SIMULATION BENCHMARK")
    print("  Author: Yagnesh Kumar Koduru | Esthien Labs")
    print("=" * 70)

    gcc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")

    json_path = os.path.join(ENGINE_DIR, "c_benchmark_results.json")

    # ------------------------------------------------------------------
    # 1. Cycle-accurate multi-core simulator (produces the JSON telemetry)
    # ------------------------------------------------------------------
    sim_exe = os.path.join(ENGINE_DIR, "snn_simulator.exe")
    sim_built = False
    if gcc is not None:
        print("[C99 Runner] Compiling snn_simulator (main.c + snn_engine.c, -O3 C99)...")
        ok = run_logged(
            [gcc, "-O3", "-Wall", "-Wextra", "-std=c99", "main.c", "snn_engine.c",
             "-o", "snn_simulator.exe", "-lm"],
            ENGINE_DIR,
        )
        sim_built = ok and os.path.exists(sim_exe)
        if not sim_built:
            print("[C99 Runner] Simulator compilation failed; skipping its run.")
    else:
        print("[C99 Runner] No C compiler found on PATH; snn_simulator cannot be built.")

    if sim_built:
        print("[C99 Runner] Running snn_simulator (4 cores, 128 neurons/core, "
              "10000 timesteps, 85% sparsity)...")
        start = time.perf_counter()
        ok = run_logged(
            [sim_exe, "--cores", "4", "--neurons", "128", "--steps", "10000",
             "--sparsity", "0.85", "--output", "c_benchmark_results.json"],
            ENGINE_DIR,
        )
        elapsed = time.perf_counter() - start
        print(f"[C99 Runner] Simulator finished in {elapsed * 1000:.1f} ms")
        if ok and os.path.exists(json_path):
            import json
            with open(json_path, "r") as f:
                data = json.load(f)
            print("\n--- Verified Benchmark Telemetry (c_benchmark_results.json) ---")
            for section in ("basic", "es_fa_event", "es_fa_stdp"):
                s = data.get(section, {})
                print(f"  [{section}]")
                for k, v in s.items():
                    print(f"    {k:24s}: {v}")
        else:
            print("[C99 Runner] Simulator ran but produced no JSON; "
                  "no results are reported from a binary that did not execute.")
    else:
        if os.path.exists(json_path):
            print(f"[C99 Runner] NOTE: {os.path.basename(json_path)} is a stale artifact "
                  "from a previous compiler run and is NOT re-reported here.")

    # ------------------------------------------------------------------
    # 2. Spike-Driven FlashAttention operation benchmark (stdout telemetry)
    # ------------------------------------------------------------------
    attn_exe = os.path.join(ENGINE_DIR, "spike_attn_bench.exe")
    if gcc is not None:
        print("[C99 Runner] Compiling spike_attn_bench (spike_attention.c, -O3 C99)...")
        run_logged([gcc, "-O3", "-std=c99", "spike_attention.c", "-o",
                    "spike_attn_bench.exe", "-lm"], ENGINE_DIR)
    if os.path.exists(attn_exe):
        print("[C99 Runner] Running SD-FlashAttention benchmark...")
        run_logged([attn_exe], ENGINE_DIR)
    else:
        print("[C99 Runner] spike_attn_bench binary unavailable and no compiler "
              "found; SD-FlashAttention benchmark skipped.")

    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
