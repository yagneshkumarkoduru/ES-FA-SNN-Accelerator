"""
Comprehensive Benchmarking Suite for ES-FA Neuromorphic SNN Accelerator.
Executes sweeps across network dimensions, event sparsity, and on-chip STDP adaptation.
Generates publication figures for IEEE TVLSI manuscript.

Requires the C99 simulator (snn_simulator.exe) from
implementations/v2_c99_cycle_accurate_engine/. It is compiled here with GCC
when a toolchain is available; otherwise the sweep is skipped with a clear
notice instead of crashing.
"""

import os
import sys
import shutil
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

ENGINE_DIR = os.path.join("implementations", "v2_c99_cycle_accurate_engine")


def ensure_simulator(engine_dir):
    """Compile snn_simulator.exe with GCC; return the exe path or None."""
    exe_path = os.path.join(engine_dir, "snn_simulator.exe")
    gcc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if gcc is None:
        print("[ES-FA BENCHMARK] No C compiler found on PATH; "
              "snn_simulator cannot be built and this sweep is skipped.")
        return None
    print("[ES-FA BENCHMARK] Compiling the C99 cycle-accurate simulator...")
    res = subprocess.run(
        [gcc, "-O3", "-Wall", "-Wextra", "-std=c99",
         "main.c", "snn_engine.c", "-o", "snn_simulator.exe", "-lm"],
        cwd=engine_dir, capture_output=True, text=True,
    )
    if res.returncode != 0 or not os.path.exists(exe_path):
        print(f"[ES-FA BENCHMARK] Simulator compilation failed: {res.stderr.strip()}")
        return None
    return exe_path


def run_c_sim(exe_path, cores, neurons, steps, sparsity, output_json):
    cmd = [
        exe_path,
        "--cores", str(cores),
        "--neurons", str(neurons),
        "--steps", str(steps),
        "--sparsity", str(sparsity),
        "--output", output_json
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running C sim: {res.stderr}")
        return None
    with open(output_json, "r") as f:
        return json.load(f)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine_dir = os.path.join(base_dir, ENGINE_DIR)
    plots_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    c_exe = ensure_simulator(engine_dir)

    print("[ES-FA BENCHMARK] Running sparsity sweep (10% to 95%)...")
    sparsities = [0.10, 0.25, 0.50, 0.70, 0.85, 0.90, 0.95]
    energy_basic = []
    energy_esfa = []
    edp_basic = []
    edp_esfa = []
    edp_reduction_factors = []

    for sp in sparsities:
        tmp_json = os.path.join(engine_dir, f"tmp_res_{int(sp*100)}.json")
        data = run_c_sim(c_exe, cores=4, neurons=128, steps=8000, sparsity=sp, output_json=tmp_json) if c_exe else None
        if data:
            eb = data["basic"]["dynamic_energy_nj"]
            ee = data["es_fa_event"]["dynamic_energy_nj"]
            edpb = data["basic"]["edp_js"]
            edpe = data["es_fa_event"]["edp_js"]
            energy_basic.append(eb)
            energy_esfa.append(ee)
            edp_basic.append(edpb)
            edp_esfa.append(edpe)
            edp_reduction_factors.append(edpb / edpe)
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    if edp_reduction_factors:
        # Plot 1: Sparsity vs Energy & EDP Reduction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot([s*100 for s in sparsities][:len(energy_basic)], energy_basic, "ro--", label="Baseline Synchronous (Round-Robin)", linewidth=2)
        ax1.plot([s*100 for s in sparsities][:len(energy_esfa)], energy_esfa, "gs-", label="ES-FA Event-Driven (Ours)", linewidth=2.5)
        ax1.set_xlabel("Spike Sparsity (%)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Dynamic Energy (nJ)", fontsize=12, fontweight="bold")
        ax1.set_title("Dynamic Energy Dissipation vs Spike Sparsity", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.plot([s*100 for s in sparsities][:len(edp_reduction_factors)], edp_reduction_factors, "b^-", label="EDP Reduction Factor (x)", linewidth=2.5, color="#2563eb")
        ax2.axhline(1.0, color="gray", linestyle=":")
        ax2.set_xlabel("Spike Sparsity (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Energy-Delay Product (EDP) Reduction (x)", fontsize=12, fontweight="bold")
        ax2.set_title("EDP Advantage Frontier vs Baseline", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plot1_path = os.path.join(plots_dir, "fig_c_engine_sparsity_speedup.png")
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved {plot1_path}")
    else:
        print("[ES-FA BENCHMARK] No simulator data available; sparsity/EDP plot skipped.")

    # Plot 2: Core Scaling & Throughput (1, 2, 4, 8, 16 Cores)
    print("[ES-FA BENCHMARK] Running core scaling sweep (1 to 16 Cores)...")
    core_counts = [1, 2, 4, 8, 16]
    throughput_gsops = []
    edp_scaling = []
    measured_core_counts = []

    for c in core_counts:
        tmp_json = os.path.join(engine_dir, f"tmp_core_{c}.json")
        data = run_c_sim(c_exe, cores=c, neurons=128, steps=5000, sparsity=0.85, output_json=tmp_json) if c_exe else None
        if data:
            measured_core_counts.append(c)
            # SYNTHETIC throughput model: peak SOP rate estimated from the
            # cycle count under an assumed 32 SOP/cycle/core issue rate and a
            # 4 ns cycle. This is a scaling MODEL estimate, not a measurement
            # of a synthesized design or board run.
            ops = data["es_fa_event"]["cycles"] * c * 32
            time_s = data["es_fa_event"]["cycles"] * 4.0e-9
            throughput_gsops.append((ops / time_s) * 1e-9)
            edp_scaling.append(data["es_fa_event"]["edp_js"])
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    if throughput_gsops:
        print("[CAVEAT] The throughput figures below are a synthetic scaling model "
              "(cycles x cores x 32 SOPs at 4 ns), NOT measured silicon throughput. "
              "Treat them as model estimates only.")

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar([str(c) for c in measured_core_counts], throughput_gsops, color="#059669", width=0.5, label="Peak Compute Throughput (GSOP/s, model estimate)")
        ax1.set_xlabel("Number of Neuromorphic Cores", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Throughput (GSOP/s)", fontsize=12, fontweight="bold", color="#059669")
        ax1.tick_params(axis="y", labelcolor="#059669")
        ax1.set_title("Multi-Core Scalability: Throughput vs Core Dimension (model estimate)", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)

        plot2_path = os.path.join(plots_dir, "fig_general_asic_fpga_edp_scaling.png")
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved {plot2_path}")

        # Summary table
        print("\n==================================================================")
        print("      ES-FA GENERAL ARCHITECTURE BENCHMARK VERIFICATION COMPLETE   ")
        print("==================================================================")
        print(f" Tested Sparsities    : {sparsities}")
        if edp_reduction_factors:
            print(f" Max EDP Reduction    : {max(edp_reduction_factors):.2f}x (cycle-accurate model)")
        print(f" Max Throughput       : {max(throughput_gsops):.2f} GSOP/s at {measured_core_counts[throughput_gsops.index(max(throughput_gsops))]} Cores (SYNTHETIC model estimate)")
        print("==================================================================\n")
    else:
        print("[ES-FA BENCHMARK] No simulator data available; core scaling plot skipped.")


if __name__ == "__main__":
    main()
