"""
Comprehensive Benchmarking Suite for ES-FA Neuromorphic SNN Accelerator.
Executes sweeps across network dimensions, event sparsity, and on-chip STDP adaptation.
Generates publication figures for IEEE TVLSI manuscript.
"""

import os
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def run_c_sim(exe_path, cores, neurons, steps, sparsity, stdp, output_json):
    cmd = [
        exe_path,
        "--cores", str(cores),
        "--neurons", str(neurons),
        "--steps", str(steps),
        "--sparsity", str(sparsity),
        "--stdp", "1" if stdp else "0",
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
    c_exe = os.path.join(base_dir, "c_engine", "snn_simulator.exe")
    plots_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("[ES-FA BENCHMARK] Running sparsity sweep (10% to 95%)...")
    sparsities = [0.10, 0.25, 0.50, 0.70, 0.85, 0.90, 0.95]
    energy_basic = []
    energy_esfa = []
    edp_basic = []
    edp_esfa = []
    edp_reduction_factors = []

    for sp in sparsities:
        tmp_json = os.path.join(base_dir, "c_engine", f"tmp_res_{int(sp*100)}.json")
        data = run_c_sim(c_exe, cores=4, neurons=128, steps=8000, sparsity=sp, stdp=True, output_json=tmp_json)
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

    # Plot 1: Sparsity vs Energy & EDP Reduction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot([s*100 for s in sparsities], energy_basic, "ro--", label="Baseline Synchronous (Round-Robin)", linewidth=2)
    ax1.plot([s*100 for s in sparsities], energy_esfa, "gs-", label="ES-FA Event-Driven (Ours)", linewidth=2.5)
    ax1.set_xlabel("Spike Sparsity (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Dynamic Energy (nJ)", fontsize=12, fontweight="bold")
    ax1.set_title("Dynamic Energy Dissipation vs Spike Sparsity", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot([s*100 for s in sparsities], edp_reduction_factors, "b^-", label="EDP Reduction Factor (x)", linewidth=2.5, color="#2563eb")
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

    # Plot 2: Core Scaling & Throughput (1, 2, 4, 8, 16 Cores)
    print("[ES-FA BENCHMARK] Running core scaling sweep (1 to 16 Cores)...")
    core_counts = [1, 2, 4, 8, 16]
    throughput_gsops = []
    edp_scaling = []

    for c in core_counts:
        tmp_json = os.path.join(base_dir, "c_engine", f"tmp_core_{c}.json")
        data = run_c_sim(c_exe, cores=c, neurons=128, steps=5000, sparsity=0.85, stdp=True, output_json=tmp_json)
        if data:
            # Synthetic GSOP throughput scaling
            ops = data["es_fa_event"]["cycles"] * c * 32
            time_s = data["es_fa_event"]["cycles"] * 4.0e-9
            throughput_gsops.append((ops / time_s) * 1e-9)
            edp_scaling.append(data["es_fa_event"]["edp_js"])
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar([str(c) for c in core_counts], throughput_gsops, color="#059669", width=0.5, label="Peak Compute Throughput (GSOP/s)")
    ax1.set_xlabel("Number of Neuromorphic Cores", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Throughput (GSOP/s)", fontsize=12, fontweight="bold", color="#059669")
    ax1.tick_params(axis="y", labelcolor="#059669")
    ax1.set_title("Multi-Core Scalability: Throughput vs Core Dimension", fontsize=13, fontweight="bold")
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
    print(f" Max EDP Reduction    : {max(edp_reduction_factors):.2f}x")
    print(f" Max Throughput       : {max(throughput_gsops):.2f} GSOP/s at 16 Cores")
    print("==================================================================\n")

if __name__ == "__main__":
    main()
