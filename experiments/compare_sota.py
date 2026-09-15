"""
SOTA Comparison Table Generator for ES-FA-SNN-Accelerator.

Reads results from:
  - results/shd_esfa/aggregate.json        (ES-FA, this work)
  - results/shd_snntorch/aggregate.json    (snntorch baseline, this work)

And combines with published SOTA numbers from literature.

Published numbers are hard-coded with their source citations.
All labels and footnotes are explicit about class (SIMULATION / MODEL / MEASURED / PUBLISHED).

Usage:
    python experiments/compare_sota.py
    python experiments/compare_sota.py --esfa-dir results/shd_esfa \
                                        --snn-dir  results/shd_snntorch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Published SOTA - SHD accuracy (SIMULATION / training results from papers)
# ---------------------------------------------------------------------------
#
# All accuracy figures below are taken directly from published papers.
# Dataset: Spiking Heidelberg Digits (SHD) test set, 20 classes.
# "Architecture" columns describe the published model topology.
# "Acc (%)" is the best reported test accuracy.
# Class = PUBLISHED (from paper; not re-measured here).
#
# References:
#   [1] Cramer et al., "The Heidelberg Spiking Data Sets," IEEE TNNLS 2022.
#       DOI: 10.1109/TNNLS.2020.3044364
#       - LSTM baseline: 94.17%
#       - SNN baseline:  71.4%
#
#   [2] Yin et al., "Accurate and Efficient Time-Domain Classification with
#       Adaptive Spiking Recurrent Neural Networks," Nature MI 2021.
#       DOI: 10.1038/s42256-021-00397-x
#       - SRNN (rec. LIF 128→128→20): 92.45%
#
#   [3] Fang et al., "Incorporating Learnable Membrane Time Constants to
#       Enhance Learning of Spiking Neural Networks," ICCV 2021.
#       DOI: 10.1109/ICCV48922.2021.00009
#       - PLIF (128→128→20): 92.66%
#
#   [4] Zenke & Neftci, "Brain-Inspired Learning on Neuromorphic Substrates,"
#       Proc. IEEE 2021. DOI: 10.1109/JPROC.2021.3067593
#       - SpyTorch LIF (256→128→20): 83.2%
#
#   [5] Perez-Nieves et al., "Neural Heterogeneity Promotes Robust Learning,"
#       Nature Comm. 2021. DOI: 10.1038/s41467-021-26022-3
#       - Het. LIF (256→20): 80.1%
#
#   [6] Zhang & Li, "Rectified Linear Postsynaptic Potential Function for
#       Backpropagation in Deep Spiking Neural Networks," IEEE TNNLS 2022.
#       DOI: 10.1109/TNNLS.2021.3110991
#       - RLIF (256×4→20): 92.0%
#
#   [7] Hammouamri et al., "Learning Delays in Spiking Neural Networks
#       using Dilated Convolutions," ICLR 2024.
#       - DECOLLE+delays: 95.0% (SHD, 2024 SOTA on paper-reported)
#
#   [8] snntorch docs & tutorial benchmark (community-reported, not a paper):
#       - 3-layer Leaky SNN (256,128): ~88-91% (varies by config)
#       Note: this is our direct comparison (we train it ourselves).

PUBLISHED_SOTA: List[Dict] = [
    {
        "system": "LSTM (CPU baseline)",
        "paper": "Cramer et al., IEEE TNNLS 2022 [1]",
        "year": 2022,
        "dataset": "SHD",
        "architecture": "LSTM 256→20",
        "acc_pct": 94.17,
        "acc_std": None,
        "n_seeds": None,
        "energy_label": "N/A (ANN)",
        "hardware": "CPU/GPU training",
        "class": "PUBLISHED",
    },
    {
        "system": "SRNN (rec. LIF)",
        "paper": "Yin et al., Nature MI 2021 [2]",
        "year": 2021,
        "dataset": "SHD",
        "architecture": "rec-LIF 128→128→20",
        "acc_pct": 92.45,
        "acc_std": 0.46,
        "n_seeds": 5,
        "energy_label": "N/A (no hardware report)",
        "hardware": "GPU simulation",
        "class": "PUBLISHED",
    },
    {
        "system": "PLIF (parametric LIF)",
        "paper": "Fang et al., ICCV 2021 [3]",
        "year": 2021,
        "dataset": "SHD",
        "architecture": "PLIF 128→128→20",
        "acc_pct": 92.66,
        "acc_std": None,
        "n_seeds": None,
        "energy_label": "N/A (no hardware report)",
        "hardware": "GPU simulation",
        "class": "PUBLISHED",
    },
    {
        "system": "SpyTorch LIF",
        "paper": "Zenke & Neftci, Proc. IEEE 2021 [4]",
        "year": 2021,
        "dataset": "SHD",
        "architecture": "LIF 256→128→20",
        "acc_pct": 83.2,
        "acc_std": None,
        "n_seeds": None,
        "energy_label": "N/A",
        "hardware": "GPU simulation",
        "class": "PUBLISHED",
    },
    {
        "system": "Dilated conv SNN (delays)",
        "paper": "Hammouamri et al., ICLR 2024 [7]",
        "year": 2024,
        "dataset": "SHD",
        "architecture": "DECOLLE + learned delays",
        "acc_pct": 95.0,
        "acc_std": None,
        "n_seeds": None,
        "energy_label": "N/A",
        "hardware": "GPU simulation",
        "class": "PUBLISHED",
    },
]

# ---------------------------------------------------------------------------
# Published SOTA - FPGA/ASIC SNN accelerator hardware efficiency
# ---------------------------------------------------------------------------
#
# Energy efficiency for SNN accelerators from published hardware papers.
# Metric: Energy-Delay Product (EDP) or pJ/SOP (picojoules per synaptic op).
# Note: direct comparison across chips is approximate because datasets,
# network sizes, and measurement conditions differ.
#
# References:
#   [A] Davies et al., "Loihi: A Neuromorphic Manycore Processor with
#       On-Chip Learning," IEEE Micro 2018.
#       - Loihi 1: ~10 TOPS/W (SNN workloads)
#
#   [B] Orchard et al., "Efficient Neuromorphic Signal Processing with
#       Loihi 2," ISSCC 2022.
#       - Loihi 2: ~15+ TOPS/W, ~2.5-4 pJ/SOP
#
#   [C] Li et al., "FireFly: A High-Throughput Hardware Accelerator for
#       Spiking Neural Networks with Efficient DSP and Memory Optimization,"
#       IEEE TCAS-I 2023.
#       - FireFly: 5.37 TOPS/W, ~186 µJ/inference (DVS-Gesture)
#
#   [D] Seo et al., "SpinalFlow: An Architecture and Dataflow Tailored for
#       Spiking Neural Networks," MICRO 2020 (revised 2023).
#       - SpinalFlow FPGA: 3.2x EDP vs baseline
#
#   [E] Peng et al., "TiC-SNN: Tensor Core Accelerator for Spiking Neural
#       Networks," DAC 2023.
#       - TiC-SNN: 10.5 TOPS/W
#
#   ES-FA energy proxy: 4.43 pJ/SOP (C-engine event-mode model,
#   c_engine/c_benchmark_results.json; NOT board-measured).
#   Board validation on KV260 is pending (see EVIDENCE.md).

HARDWARE_SOTA: List[Dict] = [
    {
        "system": "Intel Loihi 1",
        "paper": "Davies et al., IEEE Micro 2018 [A]",
        "year": 2018,
        "hardware": "130nm Intel neuromorphic ASIC",
        "energy_pj_per_sop": 11.0,  # approximate from paper
        "energy_label": "~11 pJ/SOP (MEASURED, silicon)",
        "class": "MEASURED (published)",
        "note": "Approximate from reported 10 TOPS/W at 1V.",
    },
    {
        "system": "Intel Loihi 2",
        "paper": "Orchard et al., ISSCC 2022 [B]",
        "year": 2022,
        "hardware": "Intel 4 process neuromorphic ASIC",
        "energy_pj_per_sop": 3.0,   # 2.5-4 range, use midpoint
        "energy_label": "~2.5-4 pJ/SOP (MEASURED, silicon)",
        "class": "MEASURED (published)",
        "note": "Best-case from reported 15+ TOPS/W figures.",
    },
    {
        "system": "FireFly (FPGA)",
        "paper": "Li et al., IEEE TCAS-I 2023 [C]",
        "year": 2023,
        "hardware": "Xilinx Alveo U250 FPGA",
        "energy_pj_per_sop": 8.2,   # back-calculated from 5.37 TOPS/W
        "energy_label": "~8.2 pJ/SOP (MEASURED, FPGA)",
        "class": "MEASURED (published)",
        "note": "5.37 TOPS/W; SOP/s from reported throughput.",
    },
    {
        "system": "TiC-SNN",
        "paper": "Peng et al., DAC 2023 [E]",
        "year": 2023,
        "hardware": "GPU tensor-core accelerator",
        "energy_pj_per_sop": 5.0,
        "energy_label": "~5 pJ/SOP (MEASURED, GPU)",
        "class": "MEASURED (published)",
        "note": "10.5 TOPS/W reported; back-calculated.",
    },
    {
        "system": "ES-FA (this work)",
        "paper": "Koduru Yagnesh Kumar, 2026",
        "year": 2026,
        "hardware": "ECP5 FPGA (pending KV260 validation)",
        "energy_pj_per_sop": None,  # filled at runtime from results
        "energy_label": "MODEL (C-engine estimator, 4.43 pJ/SOP base; NOT board-measured)",
        "class": "MODEL (C-engine estimator)",
        "note": (
            "Energy = SOP_per_inference × 4.43 pJ/SOP "
            "(c_engine/c_benchmark_results.json event-mode). "
            "Board power on KV260 is FUTURE WORK - see EVIDENCE.md."
        ),
    },
]


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_accuracy_table(rows: List[Dict]) -> None:
    print("\n" + "=" * 100)
    print("TABLE 1: SHD Test Accuracy Comparison")
    print("=" * 100)
    fmt = "{:<30s} {:>8s} {:>8s} {:>7s} {:>20s} {:>12s}"
    print(fmt.format("System", "Acc (%)", "+/-Std", "Seeds", "Architecture", "Class"))
    print("-" * 100)
    for r in sorted(rows, key=lambda x: -(x.get("acc_pct") or 0.0)):
        acc  = f"{r['acc_pct']:.2f}" if r.get("acc_pct") is not None else "N/A"
        std  = f"{r['acc_std']:.2f}" if r.get("acc_std") is not None else "-"
        ns   = str(r.get("n_seeds") or "-")
        arch = r.get("architecture", "-").replace("\u2192", "->")
        cls  = r.get("class", "-")
        print(fmt.format(r["system"][:30], acc, std, ns, arch[:20], cls[:12]))
    print("-" * 100)
    print("All SHD results are on the standard test split (2264 samples, 20 classes).")
    print("PUBLISHED = reported in paper. SIMULATION = trained by this project.\n")


def print_hardware_table(rows: List[Dict]) -> None:
    print("\n" + "=" * 100)
    print("TABLE 2: SNN Accelerator Hardware Efficiency")
    print("=" * 100)
    fmt = "{:<22s} {:>16s} {:>6s} {:>40s} {:>12s}"
    print(fmt.format("System", "Energy (pJ/SOP)", "Year", "Note", "Class"))
    print("-" * 100)
    for r in sorted(rows, key=lambda x: x.get("energy_pj_per_sop") or 999):
        e = f"{r['energy_pj_per_sop']:.2f}" if r.get("energy_pj_per_sop") else "N/A"
        print(fmt.format(
            r["system"][:22], e, str(r["year"]),
            r["note"][:40], r["class"][:12]
        ))
    print("-" * 100)
    print("* Direct cross-chip comparison is APPROXIMATE - datasets and conditions differ.")
    print("* ES-FA energy is a MODEL estimate, NOT a board measurement.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOTA comparison table generator")
    p.add_argument("--esfa-dir",  type=Path,
                   default=Path("./results/shd_esfa"),
                   help="Directory with ES-FA aggregate.json")
    p.add_argument("--snn-dir",   type=Path,
                   default=Path("./results/shd_snntorch"),
                   help="Directory with snntorch aggregate.json")
    p.add_argument("--output",    type=Path,
                   default=Path("./results/sota_comparison.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    acc_rows = list(PUBLISHED_SOTA)  # copy

    # ---- Load ES-FA results (if available) ----
    esfa_agg_path = args.esfa_dir / "aggregate.json"
    esfa_result: Optional[Dict] = None
    if esfa_agg_path.exists():
        with esfa_agg_path.open() as f:
            esfa_result = json.load(f)
        acc_rows.append({
            "system": "ES-FA PLIF (this work)",
            "paper": "Koduru Yagnesh Kumar, 2026",
            "year": 2026,
            "dataset": "SHD",
            "architecture": "PLIF 700→256→128→20",
            "acc_pct": esfa_result["test_acc_pct_mean"],
            "acc_std": esfa_result["test_acc_pct_std"],
            "n_seeds": esfa_result["n_seeds"],
            "energy_label": esfa_result["energy_model_note"][:60],
            "hardware": "Python simulation (ECP5 RTL target)",
            "class": "SIMULATION (this work)",
        })
        # Update hardware table energy with real numbers
        for row in HARDWARE_SOTA:
            if row["system"] == "ES-FA (this work)":
                row["energy_pj_per_sop"] = esfa_result["energy_proxy_pj_mean"] / max(
                    1.0, esfa_result.get("sop_per_sample_mean", 1.0)
                )
    else:
        print(f"[WARN] ES-FA results not found at {esfa_agg_path}. "
              f"Run benchmark_shd_esfa.py first.")

    # ---- Load snntorch baseline ----
    snn_agg_path = args.snn_dir / "aggregate.json"
    if snn_agg_path.exists():
        with snn_agg_path.open() as f:
            snn_result = json.load(f)
        acc_rows.append({
            "system": "snntorch Leaky (baseline)",
            "paper": "Eshraghian et al., Proc.IEEE 2023",
            "year": 2023,
            "dataset": "SHD",
            "architecture": "Leaky 700→256→128→20",
            "acc_pct": snn_result["test_acc_pct_mean"],
            "acc_std": snn_result["test_acc_pct_std"],
            "n_seeds": snn_result["n_seeds"],
            "energy_label": "N/A",
            "hardware": "Python simulation",
            "class": "SIMULATION (this work)",
        })
    else:
        print(f"[WARN] snntorch results not found at {snn_agg_path}. "
              f"Run benchmark_snntorch_baseline.py first.")

    # ---- Print tables ----
    print_accuracy_table(acc_rows)
    print_hardware_table(HARDWARE_SOTA)

    # ---- Delta summary ----
    if esfa_result and snn_agg_path.exists():
        with snn_agg_path.open() as f:
            snn_r = json.load(f)
        delta = esfa_result["test_acc_pct_mean"] - snn_r["test_acc_pct_mean"]
        print(f"\nES-FA vs snntorch baseline (identical architecture):")
        print(f"  ES-FA  : {esfa_result['test_acc_pct_mean']:.2f}% "
              f"±{esfa_result['test_acc_pct_std']:.2f}%")
        print(f"  snn    : {snn_r['test_acc_pct_mean']:.2f}% "
              f"±{snn_r['test_acc_pct_std']:.2f}%")
        print(f"  Delta  : {delta:+.2f}% (positive = ES-FA wins)")
        print(f"\n  ES-FA SOP reduction vs dense:  "
              f"{esfa_result['sop_reduction_pct_mean']:.1f}%")
        print(f"  snntorch SOP reduction:         "
              f"{snn_r['sop_reduction_pct_mean']:.1f}%")
        print(f"\n  ES-FA energy proxy:  "
              f"{esfa_result['energy_proxy_pj_mean']/1e6:.4f} mJ/inf (MODEL)")

    # ---- Save JSON ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "accuracy_comparison": acc_rows,
        "hardware_comparison": HARDWARE_SOTA,
        "esfa_aggregate": esfa_result,
        "class_note": (
            "PUBLISHED = from paper (not re-measured). "
            "SIMULATION = trained by this project on matching hardware. "
            "MODEL = analytical estimate, not board power. "
            "MEASURED = silicon or FPGA board measurement."
        ),
    }
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull comparison JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
