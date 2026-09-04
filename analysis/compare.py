"""Compare baseline, experiments, and iterations with plots and ranking outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1e-12:
        return [0.5 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


def collect_runs(results_root: Path) -> List[Dict[str, Any]]:
    records = []

    baseline = results_root / "baseline" / "metrics.json"
    if baseline.exists():
        data = _load_json(baseline)
        data["_source"] = str(baseline)
        records.append(data)

    for metrics in sorted((results_root / "experiments").glob("*/metrics.json")):
        data = _load_json(metrics)
        data["_source"] = str(metrics)
        records.append(data)

    for metrics in sorted((results_root / "iterations").glob("*/metrics.json")):
        data = _load_json(metrics)
        data["_source"] = str(metrics)
        records.append(data)

    return records


def rank_runs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    acc = [float(r["metrics"]["accuracy"]) for r in records]
    spars = [float(r["metrics"]["spike_sparsity"]) for r in records]
    energy = [float(r["metrics"]["energy_proxy"]) for r in records]
    latency = [float(r["metrics"]["latency_proxy"]) for r in records]

    spars_n = _normalize(spars)
    energy_n = _normalize(energy)
    latency_n = _normalize(latency)

    ranked = []
    for i, r in enumerate(records):
        score = 0.70 * acc[i] + 0.20 * spars_n[i] - 0.07 * energy_n[i] - 0.03 * latency_n[i]
        rr = dict(r)
        rr["ranking_score"] = float(score)
        ranked.append(rr)

    ranked.sort(key=lambda x: x["ranking_score"], reverse=True)
    return ranked


def _write_ranking_files(ranked: List[Dict[str, Any]], results_root: Path) -> None:
    lines = ["rank,run_name,category,accuracy,spike_sparsity,energy_proxy,latency_proxy,ranking_score"]
    for i, r in enumerate(ranked, start=1):
        m = r["metrics"]
        lines.append(
            ",".join(
                [
                    str(i),
                    str(r["run_name"]),
                    str(r["category"]),
                    f"{float(m['accuracy']):.6f}",
                    f"{float(m['spike_sparsity']):.6f}",
                    f"{float(m['energy_proxy']):.6f}",
                    f"{float(m['latency_proxy']):.6f}",
                    f"{float(r['ranking_score']):.6f}",
                ]
            )
        )
    (results_root / "ranking_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_lines = [
        "| Rank | Run | Category | Accuracy | Sparsity | Energy | Latency | Score |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, start=1):
        m = r["metrics"]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(r["run_name"]),
                    str(r["category"]),
                    f"{float(m['accuracy']):.4f}",
                    f"{float(m['spike_sparsity']):.4f}",
                    f"{float(m['energy_proxy']):.2f}",
                    f"{float(m['latency_proxy']):.4f}",
                    f"{float(r['ranking_score']):.4f}",
                ]
            )
            + " |"
        )
    (results_root / "ranking_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _scatter_plot(
    xs: List[float],
    ys: List[float],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, alpha=0.85)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _load_history(history_path: str) -> Optional[Dict[str, Any]]:
    p = Path(history_path)
    if not p.exists():
        return None
    return _load_json(p)


def _plot_training_curves(records: List[Dict[str, Any]], best_run: Dict[str, Any], plots_dir: Path) -> None:
    baseline = None
    for r in records:
        if str(r["category"]) == "baseline":
            baseline = r
            break

    candidates = [x for x in [baseline, best_run] if x is not None]
    if not candidates:
        return

    plt.figure(figsize=(9, 6))
    for run in candidates:
        hist_file = run["artifacts"]["history"]
        history = _load_history(hist_file)
        if not history:
            continue
        epochs = [h["epoch"] for h in history["history"]]
        val_acc = [h["val"]["accuracy"] for h in history["history"]]
        plt.plot(epochs, val_acc, marker="o", label=f"{run['run_name']} val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Training Curves (Validation Accuracy)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=180)
    plt.close()


def _plot_raster(run: Dict[str, Any], plots_dir: Path, suffix: str) -> None:
    export_dir = Path(run["artifacts"]["export_dir"])
    raster_npz = export_dir / "spike_raster.npz"
    if not raster_npz.exists():
        return
    data = np.load(raster_npz)
    l1 = data["layer1"]
    l2 = data["layer2"]

    # Use binary image for compact raster visualization.
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(l1.T, aspect="auto", cmap="Greys", interpolation="nearest")
    plt.title(f"{run['run_name']} L1 Raster")
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.subplot(1, 2, 2)
    plt.imshow(l2.T, aspect="auto", cmap="Greys", interpolation="nearest")
    plt.title(f"{run['run_name']} L2 Raster")
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.savefig(plots_dir / f"spike_raster_{suffix}.png", dpi=180)
    plt.close()


def analyze(results_root: Path) -> Dict[str, Any]:
    records = collect_runs(results_root)
    if not records:
        raise RuntimeError("No result files found. Run baseline/experiments/iterations first.")

    ranked = rank_runs(records)
    best = ranked[0]
    _write_ranking_files(ranked, results_root)

    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels = [r["run_name"] for r in ranked]
    acc = [float(r["metrics"]["accuracy"]) for r in ranked]
    spars = [float(r["metrics"]["spike_sparsity"]) for r in ranked]
    energy = [float(r["metrics"]["energy_proxy"]) for r in ranked]
    latency = [float(r["metrics"]["latency_proxy"]) for r in ranked]

    _scatter_plot(
        xs=energy,
        ys=acc,
        labels=labels,
        xlabel="Energy Proxy",
        ylabel="Accuracy",
        title="Accuracy vs Energy",
        out_path=plots_dir / "accuracy_vs_energy.png",
    )
    _scatter_plot(
        xs=spars,
        ys=acc,
        labels=labels,
        xlabel="Spike Sparsity",
        ylabel="Accuracy",
        title="Sparsity vs Accuracy",
        out_path=plots_dir / "sparsity_vs_accuracy.png",
    )
    _scatter_plot(
        xs=spars,
        ys=latency,
        labels=labels,
        xlabel="Spike Sparsity",
        ylabel="Latency Proxy",
        title="Latency vs Sparsity",
        out_path=plots_dir / "latency_vs_sparsity.png",
    )

    _plot_training_curves(records, best, plots_dir)
    _plot_raster(best, plots_dir, suffix="best")
    baseline = next((r for r in records if r["category"] == "baseline"), None)
    if baseline is not None:
        _plot_raster(baseline, plots_dir, suffix="baseline")

    best_summary = {
        "best_run_name": best["run_name"],
        "best_category": best["category"],
        "best_score": best["ranking_score"],
        "best_metrics": best["metrics"],
        "num_runs_compared": len(ranked),
    }
    with (results_root / "best_model.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    analysis_summary = {
        "best": best_summary,
        "ranking": [
            {
                "rank": i + 1,
                "run_name": r["run_name"],
                "category": r["category"],
                "score": r["ranking_score"],
                "metrics": r["metrics"],
            }
            for i, r in enumerate(ranked)
        ],
    }
    with (results_root / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)
    return analysis_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all runs and generate plots.")
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory containing baseline/experiments/iterations results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.results_root)
    print("Analysis complete.")
    print(f"Best run: {summary['best']['best_run_name']}")
    print(f"Plots: {(args.results_root / 'plots').resolve()}")


if __name__ == "__main__":
    main()

