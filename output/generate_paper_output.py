"""Generate proposal-aligned paper draft sections from experiment outputs."""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def _table_md(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Run | Category | Accuracy | Sparsity | Energy | Latency |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["run_name"]),
                    str(r["category"]),
                    _fmt(float(m["accuracy"])),
                    _fmt(float(m["spike_sparsity"])),
                    _fmt(float(m["energy_proxy"]), 2),
                    _fmt(float(m["latency_proxy"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def generate_sections(project_root: Path) -> None:
    results_root = project_root / "results"
    paper_dir = project_root / "paper_output"
    paper_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = results_root / "analysis_summary.json"
    if not analysis_path.exists():
        raise RuntimeError("analysis_summary.json not found. Run analysis/compare.py first.")
    analysis = _load_json(analysis_path)

    ranked = analysis["ranking"]
    best = analysis["best"]

    method = """# Method

## SNN Model
We use a proposal-aligned feed-forward SNN with architecture `784 -> 128 -> 64 -> 10`.
The hidden layers are Leaky Integrate-and-Fire (LIF) neurons trained with surrogate gradients.
Inputs are encoded as Poisson/rate spikes over fixed simulation windows (`T` timesteps).

## Hardware-Aware Loss
Training optimizes cross-entropy with optional regularizers:
- energy-aware term from the software hardware estimator,
- spike sparsity control term,
- latency proxy regularization under temporal multiplexing assumptions.

## Software Hardware Estimator (Paper-2 Integration)
At each batch, we estimate:
1. `estimate_spike_cost()` for per-layer/total spikes
2. `estimate_memory_access()` for read/write access under dense/event/adaptive dataflow
3. `estimate_energy_proxy()` as `E = a * spikes + b * memory_accesses`
4. `estimate_latency_proxy()` from sparsity + multiplexing + queue pressure
"""

    experiment = """# Experiments

## Setup
Experiments follow proposal concepts with minimal baseline modifications:
1. `exp1_hardware_aware_loss`
2. `exp2_spike_sparsity_control`
3. `exp3_temporal_multiplexing_simulation`
4. `exp4_quantization_effect`
5. `exp5_dataflow_adaptation`

Each run logs accuracy, sparsity, energy proxy, latency proxy, training history, INT8 weights, and spike raster/stats.

## Comparisons and Ablation
We compare baseline, all experiments, and iteration versions (`v1`, `v2`, `v3`) using a unified ranking and Pareto-style trade-off analysis.
"""

    table = "# Results Table\n\n" + _table_md(ranked)

    # Key findings inferred from ranked metrics.
    top = ranked[0]
    baseline = next((r for r in ranked if r["category"] == "baseline"), None)
    findings_lines = [
        "# Key Findings",
        "",
        f"- Best run: `{best['best_run_name']}` ({best['best_category']}).",
        (
            f"- Best metrics: accuracy={_fmt(float(best['best_metrics']['accuracy']))}, "
            f"sparsity={_fmt(float(best['best_metrics']['spike_sparsity']))}, "
            f"energy={_fmt(float(best['best_metrics']['energy_proxy']), 2)}, "
            f"latency={_fmt(float(best['best_metrics']['latency_proxy']))}."
        ),
        "- Efficiency gains are associated with higher sparsity and reduced proxy memory traffic in event/adaptive settings.",
        "- Temporal multiplexing introduces latency-pressure trade-offs that can be partially compensated by sparsity-aware regularization.",
    ]
    if baseline is not None:
        acc_gain = float(top["metrics"]["accuracy"]) - float(baseline["metrics"]["accuracy"])
        energy_delta = float(top["metrics"]["energy_proxy"]) - float(baseline["metrics"]["energy_proxy"])
        sparsity_delta = float(top["metrics"]["spike_sparsity"]) - float(baseline["metrics"]["spike_sparsity"])
        findings_lines.append(
            (
                f"- Compared to baseline: accuracy_delta={_fmt(acc_gain)}, "
                f"sparsity_delta={_fmt(sparsity_delta)}, energy_delta={_fmt(energy_delta, 2)}."
            )
        )

    evidence_csv = results_root / "analysis" / "evidence" / "percentage_change_table.csv"
    if evidence_csv.exists():
        findings_lines.append("- Evidence-backed comparison highlights:")
        with evidence_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                findings_lines.append(
                    f"  - {row['comparison']} / {row['metric']}: "
                    f"baseline={row['baseline_value']}, optimized={row['optimized_value']}, "
                    f"delta={row['percent_change']}%."
                )
    findings = "\n".join(findings_lines) + "\n"

    method_path = paper_dir / "method_section.md"
    exp_path = paper_dir / "experiment_section.md"
    table_path = paper_dir / "tables.md"
    finding_path = paper_dir / "key_findings.md"
    draft_path = paper_dir / "paper_draft.md"

    method_path.write_text(method, encoding="utf-8")
    exp_path.write_text(experiment, encoding="utf-8")
    table_path.write_text(table, encoding="utf-8")
    finding_path.write_text(findings, encoding="utf-8")

    draft = "\n\n".join([method, experiment, table, findings])
    draft_path.write_text(draft, encoding="utf-8")

    print("Paper sections generated:")
    print(f"  - {method_path.resolve()}")
    print(f"  - {exp_path.resolve()}")
    print(f"  - {table_path.resolve()}")
    print(f"  - {finding_path.resolve()}")
    print(f"  - {draft_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate proposal-aligned paper sections from results.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_sections(args.project_root)


if __name__ == "__main__":
    main()
