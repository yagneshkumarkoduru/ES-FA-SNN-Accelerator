"""Multi-seed reproducibility harness for the training/estimator loop.

Re-runs the best configuration from results/analysis_summary.json across
several seeds and reports mean +/- std of accuracy, spike sparsity, and the
energy proxy (plus the remaining estimator metrics), so variance can be
reported alongside the single-seed results.

Torch is required for the training loop itself. The aggregation logic is
numpy-only so it stays unit-testable in torch-free environments (see
tests/test_multiseed_aggregation.py); the torch import is guarded in main()
and the harness exits cleanly with status 0 when torch is missing, matching
the CI pattern where only numpy/pytest are installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_SEEDS: Tuple[int, ...] = (11, 22, 33, 44, 55)
# Reduced from the 5 epochs used by the single-seed runs to keep the 5-seed
# sweep runtime sane; override with --epochs.
DEFAULT_EPOCHS = 2

# Metrics aggregated in the summary (order defines the README table order).
METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "spike_sparsity",
    "energy_proxy",
    "spike_activity",
    "latency_proxy",
    "memory_accesses",
)


def aggregate_seed_metrics(per_seed_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metric dictionaries collected across seeds.

    Pure numpy, torch-free. For each metric key present in at least one
    dictionary the function reports the arithmetic mean, the sample standard
    deviation (ddof=1, 0.0 for a single sample), and the number of samples.
    Keys missing from individual dictionaries are simply excluded from that
    metric's sample set, which keeps partial runs (a seed that failed
    mid-sweep) representable instead of silently dropped.
    """
    if not per_seed_metrics:
        return {}

    summary: Dict[str, Dict[str, float]] = {}
    keys: List[str] = []
    for metrics in per_seed_metrics:
        for key in metrics:
            if key not in keys:
                keys.append(key)

    for key in keys:
        values = [float(metrics[key]) for metrics in per_seed_metrics if key in metrics]
        arr = np.asarray(values, dtype=float)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary[key] = {
            "mean": float(arr.mean()),
            "std": std,
            "n": int(arr.size),
        }
    return summary


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def resolve_best_config(project_root: Path) -> Tuple[Path, str]:
    """Locate the configuration of the best-ranked run.

    Primary source: results/analysis_summary.json (best_run_name +
    best_category). Category "experiment" maps to experiments/<run>/config.json,
    category "baseline" maps to p1_training/config_baseline.json. Falls back to
    the exp1 configuration when the summary or the mapped file is unavailable.
    Returns (config_path, selection_note).
    """
    fallback = project_root / "experiments" / "exp1_hardware_aware_loss" / "config.json"
    summary_path = project_root / "results" / "analysis_summary.json"
    if summary_path.exists():
        try:
            analysis = _load_json(summary_path)
            best = analysis.get("best", {})
            run_name = str(best.get("best_run_name", ""))
            category = str(best.get("best_category", ""))
            if run_name:
                if category == "experiment":
                    candidate = project_root / "experiments" / run_name / "config.json"
                else:
                    candidate = project_root / "p1_training" / "config_baseline.json"
                if candidate.exists():
                    return candidate, f"analysis_summary.json best run '{run_name}'"
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            print(f"[multiseed] warning: could not use {summary_path.name}: {exc}")
    return fallback, "fallback to exp1_hardware_aware_loss config"


def _format_metric(value: float) -> str:
    if abs(value) < 1000.0:
        return f"{value:.4f}"
    return f"{value:.3e}"


def write_summary_readme(
    readme_path: Path,
    config_path: Path,
    selection_note: str,
    seeds: List[int],
    epochs: int,
    aggregate: Dict[str, Dict[str, float]],
    failed_seeds: List[int],
) -> None:
    """Write the plain-text results table next to the JSON summary."""
    lines: List[str] = []
    lines.append("# Multi-seed reproducibility summary")
    lines.append("")
    lines.append(f"- Config: `{config_path.name}` ({selection_note})")
    lines.append(f"- Seeds: {', '.join(str(s) for s in seeds)}")
    lines.append(f"- Epochs per seed: {epochs}")
    succeeded = len(seeds) - len(failed_seeds)
    lines.append(f"- Runs succeeded: {succeeded}/{len(seeds)}")
    if failed_seeds:
        lines.append(f"- Failed seeds: {', '.join(str(s) for s in failed_seeds)}")
    lines.append("")
    lines.append("| metric | mean | std (ddof=1) | n |")
    lines.append("|---|---|---|---|")
    for key in METRIC_KEYS:
        stats = aggregate.get(key)
        if stats is None:
            continue
        lines.append(
            f"| {key} | {_format_metric(stats['mean'])} | "
            f"{_format_metric(stats['std'])} | {stats['n']} |"
        )
    lines.append("")
    lines.append("Per-seed outputs live in the sibling `seed_<seed>` directories; ")
    lines.append("machine-readable results are in `multiseed_summary.json`.")
    lines.append("")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def run_multiseed(
    config_path: Path,
    output_root: Path,
    data_root: Path,
    seeds: List[int],
    epochs: int,
    selection_note: str = "explicitly provided config",
) -> Dict[str, Any]:
    """Run the config once per seed and aggregate the result metrics.

    Each seed writes its full training artifacts under
    output_root / seed_<seed> (config, checkpoints, exports), mirroring the
    layout of the single-seed experiment runners.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from p1_training.training_core import load_config, train_from_config

    output_root.mkdir(parents=True, exist_ok=True)
    base_run_name = str(_load_json(config_path).get("run_name", config_path.stem))

    per_seed: List[Dict[str, Any]] = []
    failed_seeds: List[int] = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        overrides: Dict[str, Any] = {
            "seed": seed,
            "epochs": epochs,
            "run_name": f"{base_run_name}_seed{seed}",
        }
        started = time.time()
        try:
            cfg = load_config(config_path=config_path, overrides=overrides)
            result = train_from_config(cfg=cfg, output_dir=seed_dir, data_root=data_root)
            seed_entry: Dict[str, Any] = {
                "seed": seed,
                "status": "ok",
                "output_dir": str(seed_dir.resolve()),
                "runtime_sec": round(time.time() - started, 2),
                "metrics": result["metrics"],
            }
            print(
                f"[multiseed] seed={seed} acc={result['metrics']['accuracy']*100:.2f}% "
                f"sparsity={result['metrics']['spike_sparsity']*100:.2f}% "
                f"energy={result['metrics']['energy_proxy']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            failed_seeds.append(seed)
            seed_entry = {
                "seed": seed,
                "status": "error",
                "output_dir": str(seed_dir.resolve()),
                "runtime_sec": round(time.time() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[multiseed] seed={seed} FAILED: {seed_entry['error']}")
        per_seed.append(seed_entry)

    ok_metrics = [entry["metrics"] for entry in per_seed if entry["status"] == "ok"]
    aggregate = aggregate_seed_metrics(ok_metrics)

    summary: Dict[str, Any] = {
        "config_path": str(config_path.resolve()),
        "selection_note": selection_note,
        "seeds": seeds,
        "epochs_per_seed": epochs,
        "runs_succeeded": len(ok_metrics),
        "runs_failed": len(failed_seeds),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    _dump_json(output_root / "multiseed_summary.json", summary)
    write_summary_readme(
        readme_path=output_root / "README.md",
        config_path=config_path,
        selection_note=selection_note,
        seeds=seeds,
        epochs=epochs,
        aggregate=aggregate,
        failed_seeds=failed_seeds,
    )
    return summary


def _try_import_torch() -> bool:
    """Probe for torch without side effects (unit-testable in any environment)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _report_missing_torch() -> int:
    """Report the torch-missing skip path; mirrors the CI exit-0 pattern."""
    print(
        "[multiseed] torch is not installed in this environment; "
        "skipping the multi-seed run (exit 0)."
    )
    print("[multiseed] Install torch (e.g. `pip install torch`) to run the training loop.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-seed reproducibility harness.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Seeds to run (default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Epochs per seed (default: {DEFAULT_EPOCHS}; single-seed runs used 5).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit experiment config.json; default resolves the best-ranked run.",
    )
    args = parser.parse_args(argv)

    if not _try_import_torch():
        return _report_missing_torch()

    project_root = Path(__file__).resolve().parents[1]
    if args.config is not None:
        config_path = args.config.resolve()
        if not config_path.exists():
            print(f"[multiseed] error: config not found: {config_path}")
            return 1
    else:
        config_path, selection_note = resolve_best_config(project_root)
        print(f"[multiseed] config selection: {selection_note} -> {config_path.name}")

    if not config_path.exists():
        print(f"[multiseed] error: config not found: {config_path}")
        return 1

    output_root = project_root / "results" / "multiseed"
    summary = run_multiseed(
        config_path=config_path,
        output_root=output_root,
        data_root=project_root / "data",
        seeds=args.seeds,
        epochs=args.epochs,
        selection_note=selection_note if args.config is None else "explicitly provided config",
    )
    agg = summary["aggregate"]
    if "accuracy" in agg:
        stats = agg["accuracy"]
        print(
            f"[multiseed] accuracy mean={stats['mean']*100:.2f}% "
            f"std={stats['std']*100:.2f}pp over {stats['n']} seeds"
        )
    print(f"[multiseed] summary written to {output_root / 'multiseed_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
