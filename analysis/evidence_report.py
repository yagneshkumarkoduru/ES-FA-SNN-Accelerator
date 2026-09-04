"""Generate mandatory ES-FA evidence tables, comparisons, and plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _collect_software(results_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in results_root.glob("**/metrics.json"):
        if "hardware_validation" in str(p):
            continue
        try:
            d = _load(p)
        except Exception:
            continue
        d["_path"] = str(p)
        records.append(d)
    return records


def _collect_hardware(results_root: Path) -> List[Dict[str, Any]]:
    latest_by_key: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
    for p in results_root.glob("hardware_validation/**/hardware_metrics.json"):
        try:
            d = _load(p)
        except Exception:
            continue
        d["_path"] = str(p)
        key = (str(d.get("model_name", "unknown")), str(d.get("scheduler_mode", "unknown")))
        stamp = p.stat().st_mtime
        prev = latest_by_key.get(key)
        if prev is None or stamp >= prev[0]:
            latest_by_key[key] = (stamp, d)
    return [v[1] for v in latest_by_key.values()]


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _metric_row(
    comparison: str,
    metric: str,
    baseline_name: str,
    baseline_value: float,
    optimized_name: str,
    optimized_value: float,
    higher_is_better: bool,
) -> Dict[str, Any]:
    delta = optimized_value - baseline_value
    pct_change = (delta / baseline_value * 100.0) if abs(baseline_value) > 1e-12 else 0.0
    improved = delta > 0 if higher_is_better else delta < 0
    direction = "improvement" if improved else "degradation"
    return {
        "comparison": comparison,
        "metric": metric,
        "baseline_name": baseline_name,
        "baseline_value": baseline_value,
        "optimized_name": optimized_name,
        "optimized_value": optimized_value,
        "absolute_delta": delta,
        "percent_change": pct_change,
        "interpretation": (
            f"{direction}: {optimized_name} vs {baseline_name} "
            f"changes {metric} by {pct_change:.2f}% ({delta:.6f} absolute)."
        ),
    }


def _best_non_baseline(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    cands = [r for r in records if str(r.get("category", "")) != "baseline"]
    if not cands:
        return None
    cands.sort(
        key=lambda r: (
            _safe(r.get("metrics", {}).get("accuracy")),
            _safe(r.get("metrics", {}).get("spike_sparsity")),
            -_safe(r.get("metrics", {}).get("energy_proxy")),
        ),
        reverse=True,
    )
    return cands[0]


def _find_baseline(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in records:
        if str(r.get("category", "")) == "baseline":
            return r
    return None


def _find_static_adaptive(records: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    adaptive = None
    static_candidates = []
    for r in records:
        mode = str(r.get("config", {}).get("hardware_estimator", {}).get("dataflow_mode", ""))
        if mode == "adaptive":
            adaptive = r
        elif mode in {"dense", "event"} and str(r.get("category", "")) != "baseline":
            static_candidates.append(r)
    if static_candidates:
        static_candidates.sort(key=lambda r: _safe(r.get("metrics", {}).get("accuracy")), reverse=True)
        static_best = static_candidates[0]
    else:
        static_best = None
    return static_best, adaptive


def _group_dense_event_hw(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        grouped[str(r.get("model_name", "unknown"))][str(r.get("scheduler_mode", ""))] = r
    return grouped


def _find_hw_mode(records: List[Dict[str, Any]], model_name: str, mode: str) -> Optional[Dict[str, Any]]:
    for r in records:
        if str(r.get("model_name")) == model_name and str(r.get("scheduler_mode")) == mode:
            return r
    return None


def _pick_hw_optimized_model(records: List[Dict[str, Any]], baseline_model: str) -> Optional[str]:
    names = sorted({str(r.get("model_name")) for r in records if str(r.get("model_name")) != baseline_model})
    if not names:
        return None
    # Start simple here; can replace with richer multi-metric selection later.
    return names[0]


def _estimator_vs_measured_rows(hw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for r in hw_records:
        pred = _safe(r.get("estimator_reference", {}).get("latency_proxy"), default=0.0)
        meas = _safe(r.get("measured", {}).get("latency_ns"), default=0.0)
        if pred > 0 and meas > 0:
            pairs.append((r, pred, meas))
    if not pairs:
        return []

    scale = sum(meas for _, _, meas in pairs) / max(1e-12, sum(pred for _, pred, _ in pairs))
    rows = []
    for r, pred, meas in pairs:
        pred_scaled = pred * scale
        err = pred_scaled - meas
        pct = (err / meas) * 100.0 if abs(meas) > 1e-12 else 0.0
        rows.append(
            {
                "model_name": r.get("model_name"),
                "scheduler_mode": r.get("scheduler_mode"),
                "pred_latency_scaled_ns": pred_scaled,
                "measured_latency_ns": meas,
                "absolute_error_ns": err,
                "percent_error": pct,
            }
        )
    return rows


def _plot_estimator_vs_measured(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    x = [float(r["pred_latency_scaled_ns"]) for r in rows]
    y = [float(r["measured_latency_ns"]) for r in rows]
    labs = [f"{r['model_name']}:{r['scheduler_mode']}" for r in rows]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    for xi, yi, lab in zip(x, y, labs):
        plt.annotate(lab, (xi, yi), fontsize=8)
    lo = min(x + y)
    hi = max(x + y)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Estimator (scaled) latency [ns]")
    plt.ylabel("Measured latency [ns]")
    plt.title("Estimator vs Hardware Latency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_adaptive_static(static_r: Dict[str, Any], adaptive_r: Dict[str, Any], out_path: Path) -> None:
    metrics = ["accuracy", "energy_proxy", "latency_proxy"]
    svals = [_safe(static_r.get("metrics", {}).get(m)) for m in metrics]
    avals = [_safe(adaptive_r.get("metrics", {}).get(m)) for m in metrics]

    x = list(range(len(metrics)))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], svals, width=width, label=static_r.get("run_name"))
    plt.bar([i + width / 2 for i in x], avals, width=width, label=adaptive_r.get("run_name"))
    plt.xticks(x, metrics)
    plt.title("Static vs Adaptive Metrics")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def generate(results_root: Path) -> Dict[str, Any]:
    software = _collect_software(results_root)
    hardware = _collect_hardware(results_root)
    out_dir = results_root / "analysis" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    software_rows = []
    for r in software:
        m = r.get("metrics", {})
        software_rows.append(
            {
                "run_name": r.get("run_name"),
                "category": r.get("category"),
                "accuracy": _safe(m.get("accuracy")),
                "spike_sparsity": _safe(m.get("spike_sparsity")),
                "energy_proxy": _safe(m.get("energy_proxy")),
                "latency_proxy": _safe(m.get("latency_proxy")),
                "source_path": r.get("_path"),
            }
        )
    _write_csv(
        out_dir / "raw_software_metrics.csv",
        software_rows,
        ["run_name", "category", "accuracy", "spike_sparsity", "energy_proxy", "latency_proxy", "source_path"],
    )

    hardware_rows = []
    for r in hardware:
        hardware_rows.append(
            {
                "model_name": r.get("model_name"),
                "dataset_test_subset": r.get("dataset_test_subset"),
                "quantization_bit_width": r.get("quantization_bit_width"),
                "scheduler_mode": r.get("scheduler_mode"),
                "execution_mode": r.get("execution_mode"),
                "clock_frequency_mhz": _safe(r.get("clock_frequency_mhz")),
                "latency_ns": _safe(r.get("measured", {}).get("latency_ns")),
                "cycle_count": _safe(r.get("measured", {}).get("cycle_count")),
                "lut_usage": _safe(r.get("measured", {}).get("lut_usage")),
                "bram_usage": _safe(r.get("measured", {}).get("bram_usage")),
                "dsp_usage": _safe(r.get("measured", {}).get("dsp_usage")),
                "timing_slack_ns": _safe(r.get("measured", {}).get("timing_slack_ns")),
                "fmax_mhz_estimated": _safe(r.get("measured", {}).get("fmax_mhz_estimated")),
                "source_path": r.get("_path"),
            }
        )
    _write_csv(
        out_dir / "raw_hardware_metrics.csv",
        hardware_rows,
        [
            "model_name",
            "dataset_test_subset",
            "quantization_bit_width",
            "scheduler_mode",
            "execution_mode",
            "clock_frequency_mhz",
            "latency_ns",
            "cycle_count",
            "lut_usage",
            "bram_usage",
            "dsp_usage",
            "timing_slack_ns",
            "fmax_mhz_estimated",
            "source_path",
        ],
    )

    comp_rows: List[Dict[str, Any]] = []

    baseline = _find_baseline(software)
    optimized = _best_non_baseline(software)
    if baseline and optimized:
        comp_rows.append(
            _metric_row(
                "baseline_snn_vs_hardware_aware_snn",
                "accuracy",
                baseline["run_name"],
                _safe(baseline["metrics"]["accuracy"]),
                optimized["run_name"],
                _safe(optimized["metrics"]["accuracy"]),
                higher_is_better=True,
            )
        )
        comp_rows.append(
            _metric_row(
                "baseline_snn_vs_hardware_aware_snn",
                "energy_proxy",
                baseline["run_name"],
                _safe(baseline["metrics"]["energy_proxy"]),
                optimized["run_name"],
                _safe(optimized["metrics"]["energy_proxy"]),
                higher_is_better=False,
            )
        )

    grouped_hw = _group_dense_event_hw(hardware)
    for model_name, modes in grouped_hw.items():
        dense_hw = modes.get("dense")
        event_hw = modes.get("event")
        if not dense_hw or not event_hw:
            continue
        comp_rows.append(
            _metric_row(
                f"dense_vs_event_scheduling:{model_name}",
                "latency_ns",
                "dense",
                _safe(dense_hw["measured"]["latency_ns"]),
                "event",
                _safe(event_hw["measured"]["latency_ns"]),
                higher_is_better=False,
            )
        )
        comp_rows.append(
            _metric_row(
                f"dense_vs_event_scheduling:{model_name}",
                "cycle_count",
                "dense",
                _safe(dense_hw["measured"]["cycle_count"]),
                "event",
                _safe(event_hw["measured"]["cycle_count"]),
                higher_is_better=False,
            )
        )

    if baseline:
        base_name = str(baseline.get("run_name"))
        hw_opt_name = _pick_hw_optimized_model(hardware, base_name)
        base_dense = _find_hw_mode(hardware, base_name, "dense")
        opt_dense = _find_hw_mode(hardware, str(hw_opt_name), "dense") if hw_opt_name else None
        if base_dense and opt_dense:
            comp_rows.append(
                _metric_row(
                    "baseline_vs_optimized_hardware_dense",
                    "latency_ns",
                    base_name,
                    _safe(base_dense["measured"]["latency_ns"]),
                    str(hw_opt_name),
                    _safe(opt_dense["measured"]["latency_ns"]),
                    higher_is_better=False,
                )
            )
            comp_rows.append(
                _metric_row(
                    "baseline_vs_optimized_hardware_dense",
                    "lut_usage",
                    base_name,
                    _safe(base_dense["measured"]["lut_usage"]),
                    str(hw_opt_name),
                    _safe(opt_dense["measured"]["lut_usage"]),
                    higher_is_better=False,
                )
            )

    static_r, adaptive_r = _find_static_adaptive(software)
    if static_r and adaptive_r:
        comp_rows.append(
            _metric_row(
                "static_vs_adaptive_execution",
                "accuracy",
                static_r["run_name"],
                _safe(static_r["metrics"]["accuracy"]),
                adaptive_r["run_name"],
                _safe(adaptive_r["metrics"]["accuracy"]),
                higher_is_better=True,
            )
        )
        comp_rows.append(
            _metric_row(
                "static_vs_adaptive_execution",
                "latency_proxy",
                static_r["run_name"],
                _safe(static_r["metrics"]["latency_proxy"]),
                adaptive_r["run_name"],
                _safe(adaptive_r["metrics"]["latency_proxy"]),
                higher_is_better=False,
            )
        )
        comp_rows.append(
            _metric_row(
                "static_vs_adaptive_execution",
                "energy_proxy",
                static_r["run_name"],
                _safe(static_r["metrics"]["energy_proxy"]),
                adaptive_r["run_name"],
                _safe(adaptive_r["metrics"]["energy_proxy"]),
                higher_is_better=False,
            )
        )

    est_rows = _estimator_vs_measured_rows(hardware)
    if est_rows:
        for r in est_rows:
            comp_rows.append(
                _metric_row(
                    "estimator_vs_measured_hardware",
                    "latency_ns",
                    f"estimator_scaled:{r['model_name']}:{r['scheduler_mode']}",
                    _safe(r["pred_latency_scaled_ns"]),
                    f"measured:{r['model_name']}:{r['scheduler_mode']}",
                    _safe(r["measured_latency_ns"]),
                    higher_is_better=False,
                )
            )
        mean_abs_pct = sum(abs(_safe(r["percent_error"])) for r in est_rows) / len(est_rows)
        comp_rows.append(
            {
                "comparison": "estimator_vs_measured_hardware",
                "metric": "mean_absolute_percent_error_latency",
                "baseline_name": "estimator_scaled",
                "baseline_value": 0.0,
                "optimized_name": "measured_latency",
                "optimized_value": mean_abs_pct,
                "absolute_delta": mean_abs_pct,
                "percent_change": mean_abs_pct,
                "interpretation": (
                    f"Estimator/hardware latency agreement: MAPE={mean_abs_pct:.2f}% "
                    f"across {len(est_rows)} hardware runs."
                ),
            }
        )
        _write_csv(
            out_dir / "estimator_vs_hardware_rows.csv",
            est_rows,
            [
                "model_name",
                "scheduler_mode",
                "pred_latency_scaled_ns",
                "measured_latency_ns",
                "absolute_error_ns",
                "percent_error",
            ],
        )

    _write_csv(
        out_dir / "percentage_change_table.csv",
        comp_rows,
        [
            "comparison",
            "metric",
            "baseline_name",
            "baseline_value",
            "optimized_name",
            "optimized_value",
            "absolute_delta",
            "percent_change",
            "interpretation",
        ],
    )

    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "comparisons": comp_rows,
                "software_runs": len(software_rows),
                "hardware_runs": len(hardware_rows),
            },
            f,
            indent=2,
        )

    # Publication-ready additional plots.
    _plot_estimator_vs_measured(est_rows, results_root / "plots" / "estimator_vs_hardware.png")
    if static_r and adaptive_r:
        _plot_adaptive_static(static_r, adaptive_r, results_root / "plots" / "adaptive_vs_static.png")

    md_lines = ["# Evidence Interpretations", ""]
    for row in comp_rows:
        md_lines.append(f"- {row['comparison']} / {row['metric']}: {row['interpretation']}")
    (out_dir / "interpretations.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "software_rows": len(software_rows),
        "hardware_rows": len(hardware_rows),
        "comparison_rows": len(comp_rows),
        "output_dir": str(out_dir.resolve()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate ES-FA evidence tables and plots.")
    ap.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = ap.parse_args()
    summary = generate(args.results_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
