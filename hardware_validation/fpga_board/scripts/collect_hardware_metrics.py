"""Normalize local xsim + Vivado outputs into ES-FA hardware metric schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_software_metrics(project_root: Path, model_id: str) -> Optional[Dict[str, Any]]:
    candidates = list((project_root / "results").glob("**/metrics.json"))

    for p in candidates:
        try:
            data = _load(p)
        except Exception:
            continue
        if data.get("run_name") == model_id:
            return data
    return None


def _scheduler_bucket(mode: str) -> str:
    if mode == "event":
        return "event"
    if mode == "adaptive":
        return "event"
    return "dense"


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect normalized hardware metrics.")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--scheduler-mode", type=str, required=True)
    ap.add_argument("--dataset-split", type=str, default="mnist_test")
    ap.add_argument("--quantization-mode", type=str, default="int8")
    ap.add_argument("--execution-mode", type=str, default="fpga_emulation")
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    args = ap.parse_args()

    xsim = _load(args.run_dir / "xsim_metrics.json")
    vivado = _load(args.run_dir / "vivado_metrics.json")
    software = _find_software_metrics(args.project_root, args.model_id)

    xsim_bucket = _scheduler_bucket(args.scheduler_mode)
    xsim_mode = xsim.get("xsim", {}).get(xsim_bucket, {})
    impl = vivado.get("implementation", {})
    impl_util = impl.get("utilization", {})
    impl_timing = impl.get("timing", {})

    metrics = {
        "model_name": args.model_id,
        "dataset_test_subset": args.dataset_split,
        "quantization_bit_width": args.quantization_mode,
        "scheduler_mode": args.scheduler_mode,
        "execution_mode": args.execution_mode,
        "clock_frequency_mhz": args.clock_mhz,
        "measured": {
            "latency_ns": float(xsim_mode.get("latency_ns_active_window", 0.0)),
            "cycle_count": float(xsim_mode.get("active_window_cycles", 0.0)),
            "lut_usage": impl_util.get("lut"),
            "bram_usage": impl_util.get("bram"),
            "dsp_usage": impl_util.get("dsp"),
            "timing_slack_ns": impl_timing.get("wns_ns"),
            "fmax_mhz_estimated": impl_timing.get("fmax_mhz_estimated"),
        },
        "estimator_reference": {
            "latency_proxy": software.get("metrics", {}).get("latency_proxy") if software else None,
            "energy_proxy": software.get("metrics", {}).get("energy_proxy") if software else None,
            "spike_sparsity": software.get("metrics", {}).get("spike_sparsity") if software else None,
        },
        "artifacts": {
            "xsim_metrics": str((args.run_dir / "xsim_metrics.json").resolve()),
            "vivado_metrics": str((args.run_dir / "vivado_metrics.json").resolve()),
            "vivado_log": str((args.run_dir / "vivado.log").resolve()),
        },
    }

    out_path = args.run_dir / "hardware_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Hardware metrics written: {out_path}")


if __name__ == "__main__":
    main()
