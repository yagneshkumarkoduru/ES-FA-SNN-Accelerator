"""Run Vivado batch build and parse resulting reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Vivado batch flow for ES-FA RTL.")
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--scheduler-mode", type=str, choices=["dense", "event", "adaptive"], required=True)
    ap.add_argument("--clock-mhz", type=float, default=200.0)
    ap.add_argument("--part", type=str, default="xczu3eg-sbva484-1-e")
    ap.add_argument("--top", type=str, default="snn_top")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.toolchain.xilinx_toolchain import run_tool
    from hardware_validation.kv260.scripts.parse_vivado_reports import parse_reports

    tcl = project_root / "hardware_validation" / "kv260" / "vivado" / "build.tcl"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    vivado_log = args.run_dir / "vivado.log"

    run_tool(
        "vivado",
        [
            "-mode",
            "batch",
            "-source",
            str(tcl),
            "-tclargs",
            args.model_id,
            args.scheduler_mode,
            str(args.run_dir),
            args.part,
            args.top,
            str(args.clock_mhz),
        ],
        cwd=project_root,
        log_path=vivado_log,
        check=True,
    )

    parsed = parse_reports(args.run_dir / "reports", args.clock_mhz)
    parsed_out = args.run_dir / "vivado_metrics.json"
    with parsed_out.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"Vivado build complete for model={args.model_id}, mode={args.scheduler_mode}")
    print(f"Metrics: {parsed_out}")


if __name__ == "__main__":
    main()

