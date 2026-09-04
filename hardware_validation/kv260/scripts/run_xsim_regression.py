"""Run xsim regressions for dense/event scheduler modes and capture metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_result(log_text: str, clock_mhz: float) -> Dict[str, float]:
    def grab(key: str) -> float:
        m = re.search(rf"{re.escape(key)}=([-+]?\d+(?:\.\d+)?)", log_text)
        return float(m.group(1)) if m else 0.0

    cycles_total = grab("RESULT_CYCLE_COUNT")
    cycles_active = grab("RESULT_ACTIVE_WINDOW_CYCLES")
    basic_ops = grab("RESULT_BASIC_OPS")
    adv_ops = grab("RESULT_ADV_OPS")
    pe_ops = grab("RESULT_PE_OPS")
    out_spikes = grab("RESULT_OUT_SPIKES")
    mode = int(grab("RESULT_MODE"))
    period_ns = 1000.0 / max(clock_mhz, 1e-9)

    return {
        "mode_advanced": mode,
        "cycle_count": cycles_total,
        "active_window_cycles": cycles_active,
        "latency_ns_active_window": cycles_active * period_ns,
        "basic_ops": basic_ops,
        "adv_ops": adv_ops,
        "pe_ops": pe_ops,
        "out_spikes": out_spikes,
    }


def _rtl_files(project_root: Path) -> Iterable[str]:
    return [
        str(project_root / "hardware" / "memory" / "neuron_bram.v"),
        str(project_root / "hardware" / "memory" / "weight_bram_bank.v"),
        str(project_root / "hardware" / "compute" / "lif_neuron_pe.v"),
        str(project_root / "hardware" / "routing" / "spike_router.v"),
        str(project_root / "hardware" / "scheduler" / "basic_scheduler.v"),
        str(project_root / "hardware" / "scheduler" / "event_queue.v"),
        str(project_root / "hardware" / "scheduler" / "advanced_scheduler.v"),
        str(project_root / "hardware" / "top" / "snn_top.v"),
        str(project_root / "hardware_validation" / "kv260" / "tb" / "tb_kv260_modes.v"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run xsim regression for ES-FA scheduler modes.")
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--modes", type=str, default="dense,event")
    args = ap.parse_args()

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.toolchain.xilinx_toolchain import run_tool

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Dict[str, float]] = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        sim_name = f"tb_kv260_modes_sim_{mode}"
        compile_log = logs_dir / f"xvlog_{mode}.log"
        elab_log = logs_dir / f"xelab_{mode}.log"
        sim_log = logs_dir / f"xsim_{mode}.log"

        xvlog_args = []
        if mode == "event":
            xvlog_args += ["-d", "MODE_ADV_EVENT"]
        xvlog_args += list(_rtl_files(project_root))

        run_tool(
            "xvlog",
            xvlog_args,
            cwd=project_root,
            log_path=compile_log,
            check=True,
        )
        run_tool(
            "xelab",
            ["tb_kv260_modes", "-s", sim_name],
            cwd=project_root,
            log_path=elab_log,
            check=True,
        )

        sim_log = logs_dir / f"xsim_{mode}.log"
        run_tool(
            "xsim",
            [sim_name, "--runall"],
            cwd=project_root,
            log_path=sim_log,
            check=True,
        )
        parsed = _parse_result(sim_log.read_text(encoding="utf-8", errors="ignore"), args.clock_mhz)
        out[mode] = parsed

    metrics = {
        "model_id": args.model_id,
        "clock_mhz": args.clock_mhz,
        "xsim": out,
    }
    metrics_path = args.run_dir / "xsim_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"xsim regression complete for {args.model_id}: {metrics_path}")


if __name__ == "__main__":
    main()
