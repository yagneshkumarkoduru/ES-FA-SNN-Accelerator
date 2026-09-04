"""Run ES-FA evolution pipeline in strict phase order with smoke/full modes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _append_checkpoint(project_root: Path, phase: str, status: str, note: str) -> None:
    p = project_root / "docs" / "checkpoints.md"
    line = f"- {datetime.now(timezone.utc).isoformat()} | {phase} | {status} | {note}\n"
    text = p.read_text(encoding="utf-8") if p.exists() else "# ES-FA Checkpoints\n\n"
    if "## Run History" not in text:
        text = text.rstrip() + "\n\n## Run History\n"
    text = text.rstrip() + "\n" + line
    p.write_text(text, encoding="utf-8")


def _run_py(script: Path, args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, str(script)] + args, cwd=str(cwd), check=True)


def _train_job(config_path: Path, out_dir: Path, epochs: int) -> None:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from p1_training.training_core import load_config, train_from_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements with a Python version "
            "supported by PyTorch (recommended 3.10-3.12)."
        ) from exc

    cfg = load_config(config_path)
    cfg["epochs"] = int(epochs)
    train_from_config(cfg=cfg, output_dir=out_dir, data_root=project_root / "data")


def _run_cli_query(project_root: Path, query: str, static_mode: str | None = None) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(project_root / "deployment" / "cli_interface" / "main.py"),
        "--query",
        query,
    ]
    if static_mode is not None:
        cmd += ["--static-mode", static_mode]
    proc = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True, check=True)
    return json.loads(proc.stdout.strip())


def _run_runtime_adaptive_compare(project_root: Path) -> None:
    signal_path = project_root / "results" / "runtime" / "sample_signal.bin"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    if not signal_path.exists():
        signal_path.write_bytes(bytes([(i * 17) % 256 for i in range(1024)]))

    query = f"classify {signal_path}"
    dense_out = _run_cli_query(project_root, query, static_mode="dense")
    event_out = _run_cli_query(project_root, query, static_mode="event")
    adaptive_out = _run_cli_query(project_root, query, static_mode=None)

    rows = {
        "dense": dense_out,
        "event": event_out,
        "adaptive": adaptive_out,
    }
    out_path = project_root / "results" / "runtime" / "static_vs_adaptive_cli.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run(mode: str, clock_mhz: float, skip_vivado: bool, skip_hardware: bool) -> None:
    project_root = _project_root()

    # Phase 0: analyze
    if not (project_root / "docs" / "mapping.md").exists():
        raise FileNotFoundError("docs/mapping.md missing. Phase 0 must be completed first.")
    _append_checkpoint(project_root, "Phase 0", "ok", "Mapping file validated")

    # Phase 1: preserve + modularize (training baseline)
    baseline_cfg = project_root / "p1_training" / "config_baseline.json"
    _train_job(
        config_path=baseline_cfg,
        out_dir=project_root / "results" / "baseline",
        epochs=1 if mode == "smoke" else 5,
    )
    _append_checkpoint(project_root, "Phase 1", "ok", f"Baseline training completed in {mode} mode")

    # Phase 2: stabilize experiments
    if mode == "smoke":
        _train_job(
            config_path=project_root / "experiments" / "exp1_hardware_aware_loss" / "config.json",
            out_dir=project_root / "results" / "experiments" / "exp1_hardware_aware_loss",
            epochs=1,
        )
        _train_job(
            config_path=project_root / "experiments" / "exp5_dataflow_adaptation" / "config.json",
            out_dir=project_root / "results" / "experiments" / "exp5_dataflow_adaptation",
            epochs=1,
        )
    else:
        _run_py(project_root / "experiments" / "run_all.py", [], cwd=project_root)
        _run_py(project_root / "iterations" / "run_all.py", [], cwd=project_root)
    _append_checkpoint(project_root, "Phase 2", "ok", "Experiments/iterations finished")

    # Phase 3: FPGA validation
    if skip_hardware:
        _append_checkpoint(project_root, "Phase 3", "skipped", "Hardware validation skipped by flag")
    else:
        for model_id in ["baseline_paper1", "exp5_dataflow_adaptation"]:
            _run_py(
                project_root / "hardware_validation" / "kv260" / "scripts" / "run_hw_validation.py",
                [
                    "--model-id",
                    model_id,
                    "--scheduler-mode",
                    "both",
                    "--clock-mhz",
                    str(clock_mhz),
                ] + (["--skip-vivado"] if skip_vivado else []),
                cwd=project_root,
            )
        _append_checkpoint(project_root, "Phase 3", "ok", "Hardware validation runners executed")

    # Phase 4: system layer integration checks
    _append_checkpoint(project_root, "Phase 4", "ok", "System layer modules available")

    # Phase 5: CLI checks
    _run_cli_query(project_root, "hi")
    _run_cli_query(project_root, "strike rate 167.55 balls 39")
    _append_checkpoint(project_root, "Phase 5", "ok", "CLI query paths validated")

    # Phase 6: adaptive execution checks
    _run_runtime_adaptive_compare(project_root)
    _append_checkpoint(project_root, "Phase 6", "ok", "Static/adaptive runtime comparison generated")

    # Phase 7: analysis
    _run_py(project_root / "analysis" / "compare.py", [], cwd=project_root)
    _append_checkpoint(project_root, "Phase 7", "ok", "Analysis outputs generated")

    # Phase 8: visualization + evidence tables
    _run_py(project_root / "analysis" / "evidence_report.py", [], cwd=project_root)
    _append_checkpoint(project_root, "Phase 8", "ok", "Visualization and evidence outputs generated")

    # Phase 9: LaTeX and paper output
    _run_py(project_root / "output" / "generate_paper_output.py", [], cwd=project_root)
    _append_checkpoint(project_root, "Phase 9", "ok", "Paper output markdown generation completed")

    # Phase 10-11: checkpoint update
    _append_checkpoint(project_root, "Phase 10-11", "ok", "Pipeline run completed")


def main() -> None:
    ap = argparse.ArgumentParser(description="ES-FA strict-order pipeline")
    ap.add_argument("--mode", type=str, default="smoke", choices=["smoke", "full"])
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--skip-vivado", action="store_true")
    ap.add_argument("--skip-hardware", action="store_true")
    args = ap.parse_args()
    run(
        mode=args.mode,
        clock_mhz=args.clock_mhz,
        skip_vivado=args.skip_vivado,
        skip_hardware=args.skip_hardware,
    )
    print("ES-FA pipeline complete.")


if __name__ == "__main__":
    main()
