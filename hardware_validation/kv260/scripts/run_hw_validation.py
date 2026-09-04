"""End-to-end local hardware validation: xsim + Vivado + normalized metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_run_manifest(project_root: Path, model_id: str) -> Optional[Dict[str, Any]]:
    for p in (project_root / "results").glob("**/run_manifest.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("run_name") == model_id:
            return data
    return None


def _run_py(script: Path, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, str(script)] + args
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_one_mode(
    project_root: Path,
    model_id: str,
    mode: str,
    run_id: str,
    clock_mhz: float,
    skip_vivado: bool,
) -> Path:
    run_dir = project_root / "results" / "hardware_validation" / model_id / mode / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = project_root / "hardware_validation" / "kv260" / "scripts"
    _run_py(
        scripts_dir / "run_xsim_regression.py",
        [
            "--model-id",
            model_id,
            "--clock-mhz",
            str(clock_mhz),
            "--run-dir",
            str(run_dir),
            "--modes",
            mode,
        ],
        cwd=project_root,
    )

    if not skip_vivado:
        _run_py(
            scripts_dir / "run_vivado_build.py",
            [
                "--model-id",
                model_id,
                "--scheduler-mode",
                mode,
                "--clock-mhz",
                str(clock_mhz),
                "--run-dir",
                str(run_dir),
            ],
            cwd=project_root,
        )

    manifest = _find_run_manifest(project_root, model_id) or {}
    _run_py(
        scripts_dir / "collect_hardware_metrics.py",
        [
            "--project-root",
            str(project_root),
            "--run-dir",
            str(run_dir),
            "--model-id",
            model_id,
            "--scheduler-mode",
            mode,
            "--dataset-split",
            str(manifest.get("dataset_split", "mnist_test")),
            "--quantization-mode",
            str(manifest.get("quantization", {}).get("mode", "int8")),
            "--execution-mode",
            "fpga_emulation",
            "--clock-mhz",
            str(clock_mhz),
        ],
        cwd=project_root,
    )
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local ES-FA hardware validation stack.")
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--scheduler-mode", type=str, choices=["dense", "event", "both"], default="both")
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--skip-vivado", action="store_true")
    args = ap.parse_args()

    project_root = _project_root()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes = ["dense", "event"] if args.scheduler_mode == "both" else [args.scheduler_mode]

    run_dirs = []
    for mode in modes:
        print(f"[hardware_validation] model={args.model_id} mode={mode}")
        run_dirs.append(
            _run_one_mode(
                project_root=project_root,
                model_id=args.model_id,
                mode=mode,
                run_id=run_id,
                clock_mhz=args.clock_mhz,
                skip_vivado=args.skip_vivado,
            )
        )

    summary = {
        "model_id": args.model_id,
        "modes": modes,
        "clock_mhz": args.clock_mhz,
        "skip_vivado": args.skip_vivado,
        "run_id": run_id,
        "run_dirs": [str(p.resolve()) for p in run_dirs],
    }
    summary_path = project_root / "results" / "hardware_validation" / args.model_id / f"validation_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Hardware validation summary: {summary_path}")


if __name__ == "__main__":
    main()

