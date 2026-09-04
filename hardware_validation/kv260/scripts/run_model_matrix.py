"""Run hardware validation over model matrix definitions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run hardware validation matrix")
    ap.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).resolve().parent / "model_matrix.json",
    )
    ap.add_argument("--skip-vivado", action="store_true")
    args = ap.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[3]
    runner = project_root / "hardware_validation" / "kv260" / "scripts" / "run_hw_validation.py"

    for model in matrix.get("models", []):
        model_id = model["model_id"]
        clock_mhz = float(model.get("clock_mhz", 100.0))
        cmd = [
            sys.executable,
            str(runner),
            "--model-id",
            model_id,
            "--scheduler-mode",
            "both",
            "--clock-mhz",
            str(clock_mhz),
        ]
        if args.skip_vivado:
            cmd.append("--skip-vivado")
        subprocess.run(cmd, cwd=str(project_root), check=True)

    print("Hardware matrix run complete.")


if __name__ == "__main__":
    main()

