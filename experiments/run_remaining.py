"""Run remaining proposal experiments (3-5) after exp1+exp2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        project_root / "experiments" / "exp3_temporal_multiplexing_simulation" / "run.py",
        project_root / "experiments" / "exp4_quantization_effect" / "run.py",
        project_root / "experiments" / "exp5_dataflow_adaptation" / "run.py",
    ]
    for script in scripts:
        print(f"Running {script.name} ...")
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(project_root))


if __name__ == "__main__":
    main()

