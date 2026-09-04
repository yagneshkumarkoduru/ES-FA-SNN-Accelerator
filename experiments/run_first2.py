"""Run proposal-required first two experiments in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        project_root / "experiments" / "exp1_hardware_aware_loss" / "run.py",
        project_root / "experiments" / "exp2_spike_sparsity_control" / "run.py",
    ]
    for script in scripts:
        print(f"Running {script.name} ...")
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(project_root))


if __name__ == "__main__":
    main()

