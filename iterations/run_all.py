"""Run v1 -> v2 -> v3 iteration sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        project_root / "iterations" / "v1" / "run.py",
        project_root / "iterations" / "v2" / "run.py",
        project_root / "iterations" / "v3" / "run.py",
    ]
    for script in scripts:
        print(f"Running {script} ...")
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(project_root))


if __name__ == "__main__":
    main()

