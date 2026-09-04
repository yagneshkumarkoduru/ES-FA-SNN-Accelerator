"""Run all proposal-aligned experiments in the requested order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        project_root / "experiments" / "run_first2.py",
        project_root / "experiments" / "run_remaining.py",
    ]
    for script in scripts:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(project_root))


if __name__ == "__main__":
    main()

