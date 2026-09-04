"""Execute the full proposal-aligned research pipeline in required order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, script: Path, project_root: Path) -> None:
    print(f"\n=== {step_name} ===")
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(project_root))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    run_step(
        "1) Baseline model",
        project_root / "p1_training" / "train_baseline.py",
        project_root,
    )
    run_step(
        "2-4) First two experiments + logging system",
        project_root / "experiments" / "run_first2.py",
        project_root,
    )
    run_step(
        "5) Remaining experiments",
        project_root / "experiments" / "run_remaining.py",
        project_root,
    )
    run_step(
        "6) Iterations v1/v2/v3",
        project_root / "iterations" / "run_all.py",
        project_root,
    )
    run_step(
        "7) Analysis + plots",
        project_root / "analysis" / "compare.py",
        project_root,
    )
    run_step(
        "8) Paper output generation",
        project_root / "output" / "generate_paper_output.py",
        project_root,
    )

    print("\nResearch pipeline complete.")
    print(f"Results root: {(project_root / 'results').resolve()}")


if __name__ == "__main__":
    main()
