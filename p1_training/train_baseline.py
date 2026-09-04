"""Run proposal-aligned baseline training and save results/baseline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Paper-1 baseline SNN.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_baseline.json",
        help="Baseline config JSON path.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "results" / "baseline",
        help="Result directory for baseline run.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=_project_root() / "data",
        help="Dataset cache directory.",
    )
    return p.parse_args()


def main() -> None:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from p1_training.training_core import load_config, train_from_config

    args = parse_args()
    cfg = load_config(args.config)
    result = train_from_config(cfg=cfg, output_dir=args.output_dir, data_root=args.data_root)
    print("\nBaseline complete.")
    print(f"Best accuracy: {result['best_val_accuracy']*100:.2f}%")
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

