"""Shared helpers for proposal-aligned experiments."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Any


def project_root_from(path_in_experiment: Path) -> Path:
    return path_in_experiment.resolve().parents[2]


def run_experiment(config_path: Path, output_dir: Path, data_root: Path) -> Dict[str, Any]:
    project_root = output_dir.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from p1_training.training_core import load_config, train_from_config

    cfg = load_config(config_path=config_path)
    return train_from_config(cfg=cfg, output_dir=output_dir, data_root=data_root)

