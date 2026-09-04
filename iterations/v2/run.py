from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from iterations.common import build_v2_config, compare_metric_sets
    from p1_training.training_core import train_from_config

    results_root = project_root / "results"
    experiments_root = project_root / "experiments"
    output_dir = results_root / "iterations" / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, sources = build_v2_config(results_root=results_root, experiments_root=experiments_root)
    with (Path(__file__).resolve().parent / "config_generated.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    result = train_from_config(cfg=cfg, output_dir=output_dir, data_root=project_root / "data")

    prev_path = results_root / "iterations" / "v1" / "metrics.json"
    if not prev_path.exists():
        raise RuntimeError("v1 result missing. Run iterations/v1/run.py first.")
    with prev_path.open("r", encoding="utf-8") as f:
        prev = json.load(f)

    comparison = compare_metric_sets(result, prev)
    comparison["reference_type"] = "iteration_v1"
    comparison["combined_sources"] = sources

    with (output_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("v2 complete.")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

