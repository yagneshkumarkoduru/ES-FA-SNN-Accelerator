from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from iterations.common import build_v1_config, compare_metric_sets
    from p1_training.training_core import train_from_config

    results_root = project_root / "results"
    experiments_root = project_root / "experiments"
    output_dir = results_root / "iterations" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, source_exp = build_v1_config(results_root=results_root, experiments_root=experiments_root)
    with (Path(__file__).resolve().parent / "config_generated.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    result = train_from_config(cfg=cfg, output_dir=output_dir, data_root=project_root / "data")

    baseline_path = results_root / "baseline" / "metrics.json"
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        comparison = compare_metric_sets(result, baseline)
        comparison["reference_type"] = "baseline"
    else:
        comparison = compare_metric_sets(result, source_exp)
        comparison["reference_type"] = "best_experiment_fallback"

    with (output_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("v1 complete.")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

