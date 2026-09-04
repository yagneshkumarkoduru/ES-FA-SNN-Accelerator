from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from paper1_training.training_core import load_config, train_from_config

    cfg = load_config(Path(__file__).resolve().parent / "config.json")
    out_dir = project_root / "results" / "experiments" / "exp3_temporal_multiplexing_simulation"
    result = train_from_config(cfg, output_dir=out_dir, data_root=project_root / "data")
    print(f"exp3 complete: best_acc={result['best_val_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

