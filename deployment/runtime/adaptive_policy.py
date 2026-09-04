"""Explicit adaptive execution policy with decision logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class AdaptiveExecutionPolicy:
    def __init__(self, log_path: Path, sparsity_threshold: float = 0.90) -> None:
        self.log_path = log_path
        self.sparsity_threshold = sparsity_threshold
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def choose_mode(self, features: Dict[str, float], query: str, static_mode: Optional[str] = None) -> Dict[str, str]:
        # Keeping this explicit for estimator + FPGA path consistency.
        sparsity = float(features.get("spike_sparsity_est", 0.0))
        if static_mode in {"dense", "event"}:
            chosen = static_mode
            policy_mode = "static_override"
            reason = f"forced_static_mode={static_mode}"
        else:
            chosen = "event" if sparsity >= self.sparsity_threshold else "dense"
            policy_mode = "adaptive"
            reason = (
                f"sparsity={sparsity:.4f} "
                f"{'>=' if sparsity >= self.sparsity_threshold else '<'} "
                f"threshold={self.sparsity_threshold:.4f}"
            )

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "policy_mode": policy_mode,
            "scheduler_mode": chosen,
            "reason": reason,
            "features": features,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return {
            "policy_mode": policy_mode,
            "scheduler_mode": chosen,
            "reason": reason,
        }

