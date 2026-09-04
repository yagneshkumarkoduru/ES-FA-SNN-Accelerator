"""SNN execution module with optional FPGA-path metadata hooks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


class SNNModule:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def _resolve_signal_path(self, query: str) -> Path | None:
        m = re.search(r"classify\s+([^\s]+)", query, flags=re.IGNORECASE)
        if not m:
            return None
        cand = Path(m.group(1))
        if not cand.is_absolute():
            cand = (self.project_root / cand).resolve()
        return cand

    def estimate_workload_features(self, query: str) -> Dict[str, float]:
        p = self._resolve_signal_path(query)
        if p is None or not p.exists() or not p.is_file():
            return {"spike_sparsity_est": 0.90, "signal_density_est": 0.10}

        raw = p.read_bytes()
        if not raw:
            return {"spike_sparsity_est": 0.99, "signal_density_est": 0.01}

        non_zero = sum(1 for b in raw if b != 0)
        density = non_zero / float(len(raw))
        sparsity = 1.0 - density
        return {"spike_sparsity_est": sparsity, "signal_density_est": density}

    def _lookup_hardware_refs(self, scheduler_mode: str) -> Dict[str, Any]:
        base = self.project_root / "results" / "hardware_validation"
        if not base.exists():
            return {}
        candidates = list(base.glob(f"**/{scheduler_mode}/*/hardware_metrics.json"))
        if not candidates:
            return {}
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with latest.open("r", encoding="utf-8") as f:
            return json.load(f)

    def execute(self, query: str, scheduler_mode: str) -> Dict[str, Any]:
        p = self._resolve_signal_path(query)
        if p is None:
            return {
                "ok": False,
                "message": "Expected format: classify <signal.bin>",
                "execution_type": "cpu",
            }
        if not p.exists():
            return {
                "ok": False,
                "message": f"Signal file not found: {p}",
                "execution_type": "cpu",
            }

        raw = p.read_bytes()
        if not raw:
            score = 0.0
        else:
            score = sum(raw) / (255.0 * len(raw))
        label = "event_class_A" if score >= 0.50 else "event_class_B"

        hw = self._lookup_hardware_refs("event" if scheduler_mode == "adaptive" else scheduler_mode)
        latency_ns = hw.get("measured", {}).get("latency_ns") if hw else None

        return {
            "ok": True,
            "label": label,
            "score": round(score, 6),
            "scheduler_mode_used": scheduler_mode,
            "execution_type": "hybrid",
            "latency_ns_reference": latency_ns,
            "diagnostics": {
                "input_path": str(p),
                "bytes": len(raw),
                "hardware_reference_found": bool(hw),
            },
        }

