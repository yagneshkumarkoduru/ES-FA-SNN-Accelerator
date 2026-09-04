"""Rule-based intent engine (extensible to learned classifier later)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class IntentDecision:
    intent: str
    confidence: float
    features: Dict[str, float]


class IntentEngine:
    def classify(self, query: str) -> IntentDecision:
        q = query.strip().lower()
        features = {
            "has_digit": 1.0 if any(ch.isdigit() for ch in q) else 0.0,
            "has_signal_keyword": 1.0 if ("classify" in q or "signal" in q or ".bin" in q) else 0.0,
            "has_math_keyword": 1.0 if ("strike rate" in q or "balls" in q or "calculate" in q) else 0.0,
            "has_control_keyword": 1.0 if ("hi" in q or "hello" in q or "help" in q or "status" in q) else 0.0,
        }

        if features["has_signal_keyword"] > 0:
            return IntentDecision("snn_execution", 0.95, features)
        if features["has_math_keyword"] > 0:
            return IntentDecision("math_execution", 0.90, features)
        if features["has_control_keyword"] > 0:
            return IntentDecision("control_execution", 0.85, features)
        return IntentDecision("control_execution", 0.60, features)

