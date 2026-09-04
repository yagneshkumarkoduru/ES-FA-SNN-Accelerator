"""Task router for ES-FA system layer."""

from __future__ import annotations

import time
from typing import Any, Dict

from system_layer.intent_engine.engine import IntentEngine


class TaskRouter:
    def __init__(
        self,
        snn_module: Any,
        math_module: Any,
        control_module: Any,
        adaptive_policy: Any,
    ) -> None:
        self.intent_engine = IntentEngine()
        self.snn_module = snn_module
        self.math_module = math_module
        self.control_module = control_module
        self.adaptive_policy = adaptive_policy

    def route(self, query: str, static_mode: str | None = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        intent = self.intent_engine.classify(query)

        if intent.intent == "snn_execution":
            # Start simple here; can replace with learned policy later.
            snn_features = self.snn_module.estimate_workload_features(query)
            decision = self.adaptive_policy.choose_mode(
                features=snn_features,
                static_mode=static_mode,
                query=query,
            )
            payload = self.snn_module.execute(query=query, scheduler_mode=decision["scheduler_mode"])
            exec_type = payload.get("execution_type", "hybrid")
        elif intent.intent == "math_execution":
            payload = self.math_module.execute(query=query)
            decision = {"scheduler_mode": "dense", "policy_mode": "none", "reason": "math path"}
            exec_type = payload.get("execution_type", "cpu")
        else:
            payload = self.control_module.execute(query=query)
            decision = {"scheduler_mode": "dense", "policy_mode": "none", "reason": "control path"}
            exec_type = payload.get("execution_type", "cpu")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": query,
            "intent": intent.intent,
            "intent_confidence": intent.confidence,
            "execution_type": exec_type,
            "router_latency_ms": elapsed_ms,
            "policy_decision": decision,
            "result": payload,
        }

