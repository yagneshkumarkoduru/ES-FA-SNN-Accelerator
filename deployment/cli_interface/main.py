"""ES-FA runtime CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_router(project_root: Path):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from deployment.runtime.adaptive_policy import AdaptiveExecutionPolicy
    from system_layer.modules.control_module.module import ControlModule
    from system_layer.modules.math_module.module import MathModule
    from system_layer.modules.snn_module.module import SNNModule
    from system_layer.task_router.router import TaskRouter

    policy = AdaptiveExecutionPolicy(
        log_path=project_root / "results" / "runtime" / "adaptive_decisions.jsonl",
        sparsity_threshold=0.90,
    )
    router = TaskRouter(
        snn_module=SNNModule(project_root=project_root),
        math_module=MathModule(),
        control_module=ControlModule(),
        adaptive_policy=policy,
    )
    return router


def _format_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = payload.get("result", {})
    return {
        "result": result,
        "execution_type": payload.get("execution_type"),
        "latency_ms": round(float(payload.get("router_latency_ms", 0.0)), 4),
        "diagnostics": {
            "intent": payload.get("intent"),
            "intent_confidence": payload.get("intent_confidence"),
            "policy_decision": payload.get("policy_decision"),
        },
    }


def _run_query(router: Any, query: str, static_mode: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    routed = router.route(query=query, static_mode=static_mode)
    out = _format_output(routed)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ES-FA CLI runtime")
    ap.add_argument("--query", type=str, default=None, help="Single query to execute.")
    ap.add_argument(
        "--static-mode",
        type=str,
        default=None,
        choices=["dense", "event"],
        help="Force static scheduler mode for comparison.",
    )
    ap.add_argument("--json", action="store_true", help="Print raw JSON only.")
    args = ap.parse_args()

    project_root = _project_root()
    router = _build_router(project_root)

    if args.query is not None:
        out = _run_query(router, args.query, static_mode=args.static_mode)
        print(json.dumps(out, indent=2))
        return

    print("ES-FA CLI ready. Type `exit` to quit.")
    while True:
        try:
            query = input("esfa> ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        out = _run_query(router, query, static_mode=args.static_mode)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
