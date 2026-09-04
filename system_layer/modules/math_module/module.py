"""Math execution module."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("Unsupported expression")

    return _eval(node)


class MathModule:
    def execute(self, query: str) -> Dict[str, Any]:
        q = query.strip().lower()
        m = re.search(r"strike\s+rate\s+([0-9]*\.?[0-9]+)\s+balls\s+([0-9]*\.?[0-9]+)", q)
        if m:
            strike_rate = float(m.group(1))
            balls = float(m.group(2))
            runs = (strike_rate * balls) / 100.0
            return {
                "ok": True,
                "type": "strike_rate_projection",
                "runs": round(runs, 4),
                "inputs": {"strike_rate": strike_rate, "balls": balls},
                "execution_type": "cpu",
            }

        # Fallback: evaluate simple arithmetic expression.
        expr = q.replace("calculate", "").strip()
        try:
            value = _safe_eval(expr)
            return {
                "ok": True,
                "type": "arithmetic",
                "value": round(value, 6),
                "expression": expr,
                "execution_type": "cpu",
            }
        except Exception:
            return {
                "ok": False,
                "message": "Math query not understood. Example: strike rate 167.55 balls 39",
                "execution_type": "cpu",
            }

