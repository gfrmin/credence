# Role: body
"""Calculator tool: evaluates mathematical expressions via safe AST parsing.

Free, instant, only handles numerical questions. Returns None for
non-numerical questions (coverage=0 for non-numerical categories).

Security: uses ast.parse with a whitelist of operators — no raw eval().
"""

from __future__ import annotations

import ast
import math
import operator
import re

import numpy as np
from numpy.typing import NDArray

# Safe operators for AST evaluation
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe named constants and functions
_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_SAFE_FUNCS: dict[str, object] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
}


def _safe_ast_eval(expr: str) -> float | None:
    """Evaluate a mathematical expression safely using AST parsing.

    Only allows arithmetic operators, numeric constants, and whitelisted
    functions. No arbitrary code execution.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return _walk_node(tree.body)
    except Exception:
        return None


def _walk_node(node: ast.expr) -> float:
    """Recursively evaluate an AST node with whitelisted operations only."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_fn(_walk_node(node.left), _walk_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        return op_fn(_walk_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            fn = _SAFE_FUNCS[node.func.id]
            args = [_walk_node(a) for a in node.args]
            return float(fn(*args))  # type: ignore[operator]
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    raise ValueError(f"Unsupported node: {type(node)}")


def _extract_expression(question: str) -> str | None:
    """Try to extract a mathematical expression from a question."""
    # "X% of Y" pattern
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d[\d,]*(?:\.\d+)?)", question)
    if pct_match:
        pct = pct_match.group(1)
        val = pct_match.group(2).replace(",", "")
        return f"{pct} / 100 * {val}"

    # "square root of X"
    sqrt_match = re.search(r"square root of\s*(\d[\d,]*(?:\.\d+)?)", question)
    if sqrt_match:
        val = sqrt_match.group(1).replace(",", "")
        return f"sqrt({val})"

    # "X^Y" or "X to the power of Y" or "2^15"
    pow_match = re.search(r"(\d+)\s*(?:\^|\*\*|to the power of)\s*(\d+)", question)
    if pow_match:
        return f"{pow_match.group(1)} ** {pow_match.group(2)}"

    # Generic expression: digits and operators
    expr_match = re.search(r"(\d[\d\s,]*[\+\-\*\/\%\^][\d\s,\+\-\*\/\%\^\.]+\d)", question)
    if expr_match:
        expr = expr_match.group(1).replace(",", "").replace("^", "**")
        return expr

    return None


def _parse_numeric(s: str) -> float | None:
    """Parse a numeric value from a candidate string, stripping units."""
    cleaned = re.sub(r"[,$%]", "", s)
    cleaned = re.sub(r"\s*(miles|cups|mph|hours|litres?).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


class CalculatorTool:
    """Safe AST-based calculator. Free, instant, only for numerical questions."""

    name: str = "calculator"
    cost: float = 0.0
    latency: float = 0.001

    def query(self, question: str, candidates: tuple[str, ...]) -> int | None:
        """Evaluate the expression and match to closest candidate."""
        expr = _extract_expression(question)
        if expr is None:
            return None

        result = _safe_ast_eval(expr)
        if result is None:
            return None

        # Compare result to candidates numerically
        best_idx = None
        best_diff = float("inf")
        for i, c in enumerate(candidates):
            num = _parse_numeric(c)
            if num is not None:
                diff = abs(num - result)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

        # Only return if reasonably close (within 5% or 1.0 absolute)
        if best_idx is not None and best_diff <= max(abs(result) * 0.05, 1.0):
            return best_idx
        return None

    def coverage(self, categories: tuple[str, ...]) -> NDArray[np.float64]:
        """Only covers numerical questions."""
        return np.array(
            [1.0 if c == "numerical" else 0.0 for c in categories],
            dtype=np.float64,
        )
