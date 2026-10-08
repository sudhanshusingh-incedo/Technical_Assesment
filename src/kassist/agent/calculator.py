"""Safe arithmetic evaluator for the agent's calculator tool.

Never `eval()` model-generated text. Parses to an AST and walks a whitelist of numeric nodes,
with limits that stop resource-exhaustion inputs such as 9**9**9.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 64
MAX_MAGNITUDE = 1e30

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt,
          "log2": math.log2, "log10": math.log10, "ceil": math.ceil, "floor": math.floor}


class CalculatorError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("exponent too large")
        result = _BIN_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        result = _UNARY_OPS[type(node.op)](_eval(node.operand))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        result = _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    else:
        raise CalculatorError(f"unsupported syntax: {type(node).__name__}")
    if isinstance(result, complex) or abs(result) > MAX_MAGNITUDE:
        raise CalculatorError("result out of range")
    return result


def evaluate(expression: str) -> float:
    # "^" as power; drop thousands separators ("1,536") but keep argument commas ("round(x, 2)")
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression.strip().replace("^", "**"))
    if not expr or len(expr) > MAX_EXPRESSION_CHARS:
        raise CalculatorError("expression empty or too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("invalid expression") from exc
    try:
        return _eval(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc


def format_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
