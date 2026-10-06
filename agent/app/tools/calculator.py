"""计算器工具：用 AST 白名单求值，而不是 ``eval``。

**为什么不能用 eval**：模型生成的表达式来自不受信任的输入，
``eval`` 会直接执行任意代码（``__import__('os').system(...)``）。
这里改用 ``ast`` 解析后只放行算术运算符与白名单函数，
既满足「算数」这个真实需求，又不引入代码执行面。

精度方面用 :mod:`decimal` 处理十进制字面量，避免
``0.1 + 0.2 = 0.30000000000000004`` 这类结果误导模型。
"""

from __future__ import annotations

import ast
import math
import operator
from decimal import Decimal, getcontext
from typing import Any, Callable

from ..schema import ToolResult
from .base import Tool, ToolContext, ToolError

getcontext().prec = 28

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "pow": pow,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "gcd": math.gcd, "hypot": math.hypot, "degrees": math.degrees, "radians": math.radians,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"只支持数字，收到：{node.value!r}")
        return Decimal(str(node.value)) if isinstance(node.value, float) else node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 1000:
            raise ToolError("指数过大，拒绝计算（防止卡死）")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"未知标识符：{node.id}（可用常量：{', '.join(_CONSTANTS)}）")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            allowed = ", ".join(sorted(_FUNCTIONS))
            raise ToolError(f"不支持的函数调用；可用函数：{allowed}")
        if node.keywords:
            raise ToolError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        if len(args) > 100:
            raise ToolError("参数过多")
        return _FUNCTIONS[node.func.id](*args)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]
    raise ToolError(f"不支持的表达式类型：{type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """对表达式求值；非法输入抛 ``ToolError``。"""
    expression = (expression or "").strip()
    if not expression:
        raise ToolError("表达式为空")
    if len(expression) > 500:
        raise ToolError("表达式过长（上限 500 字符）")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}") from exc
    value = _eval_node(tree)
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolError("计算结果不是有限数")
        # 抹掉浮点尾差，避免把 2.9999999999999996 之类的噪音丢给模型
        rounded = round(value, 10)
        return int(rounded) if rounded == int(rounded) and abs(rounded) < 1e15 else rounded
    return value


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "计算数学表达式。支持 + - * / // % **、括号、常用函数"
        "（sqrt/log/exp/sin/cos/factorial 等）与常量 pi、e。"
        "需要精确算数时优先用它，不要自己心算。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的表达式，例如 (1-0.5)*math 写成 (1-0.5)*2；函数如 sqrt(2)、log(8, 2)",
            }
        },
        "required": ["expression"],
    }
    returns = "计算结果的数值，或语法错误说明。"

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        expression = str(arguments.get("expression", ""))
        value = safe_eval(expression)
        return ToolResult(ok=True, content=f"{expression} = {value}", raw={"value": value})
