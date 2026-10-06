"""工具基类与注册表。

三个设计要点：

1. **工具失败不抛异常**：任何工具错误都被包成 ``ToolResult(ok=False, error=...)``
   交回给模型。Agent 的价值恰恰体现在「看到失败后改变策略」，
   所以失败必须是一种可观测、可恢复的正常结果，而不是让整个流程崩掉。
2. **统一裁剪**：工具输出统一按 ``max_observation_chars`` 截断，
   防止一次网页抓取就把上下文窗口撑爆——这是 Agent 最常见的失控原因之一。
3. **参数校验前置**：调用前按 JSON Schema 检查必填项，
   避免把「模型传错参数」和「工具内部出错」混为一谈。
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..schema import ToolCall, ToolResult, ToolSpec


class ToolError(Exception):
    """工具内部错误。会被注册表捕获并转成失败结果。"""


@dataclass
class ToolContext:
    """工具执行时可用的运行时上下文（例如共享的检索器、工作目录）。"""

    workspace: str = ""
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class Tool(ABC):
    """所有工具的基类。"""

    name: str = ""
    description: str = ""
    # JSON Schema：properties + required
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    returns: str = ""
    dangerous: bool = False

    @abstractmethod
    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具。实现里可以直接抛 ``ToolError``，注册表会兜住。"""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            dangerous=self.dangerous,
            returns=self.returns,
        )

    # ---- 参数校验 ----
    def validate(self, arguments: dict[str, Any]) -> str:
        """返回错误说明；通过校验则返回空串。"""
        schema_required = self.parameters.get("required", []) or []
        missing = [k for k in schema_required if k not in arguments or arguments[k] in (None, "")]
        if missing:
            return f"缺少必填参数：{', '.join(missing)}"
        props = self.parameters.get("properties", {}) or {}
        unknown = [k for k in arguments if k not in props]
        if unknown:
            return (
                f"出现未定义参数：{', '.join(unknown)}；"
                f"可用参数：{', '.join(props.keys()) or '（无）'}"
            )
        return ""


class ToolRegistry:
    """工具注册表：负责调用、计时、裁剪与错误兜底。"""

    def __init__(self, max_observation_chars: int = 4000) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_observation_chars = max_observation_chars

    # ---- 注册与查询 ----
    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("工具必须有 name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.spec().to_openai_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """给提示词用的人类可读工具清单。"""
        lines = []
        for tool in self._tools.values():
            params = tool.parameters.get("properties", {}) or {}
            required = set(tool.parameters.get("required", []) or [])
            args = ", ".join(
                f"{k}{'*' if k in required else ''}: {v.get('type', 'any')}"
                for k, v in params.items()
            )
            lines.append(f"- {tool.name}({args})：{tool.description}")
        return "\n".join(lines)

    # ---- 调用 ----
    def invoke(self, call: ToolCall, context: ToolContext | None = None) -> ToolResult:
        context = context or ToolContext()
        started = time.perf_counter()

        tool = self.get(call.name)
        if tool is None:
            return self._fail(
                f"不存在名为 {call.name} 的工具。可用工具：{', '.join(self.names())}",
                started,
            )

        invalid = tool.validate(call.arguments or {})
        if invalid:
            return self._fail(f"参数不合法：{invalid}", started)

        try:
            result = tool.run(call.arguments or {}, context)
        except ToolError as exc:
            return self._fail(str(exc), started)
        except Exception as exc:  # 工具作者没预料到的异常也要兜住
            detail = traceback.format_exc(limit=3).strip().splitlines()[-1]
            return self._fail(f"工具内部异常：{type(exc).__name__}: {exc}（{detail}）", started)

        result.elapsed_ms = (time.perf_counter() - started) * 1000
        # 统一裁剪：太长就截断，并明确告知模型「内容被截断了」
        limit = self.max_observation_chars
        if len(result.content) > limit:
            result.content = result.content[:limit] + f"\n…（内容过长，已截断至 {limit} 字符）"
            result.truncated = True
        return result

    def _fail(self, message: str, started: float) -> ToolResult:
        return ToolResult(
            ok=False,
            content=f"工具调用失败：{message}",
            error=message,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
