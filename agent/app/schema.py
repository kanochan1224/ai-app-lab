"""Agent 全链路数据契约。

这个项目最核心的卖点之一就是**可观测性**：每一步想什么、调了什么工具、
参数是什么、返回什么、花了多久、烧了多少 token，全部结构化落盘。

因此数据契构的设计目标是「一次运行能被完整回放」——
不依赖日志文本，任何一步都能从 JSON 还原。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

StepKind = Literal["thought", "tool_call", "tool_result", "final", "error", "guard"]
StopReason = Literal[
    "finished",          # 正常给出最终答案
    "max_steps",         # 达到步数上限
    "timeout",           # 整体超时
    "no_progress",       # 连续重复同一调用，判定卡死
    "llm_error",         # 模型调用失败且无法恢复
    "aborted",           # 人工中止
]


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def signature(self) -> str:
        """用于检测「原地打转」：同名同参视为重复调用。"""
        items = sorted((k, str(v)) for k, v in self.arguments.items())
        return f"{self.name}({items})"


@dataclass
class ToolResult:
    """一次工具执行的结果。失败也是一种正常结果，交给模型决定下一步。"""

    ok: bool
    content: str                     # 给模型看的文本（已做长度裁剪）
    raw: Any = None                  # 给程序看的结构化数据
    error: str = ""                  # 失败原因（ok=False 时）
    elapsed_ms: float = 0.0
    truncated: bool = False          # 内容是否因超长被裁剪

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "truncated": self.truncated,
        }


@dataclass
class ToolSpec:
    """工具的对外声明（会被转成 function calling 的 JSON Schema）。"""

    name: str
    description: str
    parameters: dict[str, Any]            # JSON Schema
    dangerous: bool = False               # 是否需要额外确认（如执行代码）
    returns: str = ""                     # 返回内容的自然语言说明，帮模型决策

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description + (f"\n返回：{self.returns}" if self.returns else ""),
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------- #
# 知识库检索用的最小结构
#
# 为什么在 agent-lab 里也定义一套：内置迷你知识库（builtin_kb.py）需要在
# **不依赖 course-rag 项目**的前提下工作——那才是「单独 clone 就能跑」。
# 若这里去 import course-rag 的 Chunk，就又把两个仓库绑死了。
# 字段与 course-rag 的同名结构保持一致，因此检索工具的展示逻辑无需区分来源。
# --------------------------------------------------------------------------- #
@dataclass
class KBChunk:
    """知识库片段（最小结构）。"""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    section_path: str = ""
    heading: str = ""
    chunk_index: int = 0


@dataclass
class KBHit:
    """一条检索结果。"""

    chunk: KBChunk
    score: float
    channel: str = "sparse"
    rank: int = 1


@dataclass
class KBTrace:
    """检索链路（内置库只填稀疏那一路）。"""

    query: str
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    final_hits: int = 0
    rerank_used: bool = False
    stage_ms: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 执行步骤
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """Agent 循环里的一步。

    一次「思考 → 调用 → 观察」可能是多步。注意工具调用与结果都用**列表**表示：
    OpenAI 兼容接口允许模型一轮返回多个 ``tool_calls``（并行调用），
    若只处理第一个，剩下的调用就永远没有回应，服务端会直接返回 400：
    ``insufficient tool messages following tool_calls message``。
    """

    index: int
    kind: StepKind
    content: str = ""                     # 思考文本 / 最终答案 / 错误信息
    # 单个工具调用/结果（保留单数形式，兼容简单场景与历史轨迹）
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    # 并行调用：一轮里的多个调用与多个结果
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guard_note: str = ""                  # 被防护机制拦下时的说明

    def all_calls(self) -> list[ToolCall]:
        """本步涉及的全部工具调用（优先取列表字段）。"""
        if self.tool_calls:
            return list(self.tool_calls)
        return [self.tool_call] if self.tool_call else []

    def all_results(self) -> list[ToolResult]:
        if self.tool_results:
            return list(self.tool_results)
        return [self.tool_result] if self.tool_result else []

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "kind": self.kind,
            "content": self.content,
            "started_at": round(self.started_at, 3),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        # 统一输出成列表，消费方只需处理一种形状
        calls = self.all_calls()
        if calls:
            data["tool_calls"] = [asdict(c) for c in calls]
        results = self.all_results()
        if results:
            data["tool_results"] = [r.to_dict() for r in results]
        if self.guard_note:
            data["guard_note"] = self.guard_note
        return data


# --------------------------------------------------------------------------- #
# 整次运行
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    """一次 Agent 运行的完整轨迹——可落盘、可回放、可评估。"""

    run_id: str
    task: str
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: StopReason = "finished"
    brain: str = ""                       # 用的哪个决策后端
    tools_available: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""

    # ---- 派生指标（评估脚本直接用，避免各处重复计算）----
    @property
    def tool_calls(self) -> list[ToolCall]:
        """所有工具调用（按发生顺序展开，一轮多调用会展开成多条）。

        **只统计 kind="tool_call" 的步骤**：``tool_result`` 步骤上也挂着同一批
        ``ToolCall``（便于前端渲染成对展示），若只判断「有没有挂」会把每次调用数成两遍，
        进而让调用次数、失败率等所有派生指标翻倍。
        """
        calls: list[ToolCall] = []
        for step in self.steps:
            if step.kind == "tool_call":
                calls.extend(step.all_calls())
        return calls

    @property
    def tool_sequences(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def num_failed_tools(self) -> int:
        return sum(
            1
            for step in self.steps
            for result in step.all_results()
            if not result.ok
        )

    @property
    def total_ms(self) -> float:
        if self.finished_at:
            return round((self.finished_at - self.started_at) * 1000, 1)
        return round(sum(s.elapsed_ms for s in self.steps), 1)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def recovered_from_failure(self) -> bool:
        """失败后是否继续推进并最终完成——衡量「自我纠正」能力的关键指标。"""
        if self.stop_reason != "finished":
            return False
        seen_failure = False
        for step in self.steps:
            if any(not r.ok for r in step.all_results()):
                seen_failure = True
            elif seen_failure and step.all_calls():
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "brain": self.brain,
            "started_at": round(self.started_at, 3),
            "finished_at": round(self.finished_at, 3),
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "tools_available": self.tools_available,
            "metrics": {
                "num_steps": self.num_steps,
                "num_tool_calls": self.num_tool_calls,
                "num_failed_tools": self.num_failed_tools,
                "total_ms": self.total_ms,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "recovered_from_failure": self.recovered_from_failure,
            },
            "steps": [s.to_dict() for s in self.steps],
        }

    def summary_line(self) -> str:
        """一行摘要，CLI 与评估报告都用它。"""
        return (
            f"[{self.stop_reason}] 步数={self.num_steps} 工具调用={self.num_tool_calls} "
            f"(失败 {self.num_failed_tools}) 耗时={self.total_ms / 1000:.1f}s "
            f"tokens={self.total_tokens}"
        )
