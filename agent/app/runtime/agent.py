"""Agent 主循环。

这一层是项目的核心：**决策 → 执行 → 观察 → 再决策**，
并在每一步都做防护与记录。

防护机制（Agent 失控基本都源于缺了其中某一项）：

===================  ====================================================
机制                 解决的问题
===================  ====================================================
``max_steps``        无限循环、越查越发散
``max_seconds``      单次任务挂太久（配合工具级超时）
``repeat_tolerance`` 反复调用同一工具同一参数「原地打转」
``max_consecutive_failures``  工具连续失败后仍在硬试
``max_observation_chars``     一次抓取撑爆上下文窗口
===================  ====================================================

每一步都会同时写入 ``Trace``，因此「为什么走了这条路径」是可回放的。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from ..brain.base import (
    Brain,
    BrainError,
    Decision,
    append_assistant_tool_calls,
    append_tool_result,
    build_brain,
)
from ..config import Settings, settings as default_settings
from ..schema import Step, ToolCall, Trace
from ..tools.base import ToolContext, ToolRegistry


class Agent:
    """一个可观测、有防护的多步工具调用 Agent。"""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        brain: Brain | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.cfg = settings or default_settings
        self.cfg.ensure_dirs()
        if registry is None:
            from ..tools import build_registry

            registry = build_registry(self.cfg)
        self.registry = registry
        self.brain = brain or build_brain(self.cfg.brain)
        self._mock_warning = self.cfg.brain.is_mock

    # ------------------------------------------------------------------ #
    def run(self, task: str, history: list[dict[str, Any]] | None = None) -> Trace:
        """跑一个任务，返回完整轨迹（同步版本，供 CLI / API / 评估使用）。"""
        trace = Trace(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            brain=self.brain.label,
            tools_available=self.registry.names(),
        )
        for _ in self._iterate(trace, task, history):
            pass
        return trace

    def stream(self, task: str, history: list[dict[str, Any]] | None = None) -> Iterator[tuple[str, Any]]:
        """流式跑任务：每产生一个事件就 yield，前端可实时看 Agent 在干什么。

        事件类型：``step``（新增一步）、``done``（结束，带最终 Trace）。
        """
        trace = Trace(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            brain=self.brain.label,
            tools_available=self.registry.names(),
        )
        for step in self._iterate(trace, task, history):
            yield "step", step
        yield "done", trace

    # ------------------------------------------------------------------ #
    def _iterate(
        self, trace: Trace, task: str, history: list[dict[str, Any]] | None
    ) -> Iterator[Step]:
        """主循环，产出每一步。"""
        deadline = time.time() + self.cfg.guard.max_seconds
        context = ToolContext(workspace=str(self.cfg.workspace_dir), extra={})

        messages: list[dict[str, Any]] = [self.brain.system_message(self.registry.specs())]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        call_signatures: dict[str, int] = {}
        consecutive_failures = 0
        step_index = 0

        def emit(step: Step) -> Step:
            nonlocal step_index
            step.index = step_index
            step_index += 1
            trace.steps.append(step)
            return step

        while True:
            # ---- 防护 1：步数 ----
            if step_index >= self.cfg.guard.max_steps:
                trace.stop_reason = "max_steps"
                step = emit(
                    Step(
                        index=step_index,
                        kind="error",
                        content=f"已达步数上限 {self.cfg.guard.max_steps}，停止。",
                        guard_note="max_steps",
                    )
                )
                yield step
                break
            # ---- 防护 2：总时长 ----
            if time.time() > deadline:
                trace.stop_reason = "timeout"
                step = emit(
                    Step(
                        index=step_index,
                        kind="error",
                        content=f"已超过总时长上限 {self.cfg.guard.max_seconds:.0f} 秒，停止。",
                        guard_note="timeout",
                    )
                )
                yield step
                break

            # ---- 决策 ----
            step = Step(index=step_index, kind="thought")
            started = time.perf_counter()
            try:
                decision = self.brain.decide(messages, self.registry.specs())
            except BrainError as exc:
                trace.stop_reason = "llm_error"
                trace.error = str(exc)
                step.kind = "error"
                step.content = f"决策失败：{exc}"
                step.elapsed_ms = (time.perf_counter() - started) * 1000
                yield emit(step)
                break
            step.elapsed_ms = (time.perf_counter() - started) * 1000
            step.prompt_tokens = decision.prompt_tokens
            step.completion_tokens = decision.completion_tokens
            trace.prompt_tokens += decision.prompt_tokens
            trace.completion_tokens += decision.completion_tokens

            # ---- 最终答案 ----
            if decision.kind == "final":
                step.kind = "final"
                step.content = decision.thought or decision.final_answer
                yield emit(step)
                trace.final_answer = decision.final_answer
                trace.stop_reason = "finished"
                break

            calls = decision.tool_calls
            if not calls:
                step.kind = "error"
                step.content = "决策既没有调用工具也没有给出答案，停止以避免空转。"
                trace.stop_reason = "llm_error"
                yield emit(step)
                break

            step.kind = "tool_call"
            step.content = decision.thought
            step.tool_calls = list(calls)
            yield emit(step)

            # ---- 防护 3：原地打转（逐个调用分别判定）----
            repeated = [c for c in calls if _count_repeats(c, call_signatures) > self.cfg.guard.repeat_tolerance]
            if repeated:
                first = repeated[0]
                count = call_signatures[first.signature()]
                guard = emit(
                    Step(
                        index=step_index,
                        kind="guard",
                        content=(
                            f"检测到重复调用 {first.name}（相同参数已出现 {count} 次），"
                            "注入提醒并要求更换策略。"
                        ),
                        guard_note="repeat",
                    )
                )
                yield guard
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"注意：你已经用完全相同的参数调用过 {first.name} "
                            f"{count - 1} 次，结果不会改变。"
                            "请换参数、换工具，或基于已有信息直接给出最终答案。"
                        ),
                    }
                )
                if count > self.cfg.guard.repeat_tolerance + 1:
                    trace.stop_reason = "no_progress"
                    end = emit(
                        Step(
                            index=step_index,
                            kind="error",
                            content="重复调用过多，判定无进展，停止。",
                            guard_note="no_progress",
                        )
                    )
                    yield end
                    break
                continue

            # ---- 执行工具（支持一轮多个并行调用）----
            # 协议要求：assistant 的 tool_calls 必须被逐条回应，
            # 少回应任何一个，下一轮请求都会被服务端拒绝。
            results = []
            for call in calls:
                if not call.call_id:
                    call.call_id = f"call_{step_index}_{len(results)}"
                results.append(self.registry.invoke(call, context))

            result_step = Step(
                index=step_index,
                kind="tool_result",
                content="\n\n".join(
                    f"[{c.name}] {r.content}" if len(calls) > 1 else r.content
                    for c, r in zip(calls, results)
                ),
                tool_calls=list(calls),
                tool_results=results,
                elapsed_ms=round(sum(r.elapsed_ms for r in results), 1),
            )
            yield emit(result_step)

            # 回填进对话历史（assistant 的 tool_calls + 每条调用的 tool 消息）
            _ensure_assistant_tool_calls(messages, decision, calls)
            for call, result in zip(calls, results):
                append_tool_result(messages, call.call_id, call.name, result.content)
                consecutive_failures = 0 if result.ok else consecutive_failures + 1

            # ---- 防护 4：连续失败 ----
            if consecutive_failures >= self.cfg.guard.max_consecutive_failures:
                guard = emit(
                    Step(
                        index=step_index,
                        kind="guard",
                        content=(
                            f"工具已连续失败 {consecutive_failures} 次，"
                            "要求立即基于现有信息作答或说明无法完成。"
                        ),
                        guard_note="too_many_failures",
                    )
                )
                yield guard
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "注意：工具已连续多次失败。请不要再尝试调用工具，"
                            "直接基于已获得的信息给出最终答案；"
                            "若确实无法完成，明确说明缺少什么信息。"
                        ),
                    }
                )
                consecutive_failures = 0

        trace.finished_at = time.time()
        if not trace.final_answer and trace.stop_reason == "finished":
            trace.final_answer = "(模型未给出答案)"
        return

    # ------------------------------------------------------------------ #
    def preflight(self) -> dict[str, Any]:
        """启动前的自检：哪些工具真的可用、大脑是什么模式。

        这个检查很有必要：很多「Agent 不工作」其实是索引没建、没联网、
        或没配 Key，提前暴露比让用户看一堆空结果好得多。
        """
        report: dict[str, Any] = {
            "brain": self.cfg.brain.mode_label,
            "is_mock": self._mock_warning,
            "tools": {},
        }
        for name in self.registry.names():
            report["tools"][name] = self._probe(name)
        return report

    def _probe(self, name: str) -> dict[str, Any]:
        from ..tools.course_kb import CourseKBTool

        tool = self.registry.get(name)
        if isinstance(tool, CourseKBTool):
            # 工具会自动降级到内置知识库，所以这里报告「实际用的是哪个」，
            # 而不是简单地把 course-rag 缺失当成不可用
            count = tool.chunk_count
            source = tool.source_label
            if count == 0:
                return {"status": "empty", "detail": "知识库为空"}
            status = "degraded" if "内置" in source else "ok"
            return {"status": status, "detail": f"{count} 个片段 · {source}"}
        if name in {"web_search", "fetch_url"}:
            try:
                import httpx

                httpx.head("https://www.bing.com", timeout=6.0, follow_redirects=True)
                return {"status": "ok", "detail": "网络可达"}
            except Exception as exc:
                return {"status": "degraded", "detail": f"联网不可达：{type(exc).__name__}"}
        if name == "run_python":
            return {"status": "ok", "detail": f"沙箱目录 {self.cfg.workspace_dir}"}
        return {"status": "ok", "detail": "本地工具"}


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _count_repeats(call: ToolCall, signatures: dict[str, int]) -> int:
    """记录一次调用并返回该签名已出现的次数。"""
    signature = call.signature()
    signatures[signature] = signatures.get(signature, 0) + 1
    return signatures[signature]


def _ensure_assistant_tool_calls(
    messages: list[dict[str, Any]],
    decision: Any,
    calls: list[ToolCall],
) -> None:
    """确保 assistant 的 tool_calls 消息在历史里恰好存在一次。

    真实后端（OpenAIBrain）在解析响应时已按协议回填过一次；
    MockBrain 这类脚本后端不会，需要在这里补上。
    判据是「历史里最后一条 assistant 消息是否已包含本次这批调用」，
    避免重复追加造成 ``insufficient tool messages`` 之类的协议错误。
    """
    expected = {c.call_id for c in calls}
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        existing = {c.get("id") for c in (msg.get("tool_calls") or [])}
        if existing and existing == expected:
            return          # 已经回填过，且与本次调用一致
        break               # 最后一条 assistant 不是本次的 → 需要补

    append_assistant_tool_calls(
        messages,
        decision.thought or "",
        [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": _json(call.arguments)},
            }
            for call in calls
        ],
    )
