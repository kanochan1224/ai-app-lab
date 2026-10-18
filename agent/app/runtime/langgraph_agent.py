"""LangGraph 引擎：与手写的 ``Agent`` 主循环**行为等价**的另一种实现。

为什么同时保留两套？

- 手写版（``runtime/agent.py``）是理解成本最低的实现，
  每一种防护都在一个 ``while`` 里看得见，适合讲清楚「为什么会失控」；
- 这一版把同一个循环表达成**显式状态图**，代价是多了一层抽象，
  收益是拿到了手写版做不到的东西：**checkpointer 持久化**与
  ``interrupt()`` 人机协同——轨迹从「只能回放」变成「能从断点续跑」。

两者的防护语义**逐条对齐**（见下面每个节点里标注的行号，
对应 ``runtime/agent.py`` 的原始实现）。

状态设计上有一条硬约束：**state 里只能放 JSON 原生数据**。
LangGraph 的 checkpointer 走 ormsgpack 序列化，任意 dataclass（比如
``Trace`` / ``Step``）都会序列化失败。所以这里把步骤存成 ``Step.to_dict()``
的形状，需要 ``Trace`` 对象时再复用 ``trace_store._record_to_trace`` 反序列化——
**落盘格式与 checkpoint 格式因此是同一套**，这两条路不会各自漂移。

依赖：``pip install -r requirements-langgraph.txt``（可选，装不上仍可用 ``loop`` 引擎）。
"""

from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, Iterator, TypedDict

from ..brain.base import Brain, BrainError, append_tool_result, build_brain
from ..config import Settings, settings as default_settings
from ..schema import Step, ToolCall, ToolResult, Trace
from ..tools.base import ToolContext, ToolRegistry
from .agent import _count_repeats, _ensure_assistant_tool_calls, build_preflight
from .trace_store import _record_to_trace

# 装不上时的提示：这个项目对「可选依赖」的一贯要求是把话说清楚、给出下一步动作。
_LANGGRAPH_MISSING = (
    "未检测到 langgraph。它是**可选**依赖，只有 LangGraph 引擎需要：\n"
    "    pip install -r requirements-langgraph.txt\n"
    "注意不要直接 `pip install langgraph`——1.x 会强制把 langchain-core 升到 1.x，\n"
    "连带破坏 rag 项目依赖的 langchain / langchain-chroma；requirements 文件里已钉住 <1.0。\n"
    "不装也完全可以：默认的 loop 引擎不需要任何额外依赖。"
)


def _load_langgraph():
    """延迟导入——没装 langgraph 时，这个模块仍然可以被安全 import。

    首次导入会带出一条 LangChain 的 PendingDeprecation 提示
    （``langgraph/cache/base`` 导入 serde 时发出，内容是关于
    ``allowed_objects`` 默认值将来会变）。它描述的是**将来的默认值**，
    对当前用法没有影响，也不是可执行的待办。

    这条提示**屏蔽不掉**：实测 ``pytest.ini`` 的 filterwarnings、
    ``warnings.catch_warnings()`` 包住 import、包住 ``compile()`` 三种方式均无效
    （它绕过了标准的警告过滤路径）。与其加一层不起作用的包装假装解决了，
    不如在这里如实说明——看到它时知道是无害的即可。
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:  # pragma: no cover - 兼容旧版命名
            from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(_LANGGRAPH_MISSING) from exc
    return StateGraph, START, END, interrupt, Command, InMemorySaver


class AgentState(TypedDict, total=False):
    """图状态。

    三条 reducer 是刻意用上的——图里多个节点都会写这些字段，
    默认的「后写覆盖」会让先写的那次静默丢失：

    - ``trace_steps`` 追加而不是覆盖（否则每一步都会把上一步挤掉）；
    - ``prompt_tokens`` / ``completion_tokens`` 累加而不是覆盖。

    这正是「为什么 reducer 是一等公民」的实证：没有它，
    并行或连续写入同一字段就是数据丢失。
    """

    trace_steps: Annotated[list[dict[str, Any]], operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]

    task: str
    messages: list[dict[str, Any]]
    step_index: int
    consecutive_failures: int
    call_signatures: dict[str, int]
    pending_calls: list[dict[str, Any]]
    decision_thought: str
    stop_reason: str
    final_answer: str
    error: str
    route: str          # 条件边读它决定下一步（"decide"/"tools"/"end"）
    deadline: float     # 本次执行的墙钟截止时间，见 _initial_state 的说明


class LangGraphAgent:
    """用 LangGraph 状态图实现的 Agent，对外契约与 ``Agent`` 完全一致。

    鸭子类型对齐 ``run`` / ``stream`` / ``preflight`` / ``brain`` / ``cfg`` / ``registry``，
    因此 ``eval/runner.py``、API 层、CLI、前端都无需为它做任何改动。
    """

    engine = "langgraph"

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        brain: Brain | None = None,
        settings: Settings | None = None,
        checkpointer: Any = None,
        interactive: bool = False,
    ) -> None:
        StateGraph, START, END, interrupt, Command, InMemorySaver = _load_langgraph()

        self.cfg = settings or default_settings
        self.cfg.ensure_dirs()
        if registry is None:
            from ..tools import build_registry

            registry = build_registry(self.cfg)
        self.registry = registry
        self.brain = brain or build_brain(self.cfg.brain)
        self._mock_warning = self.cfg.brain.is_mock

        self._interrupt = interrupt
        self._checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        # 人机协同：危险工具执行前暂停等审批。必须配合 checkpointer 才有意义。
        self.interactive = interactive

        graph = StateGraph(AgentState)
        graph.add_node("precheck", self._precheck)
        graph.add_node("decide", self._decide)
        graph.add_node("guard_repeat", self._guard_repeat)
        graph.add_node("tools", self._tools)
        graph.add_node("guard_failures", self._guard_failures)

        # 循环必须回到 precheck 而不是 decide —— 步数与超时是「每一轮开头都要查」的，
        # 只做一次等于没有。这一个节点就是手写版 while 顶部的两段 if。
        graph.add_edge(START, "precheck")
        for node in ("precheck", "decide", "guard_repeat"):
            graph.add_conditional_edges(node, _route_of, _routes(END))
        graph.add_edge("tools", "guard_failures")
        graph.add_edge("guard_failures", "precheck")

        self._graph = graph.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------ #
    # 对外契约（与 Agent 完全一致）
    # ------------------------------------------------------------------ #
    def run(self, task: str, history: list[dict[str, Any]] | None = None) -> Trace:
        """跑一个任务，返回完整轨迹。

        取结果的顺序是**先读 checkpoint 再看返回值**：图被 ``interrupt()``
        打断时，``invoke()`` 的返回值不包含中断前已提交的那些步骤，
        直接用它会让「暂停」显示成「一步都没走」。checkpoint 才是权威状态。
        """
        trace, config, state = self._start(task, history)
        result = self._graph.invoke(state, config=config)
        return self._to_trace(trace, self._read_state(config, result or state))

    def stream(
        self, task: str, history: list[dict[str, Any]] | None = None
    ) -> Iterator[tuple[str, Any]]:
        """流式跑任务，事件名与载荷形状与手写版保持一致（前端与 SSE 依赖它）。

        用 ``stream_mode="values"``（每个超步后的**完整**状态）而不是 ``"updates"``：
        这样「哪些步还没发出去」可以直接用下标比对，**从 checkpoint 恢复时也不会
        漏发或多发**——updates 模式在恢复场景下需要自己推断哪些是历史步。
        """
        trace, config, state = self._start(task, history)
        emitted = 0
        final_state: dict[str, Any] = state
        for snapshot in self._graph.stream(state, config=config, stream_mode="values"):
            final_state = snapshot or final_state
            steps = final_state.get("trace_steps") or []
            while emitted < len(steps):
                yield "step", _step_from_dict(steps[emitted])
                emitted += 1
        yield "done", self._to_trace(trace, final_state)

    def preflight(self) -> dict[str, Any]:
        """与手写版共用同一份自检逻辑，避免两套实现给出不同结论。"""
        return build_preflight(self.cfg, self.registry, self._mock_warning)

    # ------------------------------------------------------------------ #
    # checkpoint / 中断恢复
    # ------------------------------------------------------------------ #
    def get_state(self, run_id: str) -> dict[str, Any] | None:
        """读取某个 run 的 checkpoint（不存在返回 None）。"""
        snapshot = self._graph.get_state(self._config(run_id))
        return dict(snapshot.values) if snapshot and snapshot.values else None

    def pending_interrupt(self, run_id: str) -> Any:
        """该 run 是否正停在一个人工审批点上；是则返回中断载荷。"""
        snapshot = self._graph.get_state(self._config(run_id))
        if snapshot is None:
            return None
        for task in getattr(snapshot, "tasks", ()) or ():
            for item in getattr(task, "interrupts", ()) or ():
                return getattr(item, "value", item)
        return None

    def resume(self, run_id: str, value: Any) -> Trace:
        """从中断点继续跑。``value`` 会作为 ``interrupt()`` 的返回值交给节点。

        **这是手写版做不到的事**：``TraceStore`` 只能把历史轨迹读出来展示，
        想「接着往下跑」需要模型上下文、循环计数器、防护状态全都在，
        而这些恰恰是 checkpointer 存下的东西。
        """
        Command = _load_langgraph()[4]
        config = self._config(run_id)
        snapshot = self._graph.get_state(config)
        state = dict(snapshot.values) if snapshot and snapshot.values else {}
        # 人工等待的这段时间不该计入「Agent 总时长预算」，所以续跑时重新计时
        state["deadline"] = time.time() + self.cfg.guard.max_seconds
        result = self._graph.invoke(Command(resume=value), config=config)
        trace = Trace(
            run_id=run_id,
            task=state.get("task", ""),
            brain=self.brain.label,
            tools_available=self.registry.names(),
        )
        return self._to_trace(trace, self._read_state(config, result or state))

    def resume_stream(self, run_id: str, value: Any) -> Iterator[tuple[str, Any]]:
        """``resume`` 的流式版本。"""
        Command = _load_langgraph()[4]
        config = self._config(run_id)
        snapshot = self._graph.get_state(config)
        state = dict(snapshot.values) if snapshot and snapshot.values else {}
        state["deadline"] = time.time() + self.cfg.guard.max_seconds

        trace = Trace(
            run_id=run_id,
            task=state.get("task", ""),
            brain=self.brain.label,
            tools_available=self.registry.names(),
        )
        # 已经落过盘的步骤不再重复推送（用一开始的步数当下界）
        emitted = len(state.get("trace_steps") or [])
        final_state = state
        for snapshot_values in self._graph.stream(
            Command(resume=value), config=config, stream_mode="values"
        ):
            final_state = snapshot_values or final_state
            steps = final_state.get("trace_steps") or []
            while emitted < len(steps):
                yield "step", _step_from_dict(steps[emitted])
                emitted += 1
        yield "done", self._to_trace(trace, final_state)

    # ------------------------------------------------------------------ #
    # 图节点（每个都标注了它对应手写版的哪一段）
    # ------------------------------------------------------------------ #
    def _precheck(self, state: AgentState) -> dict[str, Any]:
        """防护 1（步数）与防护 2（总时长）—— 对应 agent.py:111-137。

        注意步数计的是**已产生的步数**（含 guard/error 步），
        所以保护步本身也占用额度，这与手写版一致。
        """
        index = state.get("step_index", 0)
        if index >= self.cfg.guard.max_steps:
            step = Step(
                index=index,
                kind="error",
                content=f"已达步数上限 {self.cfg.guard.max_steps}，停止。",
                guard_note="max_steps",
            )
            return _stop(step, "max_steps")
        if time.time() > state.get("deadline", float("inf")):
            step = Step(
                index=index,
                kind="error",
                content=f"已超过总时长上限 {self.cfg.guard.max_seconds:.0f} 秒，停止。",
                guard_note="timeout",
            )
            return _stop(step, "timeout")
        return {"route": "decide"}

    def _decide(self, state: AgentState) -> dict[str, Any]:
        """调模型拿决策 —— 对应 agent.py:139-178。"""
        messages = list(state.get("messages") or [])
        index = state.get("step_index", 0)
        step = Step(index=index, kind="thought")
        started = time.perf_counter()
        try:
            decision = self.brain.decide(messages, self.registry.specs())
        except BrainError as exc:
            step.kind = "error"
            step.content = f"决策失败：{exc}"
            step.elapsed_ms = (time.perf_counter() - started) * 1000
            return {
                "trace_steps": [step.to_dict()],
                "step_index": index + 1,
                "messages": messages,
                "stop_reason": "llm_error",
                "error": str(exc),
                "route": "end",
            }
        step.elapsed_ms = (time.perf_counter() - started) * 1000
        step.prompt_tokens = decision.prompt_tokens
        step.completion_tokens = decision.completion_tokens
        tokens = {
            "prompt_tokens": decision.prompt_tokens,
            "completion_tokens": decision.completion_tokens,
        }

        if decision.kind == "final":
            step.kind = "final"
            step.content = decision.thought or decision.final_answer
            return {
                "trace_steps": [step.to_dict()],
                "step_index": index + 1,
                "messages": messages,
                "final_answer": decision.final_answer,
                "stop_reason": "finished",
                "route": "end",
                **tokens,
            }

        calls = decision.tool_calls
        if not calls:
            step.kind = "error"
            step.content = "决策既没有调用工具也没有给出答案，停止以避免空转。"
            return {
                "trace_steps": [step.to_dict()],
                "step_index": index + 1,
                "messages": messages,
                "stop_reason": "llm_error",
                "route": "end",
                **tokens,
            }

        step.kind = "tool_call"
        step.content = decision.thought
        step.tool_calls = list(calls)
        return {
            "trace_steps": [step.to_dict()],
            "step_index": index + 1,
            "messages": messages,
            "decision_thought": decision.thought,
            "pending_calls": [
                {"name": c.name, "arguments": c.arguments, "call_id": c.call_id} for c in calls
            ],
            "route": "guard_repeat",
            **tokens,
        }

    def _guard_repeat(self, state: AgentState) -> dict[str, Any]:
        """防护 3（原地打转）—— 对应 agent.py:180-219。

        两处反直觉但必须照做的语义：

        1. 触发时这一批工具调用**一个都不执行**、assistant 消息也**不回填**历史，
           只注入一条提醒就回到决策节点；
        2. 计数是「同名同参」的签名，且**所有**调用都先计数再判定，
           不是只数重复的那个。
        """
        calls = _calls_of(state)
        signatures = dict(state.get("call_signatures") or {})
        repeated = [
            c for c in calls if _count_repeats(c, signatures) > self.cfg.guard.repeat_tolerance
        ]
        index = state.get("step_index", 0)

        if not repeated:
            # 全部是新调用：签名计数写回，交给 tools 节点执行
            return {"call_signatures": signatures, "route": "tools"}

        first = repeated[0]
        count = signatures[first.signature()]
        guard = Step(
            index=index,
            kind="guard",
            content=(
                f"检测到重复调用 {first.name}（相同参数已出现 {count} 次），"
                "注入提醒并要求更换策略。"
            ),
            guard_note="repeat",
        )
        messages = list(state.get("messages") or [])
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
        update: dict[str, Any] = {
            "trace_steps": [guard.to_dict()],
            "step_index": index + 1,
            "call_signatures": signatures,
            "messages": messages,
            # 回到循环开头：这一批工具不执行，但要重新过一遍步数/超时检查
            "route": "precheck",
        }
        if count > self.cfg.guard.repeat_tolerance + 1:
            end = Step(
                index=index + 1,
                kind="error",
                content="重复调用过多，判定无进展，停止。",
                guard_note="no_progress",
            )
            update["trace_steps"] = [guard.to_dict(), end.to_dict()]
            update["step_index"] = index + 2
            update["stop_reason"] = "no_progress"
            update["route"] = "end"
        return update

    def _tools(self, state: AgentState) -> dict[str, Any]:
        """执行工具并回填历史 —— 对应 agent.py:221-247（外加人机审批）。"""
        calls = _calls_of(state)
        index = state.get("step_index", 0)
        context = ToolContext(workspace=str(self.cfg.workspace_dir), extra={})
        messages = list(state.get("messages") or [])

        # ---- 人机协同：危险工具先暂停等审批（手写版没有这个能力）----
        rejected = self._collect_approvals(calls, index)

        results = []
        for call in calls:
            if not call.call_id:
                call.call_id = f"call_{index}_{len(results)}"
            if call.call_id in rejected:
                results.append(
                    ToolResult(
                        ok=False,
                        content=f"[已拒绝] 人工审批未通过，{call.name} 未执行。",
                        error="human_rejected",
                    )
                )
                continue
            results.append(self.registry.invoke(call, context))

        result_step = Step(
            index=index,
            kind="tool_result",
            content="\n\n".join(
                f"[{c.name}] {r.content}" if len(calls) > 1 else r.content
                for c, r in zip(calls, results)
            ),
            tool_calls=list(calls),
            tool_results=results,
            elapsed_ms=round(sum(r.elapsed_ms for r in results), 1),
        )

        _ensure_assistant_tool_calls(messages, _ThoughtLike(state), calls)
        failures = state.get("consecutive_failures", 0)
        for call, result in zip(calls, results):
            append_tool_result(messages, call.call_id, call.name, result.content)
            failures = 0 if result.ok else failures + 1

        return {
            "trace_steps": [result_step.to_dict()],
            "step_index": index + 1,
            "messages": messages,
            "consecutive_failures": failures,
            "pending_calls": [],
        }

    def _guard_failures(self, state: AgentState) -> dict[str, Any]:
        """防护 4（连续失败）—— 对应 agent.py:249-273。

        刻意**不终止循环**：只注入一条提醒并清零计数，是否收手交给模型。
        直接掐断会让用户看到「突然失败」，而提醒后模型往往能自己调整。
        """
        failures = state.get("consecutive_failures", 0)
        if failures < self.cfg.guard.max_consecutive_failures:
            return {"route": "precheck"}

        index = state.get("step_index", 0)
        guard = Step(
            index=index,
            kind="guard",
            content=(
                f"工具已连续失败 {failures} 次，"
                "要求立即基于现有信息作答或说明无法完成。"
            ),
            guard_note="too_many_failures",
        )
        messages = list(state.get("messages") or [])
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
        return {
            "trace_steps": [guard.to_dict()],
            "step_index": index + 1,
            "messages": messages,
            "consecutive_failures": 0,
            "route": "precheck",
        }

    # ------------------------------------------------------------------ #
    def _collect_approvals(self, calls: list[ToolCall], index: int) -> set[str]:
        """对危险工具逐个发起人工审批，返回被拒绝的 call_id 集合。"""
        if not self.interactive:
            return set()
        rejected: set[str] = set()
        for pos, call in enumerate(calls):
            tool = self.registry.get(call.name)
            if tool is None or not getattr(tool, "dangerous", False):
                continue
            if not call.call_id:
                call.call_id = f"call_{index}_{pos}"
            verdict = self._interrupt(
                {
                    "type": "tool_approval",
                    "tool": call.name,
                    "arguments": call.arguments,
                    "question": f"工具 {call.name} 被标记为危险操作，是否放行？",
                    "options": ["approve", "reject"],
                }
            )
            if str(verdict).lower() not in {"approve", "ok", "yes", "true"}:
                rejected.add(call.call_id)
        return rejected

    # ------------------------------------------------------------------ #
    def _read_state(self, config: dict[str, Any], fallback: Any) -> dict[str, Any]:
        """从 checkpointer 读回权威状态；读不到才退回到调用方的返回值。"""
        snapshot = self._graph.get_state(config)
        if snapshot is not None and snapshot.values:
            return dict(snapshot.values)
        return dict(fallback) if isinstance(fallback, dict) else {}

    def _config(self, run_id: str) -> dict[str, Any]:
        """编译图的运行配置。

        ``recursion_limit`` 必须显式给：LangGraph 默认 25 个超步，
        而这里一轮循环最多走 5 个节点（precheck/decide/guard_repeat/tools/
        guard_failures），``max_steps`` 默认 12 就会撞上限并抛
        ``GraphRecursionError``。按步数上限推算一个够用的值，
        并且**保留 ``max_steps`` 作为唯一真正的循环约束**——
        让框架的兜底上限退到防护之后，否则报错会先于防护触发。
        """
        limit = (self.cfg.guard.max_steps + 3) * 5 + 10
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": max(25, limit),
        }

    def _start(
        self, task: str, history: list[dict[str, Any]] | None
    ) -> tuple[Trace, dict[str, Any], AgentState]:
        trace = Trace(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            brain=self.brain.label,
            tools_available=self.registry.names(),
        )
        return trace, self._config(trace.run_id), self._initial_state(trace, task, history)

    def _initial_state(
        self, trace: Trace, task: str, history: list[dict[str, Any]] | None
    ) -> AgentState:
        messages: list[dict[str, Any]] = [self.brain.system_message(self.registry.specs())]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})
        return {
            "trace_steps": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "messages": messages,
            "step_index": 0,
            "consecutive_failures": 0,
            "call_signatures": {},
            "pending_calls": [],
            "decision_thought": "",
            "stop_reason": "",
            "final_answer": "",
            "error": "",
            "route": "decide",
            # 截止时间按「本次执行」算：从 checkpoint 续跑时重新计时，
            # 这样人工审批等待的那几分钟不会把 Agent 的时长预算耗光。
            "deadline": time.time() + self.cfg.guard.max_seconds,
            "task": task,
        }

    def _to_trace(self, trace: Trace, state: dict[str, Any]) -> Trace:
        """把图状态还原成 ``Trace``。

        复用 ``trace_store`` 的反序列化器，而不是另写一份——
        **checkpoint 里的步骤格式与落盘格式是同一种**，两条路共用一套代码。
        """
        steps = list(state.get("trace_steps") or [])
        # 没有 stop_reason 说明这轮是「停在人工审批点上」，不是正常结束。
        # 用 "aborted"（这个取值此前一直声明未被使用）如实标注，避免
        # 把一次暂停误报成 finished，否则评估脚本会把它算成任务成功。
        stop_reason = state.get("stop_reason") or "aborted"
        finished = trace.finished_at or time.time()
        rebuilt = _record_to_trace(
            {
                "run_id": trace.run_id,
                "task": trace.task,
                "brain": trace.brain,
                "started_at": trace.started_at,
                "finished_at": finished,
                "final_answer": state.get("final_answer") or "",
                "stop_reason": stop_reason,
                "error": state.get("error") or "",
                "tools_available": trace.tools_available,
                "metrics": {
                    "prompt_tokens": state.get("prompt_tokens") or 0,
                    "completion_tokens": state.get("completion_tokens") or 0,
                },
                "steps": steps,
            }
        )
        # 与手写版一致：声称 finished 却没有答案时补一句占位，避免评估把它当成有效回答
        if not rebuilt.final_answer and rebuilt.stop_reason == "finished":
            rebuilt.final_answer = "(模型未给出答案)"
        return rebuilt


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
class _ThoughtLike:
    """``_ensure_assistant_tool_calls`` 只用到 ``decision.thought``，这里给个最小壳。"""

    def __init__(self, state: dict[str, Any]) -> None:
        self.thought = state.get("decision_thought") or ""


def _route_of(state: AgentState) -> str:
    """条件边：读节点写下的 ``route`` 字段决定去哪。"""
    return state.get("route") or "end"


def _routes(end: Any) -> dict[str, Any]:
    """条件边的取值表。所有节点共用一份，避免各写各的把路由目标拼错。"""
    return {
        "precheck": "precheck",
        "decide": "decide",
        "guard_repeat": "guard_repeat",
        "tools": "tools",
        "end": end,
    }


def _stop(step: Step, reason: str) -> dict[str, Any]:
    return {
        "trace_steps": [step.to_dict()],
        "step_index": step.index + 1,
        "stop_reason": reason,
        "route": "end",
    }


def _calls_of(state: AgentState) -> list[ToolCall]:
    return [
        ToolCall(
            name=raw.get("name", ""),
            arguments=raw.get("arguments") or {},
            call_id=raw.get("call_id", ""),
        )
        for raw in state.get("pending_calls") or []
    ]


def _step_from_dict(raw: dict[str, Any]) -> Step:
    """把一条序列化步骤还原成 ``Step``（与 ``_record_to_trace`` 同一套规则）。"""
    step = Step(
        index=int(raw.get("index", 0)),
        kind=raw.get("kind", "thought"),
        content=raw.get("content", ""),
        started_at=float(raw.get("started_at") or 0.0),
        elapsed_ms=float(raw.get("elapsed_ms") or 0.0),
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        guard_note=raw.get("guard_note", ""),
    )
    for item in raw.get("tool_calls") or []:
        step.tool_calls.append(
            ToolCall(
                name=item.get("name", ""),
                arguments=item.get("arguments") or {},
                call_id=item.get("call_id", ""),
            )
        )
    for item in raw.get("tool_results") or []:
        step.tool_results.append(
            ToolResult(
                ok=bool(item.get("ok")),
                content=item.get("content", ""),
                error=item.get("error", ""),
                elapsed_ms=float(item.get("elapsed_ms") or 0.0),
                truncated=bool(item.get("truncated")),
            )
        )
    if step.tool_calls:
        step.tool_call = step.tool_calls[0]
    if step.tool_results:
        step.tool_result = step.tool_results[0]
    return step
