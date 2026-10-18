"""LangGraph 引擎：等价性、checkpoint 恢复、以及对外契约。

**整个模块在未安装 langgraph 时跳过**（``importorskip``）——
「109 个测试全离线可跑、不需要任何密钥」是这个项目的承诺，
不能因为加了一个可选引擎就把它变成「必须先装一堆东西才能跑测试」。
装了 langgraph 的人会多跑这一组，没装的人不受影响。
"""

from __future__ import annotations

import json

import pytest

# 首次导入 langgraph 会带出一条 langchain 的 PendingDeprecation 提示
# （serde 默认值将来会变）。实测屏蔽不掉，见 langgraph_agent._load_langgraph 的说明。
pytest.importorskip("langgraph", reason="LangGraph 是可选依赖，未安装时跳过")

from app.brain.base import Brain, Decision                      # noqa: E402
from app.config import Settings                                 # noqa: E402
from app.runtime.agent import Agent                             # noqa: E402
from app.runtime.factory import build_agent                     # noqa: E402
from app.runtime.langgraph_agent import LangGraphAgent          # noqa: E402
from app.runtime.trace_store import TraceStore                  # noqa: E402
from app.schema import ToolCall, ToolResult                     # noqa: E402
from app.tools.base import Tool, ToolContext, ToolRegistry      # noqa: E402
from app.tools.calculator import CalculatorTool                 # noqa: E402


# --------------------------------------------------------------------------- #
# 测试替身（与 test_agent.py 同一套路：注入 Brain 子类，不 mock 网络层）
# --------------------------------------------------------------------------- #
class ScriptedBrain(Brain):
    """按剧本逐条吐决策的假大脑——完全确定，所以两个引擎必须给出同样的轨迹。"""

    label = "scripted"

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> Decision:
        if self._decisions:
            return self._decisions.pop(0)
        return Decision(kind="final", final_answer="（脚本用尽）")

    def system_message(self, tools) -> dict[str, str]:
        return {"role": "system", "content": "scripted"}


class EchoTool(Tool):
    """输出完全确定的小工具。

    **等价性测试不能用 `current_time`**：它返回墙钟时间，两次运行会差几秒，
    逐字比较就会随机失败。比较引擎行为时必须把「工具本身的不确定性」排除掉，
    否则测试变成掷骰子。
    """

    name = "echo"
    description = "原样返回输入"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, arguments, context) -> ToolResult:
        return ToolResult(ok=True, content=f"echo:{arguments.get('text', '')}")


class DangerousTool(Tool):
    """被标记为危险的工具——人工审批的拦截对象。"""

    name = "danger"
    description = "一个危险操作"
    parameters = {"type": "object", "properties": {}, "required": []}
    dangerous = True

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments, context) -> ToolResult:
        self.calls += 1
        return ToolResult(ok=True, content="危险操作已执行")


def tool_step(name: str, **kwargs) -> Decision:
    return Decision(kind="tool", thought="调用工具", tool_calls=[ToolCall(name=name, arguments=kwargs)])


def final_step(text: str) -> Decision:
    return Decision(kind="final", final_answer=text)


def make(engine: str, decisions: list[Decision], **guard_overrides):
    """按引擎名装配一个 Agent，防护阈值可逐项覆盖。"""
    cfg = Settings()
    for key, value in guard_overrides.items():
        setattr(cfg.guard, key, value)
    registry = ToolRegistry(max_observation_chars=cfg.guard.max_observation_chars)
    registry.register(EchoTool())
    registry.register(CalculatorTool())
    cls = Agent if engine == "loop" else LangGraphAgent
    return cls(registry=registry, brain=ScriptedBrain(list(decisions)), settings=cfg)


def skeleton(trace) -> tuple:
    """抽出与时间/ID 无关的结构骨架，用于「两个引擎是否等价」的比较。

    刻意排除 started_at / elapsed_ms / token / run_id：
    这些本来就该不同，把它们算进去会让等价性断言永远失败。
    """
    steps = [
        (
            step.kind,
            step.content,
            tuple((c.name, tuple(sorted(c.arguments.items()))) for c in step.all_calls()),
            tuple(r.ok for r in step.all_results()),
            step.guard_note,
        )
        for step in trace.steps
    ]
    return steps, trace.stop_reason, trace.final_answer


# --------------------------------------------------------------------------- #
# 一、等价性：同一剧本下，两个引擎必须给出同一条轨迹
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "正常完成": ([tool_step("calculator", expression="1+2"), final_step("3")], {}),
    "多步后作答": (
        [tool_step("echo", text="a"), tool_step("calculator", expression="2*3"), final_step("6")],
        {},
    ),
    # 每一步都用同一个参数 → 第 3 次触发重复提醒，第 4 次判定无进展
    "重复调用防护": (
        [tool_step("calculator", expression="1+1") for _ in range(10)],
        {"repeat_tolerance": 1},
    ),
    # 步数防护：步数上限比「重复防护触发点」更早到，考验两个引擎的先后顺序一致
    "步数防护": ([tool_step("echo", text="x") for _ in range(50)], {"max_steps": 4}),
    # 每次用不同的非法表达式，避免先撞上重复防护把这条遮蔽掉
    "连续失败防护": (
        [tool_step("calculator", expression=f"bad{i}") for i in range(6)],
        {"max_steps": 10, "max_consecutive_failures": 2},
    ),
    # 同名不同参不算重复——这条锁的是「签名」而不是「工具名」
    "参数变化不算重复": (
        [
            tool_step("calculator", expression="1+1"),
            tool_step("calculator", expression="2+2"),
            final_step("done"),
        ],
        {"repeat_tolerance": 1},
    ),
}


# pytest 会把非 ASCII 的用例名转义成 \uXXXX，读起来很痛苦，
# 所以显式给一套英文 id（场景本身仍用中文命名，便于对照业务含义）。
_SCENARIO_IDS = {
    "正常完成": "normal-finish",
    "多步后作答": "multi-step",
    "重复调用防护": "repeat-guard",
    "步数防护": "max-steps-guard",
    "连续失败防护": "consecutive-failures-guard",
    "参数变化不算重复": "changed-args-not-repeat",
}


def _scenario_id(name: str) -> str:
    return _SCENARIO_IDS.get(name, name)


@pytest.mark.parametrize("name", list(SCENARIOS), ids=_scenario_id)
def test_both_engines_produce_identical_traces(name: str):
    """核心断言：换引擎不改变行为。"""
    decisions, overrides = SCENARIOS[name]
    loop_trace = make("loop", decisions, **overrides).run("任务")
    graph_trace = make("langgraph", decisions, **overrides).run("任务")
    assert skeleton(loop_trace) == skeleton(graph_trace)


@pytest.mark.parametrize("name", list(SCENARIOS), ids=_SCENARIO_IDS)
def test_both_engines_agree_on_stop_reason(name: str):
    """等价性拆开再断言一次：失败时能立刻看出是「停止原因」错了还是「步骤」错了。"""
    decisions, overrides = SCENARIOS[name]
    assert (
        make("loop", decisions, **overrides).run("任务").stop_reason
        == make("langgraph", decisions, **overrides).run("任务").stop_reason
    )


# --------------------------------------------------------------------------- #
# 二、五种防护逐一确认（显式写明期望值，不只靠等价性）
# --------------------------------------------------------------------------- #
def test_max_steps_guard():
    trace = make("langgraph", [tool_step("echo", text="x") for _ in range(50)], max_steps=4).run("跑不停")
    assert trace.stop_reason == "max_steps"
    assert any(s.guard_note == "max_steps" for s in trace.steps)


def test_repeat_guard_injects_reminder_then_stops():
    trace = make(
        "langgraph",
        [tool_step("calculator", expression="1+1") for _ in range(10)],
        repeat_tolerance=1,
    ).run("原地打转")
    guard_notes = [s.guard_note for s in trace.steps if s.kind == "guard"]
    assert "repeat" in guard_notes
    assert trace.stop_reason in {"no_progress", "max_steps"}


def test_consecutive_failures_guard_does_not_terminate():
    """连续失败防护**不终止循环**——它只提醒，是否收手交给模型。

    所以这里脚本给的是「失败 4 次后正常作答」，期望能跑到 finished。
    """
    decisions = [tool_step("calculator", expression=f"bad{i}") for i in range(4)] + [final_step("抱歉，算不了")]
    trace = make("langgraph", decisions, max_steps=12, max_consecutive_failures=2).run("一直失败")
    assert any(s.guard_note == "too_many_failures" for s in trace.steps)
    assert trace.stop_reason == "finished"


def test_observation_is_truncated_by_registry_not_engine():
    """第五种防护在 ToolRegistry 里，与引擎无关——两个引擎都应看到同一条截断文本。"""
    decisions = [tool_step("calculator", expression="1+1"), final_step("ok")]
    for engine in ("loop", "langgraph"):
        cfg = Settings()
        cfg.guard.max_observation_chars = 5
        registry = ToolRegistry(max_observation_chars=5)
        registry.register(CalculatorTool())
        cls = Agent if engine == "loop" else LangGraphAgent
        trace = cls(registry=registry, brain=ScriptedBrain(list(decisions)), settings=cfg).run("t")
        results = [r for s in trace.steps for r in s.all_results()]
        assert results and all(r.truncated for r in results)


# --------------------------------------------------------------------------- #
# 三、checkpoint：这是 LangGraph 版相对手写版唯一的差异化能力
# --------------------------------------------------------------------------- #
def _interactive_agent(decisions):
    cfg = Settings()
    cfg.guard.max_steps = 12
    registry = ToolRegistry(max_observation_chars=cfg.guard.max_observation_chars)
    danger = DangerousTool()
    registry.register(danger)
    agent = LangGraphAgent(
        registry=registry, brain=ScriptedBrain(list(decisions)), settings=cfg, interactive=True
    )
    return agent, danger


def test_pauses_for_approval_then_resumes():
    """危险工具 → 暂停等审批 → 放行 → 跑完。手写版做不到这件事。"""
    agent, danger = _interactive_agent([tool_step("danger"), final_step("已完成")])

    paused = agent.run("执行危险操作")
    assert paused.stop_reason == "aborted", "暂停不该被记成正常结束"
    assert danger.calls == 0, "审批前绝不能真的执行"

    pending = agent.pending_interrupt(paused.run_id)
    assert pending and pending["tool"] == "danger"

    done = agent.resume(paused.run_id, "approve")
    assert done.stop_reason == "finished"
    assert danger.calls == 1
    assert done.final_answer == "已完成"


def test_rejected_approval_becomes_an_observation():
    """拒绝不是让流程崩掉，而是把「被拒绝」当成一次工具结果交回给模型。"""
    agent, danger = _interactive_agent([tool_step("danger"), final_step("那我换个方式")])

    paused = agent.run("执行危险操作")
    done = agent.resume(paused.run_id, "reject")

    assert danger.calls == 0, "拒绝后仍然执行了，这是安全问题"
    assert done.stop_reason == "finished"
    failed = [r for s in done.steps for r in s.all_results() if not r.ok]
    assert failed, "被拒绝的调用应当留下一条失败观察结果"


def test_resumed_trace_keeps_steps_from_before_the_pause():
    """续跑后的轨迹必须包含暂停前的步骤，而不是从零开始。"""
    agent, _ = _interactive_agent([tool_step("danger"), final_step("ok")])
    paused = agent.run("执行危险操作")
    before = len(paused.steps)
    done = agent.resume(paused.run_id, "approve")
    assert len(done.steps) > before
    assert [s.kind for s in done.steps[:before]] == [s.kind for s in paused.steps]


def test_checkpoint_state_is_json_serializable():
    """checkpoint 的硬约束：state 里不能有 dataclass。

    这条测试守着 `langgraph_agent` 那个「state 只放 JSON 原生数据」的设计决定——
    一旦有人图省事把 Trace/Step 对象塞进 state，序列化就会失败（而且是在运行时才炸）。
    """
    agent, _ = _interactive_agent([tool_step("danger"), final_step("ok")])
    paused = agent.run("执行危险操作")
    state = agent.get_state(paused.run_id)
    assert state is not None
    json.dumps(state, ensure_ascii=False)          # 不抛异常即通过


def test_run_id_is_the_thread_id_so_traces_stay_addressable():
    agent, _ = _interactive_agent([tool_step("danger"), final_step("ok")])
    paused = agent.run("执行危险操作")
    assert agent.get_state(paused.run_id) is not None
    assert agent.get_state("nonexistent") is None


# --------------------------------------------------------------------------- #
# 四、对外契约：API / 前端 / 评估脚本都依赖这些形状
# --------------------------------------------------------------------------- #
def test_stream_event_contract_matches_loop_engine():
    """SSE 事件名与载荷形状必须与手写版一致，否则前端和 API 测试一起挂。"""
    decisions = [tool_step("calculator", expression="1+1"), final_step("2")]
    for engine in ("loop", "langgraph"):
        events = list(make(engine, decisions).stream("t"))
        assert [name for name, _ in events][-1] == "done"
        assert all(name in {"step", "done"} for name, _ in events)
        assert events[-1][1].stop_reason == "finished"
        # 每个 step 载荷都要能原样序列化（SSE 里是 json.dumps 出去的）
        for name, payload in events[:-1]:
            assert name == "step"
            json.dumps(payload.to_dict(), ensure_ascii=False)


def test_trace_roundtrips_through_trace_store(tmp_trace_store: TraceStore):
    """LangGraph 产出的 Trace 必须能像手写版一样落盘并读回。"""
    decisions = [tool_step("calculator", expression="2+2"), final_step("4")]
    trace = make("langgraph", decisions).run("t")
    tmp_trace_store.save(trace)
    loaded = tmp_trace_store.get(trace.run_id)
    assert loaded is not None
    assert loaded.stop_reason == trace.stop_reason
    assert [s.kind for s in loaded.steps] == [s.kind for s in trace.steps]


def test_eval_report_records_the_engine():
    """评估报告要能区分两版数据，否则对比表无从生成。"""
    from app.eval.runner import run_agent_eval
    from app.eval.runner import EvalTask

    task = EvalTask(id="t1", task="1+1", expected_tools=["calculator"], answer_must_include=["2"])
    report = run_agent_eval(
        make("langgraph", [tool_step("calculator", expression="1+1"), final_step("2")]),
        [task],
        verbose=False,
    )
    assert report.engine == "langgraph"
    assert report.to_dict()["engine"] == "langgraph"


# --------------------------------------------------------------------------- #
# 五、工厂与可选项的边界
# --------------------------------------------------------------------------- #
def test_factory_defaults_to_loop_and_switches_by_name():
    assert build_agent("loop").engine == "loop"
    assert build_agent("langgraph").engine == "langgraph"


def test_factory_rejects_interactive_for_handwritten_engine():
    """手写版没有持久化状态。**明确报错，而不是静默忽略**——
    静默忽略会让人以为审批生效了，实际上危险工具照跑。"""
    with pytest.raises(ValueError, match="langgraph"):
        build_agent("loop", interactive=True)


def test_factory_rejects_unknown_engine():
    with pytest.raises(ValueError, match="未知引擎"):
        build_agent("tensorflow")
