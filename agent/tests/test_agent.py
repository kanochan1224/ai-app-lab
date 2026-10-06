"""Agent 主循环测试：防护机制、失败恢复、可观测性。

这些测试用**脚本化的假大脑**精确控制决策序列，
从而验证「当模型这样做时，循环会不会失控」——
这类测试比只跑一遍真实模型更能暴露防护逻辑的缺陷。
"""

from __future__ import annotations

import json

import pytest

from app.brain.base import Brain, Decision
from app.runtime.agent import Agent
from app.runtime.trace_store import TraceStore
from app.schema import ToolCall
from app.tools import CalculatorTool, CurrentTimeTool
from app.tools.base import ToolRegistry


# --------------------------------------------------------------------------- #
# 可编程的假大脑
# --------------------------------------------------------------------------- #
class ScriptedBrain(Brain):
    """按预设脚本依次给出决策，用于精确构造边界场景。"""

    label = "scripted-test"

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    def decide(self, messages, tools):
        self.calls += 1
        if self._decisions:
            return self._decisions.pop(0)
        return Decision(kind="final", final_answer="（脚本用尽，兜底结束）")


def tool_step(name: str, **kwargs) -> Decision:
    return Decision(
        kind="tool",
        tool_calls=[ToolCall(name=name, arguments=kwargs, call_id=f"c{name}")],
    )


def multi_tool_step(*specs: tuple[str, dict]) -> Decision:
    """一轮返回多个调用（并行调用），用于验证协议兼容性。"""
    return Decision(
        kind="tool",
        tool_calls=[
            ToolCall(name=name, arguments=args, call_id=f"c{i}-{name}")
            for i, (name, args) in enumerate(specs)
        ],
    )


def final_step(text: str = "完成") -> Decision:
    return Decision(kind="final", final_answer=text)


def make_agent(brain: Brain, **overrides) -> Agent:
    from app.config import Settings

    cfg = Settings()
    if overrides:
        for key, value in overrides.items():
            setattr(cfg.guard, key, value)
    registry = ToolRegistry(max_observation_chars=cfg.guard.max_observation_chars)
    registry.register(CurrentTimeTool())
    registry.register(CalculatorTool())
    return Agent(registry=registry, brain=brain, settings=cfg)


# --------------------------------------------------------------------------- #
# 基本流程
# --------------------------------------------------------------------------- #
def test_single_tool_then_final():
    brain = ScriptedBrain(
        [tool_step("calculator", expression="6*7"), final_step("答案是 42")]
    )
    trace = make_agent(brain).run("6 乘 7 等于多少")
    assert trace.stop_reason == "finished"
    assert trace.final_answer == "答案是 42"
    assert trace.tool_sequences == ["calculator"]
    assert trace.num_tool_calls == 1
    assert trace.num_failed_tools == 0


def test_parallel_tool_calls_in_one_turn():
    """一轮返回多个调用时必须全部执行并逐条回应。

    这是真实踩到的坑：模型一次返回两个 tool_calls，若只执行第一个，
    第二个永远没有回应，OpenAI 兼容接口会直接返回 400：
    ``insufficient tool messages following tool_calls message``。
    """
    brain = ScriptedBrain(
        [
            multi_tool_step(
                ("calculator", {"expression": "15*24"}),
                ("current_time", {}),
            ),
            final_step("两个都算完了"),
        ]
    )
    trace = make_agent(brain).run("先算 15*24，再告诉我现在几点")
    assert trace.stop_reason == "finished"
    assert trace.tool_sequences == ["calculator", "current_time"]
    assert trace.num_tool_calls == 2
    assert trace.num_failed_tools == 0

    call_step = next(s for s in trace.steps if s.kind == "tool_call")
    result_step = next(s for s in trace.steps if s.kind == "tool_result")
    assert len(call_step.all_calls()) == 2
    assert len(result_step.all_results()) == 2
    # 两个结果都要出现在注入模型的内容里，否则模型看不到第二个
    assert "360" in result_step.content
    assert "当前时间" in result_step.content


def test_parallel_calls_reflected_in_trace_json():
    brain = ScriptedBrain(
        [
            multi_tool_step(("calculator", {"expression": "1+1"}), ("current_time", {})),
            final_step(),
        ]
    )
    payload = make_agent(brain).run("并行两个").to_dict()
    step = next(s for s in payload["steps"] if s["kind"] == "tool_call")
    assert isinstance(step["tool_calls"], list)
    assert len(step["tool_calls"]) == 2
    # 每个调用都要有 id，否则无法逐条回应
    assert all(c["call_id"] for c in step["tool_calls"])


def test_parallel_calls_partial_failure_still_responds_to_all():
    """并行调用中有一个失败时，两个都要有结果，不能漏回应。"""
    brain = ScriptedBrain(
        [
            multi_tool_step(
                ("calculator", {"expression": "不是表达式"}),
                ("calculator", {"expression": "2+2"}),
            ),
            final_step("一个失败一个成功"),
        ]
    )
    trace = make_agent(brain).run("并行一好一坏")
    assert trace.num_tool_calls == 2
    assert trace.num_failed_tools == 1
    result_step = next(s for s in trace.steps if s.kind == "tool_result")
    assert len(result_step.all_results()) == 2
    assert {r.ok for r in result_step.all_results()} == {True, False}



    """回归测试：tool_call 与 tool_result 都挂同一个 ToolCall，不能数两遍。"""
    brain = ScriptedBrain(
        [
            tool_step("calculator", expression="1+1"),
            tool_step("current_time"),
            final_step(),
        ]
    )
    trace = make_agent(brain).run("随便")
    assert trace.num_tool_calls == 2, f"实际 {trace.num_tool_calls}"
    assert trace.tool_sequences == ["calculator", "current_time"]
    # 每次调用应当恰好有一条对应的结果步骤
    result_steps = [s for s in trace.steps if s.kind == "tool_result"]
    assert len(result_steps) == 2


# --------------------------------------------------------------------------- #
# 防护：步数 / 重复 / 失败
# --------------------------------------------------------------------------- #
def test_max_steps_guard_stops_runaway_loop():
    """模型一直调工具不停 → 必须被步数上限截断，而不是无限跑。"""
    brain = ScriptedBrain([tool_step("current_time") for _ in range(50)])
    trace = make_agent(brain, max_steps=4).run("永不结束的任务")
    assert trace.stop_reason == "max_steps"
    assert trace.num_steps <= 6          # 4 步 + 可能的守卫步骤
    assert any(s.guard_note == "max_steps" for s in trace.steps)


def test_repeat_guard_detects_identical_call():
    """同名同参反复调用 → 注入提醒；仍不改则判定无进展。"""
    brain = ScriptedBrain(
        [tool_step("calculator", expression="1+1") for _ in range(10)]
    )
    trace = make_agent(brain, max_steps=12, repeat_tolerance=1).run("原地打转")
    guards = [s for s in trace.steps if s.kind == "guard"]
    assert guards, "应当触发重复调用防护"
    assert any(s.guard_note == "repeat" for s in guards)
    assert trace.stop_reason in {"no_progress", "max_steps"}


def test_repeat_guard_allows_changing_arguments():
    """换了参数就不算重复，正常流程不应被误伤。"""
    brain = ScriptedBrain(
        [
            tool_step("calculator", expression="1+1"),
            tool_step("calculator", expression="2+2"),
            final_step("两个都算完了"),
        ]
    )
    trace = make_agent(brain, repeat_tolerance=1).run("算两次不同的")
    assert trace.stop_reason == "finished"
    assert trace.num_tool_calls == 2
    assert not any(s.guard_note == "repeat" for s in trace.steps)


def test_tool_failure_is_reported_to_model_and_run_continues():
    """工具失败不能中断流程，要作为观察结果交回模型。"""
    brain = ScriptedBrain(
        [
            tool_step("calculator", expression="不是表达式"),
            final_step("算不了，但我可以说明原因"),
        ]
    )
    trace = make_agent(brain).run("算个非法表达式")
    assert trace.stop_reason == "finished"
    assert trace.num_failed_tools == 1
    failed = [
        s for s in trace.steps
        if s.kind == "tool_result" and any(not r.ok for r in s.all_results())
    ]
    assert failed and "失败" in failed[0].content


def test_unknown_tool_does_not_crash_agent():
    brain = ScriptedBrain(
        [tool_step("nonexistent_tool", foo="bar"), final_step("改用已知工具")]
    )
    trace = make_agent(brain).run("调用不存在的工具")
    assert trace.stop_reason == "finished"
    assert trace.num_failed_tools == 1


def test_consecutive_failures_guard_forces_answer():
    """工具连续失败到阈值 → 注入「别再试了，直接作答」的提醒。

    注意每次用**不同的非法表达式**：否则会先触发「重复调用」防护并把流程截断，
    导致这条防护永远测不到（防护之间会互相遮蔽，这是设计测试时容易忽略的点）。
    """
    brain = ScriptedBrain(
        [tool_step("calculator", expression=f"bad{i}") for i in range(6)]
    )
    trace = make_agent(brain, max_consecutive_failures=2, max_steps=10).run("一直失败")
    guards = [s for s in trace.steps if s.guard_note == "too_many_failures"]
    assert guards, "应当触发连续失败防护"
    assert trace.num_failed_tools >= 2


def test_recovery_flag_marks_success_after_failure():
    """失败后换策略并完成 → recovered_from_failure 应为真。"""
    brain = ScriptedBrain(
        [
            tool_step("calculator", expression="bad"),
            tool_step("calculator", expression="3*3"),
            final_step("9"),
        ]
    )
    trace = make_agent(brain).run("先失败再成功")
    assert trace.num_failed_tools == 1
    assert trace.recovered_from_failure is True


def test_no_recovery_flag_when_run_fails():
    brain = ScriptedBrain([tool_step("calculator", expression="bad") for _ in range(20)])
    trace = make_agent(brain, max_steps=3, max_consecutive_failures=99).run("一直失败")
    assert trace.recovered_from_failure is False


def test_brain_error_stops_gracefully():
    from app.brain.base import BrainError

    class BrokenBrain(Brain):
        label = "broken"

        def decide(self, messages, tools):
            raise BrainError("模型服务不可用")

    trace = make_agent(BrokenBrain()).run("会失败的任务")
    assert trace.stop_reason == "llm_error"
    assert "模型服务不可用" in trace.error
    assert trace.final_answer == ""


def test_brain_returning_neither_tool_nor_answer_stops():
    """模型既不调工具也不给答案 → 不能空转到死，必须停止。"""
    brain = ScriptedBrain([Decision(kind="tool", tool_calls=[])])
    trace = make_agent(brain).run("空决策")
    assert trace.stop_reason == "llm_error"
    assert any("空转" in s.content for s in trace.steps)


# --------------------------------------------------------------------------- #
# 可观测性
# --------------------------------------------------------------------------- #
def test_trace_records_thoughts_and_observations():
    brain = ScriptedBrain(
        [
            Decision(
                kind="tool",
                thought="先算一下",
                tool_calls=[
                    ToolCall(name="calculator", arguments={"expression": "2+2"}, call_id="c1")
                ],
            ),
            final_step("4"),
        ]
    )
    trace = make_agent(brain).run("2+2")
    call_step = next(s for s in trace.steps if s.kind == "tool_call")
    assert call_step.content == "先算一下"
    result_step = next(s for s in trace.steps if s.kind == "tool_result")
    assert "4" in result_step.content
    assert result_step.all_results()[0].ok is True


def test_observation_is_truncated_before_entering_context():
    class FloodTool(CurrentTimeTool):
        name = "flood"

        def run(self, arguments, context):
            from app.schema import ToolResult

            return ToolResult(ok=True, content="A" * 10000)

    from app.config import Settings

    cfg = Settings()
    cfg.guard.max_observation_chars = 300
    registry = ToolRegistry(max_observation_chars=300)
    registry.register(FloodTool())
    brain = ScriptedBrain([tool_step("flood"), final_step()])
    trace = Agent(registry=registry, brain=brain, settings=cfg).run("灌水")
    result_step = next(s for s in trace.steps if s.kind == "tool_result")
    assert result_step.all_results()[0].truncated is True
    assert len(result_step.content) < 500


# --------------------------------------------------------------------------- #
# 轨迹持久化
# --------------------------------------------------------------------------- #
def test_trace_roundtrip_through_store(tmp_trace_store: TraceStore):
    brain = ScriptedBrain(
        [tool_step("calculator", expression="5*5"), final_step("25")]
    )
    trace = make_agent(brain).run("5 乘 5")
    tmp_trace_store.save(trace)

    loaded = tmp_trace_store.load_all()
    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.run_id == trace.run_id
    assert restored.task == trace.task
    assert restored.final_answer == trace.final_answer
    assert restored.stop_reason == trace.stop_reason
    assert restored.tool_sequences == trace.tool_sequences
    assert restored.num_tool_calls == trace.num_tool_calls
    assert restored.num_steps == trace.num_steps


def test_trace_json_is_serializable_and_complete():
    brain = ScriptedBrain([tool_step("calculator", expression="1+1"), final_step()])
    trace = make_agent(brain).run("1+1")
    payload = trace.to_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert "metrics" in payload
    assert payload["metrics"]["num_tool_calls"] == 1
    assert payload["steps"][0]["kind"] == "tool_call"
    assert "tool_results" in json.loads(text)["steps"][1]


def test_store_index_summaries(tmp_trace_store: TraceStore):
    for i in range(3):
        brain = ScriptedBrain([final_step(f"答案{i}")])
        tmp_trace_store.save(make_agent(brain).run(f"任务{i}"))
    summaries = tmp_trace_store.list_summaries()
    assert len(summaries) == 3
    assert summaries[0]["task"] == "任务2"        # 最新的在前
    assert "stop_reason" in summaries[0]


def test_store_survives_corrupted_line(tmp_trace_store: TraceStore):
    """单条损坏的轨迹不应让整次评估失败。"""
    brain = ScriptedBrain([final_step("ok")])
    tmp_trace_store.save(make_agent(brain).run("正常任务"))
    with tmp_trace_store.runs_file.open("a", encoding="utf-8") as f:
        f.write("{这行不是合法 JSON\n")
    traces = tmp_trace_store.load_all()
    assert len(traces) == 1
    assert traces[0].final_answer == "ok"
