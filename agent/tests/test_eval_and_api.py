"""评估层与 API 层测试。

API 测试用 FastAPI 的 TestClient 直接调用（不起真实端口），
并把轨迹存储隔离到临时目录，避免污染真实数据。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.brain.base import Decision
from app.eval.runner import (
    EvalTask,
    judge_task,
    load_tasks,
    report_to_markdown,
    run_agent_eval,
)
from app.schema import Step, ToolCall, ToolResult, Trace
from tests.test_agent import ScriptedBrain, final_step, make_agent, tool_step


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #
def _trace(tools: list[str], answer: str = "答案", stop: str = "finished") -> Trace:
    trace = Trace(run_id="r1", task="t", stop_reason=stop, final_answer=answer)  # type: ignore[arg-type]
    for i, name in enumerate(tools):
        call = ToolCall(name=name, arguments={}, call_id=f"c{i}")
        trace.steps.append(Step(index=i, kind="tool_call", tool_call=call))
        trace.steps.append(
            Step(index=i, kind="tool_result", tool_call=call,
                 tool_result=ToolResult(ok=True, content="ok"))
        )
    return trace


def test_judge_completed_when_tool_and_keyword_match():
    task = EvalTask(id="t1", task="问", expected_tools=["search_course_kb"],
                    answer_must_include=["增益"])
    outcome = judge_task(task, _trace(["search_course_kb"], "信息增益的定义是…"))
    assert outcome.completed is True
    assert outcome.tool_hit is True
    assert outcome.coverage > 0


def test_judge_fails_when_expected_tool_not_used():
    task = EvalTask(id="t1", task="问", expected_tools=["search_course_kb"])
    outcome = judge_task(task, _trace(["web_search"], "答案"))
    assert outcome.completed is False
    assert outcome.tool_hit is False
    assert "未使用期望工具" in outcome.note


def test_judge_fails_when_any_expected_tool_matches():
    """expected_tools 是「任一命中即可」，不是全部命中。"""
    task = EvalTask(id="t1", task="问", expected_tools=["calculator", "run_python"])
    assert judge_task(task, _trace(["run_python"])).completed is True


def test_judge_fails_on_forbidden_tool():
    """课程内问题去联网 = 失败（考察工具选择是否克制）。"""
    task = EvalTask(id="t1", task="问", forbid_tools=["web_search"])
    outcome = judge_task(task, _trace(["search_course_kb", "web_search"]))
    assert outcome.completed is False
    assert outcome.forbidden_used == ["web_search"]
    assert "不该用的工具" in outcome.note


def test_judge_fails_when_no_keyword_hit():
    task = EvalTask(id="t1", task="问", answer_must_include=["基尼", "增益"])
    outcome = judge_task(task, _trace([], "完全无关的回答"))
    assert outcome.completed is False
    assert outcome.coverage == 0.0
    assert "未命中任何预期关键词" in outcome.note


def test_judge_fails_when_not_finished():
    task = EvalTask(id="t1", task="问")
    outcome = judge_task(task, _trace([], "半截答案", stop="max_steps"))
    assert outcome.completed is False
    assert "未正常结束" in outcome.note


def test_judge_handles_empty_answer():
    task = EvalTask(id="t1", task="问")
    assert judge_task(task, _trace([], "")).completed is False


# --------------------------------------------------------------------------- #
# 任务集与报告
# --------------------------------------------------------------------------- #
def test_load_default_task_set():
    tasks = load_tasks()
    assert len(tasks) >= 10
    assert all(t.id and t.task for t in tasks)
    # 至少要覆盖「不该联网」这类约束样本，否则工具克制性无法评估
    assert any(t.forbid_tools for t in tasks)


def test_run_agent_eval_aggregates_metrics():
    tasks = [
        EvalTask(id="t1", task="算一下 1+1", expected_tools=["calculator"],
                 answer_must_include=["2"]),
        EvalTask(id="t2", task="现在几点", expected_tools=["current_time"]),
    ]
    brain = ScriptedBrain([
        tool_step("calculator", expression="1+1"), final_step("答案是 2"),
        tool_step("current_time"), final_step("现在是下午"),
    ])
    report = run_agent_eval(make_agent(brain), tasks, verbose=False)
    assert report.samples == 2
    assert report.success_rate == 1.0
    assert report.tool_accuracy == 1.0
    assert report.avg_tool_calls == 1.0
    assert report.stop_reason_dist == {"finished": 2}


def test_eval_report_marks_mock_mode():
    """mock 模式的报告必须显式声明「不代表真实能力」，避免数据被误读。"""
    tasks = [EvalTask(id="t1", task="随便")]
    brain = ScriptedBrain([final_step("ok")])
    report = run_agent_eval(make_agent(brain), tasks, verbose=False)
    assert report.is_mock is True
    md = report_to_markdown(report)
    assert "mock" in md
    assert "不能作为 Agent 能力的证据" in md
    # 工程层与能力层必须分开呈现，不能让 50% 这种数字被误读
    assert "工程层指标" in md
    assert "能力层指标" in md


def test_eval_report_separates_real_and_engine_metrics():
    """真实模型模式下应给出能力层指标，而不是「不可用」提示。"""
    from dataclasses import replace

    tasks = [EvalTask(id="t1", task="随便")]
    brain = ScriptedBrain([final_step("ok")])
    report = run_agent_eval(make_agent(brain), tasks, verbose=False)
    real = replace(report, is_mock=False, brain="deepseek-chat")
    md = report_to_markdown(real)
    assert "能力层指标（真实模型决策）" in md
    assert "本次不可用" not in md


def test_engine_metrics_are_computed_from_traces():
    """工程层指标应当来自轨迹本身，与任务是否「答对」无关。"""
    tasks = [EvalTask(id="t1", task="调用一个不存在的工具")]
    brain = ScriptedBrain([tool_step("no_such_tool", x=1), final_step("失败后我改了口径")])
    report = run_agent_eval(make_agent(brain), tasks, verbose=False)
    md = report_to_markdown(report)
    # 虽然工具失败了，但轨迹完整、系统没有崩——工程层应当如实反映
    assert "工具执行成功率" in md
    assert "轨迹完整率" in md
    assert "100.0%" in md          # 轨迹完整率应为 100%


def test_eval_report_json_serializable():
    tasks = [EvalTask(id="t1", task="随便", expected_tools=["calculator"])]
    brain = ScriptedBrain([tool_step("calculator", expression="1"), final_step("1")])
    report = run_agent_eval(make_agent(brain), tasks, verbose=False)
    text = json.dumps(report.to_dict(), ensure_ascii=False)
    payload = json.loads(text)
    assert payload["metrics"]["success_rate"] == 1.0
    assert len(payload["outcomes"]) == 1
    assert "tools" in payload["outcomes"][0]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(tmp_trace_store, monkeypatch):
    """构造一个使用临时轨迹目录的 TestClient。"""
    from app.api import main as api_main
    from app.runtime.agent import Agent
    from app.tools import CalculatorTool, CurrentTimeTool
    from app.tools.base import ToolRegistry

    cfg = Agent().cfg
    registry = ToolRegistry(max_observation_chars=cfg.guard.max_observation_chars)
    registry.register(CurrentTimeTool())
    registry.register(CalculatorTool())
    brain = ScriptedBrain([tool_step("calculator", expression="2+2"), final_step("等于 4")])
    api_main.STATE["agent"] = Agent(registry=registry, brain=brain, settings=cfg)
    api_main.STATE["store"] = tmp_trace_store
    with TestClient(api_main.app) as c:
        yield c
    api_main.STATE.clear()


def test_api_health_lists_tools(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "calculator" in data["preflight"]["tools"]
    assert data["preflight"]["is_mock"] is True


def test_api_tools_schema(client):
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()["tools"]]
    assert "calculator" in names
    calc = next(t for t in resp.json()["tools"] if t["name"] == "calculator")
    assert "expression" in calc["parameters"]["properties"]


def test_api_run_returns_trace_and_persists(client):
    resp = client.post("/api/run", json={"task": "2+2 等于多少"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["stop_reason"] == "finished"
    assert data["metrics"]["num_tool_calls"] == 1
    assert data["steps"]

    listing = client.get("/api/traces").json()
    assert listing["total"] >= 1


def test_api_get_trace_by_id(client):
    run = client.post("/api/run", json={"task": "2+2"}).json()
    got = client.get(f"/api/traces/{run['run_id']}")
    assert got.status_code == 200
    assert got.json()["run_id"] == run["run_id"]


def test_api_get_unknown_trace_returns_404(client):
    assert client.get("/api/traces/does-not-exist").status_code == 404


def test_api_stream_emits_steps_then_done(client):
    with client.stream("POST", "/api/run/stream", json={"task": "2+2"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: step" in body
    assert "event: done" in body
    # done 事件里应带完整 metrics，前端据此渲染汇总
    done_line = [ln for ln in body.splitlines() if ln.startswith("data: ")][-1]
    payload = json.loads(done_line[6:])
    assert payload["metrics"]["num_tool_calls"] == 1


def test_api_run_validates_empty_task(client):
    assert client.post("/api/run", json={"task": ""}).status_code == 422
