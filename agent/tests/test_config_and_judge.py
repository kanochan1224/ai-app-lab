"""配置解析与「假阳性」判据的回归测试。

这两组测试都来自真实踩到的坑，不是凭空写的：
1. ``.env`` 里写 ``KEY=``（留空占位）时，``os.getenv(key, default)`` 会返回空串，
   默认值失效 → 跨项目工具静默指向错误目录；
2. 工具失败后模型礼貌地说「未能获取依据」，旧判据却判它成功 → 最危险的假阳性。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.eval.runner import _NO_ANSWER_MARKERS, looks_like_non_answer
from app.schema import Step, Trace


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #
def test_empty_env_value_falls_back_to_default(monkeypatch):
    """核心回归：变量存在但为空时必须退回默认值。

    这正是 `COURSE_RAG_PATH=` 造成跨项目工具失效的原因。
    """
    from app.config import _env

    monkeypatch.setenv("AGENTLAB_TEST_KEY", "")
    assert _env("AGENTLAB_TEST_KEY", "fallback") == "fallback"

    monkeypatch.setenv("AGENTLAB_TEST_KEY", "   ")
    assert _env("AGENTLAB_TEST_KEY", "fallback") == "fallback"

    monkeypatch.setenv("AGENTLAB_TEST_KEY", ".")
    assert _env("AGENTLAB_TEST_KEY", "fallback") == "fallback"

    monkeypatch.setenv("AGENTLAB_TEST_KEY", "真实值")
    assert _env("AGENTLAB_TEST_KEY", "真实值2") == "真实值"


def test_env_path_resolves_relative_against_project_root(monkeypatch):
    """相对路径要按项目根目录解析，而不是当前工作目录。

    否则同一份配置在 CLI / API / 测试里会指向不同位置，极难排查。
    """
    from app.config import ROOT_DIR, _env_path

    default = ROOT_DIR.parent / "rag"

    monkeypatch.setenv("AGENTLAB_TEST_PATH", "")
    assert _env_path("AGENTLAB_TEST_PATH", default) == default

    monkeypatch.setenv("AGENTLAB_TEST_PATH", "../rag")
    resolved = _env_path("AGENTLAB_TEST_PATH", default)
    assert resolved.is_absolute()
    assert resolved == (ROOT_DIR / ".." / "rag").resolve()


def test_course_rag_path_defaults_to_sibling_project():
    """本项目默认应当指向同级的 course-rag，而不是自己。"""
    from app.config import DEFAULT_RAG_PATH, ROOT_DIR, settings

    assert DEFAULT_RAG_PATH == ROOT_DIR.parent / "rag"
    # 无论 .env 怎么配，都不该解析成 agent-lab 自己
    assert settings.course_rag_path.resolve() != ROOT_DIR.resolve()


def test_course_rag_path_points_at_real_project():
    """若同级存在 course-rag，则应能被正确解析到（这个仓库里它存在）。"""
    from app.config import settings

    expected = settings.course_rag_path / "app" / "retrieval" / "hybrid_retriever.py"
    if settings.course_rag_path.exists():
        assert expected.exists(), f"路径解析不对：{settings.course_rag_path}"


# --------------------------------------------------------------------------- #
# 「没答出来」检测
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "answer",
    [
        "课程知识库检索工具目前无法使用，因此未能获取到相关依据。",
        "联网搜索也没有返回相关内容，未能获取依据。",
        "我无法回答这个问题。",
        "知识库为空，请先建索引。",
        "工具调用失败，无法给出答案。",
        "",
        "   ",
    ],
)
def test_non_answer_is_detected(answer):
    assert looks_like_non_answer(answer) != ""


@pytest.mark.parametrize(
    "answer",
    [
        "信息增益衡量划分前后熵的减少量，用于 ID3；基尼指数用于 CART，不需要对数运算。",
        "15 × 24 = 360，√360 ≈ 18.9737。",
        "本课程成绩由平时作业 20%、实验报告 30%、期末闭卷 50% 构成。",
    ],
)
def test_real_answers_are_not_flagged(answer):
    assert looks_like_non_answer(answer) == ""


def test_markers_do_not_include_generic_words():
    """标记词不能过于宽泛，否则会把正常答案误判为失败。"""
    for risky in ("无法", "没有", "不", "失败"):
        assert risky not in _NO_ANSWER_MARKERS, f"标记词「{risky}」太宽泛，会误伤正常答案"


def test_judge_fails_when_answer_is_polite_refusal():
    """回归测试：工具失败 + 礼貌拒答 + 关键词碰巧命中 → 必须判为失败。

    这正是评估里出现过的假阳性：工具序列与关键词都过关，但用户什么也没拿到。
    """
    from app.eval.runner import EvalTask, judge_task

    task = EvalTask(
        id="t-fake",
        task="实验报告需要包含哪些部分？",
        expected_tools=["search_course_kb"],
        answer_must_include=["报告"],       # 拒答文案里恰好含「报告」二字
    )
    trace = Trace(run_id="r", task=task.task, stop_reason="finished")
    trace.steps.append(
        Step(
            index=0,
            kind="tool_call",
            tool_calls=[],
        )
    )
    # 直接构造：工具调用成功（让 tool_hit 通过），但答案是拒答
    from app.schema import ToolCall, ToolResult

    call = ToolCall(name="search_course_kb", arguments={}, call_id="c1")
    trace.steps[0].tool_calls = [call]
    trace.steps.append(
        Step(index=1, kind="tool_result", tool_calls=[call],
             tool_results=[ToolResult(ok=True, content="检索失败")])
    )
    trace.final_answer = (
        "课程知识库检索工具调用失败，因此未能获取到关于实验报告要求的依据。"
        "建议你查阅课程大纲中的「实验报告」章节。"
    )

    outcome = judge_task(task, trace)
    assert outcome.completed is False, "礼貌拒答不能被判为成功"
    assert "未真正作答" in outcome.note
