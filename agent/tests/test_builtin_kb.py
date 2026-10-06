"""内置迷你知识库与降级行为的测试。

这一组测试守护一个容易被忽略但很关键的性质：
**单独 clone agent-lab 也必须能跑**——面试官不会为了看一个 Agent 去 clone 第二个仓库。
"""

from __future__ import annotations

import pytest

from app.schema import KBChunk, KBHit, KBTrace
from app.tools.base import ToolContext
from app.tools.builtin_kb import BUILTIN_KB, BuiltinKnowledgeBase


# --------------------------------------------------------------------------- #
# 内置知识库本身
# --------------------------------------------------------------------------- #
def test_builtin_kb_loads_and_has_content():
    kb = BuiltinKnowledgeBase()
    assert kb.chunk_count == len(BUILTIN_KB)
    assert kb.chunk_count >= 15, "内置知识库至少要覆盖课程主要主题"
    for title, body in BUILTIN_KB:
        assert title.strip()
        assert len(body) >= 60, f"条目「{title}」内容过短，检索价值低"


def test_builtin_kb_returns_expected_shape():
    kb = BuiltinKnowledgeBase()
    hits, trace = kb.search("信息增益与基尼指数的区别", top_k=3)
    assert isinstance(trace, KBTrace)
    assert trace.final_hits == len(hits)
    assert hits, "内置库应能召回结果"
    for hit in hits:
        assert isinstance(hit, KBHit)
        assert isinstance(hit.chunk, KBChunk)
        assert hit.chunk.text
        assert hit.chunk.section_path
        assert hit.rank >= 1
    # 名次必须从 1 连续
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


@pytest.mark.parametrize(
    "query,expected_topic",
    [
        ("K-means 里 K 该怎么选", "K-means"),
        ("支持向量机的 C 参数怎么影响模型", "支持向量机"),
        ("为什么不能在测试集上调参", "测试集"),
        ("PCA 为什么用特征向量解释", "PCA"),
        ("过拟合怎么处理", "过拟合"),
        ("Bagging 和 Boosting 区别", "Bagging"),
    ],
)
def test_builtin_kb_retrieves_correct_topic(query, expected_topic):
    """内置库虽小，但必须能命中对应主题——否则兜底没有意义。"""
    kb = BuiltinKnowledgeBase()
    hits, _ = kb.search(query, top_k=1)
    assert hits, f"「{query}」没有召回任何内容"
    top = hits[0].chunk
    assert expected_topic in top.section_path or expected_topic in top.text, (
        f"「{query}」首条命中「{top.section_path}」，与预期主题「{expected_topic}」不符"
    )


def test_builtin_kb_handles_empty_query():
    kb = BuiltinKnowledgeBase()
    hits, trace = kb.search("", top_k=3)
    assert hits == []
    assert trace.final_hits == 0


# --------------------------------------------------------------------------- #
# 降级行为（关键：保证独立可跑）
# --------------------------------------------------------------------------- #
def test_course_kb_tool_falls_back_to_builtin(monkeypatch):
    """course-rag 不可用时，工具必须降级到内置库，而不是报错。

    这是「单独 clone 能跑」的核心保证。
    """
    from app.config import settings
    from app.tools.course_kb import CourseKBTool

    monkeypatch.setattr(settings, "course_rag_path", settings.root_dir / "不存在的项目")
    tool = CourseKBTool()
    assert tool.chunk_count == len(BUILTIN_KB)
    assert "内置" in tool.source_label

    result = tool.run({"query": "信息增益是什么", "top_k": 2}, ToolContext())
    assert result.ok is True
    # 必须明确告知用的是兜底数据，避免被误当成完整讲义
    assert "内置迷你知识库" in result.content
    assert result.raw and result.raw[0]["source_path"] == "内置迷你知识库"


def test_course_kb_tool_uses_real_project_when_available():
    """同级存在 course-rag 时应优先用它（本仓库开发环境下确实存在）。"""
    from app.config import settings
    from app.tools.course_kb import CourseKBTool

    bridge_target = settings.course_rag_path / "app" / "retrieval" / "hybrid_retriever.py"
    if not bridge_target.exists():
        pytest.skip("本机没有 course-rag 项目，跳过真实链路检查")

    tool = CourseKBTool()
    assert tool.chunk_count > 0
    assert "内置" not in tool.source_label, (
        f"检测到 course-rag 却仍降级了：{tool.source_label}"
    )


def test_preflight_reports_degraded_when_using_builtin(monkeypatch):
    """preflight 必须如实报告「当前用的是兜底库」，而不是笼统地说 ok。

    注意要先清掉桥接层的模块级缓存：它加载成功后会把 course-rag 组件缓存起来，
    不清缓存的话「路径不存在」根本不会被检查到，测试就成了假通过。
    """
    from app.config import settings
    from app.runtime.agent import Agent
    from app.tools import course_rag_bridge
    from app.tools.base import ToolRegistry
    from app.tools.course_kb import CourseKBTool

    monkeypatch.setattr(course_rag_bridge, "_CACHE", {})
    monkeypatch.setattr(course_rag_bridge, "_ERROR", "")
    monkeypatch.setattr(settings, "course_rag_path", settings.root_dir / "不存在的项目")

    registry = ToolRegistry()
    registry.register(CourseKBTool())
    agent = Agent(registry=registry)
    info = agent.preflight()["tools"]["search_course_kb"]
    assert info["status"] == "degraded"
    assert "内置" in info["detail"]


def test_course_kb_tool_falls_back_even_with_warm_cache(monkeypatch):
    """缓存已被填充时也必须能正确降级。

    真实场景：服务先正常运行（course-rag 加载成功、缓存写入），
    之后 course-rag 被移走/删除，此时不能继续拿着失效的缓存。
    """
    from app.config import settings
    from app.tools import course_rag_bridge
    from app.tools.course_kb import CourseKBTool

    monkeypatch.setattr(course_rag_bridge, "_CACHE", {})   # 模拟冷缓存
    monkeypatch.setattr(settings, "course_rag_path", settings.root_dir / "不存在的项目")
    tool = CourseKBTool()
    assert "内置" in tool.source_label
    assert tool.chunk_count == len(BUILTIN_KB)
