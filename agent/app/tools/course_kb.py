"""课程知识库检索工具 —— **优先复用 course-rag 项目的检索器**。

这是两个项目之间的连接点，也是「系统」而非「散件」的体现：
Agent 不重新实现一套检索，而是把已有的「混合检索 + 重排」直接当成一个工具。

**同时保证单独 clone 也能跑**：若同级不存在 course-rag 项目，
自动降级到内置迷你知识库（见 ``builtin_kb.py``），
接口完全一致，上层不感知差异——面试官 clone 单个仓库即可看到 Agent 完整工作。

返回内容里带**章节级引用**（``第 3 章 … > 3.2.4 …``），
因此 Agent 的最终答案同样可以做到可溯源——这一点在评估里会被单独检查。
"""

from __future__ import annotations

import logging
from typing import Any

from ..schema import ToolResult
from .base import Tool, ToolContext, ToolError
from .course_rag_bridge import CourseRAGUnavailable, build_retriever

logger = logging.getLogger(__name__)


class CourseKBTool(Tool):
    name = "search_course_kb"
    description = (
        "检索《机器学习导论》课程知识库（讲义、实验手册、FAQ），"
        "返回带章节出处的原文片段。"
        "凡是与机器学习课程概念、公式、实验要求、作业与考试政策相关的问题，"
        "都应优先用它，并在回答里标注出处。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索问题，用课程里的术语表述效果更好"},
            "top_k": {"type": "integer", "description": "返回片段数，默认 4，最大 8"},
        },
        "required": ["query"],
    }
    returns = "若干课程原文片段，每条含章节出处与相关度分数。"

    def __init__(self) -> None:
        self._retriever = None
        self._load_error = ""
        self._source = ""

    def _ensure(self):
        """获取检索器：优先 course-rag，缺失时降级到内置迷你知识库。

        **降级而不是报错**，是为了让单独 clone 本项目的人也能立刻跑通——
        面试官不会为了看一个 Agent 再去 clone 第二个仓库。
        两条路径返回同一种结构，上层完全不感知差异。
        """
        if self._retriever is not None:
            return self._retriever
        try:
            self._retriever = build_retriever()
            self._source = "course-rag（完整混合检索 + 重排）"
        except CourseRAGUnavailable as exc:
            # 外部项目不可用 → 用内置知识库兜底，并记下原因供 preflight 展示
            self._load_error = str(exc)
            from .builtin_kb import BuiltinKnowledgeBase

            self._retriever = BuiltinKnowledgeBase()
            self._source = "内置迷你知识库（未检测到 course-rag 项目，已自动降级）"
            logger.info("course-rag 不可用，已降级到内置知识库：%s", str(exc).splitlines()[0])
        except Exception as exc:  # 索引损坏等
            self._load_error = f"初始化课程检索器失败：{type(exc).__name__}: {exc}"
            raise ToolError(self._load_error) from exc
        return self._retriever

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("检索问题为空")
        try:
            top_k = int(arguments.get("top_k") or 4)
        except (TypeError, ValueError):
            top_k = 4
        top_k = max(1, min(top_k, 8))

        retriever = self._ensure()
        if retriever.chunk_count == 0:
            return ToolResult(
                ok=False,
                content=(
                    "课程知识库为空（索引未构建）。\n"
                    "请在 course-rag 项目中执行：python -m scripts.build_index"
                ),
                error="empty index",
            )

        hits, trace = retriever.search(query, top_k=top_k)
        if not hits:
            return ToolResult(
                ok=False,
                content="课程知识库中未检索到相关片段。建议换个说法，或改用 web_search 查询课程外内容。",
                error="no hits",
            )

        # 来源标注：用内置库时明确告知，避免把兜底数据当成完整讲义
        header = f"课程知识库检索「{query}」→ {len(hits)} 个片段"
        if self._source and "内置" in self._source:
            header += "（当前使用内置迷你知识库，未接入完整课程语料）"
        lines = [header + "：", ""]
        for i, hit in enumerate(hits, start=1):
            loc = hit.chunk.section_path or hit.chunk.source_path
            lines.append(f"[{i}] 《{hit.chunk.source_path}》 > {loc}（相关度 {hit.score:.3f}）")
            lines.append(hit.chunk.text.strip())
            lines.append("")
        return ToolResult(
            ok=True,
            content="\n".join(lines).strip(),
            raw=[
                {
                    "source_path": h.chunk.source_path,
                    "section_path": h.chunk.section_path,
                    "chunk_id": h.chunk.chunk_id,
                    "score": round(h.score, 4),
                }
                for h in hits
            ],
        )

    @property
    def source_label(self) -> str:
        """当前用的是 course-rag 还是内置库（供 preflight 展示）。"""
        try:
            self._ensure()
        except Exception:
            pass
        return self._source or "未初始化"

    @property
    def chunk_count(self) -> int:
        try:
            return self._ensure().chunk_count
        except Exception:
            return 0
