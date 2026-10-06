"""流水线测试：拒答门槛、引用对齐、无 Key 降级、可序列化。

这些测试**不访问网络、不加载模型**（EMBEDDING_PROVIDER=hash, LLM_API_KEY 为空），
因此可以在 CI 里秒级跑完。
"""

from __future__ import annotations

import json

import pytest

from app.llm.generator import (
    REFUSAL_MARKER,
    build_context,
    filter_citations,
    parse_citation_indices,
)
from app.pipeline.rag import REFUSAL_TEXT, RAGPipeline
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def pipeline(built_index) -> RAGPipeline:
    retriever = HybridRetriever(
        vector_store=built_index["store"], bm25=built_index["bm25"], reranker=None
    )
    return RAGPipeline(retriever)


# --------------------------------------------------------------------------- #
# 纯函数：上下文拼装与引用解析
# --------------------------------------------------------------------------- #
def _hit(cid: str, text: str, score: float = 0.8):
    from app.schema import Chunk, RetrievedChunk

    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=cid,
            doc_id="d",
            source_path=f"{cid}.md",
            text=text,
            section_path="第3章 > 3.2",
        ),
        score=score,
        channel="hybrid",
        rank=1,
    )


def test_build_context_numbers_sources_consistently():
    hits = [_hit("c1", "信息增益等于熵的减少量"), _hit("c2", "基尼指数不需要对数运算")]
    context, citations = build_context(hits)
    assert "[1]" in context and "[2]" in context
    assert "[1]" in context.split("[2]")[0], "编号必须按顺序出现"
    assert [c.index for c in citations] == [1, 2]
    assert citations[0].source_path == "c1.md"
    assert citations[0].section_path == "第3章 > 3.2"
    assert citations[0].chunk_id == "c1"


def test_build_context_truncates_by_budget():
    hits = [_hit(f"c{i}", "很长的一段内容" * 200) for i in range(10)]
    context, citations = build_context(hits, max_chars=500)
    assert len(citations) < 10
    assert len(context) < 3000


def test_parse_and_filter_citations():
    citations = build_context([_hit("c1", "甲"), _hit("c2", "乙"), _hit("c3", "丙")])[1]
    answer = "第一个结论成立 [1]，第二个结论也成立 [3]，重复引用 [1] 不重复出现。"
    assert parse_citation_indices(answer) == [1, 3]
    filtered = filter_citations(answer, citations)
    assert [c.index for c in filtered] == [1, 3]
    # 没有引用标注时不返回任何「装饰性引用」
    assert filter_citations("这段回答没有任何标注", citations) == []
    # 越界编号应被忽略
    assert [c.index for c in filter_citations("引用不存在的来源 [99]", citations)] == []


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def test_bm25_only_retriever_is_ready(built_index):
    """只要 BM25 有数据，检索器就应报告可用（chunk_count 取两路最大值）。"""
    retriever = HybridRetriever(vector_store=built_index["store"], bm25=BM25Index(), reranker=None)
    assert retriever.chunk_count > 0


def test_answer_returns_citations_and_trace(pipeline: RAGPipeline):
    answer = pipeline.answer("信息增益和基尼指数有什么不同", top_k=3)
    assert answer.answer
    assert not answer.refused
    assert answer.citations, "回答必须带引用来源"
    assert answer.contexts, "必须返回召回片段"
    assert answer.trace is not None
    assert answer.trace.final_hits == len(answer.contexts)
    # 引用编号必须与上下文编号体系一致
    for c in answer.citations:
        assert 1 <= c.index <= len(answer.contexts)
    assert answer.model == "extractive-fallback"


def test_answer_uses_extractive_fallback_without_api_key(pipeline: RAGPipeline):
    answer = pipeline.answer("高斯核的 gamma 参数", top_k=3)
    assert "抽取式" in answer.answer, "无 Key 时应明确标注为抽取式回答"
    assert answer.usage.get("mode") == "extractive-fallback"
    assert answer.usage.get("has_key") is False


def test_refusal_on_empty_question(pipeline: RAGPipeline):
    answer = pipeline.answer("   ")
    assert answer.refused is True
    assert answer.answer == REFUSAL_TEXT
    assert answer.citations == []
    assert answer.model == "refusal-gate"


def test_refusal_when_no_evidence(pipeline: RAGPipeline):
    """问题与语料完全无交集时不应硬答。"""
    answer = pipeline.answer("红烧肉的做法和火候掌握", top_k=3)
    if answer.refused:
        assert answer.answer == REFUSAL_TEXT
        assert answer.citations == []
    else:
        # 若抽取器找到了字面重合（例如「的」这类停用词被过滤后仍偶然重合），
        # 至少必须给出引用，不能是无来源的编造
        assert answer.citations


def test_answer_to_dict_is_json_serializable(pipeline: RAGPipeline):
    payload = pipeline.answer("预剪枝和后剪枝的区别", top_k=3).to_dict()
    text = json.dumps(payload, ensure_ascii=False)     # 不抛异常即通过
    assert "question" in payload
    assert isinstance(payload["contexts"], list)
    for ctx in payload["contexts"]:
        assert {"chunk_id", "source_path", "text", "score", "channel"} <= set(ctx)


def test_trace_records_stage_timings(pipeline: RAGPipeline):
    answer = pipeline.answer("CART 用什么指标划分", top_k=3)
    trace = answer.trace
    assert trace is not None
    assert "dense" in trace.stage_ms
    assert "sparse" in trace.stage_ms
    assert "fusion" in trace.stage_ms
    assert trace.dense_hits >= 0 and trace.sparse_hits >= 0


def test_stream_answer_emits_meta_then_delta_then_done(pipeline: RAGPipeline):
    events = list(pipeline.stream_answer("信息熵的定义是什么", top_k=3))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert "delta" in kinds
    meta = events[0]["data"]
    assert "citations" in meta and "contexts" in meta
    text = "".join(e["data"]["text"] for e in events if e["event"] == "delta")
    assert text.strip()


def test_stream_refusal_short_circuits(pipeline: RAGPipeline):
    events = list(pipeline.stream_answer("   "))
    assert events[0]["data"]["refused"] is True
    assert events[-1]["data"]["refused"] is True


def test_unknown_mode_does_not_crash(pipeline: RAGPipeline):
    # 未识别的模式会退化为「不对两路都召回」，但不应抛异常
    answer = pipeline.answer("信息增益", mode="dense", use_rerank=False)
    assert answer.answer


def test_refusal_threshold_none_disables_gate(pipeline: RAGPipeline, monkeypatch):
    """阈值留空（None）= 不启用阈值拒答。

    这是个容易写错的细节：若把「留空」解析成 0.0，配合 ``<=`` 判据
    就会把分数为 0 或负分的正常问题也拒掉。这里锁住语义。
    """
    from app.config import settings

    monkeypatch.setattr(settings.llm, "refusal_score_threshold", None)
    answer = pipeline.answer("信息增益和基尼指数有什么不同", top_k=3)
    assert answer.refused is False
    assert answer.citations


def test_refusal_threshold_gate_rejects_when_score_below(pipeline: RAGPipeline, monkeypatch):
    """阈值高于任何真实分数时，一切问题都应被检索侧闸门拦下。"""
    from app.config import settings

    monkeypatch.setattr(settings.llm, "refusal_score_threshold", 99.0)
    refused = pipeline.answer("信息增益和基尼指数有什么不同", top_k=3)
    assert refused.refused is True
    assert refused.answer == REFUSAL_TEXT
    assert refused.model == "refusal-gate"
    assert refused.usage["threshold"] == 99.0
    # 流式路径必须与一次性路径行为一致
    events = list(pipeline.stream_answer("信息增益和基尼指数有什么不同", top_k=3))
    assert events[0]["data"]["refused"] is True


def test_refusal_threshold_is_half_open_interval(pipeline: RAGPipeline, monkeypatch):
    """判据是「分数 < 阈值」：等于阈值不拒答，略高于真实分数才拒答。

    这解决了「最保守阈值」与「误拒计数」之间的矛盾——
    阈值取库内最低分时，恰好等于该分的正常问题不应被误拒。
    """
    from app.config import settings

    monkeypatch.setattr(settings.llm, "refusal_score_threshold", None)
    answer = pipeline.answer("信息增益和基尼指数有什么不同", top_k=3)
    best = pipeline.retriever.best_evidence_score(answer.contexts)
    assert best is not None, "需要拿到真实证据分才能测试边界"

    # 阈值 == 真实分数 → 不拒答（严格小于）
    monkeypatch.setattr(settings.llm, "refusal_score_threshold", best)
    assert pipeline.answer("信息增益和基尼指数有什么不同", top_k=3).refused is False
    # 阈值略高于真实分数 → 拒答
    monkeypatch.setattr(settings.llm, "refusal_score_threshold", best + 1e-6)
    assert pipeline.answer("信息增益和基尼指数有什么不同", top_k=3).refused is True


def test_extractive_answerer_marks_refusal_when_no_overlap():
    from app.llm.generator import ExtractiveAnswerer

    hits = [_hit("c1", "完全无关的另一段文字内容")]
    result = ExtractiveAnswerer().answer("量子纠缠退相干时间", hits)
    assert result.refused is True
    assert result.text == REFUSAL_MARKER


def test_extractive_mode_limitation_on_out_of_scope_but_lexically_close_question():
    """记录降级模式的已知边界（不是 bug，而是设计取舍）。

    抽取式回答器只做词汇重合判断。对于「用课程术语包装、但库里没有答案」的问题，
    它会返回片段摘要而不是拒答——生成侧的语义拒答只有在配置 LLM_API_KEY 后才会生效。
    这个测试把这个行为**固定下来**，避免以后误以为它是完整的拒答能力。
    """
    from app.llm.generator import ExtractiveAnswerer

    # 片段里恰好含有「模型」「论文」这类词，但并没有回答该问题
    hits = [_hit("c1", "本课程会介绍如何阅读一篇机器学习论文的摘要，理解模型解决的问题。")]
    result = ExtractiveAnswerer().answer("Hinton 2023 年那篇论文的结论是什么", hits)
    assert result.refused is False, "已知局限：抽取式模式下这类问题不会拒答"
    assert "[1]" in result.text

