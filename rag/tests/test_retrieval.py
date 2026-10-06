"""检索与融合测试：分词、BM25、RRF、混合检索、指标。"""

from __future__ import annotations

import pytest

from app.eval.metrics import (
    aggregate,
    dedup_docs,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_sample,
)
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from app.retrieval.tokenize import tokenize
from app.schema import Chunk, RetrievedChunk


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #
def test_tokenize_drops_stopwords_and_punctuation():
    tokens = tokenize("请问，信息增益是什么？")
    # jieba 搜索引擎模式会把「信息增益」再切出子词，因此用子串判断而非整词相等
    assert any("增益" in t for t in tokens)
    assert "请问" not in tokens
    assert "是什么" not in "".join(tokens)
    assert "，" not in tokens
    assert all(t.strip() and t not in {"的", "是", "什么"} for t in tokens)


def test_tokenize_keeps_numbers_and_technical_terms():
    tokens = tokenize("C4.5 与 ID3 的差别，k 折交叉验证")
    joined = "".join(tokens)
    assert "4" in joined or "c4" in joined
    assert any("交叉" in t or "验证" in t for t in tokens)


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


# --------------------------------------------------------------------------- #
# 构造假片段的小工具
# --------------------------------------------------------------------------- #
def _chunk(cid: str, text: str = "内容") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=f"doc-{cid}",
        source_path=f"{cid}.md",
        text=text,
        section_path="第1章 > 1.1",
        chunk_index=0,
    )


def _hit(cid: str, score: float, channel: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(cid), score=score, channel=channel, rank=rank)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
def test_bm25_ranks_relevant_chunk_first(built_index):
    bm25: BM25Index = built_index["bm25"]
    hits = bm25.search("基尼指数是怎么定义的", top_k=5)
    assert hits, "BM25 没有召回任何片段"
    assert hits[0].channel == "sparse"
    assert hits[0].rank == 1
    # 分数必须单调不增
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert any("基尼" in h.chunk.text for h in hits)


def test_bm25_exact_term_beats_generic_term(built_index):
    """正是混合检索存在的理由：精确术语匹配能力强于语义泛化。"""
    bm25: BM25Index = built_index["bm25"]
    hits = bm25.search("高斯核 gamma 参数", top_k=3)
    assert hits
    assert "高斯核" in hits[0].chunk.text or "gamma" in hits[0].chunk.text.lower()


def test_bm25_empty_query_and_empty_index():
    empty = BM25Index()
    assert empty.search("任意问题") == []
    empty.build([_chunk("a", "信息熵的定义")])
    assert empty.search("") == []
    assert empty.search("!!!") == []


def test_bm25_persistence_roundtrip(built_index, tmp_root):
    bm25: BM25Index = built_index["bm25"]
    path = tmp_root / "bm25_roundtrip.json"
    bm25.save(path)

    restored = BM25Index()
    assert restored.load(path) is True
    assert restored.size == bm25.size
    original = [h.chunk_id for h in bm25.search("信息增益", top_k=3)]
    reloaded = [h.chunk_id for h in restored.search("信息增益", top_k=3)]
    assert original == reloaded


def test_bm25_load_missing_file_returns_false(tmp_root):
    assert BM25Index().load(tmp_root / "not-exists.json") is False


def test_ensure_loaded_fills_empty_index_from_cache(built_index, tmp_root, monkeypatch):
    """回归测试：HybridRetriever 直接构造时稀疏一路不能是空的。"""
    from app.config import settings

    cache = tmp_root / "bm25_ensure.json"
    built_index["bm25"].save(cache)
    monkeypatch.setattr(settings, "bm25_cache", cache)

    fresh = BM25Index()
    assert fresh.size == 0
    assert fresh.ensure_loaded() is True
    assert fresh.size == built_index["bm25"].size

    retriever = HybridRetriever(
        vector_store=built_index["store"], bm25=BM25Index(), reranker=None
    )
    hits, trace = retriever.search("信息增益", mode="sparse", top_k=3, use_rerank=False)
    assert hits, "稀疏检索不能静默失效"
    assert trace.sparse_hits > 0


def test_ensure_loaded_rebuilds_from_vector_store(tmp_root, monkeypatch):
    """缓存缺失时应能从向量库导出片段后重建。"""
    from app.config import settings
    from app.retrieval.vector_store import VectorStore

    monkeypatch.setattr(settings, "bm25_cache", tmp_root / "definitely-missing.json")
    store = VectorStore(
        persist_dir=settings.chroma_dir, collection=settings.index_collection
    )
    if store.count == 0:
        pytest.skip("向量库为空，跳过重建路径测试")
    fresh = BM25Index()
    assert fresh.ensure_loaded(store) is True
    assert fresh.size == store.count


# --------------------------------------------------------------------------- #
# RRF 融合
# --------------------------------------------------------------------------- #
def test_rrf_favours_document_found_by_both_channels():
    dense = [_hit("a", 0.9, "dense", 1), _hit("b", 0.8, "dense", 2)]
    sparse = [_hit("b", 12.0, "sparse", 1), _hit("c", 8.0, "sparse", 2)]
    fused = reciprocal_rank_fusion([(dense, 1.0), (sparse, 1.0)], k=60, top_k=10)
    ids = [h.chunk_id for h in fused]
    assert ids[0] == "b", "同时被两路召回的文档应当排第一"
    assert set(ids) == {"a", "b", "c"}
    assert all(h.channel == "hybrid" for h in fused)
    assert [h.rank for h in fused] == [1, 2, 3]
    # 融合分数必须保留各路明细，便于解释与调试
    top = fused[0]
    assert "dense_rank" in top.sub_scores and "sparse_rank" in top.sub_scores


def test_rrf_respects_channel_weights():
    dense = [_hit("a", 0.9, "dense", 1)]
    sparse = [_hit("b", 9.0, "sparse", 1)]
    dense_heavy = reciprocal_rank_fusion([(dense, 1.0), (sparse, 0.01)], k=60, top_k=5)
    assert dense_heavy[0].chunk_id == "a"
    sparse_heavy = reciprocal_rank_fusion([(dense, 0.01), (sparse, 1.0)], k=60, top_k=5)
    assert sparse_heavy[0].chunk_id == "b"


def test_rrf_ignores_zero_weight_and_empty_lists():
    dense = [_hit("a", 0.9, "dense", 1)]
    fused = reciprocal_rank_fusion([(dense, 0.0), ([], 1.0)], k=60, top_k=5)
    assert fused == []


def test_weighted_fusion_normalizes_scale_difference():
    # BM25 分数（数十）与向量分数（0~1）量纲不同，加权融合应先归一化
    dense = [_hit("a", 0.95, "dense", 1), _hit("b", 0.90, "dense", 2)]
    sparse = [_hit("c", 50.0, "sparse", 1), _hit("d", 20.0, "sparse", 2)]
    fused = weighted_fusion([(dense, 1.0), (sparse, 1.0)], top_k=10)
    assert len(fused) == 4
    scores = {h.chunk_id: h.score for h in fused}
    # 两路各自的冠军得分应当接近（都归一到 1.0 再相加）
    assert abs(scores["a"] - scores["c"]) < 0.2


# --------------------------------------------------------------------------- #
# 混合检索
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_modes(built_index):
    retriever = HybridRetriever(
        vector_store=built_index["store"], bm25=built_index["bm25"], reranker=None
    )
    # reranker=None 时 _reranker_resolved 为 True，避免测试触发模型下载
    for mode in ["dense", "sparse", "hybrid"]:
        hits, trace = retriever.search("信息增益与基尼指数的区别", mode=mode, top_k=3,
                                       use_rerank=False)
        assert hits, f"{mode} 模式没有召回结果"
        assert len(hits) <= 3
        assert trace.query
        assert trace.final_hits == len(hits)
        assert trace.total_ms >= 0


def test_hybrid_union_not_worse_than_single_channel(built_index):
    """混合检索的召回不应差于任一路：它是两路的并集再融合。"""
    retriever = HybridRetriever(
        vector_store=built_index["store"], bm25=built_index["bm25"], reranker=None
    )
    query = "支持向量机的 C 参数和 gamma 参数"
    dense_ids = {h.chunk_id for h in retriever.search(query, mode="dense", top_k=10,
                                                      use_rerank=False)[0]}
    sparse_ids = {h.chunk_id for h in retriever.search(query, mode="sparse", top_k=10,
                                                       use_rerank=False)[0]}
    hybrid_ids = {h.chunk_id for h in retriever.search(query, mode="hybrid", top_k=10,
                                                       use_rerank=False)[0]}
    assert dense_ids | sparse_ids == hybrid_ids


def test_hybrid_retriever_empty_query(built_index):
    retriever = HybridRetriever(
        vector_store=built_index["store"], bm25=built_index["bm25"], reranker=None
    )
    hits, trace = retriever.search("   ", mode="hybrid", use_rerank=False)
    assert hits == []
    assert trace.final_hits == 0


def test_best_evidence_score_prefers_rerank_then_dense():
    reranked = RetrievedChunk(
        chunk=_chunk("a"), score=3.2, channel="rerank", rank=1,
        sub_scores={"rerank_raw": 3.2, "dense_score": 0.7},
    )
    dense_only = RetrievedChunk(
        chunk=_chunk("b"), score=0.66, channel="dense", rank=1
    )
    sparse_only = RetrievedChunk(
        chunk=_chunk("c"), score=11.0, channel="sparse", rank=1
    )
    retriever = HybridRetriever(vector_store=None, bm25=BM25Index(), reranker=None)
    assert retriever.best_evidence_score([reranked]) == pytest.approx(3.2)
    assert retriever.best_evidence_score([dense_only]) == pytest.approx(0.66)
    # 纯 BM25 分数不可比，返回 None 表示不做阈值拒答
    assert retriever.best_evidence_score([sparse_only]) is None
    assert retriever.best_evidence_score([]) is None


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def test_dedup_docs_keeps_first_occurrence():
    assert dedup_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_basic_metric_values():
    docs = ["x.md", "a.md", "b.md", "c.md"]
    relevant = {"a.md", "b.md"}
    assert hit_rate_at_k(docs, relevant, 3) == 1.0
    assert hit_rate_at_k(docs, relevant, 1) == 0.0
    assert recall_at_k(docs, relevant, 4) == 1.0
    assert recall_at_k(docs, relevant, 2) == pytest.approx(0.5)
    assert precision_at_k(docs, relevant, 4) == pytest.approx(0.5)
    assert reciprocal_rank(docs, relevant) == pytest.approx(0.5)   # 第一个命中在第 2 名
    assert reciprocal_rank(["x.md"], relevant) == 0.0
    assert 0.0 < ndcg_at_k(docs, relevant, 4) <= 1.0


def test_ndcg_perfect_ranking_is_one():
    docs = ["a.md", "b.md", "c.md"]
    assert ndcg_at_k(docs, {"a.md", "b.md"}, 3) == pytest.approx(1.0)


def test_score_sample_shape_and_no_hit():
    scores = score_sample(["a.md", "b.md"], {"a.md"}, ks=(1, 3))
    assert set(scores) >= {"hit@1", "hit@3", "recall@1", "recall@3", "mrr", "first_hit_rank"}
    assert scores["hit@1"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["first_hit_rank"] == 1.0

    miss = score_sample(["x.md", "y.md"], {"a.md"}, ks=(1, 3))
    assert miss["hit@3"] == 0.0
    assert miss["mrr"] == 0.0
    assert miss["first_hit_rank"] == 0.0


def test_aggregate_averages_rows():
    rows = [{"hit@5": 1.0, "mrr": 0.5}, {"hit@5": 0.0, "mrr": 0.1}]
    agg = aggregate(rows)
    assert agg["hit@5"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.3)
    assert aggregate([]) == {}
