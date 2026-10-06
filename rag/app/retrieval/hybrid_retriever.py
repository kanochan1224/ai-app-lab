"""混合检索与融合：整个系统的核心。

链路（默认配置）::

    问题
     ├── 稠密检索  vector_store.search  → top 20   （语义相似，怕术语不精确）
     └── 稀疏检索  bm25.search          → top 20   （词项精确，怕同义改写）
                    ↓  RRF 加权融合（只用名次，不用原始分，天然免疫量纲差异）
                 融合候选池 top 20
                    ↓  cross-encoder 精排
                 最终 top 5  →  交给生成层

**为什么用 RRF（Reciprocal Rank Fusion）而不是加权求和分数？**
向量相似度（0~1）与 BM25 分数（0~数十）量纲完全不同，直接相加需要调归一化，
而且不同 query 的分数分布还不稳定。RRF 只用「名次」：``score = w / (k + rank)``，
k 取 60，无需标定即可稳定工作，是工业界混合检索的常见默认方案。

同时保留 ``score`` 层级的加权融合模式（``fusion_mode="weighted"``）便于对比实验。
"""

from __future__ import annotations

import time
from typing import Literal

from app.config import settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore
from app.schema import RetrievalTrace, RetrievedChunk

Mode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
FusionMode = Literal["rrf", "weighted"]


# --------------------------------------------------------------------------- #
# 融合
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    result_lists: list[tuple[list[RetrievedChunk], float]],
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """加权 RRF：``score(d) = Σ_i w_i / (k + rank_i(d))``。"""
    fused_scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    detail: dict[str, dict[str, float]] = {}

    for hits, weight in result_lists:
        if weight <= 0:
            continue
        for hit in hits:
            cid = hit.chunk_id
            contribution = weight / (k + hit.rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + contribution
            detail.setdefault(cid, {})[f"{hit.channel}_score"] = hit.score
            detail[cid][f"{hit.channel}_rank"] = float(hit.rank)
            # 保留信息量更大的那个 chunk 对象（同 id 内容一致，取先到的即可）
            best_chunk.setdefault(cid, hit)

    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out: list[RetrievedChunk] = []
    for rank, (cid, score) in enumerate(ordered, start=1):
        base = best_chunk[cid]
        out.append(
            RetrievedChunk(
                chunk=base.chunk,
                score=round(score, 6),
                channel="hybrid",
                rank=rank,
                sub_scores=detail.get(cid, {}),
            )
        )
    return out


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted_fusion(
    result_lists: list[tuple[list[RetrievedChunk], float]],
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """归一化后加权求和（对比用）。"""
    agg: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    detail: dict[str, dict[str, float]] = {}
    for hits, weight in result_lists:
        if not hits or weight <= 0:
            continue
        normed = _minmax([h.score for h in hits])
        for hit, norm in zip(hits, normed):
            cid = hit.chunk_id
            agg[cid] = agg.get(cid, 0.0) + weight * norm
            detail.setdefault(cid, {})[f"{hit.channel}_score"] = hit.score
            best.setdefault(cid, hit)
    ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk=best[cid].chunk,
            score=round(score, 6),
            channel="hybrid",
            rank=rank,
            sub_scores=detail.get(cid, {}),
        )
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """统一检索入口，支持 4 种模式，便于做消融实验。"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        reranker=None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25 or BM25Index()
        self._reranker = reranker
        self._reranker_resolved = reranker is not None
        # 稀疏索引为空时自动加载缓存 / 从向量库重建，避免「静默只剩一路召回」
        self.bm25.ensure_loaded(self.vector_store)

    @property
    def reranker(self):
        if not self._reranker_resolved:
            self._reranker = get_reranker()
            self._reranker_resolved = True
        return self._reranker

    # ------------------------------------------------------------------ #
    @property
    def chunk_count(self) -> int:
        return self.bm25.size or self.vector_store.count

    def search(
        self,
        query: str,
        mode: Mode | None = None,
        top_k: int | None = None,
        fusion_mode: FusionMode | None = None,
        use_rerank: bool | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """执行检索，返回 (结果, 可观测链路)。

        ``fusion_mode`` 留空则用配置里的默认值（当前为 rrf，
        依据是 README 中的融合策略对照实验）。
        ``ENABLE_QUERY_REWRITE=true`` 时先用一次 LLM 调用把口语化提问
        扩写成教材式查询（补全术语），否则直接用原问题检索。
        """
        cfg = settings.retrieval
        rcfg = settings.rerank
        fusion_mode = fusion_mode or cfg.fusion_mode      # type: ignore[assignment]
        query = (query or "").strip()
        trace = RetrievalTrace(query=query)
        if not query:
            return [], trace

        # ---- 0) 查询改写（可选）----
        # 学生用口语提问、教材用术语书写，两者不重合时召回会整段漏掉，
        # 典型如「K 该怎么选」匹配不到「如何选择簇数 K / 肘部法 / 轮廓系数法」。
        if cfg.enable_query_rewrite:
            from app.llm.query_rewrite import merge_query, rewrite_query

            t0 = time.perf_counter()
            rewritten = rewrite_query(query)
            merged = merge_query(query, rewritten)
            trace.stage_ms["rewrite"] = round((time.perf_counter() - t0) * 1000, 2)
            trace.rewritten_query = rewritten
            query = merged

        mode = self._resolve_mode(mode)
        final_k = top_k or cfg.final_top_k

        # ---- 1) 两路召回 ----
        dense_hits: list[RetrievedChunk] = []
        sparse_hits: list[RetrievedChunk] = []

        if mode in {"dense", "hybrid", "hybrid_rerank"}:
            t0 = time.perf_counter()
            dense_hits = self.vector_store.search(query, top_k=cfg.dense_top_k)
            trace.stage_ms["dense"] = round((time.perf_counter() - t0) * 1000, 2)
        if mode in {"sparse", "hybrid", "hybrid_rerank"}:
            t0 = time.perf_counter()
            sparse_hits = self.bm25.search(query, top_k=cfg.sparse_top_k)
            trace.stage_ms["sparse"] = round((time.perf_counter() - t0) * 1000, 2)

        trace.dense_hits = len(dense_hits)
        trace.sparse_hits = len(sparse_hits)

        # ---- 2) 融合 ----
        if mode == "dense":
            fused = dense_hits[: max(final_k, rcfg.candidate_pool if self._needs_rerank(mode, use_rerank) else final_k)]
        elif mode == "sparse":
            fused = sparse_hits[: max(final_k, rcfg.candidate_pool if self._needs_rerank(mode, use_rerank) else final_k)]
        else:
            t0 = time.perf_counter()
            lists = [(dense_hits, cfg.dense_weight), (sparse_hits, cfg.sparse_weight)]
            pool = max(final_k, rcfg.candidate_pool)
            if fusion_mode == "weighted":
                fused = weighted_fusion(lists, top_k=pool)
            else:
                fused = reciprocal_rank_fusion(lists, k=cfg.rrf_k, top_k=pool)
            trace.stage_ms["fusion"] = round((time.perf_counter() - t0) * 1000, 2)
            for rank, hit in enumerate(fused, start=1):
                hit.rank = rank

        trace.fused_hits = len(fused)

        # ---- 3) 精排 ----
        if self._needs_rerank(mode, use_rerank) and fused:
            reranker = self.reranker
            if reranker is not None:
                t0 = time.perf_counter()
                final = reranker.rerank(query, fused[: rcfg.candidate_pool], top_n=final_k)
                trace.stage_ms["rerank"] = round((time.perf_counter() - t0) * 1000, 2)
                trace.rerank_used = True
            else:
                final = fused[:final_k]
        else:
            final = fused[:final_k]

        trace.final_hits = len(final)
        return final, trace

    # ------------------------------------------------------------------ #
    def _resolve_mode(self, mode: Mode | None) -> Mode:
        if mode:
            return mode
        if not settings.retrieval.enable_hybrid:
            return "dense"
        return "hybrid_rerank" if (settings.rerank.enabled and self.reranker is not None) else "hybrid"

    def _needs_rerank(self, mode: Mode, use_rerank: bool | None) -> bool:
        if use_rerank is not None:
            return use_rerank
        return mode == "hybrid_rerank"

    def best_evidence_score(self, hits: list[RetrievedChunk]) -> float | None:
        """取「最可信的那一路分数」，用于拒答判断。

        优先重排分（cross-encoder 判相关性最准），其次稠密相似度；
        两者都没有（例如纯 BM25 模式）时返回 ``None``，表示不做阈值拒答。
        """
        if not hits:
            return None
        top = hits[0]
        if "rerank_raw" in top.sub_scores and top.sub_scores["rerank_raw"] != 0.0:
            return top.sub_scores["rerank_raw"]
        if "dense_score" in top.sub_scores:
            return top.sub_scores["dense_score"]
        if top.channel == "dense":
            return top.score
        return None
