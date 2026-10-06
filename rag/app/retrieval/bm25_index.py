"""稀疏检索：BM25（中文分词后建索引）。

**为什么必须再加一路稀疏检索？**
纯向量检索对「专有名词、缩写、公式符号、数字」很不敏感——
学生问「C4.5 和 ID3 的区别」「K 折交叉验证里 K 一般取多少」，
向量模型容易召回语义相近但术语不对的段落。
BM25 对精确词项匹配敏感，两者互补，这正是混合检索的价值所在。

索引用 :mod:`rank_bm25` 的 Okapi BM25，索引体量在课程知识库（几千片段）级别下
毫秒级响应，无需上 Elasticsearch。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import settings
from app.retrieval.tokenize import tokenize
from app.schema import Chunk, RetrievedChunk


class BM25Index:
    """内存 BM25 索引，支持持久化缓存（避免每次启动都重新分词）。"""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None

    # ------------------------------------------------------------------ #
    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        t0 = time.perf_counter()
        self._tokenized = [tokenize(c.text) for c in self.chunks]
        # 全空的语料会让 BM25 除零，兜一个占位词
        safe = [toks or ["<empty>"] for toks in self._tokenized]
        self._bm25 = BM25Okapi(safe) if safe else None
        self._build_ms = round((time.perf_counter() - t0) * 1000, 1)

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def build_ms(self) -> float:
        return getattr(self, "_build_ms", 0.0)

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        if self._bm25 is None or not self.chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # 取分数最高的 top_k 个（argsort 降序）
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits: list[RetrievedChunk] = []
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue          # 完全不匹配的候选不进候选池，避免污染 RRF
            hits.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=round(score, 6),
                    channel="sparse",
                    rank=rank,
                )
            )
        return hits

    # ------------------------------------------------------------------ #
    def save(self, path: Path | None = None) -> Path:
        """持久化到 JSON（片段文本 + 元数据），下次启动直接加载。"""
        path = Path(path or settings.bm25_cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(self.chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source_path": c.source_path,
                    "text": c.text,
                    "section_path": c.section_path,
                    "heading": c.heading,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "chunk_index": c.chunk_index,
                    "token_estimate": c.token_estimate,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, path: Path | None = None) -> bool:
        """从缓存加载并重建索引；文件不存在或损坏时返回 ``False``。"""
        path = Path(path or settings.bm25_cache)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunks = [
                Chunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item.get("doc_id", ""),
                    source_path=item.get("source_path", ""),
                    text=item.get("text", ""),
                    section_path=item.get("section_path", ""),
                    heading=item.get("heading", ""),
                    start_line=item.get("start_line", 0),
                    end_line=item.get("end_line", 0),
                    chunk_index=item.get("chunk_index", 0),
                    token_estimate=item.get("token_estimate", 0),
                    metadata=item.get("metadata", {}),
                )
                for item in payload.get("chunks", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
        if not chunks:
            return False
        self.build(chunks)
        return True

    def ensure_loaded(self, vector_store=None) -> bool:
        """确保索引可用：内存为空时先读缓存，缓存缺失则从向量库重建。

        这是为了让 CLI / API 直接 ``HybridRetriever()`` 就能工作，
        而不是每个入口都自己记得加载缓存（曾因此出现稀疏检索静默失效）。
        """
        if self.size > 0:
            return True
        if self.load():
            return True
        if vector_store is not None:
            chunks = vector_store.get_all_chunks()
            if chunks:
                self.build(chunks)
                self.save()
                return True
        return False
