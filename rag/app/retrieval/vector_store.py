"""稠密检索：Chroma 向量库封装。

为什么用 `chromadb.PersistentClient` 而不是 `langchain_chroma.Chroma`：
本项目的「混合检索」需要自己控制两路召回与融合，向量库只承担「存 + 最近邻」两件事。
直接用原生客户端，省掉一层可能带来版本耦合的适配器，检索链路更透明。

（项目仍然使用 LangChain 生态：``langchain-core`` 的 Embeddings 协议、
``langchain-text-splitters`` 用于切分侧校验；见 ``docs/architecture.md`` 的说明。）
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.config import settings
from app.llm.embeddings import BaseEmbeddings, get_embeddings
from app.schema import Chunk, RetrievedChunk


class VectorStore:
    """Chroma 持久化向量库。"""

    def __init__(
        self,
        embeddings: BaseEmbeddings | None = None,
        persist_dir: Path | None = None,
        collection: str | None = None,
    ) -> None:
        self.embeddings = embeddings or get_embeddings()
        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection or settings.index_collection

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

    # ------------------------------------------------------------------ #
    def _collection(self, create: bool = False):
        if create:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        try:
            return self._client.get_collection(self.collection_name)
        except Exception as exc:  # chromadb 版本间异常类型不一致，统一转成友好提示
            raise RuntimeError(
                f"向量集合 '{self.collection_name}' 不存在，请先执行建索引："
                "python -m scripts.build_index"
            ) from exc

    @property
    def count(self) -> int:
        try:
            return self._collection().count()
        except RuntimeError:
            return 0

    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """批量写入。向量化是按 batch 调的，避免一次性把显存/内存打满。"""
        if not chunks:
            return 0
        collection = self._collection(create=True)
        written = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self.embeddings.embed_documents([c.text for c in batch])
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
            written += len(batch)
        return written

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """向量最近邻检索，返回按相似度降序的候选。"""
        if self.count == 0:
            return []
        query_vec = self.embeddings.embed_query(query)
        result = self._collection().query(
            query_embeddings=[query_vec],
            n_results=min(top_k, max(self.count, 1)),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[RetrievedChunk] = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            # hnsw:space=cosine 时 distance = 1 - cosine_similarity
            score = 1.0 - float(dist)
            if score < settings.retrieval.min_dense_score:
                continue
            hits.append(
                RetrievedChunk(
                    chunk=_meta_to_chunk(cid, doc, meta or {}),
                    score=round(score, 6),
                    channel="dense",
                    rank=rank,
                )
            )
        return hits

    def get_all_chunks(self) -> list[Chunk]:
        """导出全部片段（重建 BM25 索引、跑评估时用）。"""
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []
        data = collection.get(include=["documents", "metadatas"], limit=total)
        return [
            _meta_to_chunk(cid, doc, meta or {})
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def reset(self) -> None:
        """清空集合（重建索引时调用）。"""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass

    def drop_all(self) -> None:
        """彻底删除持久化目录，用于「从零重建」。"""
        self._client = None  # type: ignore[assignment]
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir, ignore_errors=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)


def _meta_to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=str(meta.get("chunk_id", chunk_id)),
        doc_id=str(meta.get("doc_id", "")),
        source_path=str(meta.get("source_path", "")),
        text=text or "",
        section_path=str(meta.get("section_path", "")),
        heading=str(meta.get("heading", "")),
        start_line=int(meta.get("start_line", 0) or 0),
        end_line=int(meta.get("end_line", 0) or 0),
        chunk_index=int(meta.get("chunk_index", 0) or 0),
        token_estimate=max(len(text or "") // 2, 1),
        metadata={"doc_title": meta.get("doc_title", ""), "doc_type": meta.get("doc_type", "")},
    )
