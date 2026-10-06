"""建索引 CLI：解析 → 切分 → 向量化 → 落库。

用法::

    python -m scripts.build_index                    # 用 .env 里的配置
    python -m scripts.build_index --strategy fixed   # 换切分策略做对比
    python -m scripts.build_index --chunk-size 800 --overlap 100
    python -m scripts.build_index --no-vectors       # 只重建 BM25（快速验证）

执行完会打印：文档数、片段数、切分统计、耗时，以及最终的索引规模。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许 `python -m scripts.build_index` 与 `python scripts/build_index.py` 两种调用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROCESSED_DIR, settings          # noqa: E402
from app.ingest.chunker import chunk_documents, chunk_stats  # noqa: E402
from app.ingest.loader import load_documents, load_summary   # noqa: E402
from app.retrieval.bm25_index import BM25Index          # noqa: E402
from app.retrieval.vector_store import VectorStore      # noqa: E402


def build(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    rebuild_vectors: bool = True,
    raw_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    settings.ensure_dirs()
    started = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)

    # ---- 1) 解析 ----
    log("=" * 72)
    log("第 1 步 · 文档解析")
    docs = load_documents(raw_dir)
    if not docs:
        raise SystemExit(
            f"未在 {raw_dir or settings.root_dir / 'data' / 'raw'} 找到任何受支持的文档"
            "（支持 .md / .txt / .pdf / .docx）"
        )
    log(load_summary(docs))

    # ---- 2) 切分 ----
    log("\n第 2 步 · 文本切分")
    chunks = chunk_documents(
        docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    stats = chunk_stats(chunks)
    log(
        f"策略={strategy or settings.chunk.strategy} "
        f"size={chunk_size or settings.chunk.chunk_size} "
        f"overlap={settings.chunk.chunk_overlap if chunk_overlap is None else chunk_overlap}"
    )
    log(json.dumps(stats, ensure_ascii=False, indent=2))

    # 落盘片段（便于对比不同切分策略、也方便调试引用）
    settings.chunk_cache.parent.mkdir(parents=True, exist_ok=True)
    with settings.chunk_cache.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    log(f"片段已写入 {settings.chunk_cache}")

    # ---- 3) 向量化入库 ----
    result = {
        "documents": len(docs),
        "chunks": len(chunks),
        "stats": stats,
        "vectors": 0,
        "bm25": 0,
    }
    if rebuild_vectors:
        log("\n第 3 步 · 向量化并写入向量库")
        log(f"后端：{settings.embedding.provider}:{settings.embedding.model}")
        store = VectorStore()
        store.reset()
        t0 = time.perf_counter()
        written = store.add_chunks(chunks)
        result["vectors"] = written
        log(f"写入 {written} 条向量，耗时 {time.perf_counter() - t0:.1f}s")
    else:
        log("\n第 3 步 · 跳过向量化（--no-vectors）")

    # ---- 4) BM25 ----
    log("\n第 4 步 · 构建 BM25 稀疏索引")
    bm25 = BM25Index()
    t0 = time.perf_counter()
    bm25.build(chunks)
    bm25.save(settings.bm25_cache)
    result["bm25"] = bm25.size
    log(f"BM25 索引 {bm25.size} 条，分词+建索引耗时 {time.perf_counter() - t0:.1f}s")
    log(f"缓存：{settings.bm25_cache}")

    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    log("\n" + "=" * 72)
    log(
        f"完成：{result['documents']} 篇文档 → {result['chunks']} 个片段 → "
        f"{result['vectors']} 条向量 / {result['bm25']} 条 BM25，"
        f"总耗时 {result['elapsed_s']}s"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="构建课程知识库索引")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"], default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--no-vectors", action="store_true", help="只重建 BM25 索引")
    parser.add_argument("--raw-dir", type=str, default=None)
    args = parser.parse_args()

    build(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        rebuild_vectors=not args.no_vectors,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
    )


if __name__ == "__main__":
    main()
