"""评估 CLI：跑消融实验与生成质量评估。

用法::

    python -m scripts.evaluate --retrieval-only          # 不需要 API Key
    python -m scripts.evaluate                           # 检索 + 生成（需 Key）
    python -m scripts.evaluate --limit 10 --tag quick    # 只跑 10 条，打标签

输出：
- 控制台打印对比表；
- ``data/eval/reports/report-<时间>.json`` 与 ``.md``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR, settings                           # noqa: E402
from app.eval.runner import (                                      # noqa: E402
    ABLATION_MODES,
    load_dataset,
    run_generation_eval,
    run_retrieval_eval,
    save_reports,
)
from app.pipeline.rag import RAGPipeline                           # noqa: E402
from app.retrieval.bm25_index import BM25Index                     # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever         # noqa: E402
from app.retrieval.vector_store import VectorStore                 # noqa: E402


def _pick_dataset(granularity: str) -> Path:
    """按粒度选择评估集；片段级评估集存在时 auto 优先用它。"""
    doc_set = EVAL_DIR / "eval_set.jsonl"
    sec_set = EVAL_DIR / "eval_set_chunk.jsonl"
    if granularity == "section":
        return sec_set
    if granularity == "auto" and sec_set.exists():
        return sec_set
    return doc_set


def main() -> None:
    parser = argparse.ArgumentParser(description="CourseRAG 评估")
    parser.add_argument("--dataset", type=str, default=None, help="评估集路径（JSONL）")
    parser.add_argument(
        "--granularity",
        choices=["auto", "document", "section"],
        default="auto",
        help=(
            "判定粒度。document=同一文档即算命中；"
            "section=必须命中同一章节（更严格，排除只复述问题的大纲小节）；"
            "auto=存在片段级评估集时优先用它"
        ),
    )
    parser.add_argument("--retrieval-only", action="store_true", help="只跑检索指标，不需要 Key")
    parser.add_argument("--limit", type=int, default=None, help="限制样本条数（快速验证）")
    parser.add_argument("--top-k", type=int, default=10, help="检索评估的 top-k")
    parser.add_argument("--no-rerank", action="store_true", help="跳过重排（省时间）")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="启用查询改写（需 API Key，每个问题多一次模型调用）",
    )
    parser.add_argument("--tag", type=str, default="", help="报告文件名标签")
    args = parser.parse_args()

    settings.ensure_dirs()
    if args.rewrite:
        if not settings.llm.api_key:
            raise SystemExit("查询改写需要 API Key，请先在 .env 配置 LLM_API_KEY")
        settings.retrieval.enable_query_rewrite = True
    print(f"查询改写：{'开启' if settings.retrieval.enable_query_rewrite else '关闭'}")
    dataset_path = Path(args.dataset) if args.dataset else _pick_dataset(args.granularity)
    samples = load_dataset(dataset_path, base_path=EVAL_DIR / "eval_set.jsonl")
    if args.limit:
        samples = samples[: args.limit]
    print(f"评估集：{dataset_path.name} · {len(samples)} 条样本 · 判定粒度：{samples[0].granularity}")
    print()

    # ---- 组装检索器（复用已建好的索引）----
    vector_store = VectorStore()
    bm25 = BM25Index()
    if not bm25.load(settings.bm25_cache):
        chunks = vector_store.get_all_chunks()
        if not chunks:
            raise SystemExit("索引为空，请先执行：python -m scripts.build_index")
        bm25.build(chunks)
        bm25.save(settings.bm25_cache)
    print(f"索引规模：向量 {vector_store.count} 条 / BM25 {bm25.size} 条")
    retriever = HybridRetriever(vector_store=vector_store, bm25=bm25)

    modes = ABLATION_MODES
    if args.no_rerank:
        modes = [m for m in modes if not m[2]]

    print("\n【一】检索消融实验")
    retrieval_reports = run_retrieval_eval(retriever, samples, modes=modes, top_k=args.top_k)

    generation = None
    if not args.retrieval_only:
        if not settings.llm.api_key:
            print(
                "\n【二】跳过生成质量评估：未配置 LLM_API_KEY。\n"
                "     配置后重跑本命令即可获得忠实度/相关性/引用正确性打分。"
            )
        else:
            print(f"\n【二】端到端生成评估（{len(samples)} 条，裁判模型 {settings.llm.model}）")
            pipeline = RAGPipeline(retriever)
            generation = run_generation_eval(pipeline, samples, top_k=5)

    extra = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "chunk_cache": str(settings.chunk_cache),
        "bm25_cache": str(settings.bm25_cache),
        "chroma_dir": str(settings.chroma_dir),
    }
    json_path, md_path = save_reports(
        retrieval_reports, generation, len(samples), extra=extra, tag=args.tag
    )
    print(f"\nJSON 报告：{json_path}")
    print(f"Markdown 报告：{md_path}")

    print("\n检索指标对比：")
    keys = ["hit@1", "hit@3", "hit@5", "mrr", "recall@5", "ndcg@5"]
    print(f"{'方案':<30}" + "".join(f"{k:>10}" for k in keys) + f"{'延迟(ms)':>12}")
    for r in retrieval_reports:
        row = "".join(f"{r.metrics.get(k, 0):>10.3f}" for k in keys)
        print(f"{r.mode_label:<30}{row}{r.avg_latency_ms:>12.0f}")


if __name__ == "__main__":
    main()
