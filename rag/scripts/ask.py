"""问答 CLI：命令行里直接问课程知识库。

用法::

    python -m scripts.ask "信息增益和基尼指数有什么区别"
    python -m scripts.ask "K-means 怎么选 K" --mode dense --show-context
    python -m scripts.ask "SVM 的 C 参数怎么影响模型" --top-k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                        # noqa: E402
from app.pipeline.rag import RAGPipeline               # noqa: E402


def _print_answer(answer) -> None:
    print("\n" + "=" * 72)
    print(f"问题：{answer.question}")
    print("=" * 72)
    print(answer.answer)
    if answer.citations:
        print("\n--- 引用来源 ---")
        for c in answer.citations:
            loc = c.section_path or c.source_path
            print(f"[{c.index}] {c.source_path} > {loc}")
    if answer.trace:
        t = answer.trace
        print(
            f"\n--- 检索链路 --- 模式={t.final_hits} 条 / "
            f"dense={t.dense_hits} sparse={t.sparse_hits} fused={t.fused_hits} "
            f"rerank={'是' if t.rerank_used else '否'} "
            f"耗时={t.total_ms}ms {t.stage_ms}"
        )
    print(
        f"模型={answer.model} 拒答={answer.refused} "
        f"端到端={answer.latency_ms}ms"
    )


def _print_contexts(answer) -> None:
    print("\n--- 召回的片段（重排后顺序）---")
    for i, hit in enumerate(answer.contexts, start=1):
        head = f"[{i}] {hit.chunk.source_path}"
        if hit.chunk.section_path:
            head += f" > {hit.chunk.section_path}"
        print(f"\n{head}\n  score={hit.score:.4f} channel={hit.channel} "
              f"rank={hit.rank} sub={hit.sub_scores}")
        print("  " + hit.chunk.text[:300].replace("\n", "\n  "))


def main() -> None:
    parser = argparse.ArgumentParser(description="向课程知识库提问")
    parser.add_argument("question", nargs="+", help="问题文本")
    parser.add_argument("--mode", default=None,
                        choices=["dense", "sparse", "hybrid", "hybrid_rerank"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--show-context", action="store_true", help="打印召回片段全文")
    args = parser.parse_args()

    question = " ".join(args.question)
    pipeline = RAGPipeline()
    if not pipeline.is_ready():
        raise SystemExit("索引为空，请先执行：python -m scripts.build_index")

    answer = pipeline.answer(
        question,
        top_k=args.top_k,
        mode=args.mode,
        use_rerank=False if args.no_rerank else None,
    )
    _print_answer(answer)
    if args.show_context:
        _print_contexts(answer)

    if not settings.llm.api_key:
        print(
            "\n提示：当前未配置 LLM_API_KEY，回答由抽取式降级生成。"
            "在 .env 中填入密钥即可获得生成式回答。"
        )


if __name__ == "__main__":
    main()
