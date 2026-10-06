"""拒答阈值标定：用评估集观察 rerank 分数分布，据此设 ``REFUSAL_SCORE_THRESHOLD``。

为什么需要它：
重排模型的输出是未归一化的 logits，不同模型、不同语料的尺度都不一样。
凭感觉拍一个阈值，要么把正常问题全拒掉（阈值过高），要么完全不起作用（阈值过低）。
所以先用标注数据看分布，再选一个能分开「该答」与「该拒」的分界点。

用法::

    python -m scripts.calibrate                 # 用默认评估集
    python -m scripts.calibrate --limit 20
    python -m scripts.calibrate --mode hybrid   # 对比不重排时的分数尺度
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                    # noqa: E402
from app.eval.runner import load_dataset                           # noqa: E402
from app.retrieval.bm25_index import BM25Index                     # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever         # noqa: E402
from app.retrieval.vector_store import VectorStore                 # noqa: E402


def _describe(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name:<22} 无数据")
        return
    values = sorted(values)
    n = len(values)
    print(
        f"  {name:<22} n={n:<4} min={values[0]:>8.3f} "
        f"p25={values[n // 4]:>8.3f} 中位={statistics.median(values):>8.3f} "
        f"p75={values[3 * n // 4]:>8.3f} max={values[-1]:>8.3f}"
    )


def _best_threshold(in_scope: list[float], out_scope: list[float]) -> float:
    """在候选阈值里挑「错误总数最少」的那个。

    拒答判据统一为 ``分数 < 阈值``（与 :mod:`app.pipeline.rag` 一致），因此：

    - 误拒（false positive）：库内问题分数低于阈值被错误拒答——用户最反感；
    - 漏放（false negative）：越界问题分数不低于阈值而未被拒答——幻觉风险。

    这里把两者视为同等代价取最小错误数；若业务更怕误拒，应把阈值往低处调。
    """
    candidates = sorted({round(v, 4) for v in in_scope + out_scope})
    best_t, best_err = candidates[0], None
    for t in candidates:
        err = sum(1 for v in in_scope if v < t) + sum(1 for v in out_scope if v >= t)
        if best_err is None or err < best_err:
            best_t, best_err = t, err
    # 若还存在「比所有越界分数都大」的阈值且它不误拒任何库内问题，
    # 那它严格更优（能拒掉全部越界问题），取其中最小的一个
    strict = [t for t in candidates if t > max(out_scope) and t <= min(in_scope)]
    if strict and min(strict) < best_t:
        best_t = min(strict)
    return best_t


def _threshold_variants(in_scope: list[float], out_scope: list[float]) -> list[tuple[float, str]]:
    """给出三档候选阈值：最保守 / 错误最少 / 最激进。

    判据是 ``分数 < 阈值`` 才拒答，所以「最保守」应取下界（min(in_scope)），
    这样分数等于该值的库内问题**不会**被误拒；「最激进」要取
    「大于越界最高分的最小候选值」，才能把那一条也拒掉。
    """
    candidates = sorted({round(v, 4) for v in in_scope + out_scope})
    hi = min(in_scope)
    lo_candidates = [t for t in candidates if t > max(out_scope)]
    lo = min(lo_candidates) if lo_candidates else max(candidates) + 1e-6
    best = _best_threshold(in_scope, out_scope)
    seen: list[tuple[float, str]] = []
    for t, note in [
        (hi, "最保守：不误拒任何库内问题"),
        (best, "错误总数最少"),
        (lo, "最激进：拒掉全部越界问题"),
    ]:
        if not any(abs(t - s) < 1e-9 for s, _ in seen):
            seen.append((t, note))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="标定拒答阈值")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", default="hybrid_rerank",
                        choices=["dense", "sparse", "hybrid", "hybrid_rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    samples = load_dataset(Path(args.dataset) if args.dataset else None)
    if args.limit:
        samples = samples[: args.limit]

    vector_store = VectorStore()
    bm25 = BM25Index()
    retriever = HybridRetriever(vector_store=vector_store, bm25=bm25)

    in_scope: list[float] = []      # 知识库里有答案的问题 → 分数应偏高
    out_scope: list[float] = []     # 越界问题 → 分数应偏低，用来定阈值
    per_question: list[tuple[str, float, bool]] = []

    for sample in samples:
        hits, _ = retriever.search(sample.question, mode=args.mode, top_k=args.top_k)
        score = retriever.best_evidence_score(hits)
        if score is None:
            continue
        (out_scope if sample.should_refuse else in_scope).append(score)
        per_question.append((sample.question, score, sample.should_refuse))

    print(f"\n检索模式：{args.mode}   样本：{len(samples)} 条")
    print("=" * 78)
    _describe("知识库内问题", in_scope)
    _describe("越界问题（应拒答）", out_scope)

    if in_scope and out_scope:
        lo, hi = max(out_scope), min(in_scope)      # 越界最高分 / 库内最低分
        n_in, n_out = len(in_scope), len(out_scope)   # 库内样本数 / 越界样本数
        # 判据与 app.pipeline.rag 保持一致：分数 < 阈值 才拒答
        def fp_count(t: float) -> int:
            return sum(1 for v in in_scope if v < t)

        def fn_count(t: float) -> int:
            return sum(1 for v in out_scope if v >= t)

        print("-" * 78)
        print("候选阈值对比（判据：分数 < 阈值 → 判为「知识库无依据」并拒答）")
        print(f"  {'阈值':>8}  {'误拒库内':>11}  {'漏放越界':>11}   说明")
        for t, note in _threshold_variants(in_scope, out_scope):
            print(f"  {t:>8.3f}  {fp_count(t):>8}/{n_in:<3} {fn_count(t):>8}/{n_out:<3}   {note}")

        best = _best_threshold(in_scope, out_scope)
        print("-" * 78)
        if hi > lo:
            print(f"两类分数可分：越界最高 {lo:.3f} < 库内最低 {hi:.3f}")
            print(f"建议 REFUSAL_SCORE_THRESHOLD = {(lo + hi) / 2:.3f}（两者中点）")
        else:
            print("两类分数存在重叠，单靠阈值**无法完全分开**：")
            print(f"  越界问题最高分 {lo:.3f} ≥ 库内问题最低分 {hi:.3f}")
            print()
            print("可选策略（按保守程度排序）：")
            print(f"  A. 阈值 = {hi:.3f}（= 库内最低分）：不误拒任何库内问题，"
                  f"漏放 {fn_count(hi)}/{n_out} 条越界问题")
            print(f"  B. 阈值 = {best:.3f}：误拒 {fp_count(best)}/{n_in} 条库内问题，"
                  f"漏放 {fn_count(best)}/{n_out} 条越界问题")
            aggressive = max(t for t, _ in _threshold_variants(in_scope, out_scope))
            if abs(aggressive - best) < 1e-9:
                print(f"  C. 与 B 相同（{aggressive:.3f} 已是能拒掉全部越界问题的最小阈值），"
                      f"没有更激进的独立档位。")
            else:
                print(f"  C. 阈值 = {aggressive:.3f}（最激进）：不放过任何越界问题，"
                      f"误拒 {fp_count(aggressive)}/{n_in} 条库内问题")
            print()
            print(f"  ⚠ 本评估集只有 {n_out} 条越界样本，阈值统计并不稳健。")
            print("     更可靠的做法是分层拒答：")
            print("       · 检索侧阈值只拦明显无关（分数很低）的问题；")
            print("       · 语义拒答交给提示词要求的 NO_ANSWER 标记（需配置 LLM_API_KEY）；")
            print("       · 若业务不容忍误拒，宁可把阈值设低甚至留空，完全交给模型判断。")
    else:
        print("-" * 78)
        print("缺少其中一类的数据，无法给出阈值建议。")
        print("请确认评估集里包含 should_refuse=true 的越界样本。")

    print("\n逐题明细（越低越可疑）：")
    for question, score, should_refuse in sorted(per_question, key=lambda x: x[1]):
        tag = "越界" if should_refuse else "库内"
        print(f"  [{tag}] {score:>8.3f}  {question[:52]}")

    print(
        "\n把选定的值写进 .env：\n"
        "  REFUSAL_SCORE_THRESHOLD=<上面的建议值>\n"
        "留空则不做阈值拒答，完全依赖模型判断（默认行为）。"
    )


if __name__ == "__main__":
    main()
