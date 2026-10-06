"""评估用的检索指标：Precision@k / Recall@k / HitRate@k / MRR / nDCG@k。

为什么这些指标要自己写而不是用现成库：
本项目关心的是「文档级命中」——只要召回片段所属文档在标注的相关文档集合里就算命中。
这和通用 IR 里「一个 query 对应一组相关文档 id」的设定一致，
但需要按我们的 chunk→source_path 映射来判定，自己实现反而最短。
"""

from __future__ import annotations

import math


def _hits_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> list[bool]:
    return [doc in relevant for doc in ranked_docs[:k]]


def hit_rate_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """top-k 里是否至少命中一个相关文档（0/1）。"""
    return 1.0 if any(_hits_at_k(ranked_docs, relevant, k)) else 0.0


def recall_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """top-k 覆盖了多少比例的相关文档。"""
    if not relevant:
        return 0.0
    found = {d for d in ranked_docs[:k] if d in relevant}
    return len(found) / len(relevant)


def precision_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    top = ranked_docs[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def reciprocal_rank(ranked_docs: list[str], relevant: set[str]) -> float:
    """第一个相关文档的名次的倒数；没命中记 0。"""
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """二值相关性的 nDCG@k。"""
    gains = [1.0 if doc in relevant else 0.0 for doc in ranked_docs[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def dedup_docs(ranked_docs: list[str]) -> list[str]:
    """把「片段级别的排序」压缩成「文档级别的排序」（去重但保序）。

    检索返回的是 chunk，评估关心的是文档命中，所以先按名次遍历、
    每个文档只取它最好的那次名次。
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc in ranked_docs:
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """对多条样本的指标求均值。"""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 4) for k in keys}


def score_sample(
    ranked_docs: list[str],
    relevant: set[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """单条样本的完整指标（文档级去重后计算）。"""
    docs = dedup_docs(ranked_docs)
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_rate_at_k(docs, relevant, k)
        out[f"recall@{k}"] = recall_at_k(docs, relevant, k)
        out[f"precision@{k}"] = precision_at_k(docs, relevant, k)
        out[f"ndcg@{k}"] = round(ndcg_at_k(docs, relevant, k), 4)
    out["mrr"] = round(reciprocal_rank(docs, relevant), 4)
    out["first_hit_rank"] = float(
        next((i for i, d in enumerate(docs, start=1) if d in relevant), 0)
    )
    return out
