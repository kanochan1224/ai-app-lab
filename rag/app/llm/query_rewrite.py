"""查询改写：把口语化提问扩写成更贴近教材用语的检索式。

**为什么需要它**：学生用口语提问，语料用教材用语书写，两者字面与语义都不重合。
实测案例：问「K-means 里 K 该怎么选？」时，
- 教材里的章节叫「8.3 如何选择簇数 K」，子节是「肘部法」「轮廓系数法」「Gap Statistic」；
- 稠密检索把该章节排到第 17 名，BM25 排第 14 名——**根本进不了最终上下文**；
- 模型拿不到任何选 K 的方法，于是合理地拒答。

**这不是检索器坏了，而是查询与文档的表述不对齐**，属于 RAG 的典型失败模式之一。
改写这一步用一次很便宜的 LLM 调用，把「K 该怎么选」扩写成
「K-means 聚类 如何选择簇数 K 肘部法 轮廓系数 Gap Statistic」，
让两路召回都够得着。

**失败要能降级**：改写调用失败或超时就直接用原问题，绝不能让检索因此失败。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """你是检索查询改写助手。把学生的口语化问题改写成更适合检索课程教材的查询式。

要求：
1. 补全教材里可能出现的**标准术语**（例如「K 该怎么选」→「如何选择簇数 K 肘部法 轮廓系数」）。
2. 保留原问题的核心意图，不要改变问题的指向。
3. 只输出改写后的查询，不要解释、不要引号、不要换行。
4. 长度控制在 60 字以内，用空格分隔关键词。

示例：
问：那个剪枝是干啥的
答：决策树 剪枝 预剪枝 后剪枝 过拟合

问：为啥训练集准测试集不准
答：过拟合 泛化误差 训练误差 验证集

问：K 怎么定
答：K-means 如何选择簇数 K 肘部法 轮廓系数 Gap Statistic"""


def rewrite_query(question: str, client=None) -> str:
    """把问题改写成教材式查询；失败或未配置模型时返回原问题。"""
    question = (question or "").strip()
    if not question:
        return question

    if client is None:
        from .generator import get_chat_client

        client = get_chat_client()
    if client is None:
        return question

    try:
        result = client.complete(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": f"问：{question}\n答："},
            ],
            retries=1,
        )
    except Exception as exc:      # 改写失败不能拖垮检索
        logger.warning("查询改写失败，回退原问题：%s", exc)
        return question

    rewritten = _clean(result.text)
    if not rewritten or len(rewritten) > 120:
        return question
    return rewritten


def _clean(text: str) -> str:
    """清理模型输出：去掉引号、换行、前缀说明。"""
    text = (text or "").strip()
    text = re.sub(r"^(改写|查询|答案|答)\s*[:：]\s*", "", text)
    text = text.strip().strip("\"'“”‘’")
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    return text.strip()


def merge_query(original: str, rewritten: str) -> str:
    """把原问题与改写结果拼成最终检索式。

    保留原问题在前：它携带学生真实意图，改写只是补充术语。
    这样即使改写质量一般，也不会把检索带偏。
    """
    original = (original or "").strip()
    rewritten = (rewritten or "").strip()
    if not rewritten or rewritten == original:
        return original
    return f"{original} {rewritten}"
