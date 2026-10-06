"""中文分词：BM25 稀疏检索与评估指标都依赖它。

用 :mod:`jieba` 的搜索引擎模式（``cut_for_search``），它会把长词再切出子词，
对「信息增益」「极大似然估计」这类术语的召回更友好。

停用词表内置一份轻量清单，主要是虚词与标点；不追求覆盖全，
因为 BM25 的 IDF 本身会压低高频词的权重。
"""

from __future__ import annotations

import re
from functools import lru_cache

# 轻量中文停用词（虚词、连接词、常见疑问词）
STOPWORDS: frozenset[str] = frozenset(
    """
的 了 是 在 和 与 及 或 也 都 就 而 被 把 对 为 于 以 之 其 该 这 那 有 无 不 没
我 你 他 她 它 我们 你们 他们 什么 怎么 如何 为什么 哪些 哪个 多少 请问 解释 说明
一个 一种 一些 可以 能够 需要 应该 如果 因为 所以 但是 然后 因此 例如 比如 以及
要 会 能 很 更 最 又 再 还是 或者
a an the is are was were be been of in on at to for and or not with by as it this that
""".split()
)

_PUNCT_RE = re.compile(r"[\s\u3000!-/:-@\[-`{-~，。！？；：、“”‘’（）《》【】—…·]+")


@lru_cache(maxsize=200_000)
def _cut(text: str) -> tuple[str, ...]:
    import jieba

    # 静默 jieba 的初始化日志与并行告警
    jieba.setLogLevel(60)
    return tuple(jieba.cut_for_search(text))


def tokenize(text: str) -> list[str]:
    """分词 -> 去停用词 -> 去标点 -> 英文小写。"""
    if not text:
        return []
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens: list[str] = []
    for token in _cut(cleaned):
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if token.isdigit():
            tokens.append(token)
            continue
        if len(token) == 1 and not ("\u4e00" <= token <= "\u9fff"):
            continue          # 丢掉孤立的单个英文字母
        tokens.append(token)
    return tokens


def tokenize_joined(text: str) -> str:
    """给需要「空格分隔词串」的场合用（例如调试打印）。"""
    return " ".join(tokenize(text))


def ngrams(text: str, n: int = 2) -> list[str]:
    """字符 n-gram，作为分词之外的第二路稀疏特征（可选）。"""
    chars = re.sub(r"\s+", "", text)
    if len(chars) < n:
        return [chars] if chars else []
    return [chars[i : i + n] for i in range(len(chars) - n + 1)]
