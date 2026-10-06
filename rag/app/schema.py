"""贯穿全链路的数据契构（dataclass）。

整个项目统一使用这几个对象在模块间传递数据，好处是：
- 每个 chunk 都带 ``source`` / ``section`` / 行号等元信息，答案才能做到「可溯源」；
- 检索结果带 ``channel``（dense / sparse / hybrid / rerank）与 ``score``，
  评估脚本才能做「向量 alone vs 混合 vs 混合+重排」的消融对比。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# 解析阶段
# --------------------------------------------------------------------------- #
@dataclass
class LoadedDocument:
    """一份原始文档解析后的结果。"""

    doc_id: str
    source_path: str          # 相对 data/raw 的路径，用于展示与引用
    text: str                 # 解析出的纯文本（保留标题层级标记）
    doc_type: str             # md / txt / pdf / docx
    title: str = ""           # 文档标题（取首个一级标题或文件名）
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """切分后的知识片段——检索与引用的最小单位。"""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    # 结构化定位信息：让答案里的引用能精确到「哪一章哪一节第几段」
    section_path: str = ""    # 例如 "第3章 决策树 > 3.2 信息增益"
    heading: str = ""         # 所属最近一级标题
    start_line: int = 0
    end_line: int = 0
    chunk_index: int = 0      # 在本文档内的序号
    token_estimate: int = 0   # 粗略 token 数（中文按字符估算）
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """转成向量库能存的扁平元数据（Chroma 只接受标量）。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "section_path": self.section_path,
            "heading": self.heading,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_index": self.chunk_index,
        }

    @staticmethod
    def make_id(source_path: str, chunk_index: int, text: str) -> str:
        """稳定 ID：同一文档同一序号同一内容，重建索引后 ID 不变。"""
        raw = f"{source_path}::{chunk_index}::{text[:64]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def citation(self) -> str:
        """人类可读的引用标签，例如 ``第3章 决策树 > 3.2 信息增益``。"""
        if self.section_path:
            return self.section_path
        return self.source_path


# --------------------------------------------------------------------------- #
# 检索阶段
# --------------------------------------------------------------------------- #
RetrievalChannel = Literal["dense", "sparse", "hybrid", "rerank"]


@dataclass
class RetrievedChunk:
    """一路检索返回的候选片段。"""

    chunk: Chunk
    score: float
    channel: RetrievalChannel
    rank: int = 0                      # 该路内的名次，从 1 开始
    sub_scores: dict[str, float] = field(default_factory=dict)  # 融合时的各路分数

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class RetrievalTrace:
    """一次检索的可观测记录——面试时能拿来解释「为什么召回这条」。"""

    query: str
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    final_hits: int = 0
    rerank_used: bool = False
    rewritten_query: str = ""
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stage_ms.values()), 2)


# --------------------------------------------------------------------------- #
# 生成阶段
# --------------------------------------------------------------------------- #
@dataclass
class Citation:
    """答案中的一条引用。"""

    index: int                 # 对应正文里的 [1] [2]
    chunk_id: str
    source_path: str
    section_path: str
    snippet: str


@dataclass
class Answer:
    """一次问答的完整结果。"""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[RetrievedChunk] = field(default_factory=list)
    trace: RetrievalTrace | None = None
    refused: bool = False          # True 表示证据不足，主动拒答
    model: str = ""
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # dataclass 嵌套检索结果里的 Chunk 需要展开，方便前端直接用
        data["contexts"] = [
            {
                "chunk_id": c.chunk.chunk_id,
                "source_path": c.chunk.source_path,
                "section_path": c.chunk.section_path,
                "text": c.chunk.text,
                "score": round(c.score, 4),
                "channel": c.channel,
                "rank": c.rank,
                "sub_scores": {k: round(v, 4) for k, v in c.sub_scores.items()},
            }
            for c in self.contexts
        ]
        if self.trace:
            data["trace"] = asdict(self.trace)
        return data
