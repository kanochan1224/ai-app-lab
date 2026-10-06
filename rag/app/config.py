"""项目级配置：全部通过环境变量 / .env 覆盖，代码里不出现任何密钥。

设计要点：**多后端可插拔**。向量化与生成这两处最容易绑死供应商，
所以都抽成 provider 字段，切换只改 .env，不动代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件位于 <root>/app/config.py）
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
DOCS_DIR = ROOT_DIR / "docs"

load_dotenv(ROOT_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_optional_float(key: str) -> float | None:
    """留空返回 ``None``（表示「不启用该阈值」），而不是 0.0。

    这个区分很重要：拒答阈值留空意味着「不做阈值拒答」，
    若退化成 0.0，语义就悄悄变成「分数 ≤ 0 才拒答」——看似等价，
    但一旦调整比较方式就会埋雷。
    """
    raw = _env(key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass
class ChunkSettings:
    """切分策略参数——消融实验里主要调的就是这几个值。"""

    strategy: str = field(default_factory=lambda: _env("CHUNK_STRATEGY", "recursive"))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    # semantic 策略专用：相邻句向量相似度低于该阈值就切开
    semantic_threshold: float = field(
        default_factory=lambda: _env_float("SEMANTIC_THRESHOLD", 0.62)
    )


@dataclass
class EmbeddingSettings:
    """向量化后端。

    provider:
      - ``local``      sentence-transformers 本地模型，无需 Key
      - ``siliconflow`` 硅基流动（BAAI/bge-m3 等，OpenAI 兼容）
      - ``openai``      OpenAI 官方或任意兼容端点
      - ``hash``        无依赖的确定性哈希伪向量，仅用于跑通流程/离线测试
    """

    provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "local"))
    model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
    api_base: str = field(default_factory=lambda: _env("EMBEDDING_API_BASE", ""))
    api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY", ""))
    batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))
    dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 512))
    # 查询侧指令前缀：BGE 中文系列官方建议给「查询」加前缀，能明显提召回
    query_instruction: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_QUERY_INSTRUCTION",
            "为这个句子生成表示以用于检索相关文章：",
        )
    )


@dataclass
class RerankSettings:
    """重排（cross-encoder）参数。"""

    enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", True))
    provider: str = field(default_factory=lambda: _env("RERANK_PROVIDER", "local"))
    model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "maidalun1020/bce-reranker-base_v1")
    )
    api_base: str = field(default_factory=lambda: _env("RERANK_API_BASE", ""))
    api_key: str = field(default_factory=lambda: _env("RERANK_API_KEY", ""))
    top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 5))
    # 融合后先截断到这么多候选再送去重排，控制延迟（CPU 上这是主要成本）
    candidate_pool: int = field(default_factory=lambda: _env_int("RERANK_CANDIDATE_POOL", 10))
    # 送进 cross-encoder 的文本上限：评分主要看片段开头，截断能大幅降低耗时
    # （实测 CPU 上 20 对 × 492 字要 3.6s，截到 320 字可降到约 1s）
    max_chars: int = field(default_factory=lambda: _env_int("RERANK_MAX_CHARS", 320))
    batch_size: int = field(default_factory=lambda: _env_int("RERANK_BATCH_SIZE", 8))


@dataclass
class RetrievalSettings:
    """检索参数：双路召回 + 加权融合。"""

    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 20))
    sparse_top_k: int = field(default_factory=lambda: _env_int("SPARSE_TOP_K", 20))
    fused_top_k: int = field(default_factory=lambda: _env_int("FUSED_TOP_K", 20))
    final_top_k: int = field(default_factory=lambda: _env_int("FINAL_TOP_K", 5))
    # 融合方式：rrf（只用名次，默认）或 weighted（分数 min-max 归一化后加权）
    #
    # 默认选 RRF，是实测结论（两个评估集、四种指标都更优）：
    #   RRF      Hit@1 0.429/0.543  MRR 0.649/0.762  nDCG@5 0.619/0.820
    #   weighted Hit@1 0.400/0.457  MRR 0.642/0.724  nDCG@5 0.618/0.794
    #
    # 为什么加权反而更差：min-max 归一化会把**每一路的最高分都拉成 1.0**，
    # 于是「在弱的一路里碰巧排第一」的无关片段拿到满分，
    # 而「在强的一路排第 3」的真答案反被压低（实测有查询的正确答案因此从第 7 掉到第 18）。
    # 详见 README「融合策略对照实验」。
    fusion_mode: str = field(default_factory=lambda: _env("FUSION_MODE", "rrf"))
    # RRF 平滑常数，论文推荐 60（fusion_mode=rrf 时生效）
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 1.0))
    sparse_weight: float = field(default_factory=lambda: _env_float("SPARSE_WEIGHT", 1.0))
    # 低于该相似度的候选直接丢弃（防「强行凑上下文」导致幻觉）
    min_dense_score: float = field(default_factory=lambda: _env_float("MIN_DENSE_SCORE", 0.0))
    enable_query_rewrite: bool = field(
        default_factory=lambda: _env_bool("ENABLE_QUERY_REWRITE", False)
    )
    enable_hybrid: bool = field(default_factory=lambda: _env_bool("ENABLE_HYBRID", True))


@dataclass
class LLMSettings:
    """生成后端。默认 deepseek，开箱即用的中文性价比之选。"""

    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "deepseek"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    api_base: str = field(default_factory=lambda: _env("LLM_API_BASE", "https://api.deepseek.com/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "") or _env("DEEPSEEK_API_KEY", ""))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1200))
    timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 60))
    # 无 Key 时的降级模式：只做「抽取式回答」，保证链路仍可演示
    allow_extractive_fallback: bool = field(
        default_factory=lambda: _env_bool("ALLOW_EXTRACTIVE_FALLBACK", True)
    )
    # 证据不足时的拒答阈值（重排分 ≤ 该值即拒答）。留空 = 不启用阈值拒答，
    # 完全依赖模型判断；具体取值用 scripts/calibrate.py 在评估集上标定。
    refusal_score_threshold: float | None = field(
        default_factory=lambda: _env_optional_float("REFUSAL_SCORE_THRESHOLD")
    )
    strict_citation: bool = field(default_factory=lambda: _env_bool("STRICT_CITATION", True))


@dataclass
class Settings:
    project_name: str = "CourseRAG"
    knowledge_base_name: str = field(
        default_factory=lambda: _env("KB_NAME", "《机器学习导论》课程知识库")
    )
    chunk: ChunkSettings = field(default_factory=ChunkSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    index_collection: str = field(default_factory=lambda: _env("INDEX_COLLECTION", "course_kb"))
    # 允许用环境变量覆盖索引落盘位置（容器里常挂载到别的路径）
    chroma_dir_override: str = field(default_factory=lambda: _env("CHROMA_DIR", ""))
    bm25_cache: Path = field(default_factory=lambda: INDEX_DIR / "bm25_cache.json")
    chunk_cache: Path = field(default_factory=lambda: PROCESSED_DIR / "chunks.jsonl")

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @property
    def chroma_dir(self) -> Path:
        if self.chroma_dir_override:
            p = Path(self.chroma_dir_override)
            return p if p.is_absolute() else ROOT_DIR / p
        return INDEX_DIR / "chroma"

    def ensure_dirs(self) -> None:
        for p in (RAW_DIR, PROCESSED_DIR, EVAL_DIR, INDEX_DIR, DOCS_DIR, self.chroma_dir):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, str]:
        """给 /health 与 CLI 用的配置快照（不含密钥）。"""
        return {
            "knowledge_base": self.knowledge_base_name,
            "chunk_strategy": f"{self.chunk.strategy} (size={self.chunk.chunk_size}, overlap={self.chunk.chunk_overlap})",
            "embedding": f"{self.embedding.provider}:{self.embedding.model}",
            "rerank": f"{self.rerank.provider}:{self.rerank.model}" if self.rerank.enabled else "disabled",
            "llm": f"{self.llm.provider}:{self.llm.model}" + ("" if self.llm.api_key else " (no-key: 抽取式降级)"),
            "retrieval": (
                f"hybrid={self.retrieval.enable_hybrid} "
                f"dense_top_k={self.retrieval.dense_top_k} "
                f"sparse_top_k={self.retrieval.sparse_top_k} "
                f"final_top_k={self.retrieval.final_top_k}"
            ),
        }


settings = Settings()
