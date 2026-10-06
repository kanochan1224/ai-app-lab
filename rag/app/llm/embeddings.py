"""第 3 步：向量化（Embedding）。

统一实现 LangChain 的 :class:`~langchain_core.embeddings.Embeddings` 接口，
所以本地模型、OpenAI 兼容端点、无依赖的哈希伪向量三种后端可以互换，
上层的向量库 / 检索器完全不感知差异。

- ``local``       sentence-transformers 加载本地模型（默认 BAAI/bge-small-zh-v1.5，约 100MB）
- ``siliconflow`` / ``openai`` / ``dashscope``  走 OpenAI 兼容的 ``/embeddings``
- ``hash``        确定性哈希向量，不下载任何模型，用于离线跑通流程与单元测试

约定：所有后端返回的向量都已做 L2 归一化，因此「点积 == 余弦相似度」。
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings

# 允许用环境变量指向国内镜像，避免 huggingface.co 直连超时
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


class BaseEmbeddings(ABC):
    """向量化后端统一接口（与 LangChain Embeddings 协议一致）。"""

    provider: str = "base"
    model: str = ""
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档片段。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """向量化查询（部分模型需要加指令前缀，故单独一个方法）。"""

    # LangChain 兼容别名
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


# --------------------------------------------------------------------------- #
# 1) 本地 sentence-transformers
# --------------------------------------------------------------------------- #
class LocalSTEmbeddings(BaseEmbeddings):
    """本地模型后端（BGE / M3E / text2vec 等均可）。"""

    provider = "local"

    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self._model = None  # 懒加载：避免 import 阶段就吃内存

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        # 已缓存则强制离线，避免每次启动都去 huggingface.co 查版本（国内会超时重试）
        if _is_model_cached(self.model):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用本地向量模型。\n"
                "请执行：pip install -r requirements-ml.txt\n"
                "或把 EMBEDDING_PROVIDER 换成云端（如 siliconflow）。"
            ) from exc
        self._model = SentenceTransformer(self.model, device=self.device)
        # sentence-transformers 新版把方法改名为 get_embedding_dimension，这里两者都兼容
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(getter())
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # 归一化后点积即余弦
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        # BGE 中文系列：查询侧加指令前缀可明显提升召回
        payload = f"{self.query_instruction}{text}" if self.query_instruction else text
        return self._encode([payload])[0]


# --------------------------------------------------------------------------- #
# 2) OpenAI 兼容端点（硅基流动 / 通义 / OpenAI / 本地 vLLM 均可）
# --------------------------------------------------------------------------- #
class OpenAICompatEmbeddings(BaseEmbeddings):
    """走 ``POST {api_base}/embeddings`` 的通用客户端。

    硅基流动：``EMBEDDING_API_BASE=https://api.siliconflow.cn/v1``，模型 ``BAAI/bge-m3``
    通义：    ``EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1``
    """

    provider = "openai-compatible"

    def __init__(self, model: str, api_base: str, api_key: str, batch_size: int = 32) -> None:
        if not api_base:
            raise ValueError("云端向量化需要配置 EMBEDDING_API_BASE")
        if not api_key:
            raise ValueError("云端向量化需要配置 EMBEDDING_API_KEY")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.dim = settings.embedding.dim

    def _post(self, texts: list[str]) -> list[list[float]]:
        import httpx

        resp = httpx.post(
            f"{self.api_base}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts, "encoding_format": "float"},
            timeout=60.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        # 按 index 排序，防止服务端乱序返回导致向量与文本错位
        items = sorted(payload["data"], key=lambda d: d.get("index", 0))
        vectors = [item["embedding"] for item in items]
        if vectors:
            self.dim = len(vectors[0])
        return [_l2_normalize(v) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(texts[i : i + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]


# --------------------------------------------------------------------------- #
# 3) 哈希伪向量：零依赖，仅用于离线跑通与测试
# --------------------------------------------------------------------------- #
class HashEmbeddings(BaseEmbeddings):
    """把文本哈希到固定维度向量。

    **不具备语义能力**，仅用于：
    - 没有网络 / 没装模型时验证「解析->切分->入库->检索->生成」链路是否通畅；
    - 单元测试里避免下载模型（可复现、秒级）。
    """

    provider = "hash"

    def __init__(self, dim: int = 512, model: str = "hash-512") -> None:
        self.dim = dim
        self.model = model

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _char_ngrams(text)
        for token in tokens:
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _char_ngrams(text: str, n: int = 2) -> list[str]:
    cleaned = "".join(ch for ch in text.lower() if not ch.isspace())
    if len(cleaned) <= n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _is_model_cached(model_name: str) -> bool:
    """判断 sentence-transformers 模型是否已在本地缓存。

    只做目录检查，不依赖 huggingface_hub。命中则强制离线加载，
    避免每次启动都去 huggingface.co 查询版本（国内网络会连续超时重试）。
    """
    from pathlib import Path

    if os.path.sep in model_name or model_name.startswith("."):
        return Path(model_name).exists()
    cache_root = Path(
        os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    if os.getenv("HF_HOME"):
        cache_root = Path(os.environ["HF_HOME"]) / "hub"
    snapshots = cache_root / ("models--" + model_name.replace("/", "--")) / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        has_config = any(snap.glob("*.json"))
        has_weights = any(
            list(snap.glob("*.bin")) + list(snap.glob("*.safetensors")) + list(snap.glob("*.onnx"))
        )
        if has_config and has_weights:
            return True
    return False


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
_CACHE: dict[str, BaseEmbeddings] = {}

_CLOUD_PROVIDERS = {"openai", "siliconflow", "dashscope", "qwen", "deepseek", "openai-compatible"}


def get_embeddings(provider: str | None = None, model: str | None = None) -> BaseEmbeddings:
    """按配置返回向量化后端实例（带缓存，避免重复加载模型）。"""
    cfg = settings.embedding
    provider = (provider or cfg.provider).lower()
    model = model or cfg.model
    cache_key = f"{provider}:{model}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if provider == "local":
        backend: BaseEmbeddings = LocalSTEmbeddings(
            model_name=model,
            query_instruction=cfg.query_instruction,
            batch_size=cfg.batch_size,
        )
    elif provider in _CLOUD_PROVIDERS:
        backend = OpenAICompatEmbeddings(
            model=model,
            api_base=cfg.api_base or _default_base(provider),
            api_key=cfg.api_key,
            batch_size=cfg.batch_size,
        )
    elif provider == "hash":
        backend = HashEmbeddings(dim=cfg.dim, model=model or "hash")
    else:
        raise ValueError(
            f"未知 EMBEDDING_PROVIDER: {provider}"
            "（可选 local / siliconflow / openai / dashscope / hash）"
        )

    _CACHE[cache_key] = backend
    return backend


def _default_base(provider: str) -> str:
    return {
        "siliconflow": "https://api.siliconflow.cn/v1",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }.get(provider, "")


def embedding_info() -> dict[str, Any]:
    cfg = settings.embedding
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "api_base": cfg.api_base or _default_base(cfg.provider),
        "has_key": bool(cfg.api_key),
        "dim": cfg.dim,
    }


__all__ = [
    "BaseEmbeddings",
    "LocalSTEmbeddings",
    "OpenAICompatEmbeddings",
    "HashEmbeddings",
    "get_embeddings",
    "embedding_info",
]
