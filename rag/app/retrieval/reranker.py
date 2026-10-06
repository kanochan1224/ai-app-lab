"""重排（Rerank）：cross-encoder 精排。

**为什么召回之后还要重排？**
双路召回是「粗排」，用向量点积 / BM25 分数排序，追求的是「别漏」；
cross-encoder 把 (问题, 片段) 拼在一起过一遍模型，能建模细粒度交互，
排序质量明显更好，但计算量大——所以只对候选池（默认 20 条）做，成本可控。

前端展示与答案生成都只用重排后的 top_n 条，这就是 RAG 里最典型的一段
「召回宽、精排窄」的工程权衡。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.schema import RetrievedChunk

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


def _is_model_cached(model_name: str) -> bool:
    """判断模型是否已在本地缓存，用于决定是否强制离线加载。

    只做轻量的目录检查，不引入 huggingface_hub 的可选依赖。
    """
    if os.path.sep in model_name or model_name.startswith("."):
        return Path(model_name).exists()
    cache_root = Path(
        os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    if os.getenv("HF_HOME"):
        cache_root = Path(os.environ["HF_HOME"]) / "hub"
    folder = "models--" + model_name.replace("/", "--")
    snapshots = cache_root / folder / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        # 至少要能看到权重或配置，才算「缓存可用」
        if any(snap.glob("*.json")) and any(
            list(snap.glob("*.bin")) + list(snap.glob("*.safetensors")) + list(snap.glob("*.onnx"))
        ):
            return True
    return False


class BaseReranker:
    """重排后端统一接口。"""

    provider = "base"
    model = ""

    def rerank(self, query: str, hits: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 本地 cross-encoder
# --------------------------------------------------------------------------- #
class LocalCrossEncoderReranker(BaseReranker):
    """sentence-transformers 的 CrossEncoder（默认 BCE-reranker-base_v1，中英双语）。"""

    provider = "local"

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_chars: int = 320,
        batch_size: int = 8,
    ) -> None:
        self.model = model_name
        self.device = device or os.getenv("RERANK_DEVICE", "cpu")
        self.max_chars = max_chars
        self.batch_size = batch_size
        self._model = None          # 懒加载
        self.available = True

    def _ensure_model(self):
        if self._model is not None or not self.available:
            return self._model
        os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        # 关键：模型已缓存时必须离线加载。
        # 否则每次构造 CrossEncoder 都会去 huggingface.co 查询 commit 版本，
        # 在国内网络下会连续重试超时（实测让每次重排多花 3.7~17 秒）。
        if _is_model_cached(self.model):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            self.available = False
            return None
        try:
            self._model = CrossEncoder(self.model, device=self.device)
        except Exception:
            # 模型下载失败（离线 / 断网）不该让整个问答失败，退化为不重排
            self.available = False
            self._model = None
        return self._model

    def rerank(self, query: str, hits: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        model = self._ensure_model()
        if model is None or not hits:
            return _passthrough(hits, top_n)
        # 截断：cross-encoder 算分主要依据片段开头，截断换来的速度提升很划算
        pairs = [(query, h.chunk.text[: self.max_chars]) for h in hits]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        rescored: list[RetrievedChunk] = []
        for hit, score in zip(hits, scores):
            value = float(score)
            # 有些模型输出 logits、有些已过 sigmoid，这里统一按 logits 语义使用；
            # 归一化到 0~1 只是为了前端展示与阈值判断更方便。
            rescored.append(
                RetrievedChunk(
                    chunk=hit.chunk,
                    score=round(value, 6),
                    channel="rerank",
                    sub_scores={**hit.sub_scores, "rerank_raw": round(value, 6)},
                )
            )
        rescored.sort(key=lambda h: h.score, reverse=True)
        for rank, hit in enumerate(rescored, start=1):
            hit.rank = rank
        return rescored[:top_n]


# --------------------------------------------------------------------------- #
# 云端重排（硅基流动 / Jina / Cohere 风格接口）
# --------------------------------------------------------------------------- #
class ApiReranker(BaseReranker):
    """走 ``POST {api_base}/rerank`` 的云端重排，接口形态与硅基流动一致。"""

    provider = "api"

    def __init__(self, model: str, api_base: str, api_key: str) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.available = bool(api_base and api_key)

    def rerank(self, query: str, hits: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        if not self.available or not hits:
            return _passthrough(hits, top_n)
        import httpx

        try:
            resp = httpx.post(
                f"{self.api_base}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [h.chunk.text for h in hits],
                    "top_n": top_n,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception:
            return _passthrough(hits, top_n)

        out: list[RetrievedChunk] = []
        for rank, item in enumerate(results[:top_n], start=1):
            idx = int(item.get("index", 0))
            if idx >= len(hits):
                continue
            base = hits[idx]
            out.append(
                RetrievedChunk(
                    chunk=base.chunk,
                    score=round(float(item.get("relevance_score", 0.0)), 6),
                    channel="rerank",
                    rank=rank,
                    sub_scores={**base.sub_scores, "rerank_raw": float(item.get("relevance_score", 0.0))},
                )
            )
        return out or _passthrough(hits, top_n)


def _passthrough(hits: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """重排不可用时的优雅降级：沿用融合后的顺序。"""
    out = []
    for rank, hit in enumerate(hits[:top_n], start=1):
        out.append(
            RetrievedChunk(
                chunk=hit.chunk,
                score=hit.score,
                channel=hit.channel,
                rank=rank,
                sub_scores={**hit.sub_scores, "rerank_raw": 0.0},
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
_RERANKER_CACHE: dict[str, BaseReranker] = {}


def get_reranker() -> BaseReranker | None:
    cfg = settings.rerank
    if not cfg.enabled:
        return None
    key = f"{cfg.provider}:{cfg.model}"
    if key in _RERANKER_CACHE:
        return _RERANKER_CACHE[key]

    if cfg.provider == "local":
        reranker: BaseReranker = LocalCrossEncoderReranker(
            cfg.model, max_chars=cfg.max_chars, batch_size=cfg.batch_size
        )
    elif cfg.provider in {"api", "siliconflow", "jina", "cohere"}:
        reranker = ApiReranker(
            model=cfg.model,
            api_base=cfg.api_base or "https://api.siliconflow.cn/v1",
            api_key=cfg.api_key,
        )
    else:
        raise ValueError(f"未知 RERANK_PROVIDER: {cfg.provider}（可选 local / api）")

    _RERANKER_CACHE[key] = reranker
    return reranker


def rerank_info() -> dict[str, Any]:
    cfg = settings.rerank
    return {
        "enabled": cfg.enabled,
        "provider": cfg.provider,
        "model": cfg.model,
        "top_n": cfg.top_n,
        "candidate_pool": cfg.candidate_pool,
        "max_chars": cfg.max_chars,
        "batch_size": cfg.batch_size,
    }
