"""course-rag 项目的桥接层。

**为什么需要单独一层**：两个项目都有自己的 ``app`` 包。若天真地把
course-rag（rag/）的根目录插到 ``sys.path`` 然后直接 ``import app.xxx``，
就会和本项目自己的 ``app`` 包撞名——表现为「某些模块突然变成另一个项目的实现」，
而且报错位置和真实原因完全不相干，属于最难排查的一类 bug。

这里的做法是有意做一次**显式越界**：
1. 只在真正需要时才加载（懒加载），不拖慢普通启动；
2. 加载期间临时切换 ``sys.path``，**加载完立刻还原**；
3. 通过 ``importlib`` 按文件路径加载 course-rag（rag/）的检索器与配置，
   并在 ``sys.modules`` 里用唯一别名注册，避免与本体冲突；
4. 失败时给出可操作的中文提示，而不是抛一个 ImportError 让人猜。
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from typing import Any

from ..config import settings

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}
_ERROR: str = ""


class CourseRAGUnavailable(RuntimeError):
    """复用 course-rag 失败（通常是路径不对或缺依赖）。"""


def _load_course_rag() -> dict[str, Any]:
    """把 course-rag 的检索器加载进来。

    关键点：course-rag 内部大量使用绝对导入（``from app.config import settings``），
    因此切换 sys.path 必须同时清掉 ``app`` 相关的模块缓存，
    加载完成后**立刻还原**，把对本体导入系统的影响限制在这一小段临界区内。
    """
    global _ERROR
    root = Path(settings.course_rag_path)
    if not (root / "app" / "retrieval" / "hybrid_retriever.py").exists():
        _ERROR = (
            f"未找到 course-rag 项目：{root}\n"
            "请检查 .env 里的 COURSE_RAG_PATH：\n"
            "  · 把该行留空或删除 → 自动使用同级目录 "
            f"{settings.course_rag_path.parent / 'rag'}\n"
            "  · 或填写该项目根目录的绝对路径"
            "（该目录下应存在 app/retrieval/hybrid_retriever.py）"
        )
        raise CourseRAGUnavailable(_ERROR)

    saved_path = list(sys.path)
    # 记录并清掉可能已存在的 app 包缓存，避免解析到本项目的 app
    saved_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "app" or name.startswith("app.")
    }
    sys.path.insert(0, str(root))
    try:
        config_mod = importlib.import_module("app.config")
        bm25_mod = importlib.import_module("app.retrieval.bm25_index")
        vs_mod = importlib.import_module("app.retrieval.vector_store")
        hr_mod = importlib.import_module("app.retrieval.hybrid_retriever")
        payload = {
            "rag_settings": config_mod.settings,
            "BM25Index": bm25_mod.BM25Index,
            "VectorStore": vs_mod.VectorStore,
            "HybridRetriever": hr_mod.HybridRetriever,
            "root": root,
        }
    except Exception as exc:  # 缺依赖、索引损坏、版本不兼容都在这里兜住
        _ERROR = (
            f"加载 course-rag 失败：{type(exc).__name__}: {exc}\n"
            f"项目路径：{root}\n"
            "请确认该项目已安装依赖（pip install -r requirements-core.txt）"
        )
        raise CourseRAGUnavailable(_ERROR) from exc
    finally:
        # 还原导入环境：把本项目的 app 包缓存放回去，sys.path 恢复原状
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            sys.modules[name] = module
    return payload


def get_course_rag() -> dict[str, Any]:
    """线程安全地获取（并缓存）course-rag 的检索组件。"""
    if _CACHE:
        return _CACHE
    with _LOCK:
        if not _CACHE:
            _CACHE.update(_load_course_rag())
    return _CACHE


def build_retriever():
    """构造一个可直接用的 HybridRetriever（复用 course-rag 已建好的索引）。"""
    parts = get_course_rag()
    store = parts["VectorStore"]()
    bm25 = parts["BM25Index"]()
    bm25.ensure_loaded(store)          # 索引缺失时会自动从向量库重建
    return parts["HybridRetriever"](vector_store=store, bm25=bm25)


def availability() -> dict[str, Any]:
    """供 /health 与 CLI 展示的可用性信息，不抛异常。"""
    root = Path(settings.course_rag_path)
    info = {
        "path": str(root),
        "exists": (root / "app" / "retrieval" / "hybrid_retriever.py").exists(),
        "loaded": bool(_CACHE),
        "error": _ERROR,
        "chunks": _CACHE.get("_chunks", 0) if _CACHE else 0,
    }
    return info
