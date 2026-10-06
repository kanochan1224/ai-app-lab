"""pytest 全局配置。

关键点：**在导入 app 之前**把环境变量设为离线可跑的值，
否则 app.config 会在 import 时读取到开发用的 .env（可能要下模型、连外网）。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- 必须在 import app.* 之前设置 ----
_TMP = Path(tempfile.mkdtemp(prefix="courserag-test-"))
os.environ.update(
    {
        "EMBEDDING_PROVIDER": "hash",        # 不下载模型
        "EMBEDDING_DIM": "256",
        "RERANK_ENABLED": "false",           # 不下载重排模型
        "LLM_API_KEY": "",                   # 走抽取式降级，不发网络请求
        "DEEPSEEK_API_KEY": "",
        "INDEX_COLLECTION": "test_kb",
        "REFUSAL_SCORE_THRESHOLD": "",
    }
)

from app.config import settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolate_index_dirs():
    """把索引目录指到临时目录，避免污染真实 data/index。"""
    settings.chroma_dir.parent.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_raw_dir() -> Path:
    """测试用的小语料目录。"""
    return ROOT / "tests" / "data" / "raw"


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def built_index(sample_raw_dir: Path, tmp_root: Path):
    """建好一个小型索引（向量 + BM25），供检索与流水线测试复用。"""
    from app.ingest.chunker import chunk_documents
    from app.ingest.loader import load_documents
    from app.retrieval.bm25_index import BM25Index
    from app.retrieval.vector_store import VectorStore

    docs = load_documents(sample_raw_dir)
    chunks = chunk_documents(docs, strategy="recursive", chunk_size=300, chunk_overlap=50)

    store = VectorStore(persist_dir=tmp_root / "chroma", collection="test_kb")
    store.reset()
    store.add_chunks(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)

    return {"docs": docs, "chunks": chunks, "store": store, "bm25": bm25}
