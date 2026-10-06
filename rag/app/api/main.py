"""FastAPI 服务层。

对外提供「问答 / 流式问答 / 检索调试 / 重建索引 / 健康检查」五组接口。
启动时自动加载 BM25 索引与向量库；未建索引时服务照常起，但问答会明确提示先建索引。

启动：
    uvicorn app.api.main:app --port 8000
或：
    python -m scripts.serve
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT_DIR, settings
from app.llm.embeddings import embedding_info
from app.llm.generator import llm_info
from app.pipeline.rag import RAGPipeline
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import rerank_info
from app.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("course-rag")

STATE: dict[str, Any] = {
    "pipeline": None,
    "vector_store": None,
    "bm25": None,
    "loaded_at": None,
    "index_error": None,
    "indexing": False,
}


# --------------------------------------------------------------------------- #
# 启动 / 关闭
# --------------------------------------------------------------------------- #
def bootstrap(force_reload: bool = False) -> dict[str, Any]:
    """加载向量库与 BM25 索引，构造问答流水线。"""
    settings.ensure_dirs()
    try:
        vector_store = VectorStore()
        bm25 = BM25Index()
        loaded = bm25.load(settings.bm25_cache)
        if not loaded:
            chunks = vector_store.get_all_chunks()
            if chunks:
                bm25.build(chunks)
                bm25.save(settings.bm25_cache)
        retriever = HybridRetriever(vector_store=vector_store, bm25=bm25)
        STATE.update(
            pipeline=RAGPipeline(retriever),
            vector_store=vector_store,
            bm25=bm25,
            loaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            index_error=None,
        )
        logger.info(
            "索引加载完成：向量 %d 条 / BM25 %d 条", vector_store.count, bm25.size
        )
    except Exception as exc:  # 索引缺失或损坏不应导致服务起不来
        STATE["index_error"] = str(exc)
        logger.warning("索引加载失败：%s", exc)
    return STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    yield
    STATE.clear()


app = FastAPI(
    title="CourseRAG · 课程知识库问答",
    description=(
        "文档解析 → 文本切分 → 向量化 → 混合检索(BM25+向量+RRF) → 重排 → 带引用生成。\n\n"
        "所有回答均可溯源到课程资料的具体章节。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# 请求 / 响应模型
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="学生问题")
    top_k: int | None = Field(None, ge=1, le=20, description="最终送入生成的片段数")
    mode: str | None = Field(
        None,
        description="检索模式：dense（仅向量）/ sparse（仅BM25）/ hybrid（融合）/ hybrid_rerank（融合+重排）",
    )
    use_rerank: bool | None = Field(None, description="是否启用重排；留空按配置")
    history: list[dict[str, str]] | None = Field(None, description="多轮对话历史")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = "hybrid_rerank"
    top_k: int = Field(10, ge=1, le=50)


class ReindexRequest(BaseModel):
    strategy: str | None = Field(None, description="fixed / recursive / semantic")
    chunk_size: int | None = Field(None, ge=80, le=2000)
    rebuild_vectors: bool = Field(True, description="是否重建向量索引（否则只重建 BM25）")


class SourceHit(BaseModel):
    chunk_id: str
    source_path: str
    section_path: str
    text: str
    score: float
    channel: str
    rank: int
    sub_scores: dict[str, float] = {}


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    model: str
    latency_ms: float
    citations: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    trace: dict[str, Any] | None = None
    usage: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
@app.get("/api/health", summary="健康检查与配置快照")
def health() -> dict[str, Any]:
    pipeline: RAGPipeline | None = STATE.get("pipeline")
    vector_store: VectorStore | None = STATE.get("vector_store")
    bm25: BM25Index | None = STATE.get("bm25")
    return {
        "status": "ok",
        "ready": bool(pipeline and pipeline.is_ready()),
        "loaded_at": STATE.get("loaded_at"),
        "index_error": STATE.get("index_error"),
        "index": {
            "vectors": vector_store.count if vector_store else 0,
            "bm25": bm25.size if bm25 else 0,
            "collection": settings.index_collection,
        },
        "config": settings.describe(),
        "backends": {
            "embedding": embedding_info(),
            "rerank": rerank_info(),
            "llm": llm_info(),
        },
    }


@app.post("/api/query", response_model=QueryResponse, summary="知识库问答（一次性返回）")
def query(req: QueryRequest) -> QueryResponse:
    pipeline: RAGPipeline | None = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪，请稍后重试")
    if not pipeline.is_ready():
        raise HTTPException(
            status_code=409,
            detail="索引为空，请先执行：python -m scripts.build_index",
        )
    try:
        answer = pipeline.answer(
            req.question,
            top_k=req.top_k,
            mode=req.mode,          # type: ignore[arg-type]
            history=req.history,
            use_rerank=req.use_rerank,
        )
    except Exception as exc:
        logger.exception("问答失败")
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc
    return QueryResponse(**answer.to_dict())


@app.post("/api/query/stream", summary="知识库问答（SSE 流式）")
def query_stream(req: QueryRequest) -> StreamingResponse:
    pipeline: RAGPipeline | None = STATE.get("pipeline")
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(status_code=409, detail="索引未就绪，请先执行 python -m scripts.build_index")

    def event_source():
        try:
            for event in pipeline.stream_answer(
                req.question, top_k=req.top_k, mode=req.mode, history=req.history
            ):
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {payload}\n\n"
        except Exception as exc:  # 流已开始，只能以错误事件收尾
            logger.exception("流式问答失败")
            payload = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/retrieve", summary="只检索不生成（调试召回效果）")
def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    pipeline: RAGPipeline | None = STATE.get("pipeline")
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(status_code=409, detail="索引未就绪")
    hits, trace = pipeline.retriever.search(req.query, mode=req.mode, top_k=req.top_k)  # type: ignore[arg-type]
    return {
        "query": req.query,
        "count": len(hits),
        "trace": trace.__dict__,
        "results": [
            SourceHit(
                chunk_id=h.chunk.chunk_id,
                source_path=h.chunk.source_path,
                section_path=h.chunk.section_path,
                text=h.chunk.text,
                score=round(h.score, 6),
                channel=h.channel,
                rank=h.rank,
                sub_scores={k: round(v, 6) for k, v in h.sub_scores.items()},
            ).model_dump()
            for h in hits
        ],
    }


@app.get("/api/documents", summary="列出已入库的文档与片段分布")
def documents() -> dict[str, Any]:
    bm25: BM25Index | None = STATE.get("bm25")
    if bm25 is None or bm25.size == 0:
        return {"documents": [], "total_chunks": 0}
    stats: dict[str, dict[str, Any]] = {}
    for chunk in bm25.chunks:
        item = stats.setdefault(
            chunk.source_path,
            {"source_path": chunk.source_path, "chunks": 0, "chars": 0, "sections": set()},
        )
        item["chunks"] += 1
        item["chars"] += len(chunk.text)
        if chunk.section_path:
            item["sections"].add(chunk.section_path)
    docs = [
        {
            "source_path": v["source_path"],
            "chunks": v["chunks"],
            "chars": v["chars"],
            "sections": len(v["sections"]),
        }
        for v in sorted(stats.values(), key=lambda x: x["source_path"])
    ]
    return {"documents": docs, "total_chunks": bm25.size}


def _run_reindex(req: ReindexRequest) -> None:
    """后台重建索引（避免请求超时）。"""
    from scripts.build_index import build as build_index

    STATE["indexing"] = True
    try:
        build_index(
            strategy=req.strategy,
            chunk_size=req.chunk_size,
            rebuild_vectors=req.rebuild_vectors,
        )
        bootstrap()
    finally:
        STATE["indexing"] = False


@app.post("/api/reindex", summary="重建索引（后台执行）")
def reindex(req: ReindexRequest, background: BackgroundTasks) -> dict[str, Any]:
    if STATE.get("indexing"):
        raise HTTPException(status_code=409, detail="已有重建任务在执行中")
    background.add_task(_run_reindex, req)
    return {"status": "started", "message": "索引重建已在后台开始，可轮询 /api/health 查看进度"}


# --------------------------------------------------------------------------- #
# 前端静态页
# --------------------------------------------------------------------------- #
WEB_DIR = ROOT_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
