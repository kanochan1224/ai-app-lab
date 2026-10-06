"""FastAPI 服务：把 Agent 暴露成接口，并提供一个轨迹检视前端。

接口设计的一个要点：**长任务用 SSE 流式返回每一步**。
Agent 一次任务可能跑十几秒，若用普通请求，前端只能干等；
流式返回后用户能实时看到「它正在调什么工具」，
这既是体验问题，也是可解释性问题。
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import ROOT_DIR, settings
from ..eval.runner import load_tasks, save_report, run_agent_eval
from ..runtime.agent import Agent
from ..runtime.trace_store import TraceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agentlab")

STATE: dict[str, Any] = {"agent": None, "store": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    STATE["store"] = TraceStore()
    try:
        STATE["agent"] = Agent()
        logger.info("Agent 就绪：%s", STATE["agent"].brain.label)
    except Exception as exc:            # 工具装配失败也要让服务起得来
        logger.warning("Agent 初始化失败：%s", exc)
    yield
    STATE.clear()


app = FastAPI(
    title="AgentLab · 工具调用 Agent",
    description=(
        "多步工具调用 Agent：课程知识库检索（复用 course-rag）→ 联网搜索 → "
        "网页抓取 → 沙箱代码执行 → 计算器 → 时间。\n\n"
        "每次运行都有完整轨迹，可回放每一步的决策与工具调用。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=2000)
    max_steps: int | None = Field(None, ge=1, le=30)


class EvalRequest(BaseModel):
    limit: int | None = Field(None, ge=1, le=100)


def _agent() -> Agent:
    agent = STATE.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未就绪")
    return agent


@app.get("/api/health", summary="健康检查与工具自检")
def health() -> dict[str, Any]:
    agent = STATE.get("agent")
    store: TraceStore | None = STATE.get("store")
    if agent is None:
        return {"status": "degraded", "detail": "Agent 未初始化"}
    return {
        "status": "ok",
        "config": settings.describe(),
        "preflight": agent.preflight(),
        "traces": store.count if store else 0,
    }


@app.get("/api/tools", summary="列出可用工具及其 Schema")
def tools() -> dict[str, Any]:
    agent = _agent()
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "dangerous": spec.dangerous,
            }
            for spec in agent.registry.specs()
        ]
    }


@app.post("/api/run", summary="运行任务（一次性返回完整轨迹）")
def run(req: RunRequest) -> dict[str, Any]:
    agent = _agent()
    if req.max_steps:
        settings.guard.max_steps = req.max_steps
    trace = agent.run(req.task)
    store: TraceStore | None = STATE.get("store")
    if store:
        store.save(trace)
    return trace.to_dict()


@app.post("/api/run/stream", summary="运行任务（SSE 流式，逐步推送）")
def run_stream(req: RunRequest) -> StreamingResponse:
    agent = _agent()
    if req.max_steps:
        settings.guard.max_steps = req.max_steps

    def event_source() -> Iterator[str]:
        trace = None
        try:
            for event, payload in agent.stream(req.task):
                if event == "step":
                    data = payload.to_dict()
                else:
                    trace = payload
                    data = payload.to_dict()
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("流式运行失败")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
            return
        if trace is not None:
            store: TraceStore | None = STATE.get("store")
            if store:
                store.save(trace)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/traces", summary="历史运行摘要列表")
def list_traces(limit: int = 30, failed_only: bool = False, tool: str | None = None) -> dict[str, Any]:
    store: TraceStore | None = STATE.get("store")
    if store is None:
        return {"traces": []}
    rows = store.list_summaries(limit=200)
    if failed_only:
        rows = [r for r in rows if r.get("stop_reason") != "finished"]
    if tool:
        rows = [r for r in rows if tool in (r.get("tools") or [])]
    return {"traces": rows[:limit], "total": store.count}


@app.get("/api/traces/{run_id}", summary="读取单条轨迹（用于回放）")
def get_trace(run_id: str) -> dict[str, Any]:
    store: TraceStore | None = STATE.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail="轨迹存储未就绪")
    trace = store.get(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"未找到轨迹 {run_id}")
    return trace.to_dict()


@app.post("/api/eval", summary="跑任务评估集（同步，耗时较长）")
def run_eval(req: EvalRequest) -> dict[str, Any]:
    agent = _agent()
    tasks = load_tasks()
    if req.limit:
        tasks = tasks[: req.limit]
    report = run_agent_eval(agent, tasks, store=STATE.get("store"), verbose=False)
    json_path, md_path = save_report(report, tag="api")
    payload = report.to_dict()
    payload["report_files"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


WEB_DIR = ROOT_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
