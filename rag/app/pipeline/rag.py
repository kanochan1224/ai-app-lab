"""RAG 主流程编排：把前面各步串成一次可观测的问答。

``answer`` 的关键行为：

1. **检索** → 混合召回 + 精排，拿到 top-k 片段与链路耗时；
2. **证据门槛** → 命中为空，或最优证据分低于 ``REFUSAL_SCORE_THRESHOLD`` 时直接拒答，
   根本不调用大模型（省钱、且避免「拿着不相关片段硬答」）；
3. **生成** → 把片段编号成 ``[1] [2]`` 交给模型，要求逐句标注来源；
4. **引用对齐** → 只回填答案里真正引用到的来源，并校验编号不越界。
"""

from __future__ import annotations

import time
from typing import Iterator

from app.config import settings
from app.llm.generator import (
    REFUSAL_MARKER,
    ExtractiveAnswerer,
    build_context,
    build_messages,
    filter_citations,
    get_chat_client,
    parse_citation_indices,
)
from app.retrieval.hybrid_retriever import HybridRetriever, Mode
from app.schema import Answer

REFUSAL_TEXT = (
    "这个问题在课程知识库中没有找到足够依据，我不做推测。\n\n"
    "建议：换一种问法（补上章节名或术语），或确认该内容是否在已上传的课程资料里。"
)


class RAGPipeline:
    """一次问答的完整编排。"""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    # ------------------------------------------------------------------ #
    def is_ready(self) -> bool:
        return self.retriever.chunk_count > 0

    # ------------------------------------------------------------------ #
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        mode: Mode | None = None,
        history: list[dict[str, str]] | None = None,
        use_rerank: bool | None = None,
    ) -> Answer:
        started = time.perf_counter()
        cfg = settings

        hits, trace = self.retriever.search(
            question, mode=mode, top_k=top_k, use_rerank=use_rerank
        )

        # ---- 证据门槛：宁可拒答，也不猜 ----
        # 判据：分数 < 阈值 才拒答（scripts/calibrate.py 用同一判据给出阈值建议）
        # 阈值为 None 表示「不启用阈值拒答」，只依赖「一条都没召回」这个条件
        best_score = self.retriever.best_evidence_score(hits)
        threshold = cfg.llm.refusal_score_threshold
        below_threshold = (
            threshold is not None and best_score is not None and best_score < threshold
        )
        if not hits or below_threshold:
            return Answer(
                question=question,
                answer=REFUSAL_TEXT,
                citations=[],
                contexts=hits,
                trace=trace,
                refused=True,
                model="refusal-gate",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                usage={"retrieval_hits": len(hits), "best_score": best_score, "threshold": threshold},
            )

        context, citations = build_context(hits)
        chat = get_chat_client()

        # ---- 无 Key：抽取式降级，链路照跑 ----
        if chat is None:
            if not cfg.llm.allow_extractive_fallback:
                raise RuntimeError(
                    "未配置 LLM_API_KEY，且 ALLOW_EXTRACTIVE_FALLBACK=false。"
                    "请在 .env 中填入密钥，或允许抽取式降级。"
                )
            result = ExtractiveAnswerer().answer(question, hits)
            text = REFUSAL_TEXT if result.refused else result.text
            return Answer(
                question=question,
                answer=text,
                citations=[] if result.refused else (filter_citations(text, citations) or citations[: len(hits)]),
                contexts=hits,
                trace=trace,
                refused=result.refused,
                model=result.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                usage={"mode": "extractive-fallback", "has_key": False},
            )

        # ---- 生成 ----
        messages = build_messages(question, context, history=history)
        result = chat.complete(messages)
        text = result.text
        refused = result.refused or REFUSAL_MARKER in text
        if refused:
            text = REFUSAL_TEXT

        used_citations = filter_citations(text, citations)
        return Answer(
            question=question,
            answer=text,
            citations=used_citations,
            contexts=hits,
            trace=trace,
            refused=refused,
            model=result.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            usage={**result.usage, "mode": "generative", "cited": parse_citation_indices(text)},
        )

    # ------------------------------------------------------------------ #
    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        mode: Mode | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict]:
        """流式问答：先吐 ``meta``（引用与召回片段），再逐段吐 ``delta``，最后 ``done``。

        SSE 事件格式由 API 层负责序列化，这里只产出结构化事件字典。
        """
        hits, trace = self.retriever.search(question, mode=mode, top_k=top_k)
        # 与 answer() 保持完全一致的证据门槛，避免两条路径行为不一致
        best_score = self.retriever.best_evidence_score(hits)
        threshold = settings.llm.refusal_score_threshold
        below_threshold = (
            threshold is not None and best_score is not None and best_score < threshold
        )
        if not hits or below_threshold:
            yield {"event": "meta", "data": {"citations": [], "contexts": [], "refused": True}}
            yield {"event": "delta", "data": {"text": REFUSAL_TEXT}}
            yield {"event": "done", "data": {"refused": True, "model": "refusal-gate"}}
            return

        context, citations = build_context(hits)
        yield {
            "event": "meta",
            "data": {
                "citations": [c.__dict__ for c in citations],
                "contexts": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "source_path": h.chunk.source_path,
                        "section_path": h.chunk.section_path,
                        "score": round(h.score, 4),
                        "channel": h.channel,
                    }
                    for h in hits
                ],
                "trace": trace.__dict__,
                "refused": False,
            },
        }

        chat = get_chat_client()
        if chat is None:
            result = ExtractiveAnswerer().answer(question, hits)
            yield {
                "event": "delta",
                "data": {"text": REFUSAL_TEXT if result.refused else result.text},
            }
            yield {"event": "done", "data": {"refused": result.refused, "model": result.model}}
            return

        buffer = ""
        for piece in chat.stream(build_messages(question, context, history=history)):
            buffer += piece
            yield {"event": "delta", "data": {"text": piece}}
        yield {
            "event": "done",
            "data": {
                "refused": REFUSAL_MARKER in buffer,
                "model": chat.model,
                "cited": parse_citation_indices(buffer),
            },
        }
