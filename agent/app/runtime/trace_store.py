"""轨迹存储：把每次运行完整落盘，支持按时间回放与批量加载评估。

落盘格式刻意用 **JSONL（每行一条运行）** 而不是每条一个文件：
评估脚本要一次读几百条轨迹，JSONL 可以流式读取，不用管理几百个小文件。

同时提供 ``index.jsonl`` 摘要索引，便于前端快速列出历史运行而不必读全量轨迹。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import settings
from ..schema import Step, ToolCall, ToolResult, Trace


class TraceStore:
    """轨迹读写。"""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory or settings.trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.dir / "runs.jsonl"
        self.index_file = self.dir / "index.jsonl"

    # ------------------------------------------------------------------ #
    def save(self, trace: Trace) -> Path:
        """追加写入轨迹与索引摘要。"""
        record = trace.to_dict()
        with self.runs_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "run_id": trace.run_id,
            "task": trace.task,
            "brain": trace.brain,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stop_reason": trace.stop_reason,
            "final_answer": trace.final_answer[:300],
            "tools": trace.tool_sequences,
            **record["metrics"],
        }
        with self.index_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return self.runs_file

    # ------------------------------------------------------------------ #
    def load_all(self) -> list[Trace]:
        """加载全部轨迹（评估用）。"""
        if not self.runs_file.exists():
            return []
        traces: list[Trace] = []
        for line in self.runs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(_record_to_trace(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue          # 单条损坏不该拖垮整次评估
        return traces

    def get(self, run_id: str) -> Trace | None:
        for trace in self.load_all():
            if trace.run_id == run_id:
                return trace
        return None

    def list_summaries(self, limit: int = 50) -> list[dict]:
        """读取索引摘要，最新的在前。"""
        if not self.index_file.exists():
            return []
        rows: list[dict] = []
        for line in self.index_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(rows))[:limit]

    def clear(self) -> None:
        for path in (self.runs_file, self.index_file):
            path.unlink(missing_ok=True)

    @property
    def count(self) -> int:
        if not self.index_file.exists():
            return 0
        return sum(1 for _ in self.index_file.open(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 反序列化
# --------------------------------------------------------------------------- #
def _record_to_trace(record: dict) -> Trace:
    trace = Trace(
        run_id=record.get("run_id", ""),
        task=record.get("task", ""),
        started_at=float(record.get("started_at") or 0.0),
        finished_at=float(record.get("finished_at") or 0.0),
        final_answer=record.get("final_answer", ""),
        stop_reason=record.get("stop_reason", "finished"),
        brain=record.get("brain", ""),
        tools_available=list(record.get("tools_available") or []),
        error=record.get("error", ""),
    )
    metrics = record.get("metrics") or {}
    trace.prompt_tokens = int(metrics.get("prompt_tokens") or 0)
    trace.completion_tokens = int(metrics.get("completion_tokens") or 0)

    for raw in record.get("steps") or []:
        step = Step(
            index=int(raw.get("index", 0)),
            kind=raw.get("kind", "thought"),
            content=raw.get("content", ""),
            started_at=float(raw.get("started_at") or 0.0),
            elapsed_ms=float(raw.get("elapsed_ms") or 0.0),
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            guard_note=raw.get("guard_note", ""),
        )
        # 兼容两种历史格式：新的 tool_calls/tool_results 列表，以及旧的单数字段
        raw_calls = raw.get("tool_calls")
        if raw_calls is None and raw.get("tool_call"):
            raw_calls = [raw["tool_call"]]
        for item in raw_calls or []:
            step.tool_calls.append(
                ToolCall(
                    name=item.get("name", ""),
                    arguments=item.get("arguments") or {},
                    call_id=item.get("call_id", ""),
                )
            )
        raw_results = raw.get("tool_results")
        if raw_results is None and raw.get("tool_result"):
            raw_results = [raw["tool_result"]]
        for item in raw_results or []:
            step.tool_results.append(
                ToolResult(
                    ok=bool(item.get("ok")),
                    content=item.get("content", ""),
                    error=item.get("error", ""),
                    elapsed_ms=float(item.get("elapsed_ms") or 0.0),
                    truncated=bool(item.get("truncated")),
                )
            )
        # 同时填单数字段，方便只关心单次调用的消费方
        if step.tool_calls:
            step.tool_call = step.tool_calls[0]
        if step.tool_results:
            step.tool_result = step.tool_results[0]
        trace.steps.append(step)
    return trace
