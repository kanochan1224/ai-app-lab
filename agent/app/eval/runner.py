"""Agent 任务评估：把「Agent 行不行」变成可量化的数字。

评估 Agent 比评估 RAG 麻烦，因为它没有唯一正确答案，而且**过程也重要**。
所以这里分两层指标：

**结果层**
- ``success_rate``      任务是否完成（按每条样本的判据）
- ``answer_coverage``   预期关键词在最终答案里出现了多少

**过程层**（这是 Agent 评估区别于普通问答的地方）
- ``avg_steps`` / ``avg_tool_calls``   效率：能不能少走弯路
- ``tool_accuracy``                    工具选择是否正确（首轮是否用了该用的工具）
- ``recovery_rate``                    工具失败后能否自我纠正并最终完成
- ``guard_rate``                       触发防护的比例——过高说明提示词或工具设计有问题
- ``avg_cost``                         平均 token 与耗时（真实成本）

每条样本的判据用三种方式组合，避免「用关键词硬判」的脆弱性：
``expected_tools`` 检查工具使用，``answer_must_include`` 检查答案内容，
``forbid_tools`` 检查不该动用的工具（例如纯课程问题不该去联网）。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import EVAL_DIR, settings
from ..runtime.agent import Agent
from ..runtime.trace_store import TraceStore
from ..schema import Trace


@dataclass
class EvalTask:
    """一条评估任务。"""

    id: str
    task: str
    expected_tools: list[str] = field(default_factory=list)      # 期望用到的工具（任一命中即可）
    answer_must_include: list[str] = field(default_factory=list)  # 答案里应出现的关键词（任一）
    forbid_tools: list[str] = field(default_factory=list)         # 不应动用的工具
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)


def load_tasks(path: Path | None = None) -> list[EvalTask]:
    path = Path(path or EVAL_DIR / "tasks.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"评估任务集不存在：{path}")
    tasks: list[EvalTask] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"任务集第 {line_no} 行不是合法 JSON：{exc}") from exc
        tasks.append(
            EvalTask(
                id=item.get("id", f"t{line_no:03d}"),
                task=item["task"],
                expected_tools=list(item.get("expected_tools") or []),
                answer_must_include=list(item.get("answer_must_include") or []),
                forbid_tools=list(item.get("forbid_tools") or []),
                difficulty=item.get("difficulty", "medium"),
                tags=list(item.get("tags") or []),
            )
        )
    return tasks


# 这些措辞说明模型**没有真正回答**（工具失败、无依据、拒答）。
# 若不检查它们，就会出现「工具序列看着对、关键词碰巧命中，但答案是『我无法回答』」
# 却被判成功的情况——这是最危险的一类假阳性。
_NO_ANSWER_MARKERS = (
    "未能获取", "未获取到", "无法获取", "未能完成", "无法完成",
    "无法回答", "不能回答", "不做猜测", "不便凭印象", "没有找到依据",
    "未检索到", "检索工具调用失败", "工具调用失败", "知识库为空",
    "无法给出", "无法确认",
)


def looks_like_non_answer(text: str) -> str:
    """判断答案是否属于「没答出来」。返回命中的标记，未命中返回空串。"""
    body = (text or "").strip()
    if not body:
        return "答案为空"
    for marker in _NO_ANSWER_MARKERS:
        if marker in body:
            return marker
    return ""


@dataclass
class TaskOutcome:
    """单条任务的评估结果。"""

    id: str
    task: str
    trace: Trace
    completed: bool
    tool_hit: bool | None
    forbidden_used: list[str]
    coverage: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "completed": self.completed,
            "tool_hit": self.tool_hit,
            "forbidden_used": self.forbidden_used,
            "coverage": round(self.coverage, 3),
            "note": self.note,
            "stop_reason": self.trace.stop_reason,
            "steps": self.trace.num_steps,
            "tool_calls": self.trace.num_tool_calls,
            "failed_tools": self.trace.num_failed_tools,
            "tools": self.trace.tool_sequences,
            "elapsed_ms": self.trace.total_ms,
            "tokens": self.trace.total_tokens,
            "final_answer": self.trace.final_answer[:400],
        }


def judge_task(task: EvalTask, trace: Trace) -> TaskOutcome:
    """按判据给一条任务打分。"""
    used = trace.tool_sequences
    # 结果层：必须正常结束，且答案非空
    completed = trace.stop_reason == "finished" and bool(trace.final_answer.strip())

    tool_hit: bool | None = None
    if task.expected_tools:
        tool_hit = any(t in used for t in task.expected_tools)
        completed = completed and tool_hit

    forbidden_used = [t for t in used if t in task.forbid_tools]
    if forbidden_used:
        completed = False

    coverage = 0.0
    if task.answer_must_include:
        hits = sum(1 for kw in task.answer_must_include if kw and kw in trace.final_answer)
        coverage = hits / len(task.answer_must_include)
        # 关键词一个都没命中，判定为没真正回答（避免「答了一堆但不对题」）
        completed = completed and coverage > 0

    # 「没答出来」检测：工具失败、无依据、明确拒答都算失败。
    # 不做这一步会出现假阳性——例如工具报错后模型礼貌地说明「未能获取依据」，
    # 工具序列与关键词判据都可能通过，但用户其实什么也没得到。
    non_answer = looks_like_non_answer(trace.final_answer)
    if non_answer:
        completed = False

    note = ""
    if forbidden_used:
        note = f"动用了不该用的工具：{', '.join(forbidden_used)}"
    elif task.expected_tools and tool_hit is False:
        note = f"未使用期望工具（期望 {task.expected_tools}，实际 {used}）"
    elif task.answer_must_include and coverage == 0:
        note = "答案未命中任何预期关键词"
    elif non_answer:
        note = f"未真正作答（命中标记「{non_answer}」）"
    elif trace.stop_reason != "finished":
        note = f"未正常结束：{trace.stop_reason}"

    return TaskOutcome(
        id=task.id,
        task=task.task,
        trace=trace,
        completed=completed,
        tool_hit=tool_hit,
        forbidden_used=forbidden_used,
        coverage=coverage,
        note=note,
    )


@dataclass
class AgentEvalReport:
    samples: int
    success_rate: float
    avg_coverage: float
    tool_accuracy: float
    avg_steps: float
    avg_tool_calls: float
    avg_elapsed_ms: float
    avg_tokens: float
    guard_rate: float
    recovery_rate: float
    stop_reason_dist: dict[str, int] = field(default_factory=dict)
    outcomes: list[TaskOutcome] = field(default_factory=list)
    brain: str = ""
    is_mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "brain": self.brain,
            "is_mock": self.is_mock,
            "metrics": {
                "success_rate": self.success_rate,
                "avg_coverage": self.avg_coverage,
                "tool_accuracy": self.tool_accuracy,
                "avg_steps": self.avg_steps,
                "avg_tool_calls": self.avg_tool_calls,
                "avg_elapsed_ms": self.avg_elapsed_ms,
                "avg_tokens": self.avg_tokens,
                "guard_rate": self.guard_rate,
                "recovery_rate": self.recovery_rate,
            },
            "stop_reason_dist": self.stop_reason_dist,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def run_agent_eval(
    agent: Agent,
    tasks: list[EvalTask],
    limit: int | None = None,
    store: TraceStore | None = None,
    verbose: bool = True,
) -> AgentEvalReport:
    """逐条跑任务并汇总指标。"""
    selected = tasks[:limit] if limit else tasks
    store = store or TraceStore()
    outcomes: list[TaskOutcome] = []
    stop_dist: dict[str, int] = {}

    for task in selected:
        started = time.perf_counter()
        trace = agent.run(task.task)
        store.save(trace)
        outcome = judge_task(task, trace)
        outcomes.append(outcome)
        stop_dist[trace.stop_reason] = stop_dist.get(trace.stop_reason, 0) + 1
        if verbose:
            flag = "✔" if outcome.completed else "✘"
            print(
                f"  {flag} [{task.id}] {trace.stop_reason:<12} "
                f"步数={trace.num_steps:<2} 工具={','.join(trace.tool_sequences) or '-':<28} "
                f"{trace.total_ms / 1000:.1f}s"
                + (f"  ← {outcome.note}" if outcome.note else "")
            )

    n = len(outcomes) or 1
    successes = sum(1 for o in outcomes if o.completed)
    tool_judged = [o for o in outcomes if o.tool_hit is not None]
    guard_traces = [o for o in outcomes if any(s.kind == "guard" for s in o.trace.steps)]

    return AgentEvalReport(
        samples=len(outcomes),
        success_rate=round(successes / n, 4),
        avg_coverage=round(statistics.mean([o.coverage for o in outcomes]), 4) if outcomes else 0.0,
        tool_accuracy=round(
            sum(1 for o in tool_judged if o.tool_hit) / len(tool_judged), 4
        ) if tool_judged else 0.0,
        avg_steps=round(statistics.mean([o.trace.num_steps for o in outcomes]), 2) if outcomes else 0.0,
        avg_tool_calls=round(
            statistics.mean([o.trace.num_tool_calls for o in outcomes]), 2
        ) if outcomes else 0.0,
        avg_elapsed_ms=round(statistics.mean([o.trace.total_ms for o in outcomes]), 1) if outcomes else 0.0,
        avg_tokens=round(statistics.mean([o.trace.total_tokens for o in outcomes]), 1) if outcomes else 0.0,
        guard_rate=round(len(guard_traces) / n, 4),
        recovery_rate=round(
            sum(1 for o in outcomes if o.trace.recovered_from_failure) / n, 4
        ),
        stop_reason_dist=stop_dist,
        outcomes=outcomes,
        brain=agent.brain.label,
        is_mock=agent.cfg.brain.is_mock,
    )


def report_to_markdown(report: AgentEvalReport) -> str:
    lines = ["# AgentLab 任务评估报告", ""]
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 决策后端：**{report.brain}**")
    lines.append(f"- 任务数：{report.samples}")
    lines.append("")

    # 工程层指标：无论真实模型还是 mock 都成立，是「系统是否可靠」的证据
    lines.append("## 一、工程层指标（与决策后端无关，任何模式下都成立）")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 工具执行成功率 | {_tool_success_rate(report):.1%} | 工具调用未抛异常、未超时的比例 |"
    )
    lines.append(f"| 轨迹完整率 | {_trace_integrity(report):.1%} | 步骤序列完整且可反序列化的比例 |")
    lines.append(f"| 防护触发率 | {report.guard_rate:.1%} | 步数/重复/连续失败防护被激活的比例 |")
    lines.append(f"| 平均耗时 | {report.avg_elapsed_ms / 1000:.1f} s | 端到端，含模型与工具时间 |")
    lines.append("")

    if report.is_mock:
        lines.append("## 二、能力层指标 ⚠️ 本次不可用")
        lines.append("")
        lines.append(
            "本次运行使用 **离线 mock 策略**（未配置 `LLM_API_KEY`）。"
            "mock 只做关键词到工具的固定映射，**不具备理解意图、多步规划、"
            "根据观察调整策略的能力**，因此下面这些指标反映的是 mock 的能力上限，"
            "**不能作为 Agent 能力的证据**："
        )
        lines.append("")
        lines.append(f"- 任务成功率 {report.success_rate:.1%}（受 mock 规则覆盖度限制）")
        lines.append(f"- 工具选择准确率 {report.tool_accuracy:.1%}")
        lines.append(f"- 失败恢复率 {report.recovery_rate:.1%}（mock 不会真正重规划）")
        lines.append("")
        lines.append(
            "> 配置 `LLM_API_KEY` 后重跑 `python -m scripts.evaluate`，"
            "这一节才会变成有意义的真实数据。"
        )
        lines.append("")
    else:
        lines.append("## 二、能力层指标（真实模型决策）")
        lines.append("")
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("|---|---|---|")
        lines.append(f"| 任务成功率 | **{report.success_rate:.1%}** | 工具使用 + 答案关键词综合判定 |")
        lines.append(f"| 答案覆盖度 | {report.avg_coverage:.1%} | 预期关键词命中比例 |")
        lines.append(f"| 工具选择准确率 | {report.tool_accuracy:.1%} | 是否用了该用的工具 |")
        lines.append(f"| 平均步数 | {report.avg_steps} | 越少越高效 |")
        lines.append(f"| 平均工具调用 | {report.avg_tool_calls} | 反映绕路程度 |")
        lines.append(f"| 平均 token | {report.avg_tokens:.0f} | 真实成本 |")
        lines.append(f"| 失败恢复率 | {report.recovery_rate:.1%} | 工具报错后仍完成的比例 |")
        lines.append("")

    lines.append("### 停止原因分布")
    lines.append("")
    lines.append("| 原因 | 次数 |")
    lines.append("|---|---|")
    for reason, count in sorted(report.stop_reason_dist.items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append("## 三、逐题明细")
    lines.append("")
    lines.append("| 任务 | 结果 | 停止原因 | 步数 | 工具序列 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    for o in report.outcomes:
        mark = "✔" if o.completed else "✘"
        tools = " → ".join(o.trace.tool_sequences) or "-"
        lines.append(
            f"| {o.id} | {mark} | {o.trace.stop_reason} | {o.trace.num_steps} | {tools} | {o.note} |"
        )
    lines.append("")
    return "\n".join(lines)


def _tool_success_rate(report: AgentEvalReport) -> float:
    """工具执行成功率：失败调用占全部调用的比例。"""
    total = sum(o.trace.num_tool_calls for o in report.outcomes)
    failed = sum(o.trace.num_failed_tools for o in report.outcomes)
    return 1.0 if total == 0 else (total - failed) / total


def _trace_integrity(report: AgentEvalReport) -> float:
    """轨迹完整率：有步骤、有明确停止原因、且能重新序列化。"""
    if not report.outcomes:
        return 0.0
    good = 0
    for outcome in report.outcomes:
        trace = outcome.trace
        try:
            payload = trace.to_dict()
            ok = bool(payload.get("steps")) and bool(trace.stop_reason) and "metrics" in payload
        except Exception:
            ok = False
        good += 1 if ok else 0
    return good / len(report.outcomes)


def save_report(report: AgentEvalReport, tag: str = "") -> tuple[Path, Path]:
    out_dir = EVAL_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    json_path = out_dir / f"agent-eval-{stamp}{suffix}.json"
    md_path = out_dir / f"agent-eval-{stamp}{suffix}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path
