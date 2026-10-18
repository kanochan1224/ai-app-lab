"""跑一个任务：``python -m scripts.run "你的任务"``。

三种查看方式：
- 默认打印每一步的决策与工具调用（实时流式）；
- ``--json`` 输出完整轨迹 JSON，便于程序消费；
- 结束后轨迹自动落盘，可用 ``python -m scripts.traces`` 回看。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                      # noqa: E402
from app.runtime.factory import build_agent          # noqa: E402
from app.runtime.trace_store import TraceStore       # noqa: E402

_COLORS = {
    "tool_call": "\033[36m",      # 青色
    "tool_result": "\033[32m",    # 绿色
    "final": "\033[35m",          # 紫色
    "guard": "\033[33m",          # 黄色
    "error": "\033[31m",          # 红色
    "thought": "\033[90m",        # 灰色
}
_RESET = "\033[0m"


def _paint(kind: str, text: str) -> str:
    return f"{_COLORS.get(kind, '')}{text}{_RESET}"


def _print_step(step) -> None:
    head = f"[{step.index:>2}] {step.kind:<11}"
    calls = step.all_calls()
    if calls:
        shown = []
        for call in calls[:3]:
            args = ", ".join(f"{k}={v!r}" for k, v in list(call.arguments.items())[:2])
            shown.append(f"{call.name}({args[:60]})")
        head += " " + " + ".join(shown)
        if len(calls) > 3:
            head += f" …（共 {len(calls)} 个并行调用）"
    print(_paint(step.kind, head))
    if step.kind in {"thought", "final"} and step.content:
        print(f"      {step.content[:600]}")
    elif step.kind == "tool_result":
        preview = step.content[:400].replace("\n", "\n      ")
        print(f"      {preview}")
        for failed in [r for r in step.all_results() if not r.ok]:
            print(_paint("error", f"      ⚠ 失败：{failed.error[:200]}"))
    elif step.kind in {"guard", "error"}:
        print(_paint(step.kind, f"      {step.content[:300]}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Agent 任务")
    parser.add_argument("task", nargs="*", help="任务描述")
    parser.add_argument("--json", action="store_true", help="输出完整轨迹 JSON")
    parser.add_argument("--no-save", action="store_true", help="不落盘轨迹")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖步数上限")
    parser.add_argument("--preflight", action="store_true", help="运行前先做工具自检")
    parser.add_argument(
        "--engine",
        choices=["loop", "langgraph"],
        default=None,
        help="主循环实现（默认取 AGENT_ENGINE，即 loop）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="危险工具执行前暂停等人工审批（需 --engine langgraph）",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="从某个被中断的 run 继续跑（需 --engine langgraph + checkpointer）",
    )
    parser.add_argument(
        "--approve",
        choices=["approve", "reject"],
        default="approve",
        help="配合 --resume 使用：放行或拒绝那个待审批的危险工具",
    )
    args = parser.parse_args()

    if not args.task and not args.resume:
        parser.error("要么给一个任务描述，要么用 --resume 续跑一个已存在的 run")

    task = " ".join(args.task)
    if args.max_steps:
        settings.guard.max_steps = args.max_steps

    # 续跑必须用 LangGraph 引擎：手写版没有持久化状态，无从恢复
    engine = args.engine or ("langgraph" if (args.resume or args.interactive) else None)
    agent = build_agent(engine, interactive=args.interactive)

    if args.resume:
        return _run_resume(agent, args)

    if args.preflight:
        report = agent.preflight()
        print("=" * 72)
        print(f"决策引擎：{getattr(agent, 'engine', '?')}    决策后端：{report['brain']}")
        for name, info in report["tools"].items():
            print(f"  {name:<18} {info['status']:<12} {info['detail']}")
        print("=" * 72)

    if not args.json:
        print(f"\n任务：{task}")
        print(
            f"引擎：{getattr(agent, 'engine', '?')}   "
            f"后端：{agent.brain.label}   工具：{', '.join(agent.registry.names())}"
        )
        print("-" * 72)

    trace = None
    for event, payload in agent.stream(task):
        if event == "step":
            if not args.json:
                _print_step(payload)
        else:
            trace = payload

    assert trace is not None
    if args.interactive:
        trace = _approve_loop(agent, trace, args)
    _finish(agent, trace, args)


def _approve_loop(agent, trace, args):
    """在**同一个进程内**完成「暂停 → 询问 → 续跑」的循环。

    为什么必须同进程：默认的 InMemorySaver 是进程内状态，换个进程就找不到
    checkpoint 了。想真正跨进程续跑，得把 checkpointer 换成 SqliteSaver
    （见 requirements-langgraph.txt 里那行注释，装 langgraph-checkpoint-sqlite 即可）。
    """
    probe = getattr(agent, "pending_interrupt", None)
    if not callable(probe):
        return trace
    while True:
        pending = probe(trace.run_id)
        if not pending:
            return trace
        print("-" * 72)
        print(_paint("guard", "⏸  暂停：危险工具等待人工审批"))
        print(f"   工具：{pending.get('tool')}")
        print(f"   参数：{pending.get('arguments')}")
        try:
            answer = input("   放行？[a]pprove / [r]eject（直接回车 = 拒绝）> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n   非交互环境，按「拒绝」处理。")
            answer = "r"
        verdict = "approve" if answer.startswith("a") else "reject"
        print(f"   → {verdict}")
        if args.json:
            # JSON 模式不打印步骤，但仍要推进流程
            for event, payload in agent.resume_stream(trace.run_id, verdict):
                if event == "done":
                    trace = payload
        else:
            for event, payload in agent.resume_stream(trace.run_id, verdict):
                if event == "step":
                    _print_step(payload)
                else:
                    trace = payload


def _run_resume(agent, args) -> None:
    """从中断点续跑——手写引擎做不到这件事，所以这里必须挡住并说清楚原因。"""
    if not hasattr(agent, "resume_stream"):
        raise SystemExit(
            "当前引擎不支持续跑。手写循环没有持久化状态，无法从中断处恢复；"
            "请加 --engine langgraph（或用 AGENT_ENGINE=langgraph）。"
        )
    trace = None
    for event, payload in agent.resume_stream(args.resume, args.approve):
        if event == "step":
            if not args.json:
                _print_step(payload)
        else:
            trace = payload
    if trace is None:
        raise SystemExit(f"没找到 run_id={args.resume} 的 checkpoint，无法续跑。")
    _finish(agent, trace, args)


def _finish(agent, trace, args) -> None:
    """两条路径（首次运行 / 续跑）共用的收尾：落盘、打印、以及中断提示。"""
    if not args.no_save:
        TraceStore().save(trace)

    if args.json:
        print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
        return

    # 停在人工审批点上 ≠ 失败，要把「下一步该敲什么命令」直接给出来
    pending = None
    probe = getattr(agent, "pending_interrupt", None)
    if callable(probe):
        try:
            pending = probe(trace.run_id)
        except Exception:
            pending = None

    print("-" * 72)
    if pending:
        print(_paint("guard", "⏸  已暂停，等待人工审批"))
        print(f"   工具：{pending.get('tool')}   参数：{pending.get('arguments')}")
        print(f"   放行：python -m scripts.run --resume {trace.run_id} --approve approve")
        print(f"   拒绝：python -m scripts.run --resume {trace.run_id} --approve reject")
        print(f"{trace.summary_line()}  run_id={trace.run_id}")
        return

    print(_paint("final", "最终答案："))
    print(trace.final_answer or "(空)")
    print("-" * 72)
    print(f"{trace.summary_line()}  run_id={trace.run_id}")
    if settings.brain.is_mock:
        print(
            _paint(
                "guard",
                "提示：当前是离线 mock 策略（未配置 LLM_API_KEY），"
                "以上不是真实模型决策。配置密钥后可获得真正的推理能力。",
            )
        )


if __name__ == "__main__":
    main()
