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
from app.runtime.agent import Agent                  # noqa: E402
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
    parser.add_argument("task", nargs="+", help="任务描述")
    parser.add_argument("--json", action="store_true", help="输出完整轨迹 JSON")
    parser.add_argument("--no-save", action="store_true", help="不落盘轨迹")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖步数上限")
    parser.add_argument("--preflight", action="store_true", help="运行前先做工具自检")
    args = parser.parse_args()

    task = " ".join(args.task)
    if args.max_steps:
        settings.guard.max_steps = args.max_steps

    agent = Agent()

    if args.preflight:
        report = agent.preflight()
        print("=" * 72)
        print(f"决策后端：{report['brain']}")
        for name, info in report["tools"].items():
            print(f"  {name:<18} {info['status']:<12} {info['detail']}")
        print("=" * 72)

    if not args.json:
        print(f"\n任务：{task}")
        print(f"后端：{agent.brain.label}   工具：{', '.join(agent.registry.names())}")
        print("-" * 72)

    trace = None
    for event, payload in agent.stream(task):
        if event == "step":
            if not args.json:
                _print_step(payload)
        else:
            trace = payload

    assert trace is not None
    if not args.no_save:
        store = TraceStore()
        store.save(trace)

    if args.json:
        print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
        return

    print("-" * 72)
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
