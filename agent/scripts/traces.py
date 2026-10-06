"""查看历史轨迹：``python -m scripts.traces``。

相当于给 Agent 装了个「行车记录仪」：可以列出历史运行、回放某一次、
按条件筛选（失败 / 触发防护 / 用了某工具）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.runtime.trace_store import TraceStore      # noqa: E402


def _list(store: TraceStore, limit: int, only_failed: bool, tool: str | None) -> None:
    rows = store.list_summaries(limit=500)
    if only_failed:
        rows = [r for r in rows if r.get("stop_reason") != "finished"]
    if tool:
        rows = [r for r in rows if tool in (r.get("tools") or [])]
    rows = rows[:limit]

    if not rows:
        print("没有匹配的轨迹。先跑几个任务：python -m scripts.run \"你的任务\"")
        return

    print(f"{'run_id':<14}{'时间':<21}{'停止':<12}{'步数':>5}{'工具':>5}{'秒':>7}  任务")
    print("-" * 100)
    for r in rows:
        print(
            f"{r.get('run_id', ''):<14}{r.get('saved_at', ''):<21}"
            f"{r.get('stop_reason', ''):<12}{r.get('num_steps', 0):>5}"
            f"{r.get('num_tool_calls', 0):>5}{r.get('total_ms', 0) / 1000:>7.1f}  "
            f"{r.get('task', '')[:40]}"
        )
    print(f"\n共 {len(rows)} 条（最新在前）")


def _replay(store: TraceStore, run_id: str) -> None:
    trace = store.get(run_id)
    if trace is None:
        print(f"未找到轨迹 {run_id}")
        return
    print("=" * 80)
    print(f"任务：{trace.task}")
    print(f"后端：{trace.brain}    {trace.summary_line()}")
    print("=" * 80)
    for step in trace.steps:
        head = f"[{step.index:>2}] {step.kind:<11}"
        calls = step.all_calls()
        if calls:
            head += " " + ", ".join(f"{c.name}({c.arguments})" for c in calls)
        print(head)
        if step.content:
            for line in step.content[:500].splitlines():
                print(f"      {line}")
        if step.guard_note:
            print(f"      ⚠ 防护：{step.guard_note}")
    print("=" * 80)
    print("最终答案：")
    print(trace.final_answer or "(空)")


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 / 回放 Agent 轨迹")
    parser.add_argument("--show", type=str, help="回放指定 run_id")
    parser.add_argument("--limit", type=int, default=20, help="列出条数")
    parser.add_argument("--failed", action="store_true", help="只看未正常结束的")
    parser.add_argument("--tool", type=str, default=None, help="只看用过某工具的")
    parser.add_argument("--clear", action="store_true", help="清空全部轨迹")
    args = parser.parse_args()

    store = TraceStore()
    if args.clear:
        store.clear()
        print("已清空轨迹。")
        return
    if args.show:
        _replay(store, args.show)
        return
    _list(store, args.limit, args.failed, args.tool)


if __name__ == "__main__":
    main()
