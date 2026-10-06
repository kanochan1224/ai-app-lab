"""任务评估：``python -m scripts.evaluate [--limit N]``。

不需要 API Key 也能跑（会用 mock 策略），但**报告里会明确标注那不是真实能力**。
想看真实数据，先在 .env 配置 LLM_API_KEY。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR, settings              # noqa: E402
from app.eval.runner import (                          # noqa: E402
    load_tasks,
    run_agent_eval,
    save_report,
)
from app.runtime.agent import Agent                    # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 任务评估")
    parser.add_argument("--dataset", type=str, default=None, help="任务集路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--tag", type=str, default="", help="报告文件名标签")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.dataset) if args.dataset else None)
    if args.limit:
        tasks = tasks[: args.limit]

    agent = Agent()
    print("=" * 78)
    print(f"决策后端：{agent.brain.label}")
    if agent.cfg.brain.is_mock:
        print(
            "⚠️  当前为离线 mock 策略（未配置 LLM_API_KEY）。\n"
            "    本次结果只验证工程链路，不代表 Agent 的真实能力。"
        )
    print(f"任务数：{len(tasks)}    工具：{', '.join(agent.registry.names())}")
    print("=" * 78)

    report = run_agent_eval(agent, tasks, store=None)
    json_path, md_path = save_report(report, tag=args.tag)

    print("-" * 78)
    print(f"任务成功率      : {report.success_rate:.1%}")
    print(f"答案覆盖度      : {report.avg_coverage:.1%}")
    print(f"工具选择准确率  : {report.tool_accuracy:.1%}")
    print(f"平均步数/调用   : {report.avg_steps} / {report.avg_tool_calls}")
    print(f"平均耗时        : {report.avg_elapsed_ms / 1000:.1f} s")
    print(f"平均 token      : {report.avg_tokens:.0f}")
    print(f"防护触发率      : {report.guard_rate:.1%}")
    print(f"失败恢复率      : {report.recovery_rate:.1%}")
    print(f"停止原因分布    : {report.stop_reason_dist}")
    print("-" * 78)
    print(f"JSON 报告：{json_path}")
    print(f"Markdown 报告：{md_path}")


if __name__ == "__main__":
    main()
