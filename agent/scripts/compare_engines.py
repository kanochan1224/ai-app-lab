"""两个引擎跑同一批任务，产出对比表：``python -m scripts.compare_engines``。

这个脚本存在的意义不是「证明 LangGraph 更好」——**它们的行为本来就该一样**。
它要回答两个不同的问题：

1. **行为是否等价**（``--mock``，离线免费）：
   用脚本化假大脑跑，两个引擎的每一条轨迹都应该逐字一致。
   这是「重写是行为保持的」的证明——重构最容易悄悄改坏东西，而等价性断言能抓住。

2. **成本差多少**（不加 ``--mock``，走真实模型）：
   步数、耗时、token 应当接近，差值主要来自图调度的开销。
   这里如实报告，不做「框架更快」这种没有依据的结论。

用法：
    python -m scripts.compare_engines --mock              # 离线等价性验证
    python -m scripts.compare_engines                     # 真实模型对比
    python -m scripts.compare_engines --limit 3 --mock    # 快速冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR, settings                # noqa: E402
from app.eval.runner import load_tasks, run_agent_eval   # noqa: E402
from app.runtime.factory import build_agent              # noqa: E402

ENGINES = ("loop", "langgraph")

# 对比表里关心的指标（键 -> 表头）
_METRICS = [
    ("success_rate", "任务成功率"),
    ("tool_accuracy", "工具选择准确率"),
    ("avg_steps", "平均步数"),
    ("avg_tool_calls", "平均工具调用"),
    ("avg_tokens", "平均 token"),
    ("guard_rate", "防护触发率"),
]


def _skeleton(trace, strict: bool) -> list[tuple]:
    """抽出「引擎可控」的那部分行为，用于跨引擎比对。

    哪些东西**不能**算进等价性判据，取决于用的是谁的大脑：

    - ``strict=True``（脚本化假大脑）：连**工具参数**一起比。参数由剧本给定，
      两个引擎必须给出完全一样的调用——这是最强的断言，也是等价性的主证据。
    - ``strict=False``（真实模型）：**只比结构**——步骤类型与顺序、工具名序列、
      成败标志、防护标记、停止原因。参数和正文都是模型这一次生成的，
      两次独立的 API 调用本来就会不同，拿它判「引擎有 bug」是冤枉。
      即使 temperature=0，供应商也不保证跨调用完全一致。

    两种情况都排除时间与 ID（``started_at`` / ``elapsed_ms`` / token / run_id）
    以及**工具返回的具体内容**（取决于环境，如 ``current_time`` 的秒数）；
    工具失败与否（``ok``）要保留——那是引擎必须正确传递的信息。
    """
    skeleton = []
    for step in trace.steps:
        # 只保留「引擎自己写」的文本：guard / error 出自防护逻辑，两个引擎必须一致。
        # final 正文来自模型、tool_result 正文来自工具，都不由引擎负责
        # （mock 大脑更是直接复述工具输出，会让时间戳从工具一路泄漏进 final）。
        content = step.content if step.kind in {"guard", "error"} else ""
        calls = tuple(
            (
                c.name,
                tuple(sorted(c.arguments.items())) if strict else (),
            )
            for c in step.all_calls()
        )
        skeleton.append(
            (
                step.kind,
                content,
                calls,
                tuple(r.ok for r in step.all_results()),
                step.guard_note,
            )
        )
    return skeleton


def _tool_outputs_differ(traces_by_engine: dict, task_id: str) -> bool:
    """工具返回内容是否逐字相同——只作为信息展示，不作为等价性判据。"""
    outs = []
    for engine in ENGINES:
        trace = traces_by_engine.get(engine, {}).get(task_id)
        outs.append(
            tuple(r.content for s in trace.steps for r in s.all_results()) if trace else None
        )
    return outs[0] != outs[1]


def _verdict_marks(fmt: str) -> dict[str, str]:
    """终端用符号，Markdown 用文字——把 ASCII 符号塞进报告文件很难看。"""
    if fmt == "md":
        return {"ok": "✅", "diff": "❌ 不一致"}
    return {"ok": "[一致]", "diff": "[不一致]"}


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 loop 与 langgraph 两个引擎")
    parser.add_argument("--dataset", type=str, default=None, help="任务集路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--mock", action="store_true", help="强制离线假大脑（验证等价性）")
    parser.add_argument("--save", action="store_true", help="把对比结果写入报告文件")
    args = parser.parse_args()

    if args.mock:
        # 必须在构造 Agent 之前改，因为 build_brain 是在 Agent 初始化时读配置的
        settings.brain.backend = "mock"
        settings.brain.api_key = ""

    tasks = load_tasks(Path(args.dataset) if args.dataset else None)
    if args.limit:
        tasks = tasks[: args.limit]

    mode = "离线假大脑（验证行为等价）" if args.mock else "真实模型（对比成本）"
    print("=" * 78)
    print(f"  引擎对比 · {mode}")
    print(f"  任务数：{len(tasks)}    引擎：{' / '.join(ENGINES)}")
    print("=" * 78)

    reports: dict[str, object] = {}
    raw_traces: dict[str, dict[str, object]] = {}

    for engine in ENGINES:
        try:
            agent = build_agent(engine)
        except RuntimeError as exc:          # 未安装 langgraph 时的友好退出
            print(f"\n跳过 {engine}：{exc}")
            continue
        print(f"\n▶ 正在跑 {engine} …")
        report = run_agent_eval(agent, tasks, verbose=False)
        reports[engine] = report
        raw_traces[engine] = {o.id: o.trace for o in report.outcomes}

    missing = [e for e in ENGINES if e not in reports]
    if missing:
        print(f"\n缺少引擎数据：{missing}，无法对比。")
        raise SystemExit(1)

    # ---- 指标表 ----
    print("\n" + "=" * 78)
    print("一、指标对比")
    print("=" * 78)
    header = f"{'指标':<16}" + "".join(f"{e:>14}" for e in ENGINES) + f"{'差异':>12}"
    print(header)
    print("-" * 78)
    rows = []
    for key, label in _METRICS:
        a = getattr(reports["loop"], key)
        b = getattr(reports["langgraph"], key)
        diff = "一致" if a == b else f"{b - a:+.4g}"
        print(f"{label:<16}{a:>14.4g}{b:>14.4g}{diff:>12}")
        rows.append((label, a, b, diff))

    # ---- 逐条轨迹比对 ----
    print("\n" + "=" * 78)
    print("二、逐条轨迹比对（只比结构，不比时间与 ID）")
    print("=" * 78)
    strict = args.mock
    scope = "工具参数也一并比对" if strict else "只比结构（参数与正文由模型生成，不比）"
    print(f"\n判据：{scope}\n")

    marks = _verdict_marks("md")
    same, diff_ids, tool_diff_ids = 0, [], []
    for task in tasks:
        a_trace = raw_traces["loop"].get(task.id)
        b_trace = raw_traces["langgraph"].get(task.id)
        a = _skeleton(a_trace, strict) if a_trace else None
        b = _skeleton(b_trace, strict) if b_trace else None
        tool_diff = _tool_outputs_differ(raw_traces, task.id)
        if tool_diff:
            tool_diff_ids.append(task.id)
        if a == b and a is not None:
            same += 1
            print(f"  {marks['ok']}  [{task.id}] {task.task[:38]}")
        else:
            diff_ids.append(task.id)
            print(f"  {marks['diff']}  [{task.id}] {task.task[:38]}")
            print(f"         loop: {len(a or [])} 步 / langgraph: {len(b or [])} 步")

    print("-" * 78)
    print(f"引擎行为一致：{same}/{len(tasks)}")
    if tool_diff_ids:
        print(
            f"另有 {len(tool_diff_ids)} 条的工具返回内容两次不同"
            f"（如 current_time 返回的秒数）：{', '.join(tool_diff_ids)}"
        )
        print("  这不属于引擎行为差异，已从等价性判据中排除。")
    if diff_ids:
        print(f"⚠️  行为不一致的任务：{', '.join(diff_ids)}")
        print("   不一致不一定是 bug，但必须逐个查清原因再下结论——")
        print("   先看是不是「模型两次生成的参数/措辞不同」，再看引擎本身。")

    if args.save:
        _save(args, mode, rows, same, len(tasks), diff_ids, tool_diff_ids, reports, strict)


def _save(args, mode, rows, same, total, diff_ids, tool_diff_ids, reports, strict) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = EVAL_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"engine-compare-{stamp}.md"
    json_path = out_dir / f"engine-compare-{stamp}.json"

    lines = [
        "# 引擎对比：手写循环 vs LangGraph",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模式：{mode}",
        f"- 任务数：{total}",
        f"- 决策后端：{reports['loop'].brain}",
        "",
        "> 两个引擎**共享同一套工具、防护参数与判定逻辑**，只有主循环实现不同。",
        "> 离线模式下轨迹应当逐字一致——这是「重写保持了行为」的证据。",
        "",
        "## 一、指标对比",
        "",
        "| 指标 | loop（手写） | langgraph（状态图） | 差异 |",
        "| --- | --- | --- | --- |",
    ]
    for label, a, b, diff in rows:
        lines.append(f"| {label} | {a:.4g} | {b:.4g} | {diff} |")
    strict_note = (
        "本次为**离线假大脑**，判据从严：连**工具参数**一起比对，"
        "两个引擎应当逐字一致。"
        if strict
        else "本次为**真实模型**，判据只比结构：步骤类型与顺序、工具名序列、"
        "成败标志、防护标记、停止原因。**工具参数与答案正文由模型当次生成，"
        "两次独立调用本就不同**（即使 temperature=0 也不保证跨调用一致），"
        "因此不计入等价性判据。"
    )
    lines += [
        "",
        "## 二、逐条轨迹一致性",
        "",
        f"**{same}/{total}** 条任务的引擎行为一致。",
        "",
        strict_note,
        "",
        "两种情况都排除：时间戳、耗时、token、run_id（本来就该不同），",
        "以及**工具返回的具体内容**（取决于环境而非引擎）。",
    ]
    if tool_diff_ids:
        lines += [
            "",
            f"另有 {len(tool_diff_ids)} 条任务的工具返回内容两次不同"
            f"（{', '.join(tool_diff_ids)}）——例如 `current_time` 返回的秒数，",
            "这属于环境差异，不是引擎行为差异。",
        ]
    if diff_ids:
        lines += ["", f"⚠️ **行为不一致**：{', '.join(diff_ids)}，需逐条归因后再下结论。"]
    lines += [
        "",
        "## 三、怎么读这张表",
        "",
        "- 离线模式下**指标与行为都应当完全一致**；出现差异说明重写改变了行为，是 bug 而非特性。",
        "- 真实模型模式下，**只有结构可比**。步数/token 接近即可，",
        "  差值来自模型每次的决策差异与图调度的固定开销；",
        "  延迟差异会随任务复杂度放大，**不应据此断言某个引擎更快**。",
        "- LangGraph 版真正的增量不在这些指标里，而在**它多出来的能力**：",
        "  checkpoint 持久化、中断续跑、危险工具人工审批。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "mode": mode,
        "samples": total,
        "identical_traces": same,
        "differing_tasks": diff_ids,
        "metrics": {
            engine: {k: getattr(rep, k) for k, _ in _METRICS}
            for engine, rep in reports.items()
        },
        "full_reports": {engine: rep.to_dict() for engine, rep in reports.items()},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n对比报告：{md_path}")
    print(f"对比数据：{json_path}")


if __name__ == "__main__":
    main()
