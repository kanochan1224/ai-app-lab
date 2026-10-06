"""启动 API 服务：``python -m scripts.serve``。

启动前会做一次工具自检，把「哪些工具真的能用」直接打在终端上——
这比让你打开网页后再发现索引没建、联网不通要好得多。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings      # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 AgentLab API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    settings.ensure_dirs()
    print("=" * 72)
    print("  AgentLab · 工具调用 Agent 实验台")
    print("=" * 72)
    for key, value in settings.describe().items():
        print(f"  {key:<12}: {value}")

    if not args.skip_preflight:
        try:
            from app.runtime.agent import Agent

            agent = Agent()
            report = agent.preflight()
            print("-" * 72)
            print(f"  决策后端：{report['brain']}")
            for name, info in report["tools"].items():
                print(f"    {name:<18} {info['status']:<12} {info['detail']}")
            if report["is_mock"]:
                print("-" * 72)
                print("  ⚠ 未配置 LLM_API_KEY：当前为离线 mock 策略。")
                print("    工具链路、轨迹、防护都能正常演示，但决策不是真实模型做的。")
        except Exception as exc:
            print(f"  自检失败（服务仍会启动）：{type(exc).__name__}: {exc}")

    print("-" * 72)
    print(f"  实验台页面 : http://{args.host}:{args.port}/")
    print(f"  接口文档   : http://{args.host}:{args.port}/docs")
    print("=" * 72)

    import uvicorn

    uvicorn.run("app.api.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
