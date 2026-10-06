"""启动 API 服务：``python -m scripts.serve``。

等价于 ``uvicorn app.api.main:app --host 0.0.0.0 --port 8000``，
但会先检查索引状态并打印可点击的地址。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings          # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 CourseRAG API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    settings.ensure_dirs()
    has_index = settings.bm25_cache.exists()
    print("=" * 68)
    print(f"  {settings.knowledge_base_name} · 问答服务")
    print("=" * 68)
    for key, value in settings.describe().items():
        print(f"  {key:<16}: {value}")
    print("-" * 68)
    print(f"  索引缓存      : {'已就绪' if has_index else '未构建（请先跑 build_index）'}")
    print(f"  问答页面      : http://{args.host}:{args.port}/")
    print(f"  接口文档      : http://{args.host}:{args.port}/docs")
    print("=" * 68)

    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
