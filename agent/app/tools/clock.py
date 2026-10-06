"""时间工具。

看起来最没用，但 Agent 评估里经常需要它：
「今年」「最近」「当前版本」这类相对时间表述，模型必须知道今天是几号才不会答错。
另外它也是最小、最稳的工具，适合用来自检工具调用链路是否通畅。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..schema import ToolResult
from .base import Tool, ToolContext, ToolError

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class CurrentTimeTool(Tool):
    name = "current_time"
    description = (
        "获取当前日期与时间。当问题涉及「今天/今年/最近/现在」等相对时间，"
        "或需要判断信息时效性时，先用它确认当前时间，不要凭训练数据猜测。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA 时区名，例如 Asia/Shanghai、UTC、America/New_York，默认 Asia/Shanghai",
            }
        },
        "required": [],
    }
    returns = "当前日期、时间、星期与时区名。"

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        tz_name = str(arguments.get("timezone") or "Asia/Shanghai").strip()
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
            raise ToolError(f"未知时区：{tz_name}（应使用 IANA 名称，如 Asia/Shanghai）") from exc

        now = datetime.now(tz)
        utc = datetime.now(timezone.utc)
        weekday = _WEEKDAYS[now.weekday()]
        content = (
            f"当前时间（{tz_name}）：{now.strftime('%Y-%m-%d %H:%M:%S')} {weekday}\n"
            f"（UTC：{utc.strftime('%Y-%m-%d %H:%M:%S')}）"
        )
        return ToolResult(
            ok=True,
            content=content,
            raw={"iso": now.isoformat(), "date": now.strftime("%Y-%m-%d"), "weekday": weekday},
        )
