"""工具集合：注册表工厂 + 统一导出。"""

from __future__ import annotations

from ..config import Settings, settings as default_settings
from .base import Tool, ToolContext, ToolError, ToolRegistry
from .calculator import CalculatorTool, safe_eval
from .clock import CurrentTimeTool
from .code_exec import CodeExecTool
from .course_kb import CourseKBTool
from .web import FetchUrlTool, WebSearchTool

__all__ = [
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "CalculatorTool",
    "CurrentTimeTool",
    "CodeExecTool",
    "CourseKBTool",
    "WebSearchTool",
    "FetchUrlTool",
    "safe_eval",
    "build_registry",
]


def build_registry(settings: Settings | None = None) -> ToolRegistry:
    """按配置装配工具集。

    工具的启停通过 ``ENABLE_*`` 环境变量控制——评估不同工具组合对任务成功率的
    影响时，这个开关是必需的（例如「去掉联网后还能做对吗」）。
    """
    cfg = settings or default_settings
    cfg.ensure_dirs()
    registry = ToolRegistry(max_observation_chars=cfg.guard.max_observation_chars)

    # 时间与计算器是纯本地、零副作用的工具，永远开启
    registry.register(CurrentTimeTool())
    registry.register(CalculatorTool())

    if cfg.tools.enable_course_kb:
        registry.register(CourseKBTool())
    if cfg.tools.enable_web_search:
        registry.register(WebSearchTool(cfg.tools))
        registry.register(FetchUrlTool(cfg.tools))
    if cfg.tools.enable_code_exec:
        registry.register(CodeExecTool(cfg.tools.code_timeout, cfg.workspace_dir))

    return registry
