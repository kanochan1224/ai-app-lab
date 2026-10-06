"""pytest 全局配置。

关键：**在导入 app.* 之前**把环境设成「完全离线、可复现」：
- ``BRAIN_BACKEND=mock``：不调用任何大模型，决策由脚本策略给出；
- 关掉联网工具的真实请求（搜索结果页结构会变，不适合放进单元测试）；
- 轨迹写到临时目录，避免污染真实数据。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="agentlab-test-"))
os.environ.update(
    {
        "BRAIN_BACKEND": "mock",
        "LLM_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
        "ENABLE_WEB_SEARCH": "false",     # 单元测试不依赖外网与页面结构
        "ENABLE_CODE_EXEC": "true",
        "MAX_STEPS": "6",
        "MAX_SECONDS": "30",
        "REPEAT_TOLERANCE": "1",
        "MAX_CONSECUTIVE_FAILURES": "2",
        "MAX_OBSERVATION_CHARS": "800",
    }
)

from app.config import settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolate_dirs():
    """轨迹与沙箱目录指向临时目录。"""
    settings.trace_dir = _TMP / "traces"
    settings.workspace_dir = _TMP / "workspace"
    settings.ensure_dirs()
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture()
def tmp_trace_store(tmp_path: Path):
    from app.runtime.trace_store import TraceStore

    return TraceStore(tmp_path / "traces")


@pytest.fixture()
def agent():
    """一个只挂本地工具（计算器 / 时间 / 沙箱）的 Agent，不碰网络与索引。"""
    from app.config import Settings
    from app.brain.base import MockBrain
    from app.runtime.agent import Agent
    from app.tools import CalculatorTool, CodeExecTool, CurrentTimeTool
    from app.tools.base import ToolRegistry

    cfg = Settings()
    cfg.workspace_dir = _TMP / "workspace"
    cfg.trace_dir = _TMP / "traces"
    cfg.ensure_dirs()
    registry = ToolRegistry(max_observation_chars=cfg.guard.max_observation_chars)
    registry.register(CurrentTimeTool())
    registry.register(CalculatorTool())
    registry.register(CodeExecTool(workspace=cfg.workspace_dir))
    return Agent(registry=registry, brain=MockBrain(), settings=cfg)
