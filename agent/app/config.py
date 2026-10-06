"""配置：环境变量驱动，代码里不出现密钥。

另一个关键点是 **跨项目复用**：本 Agent 的 ``search_course_kb`` 工具直接调用
course-rag 项目（本仓库的 rag/ 目录）里的检索器，而不是重新实现一套。通过 ``COURSE_RAG_PATH``
把那个项目的根目录挂到 ``sys.path`` 上即可——这也是「系统」与「散件」的区别。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRACE_DIR = DATA_DIR / "traces"
EVAL_DIR = DATA_DIR / "eval"
WORKSPACE_DIR = DATA_DIR / "workspace"      # 沙箱可写目录

# 被复用的 RAG 项目（单仓库里同级目录名为 rag）
DEFAULT_RAG_PATH = ROOT_DIR.parent / "rag"

load_dotenv(ROOT_DIR / ".env")

# 这些写法都表示「未填写」，应当退回默认值
_BLANK_MARKERS = {"", ".", "./", ".\\", "none", "null", "todo", "changeme"}


def _env(key: str, default: str = "") -> str:
    """读字符串配置。

    **踩过的坑**：``os.getenv(key, default)`` 在「变量存在但值为空」时返回空串，
    默认值不会生效。而 ``.env`` 里写一行 ``COURSE_RAG_PATH=``（留空占位）
    恰好就是这种情况——于是相对路径默认值 ``"."`` 被当成真实配置，
    导致跨项目工具静默失效。所以这里显式判断空值。
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lower() in _BLANK_MARKERS:
        return default
    return raw


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key).lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


def _env_path(key: str, default: Path) -> Path:
    """读路径配置：相对路径按项目根目录解析，避免受「当前工作目录」影响。

    这也是踩过的坑：``Path(".")`` 会随调用方 cwd 变化，
    在不同入口（CLI / API / 测试）表现出不同行为，属于最难排查的一类问题。
    """
    raw = _env(key)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


@dataclass
class BrainSettings:
    """决策后端。

    - ``openai``  任意 OpenAI 兼容端点，支持 function calling（DeepSeek / 硅基流动 / 通义…）
    - ``mock``    离线脚本化策略，用于无 Key 演示与单元测试（**不做真实决策**）
    """

    backend: str = field(default_factory=lambda: _env("BRAIN_BACKEND", "openai"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    api_base: str = field(default_factory=lambda: _env("LLM_API_BASE", "https://api.deepseek.com/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "") or _env("DEEPSEEK_API_KEY", ""))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1500))
    timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 90))
    # 工具调用模式：auto 由模型决定；required 强制本轮必须调工具（首轮常用）
    tool_choice: str = field(default_factory=lambda: _env("TOOL_CHOICE", "auto"))

    @property
    def is_mock(self) -> bool:
        return self.backend == "mock" or not self.api_key

    @property
    def mode_label(self) -> str:
        if self.backend == "mock":
            return "mock（脚本化策略，非真实决策）"
        if not self.api_key:
            return "mock（未配置 LLM_API_KEY，自动降级）"
        return f"{self.backend}:{self.model}"


@dataclass
class GuardSettings:
    """防护参数——Agent 失控通常就栽在这几个值上。"""

    max_steps: int = field(default_factory=lambda: _env_int("MAX_STEPS", 12))
    max_seconds: float = field(default_factory=lambda: _env_float("MAX_SECONDS", 180.0))
    # 连续重复同一工具调用多少次就判定「原地打转」
    repeat_tolerance: int = field(default_factory=lambda: _env_int("REPEAT_TOLERANCE", 2))
    # 单个工具结果注入模型上下文的字符上限（防止一次抓取撑爆上下文）
    max_observation_chars: int = field(
        default_factory=lambda: _env_int("MAX_OBSERVATION_CHARS", 4000)
    )
    # 工具调用连续失败多少次后强制要求给出最终答案
    max_consecutive_failures: int = field(
        default_factory=lambda: _env_int("MAX_CONSECUTIVE_FAILURES", 4)
    )


@dataclass
class ToolSettings:
    """工具相关配置。"""

    enable_web_search: bool = field(default_factory=lambda: _env_bool("ENABLE_WEB_SEARCH", True))
    enable_code_exec: bool = field(default_factory=lambda: _env_bool("ENABLE_CODE_EXEC", True))
    enable_course_kb: bool = field(default_factory=lambda: _env_bool("ENABLE_COURSE_KB", True))
    search_max_results: int = field(default_factory=lambda: _env_int("SEARCH_MAX_RESULTS", 5))
    fetch_max_chars: int = field(default_factory=lambda: _env_int("FETCH_MAX_CHARS", 6000))
    http_timeout: float = field(default_factory=lambda: _env_float("HTTP_TIMEOUT", 15.0))
    code_timeout: float = field(default_factory=lambda: _env_float("CODE_TIMEOUT", 10.0))
    user_agent: str = field(
        default_factory=lambda: _env(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        )
    )


@dataclass
class Settings:
    project_name: str = "AgentLab"
    brain: BrainSettings = field(default_factory=BrainSettings)
    guard: GuardSettings = field(default_factory=GuardSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    course_rag_path: Path = field(
        default_factory=lambda: _env_path("COURSE_RAG_PATH", DEFAULT_RAG_PATH)
    )
    trace_dir: Path = field(default_factory=lambda: TRACE_DIR)
    workspace_dir: Path = field(default_factory=lambda: WORKSPACE_DIR)

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    def ensure_dirs(self) -> None:
        for p in (self.trace_dir, EVAL_DIR, self.workspace_dir):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, str]:
        return {
            "brain": self.brain.mode_label,
            "guard": (
                f"max_steps={self.guard.max_steps} "
                f"max_seconds={self.guard.max_seconds:.0f} "
                f"repeat_tolerance={self.guard.repeat_tolerance}"
            ),
            "course_rag": str(self.course_rag_path),
            "workspace": str(self.workspace_dir),
        }


settings = Settings()
