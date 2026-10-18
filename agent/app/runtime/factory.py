"""按配置构造 Agent —— 让调用方不需要知道用的是哪个引擎。

两个引擎（``loop`` / ``langgraph``）对外契约完全一致：
``run`` / ``stream`` / ``preflight`` / ``brain`` / ``cfg`` / ``registry``。
所以 CLI、API、评估脚本都只需要 ``build_agent()``，不需要 import 具体实现——
这也是为什么加第三个引擎时改动会很小。
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, settings as default_settings
from .agent import Agent

_LOOP_ALIASES = {"loop", "handwritten", "native", "builtin"}
_GRAPH_ALIASES = {"langgraph", "graph", "lg"}


def build_agent(
    engine: str | None = None,
    *,
    settings: Settings | None = None,
    interactive: bool = False,
    checkpointer: Any = None,
) -> Any:
    """构造一个 Agent。

    ``engine`` 为空时取配置里的 ``AGENT_ENGINE``（默认 ``loop``）。

    ``interactive`` / ``checkpointer`` 只对 LangGraph 引擎有意义——
    手写版没有持久化状态，天然无法「暂停后换个进程接着跑」。
    传了它们却用 loop 引擎时会**明确报错**，而不是静默忽略：
    静默忽略会让人以为审批生效了，实际危险工具照跑。

    未知引擎名同样直接报错并列出可选值，不做「猜一个最接近的」。
    """
    cfg = settings or default_settings
    name = (engine or cfg.engine or "loop").strip().lower()

    if name in _LOOP_ALIASES:
        if interactive or checkpointer is not None:
            raise ValueError(
                f"loop 引擎不支持 interactive/checkpointer（无持久化状态）。"
                f"请改用 engine='langgraph'，即 AGENT_ENGINE=langgraph。"
            )
        return Agent(settings=cfg)

    if name in _GRAPH_ALIASES:
        from .langgraph_agent import LangGraphAgent

        return LangGraphAgent(
            settings=cfg, interactive=interactive, checkpointer=checkpointer
        )

    raise ValueError(
        f"未知引擎 {name!r}。可选：loop（手写循环，默认）/ langgraph（状态图）。"
        f"对应环境变量 AGENT_ENGINE。"
    )
