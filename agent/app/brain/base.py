"""决策后端（Agent 的「大脑」）。

两种实现：

- :class:`OpenAIBrain` —— 走任意 OpenAI 兼容端点的 function calling，
  由模型自己决定「调哪个工具」还是「直接回答」；
- :class:`MockBrain`   —— 离线脚本化策略，**不做真实决策**，
  仅在无 Key 时用于演示链路与单元测试，会明确标注自己不是真决策。

一个容易被忽略但必踩的坑：OpenAI 协议要求
**带 tool_calls 的 assistant 消息必须原样回填进历史**，
并且每条 tool 消息的 ``tool_call_id`` 要与之对应。
少了任何一环，服务端会直接返回 400，而错误信息往往看不懂。
这里用 :func:`append_assistant_tool_calls` 统一处理。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import BrainSettings
from ..schema import ToolCall, ToolSpec

SYSTEM_PROMPT = """你是一个会使用工具的智能助理。你的目标是**用最少的步骤**给出可靠答案。

可用工具：
{tools}

工作方式：
1. 先判断问题需要哪些信息。不确定时优先检索（课程知识库或联网），不要凭记忆作答。
2. 每次只调用一个工具，观察结果后再决定下一步。
3. 工具报错时不要重复同样的调用——换参数、换工具，或改用其他信息源。
4. 信息足够后立即给出最终答案，不要为了「更完整」而无限检索。

答案要求：
- 用中文回答，直接给结论，不要写「根据工具返回」这类过程性废话。
- 用到课程知识库时必须标注章节出处，格式如：`（《03-第3章-决策树.md》 > 3.2.4 信息增益的缺陷）`。
- 用到联网结果时给出关键来源。
- 如果工具都没能提供有效信息，明确说明「未能获取到相关依据」，不要编造。
- 需要计算时用 calculator 或 run_python，不要心算大数。

当你已经可以回答时，直接输出最终答案（不要再调用工具）。"""


@dataclass
class Decision:
    """一次决策的结果：要么调工具（可能一轮多个），要么给最终答案。"""

    kind: str                                   # "tool" | "final"
    thought: str = ""                           # 模型的思考过程（可能为空）
    # 一轮可能返回多个调用（并行调用）。只处理第一个会导致协议错误，详见 Step 的说明。
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_call(self) -> ToolCall | None:
        """首个调用，方便只关心单调用的调用方与测试。"""
        return self.tool_calls[0] if self.tool_calls else None


class BrainError(RuntimeError):
    """决策后端调用失败。"""


def append_assistant_tool_calls(
    messages: list[dict[str, Any]],
    content: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    """把带 tool_calls 的 assistant 消息按协议要求回填。

    必须保留 ``tool_calls`` 字段本身，只发 content 会导致服务端拒绝后续请求。
    """
    messages.append(
        {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls,
        }
    )


def append_tool_result(messages: list[dict[str, Any]], call_id: str, name: str, content: str) -> None:
    messages.append(
        {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}
    )


class Brain(ABC):
    """决策后端接口。"""

    label: str = "brain"

    @abstractmethod
    def decide(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Decision:
        """给定对话历史与可用工具，产出下一步决策。"""

    def system_message(self, tools: list[ToolSpec]) -> dict[str, str]:
        lines = []
        for spec in tools:
            params = spec.parameters.get("properties", {}) or {}
            required = set(spec.parameters.get("required", []) or [])
            args = ", ".join(
                f"{k}{'*' if k in required else ''}" for k in params
            )
            lines.append(f"- {spec.name}({args})：{spec.description}")
        return {"role": "system", "content": SYSTEM_PROMPT.format(tools="\n".join(lines))}


# --------------------------------------------------------------------------- #
# 真实后端
# --------------------------------------------------------------------------- #
class OpenAIBrain(Brain):
    """OpenAI 兼容的 function calling 后端。"""

    def __init__(self, cfg: BrainSettings) -> None:
        if not cfg.api_key:
            raise BrainError("未配置 LLM_API_KEY")
        self.cfg = cfg
        self.label = f"{cfg.model}"
        self.api_base = cfg.api_base.rstrip("/")

    def decide(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Decision:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_openai_schema() for t in tools]
            payload["tool_choice"] = self.cfg.tool_choice

        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.cfg.timeout,
            )
        except httpx.HTTPError as exc:
            raise BrainError(f"请求决策模型失败：{exc}") from exc
        if resp.status_code != 200:
            raise BrainError(f"决策模型返回 {resp.status_code}：{resp.text[:300]}")

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        raw_calls = message.get("tool_calls") or []

        # 把原始 assistant 消息回填进历史（含 tool_calls），供调用方拼下一轮上下文
        if raw_calls:
            append_assistant_tool_calls(
                messages, message.get("content") or "", raw_calls
            )

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        if raw_calls:
            calls: list[ToolCall] = []
            for raw_call in raw_calls:
                fn = raw_call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append(
                    ToolCall(
                        name=fn.get("name", ""),
                        arguments=args,
                        call_id=raw_call.get("id", ""),
                    )
                )
            return Decision(
                kind="tool",
                thought=(message.get("content") or "").strip(),
                tool_calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw={
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "parallel": len(calls) > 1,
                },
            )

        content = (message.get("content") or "").strip()
        return Decision(
            kind="final",
            thought="",
            final_answer=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw={"elapsed_ms": round((time.perf_counter() - started) * 1000, 1)},
        )


# --------------------------------------------------------------------------- #
# 离线后端
# --------------------------------------------------------------------------- #
class MockBrain(Brain):
    """脚本化策略：按关键词决定调哪个工具，然后给出一段说明性答案。

    **它不是 Agent 的智能部分**，只在没有 API Key 时保证：
    - CLI / API / 轨迹落盘 / 防护机制 / 评估流程都能被完整验证；
    - 单元测试不依赖网络与密钥。

    任何评估结论都不会来自它——评估报告里会标注 brain=mock。
    """

    label = "mock-scripted"

    # 关键词 → 工具调用（顺序即优先级）
    RULES: list[tuple[tuple[str, ...], str, dict[str, Any]]] = [
        (("课程", "信息增益", "基尼", "决策树", "SVM", "支持向量", "聚类", "K-means",
          "过拟合", "交叉验证", "正则", "神经网络", "梯度", "集成", "PCA", "作业", "实验", "考试"),
         "search_course_kb", {"query": "{task}", "top_k": 3}),
        (("最新", "新闻", "今年", "2024", "2025", "2026", "现在", "发布", "版本"),
         "web_search", {"query": "{task}", "max_results": 3}),
        (("计算", "多少", "求和", "等于", "平方", "开方"),
         "calculator", {"expression": "1+1"}),
    ]

    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self._used: list[str] = []

    def system_message(self, tools: list[ToolSpec]) -> dict[str, str]:
        return {
            "role": "system",
            "content": "（mock 策略：这不是真实模型决策，仅用于离线演示工具链路）",
        }

    def decide(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Decision:
        names = {t.name for t in tools}
        task = ""
        for msg in messages:
            if msg.get("role") == "user":
                task = str(msg.get("content", ""))
                break
        observations = [m for m in messages if m.get("role") == "tool"]

        # 已经有工具结果 → 直接收尾（模拟「信息够了就回答」）
        if observations:
            last = observations[-1]
            body = str(last.get("content", ""))[:1200]
            answer = (
                "（离线 mock 模式，非真实模型生成；以下为工具返回的原始信息，"
                "配置 LLM_API_KEY 后可获得真正的推理与归纳）\n\n"
                f"{body}"
            )
            return Decision(kind="final", final_answer=answer)

        # 首轮：按关键词挑工具
        for keywords, tool_name, args_template in self.RULES:
            if tool_name not in names:
                continue
            if any(k.lower() in task.lower() for k in keywords):
                args = {
                    k: (task if v == "{task}" else v) for k, v in args_template.items()
                }
                self._used.append(tool_name)
                return Decision(
                    kind="tool",
                    thought=f"（mock）任务包含关键词，试用 {tool_name}",
                    tool_calls=[
                        ToolCall(
                            name=tool_name, arguments=args, call_id=f"mock-{len(self._used)}"
                        )
                    ],
                )

        # 没有匹配规则：用时间工具兜一下，保证工具链路仍被覆盖
        if "current_time" in names and "current_time" not in self._used:
            self._used.append("current_time")
            return Decision(
                kind="tool",
                thought="（mock）无匹配规则，调用时间工具以验证链路",
                tool_calls=[
                    ToolCall(name="current_time", arguments={}, call_id="mock-time")
                ],
            )

        return Decision(
            kind="final",
            final_answer=(
                "（离线 mock 模式）没有可用工具能回答该问题，也无法进行真实推理。\n"
                "请在 .env 中配置 LLM_API_KEY 后重试，以获得真实的 Agent 决策能力。"
            ),
        )


def build_brain(cfg: BrainSettings) -> Brain:
    """按配置选择后端；缺 Key 或显式指定 mock 时走脚本化策略。"""
    if cfg.backend == "mock" or not cfg.api_key:
        return MockBrain()
    return OpenAIBrain(cfg)
