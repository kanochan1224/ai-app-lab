"""第 5 步：生成（Generation）。

设计取舍：
- **不用 LangChain 的 ChatModel 封装**，直接走 OpenAI 兼容的 ``/chat/completions``。
  原因：DeepSeek / 硅基流动 / 通义 / vLLM / Ollama 全是这套协议，一个客户端通吃；
  而流式解析与超时控制自己写反而更清楚，也少一层抽象。
- **强制引用**：提示词要求每个结论后用 ``[编号]`` 标注依据，且只允许引用给定片段。
- **可拒答**：证据不足时要求模型输出固定标记，服务层据此返回「未在课程资料中找到依据」，
  而不是编一个看似合理的答案——这是 RAG 落地时最重要的安全阀。
- **无 Key 降级**：没配 API Key 时退回「抽取式回答」（从召回片段里选最相关的句子），
  保证演示时链路完整、不报错。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from app.config import settings
from app.schema import Citation, RetrievedChunk

# 模型被要求在没有依据时输出这个标记
REFUSAL_MARKER = "NO_ANSWER"

SYSTEM_PROMPT = """你是一位严谨的课程助教，只依据给定的「课程资料片段」回答学生问题。

必须遵守的规则：
1. 只使用资料片段中的信息作答，不要引入片段之外的知识，也不要凭常识补全。
2. 每一条结论后面都要标注来源编号，格式为 [1]、[2]；一句话用到多段就写 [1][3]。
3. 如果资料片段不足以回答问题，只回复 {refusal}，不要解释、不要道歉、不要编造。
4. 回答面向学生，先给直接结论，再给必要的推导或例子；数学公式用 LaTeX 行内写法 $...$。
5. 如果资料中存在相互矛盾的说法，指出矛盾并分别标注来源。
6. 使用中文回答，不要输出「根据提供的资料」这类套话，直接给答案。""".format(
    refusal=REFUSAL_MARKER
)

USER_TEMPLATE = """【课程资料片段】
{context}

【学生问题】
{question}

请依据上面的片段作答，并标注来源编号。"""


# 需要重试的服务端状态码：限流与临时故障；4xx 参数错误不重试
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """调用大模型失败。"""


# --------------------------------------------------------------------------- #
# 上下文拼装
# --------------------------------------------------------------------------- #
def build_context(hits: list[RetrievedChunk], max_chars: int = 6000) -> tuple[str, list[Citation]]:
    """把召回片段拼成带编号的上下文，同时产出引用表。

    编号即答案里的 ``[n]``，所以顺序必须与最终返回给前端的 citations 一致。
    """
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[{i}] 来源：{chunk.source_path}"
        if chunk.section_path:
            header += f" > {chunk.section_path}"
        block = f"{header}\n{chunk.text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
        citations.append(
            Citation(
                index=i,
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                section_path=chunk.section_path,
                snippet=_clip(chunk.text, 160),
            )
        )
    return "\n\n".join(blocks), citations


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def parse_citation_indices(answer: str) -> list[int]:
    """从答案正文里抽出用到的引用编号（去重且保持出现顺序）。"""
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", answer):
        idx = int(m.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def filter_citations(answer: str, citations: list[Citation]) -> list[Citation]:
    """只保留答案真正引用到的来源，避免前端显示一堆没用上的「装饰性引用」。"""
    used = parse_citation_indices(answer)
    if not used:
        return []
    by_index = {c.index: c for c in citations}
    return [by_index[i] for i in used if i in by_index]


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    text: str
    model: str
    usage: dict[str, Any]
    latency_ms: float
    refused: bool = False


class OpenAICompatChat:
    """OpenAI 兼容的对话客户端（DeepSeek / 硅基流动 / 通义 / vLLM / Ollama 通用）。"""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        timeout: int = 60,
        retries: int = 2,
    ) -> None:
        if not api_base:
            raise LLMError("缺少 LLM_API_BASE 配置")
        if not api_key:
            raise LLMError("缺少 LLM_API_KEY 配置")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 连接超时/5xx/429 的重试次数（评估时网络抖动不该毁掉整轮结果）
        self.retries = max(0, retries)
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    def complete(
        self, messages: list[dict[str, str]], retries: int | None = None
    ) -> GenerationResult:
        """发起一次对话补全。

        **带重试**：真实评估里网络抖动一次就丢掉整轮结果（几十条样本、几十分钟）
        是不可接受的，所以对连接超时与 5xx/429 做指数退避重试；
        4xx（参数/鉴权错误）不重试——重试也不会变好，只会浪费时间掩盖问题。
        """
        attempts = self.retries if retries is None else retries
        started = time.perf_counter()
        last_error = ""

        for attempt in range(attempts + 1):
            try:
                resp = httpx.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._headers,
                    json=self._payload(messages, stream=False),
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts:
                    time.sleep(self._backoff(attempt))
                    continue
                raise LLMError(
                    f"请求大模型失败（已重试 {attempts} 次）：{last_error}"
                ) from exc

            if resp.status_code == 200:
                data = resp.json()
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                latency = (time.perf_counter() - started) * 1000
                return GenerationResult(
                    text=text.strip(),
                    model=data.get("model", self.model),
                    usage=data.get("usage", {}) or {},
                    latency_ms=round(latency, 1),
                    refused=REFUSAL_MARKER in text,
                )

            # 可重试的服务端错误
            if resp.status_code in _RETRYABLE_STATUS and attempt < attempts:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(self._backoff(attempt))
                continue
            raise LLMError(f"大模型返回 {resp.status_code}：{resp.text[:300]}")

        raise LLMError(f"请求大模型失败：{last_error}")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """指数退避：1s、2s、4s……（加上抖动避免同时重试）"""
        import random

        return min(2 ** attempt + random.uniform(0, 0.5), 8.0)

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """流式输出：逐 token 让前端有「打字机」体验。"""
        try:
            with httpx.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, stream=True),
                timeout=self.timeout,
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise LLMError(f"大模型返回 {resp.status_code}：{body[:300]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (payload.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"流式请求大模型失败：{exc}") from exc


class ExtractiveAnswerer:
    """无 Key 时的降级回答器：从召回片段中抽取与问题最相关的句子。

    它不生成新内容，因此**不会幻觉**，但也无法做归纳、推理与计算。
    定位是「让整条链路在没有任何密钥时也能演示」，不是替代大模型。

    **拒答能力的边界（重要）**：
    它只能靠词汇重合判断，因此只在「问题与知识库几乎没有任何共同词」时才会拒答
    （例如问食堂几点开门）。对于「用课程术语包装、但库里其实没有答案」的问题
    （例如问某篇并不在库中的论文），它会因为捡到若干共同词而给出片段摘要。
    也就是说 **生成侧的语义拒答闸门只有在配置了 LLM_API_KEY 时才真正生效**。
    这是无 Key 降级模式的固有局限，已在 README 中如实标注。
    """

    model = "extractive-fallback"

    def answer(self, question: str, hits: list[RetrievedChunk]) -> GenerationResult:
        from app.retrieval.tokenize import tokenize

        q_tokens = set(tokenize(question))
        scored: list[tuple[float, str, int]] = []
        for i, hit in enumerate(hits, start=1):
            for sent in re.split(r"(?<=[。！？!?；;])", hit.chunk.text):
                sent = sent.strip()
                if len(sent) < 8:
                    continue
                # FAQ 类资料里的「Q: ...」是提问行，不是答案，抽出来会显得很傻
                if re.match(r"^[Qq]\s*[:：]", sent) or sent.endswith("？") and "Q" in sent[:3]:
                    continue
                s_tokens = set(tokenize(sent))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / (len(q_tokens) + 1e-6)
                if overlap > 0:
                    scored.append((overlap, sent, i))
        if not scored:
            return GenerationResult(
                text=REFUSAL_MARKER,
                model=self.model,
                usage={},
                latency_ms=0.0,
                refused=True,
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        picked: list[tuple[float, str, int]] = []
        seen: set[str] = set()
        for item in scored:
            if item[1] in seen:
                continue
            seen.add(item[1])
            picked.append(item)
            if len(picked) >= 4:
                break
        lines = ["（当前未配置大模型 API Key，以下为检索结果的抽取式摘要，非生成式回答）", ""]
        for _, sent, idx in picked:
            lines.append(f"- {sent} [{idx}]")
        return GenerationResult(
            text="\n".join(lines),
            model=self.model,
            usage={},
            latency_ms=0.0,
        )


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
_CHAT_CACHE: dict[str, OpenAICompatChat] = {}

_PROVIDER_BASES = {
    "deepseek": "https://api.deepseek.com/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}


def resolve_api_base(provider: str, configured: str = "") -> str:
    return configured or _PROVIDER_BASES.get(provider.lower(), "")


def get_chat_client() -> OpenAICompatChat | None:
    """返回对话客户端；未配置 Key 时返回 ``None``（调用方走降级路径）。"""
    cfg = settings.llm
    if not cfg.api_key:
        return None
    key = f"{cfg.provider}:{cfg.model}:{cfg.api_base}"
    if key not in _CHAT_CACHE:
        _CHAT_CACHE[key] = OpenAICompatChat(
            model=cfg.model,
            api_base=resolve_api_base(cfg.provider, cfg.api_base),
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
    return _CHAT_CACHE[key]


def build_messages(question: str, context: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """组装对话消息：系统提示 + 历史（可选）+ 当前问题。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)})
    return messages


def llm_info() -> dict[str, Any]:
    cfg = settings.llm
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "api_base": resolve_api_base(cfg.provider, cfg.api_base),
        "has_key": bool(cfg.api_key),
        "mode": "generative" if cfg.api_key else "extractive-fallback",
    }
