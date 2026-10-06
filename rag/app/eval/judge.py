"""生成质量评估：用大模型当裁判（LLM-as-a-Judge）。

检索指标能算出「有没有找对资料」，但答不出来「答得对不对」。
所以再加三项裁判打分（1~5 分）：

- **faithfulness 忠实度**：答案里的每句话是否都能从给定片段中推出（防幻觉的核心指标）
- **answer_relevance 相关性**：是否正面回答了学生的问题
- **citation_correctness 引用正确性**：``[n]`` 标注的位置是否确实支持该结论

裁判与被测模型是同一个供应商也没关系——关键是提示词要求它**先逐条核对再打分**，
并要求严格 JSON 输出，便于程序解析。没有 API Key 时整段跳过并如实标注。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

JUDGE_SYSTEM = """你是一名严格的 RAG 系统评估专家。你会看到一个学生问题、系统检索到的资料片段、以及系统给出的回答。

请对回答做三项 1~5 分的打分，并严格输出 JSON：

{
  "faithfulness": <int>,
  "answer_relevance": <int>,
  "citation_correctness": <int>,
  "hallucinated_claims": [<回答中无法从片段推出、疑似编造的具体句子>],
  "reason": "<一句话说明扣分原因>"
}

**打分前必须逐句核对**：先把回答拆成若干事实性陈述，逐条回到片段里找依据，
再根据「有多少条找不到依据 / 有多少条答偏」来决定分数。**不允许凭整体印象给分。**

各分值的确切含义（请严格对照，不要轻易给 5 分）：

faithfulness（事实性陈述有多少能在片段中找到依据）
- 5：每一条事实性陈述都能在片段里找到明确依据，没有自行补充
- 4：有 1 条次要陈述缺乏依据，但不影响核心结论
- 3：有多条陈述缺乏依据，或把片段内容做了超出原文的引申
- 2：核心结论缺乏依据，或把不同片段的说法错误地合并
- 1：主要内容为编造

answer_relevance（是否正面、完整地回答了学生的问题）
- 5：正面回答且覆盖问题的全部要点
- 4：正面回答，但遗漏了次要要点
- 3：只答了一部分，或答了相关但非所问的内容
- 2：明显答偏，需要学生再问一次
- 1：答非所问

citation_correctness（[n] 标注是否准确）
- 5：每个 [n] 都标在正确位置，且标注的片段确实支持该结论
- 4：个别引用编号有偏差，但读者仍能定位到正确来源
- 3：引用位置大体正确但存在错配，或关键结论漏标引用
- 2：引用编号与结论明显不符
- 1：引用完全错误

**给 5 分是有门槛的**：只有该维度确实挑不出问题才给 5；
发现任何瑕疵（哪怕很小）都应给 4 分或更低，并把瑕疵写进 reason。
若回答是系统主动拒答（refused=true），三项均记 5 分。

只输出 JSON，不要输出任何解释性文字或 Markdown 代码块。"""

JUDGE_USER = """【学生问题】
{question}

【系统检索到的资料片段】
{context}

【系统的回答】
{answer}

请按上述要求输出 JSON。"""


@dataclass
class JudgeScore:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    citation_correctness: float = 0.0
    hallucinated_claims: list[str] = field(default_factory=list)
    reason: str = ""
    ok: bool = True


def _extract_json(text: str) -> dict[str, Any] | None:
    """模型有时会套一层 ```json 代码块，这里做兼容提取。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def judge_answer(
    question: str,
    context: str,
    answer: str,
    refused: bool = False,
    client=None,
) -> JudgeScore:
    """调用大模型裁判；无法解析时返回 ``ok=False`` 并在汇总里剔除。"""
    if refused:
        # 拒答本身是「正确行为」，不参与扣分
        return JudgeScore(5.0, 5.0, 5.0, reason="系统主动拒答，按规则记满分")

    from app.llm.generator import get_chat_client

    client = client or get_chat_client()
    if client is None:
        return JudgeScore(ok=False, reason="未配置 LLM_API_KEY，无法进行裁判打分")

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                question=question, context=_clip(context, 4000), answer=_clip(answer, 2000)
            ),
        },
    ]
    try:
        result = client.complete(messages)
    except Exception as exc:
        return JudgeScore(ok=False, reason=f"裁判调用失败：{exc}")

    payload = _extract_json(result.text)
    if not payload:
        return JudgeScore(ok=False, reason="裁判输出无法解析为 JSON")

    def num(key: str) -> float:
        try:
            return float(payload.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    return JudgeScore(
        faithfulness=num("faithfulness"),
        answer_relevance=num("answer_relevance"),
        citation_correctness=num("citation_correctness"),
        hallucinated_claims=list(payload.get("hallucinated_claims") or []),
        reason=str(payload.get("reason", "")),
        ok=True,
    )
