"""评估执行器：跑检索指标 + 消融实验 + 生成质量，输出 JSON / Markdown 报告。

两个入口：

1. :func:`run_retrieval_eval` —— **不需要任何 API Key**，只要建好索引就能跑，
   对同一套标注问题，换不同检索模式/切分参数，得到可对比的指标表。
   这是整个项目最能体现工程能力的地方：用数据证明「混合检索 + 重排」到底带来多少提升。
2. :func:`run_generation_eval` —— 需要 API Key，用大模型裁判给忠实度/相关性打分。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import EVAL_DIR, settings
from app.eval.judge import JudgeScore, judge_answer
from app.eval.metrics import aggregate, dedup_docs, score_sample
from app.llm.generator import build_context, build_messages, get_chat_client
from app.pipeline.rag import RAGPipeline
from app.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_KS = (1, 3, 5, 10)

# 消融实验用到的检索模式：(标签, 检索模式, 是否重排, 融合方式)
ABLATION_MODES: list[tuple[str, str, bool, str]] = [
    ("仅向量检索 (dense)", "dense", False, "rrf"),
    ("仅 BM25 (sparse)", "sparse", False, "rrf"),
    ("混合检索 RRF (hybrid)", "hybrid", False, "rrf"),
    ("混合检索 加权分数融合 (hybrid)", "hybrid", False, "weighted"),
    ("混合检索 + 重排 (hybrid_rerank)", "hybrid_rerank", True, "rrf"),
]


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@dataclass
class EvalSample:
    """一条标注样本。"""

    id: str
    question: str
    ground_truth: str = ""
    relevant_sources: list[str] = field(default_factory=list)   # 相对 data/raw 的路径
    # 更严格的片段级标签：[(source_path, section_path), ...]
    # 给了它就按「章节命中」判定，比文档级更能反映真实质量（见 eval_set_chunk.jsonl 的说明）
    relevant_sections: list[tuple[str, str]] = field(default_factory=list)
    should_refuse: bool = False                                  # 是否属于「库里没有」的越界问题
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)

    @property
    def granularity(self) -> str:
        return "section" if self.relevant_sections else "document"


def load_dataset(path: Path | None = None, base_path: Path | None = None) -> list[EvalSample]:
    """从 JSONL 读评估集，每行一个样本。

    若当前评估集缺少 ``ground_truth`` / ``difficulty``（片段级评估集就是这样），
    会自动从 ``base_path``（默认 ``eval_set.jsonl``）按 ``id`` 补齐，
    避免为同一批问题重复维护标签。
    """
    path = Path(path or EVAL_DIR / "eval_set.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在：{path}")

    base: dict[str, dict] = {}
    base_path = Path(base_path) if base_path else EVAL_DIR / "eval_set.jsonl"
    if base_path.exists() and base_path.resolve() != path.resolve():
        for line in base_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            base[item.get("id", "")] = item

    samples: list[EvalSample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评估集第 {line_no} 行不是合法 JSON：{exc}") from exc
        fallback = base.get(item.get("id", ""), {})
        sections = [
            (str(pair[0]), str(pair[1]))
            for pair in item.get("relevant_sections", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        samples.append(
            EvalSample(
                id=item.get("id", f"q{line_no:03d}"),
                question=item["question"],
                ground_truth=item.get("ground_truth") or fallback.get("ground_truth", ""),
                relevant_sources=item.get("relevant_sources") or fallback.get("relevant_sources", []),
                relevant_sections=sections,
                should_refuse=bool(item.get("should_refuse", fallback.get("should_refuse", False))),
                difficulty=item.get("difficulty") or fallback.get("difficulty", "medium"),
                tags=list(item.get("tags") or fallback.get("tags", [])),
            )
        )
    return samples


# --------------------------------------------------------------------------- #
# 检索评估
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalReport:
    mode: str
    mode_label: str
    samples: int
    metrics: dict[str, float]
    avg_latency_ms: float
    rerank_used: bool
    granularity: str = "document"
    fusion_mode: str = "rrf"
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


def evaluate_mode(
    retriever: HybridRetriever,
    samples: list[EvalSample],
    mode: str,
    use_rerank: bool = False,
    top_k: int = 10,
    ks: Iterable[int] = DEFAULT_KS,
    fusion_mode: str = "rrf",
) -> RetrievalReport:
    """在给定检索模式下跑完整评估集。

    ``fusion_mode`` 用于对照两种融合策略：
    - ``rrf``      只用名次（对单路强信号不敏感）
    - ``weighted`` 分数归一化后加权求和（保留强度信息）
    """
    rows: list[dict[str, float]] = []
    per_sample: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    rerank_used = False
    granularity = samples[0].granularity if samples else "document"

    for sample in samples:
        hits, trace = retriever.search(
            sample.question,
            mode=mode,          # type: ignore[arg-type]
            top_k=top_k,
            use_rerank=use_rerank,
            fusion_mode=fusion_mode,    # type: ignore[arg-type]
        )
        rerank_used = rerank_used or trace.rerank_used
        latencies.append(trace.total_ms)

        if sample.should_refuse:
            # 越界问题不进检索指标（没有相关文档可算命中率），
            # 单独统计「是否正确拒答」在生成评估里
            continue

        if sample.relevant_sections:
            # 片段级：必须命中「同一文档的同一章节」才算相关
            relevant = {f"{p}::{s}" for p, s in sample.relevant_sections}
            ranked = [f"{h.chunk.source_path}::{h.chunk.section_path}" for h in hits]
            display = [h.chunk.section_path or h.chunk.source_path for h in hits]
        else:
            relevant = set(sample.relevant_sources)
            ranked = [h.chunk.source_path for h in hits]
            display = list(ranked)

        scores = score_sample(ranked, relevant, ks=tuple(ks))
        rows.append(scores)
        per_sample.append(
            {
                "id": sample.id,
                "question": sample.question,
                "first_hit_rank": int(scores["first_hit_rank"]),
                "hit@5": scores["hit@5"],
                "mrr": scores["mrr"],
                "top_docs": dedup_docs(display)[:5],
            }
        )
        if scores["hit@5"] == 0.0:
            failures.append(
                {
                    "id": sample.id,
                    "question": sample.question,
                    "expected": [s for _, s in sample.relevant_sections] or sample.relevant_sources,
                    "got": dedup_docs(display)[:5],
                }
            )

    metrics = aggregate(rows)
    metrics["samples"] = float(len(rows))
    return RetrievalReport(
        mode=mode,
        mode_label=mode,
        samples=len(rows),
        metrics=metrics,
        avg_latency_ms=round(statistics.mean(latencies), 1) if latencies else 0.0,
        rerank_used=rerank_used,
        granularity=granularity,
        per_sample=per_sample,
        failures=failures,
    )


def run_retrieval_eval(
    retriever: HybridRetriever,
    samples: list[EvalSample],
    modes: list[tuple] | None = None,
    top_k: int = 10,
) -> list[RetrievalReport]:
    """跑消融实验：同一份评估集，逐个检索模式对比。

    每个条目形如 ``(标签, 检索模式, 是否重排[, 融合方式])``，
    融合方式缺省为 rrf（兼容旧的两元素/三元素写法）。
    """
    modes = modes or ABLATION_MODES
    reports: list[RetrievalReport] = []
    for entry in modes:
        # 兼容 (标签, 模式, 重排) 与 (标签, 模式, 重排, 融合方式) 两种写法
        label, mode, use_rerank = entry[0], entry[1], entry[2]
        fusion_mode = entry[3] if len(entry) > 3 else "rrf"
        started = time.perf_counter()
        report = evaluate_mode(
            retriever, samples, mode, use_rerank, top_k=top_k, fusion_mode=fusion_mode
        )
        report.mode_label = label
        report.fusion_mode = fusion_mode
        reports.append(report)
        print(
            f"  [{label:<32}] Hit@5={report.metrics.get('hit@5', 0):.3f} "
            f"MRR={report.metrics.get('mrr', 0):.3f} "
            f"Recall@5={report.metrics.get('recall@5', 0):.3f} "
            f"耗时 {time.perf_counter() - started:.1f}s"
        )
    return reports


# --------------------------------------------------------------------------- #
# 生成评估
# --------------------------------------------------------------------------- #
@dataclass
class GenerationReport:
    samples: int
    judged: int
    faithfulness: float
    answer_relevance: float
    citation_correctness: float
    refusal_accuracy: float
    avg_latency_ms: float
    hallucinations: list[dict[str, Any]] = field(default_factory=list)
    # 单条样本的失败记录（生成失败 / 裁判失败），不影响整轮评估
    failures: list[dict[str, Any]] = field(default_factory=list)
    per_sample: list[dict[str, Any]] = field(default_factory=list)


def run_generation_eval(
    pipeline: RAGPipeline,
    samples: list[EvalSample],
    limit: int | None = None,
    top_k: int = 5,
) -> GenerationReport:
    """端到端评估：检索 → 生成 → 裁判打分。需要 LLM_API_KEY。

    **单条样本失败不会中断整轮**：网络超时、模型偶发报错都只影响该条，
    记录原因后继续跑。否则一次抖动就会让几十条样本、几十分钟的计算全部白费
    （这是实际踩过的坑）。
    """
    client = get_chat_client()
    selected = samples[:limit] if limit else samples
    judged_scores: list[JudgeScore] = []
    per_sample: list[dict[str, Any]] = []
    hallucinations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []

    # 拒答准确率：越界问题应当拒答，正常问题不应拒答
    refusal_hits = 0
    refusal_total = 0

    for sample in selected:
        try:
            answer = pipeline.answer(sample.question, top_k=top_k)
        except Exception as exc:
            failures.append({"id": sample.id, "stage": "generate", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{sample.id}] ✘ 生成失败：{type(exc).__name__}: {str(exc)[:90]}")
            continue
        latencies.append(answer.latency_ms)

        refusal_total += 1
        if sample.should_refuse == answer.refused:
            refusal_hits += 1

        context, _ = build_context(answer.contexts)
        if client is not None and client.model != "extractive-fallback":
            score = judge_answer(sample.question, context, answer.answer, answer.refused, client)
        else:
            score = JudgeScore(ok=False, reason="未配置 LLM_API_KEY，跳过裁判打分")
        if score.ok:
            judged_scores.append(score)
            for claim in score.hallucinated_claims:
                hallucinations.append({"id": sample.id, "claim": claim, "reason": score.reason})
        else:
            failures.append({"id": sample.id, "stage": "judge", "error": score.reason})

        per_sample.append(
            {
                "id": sample.id,
                "question": sample.question,
                "answer": answer.answer,
                "refused": answer.refused,
                "should_refuse": sample.should_refuse,
                "model": answer.model,
                "latency_ms": answer.latency_ms,
                "citations": [c.__dict__ for c in answer.citations],
                "judge": asdict(score),
            }
        )
        flag = "拒答" if answer.refused else "作答"
        print(
            f"  [{sample.id}] {flag} 忠实度={score.faithfulness:.0f} "
            f"相关性={score.answer_relevance:.0f} 引用={score.citation_correctness:.0f} "
            f"({answer.latency_ms:.0f}ms)"
        )

    def mean(attr: str) -> float:
        if not judged_scores:
            return 0.0
        return round(sum(getattr(s, attr) for s in judged_scores) / len(judged_scores), 3)

    return GenerationReport(
        samples=len(selected),
        judged=len(judged_scores),
        faithfulness=mean("faithfulness"),
        answer_relevance=mean("answer_relevance"),
        citation_correctness=mean("citation_correctness"),
        refusal_accuracy=round(refusal_hits / refusal_total, 3) if refusal_total else 0.0,
        avg_latency_ms=round(statistics.mean(latencies), 1) if latencies else 0.0,
        hallucinations=hallucinations,
        failures=failures,
        per_sample=per_sample,
    )


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def report_to_markdown(
    retrieval_reports: list[RetrievalReport],
    generation: GenerationReport | None = None,
    dataset_size: int = 0,
    extra: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = ["# CourseRAG 评估报告", ""]
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 知识库：{settings.knowledge_base_name}")
    lines.append(f"- 评估集规模：{dataset_size} 条（含越界拒答样本）")
    if retrieval_reports:
        g = retrieval_reports[0].granularity
        lines.append(
            "- 判定粒度："
            + ("**片段级（章节命中）**" if g == "section" else "文档级（同一文档即算命中）")
        )
    lines.append(f"- 切分策略：{settings.chunk.strategy} / size={settings.chunk.chunk_size} / overlap={settings.chunk.chunk_overlap}")
    lines.append(f"- 向量模型：{settings.embedding.provider}:{settings.embedding.model}")
    lines.append(f"- 重排模型：{settings.rerank.model if settings.rerank.enabled else '未启用'}")
    lines.append("")

    lines.append("## 一、检索消融实验")
    lines.append("")
    max_hit5 = max((r.metrics.get("hit@5", 0.0) for r in retrieval_reports), default=0.0)
    if max_hit5 >= 0.98:
        lines.append(
            "> 说明：本轮的 `Hit@5` 已接近饱和（知识库文档数少时容易如此），"
            "**此时真正有区分度的是 `Hit@1` 与 `MRR`**（正确答案有没有排在第一位）。"
        )
    else:
        lines.append(
            f"> 说明：本轮 `Hit@5` 最高为 {max_hit5:.3f}，未饱和，"
            "因此 `Hit@5` / `MRR` / `Recall@5` 都具备区分度。"
        )
    lines.append("")
    keys = ["hit@1", "hit@3", "hit@5", "mrr", "recall@5", "ndcg@5", "precision@5"]
    header = "| 检索方案 | " + " | ".join(keys) + " | 平均耗时(ms) |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines.extend([header, sep])
    for r in retrieval_reports:
        cells = " | ".join(f"{r.metrics.get(k, 0):.3f}" for k in keys)
        lines.append(f"| {r.mode_label} | {cells} | {r.avg_latency_ms:.0f} |")
    lines.append("")

    # 基线：优先与「单路检索」比较，因为两段式方案的价值就是相对单路召回的精排收益
    baseline = next(
        (r for r in retrieval_reports if r.mode in {"dense", "sparse"}),
        retrieval_reports[0],
    )
    best = max(retrieval_reports, key=lambda r: (r.metrics.get("mrr", 0), r.metrics.get("hit@1", 0)))
    lines.append("### 相对基线的提升")
    lines.append("")
    lines.append(f"- 基线：**{baseline.mode_label}**")
    lines.append(f"- 最优（按 MRR / Hit@1）：**{best.mode_label}**")
    lines.append("")
    lines.append("| 指标 | 基线 | 最优 | 变化 | 延迟(ms) |")
    lines.append("|---|---|---|---|---|")
    for k in ("hit@1", "mrr", "ndcg@5", "recall@5"):
        b, o = baseline.metrics.get(k, 0), best.metrics.get(k, 0)
        delta = f"{(o - b) / b * 100:+.1f}%" if b > 0 else "—"
        lines.append(f"| {k} | {b:.3f} | {o:.3f} | **{delta}** | {baseline.avg_latency_ms:.0f} → {best.avg_latency_ms:.0f} |")
    lines.append("")
    if best.avg_latency_ms > baseline.avg_latency_ms * 3 and baseline.avg_latency_ms > 0:
        lines.append(
            f"> 权衡提示：最优方案把延迟从 {baseline.avg_latency_ms:.0f} ms 提到 "
            f"{best.avg_latency_ms:.0f} ms（约 {best.avg_latency_ms / baseline.avg_latency_ms:.1f} 倍）。"
            "重排是 CPU 上最贵的一步，可调小 RERANK_CANDIDATE_POOL / RERANK_MAX_CHARS，"
            "或改用云端 rerank 接口（RERANK_PROVIDER=api）来压低延迟。"
        )
        lines.append("")

    if generation:
        lines.append("## 二、生成质量（LLM-as-a-Judge，1~5 分）")
        lines.append("")
        lines.append("| 指标 | 得分 |")
        lines.append("|---|---|")
        lines.append(f"| 忠实度 faithfulness | {generation.faithfulness} |")
        lines.append(f"| 相关性 answer_relevance | {generation.answer_relevance} |")
        lines.append(f"| 引用正确性 citation_correctness | {generation.citation_correctness} |")
        lines.append(f"| 拒答准确率 refusal_accuracy | {generation.refusal_accuracy} |")
        lines.append(f"| 参与打分样本数 | {generation.judged}/{generation.samples} |")
        lines.append(f"| 平均端到端延迟 | {generation.avg_latency_ms:.0f} ms |")
        lines.append("")
        if generation.hallucinations:
            lines.append("### 疑似幻觉片段")
            lines.append("")
            for h in generation.hallucinations[:10]:
                lines.append(f"- `{h['id']}`：{h['claim']}")
            lines.append("")
        else:
            lines.append("未检出幻觉片段。")
            lines.append("")

    lines.append("## 三、错误分析")
    lines.append("")
    lines.append("### 未把正确文档排在第一位的样本（按方案统计）")
    lines.append("")
    for r in retrieval_reports:
        missed = [s for s in r.per_sample if s.get("first_hit_rank", 0) != 1]
        lines.append(f"**{r.mode_label}**：{len(missed)}/{len(r.per_sample)} 条未命中第 1 位")
        if missed:
            lines.append("")
            lines.append("| 样本 | 问题 | 首次命中名次 | 实际 Top3 |")
            lines.append("|---|---|---|---|")
            for s in missed[:6]:
                lines.append(
                    f"| {s['id']} | {s['question'][:34]} | {s['first_hit_rank']} | "
                    f"{', '.join(d[:22] for d in s['top_docs'][:3])} |"
                )
        lines.append("")

    failures = [(r.mode_label, f) for r in retrieval_reports for f in r.failures]
    lines.append("### 完全未召回（Hit@5 = 0）的案例")
    lines.append("")
    if failures:
        lines.append("| 方案 | 问题 | 期望文档 | 实际召回 |")
        lines.append("|---|---|---|---|")
        for label, f in failures[:15]:
            lines.append(
                f"| {label} | {f['question'][:30]} | {', '.join(f['expected'])[:30]} | "
                f"{', '.join(f['got'])[:50]} |"
            )
    else:
        lines.append("本轮实验中没有出现完全未召回的情况（所有问题的正确文档都进了 top-5）。")
        lines.append("")
        lines.append(
            "这说明当前的失败模式是**排序问题**而不是**召回问题**："
            "正确文档已经进入候选池，但名次不够靠前——这正是引入精排（cross-encoder）的直接动机。"
        )
    lines.append("")

    if extra:
        lines.append("## 四、运行环境")
        lines.append("")
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def save_reports(
    retrieval_reports: list[RetrievalReport],
    generation: GenerationReport | None,
    dataset_size: int,
    out_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
    tag: str = "",
) -> tuple[Path, Path]:
    """同时落盘 JSON（给程序读）与 Markdown（给人读）。"""
    out_dir = Path(out_dir or EVAL_DIR / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    json_path = out_dir / f"report-{stamp}{suffix}.json"
    md_path = out_dir / f"report-{stamp}{suffix}.md"

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": dataset_size,
        "config": settings.describe(),
        "retrieval": [
            {
                "mode": r.mode,
                "label": r.mode_label,
                "samples": r.samples,
                "metrics": r.metrics,
                "avg_latency_ms": r.avg_latency_ms,
                "rerank_used": r.rerank_used,
                "granularity": r.granularity,
                "fusion_mode": r.fusion_mode,
                "failures": r.failures,
                # 逐题明细要落盘：错误分析、复现实验、以及回答「哪类问题排不准」都依赖它
                "per_sample": r.per_sample,
            }
            for r in retrieval_reports
        ],
        "generation": asdict(generation) if generation else None,
        "extra": extra or {},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        report_to_markdown(retrieval_reports, generation, dataset_size, extra),
        encoding="utf-8",
    )
    return json_path, md_path
