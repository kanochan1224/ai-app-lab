"""eval 子包：检索指标、裁判打分、消融实验。"""

from app.eval.judge import JudgeScore, judge_answer
from app.eval.metrics import (
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_sample,
)
from app.eval.runner import (
    ABLATION_MODES,
    EvalSample,
    GenerationReport,
    RetrievalReport,
    evaluate_mode,
    load_dataset,
    report_to_markdown,
    run_generation_eval,
    run_retrieval_eval,
    save_reports,
)

__all__ = [
    "JudgeScore",
    "judge_answer",
    "aggregate",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_sample",
    "ABLATION_MODES",
    "EvalSample",
    "GenerationReport",
    "RetrievalReport",
    "evaluate_mode",
    "load_dataset",
    "report_to_markdown",
    "run_generation_eval",
    "run_retrieval_eval",
    "save_reports",
]
