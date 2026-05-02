"""
rag_eval — Rigorous evaluation engine for Research QA RAG systems.

Quick start
-----------
from rag_eval import RAGEvaluator, EvalInput

evaluator = RAGEvaluator()

# Single sample
result = evaluator.evaluate(
    question  = "What is the attention mechanism in transformers?",
    answer    = "The attention mechanism allows...",
    context   = "Vaswani et al. (2017) introduced...",   # optional
    reference = "Attention computes weighted...",          # optional
)

print(f"Overall: {result.overall_score:.3f}")
for m in result.metrics:
    print(f"  {m.name.value}: {m.score:.3f} — {m.reasoning}")

# Batch
report = evaluator.evaluate_batch(list_of_eval_inputs)
print(report.summary_table())

# Export
import json
with open("eval_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
"""

from .evaluator import RAGEvaluator
from .schemas import (
    EvalInput, EvalResult, EvalReport, MetricResult, MetricName,
)
from .metrics import (
    ALL_METRICS,
    AnswerRelevanceMetric, FaithfulnessMetric, CorrectnessMetric,
    CompletenessMetric, HallucinationMetric, ConcisenessMetric, Metric,
)
from .judge import LLMJudge

__all__ = [
    "RAGEvaluator",
    "EvalInput", "EvalResult", "EvalReport", "MetricResult", "MetricName",
    "ALL_METRICS",
    "AnswerRelevanceMetric", "FaithfulnessMetric", "CorrectnessMetric",
    "CompletenessMetric", "HallucinationMetric", "ConcisenessMetric", "Metric",
    "LLMJudge",
]