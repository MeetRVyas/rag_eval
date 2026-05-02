"""
rag_eval — Rigorous evaluation engine for Research QA RAG systems.

Multi-provider LLM-as-judge: anthropic (default), google, openai, groq, ollama.

Quick start
-----------
from rag_eval import RAGEvaluator, EvalInput

# Default (Anthropic)
evaluator = RAGEvaluator()

# Other providers
evaluator = RAGEvaluator(provider="google",    model="gemini-2.5-pro")
evaluator = RAGEvaluator(provider="openai")
evaluator = RAGEvaluator(provider="groq",      model="llama-3.3-70b-versatile")
evaluator = RAGEvaluator(provider="ollama",    model="llama3.2")

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

# List all providers + models
from rag_eval import list_models
list_models()              # all providers
list_models("groq")        # one provider
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
from .providers import (
    Provider, PROVIDER_MODELS, create_judge,
    default_model, list_models,
)
from .providers.base import BaseJudge
from .providers.anthropic import AnthropicJudge
from .providers.google import GoogleJudge
from .providers.openai_compat import OpenAIJudge, GroqJudge, OllamaJudge

__all__ = [
    # Core
    "RAGEvaluator",
    # Schemas
    "EvalInput", "EvalResult", "EvalReport", "MetricResult", "MetricName",
    # Metrics
    "ALL_METRICS",
    "AnswerRelevanceMetric", "FaithfulnessMetric", "CorrectnessMetric",
    "CompletenessMetric", "HallucinationMetric", "ConcisenessMetric", "Metric",
    # Providers
    "Provider", "PROVIDER_MODELS", "create_judge",
    "default_model", "list_models",
    "BaseJudge",
    "LLMJudge",           # backward compat alias for AnthropicJudge
    "AnthropicJudge",
    "GoogleJudge",
    "OpenAIJudge",
    "GroqJudge",
    "OllamaJudge",
]
