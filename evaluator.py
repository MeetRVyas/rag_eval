"""
RAGEvaluator — the main entry point for the evaluation engine.

Supports multiple LLM providers as judge:  anthropic (default), google,
openai, groq, ollama.

Usage
-----
from rag_eval import RAGEvaluator, EvalInput

# Anthropic (original behaviour — unchanged)
evaluator = RAGEvaluator()
evaluator = RAGEvaluator(provider="anthropic", model="claude-opus-4-6")

# Google Gemini
evaluator = RAGEvaluator(provider="google", model="gemini-2.5-pro")

# OpenAI
evaluator = RAGEvaluator(provider="openai")

# Groq (fast inference)
evaluator = RAGEvaluator(provider="groq", model="llama-3.3-70b-versatile")

# Ollama (local)
evaluator = RAGEvaluator(provider="ollama", model="llama3.2")
evaluator = RAGEvaluator(provider="ollama", ollama_base_url="http://192.168.1.10:11434")

# Single evaluation
result = evaluator.evaluate(
    question = "What causes transformer attention to scale quadratically?",
    answer   = "...",
    context  = "...",    # optional — enables faithfulness & hallucination
    reference= "...",    # optional — enables correctness
)

# Batch evaluation
report = evaluator.evaluate_batch(inputs)
print(report.summary_table())
"""
from __future__ import annotations

import time
import concurrent.futures
from typing import Optional

from .schemas import (
    EvalInput, EvalResult, EvalReport, MetricName, MetricResult
)
from .providers import create_judge, Provider
from .providers.base import BaseJudge
from .metrics import (
    ALL_METRICS,
    AnswerRelevanceMetric, FaithfulnessMetric, CorrectnessMetric,
    CompletenessMetric, HallucinationMetric, ConcisenessMetric,
)

_DEFAULT_WEIGHTS: dict[MetricName, float] = {
    MetricName.ANSWER_RELEVANCE: 0.25,
    MetricName.FAITHFULNESS:     0.20,
    MetricName.CORRECTNESS:      0.20,
    MetricName.COMPLETENESS:     0.20,
    MetricName.HALLUCINATION:    0.10,
    MetricName.CONCISENESS:      0.05,
}


class RAGEvaluator:
    """
    Evaluates RAG system outputs using an LLM-as-judge approach.

    Parameters
    ----------
    provider        : LLM provider to use as judge.
                      One of: "anthropic" (default), "google", "openai",
                      "groq", "ollama"  — or the Provider enum.
    model           : Model identifier for the chosen provider.
                      Omit to use each provider's recommended default.
    api_key         : Provider API key. Falls back to the standard env-var
                      for each provider when not supplied:
                        anthropic → ANTHROPIC_API_KEY
                        google    → GOOGLE_API_KEY  (or GEMINI_API_KEY)
                        openai    → OPENAI_API_KEY
                        groq      → GROQ_API_KEY
                        ollama    → (no key required)
    temperature     : Judge temperature. 0 = deterministic judgments.
    metrics         : List of metric instances. Defaults to ALL_METRICS.
    weights         : Dict[MetricName, float] custom weights for overall score.
                      Skipped metrics are excluded from the weighted mean.
    max_workers     : Thread pool size for parallel batch evaluation.
    ollama_base_url : Ollama server URL (only used when provider="ollama").
                      Default: "http://localhost:11434"
    """

    def __init__(
        self,
        provider:        str | Provider = Provider.ANTHROPIC,
        model:           Optional[str] = None,
        api_key:         Optional[str] = None,
        temperature:     float = 0.0,
        metrics:         Optional[list] = None,
        weights:         Optional[dict[MetricName, float]] = None,
        max_workers:     int = 4,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self._judge = create_judge(
            provider        = provider,
            model           = model,
            api_key         = api_key,
            temperature     = temperature,
            ollama_base_url = ollama_base_url,
        )
        self._metrics     = metrics if metrics is not None else ALL_METRICS
        self._weights     = weights if weights is not None else _DEFAULT_WEIGHTS
        self._max_workers = max_workers

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        question:  str,
        answer:    str,
        context:   Optional[str | list[str]] = None,
        reference: Optional[str] = None,
        metadata:  Optional[dict] = None,
    ) -> EvalResult:
        """
        Evaluate a single question/answer pair.

        Returns an EvalResult with per-metric scores and a weighted overall score.
        """
        inp = EvalInput(
            question  = question,
            answer    = answer,
            context   = context,
            reference = reference,
            metadata  = metadata or {},
        )
        return self._run(inp)

    def evaluate_input(self, inp: EvalInput) -> EvalResult:
        """Evaluate a pre-built EvalInput object."""
        return self._run(inp)

    def evaluate_batch(
        self,
        inputs:        list[EvalInput],
        show_progress: bool = True,
    ) -> EvalReport:
        """
        Evaluate a list of EvalInput objects in parallel.

        Returns an EvalReport with per-result details and aggregate statistics.
        """
        results: list[EvalResult] = []
        total   = len(inputs)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as pool:
            futures = {pool.submit(self._run, inp): i
                       for i, inp in enumerate(inputs)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if show_progress:
                        done = len(results)
                        print(
                            f"  [{done}/{total}] "
                            f"overall={result.overall_score:.3f} "
                            f"| {inputs[idx].question[:60]}..."
                        )
                except Exception as exc:
                    print(f"  [ERROR] sample {idx}: {exc}")

        # Sort back into original submission order
        order = {id(inp): i for i, inp in enumerate(inputs)}
        results.sort(key=lambda r: order.get(id(r.input), 0))

        return EvalReport(
            results=results,
            config={
                "provider":    self._judge.provider,
                "model":       self._judge.model,
                "temperature": self._judge.temperature,
                "metrics":     [m.name.value for m in self._metrics],
                "weights":     {k.value: v for k, v in self._weights.items()},
            },
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def judge(self) -> BaseJudge:
        """The underlying judge instance (useful for custom metrics)."""
        return self._judge

    @property
    def provider(self) -> str:
        return self._judge.provider

    @property
    def model(self) -> str:
        return self._judge.model

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, inp: EvalInput) -> EvalResult:
        t0             = time.perf_counter()
        metric_results = []

        for metric in self._metrics:
            result = metric(inp, self._judge)
            metric_results.append(result)

        overall = self._weighted_mean(metric_results)
        latency = (time.perf_counter() - t0) * 1000

        return EvalResult(
            input         = inp,
            metrics       = metric_results,
            overall_score = overall,
            latency_ms    = latency,
            model         = self._judge.model,
        )

    def _weighted_mean(self, results: list[MetricResult]) -> float:
        """
        Compute a weighted mean, excluding any metric that was skipped
        (those have reasoning starting with "Skipped").
        """
        total_weight = 0.0
        total_score  = 0.0

        for r in results:
            if r.reasoning.startswith("Skipped"):
                continue
            w             = self._weights.get(r.name, 1.0)
            total_score  += r.score * w
            total_weight += w

        return 0.0 if total_weight == 0 else total_score / total_weight

    def __repr__(self) -> str:
        return (
            f"RAGEvaluator(provider={self.provider!r}, model={self.model!r}, "
            f"metrics={len(self._metrics)}, max_workers={self._max_workers})"
        )
