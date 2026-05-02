"""
RAGEvaluator — the main entry point for the evaluation engine.

Usage
-----
from rag_eval import RAGEvaluator, EvalInput

evaluator = RAGEvaluator()                    # uses env var ANTHROPIC_API_KEY
evaluator = RAGEvaluator(api_key="sk-...")    # explicit key
evaluator = RAGEvaluator(model="claude-opus-4-20250514")

# Single evaluation
result = evaluator.evaluate(
    question = "What causes transformer attention to scale quadratically?",
    answer   = "...",
    context  = "...",          # optional – enables faithfulness & hallucination
    reference= "...",          # optional – enables correctness
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
from .judge import LLMJudge
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
    model       : Judge model (default: claude-sonnet-4-20250514)
    api_key     : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    temperature : Judge temperature. 0 = deterministic.
    metrics     : List of metric instances to use. Defaults to ALL_METRICS.
    weights     : Dict[MetricName, float] custom weights for overall score.
                  Skipped metrics are excluded from the weighted mean.
    max_workers : Thread pool size for parallel batch evaluation.
    """

    def __init__(
        self,
        model:       str = "claude-sonnet-4-20250514",
        api_key:     Optional[str] = None,
        temperature: float = 0.0,
        metrics:     Optional[list] = None,
        weights:     Optional[dict[MetricName, float]] = None,
        max_workers: int = 4,
    ) -> None:
        self._judge      = LLMJudge(model=model, api_key=api_key,
                                    temperature=temperature)
        self._metrics    = metrics if metrics is not None else ALL_METRICS
        self._weights    = weights if weights is not None else _DEFAULT_WEIGHTS
        self._max_workers = max_workers

    # ── Public API ───────────────────────────────────────────────────────────

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
        inputs: list[EvalInput],
        show_progress: bool = True,
    ) -> EvalReport:
        """
        Evaluate a list of EvalInput objects in parallel.

        Returns an EvalReport with per-result details and aggregate statistics.
        """
        results: list[EvalResult] = []
        total = len(inputs)

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
                        print(f"  [{done}/{total}] overall={result.overall_score:.3f} "
                              f"| {inputs[idx].question[:60]}...")
                except Exception as exc:
                    print(f"  [ERROR] sample {idx}: {exc}")

        # Sort back into original order
        order = {id(inp): i for i, inp in enumerate(inputs)}
        results.sort(key=lambda r: order.get(id(r.input), 0))

        return EvalReport(
            results=results,
            config={
                "model":       self._judge.model,
                "temperature": self._judge.temperature,
                "metrics":     [m.name.value for m in self._metrics],
                "weights":     {k.value: v for k, v in self._weights.items()},
            },
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self, inp: EvalInput) -> EvalResult:
        t0 = time.perf_counter()
        metric_results: list[MetricResult] = []

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
        Compute a weighted mean, skipping metrics that were not run
        (score=0 with a 'Skipped' reasoning).
        """
        total_weight = 0.0
        total_score  = 0.0

        for r in results:
            if r.reasoning.startswith("Skipped"):
                continue
            w = self._weights.get(r.name, 1.0)
            total_score  += r.score * w
            total_weight += w

        if total_weight == 0:
            return 0.0
        return total_score / total_weight
