"""
Data models for the RAG evaluation engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class MetricName(str, Enum):
    ANSWER_RELEVANCE   = "answer_relevance"
    FAITHFULNESS       = "faithfulness"
    CORRECTNESS        = "correctness"
    COMPLETENESS       = "completeness"
    HALLUCINATION      = "hallucination"
    CONCISENESS        = "conciseness"


@dataclass
class EvalInput:
    """
    Represents a single evaluation sample.

    Required:
        question    – the user question sent to the RAG system
        answer      – the bot's response

    Optional (unlock additional metrics when provided):
        context     – retrieved chunks / passages used to generate the answer
                      (enables: faithfulness, hallucination)
        reference   – ground-truth / ideal answer
                      (enables: correctness)
        metadata    – arbitrary key/value bag for tracing / grouping
    """
    question:  str
    answer:    str
    context:   Optional[str | list[str]] = None
    reference: Optional[str] = None
    metadata:  dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise context to a single string for prompts
        if isinstance(self.context, list):
            self.context = "\n\n---\n\n".join(self.context)


@dataclass
class MetricResult:
    """Score + reasoning for one metric."""
    name:       MetricName
    score:      float          # 0.0 – 1.0
    reasoning:  str
    raw:        dict[str, Any] = field(default_factory=dict)   # full LLM JSON

    def passed(self, threshold: float = 0.5) -> bool:
        return self.score >= threshold


@dataclass
class EvalResult:
    """Aggregated result for one EvalInput."""
    input:          EvalInput
    metrics:        list[MetricResult]
    overall_score:  float          # weighted mean
    latency_ms:     float = 0.0
    model:          str = ""
    timestamp:      float = field(default_factory=time.time)

    # ── Convenience helpers ──────────────────────────────────────────────────

    def get(self, name: MetricName) -> Optional[MetricResult]:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def passed(self, threshold: float = 0.5) -> bool:
        return self.overall_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "question":      self.input.question,
            "answer":        self.input.answer,
            "overall_score": round(self.overall_score, 4),
            "passed":        self.passed(),
            "latency_ms":    round(self.latency_ms, 1),
            "model":         self.model,
            "timestamp":     self.timestamp,
            "metrics": {
                m.name.value: {
                    "score":     round(m.score, 4),
                    "reasoning": m.reasoning,
                }
                for m in self.metrics
            },
            "metadata": self.input.metadata,
        }


@dataclass
class EvalReport:
    """Aggregated report over a batch of EvalResults."""
    results:      list[EvalResult]
    config:       dict[str, Any] = field(default_factory=dict)
    created_at:   float = field(default_factory=time.time)

    # ── Aggregate stats ──────────────────────────────────────────────────────

    @property
    def mean_overall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    def metric_mean(self, name: MetricName) -> Optional[float]:
        scores = [
            m.score
            for r in self.results
            for m in r.metrics
            if m.name == name
        ]
        return (sum(scores) / len(scores)) if scores else None

    def pass_rate(self, threshold: float = 0.5) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed(threshold) for r in self.results) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        metric_names = {m.name for r in self.results for m in r.metrics}
        return {
            "summary": {
                "n":            len(self.results),
                "mean_overall": round(self.mean_overall, 4),
                "pass_rate":    round(self.pass_rate(), 4),
                "metrics": {
                    n.value: round(self.metric_mean(n), 4)
                    for n in metric_names
                    if self.metric_mean(n) is not None
                },
            },
            "config":     self.config,
            "created_at": self.created_at,
            "results":    [r.to_dict() for r in self.results],
        }

    def summary_table(self) -> str:
        """Pretty-print a concise summary table."""
        metric_names = sorted({m.name for r in self.results for m in r.metrics},
                               key=lambda x: x.value)
        lines = [
            "╔══════════════════════════════════════╗",
            "║       RAG Evaluation  Summary        ║",
            "╚══════════════════════════════════════╝",
            f"  Samples evaluated : {len(self.results)}",
            f"  Overall score     : {self.mean_overall:.3f}",
            f"  Pass rate (≥0.5)  : {self.pass_rate():.1%}",
            "",
            "  Per-metric averages:",
        ]
        for name in metric_names:
            mean = self.metric_mean(name)
            bar  = "█" * int((mean or 0) * 20)
            lines.append(f"    {name.value:<22} {mean:.3f}  {bar}")
        return "\n".join(lines)
