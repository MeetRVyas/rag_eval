"""
Individual evaluation metrics for the RAG evaluation engine.

Each metric is a callable class with a __call__(input, judge) -> MetricResult.
All prompts follow a strict JSON output contract so scores are deterministic.
"""
from __future__ import annotations

from typing import Protocol

from ..schemas import EvalInput, MetricName, MetricResult
from ..judge import LLMJudge


# ── Metric Protocol ──────────────────────────────────────────────────────────

class Metric(Protocol):
    name: MetricName

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult: ...


# ── Shared prompt scaffold ───────────────────────────────────────────────────

_SYSTEM = """\
You are an expert evaluator for research QA RAG systems. Your role is to assess
the quality of AI-generated answers with rigour and objectivity.

You MUST respond with ONLY a valid JSON object — no prose, no markdown fences,
no commentary outside the JSON.

Required JSON schema (always include all keys):
{
  "score": <float between 0.0 and 1.0, two decimal places>,
  "reasoning": "<concise 1-3 sentence explanation of the score>",
  "evidence": "<specific excerpt from the answer that most influenced your score>"
}

Scoring scale:
  1.0 = perfect / fully satisfies the criterion
  0.8 = good, minor gaps
  0.6 = acceptable, notable gaps
  0.4 = poor, significant problems
  0.2 = very poor
  0.0 = completely fails
"""


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ── Metric 1: Answer Relevance ───────────────────────────────────────────────

class AnswerRelevanceMetric:
    """Does the answer actually address the question asked?"""
    name = MetricName.ANSWER_RELEVANCE

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
        prompt = f"""\
Evaluate how well the ANSWER addresses the QUESTION.

QUESTION:
{inp.question}

ANSWER:
{inp.answer}

Criterion — Answer Relevance:
- Does the answer directly respond to what was asked?
- Is it on-topic throughout, or does it drift into unrelated content?
- A high score requires the answer to be clearly and specifically responsive.

Respond with JSON only."""

        raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
        return MetricResult(
            name=self.name, score=_clamp(raw["score"]),
            reasoning=raw["reasoning"], raw=raw,
        )


# ── Metric 2: Faithfulness ───────────────────────────────────────────────────

class FaithfulnessMetric:
    """Are all claims in the answer supported by the retrieved context?"""
    name = MetricName.FAITHFULNESS

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
        if not inp.context:
            return MetricResult(name=self.name, score=0.0,
                                reasoning="Skipped — no context provided.")

        prompt = f"""\
Evaluate the faithfulness of the ANSWER with respect to the CONTEXT.

CONTEXT (retrieved passages):
{inp.context}

QUESTION:
{inp.question}

ANSWER:
{inp.answer}

Criterion — Faithfulness:
- Every factual claim in the answer must be directly supported by the context.
- Claims that go beyond the context (even if plausibly true) lower the score.
- Contradicting the context is a critical failure (score <= 0.2).

Respond with JSON only."""

        raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
        return MetricResult(name=self.name, score=_clamp(raw["score"]),
                            reasoning=raw["reasoning"], raw=raw)


# ── Metric 3: Correctness ────────────────────────────────────────────────────

class CorrectnessMetric:
    """Semantic match between bot answer and a reference answer."""
    name = MetricName.CORRECTNESS

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
        if not inp.reference:
            return MetricResult(name=self.name, score=0.0,
                                reasoning="Skipped — no reference answer provided.")

        prompt = f"""\
Compare the ANSWER to the REFERENCE ANSWER and score correctness.

QUESTION:
{inp.question}

REFERENCE ANSWER (ground truth):
{inp.reference}

BOT ANSWER:
{inp.answer}

Criterion — Correctness:
- Assess factual accuracy and semantic agreement with the reference.
- Wording differences are fine; factual disagreements are not.
- Additional valid information in the answer should not be penalised.
- Penalise missing key facts from the reference proportionally.

Respond with JSON only."""

        raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
        return MetricResult(name=self.name, score=_clamp(raw["score"]),
                            reasoning=raw["reasoning"], raw=raw)


# ── Metric 4: Completeness ───────────────────────────────────────────────────

class CompletenessMetric:
    """Does the answer cover all significant aspects of the question?"""
    name = MetricName.COMPLETENESS

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
        context_block = (f"\nCONTEXT:\n{inp.context}\n" if inp.context else "")
        reference_block = (f"\nREFERENCE:\n{inp.reference}\n" if inp.reference else "")

        prompt = f"""\
Evaluate how completely the ANSWER covers the QUESTION.

QUESTION:
{inp.question}
{context_block}{reference_block}
ANSWER:
{inp.answer}

Criterion — Completeness:
- Identify all distinct aspects or sub-questions in the question.
- Check whether each aspect is addressed in the answer.
- A partial answer that only covers one part of a multi-part question scores low.

Respond with JSON only."""

        raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
        return MetricResult(name=self.name, score=_clamp(raw["score"]),
                            reasoning=raw["reasoning"], raw=raw)


# ── Metric 5: Hallucination ──────────────────────────────────────────────────

class HallucinationMetric:
    """
    Detects fabricated or unsupported specific facts.
    Score: 1.0 = no hallucinations, 0.0 = severe hallucinations.
    """
    name = MetricName.HALLUCINATION

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
        if not inp.context:
            return MetricResult(name=self.name, score=0.0,
                                reasoning="Skipped — no context provided.")

        prompt = f"""\
Detect hallucinations in the ANSWER relative to the CONTEXT.

CONTEXT (retrieved passages):
{inp.context}

QUESTION:
{inp.question}

ANSWER:
{inp.answer}

Criterion — Hallucination (score = 1 - hallucination_severity):
- Identify any specific facts, numbers, citations, or claims in the answer
  that are NOT supported by the context and appear to be fabricated.
- Universal background knowledge is acceptable.
- Specific invented details (wrong statistics, made-up citations) are not.
- Score 1.0 = zero hallucinations. Score 0.0 = substantially fabricated.

Respond with JSON only."""

        raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
        return MetricResult(name=self.name, score=_clamp(raw["score"]),
                            reasoning=raw["reasoning"], raw=raw)


# ── Metric 6: Conciseness ────────────────────────────────────────────────────

class ConcisenessMetric:
    """Is the answer appropriately concise — no padding or repetition?"""
    name = MetricName.CONCISENESS

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
        prompt = f"""\
Evaluate whether the ANSWER is appropriately concise.

QUESTION:
{inp.question}

ANSWER:
{inp.answer}

Criterion — Conciseness:
- The answer should convey necessary information without filler phrases,
  unnecessary repetition, or excessive caveats.
- Longer answers are fine if the question warrants it.
- Penalise redundancy, repetition, or content that adds no value.

Respond with JSON only."""

        raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
        return MetricResult(name=self.name, score=_clamp(raw["score"]),
                            reasoning=raw["reasoning"], raw=raw)


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_METRICS: list = [
    AnswerRelevanceMetric(),
    FaithfulnessMetric(),
    CorrectnessMetric(),
    CompletenessMetric(),
    HallucinationMetric(),
    ConcisenessMetric(),
]
