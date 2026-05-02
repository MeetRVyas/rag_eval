"""
examples.py — Usage examples for rag_eval.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples.py
"""
import json
import os
import sys

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval import (
    RAGEvaluator, EvalInput, MetricName,
    AnswerRelevanceMetric, FaithfulnessMetric, CorrectnessMetric,
    CompletenessMetric, HallucinationMetric,
)


# ── Sample data ──────────────────────────────────────────────────────────────

CONTEXT = """\
Transformer models rely on a mechanism called self-attention, introduced by
Vaswani et al. in "Attention Is All You Need" (2017). Self-attention allows each
token in a sequence to attend to all other tokens, computing a weighted sum of
value vectors. The computational complexity of self-attention is O(n²·d) where
n is the sequence length and d is the model dimension, making it quadratic in
sequence length. This quadratic scaling is the primary bottleneck for processing
very long documents. Variants such as Sparse Transformers and Longformer
introduce approximate attention to reduce this cost to O(n·log n) or O(n).
"""

REFERENCE = (
    "Self-attention in transformers scales quadratically with sequence length "
    "because every token attends to every other token, resulting in O(n²) "
    "complexity. This makes processing long sequences computationally expensive."
)


# ── Example 1: Single evaluation (full context + reference) ──────────────────

def example_single_full():
    print("\n" + "="*60)
    print("EXAMPLE 1 — Single evaluation (context + reference)")
    print("="*60)

    evaluator = RAGEvaluator()

    result = evaluator.evaluate(
        question  = "Why does transformer attention scale quadratically?",
        answer    = (
            "Transformer self-attention has O(n²) complexity because each of "
            "the n tokens must compute attention scores against all other n "
            "tokens. This quadratic growth becomes a serious bottleneck for "
            "long sequences. Sparse attention variants reduce this cost."
        ),
        context   = CONTEXT,
        reference = REFERENCE,
        metadata  = {"source": "gpt-4o", "version": "2025-01"},
    )

    print(f"\nOverall score : {result.overall_score:.3f}")
    print(f"Latency       : {result.latency_ms:.0f} ms")
    print(f"Model         : {result.model}")
    print("\nPer-metric breakdown:")
    for m in result.metrics:
        skipped = m.reasoning.startswith("Skipped")
        status  = "–" if skipped else f"{m.score:.3f}"
        print(f"  {m.name.value:<22} {status}")
        if not skipped:
            print(f"    ↳ {m.reasoning}")

    return result


# ── Example 2: Evaluation without context/reference ──────────────────────────

def example_single_minimal():
    print("\n" + "="*60)
    print("EXAMPLE 2 — Single evaluation (question + answer only)")
    print("="*60)

    evaluator = RAGEvaluator()

    result = evaluator.evaluate(
        question = "What is RLHF?",
        answer   = (
            "RLHF stands for Reinforcement Learning from Human Feedback. "
            "It is a technique where a language model is fine-tuned using a "
            "reward model trained on human preference data, allowing the model "
            "to better align with human values and instructions."
        ),
    )

    print(f"\nOverall score : {result.overall_score:.3f}")
    for m in result.metrics:
        if not m.reasoning.startswith("Skipped"):
            print(f"  {m.name.value:<22} {m.score:.3f}")

    return result


# ── Example 3: Hallucination detection ───────────────────────────────────────

def example_hallucination():
    print("\n" + "="*60)
    print("EXAMPLE 3 — Hallucination detection")
    print("="*60)

    evaluator = RAGEvaluator()

    # Deliberately inject a hallucinated statistic
    result = evaluator.evaluate(
        question = "What is the complexity of transformer self-attention?",
        answer   = (
            "Transformer self-attention has O(n²) complexity. Studies from "
            "MIT in 2023 showed that on sequences over 100k tokens, attention "
            "consumes 87.3% of GPU memory — a figure that has been validated "
            "across 14 independent benchmarks."
        ),
        context  = CONTEXT,
    )

    h = result.get(MetricName.HALLUCINATION)
    f = result.get(MetricName.FAITHFULNESS)
    print(f"\nHallucination score : {h.score:.3f}")
    print(f"  ↳ {h.reasoning}")
    print(f"\nFaithfulness score  : {f.score:.3f}")
    print(f"  ↳ {f.reasoning}")

    return result


# ── Example 4: Batch evaluation with custom metric subset ────────────────────

def example_batch():
    print("\n" + "="*60)
    print("EXAMPLE 4 — Batch evaluation with custom metrics")
    print("="*60)

    # Only run the metrics we care about
    evaluator = RAGEvaluator(
        metrics=[
            AnswerRelevanceMetric(),
            FaithfulnessMetric(),
            HallucinationMetric(),
        ],
        weights={
            MetricName.ANSWER_RELEVANCE: 0.4,
            MetricName.FAITHFULNESS:     0.4,
            MetricName.HALLUCINATION:    0.2,
        },
        max_workers=3,
    )

    inputs = [
        EvalInput(
            question  = "What is self-attention?",
            answer    = "Self-attention lets each token attend to all others.",
            context   = CONTEXT,
            metadata  = {"id": "q1"},
        ),
        EvalInput(
            question  = "What is the complexity of self-attention?",
            answer    = "It is O(n²) where n is the sequence length.",
            context   = CONTEXT,
            metadata  = {"id": "q2"},
        ),
        EvalInput(
            question  = "Who invented the transformer?",
            answer    = "The transformer was invented by Yann LeCun in 2015.",
            context   = CONTEXT,
            metadata  = {"id": "q3"},
        ),
    ]

    print("\nRunning batch evaluation...")
    report = evaluator.evaluate_batch(inputs)

    print()
    print(report.summary_table())

    # Export to JSON
    out_path = "/tmp/eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report saved → {out_path}")

    return report


# ── Example 5: Custom metric ──────────────────────────────────────────────────

def example_custom_metric():
    """
    Shows how to write and plug in a custom metric.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5 — Custom metric (Citation Quality)")
    print("="*60)

    from rag_eval import MetricName, MetricResult, LLMJudge, EvalInput

    _SYSTEM = """\
You are an expert evaluator. Respond ONLY with valid JSON:
{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>", "evidence": "<excerpt>"}
"""

    class CitationQualityMetric:
        """Does the answer cite its sources clearly and appropriately?"""
        name = MetricName.CONCISENESS   # reuse an existing slot, or extend the enum

        def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
            prompt = f"""\
Evaluate the quality of citations in the ANSWER.

QUESTION: {inp.question}
ANSWER: {inp.answer}

Criterion: Does the answer reference its sources (author, year, paper name)?
Are citations accurate and relevant? Score 1.0 if well-cited, 0 if none.

Respond with JSON only."""
            raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
            return MetricResult(
                name=self.name, score=max(0.0, min(1.0, float(raw["score"]))),
                reasoning=raw["reasoning"], raw=raw,
            )

    evaluator = RAGEvaluator(metrics=[CitationQualityMetric()])

    result = evaluator.evaluate(
        question="What introduced the transformer architecture?",
        answer=(
            'The transformer architecture was introduced in "Attention Is All '
            'You Need" by Vaswani et al. (2017), published at NeurIPS.'
        ),
    )

    m = result.metrics[0]
    print(f"\nCitation quality: {m.score:.3f}")
    print(f"  ↳ {m.reasoning}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    example_single_full()
    example_single_minimal()
    example_hallucination()
    example_batch()
    example_custom_metric()

    print("\n✓ All examples completed.")
