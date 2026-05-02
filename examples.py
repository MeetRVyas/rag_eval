"""
examples.py — Usage examples for rag_eval.

Run with:
    ANTHROPIC_API_KEY=sk-ant-...   python examples.py
    GOOGLE_API_KEY=AIza...         python examples.py
    GROQ_API_KEY=gsk_...           python examples.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval import (
    RAGEvaluator, EvalInput, MetricName, list_models,
    AnswerRelevanceMetric, FaithfulnessMetric, HallucinationMetric,
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


# ── Example 0: List all available providers & models ─────────────────────────

def example_list_models():
    print("\n" + "="*60)
    print("EXAMPLE 0 — Available providers and models")
    print("="*60)
    list_models()


# ── Example 1: Anthropic (default) ───────────────────────────────────────────

def example_anthropic():
    print("\n" + "="*60)
    print("EXAMPLE 1 — Anthropic judge (default provider)")
    print("="*60)

    evaluator = RAGEvaluator()   # provider="anthropic", model="claude-sonnet-4-6"
    print(f"\nJudge: {evaluator}")

    result = evaluator.evaluate(
        question  = "Why does transformer attention scale quadratically?",
        answer    = (
            "Transformer self-attention has O(n²) complexity because each of "
            "the n tokens must compute attention scores against all other n "
            "tokens. Sparse attention variants reduce this cost."
        ),
        context   = CONTEXT,
        reference = REFERENCE,
    )

    _print_result(result)
    return result


# ── Example 2: Google Gemini ──────────────────────────────────────────────────

def example_google():
    print("\n" + "="*60)
    print("EXAMPLE 2 — Google Gemini judge")
    print("="*60)

    evaluator = RAGEvaluator(provider="google", model="gemini-2.5-flash")
    print(f"\nJudge: {evaluator}")

    result = evaluator.evaluate(
        question  = "What is RLHF?",
        answer    = (
            "RLHF stands for Reinforcement Learning from Human Feedback. "
            "It fine-tunes a language model using a reward model trained on "
            "human preference data, aligning the model with human values."
        ),
        reference = (
            "RLHF is a technique that trains a reward model from human "
            "preference labels and then fine-tunes the LLM with RL to "
            "maximise that reward."
        ),
    )

    _print_result(result)
    return result


# ── Example 3: Groq (fast inference) ─────────────────────────────────────────

def example_groq():
    print("\n" + "="*60)
    print("EXAMPLE 3 — Groq judge (fast inference)")
    print("="*60)

    evaluator = RAGEvaluator(
        provider = "groq",
        model    = "llama-3.3-70b-versatile",
    )
    print(f"\nJudge: {evaluator}")

    result = evaluator.evaluate(
        question = "What is the complexity of transformer self-attention?",
        answer   = "It is O(n²) where n is the sequence length.",
        context  = CONTEXT,
    )

    _print_result(result)
    return result


# ── Example 4: OpenAI ─────────────────────────────────────────────────────────

def example_openai():
    print("\n" + "="*60)
    print("EXAMPLE 4 — OpenAI judge")
    print("="*60)

    evaluator = RAGEvaluator(provider="openai")  # gpt-5-2025-08-07 default
    print(f"\nJudge: {evaluator}")

    result = evaluator.evaluate(
        question = "Who introduced the transformer architecture?",
        answer   = (
            'The transformer was introduced in "Attention Is All You Need" '
            "by Vaswani et al. (2017) at NeurIPS."
        ),
        context  = CONTEXT,
    )

    _print_result(result)
    return result


# ── Example 5: Ollama (local) ─────────────────────────────────────────────────

def example_ollama():
    print("\n" + "="*60)
    print("EXAMPLE 5 — Ollama judge (local model, no API key)")
    print("="*60)
    print("  Requires: `ollama serve` + `ollama pull llama3.2`")

    evaluator = RAGEvaluator(
        provider = "ollama",
        model    = "llama3.2",
        # ollama_base_url = "http://localhost:11434",  # default
    )
    print(f"\nJudge: {evaluator}")

    result = evaluator.evaluate(
        question = "What does O(n²) mean in transformer attention?",
        answer   = (
            "O(n²) means the computation grows quadratically with sequence "
            "length n, so doubling n quadruples the cost."
        ),
        context  = CONTEXT,
    )

    _print_result(result)
    return result


# ── Example 6: Cross-provider batch comparison ────────────────────────────────

def example_cross_provider_batch():
    """
    Evaluate the same inputs with two different providers and compare scores.
    Useful for auditing judge consistency across models.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6 — Cross-provider comparison (Anthropic vs Groq)")
    print("="*60)

    inputs = [
        EvalInput(
            question = "What is self-attention?",
            answer   = "Self-attention lets each token attend to all others.",
            context  = CONTEXT,
            metadata = {"id": "q1"},
        ),
        EvalInput(
            question = "Who invented the transformer?",
            answer   = "The transformer was invented by Yann LeCun in 2015.",  # wrong
            context  = CONTEXT,
            metadata = {"id": "q2"},
        ),
    ]

    providers = [
        ("anthropic", "claude-sonnet-4-6"),
        ("groq",      "llama-3.3-70b-versatile"),
    ]

    for prov, model in providers:
        evaluator = RAGEvaluator(
            provider    = prov,
            model       = model,
            metrics     = [AnswerRelevanceMetric(), FaithfulnessMetric(), HallucinationMetric()],
            weights     = {MetricName.ANSWER_RELEVANCE: 0.4,
                           MetricName.FAITHFULNESS:     0.4,
                           MetricName.HALLUCINATION:    0.2},
            max_workers = 2,
        )
        print(f"\n─── Provider: {prov} / {model} ───")
        report = evaluator.evaluate_batch(inputs, show_progress=False)
        print(report.summary_table())


# ── Example 7: Custom metric with provider choice ────────────────────────────

def example_custom_metric_with_provider():
    print("\n" + "="*60)
    print("EXAMPLE 7 — Custom metric + Google Gemini")
    print("="*60)

    from rag_eval import MetricResult, BaseJudge, EvalInput

    _SYSTEM = """\
You are an expert evaluator. Respond ONLY with valid JSON:
{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>", "evidence": "<excerpt>"}
"""

    class CitationQualityMetric:
        """Does the answer clearly cite its sources?"""
        name = MetricName.CONCISENESS   # reuse an existing enum slot

        def __call__(self, inp: EvalInput, judge: BaseJudge) -> MetricResult:
            prompt = (
                f"Does the ANSWER reference sources (author, year, paper)?\n\n"
                f"QUESTION: {inp.question}\nANSWER: {inp.answer}\n\n"
                "Score 1.0 if well-cited, 0 if none. Respond with JSON only."
            )
            raw = judge.judge(_SYSTEM, prompt, ["score", "reasoning", "evidence"])
            return MetricResult(
                name=self.name, score=max(0.0, min(1.0, float(raw["score"]))),
                reasoning=raw["reasoning"], raw=raw,
            )

    evaluator = RAGEvaluator(
        provider = "google",
        model    = "gemini-2.5-flash",
        metrics  = [CitationQualityMetric()],
    )

    result = evaluator.evaluate(
        question = "What introduced the transformer architecture?",
        answer   = (
            'The transformer architecture was introduced in "Attention Is All '
            'You Need" by Vaswani et al. (2017), published at NeurIPS.'
        ),
    )

    m = result.metrics[0]
    print(f"\nCitation quality ({evaluator.provider}/{evaluator.model}): {m.score:.3f}")
    print(f"  ↳ {m.reasoning}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_result(result) -> None:
    print(f"\n  Overall score : {result.overall_score:.3f}")
    print(f"  Latency       : {result.latency_ms:.0f} ms")
    print(f"  Per-metric:")
    for m in result.metrics:
        skipped = m.reasoning.startswith("Skipped")
        status  = "  –  " if skipped else f"{m.score:.3f}"
        print(f"    {m.name.value:<22} {status}")
        if not skipped:
            print(f"      ↳ {m.reasoning}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run examples based on which API keys are available
    example_list_models()

    if os.getenv("ANTHROPIC_API_KEY"):
        example_anthropic()
    else:
        print("\n⚠  ANTHROPIC_API_KEY not set — skipping Anthropic examples.")

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        example_google()
        example_custom_metric_with_provider()
    else:
        print("\n⚠  GOOGLE_API_KEY not set — skipping Google examples.")

    if os.getenv("OPENAI_API_KEY"):
        example_openai()
    else:
        print("\n⚠  OPENAI_API_KEY not set — skipping OpenAI examples.")

    if os.getenv("GROQ_API_KEY"):
        example_groq()
    else:
        print("\n⚠  GROQ_API_KEY not set — skipping Groq examples.")

    # Ollama: only run if server appears reachable
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        example_ollama()
    except Exception:
        print("\n⚠  Ollama not reachable at localhost:11434 — skipping.")

    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("GROQ_API_KEY"):
        example_cross_provider_batch()

    print("\n✓ Done.")
