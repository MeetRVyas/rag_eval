# rag_eval — RAG Evaluation Engine

A rigorous, modular evaluation engine for **Research QA RAG systems**.  
Uses **LLM-as-judge** to score answers across six metrics — now with **multi-provider support**.

---

## Providers

| Provider | Default model | Key env-var | Install extra |
|----------|--------------|-------------|---------------|
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `pip install 'rag-eval[anthropic]'` |
| `google` | `gemini-2.5-flash` | `GOOGLE_API_KEY` | `pip install 'rag-eval[google]'` |
| `openai` | `gpt-5-2025-08-07` | `OPENAI_API_KEY` | `pip install 'rag-eval[openai]'` |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | `pip install 'rag-eval[groq]'` |
| `ollama` | `llama3.2` | *(none)* | `pip install 'rag-eval[ollama]'` |

> **Groq** is recommended for cost-sensitive or high-throughput eval pipelines — same quality, much faster.  
> **Ollama** is fully air-gapped: no API key, no data leaves your machine.

---

## Installation

```bash
# Anthropic only (original behaviour)
pip install 'rag-eval[anthropic]'

# Add more providers as needed
pip install 'rag-eval[anthropic,google,groq]'

# Everything
pip install 'rag-eval[all]'

# Ollama / Groq reuse the openai SDK — no extra package
pip install 'rag-eval[openai]'
```

Set the API key for your chosen provider:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # anthropic
export GOOGLE_API_KEY="AIza..."         # google  (or GEMINI_API_KEY)
export OPENAI_API_KEY="sk-..."          # openai
export GROQ_API_KEY="gsk_..."           # groq
# ollama needs no key — just run: ollama serve
```

---

## Quick Start

```python
from rag_eval import RAGEvaluator

# Pick any provider
evaluator = RAGEvaluator()                                        # anthropic (default)
evaluator = RAGEvaluator(provider="google")                       # Gemini 2.5 Flash
evaluator = RAGEvaluator(provider="groq")                         # Llama 3.3 70B on Groq
evaluator = RAGEvaluator(provider="ollama", model="llama3.2")     # local, no key
evaluator = RAGEvaluator(provider="openai", model="gpt-5-2025-08-07")

result = evaluator.evaluate(
    question  = "Why does transformer attention scale quadratically?",
    answer    = "Because each token attends to all others, giving O(n²) cost.",
    context   = "..retrieved passages..",   # optional — unlocks faithfulness & hallucination
    reference = "..ground truth answer..",  # optional — unlocks correctness
)

print(f"Overall: {result.overall_score:.3f}")
for m in result.metrics:
    print(f"  {m.name.value}: {m.score:.3f} — {m.reasoning}")
```

---

## Available Models

```python
from rag_eval import list_models

list_models()           # all providers
list_models("groq")     # one provider
```

```
  ┌─ ANTHROPIC  (Requires ANTHROPIC_API_KEY)
  │   claude-opus-4-6
  │   claude-sonnet-4-6 ← default
  │   claude-haiku-4-5-20251001
  └──────────────────────────────────────────────────

  ┌─ GOOGLE  (Requires GOOGLE_API_KEY (or GEMINI_API_KEY))
  │   gemini-2.5-pro
  │   gemini-2.5-flash ← default
  │   gemini-2.5-flash-lite
  │   gemini-3-flash-preview
  │   gemini-3.1-flash-lite-preview
  └──────────────────────────────────────────────────

  ┌─ OPENAI  (Requires OPENAI_API_KEY)
  │   gpt-5-2025-08-07 ← default
  │   gpt-5.4-2026-03-05
  │   gpt-5.5-2026-04-23
  │   gpt-5-mini-2025-08-07
  │   gpt-5.4-mini-2026-03-17
  │   gpt-5.4-nano-2026-03-17
  └──────────────────────────────────────────────────

  ┌─ GROQ  (Requires GROQ_API_KEY — very fast inference)
  │   llama-3.3-70b-versatile ← default
  │   deepseek-r1-distill-llama-70b
  │   qwen-qwq-32b
  │   mixtral-8x7b-32768
  │   llama-3.1-8b-instant
  └──────────────────────────────────────────────────

  ┌─ OLLAMA  (No API key needed — Ollama must be running locally)
  │   llama3.2 ← default
  │   llama3.3
  │   qwen2.5
  │   mistral
  │   phi4
  │   gemma3:9b
  └──────────────────────────────────────────────────
```

---

## Metrics

| Metric | Description | Requires |
|--------|-------------|----------|
| `answer_relevance` | Does the answer address the question? | question + answer |
| `faithfulness` | Are claims supported by the retrieved context? | + context |
| `correctness` | Semantic match to a reference answer | + reference |
| `completeness` | Are all aspects of the question covered? | question + answer |
| `hallucination` | Detects fabricated / unsupported facts (1.0 = none) | + context |
| `conciseness` | No padding, repetition, or filler | question + answer |

Metrics that lack required inputs are automatically **skipped** and excluded from the overall score.

---

## EvalInput

```python
from rag_eval import EvalInput

inp = EvalInput(
    question  = "...",
    answer    = "...",
    context   = "..." | ["chunk1", "chunk2"],   # list is joined automatically
    reference = "...",
    metadata  = {"model": "gpt-5", "run_id": "abc123"},
)
```

---

## Batch Evaluation

```python
from rag_eval import RAGEvaluator, EvalInput

evaluator = RAGEvaluator(provider="groq", max_workers=8)

inputs = [EvalInput(...), EvalInput(...), ...]
report = evaluator.evaluate_batch(inputs)

print(report.summary_table())

import json
with open("report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

---

## Custom Metrics

Custom metrics receive the `BaseJudge` interface — they work identically
regardless of which provider you choose.

```python
from rag_eval import (
    RAGEvaluator, MetricName, MetricResult, EvalInput, BaseJudge,
    AnswerRelevanceMetric, FaithfulnessMetric,
)

class MyCitationMetric:
    name = MetricName.CONCISENESS   # reuse or extend MetricName

    def __call__(self, inp: EvalInput, judge: BaseJudge) -> MetricResult:
        raw = judge.judge(
            system_prompt = "Respond ONLY with JSON: {score, reasoning, evidence}",
            user_prompt   = f"Does this answer cite its sources?\n{inp.answer}",
            required_keys = ["score", "reasoning", "evidence"],
        )
        return MetricResult(
            name      = self.name,
            score     = float(raw["score"]),
            reasoning = raw["reasoning"],
            raw       = raw,
        )

evaluator = RAGEvaluator(
    provider = "google",
    model    = "gemini-2.5-pro",
    metrics  = [AnswerRelevanceMetric(), FaithfulnessMetric(), MyCitationMetric()],
    weights  = {
        MetricName.ANSWER_RELEVANCE: 0.4,
        MetricName.FAITHFULNESS:     0.4,
        MetricName.CONCISENESS:      0.2,
    },
)
```

---

## Configuration Options

```python
RAGEvaluator(
    provider        = "groq",                    # provider name or Provider enum
    model           = "llama-3.3-70b-versatile", # omit for provider's default
    api_key         = "gsk_...",                 # or set env-var
    temperature     = 0.0,                        # 0 = deterministic judgments
    metrics         = [...],                      # default: all 6 metrics
    weights         = {...},                      # default: see evaluator.py
    max_workers     = 4,                          # parallel batch workers
    ollama_base_url = "http://localhost:11434",   # Ollama only
)
```

---

## Provider enum (optional)

```python
from rag_eval import Provider

evaluator = RAGEvaluator(provider=Provider.GROQ)
evaluator = RAGEvaluator(provider=Provider.OLLAMA, model="phi4")
```

---

## Output Structure

```
EvalResult
├── overall_score : float          # weighted mean of active metrics
├── latency_ms    : float
├── model         : str
├── metrics       : list[MetricResult]
│   ├── name      : MetricName
│   ├── score     : float (0–1)
│   ├── reasoning : str
│   └── raw       : dict           # full LLM JSON
└── input         : EvalInput

EvalReport (batch)
├── mean_overall  : float
├── pass_rate()   : float
├── metric_mean(name) : float
├── summary_table()   : str
├── to_dict()         : dict       # JSON-serialisable
└── results       : list[EvalResult]
```

---

## Backward Compatibility

All v1 code continues to work unchanged:

```python
from rag_eval import LLMJudge          # still works → alias for AnthropicJudge
from rag_eval import RAGEvaluator

evaluator = RAGEvaluator()             # still defaults to Anthropic
evaluator = RAGEvaluator(model="claude-opus-4-6", api_key="sk-ant-...")
```

The only breaking change in v2 is the removal of the `model=` parameter as a
positional argument to `RAGEvaluator` — it must now be passed alongside `provider=`.

---

## Running Examples

```bash
ANTHROPIC_API_KEY=sk-ant-...  python -m rag_eval.examples
GOOGLE_API_KEY=AIza...        python -m rag_eval.examples
GROQ_API_KEY=gsk_...          python -m rag_eval.examples
```
