# rag_eval — RAG Evaluation Engine

A rigorous, modular evaluation engine for **Research QA RAG systems**.  
Uses **LLM-as-judge** (Claude) to score answers across six metrics — works with any LLM or retrieval framework.

---

## Installation

```bash
pip install anthropic          # only dependency
# then drop the rag_eval/ folder into your project, or:
pip install -e .               # install as an editable package
```

Set your API key:
```bash
export ANTHROPIC_API_KEY="sk-..."
```

---

## Quick Start

```python
from rag_eval import RAGEvaluator

evaluator = RAGEvaluator()

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
    metadata  = {"model": "gpt-4o", "run_id": "abc123"},
)
```

---

## Batch Evaluation

```python
from rag_eval import RAGEvaluator, EvalInput

evaluator = RAGEvaluator(max_workers=8)  # parallel by default

inputs = [EvalInput(...), EvalInput(...), ...]
report = evaluator.evaluate_batch(inputs)

print(report.summary_table())

import json
with open("report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

---

## Custom Metrics & Weights

```python
from rag_eval import (
    RAGEvaluator, MetricName, MetricResult, EvalInput, LLMJudge,
    AnswerRelevanceMetric, FaithfulnessMetric,
)

class MyCitationMetric:
    name = MetricName.CONCISENESS   # reuse or extend MetricName

    def __call__(self, inp: EvalInput, judge: LLMJudge) -> MetricResult:
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
    metrics = [AnswerRelevanceMetric(), FaithfulnessMetric(), MyCitationMetric()],
    weights = {
        MetricName.ANSWER_RELEVANCE: 0.4,
        MetricName.FAITHFULNESS:     0.4,
        MetricName.CONCISENESS:      0.2,   # MyCitationMetric reuses this slot
    },
)
```

---

## Configuration Options

```python
RAGEvaluator(
    model       = "claude-opus-4-20250514",   # any Anthropic model
    api_key     = "sk-...",                    # or set ANTHROPIC_API_KEY
    temperature = 0.0,                         # 0 = deterministic judgments
    metrics     = [...],                       # default: all 6 metrics
    weights     = {...},                       # default: see evaluator.py
    max_workers = 4,                           # parallel batch workers
)
```

---

## Output Structure

```python
EvalResult
├── overall_score: float          # weighted mean of active metrics
├── latency_ms:    float
├── model:         str
├── metrics: list[MetricResult]
│   ├── name:      MetricName
│   ├── score:     float (0–1)
│   ├── reasoning: str
│   └── raw:       dict           # full LLM JSON
└── input:         EvalInput

EvalReport (batch)
├── mean_overall:  float
├── pass_rate():   float
├── metric_mean(name): float
├── summary_table(): str
├── to_dict():     dict           # JSON-serialisable
└── results: list[EvalResult]
```

---

## Running Examples

```bash
ANTHROPIC_API_KEY=sk-... python -m rag_eval.examples
```
