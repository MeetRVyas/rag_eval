"""
judge.py — Backward-compatible facade for LLMJudge.

The original `LLMJudge` class is preserved here as an alias for
`AnthropicJudge` so existing code continues to work unchanged:

    from rag_eval import LLMJudge          # still works
    judge = LLMJudge(model="...", ...)     # still works

For multi-provider usage, prefer `create_judge()` or the `provider=`
argument on `RAGEvaluator`.
"""
from __future__ import annotations

# Re-export the factory and base class for public use
from .providers import create_judge, Provider, PROVIDER_MODELS          # noqa: F401
from .providers.base import BaseJudge                                    # noqa: F401
from .providers.anthropic import AnthropicJudge                          # noqa: F401


# ── Backward-compatible alias ─────────────────────────────────────────────────

class LLMJudge(AnthropicJudge):
    """
    Backward-compatible wrapper.  Identical to AnthropicJudge.

    Preserved so any existing code that does:

        from rag_eval import LLMJudge
        judge = LLMJudge(model=..., api_key=..., temperature=...)

    continues to work without modification.

    For new code, prefer using `RAGEvaluator(provider=..., model=...)`.
    """

    def __init__(
        self,
        model:       str = "claude-sonnet-4-6",
        api_key      = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        super().__init__(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )
