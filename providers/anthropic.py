"""
AnthropicJudge — LLM judge backed by the Anthropic Messages API.

Supported models (see providers/__init__.py for full registry):
  claude-opus-4-6           most capable
  claude-sonnet-4-6         default — best quality/cost ratio
  claude-haiku-4-5-20251001 fastest, cheapest

Install:
    pip install anthropic>=0.40.0
    export ANTHROPIC_API_KEY="sk-ant-..."
"""
from __future__ import annotations

from typing import Optional

from .base import BaseJudge


class AnthropicJudge(BaseJudge):
    """Judge that calls the Anthropic Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        model:       str,
        api_key:     Optional[str] = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for the Anthropic provider.\n"
                "Install it with:  pip install 'rag-eval[anthropic]'\n"
                "  or:             pip install anthropic"
            ) from exc

        self._client = (
            _anthropic.Anthropic(api_key=api_key)
            if api_key
            else _anthropic.Anthropic()
        )

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model       = self.model,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
            system      = system_prompt,
            messages    = [{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
