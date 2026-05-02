"""
OpenAI-compatible judges — OpenAI, Groq, and Ollama.

All three use the OpenAI wire protocol, so they share one base class
(OpenAICompatibleJudge) and differ only in base_url + API key resolution.

Supported models
----------------
OpenAI:
  gpt-5-2025-08-07         flagship, most capable   ← default
  gpt-5.4-2026-03-05       next-gen capable
  gpt-5.5-2026-04-23       cutting-edge preview
  gpt-5-mini-2025-08-07    efficient
  gpt-5.4-mini-2026-03-17  mini next-gen
  gpt-5.4-nano-2026-03-17  ultra-light

Groq:
  llama-3.3-70b-versatile      best quality on Groq  ← default
  deepseek-r1-distill-llama-70b strong reasoning
  qwen-qwq-32b                  good reasoning, smaller
  mixtral-8x7b-32768            long context, balanced
  llama-3.1-8b-instant          fastest / cheapest

Ollama (local):
  llama3.2    default — solid all-rounder
  llama3.3    70B, best quality locally
  qwen2.5     strong reasoning + coding
  mistral     fast, good instruction following
  phi4        Microsoft, compact + capable
  gemma3:9b   Google, efficient

Install:
    pip install openai>=1.50.0            # covers OpenAI + Groq + Ollama
    export OPENAI_API_KEY="sk-..."
    export GROQ_API_KEY="gsk_..."
    # Ollama needs no API key — just run `ollama serve`
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseJudge

_OPENAI_BASE_URL = "https://api.openai.com/v1"
_GROQ_BASE_URL   = "https://api.groq.com/openai/v1"


# ── Shared base ───────────────────────────────────────────────────────────────

class OpenAICompatibleJudge(BaseJudge):
    """
    Judge that calls any OpenAI-compatible REST endpoint.
    Sub-classes set `provider`, `_default_base_url`, and `_env_key_name`.
    """

    provider:           str = "openai_compat"
    _default_base_url:  str = _OPENAI_BASE_URL
    _env_key_name:      str = "OPENAI_API_KEY"

    def __init__(
        self,
        model:       str,
        api_key:     Optional[str] = None,
        base_url:    Optional[str] = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the OpenAI / Groq / Ollama providers.\n"
                "Install it with:  pip install 'rag-eval[openai]'\n"
                "  or:             pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get(self._env_key_name, "none")
        resolved_url = base_url or self._default_base_url

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model       = self.model,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIJudge(OpenAICompatibleJudge):
    """Judge backed by the OpenAI API."""

    provider          = "openai"
    _default_base_url = _OPENAI_BASE_URL
    _env_key_name     = "OPENAI_API_KEY"

    def __init__(
        self,
        model:       str,
        api_key:     Optional[str] = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        super().__init__(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )


# ── Groq ─────────────────────────────────────────────────────────────────────

class GroqJudge(OpenAICompatibleJudge):
    """
    Judge backed by Groq — extremely fast LLM inference.

    Uses the OpenAI SDK pointed at Groq's endpoint.
    Requires: GROQ_API_KEY environment variable.
    """

    provider          = "groq"
    _default_base_url = _GROQ_BASE_URL
    _env_key_name     = "GROQ_API_KEY"

    def __init__(
        self,
        model:       str,
        api_key:     Optional[str] = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        super().__init__(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaJudge(OpenAICompatibleJudge):
    """
    Judge backed by a locally running Ollama instance.

    Uses Ollama's OpenAI-compatible endpoint (/v1).
    No API key required — Ollama does not authenticate.

    Make sure Ollama is running:  `ollama serve`
    Pull a model first:           `ollama pull llama3.2`
    """

    provider      = "ollama"
    _env_key_name = ""   # Ollama does not use an API key

    def __init__(
        self,
        model:       str,
        temperature: float = 0.0,
        max_tokens:  int = 512,
        base_url:    str = "http://localhost:11434",
    ) -> None:
        # Normalize: strip trailing /v1 if user accidentally included it
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        super().__init__(
            model       = model,
            api_key     = "ollama",   # dummy key; Ollama ignores it
            base_url    = base_url,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
