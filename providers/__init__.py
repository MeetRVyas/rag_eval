"""
rag_eval.providers — Multi-provider LLM judge support.

Supported providers
-------------------
  anthropic  →  Claude models (default provider)
  google     →  Gemini models via google-genai SDK
  openai     →  GPT models via openai SDK
  groq       →  Fast inference via groq SDK (OpenAI-compatible)
  ollama     →  Local models via Ollama (OpenAI-compatible endpoint)

Quick usage
-----------
from rag_eval import RAGEvaluator
from rag_eval.providers import Provider, list_models

evaluator = RAGEvaluator(provider="google", model="gemini-2.5-pro")
evaluator = RAGEvaluator(provider="ollama", model="llama3.2")
evaluator = RAGEvaluator(provider="groq")   # uses default model

list_models()   # prints all providers and their available models
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseJudge


# ── Provider enum ─────────────────────────────────────────────────────────────

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    GOOGLE    = "google"
    OPENAI    = "openai"
    GROQ      = "groq"
    OLLAMA    = "ollama"


# ── Model registry ────────────────────────────────────────────────────────────
#
# Structure per provider:
#   default : str          — used when the user doesn't specify a model
#   models  : list[str]    — all supported model identifiers
#   notes   : str          — shown in list_models() output
#
PROVIDER_MODELS: dict[Provider, dict] = {

    Provider.ANTHROPIC: {
        "default": "claude-sonnet-4-6",
        "models": [
            "claude-opus-4-6",           # most capable, highest cost
            "claude-sonnet-4-6",         # ← default: best quality/cost ratio
            "claude-haiku-4-5-20251001", # fastest, cheapest
        ],
        "notes": "Requires ANTHROPIC_API_KEY",
    },

    Provider.GOOGLE: {
        "default": "gemini-2.5-flash",
        "models": [
            "gemini-2.5-pro",               # most capable
            "gemini-2.5-flash",             # ← default: fast + capable
            "gemini-2.5-flash-lite",        # lightweight
            "gemini-3-flash-preview",       # next-gen preview
            "gemini-3.1-flash-lite-preview",# next-gen lite preview
        ],
        "notes": "Requires GOOGLE_API_KEY (or GEMINI_API_KEY)",
    },

    Provider.OPENAI: {
        "default": "gpt-5-2025-08-07",
        "models": [
            "gpt-5-2025-08-07",          # flagship, most capable ← default
            "gpt-5.4-2026-03-05",        # next-gen capable
            "gpt-5.5-2026-04-23",        # cutting-edge preview
            "gpt-5-mini-2025-08-07",     # efficient
            "gpt-5.4-mini-2026-03-17",   # mini next-gen
            "gpt-5.4-nano-2026-03-17",   # ultra-light
        ],
        "notes": "Requires OPENAI_API_KEY",
    },

    Provider.GROQ: {
        "default": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile",    # ← default: best quality on Groq
            "deepseek-r1-distill-llama-70b", # strong reasoning
            "qwen-qwq-32b",               # good reasoning, smaller
            "mixtral-8x7b-32768",         # long context, balanced
            "llama-3.1-8b-instant",       # fastest / cheapest
        ],
        "notes": "Requires GROQ_API_KEY — very fast inference",
    },

    Provider.OLLAMA: {
        "default": "llama3.2",
        "models": [
            "llama3.2",    # ← default: solid 3B/8B all-rounder
            "llama3.3",    # 70B, best quality locally
            "qwen2.5",     # strong reasoning + coding
            "mistral",     # fast, good instruction following
            "phi4",        # Microsoft, compact + capable
            "gemma3:9b",   # Google, efficient
        ],
        "notes": "No API key needed — Ollama must be running locally",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def default_model(provider: Provider | str) -> str:
    """Return the default model string for a given provider."""
    p = Provider(provider)
    return PROVIDER_MODELS[p]["default"]


def list_models(provider: Optional[Provider | str] = None) -> None:
    """Pretty-print available models. Pass a provider to filter."""
    providers = [Provider(provider)] if provider else list(Provider)
    print()
    for p in providers:
        cfg = PROVIDER_MODELS[p]
        print(f"  ┌─ {p.value.upper()}  ({cfg['notes']})")
        for m in cfg["models"]:
            marker = " ← default" if m == cfg["default"] else ""
            print(f"  │   {m}{marker}")
        print(f"  └{'─' * 50}")
    print()


def _resolve_provider(provider: Provider | str) -> Provider:
    try:
        return Provider(provider)
    except ValueError:
        valid = [p.value for p in Provider]
        raise ValueError(
            f"Unknown provider '{provider}'. Valid options: {valid}"
        )


def _resolve_model(provider: Provider, model: Optional[str]) -> str:
    """Validate and return the model string, defaulting if None."""
    cfg = PROVIDER_MODELS[provider]
    if model is None:
        return cfg["default"]
    if model not in cfg["models"]:
        import warnings
        warnings.warn(
            f"Model '{model}' is not in the known list for provider "
            f"'{provider.value}'. Proceeding anyway — double-check the name. "
            f"Known models: {cfg['models']}",
            UserWarning, stacklevel=3,
        )
    return model


# ── Judge factory ─────────────────────────────────────────────────────────────

def create_judge(
    provider:        Provider | str = Provider.ANTHROPIC,
    model:           Optional[str] = None,
    api_key:         Optional[str] = None,
    temperature:     float = 0.0,
    max_tokens:      int = 512,
    ollama_base_url: str = "http://localhost:11434",
) -> "BaseJudge":
    """
    Instantiate and return the correct judge for the requested provider.

    Parameters
    ----------
    provider        : One of "anthropic", "google", "openai", "groq", "ollama"
                      (or the Provider enum).
    model           : Model identifier — defaults to each provider's recommended
                      default if not specified.
    api_key         : API key. Falls back to the standard env-var for each
                      provider (ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.).
    temperature     : Sampling temperature. 0 = fully deterministic.
    max_tokens      : Max tokens in the judge response (keep small; 512 is fine).
    ollama_base_url : Base URL for the Ollama server (Ollama provider only).
    """
    p     = _resolve_provider(provider)
    model = _resolve_model(p, model)

    if p is Provider.ANTHROPIC:
        from .anthropic import AnthropicJudge
        return AnthropicJudge(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )

    if p is Provider.GOOGLE:
        from .google import GoogleJudge
        return GoogleJudge(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )

    if p is Provider.OPENAI:
        from .openai_compat import OpenAIJudge
        return OpenAIJudge(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )

    if p is Provider.GROQ:
        from .openai_compat import GroqJudge
        return GroqJudge(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )

    if p is Provider.OLLAMA:
        from .openai_compat import OllamaJudge
        return OllamaJudge(
            model=model,
            temperature=temperature, max_tokens=max_tokens,
            base_url=ollama_base_url,
        )

    raise ValueError(f"Unhandled provider: {p}")  # pragma: no cover


__all__ = [
    "Provider",
    "PROVIDER_MODELS",
    "create_judge",
    "default_model",
    "list_models",
]
