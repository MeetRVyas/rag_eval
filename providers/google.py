"""
GoogleJudge — LLM judge backed by the Google Gemini API.

Supported models (see providers/__init__.py for full registry):
  gemini-2.5-pro              most capable
  gemini-2.5-flash            default — fast + capable
  gemini-2.5-flash-lite       lightweight
  gemini-3-flash-preview      next-gen preview
  gemini-3.1-flash-lite-preview next-gen lite preview

Install:
    pip install google-genai>=1.0.0
    export GOOGLE_API_KEY="AIza..."   # or GEMINI_API_KEY
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseJudge


class GoogleJudge(BaseJudge):
    """Judge that calls the Google Gemini API via the google-genai SDK."""

    provider = "google"

    def __init__(
        self,
        model:       str,
        api_key:     Optional[str] = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for the Google provider.\n"
                "Install it with:  pip install 'rag-eval[google]'\n"
                "  or:             pip install google-genai"
            ) from exc

        self._genai  = _genai
        self._types  = _types

        # Resolve API key: explicit > GOOGLE_API_KEY > GEMINI_API_KEY
        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
                "in your environment, or pass api_key= to RAGEvaluator."
            )

        self._client = _genai.Client(api_key=resolved_key)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.models.generate_content(
            model    = self.model,
            config   = self._types.GenerateContentConfig(
                system_instruction = system_prompt,
                temperature        = self.temperature,
                max_output_tokens  = self.max_tokens,
            ),
            contents = user_prompt,
        )
        return response.text
