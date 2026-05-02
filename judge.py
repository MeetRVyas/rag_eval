"""
LLM-as-Judge: thin wrapper around the Anthropic Messages API.
Supports any model; defaults to claude-sonnet-4-20250514.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import anthropic

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_MAX_RETRIES   = 3
_RETRY_DELAY   = 1.5   # seconds


class LLMJudge:
    """
    Sends a structured prompt to Claude and returns a validated JSON dict.

    Parameters
    ----------
    model       : Anthropic model string
    api_key     : optional; falls back to ANTHROPIC_API_KEY env var
    temperature : sampling temperature (0 = deterministic)
    max_tokens  : max tokens for the judge response
    """

    def __init__(
        self,
        model:       str = _DEFAULT_MODEL,
        api_key:     Optional[str] = None,
        temperature: float = 0.0,
        max_tokens:  int = 512,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._client     = anthropic.Anthropic(api_key=api_key) if api_key \
                           else anthropic.Anthropic()

    # ── Public API ───────────────────────────────────────────────────────────

    def judge(
        self,
        system_prompt: str,
        user_prompt:   str,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM and return a parsed JSON dict.

        Retries up to _MAX_RETRIES times on malformed JSON or missing keys.
        Raises `RuntimeError` if all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model       = self.model,
                    max_tokens  = self.max_tokens,
                    temperature = self.temperature,
                    system      = system_prompt,
                    messages    = [{"role": "user", "content": user_prompt}],
                )
                raw_text = response.content[0].text
                parsed   = self._parse_json(raw_text)

                if required_keys:
                    missing = [k for k in required_keys if k not in parsed]
                    if missing:
                        raise ValueError(f"LLM response missing keys: {missing}. Got: {parsed}")

                return parsed

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)

        raise RuntimeError(
            f"LLMJudge failed after {_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """
        Extract JSON from LLM output that may contain markdown fences,
        leading/trailing prose, or minor whitespace issues.
        """
        # 1. Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try to extract content from ```json ... ``` or ``` ... ``` blocks
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Grab the first {...} block in the text
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)
