"""
BaseJudge — abstract base class for all LLM judge implementations.

All provider-specific judges extend this class and only need to implement
`_call_api(system_prompt, user_prompt) -> str`.

The public `judge()` method adds:
  • retry logic (up to _MAX_RETRIES attempts with exponential back-off)
  • JSON extraction from raw LLM text (strips fences, finds first {...})
  • required-key validation
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

_MAX_RETRIES = 3
_RETRY_DELAY = 1.5   # seconds; multiplied by attempt index


class BaseJudge(ABC):
    """
    Abstract LLM judge.  Sub-classes implement `_call_api()`.

    Attributes
    ----------
    model       : Model identifier string used for this judge instance.
    temperature : Sampling temperature (0 = deterministic).
    max_tokens  : Upper bound on judge response tokens.
    provider    : Human-readable provider name (set by subclasses).
    """

    provider: str = "base"   # overridden by each subclass

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def judge(
        self,
        system_prompt: str,
        user_prompt:   str,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM, parse JSON, validate required keys.

        Retries up to `_MAX_RETRIES` times on malformed JSON or missing keys.
        Raises `RuntimeError` after all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                raw_text = self._call_api(system_prompt, user_prompt)
                parsed   = self._parse_json(raw_text)

                if required_keys:
                    missing = [k for k in required_keys if k not in parsed]
                    if missing:
                        raise ValueError(
                            f"LLM response missing keys {missing}. Got: {list(parsed)}"
                        )

                return parsed

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)

        raise RuntimeError(
            f"[{self.provider}] LLMJudge failed after {_MAX_RETRIES} attempts. "
            f"Model: {self.model}. Last error: {last_error}"
        )

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make the actual API call and return the raw response text.
        Sub-classes implement provider-specific SDK calls here.
        """

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """
        Extract a JSON object from LLM output that may contain markdown fences,
        leading/trailing prose, or minor whitespace variations.

        Tries three strategies in order:
          1. Direct json.loads on the full text.
          2. Extract content from ```json...``` or ```...``` fences.
          3. Find the first {...} block in the text.
        """
        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Markdown fences: ```json ... ``` or ``` ... ```
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. First {...} block (handles preamble / postamble prose)
        brace = re.search(r"\{[\s\S]*\}", text)
        if brace:
            try:
                return json.loads(brace.group())
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider!r}, model={self.model!r}, "
            f"temperature={self.temperature})"
        )
