"""
LLM client abstraction for genre subsystem tooling.

Used by vocab normalization (Phase 1) and embedding prior generation (Phase 2).
Not on the hot path — offline/artifact-build use only.

Supported providers: "anthropic" (default), "openai".
Both support dry_run=True for deterministic testing without API calls.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

JSONResponse = dict[str, Any] | list[Any]


class LLMClient(abc.ABC):
    """Minimal interface: prompt → parsed JSON object or array."""

    provider: str
    model: str

    @abc.abstractmethod
    def complete_json(self, prompt: str, *, max_retries: int = 2) -> JSONResponse:
        """Send prompt and return a parsed JSON object or array."""
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicLLMClient(LLMClient):
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        retry_sleep: float = 1.5,
    ) -> None:
        self.model = model
        self._retry_sleep = retry_sleep
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required: pip install anthropic"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY env var is not set and api_key was not provided"
            )
        self._client = _anthropic.Anthropic(api_key=key)

    def complete_json(self, prompt: str, *, max_retries: int = 2) -> JSONResponse:
        import anthropic

        for attempt in range(max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    system=(
                        "You are a helpful assistant. Always respond with valid JSON only, "
                        "no markdown fences, no preamble."
                    ),
                )
                text = response.content[0].text.strip()
                return json.loads(text)
            except (json.JSONDecodeError, TypeError, IndexError, AttributeError) as exc:
                if attempt < max_retries:
                    logger.warning("JSON parse failed on attempt %d: %s", attempt + 1, exc)
                    time.sleep(self._retry_sleep)
                    continue
                raise RuntimeError(
                    f"AnthropicLLMClient: failed to parse JSON after {max_retries + 1} attempts"
                ) from exc
            except anthropic.APIError as exc:
                if attempt < max_retries:
                    logger.warning("API error on attempt %d: %s", attempt + 1, exc)
                    time.sleep(self._retry_sleep)
                    continue
                raise RuntimeError(f"AnthropicLLMClient: API error: {exc}") from exc
        raise RuntimeError("AnthropicLLMClient: exhausted retries")  # unreachable


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        retry_sleep: float = 1.5,
    ) -> None:
        self.model = model
        self._retry_sleep = retry_sleep
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY env var is not set and api_key was not provided"
            )
        self._client = _openai.OpenAI(api_key=key)

    def complete_json(self, prompt: str, *, max_retries: int = 2) -> JSONResponse:
        import openai

        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. Always respond with valid JSON only, "
                                "no markdown fences, no preamble."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
                return json.loads(text)
            except (json.JSONDecodeError, TypeError, IndexError, AttributeError) as exc:
                if attempt < max_retries:
                    time.sleep(self._retry_sleep)
                    continue
                raise RuntimeError(
                    f"OpenAILLMClient: failed to parse JSON after {max_retries + 1} attempts"
                ) from exc
            except openai.APIError as exc:
                if attempt < max_retries:
                    time.sleep(self._retry_sleep)
                    continue
                raise RuntimeError(f"OpenAILLMClient: API error: {exc}") from exc
        raise RuntimeError("OpenAILLMClient: exhausted retries")  # unreachable


# ---------------------------------------------------------------------------
# Dry-run (deterministic mock — no API calls)
# ---------------------------------------------------------------------------

class DryRunLLMClient(LLMClient):
    """
    Returns deterministic mock JSON derived from the prompt hash.
    Suitable for tests and cost-free dry-run previews.

    The caller is responsible for injecting realistic mock data in tests by
    subclassing or patching `_mock_response`.
    """

    provider = "dry-run"

    def __init__(self, model: str = "dry-run") -> None:
        self.model = model
        # Tests can patch this dict: {prompt_hash -> parsed JSON response}
        self._overrides: dict[str, JSONResponse] = {}

    def complete_json(self, prompt: str, *, max_retries: int = 2) -> JSONResponse:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if h in self._overrides:
            return self._overrides[h]
        return self._mock_response(prompt, h)

    def _mock_response(self, prompt: str, prompt_hash: str) -> JSONResponse:
        return {"_dry_run": True, "prompt_hash": prompt_hash}

    def prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_llm_client(
    provider: str = "anthropic",
    model: str | None = None,
    *,
    dry_run: bool = False,
    api_key: str | None = None,
) -> LLMClient:
    """Build an LLMClient for the given provider.

    Args:
        provider: "anthropic" | "openai" | "dry-run"
        model: model name; defaults to provider's recommended fast model
        dry_run: if True, return a DryRunLLMClient regardless of provider
        api_key: API key (falls back to env var)
    """
    if dry_run or provider == "dry-run":
        return DryRunLLMClient(model=model or "dry-run")

    if provider == "anthropic":
        return AnthropicLLMClient(
            model=model or "claude-haiku-4-5-20251001",
            api_key=api_key,
        )
    if provider == "openai":
        return OpenAILLMClient(
            model=model or "gpt-4o-mini",
            api_key=api_key,
        )
    raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic', 'openai', or 'dry-run'.")
