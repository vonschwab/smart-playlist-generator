from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .models import validate_ai_response
from .pricing import estimate_cost_usd
from .prompt import SYSTEM_INSTRUCTIONS
from .routing import WebMode

DEFAULT_ALLOWED_WEB_DOMAINS: list[str] = []


@dataclass(frozen=True)
class EnrichmentResult:
    status: str
    response_json: dict[str, Any]
    token_usage: dict[str, int]
    estimated_cost_usd: float | None = None
    error_message: str | None = None


class OpenAIEnrichmentClient:
    """Small synchronous OpenAI wrapper with a dry-run path and retry behavior."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        dry_run: bool = False,
        web_mode: WebMode | str = WebMode.OFF,
        allowed_web_domains: list[str] | None = None,
        max_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
        ) -> None:
        self.model = model
        self.dry_run = dry_run
        self.web_mode = WebMode(web_mode)
        self.allowed_web_domains = list(DEFAULT_ALLOWED_WEB_DOMAINS if allowed_web_domains is None else allowed_web_domains)
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def enrich(
        self,
        payload: dict[str, Any],
        prompt: str,
        response_format: dict[str, Any],
        *,
        instructions: str = SYSTEM_INSTRUCTIONS,
    ) -> EnrichmentResult:
        if self.dry_run:
            estimated_chars = len(prompt)
            estimated_prompt_tokens = max(1, estimated_chars // 4)
            estimated_output_tokens = 900
            return EnrichmentResult(
                status="skipped",
                response_json={
                    "dry_run": True,
                    "model": self.model,
                    "payload": payload,
                    "web_mode": self.web_mode.value,
                    "allowed_web_domains": self.allowed_web_domains if self.web_mode != WebMode.OFF else [],
                    "estimated_prompt_chars": estimated_chars,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                },
                token_usage={
                    "estimated_prompt_chars": estimated_chars,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                },
                estimated_cost_usd=estimate_cost_usd(
                    self.model,
                    input_tokens=estimated_prompt_tokens,
                    output_tokens=estimated_output_tokens,
                ),
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return EnrichmentResult(
                status="failed",
                response_json={},
                token_usage={},
                error_message="OPENAI_API_KEY is not set",
            )
        if self.allowed_web_domains and not _supports_web_search_domain_filters(self.model):
            return EnrichmentResult(
                status="failed",
                response_json={},
                token_usage={},
                error_message=(
                    f"Model {self.model!r} does not support web_search domain filters in this SDK/API path. "
                    "Omit --allowed-web-domain or use a model that supports filtered web_search."
                ),
            )

        try:
            response_json: dict[str, Any] | None = None
            token_usage: dict[str, int] = {}
            validation_error: Exception | None = None
            for attempt in range(2):
                retry_instructions = instructions
                if attempt and validation_error is not None:
                    retry_instructions = _retry_instructions(instructions, validation_error)
                response = self._call_openai(prompt, response_format, instructions=retry_instructions)
                token_usage = _combine_token_usage(token_usage, _extract_token_usage(response))
                response_json = _extract_response_json(response)
                try:
                    validate_ai_response(response_json)
                    validation_error = None
                    break
                except ValueError as exc:
                    validation_error = exc
            if validation_error is not None:
                raise validation_error
            if response_json is None:
                raise ValueError("OpenAI response did not include parseable JSON output")
            return EnrichmentResult(
                status="complete",
                response_json=response_json,
                token_usage=token_usage,
                estimated_cost_usd=estimate_cost_usd(
                    self.model,
                    input_tokens=token_usage.get("input_tokens", 0),
                    output_tokens=token_usage.get("output_tokens", 0),
                ),
            )
        except Exception as exc:
            return EnrichmentResult(status="failed", response_json={}, token_usage={}, error_message=str(exc))

    def _call_openai(self, prompt: str, response_format: dict[str, Any], *, instructions: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Install the optional dependency with `pip install openai`."
            ) from exc

        client = OpenAI()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "instructions": instructions,
                    "input": [{"role": "user", "content": prompt}],
                    "text": {"format": response_format},
                }
                if self.web_mode != WebMode.OFF:
                    web_tool: dict[str, Any] = {"type": "web_search"}
                    if self.allowed_web_domains:
                        web_tool["filters"] = {"allowed_domains": self.allowed_web_domains}
                    kwargs["tools"] = [web_tool]
                    kwargs["tool_choice"] = "required" if self.web_mode == WebMode.REQUIRED else "auto"
                    kwargs["include"] = ["web_search_call.action.sources"]
                return client.responses.create(
                    **kwargs,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(f"OpenAI request failed after retries: {last_exc}") from last_exc


def _extract_response_json(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, dict):
        return parsed

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return json.loads(output_text)

    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", None) or []
            for part in content:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    return json.loads(text)
    raise ValueError("OpenAI response did not include parseable JSON output")


def _extract_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    result: dict[str, int] = {}
    if input_tokens is not None:
        result["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        result["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        result["total_tokens"] = int(total_tokens)
    elif "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _combine_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    combined = dict(left)
    for key in {"input_tokens", "output_tokens", "total_tokens"}:
        if key in right:
            combined[key] = combined.get(key, 0) + right[key]
    if "total_tokens" not in combined and {"input_tokens", "output_tokens"} <= combined.keys():
        combined["total_tokens"] = combined["input_tokens"] + combined["output_tokens"]
    return combined


def _retry_instructions(instructions: str, validation_error: Exception) -> str:
    return (
        instructions
        + "\n\nPrevious response failed validation: "
        + str(validation_error)
        + "\nReturn the same strict JSON schema again, correcting only the invalid provenance, "
        "basis, source indexes, descriptor classification, prune policy, or auto-apply fields. "
        "Do not cite authoritative_source or hybrid without valid authoritative supporting_source_indexes."
    )


def _supports_web_search_domain_filters(model: str) -> bool:
    return model not in {"gpt-4o-mini"}
