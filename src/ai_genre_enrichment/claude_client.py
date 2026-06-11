"""Claude-Code-backed enrichment client (Agent SDK; runs on the user's subscription).

Mirrors OpenAIEnrichmentClient's public surface so call sites are provider-neutral:
- call_structured(prompt, response_format, *, instructions) -> parsed JSON dict
- enrich(...) / request_structured(...) -> EnrichmentResult
- call_structured_batch(...) -> chunked multi-item classification

The Agent SDK has no strict structured-output mode; the JSON schema from
``response_format`` is rendered into the prompt as an output contract and the
existing validators + retry loop enforce it.

The SDK is imported lazily inside the default runners only, so unit tests can
inject fake runners without the SDK installed. Construction fails loudly when
the SDK is missing (project rule: a knob that can't act is an error, not a
silent no-op); CLI/auth problems surface as RuntimeError on the first call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models import validate_ai_response
from .prompt import SYSTEM_INSTRUCTIONS
from .client import EnrichmentResult, _combine_token_usage, _retry_instructions

# (prompt, instructions) -> (response_text, normalized_usage)
SingleRunner = Callable[[str, str], tuple[str, dict[str, int]]]
# (prompts, instructions) -> list of (response_text, normalized_usage), aligned with prompts
BatchRunner = Callable[[list[str], str], list[tuple[str, dict[str, int]]]]

_JSON_CONTRACT = (
    "\n\nRespond with ONLY one JSON object (no prose, no code fences) "
    "matching this JSON schema:\n"
)


@dataclass(frozen=True)
class BatchItemResult:
    item_id: str
    status: str  # "ok" | "failed" | "dry_run"
    output: dict[str, Any] | None
    error: str | None


def _parse_json_text(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model output text (tolerates fences/prose)."""
    raw = text.strip()
    if raw.startswith("```"):
        # strip the first fence line and a trailing fence
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # last resort: first '{' to last '}'
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError("Claude response did not include a parseable JSON object")


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    if not usage:
        return {}
    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens"):
        if usage.get(key) is not None:
            out[key] = int(usage[key])
    if "input_tokens" in out and "output_tokens" in out:
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


class ClaudeCodeEnrichmentClient:
    """Synchronous wrapper over the Claude Agent SDK with no tools enabled."""

    provider = "claude_code"

    def __init__(
        self,
        *,
        model: str = "haiku",
        dry_run: bool = False,
        max_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
        single_runner: SingleRunner | None = None,
        batch_runner: BatchRunner | None = None,
    ) -> None:
        self.model = model
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.last_token_usage: dict[str, int] = {}
        self._single_runner: SingleRunner = single_runner or self._sdk_single_runner
        self._batch_runner: BatchRunner = batch_runner or self._sdk_batch_runner
        # Fail loudly at construction when the backend can't possibly work.
        if single_runner is None and batch_runner is None and not dry_run:
            self._ensure_sdk()

    # ── availability ──────────────────────────────────────────────────────

    @staticmethod
    def _ensure_sdk() -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk is not installed. Install with `pip install -e .[ai]` "
                "and ensure Claude Code is installed and authenticated "
                "(run `claude` once interactively)."
            ) from exc

    # ── public surface ────────────────────────────────────────────────────

    def call_structured(
        self, prompt: str, response_format: dict[str, Any], *, instructions: str
    ) -> dict[str, Any]:
        """One structured call. Returns the parsed JSON dict; sets last_token_usage."""
        full_prompt = self._render_structured_prompt(prompt, response_format)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                text, usage = self._single_runner(full_prompt, instructions)
                self.last_token_usage = _normalize_usage(usage)
                return _parse_json_text(text)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(f"Claude Code request failed after retries: {last_exc}") from last_exc

    def enrich(
        self,
        payload: dict[str, Any],
        prompt: str,
        response_format: dict[str, Any],
        *,
        instructions: str = SYSTEM_INSTRUCTIONS,
    ) -> EnrichmentResult:
        if self.dry_run:
            return self._dry_run_result(payload, prompt, estimated_output_tokens=900)
        try:
            token_usage: dict[str, int] = {}
            response_json: dict[str, Any] | None = None
            validation_error: Exception | None = None
            for attempt in range(2):
                retry_instructions = instructions
                if attempt and validation_error is not None:
                    retry_instructions = _retry_instructions(instructions, validation_error)
                response_json = self.call_structured(
                    prompt, response_format, instructions=retry_instructions
                )
                token_usage = _combine_token_usage(token_usage, self.last_token_usage)
                try:
                    validate_ai_response(response_json)
                    validation_error = None
                    break
                except ValueError as exc:
                    validation_error = exc
            if validation_error is not None:
                raise validation_error
            assert response_json is not None
            return EnrichmentResult(
                status="complete",
                response_json=response_json,
                token_usage=token_usage,
                estimated_cost_usd=None,
            )
        except Exception as exc:
            return EnrichmentResult(
                status="failed", response_json={}, token_usage={}, error_message=str(exc)
            )

    def request_structured(
        self,
        *,
        payload: dict[str, Any],
        prompt: str,
        response_format: dict[str, Any],
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        instructions: str,
        estimated_output_tokens: int,
    ) -> EnrichmentResult:
        if self.dry_run:
            return self._dry_run_result(
                payload, prompt, estimated_output_tokens=estimated_output_tokens
            )
        try:
            response_json = validator(
                self.call_structured(prompt, response_format, instructions=instructions)
            )
            return EnrichmentResult(
                status="complete",
                response_json=response_json,
                token_usage=dict(self.last_token_usage),
                estimated_cost_usd=None,
            )
        except Exception as exc:
            return EnrichmentResult(
                status="failed", response_json={}, token_usage={}, error_message=str(exc)
            )

    def _dry_run_result(
        self, payload: dict[str, Any], prompt: str, *, estimated_output_tokens: int
    ) -> EnrichmentResult:
        estimated_chars = len(prompt)
        estimated_prompt_tokens = max(1, estimated_chars // 4)
        return EnrichmentResult(
            status="skipped",
            response_json={
                "dry_run": True,
                "model": self.model,
                "payload": payload,
                "web_mode": "off",
                "estimated_prompt_chars": estimated_chars,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_output_tokens": estimated_output_tokens,
            },
            token_usage={
                "estimated_prompt_chars": estimated_chars,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_output_tokens": estimated_output_tokens,
            },
            estimated_cost_usd=None,
        )

    def call_structured_batch(
        self,
        items: list[tuple[str, str]],
        *,
        item_schema: dict[str, Any],
        instructions: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        chunk_size: int = 30,
    ) -> dict[str, BatchItemResult]:
        """Classify many items in chunked calls; per-item validation + fallback.

        Returns a dict keyed by item_id. A chunk-level backend failure raises
        (the run aborts; completed work is the caller's to persist/resume).
        """
        if self.dry_run:
            return {
                item_id: BatchItemResult(item_id, "dry_run", None, None)
                for item_id, _ in items
            }
        schema_text = json.dumps(item_schema.get("schema", item_schema), sort_keys=True)
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        prompts = [self._render_chunk_prompt(chunk, schema_text) for chunk in chunks]
        responses = self._batch_runner(prompts, instructions)

        total_usage: dict[str, int] = {}
        results: dict[str, BatchItemResult] = {}
        fallbacks: list[tuple[str, str]] = []
        for chunk, (text, usage) in zip(chunks, responses):
            total_usage = _combine_token_usage(total_usage, _normalize_usage(usage))
            try:
                by_id = {
                    str(entry.get("item_id")): entry.get("output")
                    for entry in _parse_json_text(text).get("results", [])
                }
            except ValueError:
                by_id = {}
            for item_id, item_prompt in chunk:
                output = by_id.get(item_id)
                if isinstance(output, dict):
                    try:
                        validated = validator(output) if validator else output
                        results[item_id] = BatchItemResult(item_id, "ok", validated, None)
                        continue
                    except ValueError:
                        pass
                fallbacks.append((item_id, item_prompt))

        for item_id, item_prompt in fallbacks:
            try:
                output = self.call_structured(item_prompt, item_schema, instructions=instructions)
                total_usage = _combine_token_usage(total_usage, self.last_token_usage)
                validated = validator(output) if validator else output
                results[item_id] = BatchItemResult(item_id, "ok", validated, None)
            except Exception as exc:
                results[item_id] = BatchItemResult(item_id, "failed", None, str(exc))

        self.last_token_usage = total_usage
        return results

    @staticmethod
    def _render_chunk_prompt(items: list[tuple[str, str]], item_schema_text: str) -> str:
        blocks = "\n\n".join(
            f"### item_id: {item_id}\n{item_prompt}" for item_id, item_prompt in items
        )
        return (
            "Process each item below independently.\n\n"
            + blocks
            + "\n\nRespond with ONLY one JSON object (no prose, no code fences) of the form "
            '{"results": [{"item_id": "<id>", "output": <item_output>}, ...]} '
            "with exactly one entry per item_id above, where each <item_output> matches "
            "this JSON schema:\n" + item_schema_text
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _render_structured_prompt(prompt: str, response_format: dict[str, Any]) -> str:
        schema = response_format.get("schema", response_format)
        return prompt + _JSON_CONTRACT + json.dumps(schema, sort_keys=True)

    # ── default SDK runners (lazy import; not exercised by unit tests) ────

    def _sdk_single_runner(self, prompt: str, instructions: str) -> tuple[str, dict[str, int]]:
        self._ensure_sdk()
        import asyncio

        return asyncio.run(self._sdk_single(prompt, instructions))

    async def _sdk_single(self, prompt: str, instructions: str) -> tuple[str, dict[str, int]]:
        from claude_agent_sdk import query

        async def _aiter():
            async for message in query(prompt=prompt, options=self._sdk_options(instructions)):
                yield message

        return await self._collect_response(_aiter())

    def _sdk_batch_runner(
        self, prompts: list[str], instructions: str
    ) -> list[tuple[str, dict[str, int]]]:
        self._ensure_sdk()
        import asyncio

        return asyncio.run(self._sdk_batch(prompts, instructions))

    async def _sdk_batch(
        self, prompts: list[str], instructions: str
    ) -> list[tuple[str, dict[str, int]]]:
        """All chunks in ONE SDK session (consecutive turns keep the prefix cached)."""
        from claude_agent_sdk import ClaudeSDKClient

        out: list[tuple[str, dict[str, int]]] = []
        async with ClaudeSDKClient(options=self._sdk_options(instructions)) as client:
            for prompt in prompts:
                await client.query(prompt)
                out.append(await self._collect_response(client.receive_response()))
        return out

    def _sdk_options(self, instructions: str):
        from claude_agent_sdk import ClaudeAgentOptions

        # No tools, deny anything not pre-approved: pure prompt -> JSON text.
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=instructions,
            allowed_tools=[],
            permission_mode="dontAsk",
        )

    @staticmethod
    async def _collect_response(messages) -> tuple[str, dict[str, int]]:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        text_parts: list[str] = []
        usage: dict[str, int] = {}
        result_text: str | None = None
        is_error = False
        subtype = ""
        async for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                usage = _normalize_usage(message.usage)
                result_text = message.result
                is_error = bool(message.is_error)
                subtype = message.subtype
        if is_error:
            raise RuntimeError(
                f"Claude Code call failed (subtype={subtype!r}): {result_text or ''}"
            )
        return (result_text or "".join(text_parts)), usage
