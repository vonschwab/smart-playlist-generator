# Claude-Code Enrichment Backend Implementation Plan (Phase 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the enrichment pipeline's OpenAI backend with a Claude-Code-backed client (Agent SDK, runs on Dylan's Max subscription), including a chunked batch-call mode, behind a provider factory.

**Architecture:** New `ClaudeCodeEnrichmentClient` mirrors `OpenAIEnrichmentClient`'s public surface (`enrich`, `request_structured`, plus a new provider-neutral `call_structured`) and adds `call_structured_batch` for chunked classification. A factory in `src/ai_genre_enrichment/provider.py` selects the provider from `config.yaml` (`ai_genre.provider`, default `claude_code`) with a `PG_AI_PROVIDER` env override for tests. All production call sites switch to the factory. No web search anywhere (Bandcamp locator is out of scope per spec).

**Tech Stack:** Python 3.11, `claude-agent-sdk` (drives the locally installed, already-authenticated Claude Code CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-analyze-library-graph-claude-design.md` (Phase 1 section). Phases 2 (analyze stages) and 3 (web GUI) get their own plans later.

**Verified SDK facts (fetched 2026-06-10 from code.claude.com/docs/en/agent-sdk/python):**
- `from claude_agent_sdk import query, ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage`
- `query(prompt=str, options=ClaudeAgentOptions) -> AsyncIterator[Message]`
- `ClaudeAgentOptions(model=..., system_prompt=..., allowed_tools=[...], permission_mode=..., max_turns=...)`; `permission_mode="dontAsk"` denies anything not pre-approved.
- Multi-turn: `async with ClaudeSDKClient(options=options) as client: await client.query(p); async for msg in client.receive_response(): ...`
- `ResultMessage` fields: `subtype: str`, `is_error: bool`, `result: str | None`, `usage: dict | None` (keys `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`), `total_cost_usd`.
- `AssistantMessage.content` is a list of blocks; text lives in `TextBlock.text`.

**Conventions for this plan:**
- Run tests with: `python -m pytest <path> -v` from the repo root.
- The SDK is imported **lazily inside the default runners only** — unit tests inject fake runners and never need the SDK installed.
- `estimate_cost_usd()` already returns `None` for unknown models, so Claude models get `estimated_cost_usd=None` with no pricing change.

---

## File structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/config_loader.py` | Modify | `ai_genre_provider` / `ai_genre_claude_model` properties |
| `config.example.yaml` | Modify | document the `ai_genre:` block |
| `src/ai_genre_enrichment/claude_client.py` | Create | `ClaudeCodeEnrichmentClient`, `BatchItemResult`, JSON-text parsing |
| `src/ai_genre_enrichment/provider.py` | Create | `create_enrichment_client`, `get_enrichment_provider`, `resolve_enrichment_model` |
| `src/ai_genre_enrichment/client.py` | Modify | add `provider = "openai"` attr + `call_structured()` + `last_token_usage` |
| `src/ai_genre_enrichment/graph_growth.py` | Modify | `propose_placement` uses `client.call_structured` |
| `src/ai_genre_enrichment/tag_adjudicator.py` | Modify | route through factory client; fail loud |
| `src/ai_genre_enrichment/storage.py` | Modify | `classify_source_tags` model param default `None` |
| `scripts/ai_genre_enrich.py` | Modify | 4 client constructions → factory; `provider="openai"` → dynamic; `--model` defaults → `None` |
| `scripts/run_model_prior_album_tests.py` | Modify | factory |
| `pyproject.toml` | Modify | add `claude-agent-sdk` to `[ai]` extra |
| `tests/unit/test_claude_client.py` | Create | client + factory tests |
| `tests/unit/test_graph_growth.py` | Modify | fake client exposes `call_structured` |
| `tests/unit/test_ai_genre_enrichment.py` | Modify | adjudicator test injects fake client |
| `tests/unit/test_ai_genre_hybrid_cli.py`, `tests/unit/test_ai_genre_model_prior.py` | Modify | pin `PG_AI_PROVIDER=openai` where they patch the OpenAI class |

---

### Task 1: Config plumbing (`ai_genre.provider`, `ai_genre.claude_model`)

**Files:**
- Modify: `src/config_loader.py` (after the `openai_api_key` property, ~line 92)
- Modify: `config.example.yaml`
- Test: `tests/unit/test_claude_client.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_claude_client.py`:

```python
"""Tests for the Claude-Code enrichment client, provider factory, and config plumbing."""
from __future__ import annotations

from src.config_loader import Config


def _write_config(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_config_ai_genre_defaults(tmp_path):
    cfg = Config(_write_config(tmp_path, "library:\n  database_path: data/metadata.db\n"))
    assert cfg.ai_genre_provider == "claude_code"
    assert cfg.ai_genre_claude_model == "haiku"


def test_config_ai_genre_explicit(tmp_path):
    cfg = Config(_write_config(
        tmp_path,
        "ai_genre:\n  provider: openai\n  claude_model: sonnet\n",
    ))
    assert cfg.ai_genre_provider == "openai"
    assert cfg.ai_genre_claude_model == "sonnet"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'ai_genre_provider'`

- [ ] **Step 3: Add the properties to Config**

In `src/config_loader.py`, directly after the `openai_api_key` property block, add:

```python
    @property
    def ai_genre_provider(self) -> str:
        """LLM provider for the genre enrichment pipeline ('claude_code' or 'openai')."""
        return (self.config.get('ai_genre') or {}).get('provider', 'claude_code')

    @property
    def ai_genre_claude_model(self) -> str:
        """Claude model alias for the claude_code provider (e.g. 'haiku', 'sonnet')."""
        return (self.config.get('ai_genre') or {}).get('claude_model', 'haiku')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: 2 PASS

- [ ] **Step 5: Document in config.example.yaml**

Add this block to `config.example.yaml` (next to the existing `openai:` block):

```yaml
# AI genre enrichment LLM backend.
# provider: claude_code  -> uses the locally installed, authenticated Claude Code
#                           (Agent SDK; runs on your Claude subscription, no API key)
# provider: openai       -> legacy; requires OPENAI_API_KEY or openai.api_key
ai_genre:
  provider: claude_code
  claude_model: haiku    # model alias for claude_code calls ('haiku' | 'sonnet' | 'opus')
```

- [ ] **Step 6: Commit**

```bash
git add src/config_loader.py config.example.yaml tests/unit/test_claude_client.py
git commit -m "feat(enrichment): add ai_genre provider/claude_model config keys"
```

---

### Task 2: ClaudeCodeEnrichmentClient core (`call_structured`, JSON parsing, guards)

**Files:**
- Create: `src/ai_genre_enrichment/claude_client.py`
- Test: `tests/unit/test_claude_client.py` (append)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_claude_client.py`, add to the **top-of-file import block** (not appended after the tests — keep imports at the top):

```python
import json
import sys

import pytest

from src.ai_genre_enrichment.claude_client import (
    ClaudeCodeEnrichmentClient,
    _parse_json_text,
)
```

Then append the tests:

```python


def _runner_returning(*texts):
    """Fake single-call runner: pops canned texts, records calls."""
    queue = list(texts)
    calls: list[tuple[str, str]] = []

    def runner(prompt: str, instructions: str):
        calls.append((prompt, instructions))
        return queue.pop(0), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_parse_json_text_plain():
    assert _parse_json_text('{"a": 1}') == {"a": 1}


def test_parse_json_text_fenced():
    assert _parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_text_with_prose():
    assert _parse_json_text('Here you go:\n{"a": 1}\nDone.') == {"a": 1}


def test_parse_json_text_invalid_raises():
    with pytest.raises(ValueError):
        _parse_json_text("no json here")


def test_construction_fails_loudly_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(RuntimeError, match="claude-agent-sdk"):
        ClaudeCodeEnrichmentClient(model="haiku")


def test_construction_with_injected_runner_skips_sdk_check():
    client = ClaudeCodeEnrichmentClient(model="haiku", single_runner=_runner_returning("{}"))
    assert client.provider == "claude_code"
    assert client.model == "haiku"


def test_call_structured_parses_json_and_records_usage():
    runner = _runner_returning('{"name": "dream pop"}')
    client = ClaudeCodeEnrichmentClient(model="haiku", single_runner=runner)
    schema = {"type": "json_schema", "name": "x", "schema": {"type": "object"}, "strict": True}
    data = client.call_structured("place this", schema, instructions="be terse")
    assert data == {"name": "dream pop"}
    assert client.last_token_usage["input_tokens"] == 10
    prompt, instructions = runner.calls[0]
    assert "place this" in prompt
    assert '"type": "object"' in prompt          # schema embedded in prompt
    assert "ONLY one JSON object" in prompt      # output contract present
    assert instructions == "be terse"


def test_call_structured_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky(prompt, instructions):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return '{"ok": true}', {"input_tokens": 1, "output_tokens": 1}

    client = ClaudeCodeEnrichmentClient(
        model="haiku", single_runner=flaky, max_retries=1, retry_sleep_seconds=0.0
    )
    assert client.call_structured("p", {"schema": {}}, instructions="i") == {"ok": True}
    assert attempts["n"] == 2


def test_call_structured_exhausted_retries_raises():
    def always_fails(prompt, instructions):
        raise RuntimeError("boom")

    client = ClaudeCodeEnrichmentClient(
        model="haiku", single_runner=always_fails, max_retries=1, retry_sleep_seconds=0.0
    )
    with pytest.raises(RuntimeError, match="failed after retries"):
        client.call_structured("p", {"schema": {}}, instructions="i")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: FAIL with `ImportError: cannot import name 'ClaudeCodeEnrichmentClient'` (Task 1 tests still pass)

- [ ] **Step 3: Create the client module**

Create `src/ai_genre_enrichment/claude_client.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/claude_client.py tests/unit/test_claude_client.py
git commit -m "feat(enrichment): ClaudeCodeEnrichmentClient core with call_structured"
```

---

### Task 3: `enrich()` and `request_structured()` on the Claude client

**Files:**
- Modify: `src/ai_genre_enrichment/claude_client.py`
- Test: `tests/unit/test_claude_client.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_claude_client.py`:

```python
def test_enrich_dry_run_matches_openai_shape():
    client = ClaudeCodeEnrichmentClient(model="haiku", dry_run=True)
    result = client.enrich({"artist": "A"}, "prompt text", {"schema": {}})
    assert result.status == "skipped"
    assert result.response_json["dry_run"] is True
    assert result.response_json["model"] == "haiku"
    assert result.token_usage["estimated_prompt_tokens"] >= 1
    assert result.estimated_cost_usd is None  # subscription usage, not billable


def test_enrich_retries_validation_then_succeeds(monkeypatch):
    import src.ai_genre_enrichment.claude_client as cc

    seen = {"n": 0}

    def fake_validate(data):
        seen["n"] += 1
        if seen["n"] == 1:
            raise ValueError("bad provenance")

    monkeypatch.setattr(cc, "validate_ai_response", fake_validate)
    runner = _runner_returning('{"first": true}', '{"second": true}')
    client = ClaudeCodeEnrichmentClient(model="haiku", single_runner=runner)
    result = client.enrich({}, "p", {"schema": {}}, instructions="base instructions")
    assert result.status == "complete"
    assert result.response_json == {"second": True}
    assert result.token_usage["input_tokens"] == 20  # combined across both attempts
    # second attempt carries the validation error back to the model
    assert "bad provenance" in runner.calls[1][1]


def test_enrich_returns_failed_after_validation_exhausted(monkeypatch):
    import src.ai_genre_enrichment.claude_client as cc

    def always_invalid(data):
        raise ValueError("never valid")

    monkeypatch.setattr(cc, "validate_ai_response", always_invalid)
    runner = _runner_returning('{"a": 1}', '{"a": 2}')
    client = ClaudeCodeEnrichmentClient(model="haiku", single_runner=runner)
    result = client.enrich({}, "p", {"schema": {}})
    assert result.status == "failed"
    assert "never valid" in (result.error_message or "")


def test_request_structured_applies_validator():
    runner = _runner_returning('{"genres": ["slowcore"]}')
    client = ClaudeCodeEnrichmentClient(model="haiku", single_runner=runner)

    def validator(data):
        assert data["genres"] == ["slowcore"]
        return data

    result = client.request_structured(
        payload={}, prompt="p", response_format={"schema": {}},
        validator=validator, instructions="i", estimated_output_tokens=300,
    )
    assert result.status == "complete"
    assert result.response_json == {"genres": ["slowcore"]}


def test_request_structured_failed_on_validator_error():
    runner = _runner_returning('{"genres": []}')
    client = ClaudeCodeEnrichmentClient(model="haiku", single_runner=runner)

    def validator(data):
        raise ValueError("empty genres")

    result = client.request_structured(
        payload={}, prompt="p", response_format={"schema": {}},
        validator=validator, instructions="i", estimated_output_tokens=300,
    )
    assert result.status == "failed"
    assert "empty genres" in (result.error_message or "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'enrich'`

- [ ] **Step 3: Implement `enrich` and `request_structured`**

Add to `ClaudeCodeEnrichmentClient` (after `call_structured`, before the helpers section):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/claude_client.py tests/unit/test_claude_client.py
git commit -m "feat(enrichment): enrich/request_structured on Claude client with validation retry"
```

---

### Task 4: `call_structured_batch` (chunked classification)

**Files:**
- Modify: `src/ai_genre_enrichment/claude_client.py`
- Test: `tests/unit/test_claude_client.py` (append)

**Design notes (from spec):** items = `(item_id, item_prompt)` pairs. One prompt per chunk; response contract `{"results": [{"item_id": ..., "output": {...}}]}`. Per-item validation; invalid/missing items fall back to a per-item `call_structured` call (simplification of the spec's "re-queue then fall back" — failures go straight to per-item fallback). A chunk-level runner exception propagates (run aborts, resumable). Usage accumulates across chunks into `last_token_usage`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_claude_client.py`:

```python
def _batch_runner_returning(*texts):
    queue = list(texts)
    calls: list[tuple[list[str], str]] = []

    def runner(prompts: list[str], instructions: str):
        calls.append((list(prompts), instructions))
        return [
            (queue.pop(0), {"input_tokens": 100, "output_tokens": 50})
            for _ in prompts
        ]

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def _batch_text(results: list[dict]) -> str:
    return json.dumps({"results": results})


def test_batch_chunks_and_returns_ok_items():
    items = [("r1", "classify r1"), ("r2", "classify r2"), ("r3", "classify r3")]
    runner = _batch_runner_returning(
        _batch_text([
            {"item_id": "r1", "output": {"genre": "slowcore"}},
            {"item_id": "r2", "output": {"genre": "dream pop"}},
        ]),
        _batch_text([{"item_id": "r3", "output": {"genre": "shoegaze"}}]),
    )
    client = ClaudeCodeEnrichmentClient(model="haiku", batch_runner=runner)
    results = client.call_structured_batch(
        items, item_schema={"schema": {"type": "object"}}, instructions="classify", chunk_size=2
    )
    # 3 items / chunk_size 2 -> one runner invocation with 2 chunk prompts
    assert len(runner.calls) == 1
    assert len(runner.calls[0][0]) == 2
    assert results["r1"].status == "ok" and results["r1"].output == {"genre": "slowcore"}
    assert results["r3"].status == "ok" and results["r3"].output == {"genre": "shoegaze"}
    # chunk prompts embed item ids and the per-item schema contract
    chunk1 = runner.calls[0][0][0]
    assert "item_id: r1" in chunk1 and "classify r1" in chunk1
    assert '"results"' in chunk1


def test_batch_missing_item_falls_back_to_single_call():
    items = [("r1", "classify r1"), ("r2", "classify r2")]
    batch = _batch_runner_returning(
        _batch_text([{"item_id": "r1", "output": {"genre": "slowcore"}}])  # r2 missing
    )
    single = _runner_returning('{"genre": "post-rock"}')
    client = ClaudeCodeEnrichmentClient(model="haiku", batch_runner=batch, single_runner=single)
    results = client.call_structured_batch(
        items, item_schema={"schema": {}}, instructions="classify", chunk_size=10
    )
    assert results["r1"].status == "ok"
    assert results["r2"].status == "ok" and results["r2"].output == {"genre": "post-rock"}
    assert "classify r2" in single.calls[0][0]  # fallback used the item prompt


def test_batch_validator_rejects_item_then_fallback_fails():
    items = [("r1", "classify r1")]
    batch = _batch_runner_returning(
        _batch_text([{"item_id": "r1", "output": {"genre": ""}}])
    )

    def failing_single(prompt, instructions):
        raise RuntimeError("rate window exhausted")

    def validator(output):
        if not output.get("genre"):
            raise ValueError("empty genre")
        return output

    client = ClaudeCodeEnrichmentClient(
        model="haiku", batch_runner=batch, single_runner=failing_single,
        max_retries=0, retry_sleep_seconds=0.0,
    )
    results = client.call_structured_batch(
        items, item_schema={"schema": {}}, instructions="classify",
        validator=validator, chunk_size=10,
    )
    assert results["r1"].status == "failed"
    assert "rate window" in (results["r1"].error or "")


def test_batch_runner_exception_propagates():
    def dead_runner(prompts, instructions):
        raise RuntimeError("session died")

    client = ClaudeCodeEnrichmentClient(model="haiku", batch_runner=dead_runner)
    with pytest.raises(RuntimeError, match="session died"):
        client.call_structured_batch(
            [("r1", "p")], item_schema={"schema": {}}, instructions="i"
        )


def test_batch_dry_run_returns_dry_run_items():
    client = ClaudeCodeEnrichmentClient(model="haiku", dry_run=True)
    results = client.call_structured_batch(
        [("r1", "p1"), ("r2", "p2")], item_schema={"schema": {}}, instructions="i"
    )
    assert all(r.status == "dry_run" for r in results.values())


def test_batch_accumulates_usage():
    items = [("r1", "p1"), ("r2", "p2"), ("r3", "p3")]
    runner = _batch_runner_returning(
        _batch_text([
            {"item_id": "r1", "output": {"g": 1}},
            {"item_id": "r2", "output": {"g": 2}},
        ]),
        _batch_text([{"item_id": "r3", "output": {"g": 3}}]),
    )
    client = ClaudeCodeEnrichmentClient(model="haiku", batch_runner=runner)
    client.call_structured_batch(
        items, item_schema={"schema": {}}, instructions="i", chunk_size=2
    )
    assert client.last_token_usage["input_tokens"] == 200  # 100 per chunk x 2 chunks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: new tests FAIL with `AttributeError: ... no attribute 'call_structured_batch'`

- [ ] **Step 3: Implement `call_structured_batch`**

Add to `ClaudeCodeEnrichmentClient` (after `request_structured`/`_dry_run_result`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/claude_client.py tests/unit/test_claude_client.py
git commit -m "feat(enrichment): chunked call_structured_batch with per-item fallback"
```

---

### Task 5: Provider-neutral `call_structured` on the OpenAI client; switch `graph_growth`

**Files:**
- Modify: `src/ai_genre_enrichment/client.py`
- Modify: `src/ai_genre_enrichment/graph_growth.py:225-241`
- Test: `tests/unit/test_claude_client.py` (append), `tests/unit/test_graph_growth.py:88-95`

- [ ] **Step 1: Write the failing test for the OpenAI alias**

Append to `tests/unit/test_claude_client.py`:

```python
def test_openai_client_call_structured_returns_parsed_json(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    class FakeResp:
        output_text = '{"name": "boom bap"}'
        usage = {"input_tokens": 7, "output_tokens": 3}

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", web_mode="off")
    monkeypatch.setattr(client, "_call_openai", lambda *a, **k: FakeResp())
    data = client.call_structured("p", {"schema": {}}, instructions="i")
    assert data == {"name": "boom bap"}
    assert client.provider == "openai"
    assert client.last_token_usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_claude_client.py::test_openai_client_call_structured_returns_parsed_json -v`
Expected: FAIL with `AttributeError: ... no attribute 'call_structured'`

- [ ] **Step 3: Add provider attr, usage field, and the alias to `OpenAIEnrichmentClient`**

In `src/ai_genre_enrichment/client.py`:

(a) Add a class attribute right under the class docstring (`class OpenAIEnrichmentClient:` block):

```python
class OpenAIEnrichmentClient:
    """Small synchronous OpenAI wrapper with a dry-run path and retry behavior."""

    provider = "openai"
```

(b) At the end of `__init__`, after `self._api_key = api_key`, add:

```python
        self.last_token_usage: dict[str, int] = {}
```

(c) Add this method directly after `__init__` (before `enrich`):

```python
    def call_structured(
        self, prompt: str, response_format: dict[str, Any], *, instructions: str
    ) -> dict[str, Any]:
        """Provider-neutral structured call: returns parsed JSON, records usage."""
        response = self._call_openai(prompt, response_format, instructions=instructions)
        self.last_token_usage = _extract_token_usage(response)
        return _extract_response_json(response)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_claude_client.py::test_openai_client_call_structured_returns_parsed_json -v`
Expected: PASS

- [ ] **Step 5: Switch `graph_growth.propose_placement` to `call_structured`**

In `src/ai_genre_enrichment/graph_growth.py`, replace (currently lines 225-241):

```python
    """Ask the model to place one candidate. `client` exposes `_call_openai`."""
    from .client import _extract_response_json

    context_names = _build_taxonomy_context(taxonomy, candidate)
    prompt = json.dumps({
        "candidate_term": candidate.term,
        "album_frequency": candidate.album_frequency,
        "cooccurring_tags": candidate.cooccurring_tags,
        "spelling_variants": candidate.variants,
        "examples": candidate.examples,
        "existing_taxonomy_names": context_names,
    }, ensure_ascii=False, sort_keys=True)
    raw = client._call_openai(
        prompt, growth_proposal_response_format(),
        instructions=GROWTH_PROPOSAL_INSTRUCTIONS,
    )
    data = _extract_response_json(raw)
```

with:

```python
    """Ask the model to place one candidate. `client` exposes `call_structured`."""
    context_names = _build_taxonomy_context(taxonomy, candidate)
    prompt = json.dumps({
        "candidate_term": candidate.term,
        "album_frequency": candidate.album_frequency,
        "cooccurring_tags": candidate.cooccurring_tags,
        "spelling_variants": candidate.variants,
        "examples": candidate.examples,
        "existing_taxonomy_names": context_names,
    }, ensure_ascii=False, sort_keys=True)
    data = client.call_structured(
        prompt, growth_proposal_response_format(),
        instructions=GROWTH_PROPOSAL_INSTRUCTIONS,
    )
```

- [ ] **Step 6: Update the graph-growth test fake**

In `tests/unit/test_graph_growth.py`, replace the `_FakeClient` class (lines 88-95):

```python
class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def _call_openai(self, prompt, response_format, *, instructions):
        self.calls.append(prompt)
        return _FakeResp(self._payload)
```

with:

```python
class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def call_structured(self, prompt, response_format, *, instructions):
        self.calls.append(prompt)
        return self._payload
```

If `_FakeResp` (line 83-85) is now unused in that file (check with `rg "_FakeResp" tests/unit/test_graph_growth.py`), delete it.

- [ ] **Step 7: Run the affected suites**

Run: `python -m pytest tests/unit/test_graph_growth.py tests/unit/test_claude_client.py tests/unit/test_ai_genre_enrichment.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add src/ai_genre_enrichment/client.py src/ai_genre_enrichment/graph_growth.py tests/unit/test_graph_growth.py tests/unit/test_claude_client.py
git commit -m "refactor(enrichment): provider-neutral call_structured; graph_growth uses it"
```

---

### Task 6: Provider factory (`src/ai_genre_enrichment/provider.py`)

**Files:**
- Create: `src/ai_genre_enrichment/provider.py`
- Test: `tests/unit/test_claude_client.py` (append)

**Design notes:** provider resolution order = explicit `provider=` arg → `PG_AI_PROVIDER` env var (test/CI seam) → `config.yaml` `ai_genre.provider` → default `claude_code`. The claude path rejects any non-off `web_mode` loudly (no web search support — Bandcamp is out of scope). Config reads are `lru_cache`d per path (CLI process lifetime).

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_claude_client.py`, add to the **top-of-file import block**:

```python
from src.ai_genre_enrichment.provider import (
    create_enrichment_client,
    get_enrichment_provider,
    resolve_enrichment_model,
)
```

Then append the tests:

```python


def test_factory_explicit_provider_openai():
    client = create_enrichment_client(provider="openai", model=None)
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"


def test_factory_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PG_AI_PROVIDER", "openai")
    cfg = _write_config(tmp_path, "ai_genre:\n  provider: claude_code\n")
    assert get_enrichment_provider(config_path=cfg) == "openai"


def test_factory_claude_default_model_from_config(monkeypatch, tmp_path):
    monkeypatch.delenv("PG_AI_PROVIDER", raising=False)
    cfg = _write_config(tmp_path, "ai_genre:\n  provider: claude_code\n  claude_model: sonnet\n")
    assert resolve_enrichment_model(None, config_path=cfg) == "sonnet"
    assert resolve_enrichment_model("haiku", config_path=cfg) == "haiku"  # explicit wins


def test_factory_claude_rejects_web_mode(monkeypatch):
    monkeypatch.delenv("PG_AI_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="web"):
        create_enrichment_client(provider="claude_code", web_mode="required")


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown"):
        create_enrichment_client(provider="copilot")


def test_factory_claude_client_construction(monkeypatch):
    # SDK may not be installed in CI; dry_run skips the construction guard.
    client = create_enrichment_client(provider="claude_code", model="sonnet", dry_run=True)
    assert client.provider == "claude_code"
    assert client.model == "sonnet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.provider'`

- [ ] **Step 3: Create the factory module**

Create `src/ai_genre_enrichment/provider.py`:

```python
"""Provider selection for enrichment LLM clients.

Resolution order: explicit ``provider=`` arg -> PG_AI_PROVIDER env var (test
seam) -> config.yaml ``ai_genre.provider`` -> 'claude_code'.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .claude_client import ClaudeCodeEnrichmentClient
from .client import OpenAIEnrichmentClient
from .routing import WebMode

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
CLAUDE_DEFAULT_MODEL = "haiku"
KNOWN_PROVIDERS = ("claude_code", "openai")


@lru_cache(maxsize=8)
def _config_ai_genre(config_path: str | None) -> tuple[str, str]:
    """(provider, claude_model) from config.yaml; safe defaults if unreadable."""
    try:
        from src.config_loader import Config

        cfg = Config(config_path) if config_path else Config()
        return cfg.ai_genre_provider, cfg.ai_genre_claude_model
    except (FileNotFoundError, KeyError, AttributeError):
        return "claude_code", CLAUDE_DEFAULT_MODEL


def get_enrichment_provider(config_path: str | None = None) -> str:
    provider = os.environ.get("PG_AI_PROVIDER") or _config_ai_genre(config_path)[0]
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(
            f"Unknown ai_genre.provider {provider!r}; expected one of {KNOWN_PROVIDERS}"
        )
    return provider


def resolve_enrichment_model(
    model: str | None = None, *, config_path: str | None = None
) -> str:
    """Explicit model wins; otherwise the active provider's default."""
    if model:
        return model
    if get_enrichment_provider(config_path) == "claude_code":
        return _config_ai_genre(config_path)[1]
    return OPENAI_DEFAULT_MODEL


def create_enrichment_client(
    *,
    model: str | None = None,
    dry_run: bool = False,
    web_mode: WebMode | str = WebMode.OFF,
    allowed_web_domains: list[str] | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    config_path: str | None = None,
) -> Any:
    if provider is None:
        provider = get_enrichment_provider(config_path)
    elif provider not in KNOWN_PROVIDERS:
        raise ValueError(
            f"Unknown ai_genre.provider {provider!r}; expected one of {KNOWN_PROVIDERS}"
        )
    if provider == "openai":
        return OpenAIEnrichmentClient(
            model=model or OPENAI_DEFAULT_MODEL,
            dry_run=dry_run,
            web_mode=web_mode,
            allowed_web_domains=allowed_web_domains,
            api_key=api_key,
            max_retries=max_retries,
        )
    if WebMode(web_mode) != WebMode.OFF:
        raise ValueError(
            "ai_genre.provider=claude_code does not support web search; "
            "this call requires web_mode='off' (Bandcamp locator is OpenAI-only and unwired)"
        )
    return ClaudeCodeEnrichmentClient(
        model=model or _config_ai_genre(config_path)[1],
        dry_run=dry_run,
        max_retries=max_retries,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_claude_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/provider.py tests/unit/test_claude_client.py
git commit -m "feat(enrichment): provider factory with PG_AI_PROVIDER override"
```

---

### Task 7: Switch `tag_adjudicator` to the factory client (fail loud)

**Files:**
- Modify: `src/ai_genre_enrichment/tag_adjudicator.py`
- Modify: `src/ai_genre_enrichment/storage.py` (`classify_source_tags` signature, ~line 1197)
- Test: `tests/unit/test_ai_genre_enrichment.py:3441-3486`

**Behavior change (intentional, per spec):** `adjudicate_tags` no longer returns `{}` when the backend is unavailable or errors — it raises. The user explicitly requested AI adjudication (`--adjudicate`); an unavailable backend is an error, not a silent skip. The existing graceful `{}` returns for empty input and dry-run stay.

- [ ] **Step 1: Rewrite the existing adjudicator test to inject a fake client**

In `tests/unit/test_ai_genre_enrichment.py`, replace `test_adjudicate_tags_returns_classifications` (lines 3441-3486) with:

```python
def test_adjudicate_tags_returns_classifications():
    from src.ai_genre_enrichment.tag_adjudicator import adjudicate_tags

    fake_response = {
        "tag_classifications": [
            {
                "raw_tag": "ambient pop",
                "normalized_tag": "ambient pop",
                "classification": "genre_style",
                "confidence": 0.85,
                "reason": "Recognized subgenre.",
            },
            {
                "raw_tag": "stage & screen",
                "normalized_tag": "stage & screen",
                "classification": "review_only",
                "confidence": 0.30,
                "reason": "Market category, not a genre.",
            },
        ],
        "warnings": [],
    }

    class FakeClient:
        last_token_usage = {"input_tokens": 5, "output_tokens": 5}

        def call_structured(self, prompt, response_format, *, instructions):
            assert "ambient pop" in prompt
            assert "Classify source-provided release tags" in instructions
            return fake_response

    results = adjudicate_tags(
        [("ambient pop", "ambient pop"), ("stage & screen", "stage & screen")],
        client=FakeClient(),
    )

    assert len(results) == 2
    assert results["ambient pop"]["classification"] == "genre_style"
    assert results["stage & screen"]["classification"] == "review_only"


def test_adjudicate_tags_propagates_backend_failure():
    from src.ai_genre_enrichment.tag_adjudicator import adjudicate_tags

    class DeadClient:
        def call_structured(self, prompt, response_format, *, instructions):
            raise RuntimeError("Claude Code request failed after retries: boom")

    with pytest.raises(RuntimeError, match="failed after retries"):
        adjudicate_tags([("tag", "tag")], client=DeadClient())
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/unit/test_ai_genre_enrichment.py -k adjudicate_tags -v`
Expected: FAIL with `TypeError: adjudicate_tags() got an unexpected keyword argument 'client'`

- [ ] **Step 3: Rewrite `adjudicate_tags`**

In `src/ai_genre_enrichment/tag_adjudicator.py`:

(a) Delete the module-level OpenAI import block (lines 7, 13-16):

```python
import os
...
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]
```

(keep `import json`? — no longer needed either; delete it. Keep `logging`, `deepcopy`, `Any`.)

(b) Replace the whole `adjudicate_tags` function (from `def adjudicate_tags(` to end of file) with:

```python
def adjudicate_tags(
    tags: list[tuple[str, str]],
    *,
    model: str | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Call AI to classify a batch of unknown tags.

    Args:
        tags: List of (raw_tag, normalized_tag) pairs to classify.
        model: Model override; None uses the active provider's default.
        dry_run: If True, return empty results without calling the backend.
        client: Injected enrichment client (tests); None builds one via the factory.

    Returns:
        Dict keyed by normalized_tag -> {"classification", "confidence", "reason"}.

    Raises:
        RuntimeError/ValueError: backend unavailable or call failed after retries.
        An explicitly requested adjudication that cannot run is an error, not a
        silent no-op.

    Note: No hard batch-size limit is enforced here. At scale, callers should
    chunk large batches to avoid token-limit failures (the analyze 'enrich'
    stage uses call_structured_batch for that).
    """
    if not tags or dry_run:
        return {}

    if client is None:
        from .provider import create_enrichment_client

        client = create_enrichment_client(model=model)

    tag_list = "\n".join(f"- raw: {raw!r}, normalized: {norm!r}" for raw, norm in tags)
    prompt = f"Classify the following source tags:\n\n{tag_list}"

    data = client.call_structured(
        prompt,
        tag_adjudicator_response_format(),
        instructions=TAG_ADJUDICATOR_INSTRUCTIONS,
    )

    results: dict[str, dict[str, Any]] = {}
    for item in data.get("tag_classifications", []):
        norm = item.get("normalized_tag", "").strip().casefold()
        if norm:
            results[norm] = {
                "classification": item["classification"],
                "confidence": item["confidence"],
                "reason": item.get("reason", ""),
            }

    usage = getattr(client, "last_token_usage", None) or {}
    logger.info(
        "AI adjudication: %d tags, %d input + %d output tokens",
        len(tags),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return results
```

(c) In `src/ai_genre_enrichment/storage.py`, change the `classify_source_tags` signature default (~line 1197) from:

```python
        model: str = "gpt-4o-mini",
```

to:

```python
        model: str | None = None,
```

(`adjudicate_tags(ai_input, model=model)` at line 1341 then passes `None` through to the factory, which resolves the provider default.)

- [ ] **Step 4: Run the affected suites**

Run: `python -m pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_claude_client.py -v`
Expected: all PASS. If other tests in the file relied on the old "no API key -> returns {}" behavior, update them to inject a fake client or expect the raise (the silent-skip behavior is intentionally gone).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/tag_adjudicator.py src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "refactor(enrichment): tag adjudicator uses provider factory, fails loud"
```

---

### Task 8: Switch `scripts/ai_genre_enrich.py` + `run_model_prior_album_tests.py` to the factory

**Files:**
- Modify: `scripts/ai_genre_enrich.py` (imports; `main()`; `--model` defaults at lines 124, 200, 220, 231, 337, 344, 380, 392; client sites at 1210, 1887, 1942, 2308; `provider="openai"` at 1875, 1908, 1931, 1962; `model =` at 1928, 581)
- Modify: `scripts/run_model_prior_album_tests.py:61`
- Test: existing suites (`test_ai_genre_hybrid_cli.py`, `test_ai_genre_model_prior.py`, `test_ai_genre_hybrid_evidence.py`) pinned to the OpenAI provider where they patch the OpenAI class

**Key idea:** `main()` resolves `args.model` once via `resolve_enrichment_model`, so every downstream use (`store.record_*`, cache lookups, client construction) sees a concrete model name. `provider="openai"` hardcodes become `get_enrichment_provider()` — this means cached model priors are keyed by the live provider, so switching providers correctly invalidates the prior cache.

- [ ] **Step 1: Pin existing CLI tests to the OpenAI provider**

The hybrid-CLI and model-prior tests patch `src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai` / `.request_structured` at class level (e.g. `tests/unit/test_ai_genre_hybrid_cli.py:38,108`). With the factory defaulting to `claude_code`, those patches would never be hit. Add an autouse fixture **at the top of each of these files** (`tests/unit/test_ai_genre_hybrid_cli.py`, `tests/unit/test_ai_genre_model_prior.py`):

```python
@pytest.fixture(autouse=True)
def _pin_openai_provider(monkeypatch):
    """These tests stub the OpenAI client class; pin the factory to it."""
    monkeypatch.setenv("PG_AI_PROVIDER", "openai")
```

(If a file lacks `import pytest`, add it.)

- [ ] **Step 2: Update imports and `main()` in `scripts/ai_genre_enrich.py`**

(a) Add to the imports block (next to the existing `from src.ai_genre_enrichment...` imports):

```python
from src.ai_genre_enrichment.provider import (
    create_enrichment_client,
    get_enrichment_provider,
    resolve_enrichment_model,
)
```

(b) In `main()`, immediately after `args = parser.parse_args(argv)` (line 56), add:

```python
    # Resolve --model once: explicit flag wins, else the active provider's
    # default. Downstream store records and cache lookups need a concrete name.
    if hasattr(args, "model") and args.model is None:
        args.model = resolve_enrichment_model(None)
```

- [ ] **Step 3: Change `--model` defaults to None**

Replace `default=DEFAULT_MODEL` with `default=None` on these `add_argument("--model", ...)` lines: 124 (shared), 200 (classify), 220 (ingest-local), 231 (extract-lastfm), 337 (model-prior-one), 344 (model-prior), 380 (hybrid-enrich-one), 392 (propose-growth). **Leave line 248 (`extract_bandcamp`) unchanged** — Bandcamp stays OpenAI-only and unwired. Keep the `DEFAULT_MODEL` constant (line 43); it is still the bandcamp default.

Also update the two `getattr` fallbacks:
- Line 581 (`cmd_classify_tags`): `model = getattr(args, "model", DEFAULT_MODEL)` → `model = getattr(args, "model", None) or resolve_enrichment_model(None)`
- Line 1928 (`_ensure_model_prior_for_hybrid`): `model = getattr(args, "model", DEFAULT_MODEL)` → `model = getattr(args, "model", None) or resolve_enrichment_model(None)`

- [ ] **Step 4: Switch the four client constructions to the factory**

Site 1 (~line 1210, run/run-one enrichment):

```python
        client = OpenAIEnrichmentClient(
            model=args.model,
            dry_run=args.dry_run,
            web_mode=effective_web_mode,
            allowed_web_domains=getattr(args, "allowed_web_domains", None),
            api_key=_api_key,
        )
```
→
```python
        client = create_enrichment_client(
            model=args.model,
            dry_run=args.dry_run,
            web_mode=effective_web_mode,
            allowed_web_domains=getattr(args, "allowed_web_domains", None),
            api_key=_api_key,
        )
```

Site 2 (~line 1887, model-prior-one):

```python
    client = OpenAIEnrichmentClient(model=args.model, dry_run=args.dry_run, web_mode="off")
```
→
```python
    client = create_enrichment_client(model=args.model, dry_run=args.dry_run, web_mode="off")
```

Site 3 (~line 1942, `_ensure_model_prior_for_hybrid`):

```python
    client = OpenAIEnrichmentClient(model=model, dry_run=False, web_mode="off")
```
→
```python
    client = create_enrichment_client(model=model, dry_run=False, web_mode="off")
```

Site 4 (~line 2308, propose-growth):

```python
    client = OpenAIEnrichmentClient(model=args.model, api_key=api_key,
                                    web_mode=args.web_mode)
```
→
```python
    client = create_enrichment_client(model=args.model, api_key=api_key,
                                      web_mode=args.web_mode)
```

(The claude path raises ValueError if someone passes a non-off `--web-mode` to propose-growth — loud by design.)

If `OpenAIEnrichmentClient` is no longer referenced in the file (check: `rg "OpenAIEnrichmentClient" scripts/ai_genre_enrich.py`), remove it from the imports.

- [ ] **Step 5: Make the model-prior provider string dynamic**

At lines 1875, 1908, 1931, 1962 replace `provider="openai"` with `provider=get_enrichment_provider()`. Example (line 1875):

```python
            release_key=release.release_key, provider="openai", model=args.model,
```
→
```python
            release_key=release.release_key, provider=get_enrichment_provider(), model=args.model,
```

- [ ] **Step 6: Switch `scripts/run_model_prior_album_tests.py`**

Line 61:

```python
    client = OpenAIEnrichmentClient(model="gpt-4o-mini", web_mode="off")
```
→
```python
    client = create_enrichment_client(web_mode="off")
```

and change its import of `OpenAIEnrichmentClient` to:

```python
from src.ai_genre_enrichment.provider import create_enrichment_client
```

- [ ] **Step 7: Run the full enrichment-adjacent suites**

Run: `python -m pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_ai_genre_model_prior.py tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_graph_growth.py tests/unit/test_claude_client.py tests/unit/test_bandcamp_locator.py -v`
Expected: all PASS. Failures here are almost certainly (a) a test constructing a client via a path now routed through the factory without `PG_AI_PROVIDER=openai` — add the autouse fixture to that file too; or (b) an assertion on `provider="openai"` in stored rows — those tests write their own rows and should still pass; do NOT weaken assertions, find the actual call path.

- [ ] **Step 8: Lint**

Run: `ruff check scripts/ai_genre_enrich.py scripts/run_model_prior_album_tests.py src/ai_genre_enrichment/`
Expected: clean (catches unused imports left behind).

- [ ] **Step 9: Commit**

```bash
git add scripts/ai_genre_enrich.py scripts/run_model_prior_album_tests.py tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_ai_genre_model_prior.py
git commit -m "refactor(enrichment): all CLI call sites use the provider factory"
```

---

### Task 9: Dependency, docs touch, full verification

**Files:**
- Modify: `pyproject.toml:33-35` (`ai` extra)
- Modify: `docs/AI_GENRE_ENRICHMENT.md` (provider section)

- [ ] **Step 1: Add the SDK dependency**

In `pyproject.toml`, extend the `ai` extra:

```toml
ai = [
    "openai>=1.68.0",
    "claude-agent-sdk>=0.1.0",
]
```

Then install: `pip install -e .[ai]`
Expected: `claude-agent-sdk` installs cleanly.

- [ ] **Step 2: Smoke the real backend once (manual, not CI)**

Run: `python -c "from src.ai_genre_enrichment.provider import create_enrichment_client; c = create_enrichment_client(provider='claude_code', model='haiku'); print(c.call_structured('Return the genre of the band Slowdive as JSON.', {'schema': {'type': 'object', 'properties': {'genre': {'type': 'string'}}, 'required': ['genre']}}, instructions='You classify music genres. JSON only.'))"`
Expected: a dict like `{'genre': 'shoegaze'}` printed within ~10-30s. This verifies CLI auth + SDK wiring end-to-end on the Max subscription. If it raises RuntimeError about the SDK or CLI, fix the environment before proceeding — do not mark this step done on a failure.

- [ ] **Step 3: Document the provider in `docs/AI_GENRE_ENRICHMENT.md`**

Add a short section (wherever the doc currently mentions `OPENAI_API_KEY` / model selection):

```markdown
## LLM provider

The enrichment pipeline calls its LLM through a provider factory
(`src/ai_genre_enrichment/provider.py`), configured in `config.yaml`:

```yaml
ai_genre:
  provider: claude_code   # default — local Claude Code (Agent SDK, subscription auth)
  claude_model: haiku     # 'haiku' | 'sonnet' | 'opus'
```

- `claude_code` (default): requires Claude Code installed and authenticated
  (`claude` login) plus `pip install -e .[ai]`. No API key; usage draws on the
  Claude subscription's rate windows. No web search support.
- `openai` (legacy): requires `OPENAI_API_KEY`; still used by the unwired
  `extract-bandcamp` subcommand.
- `PG_AI_PROVIDER` env var overrides the config (used by tests).
- Costs: claude_code calls report token usage only (`estimated_cost_usd` is
  null — subscription usage is not billable per token).
```

- [ ] **Step 4: Full test suite**

Run: `python -m pytest -m "not slow" -q`
Expected: pass, modulo the documented pre-existing deselect list (see memory `project_sp3a_taxonomy_growth`: 13 known pre-existing failures unrelated to this work). Any NEW failure must be fixed before commit.

- [ ] **Step 5: Lint + types**

Run: `ruff check src/ai_genre_enrichment/ scripts/ && mypy src/ai_genre_enrichment/claude_client.py src/ai_genre_enrichment/provider.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml docs/AI_GENRE_ENRICHMENT.md
git commit -m "chore(enrichment): claude-agent-sdk dependency + provider docs"
```

---

## Out of scope (later plans)

- **Phase 2** (`lastfm`/`enrich`/`publish` analyze stages, batch wiring into `classify_source_tags` via `call_structured_batch`): separate plan after this lands.
- **Phase 3** (web `/api/tools/*` + Tools panel): separate plan.
- `bandcamp_enrichment.py`, `extract-bandcamp`, `prepare-batch`/`collect-batch` (OpenAI Batch API era): untouched.
- Removing the OpenAI client: not planned; it remains the test double surface and the bandcamp path's backend.

