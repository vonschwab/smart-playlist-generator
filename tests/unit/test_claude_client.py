"""Tests for the Claude-Code enrichment client, provider factory, and config plumbing."""
from __future__ import annotations

import json
import sys

import pytest

from src.ai_genre_enrichment.claude_client import (
    ClaudeCodeEnrichmentClient,
    _parse_json_text,
)
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
        "library:\n  database_path: data/metadata.db\nai_genre:\n  provider: openai\n  claude_model: sonnet\n",
    ))
    assert cfg.ai_genre_provider == "openai"
    assert cfg.ai_genre_claude_model == "sonnet"


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
