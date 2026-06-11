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
