from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def test_request_structured_uses_supplied_validator(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    seen = []

    class Response:
        output_text = '{"genres":[],"warnings":[]}'
        usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", api_key="test", max_retries=0)
    monkeypatch.setattr(client, "_call_openai", lambda *_args, **_kwargs: Response())

    result = client.request_structured(
        payload={"artist": "Duster"},
        prompt="classify",
        response_format={"type": "json_schema", "name": "prior", "schema": {}},
        validator=lambda value: seen.append(value) or value,
        instructions="No web.",
        estimated_output_tokens=300,
    )

    assert result.status == "complete"
    assert result.response_json == {"genres": [], "warnings": []}
    assert seen == [{"genres": [], "warnings": []}]


def test_request_structured_dry_run_does_not_call_openai(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", dry_run=True)
    monkeypatch.setattr(
        client,
        "_call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called")),
    )

    result = client.request_structured(
        payload={"artist": "Duster"},
        prompt="classify",
        response_format={"type": "json_schema", "name": "prior", "schema": {}},
        validator=lambda value: value,
        instructions="No web.",
        estimated_output_tokens=300,
    )

    assert result.status == "skipped"
    assert result.response_json["dry_run"] is True
    assert result.response_json["estimated_output_tokens"] == 300
