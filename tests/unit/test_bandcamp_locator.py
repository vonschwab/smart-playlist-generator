"""Tests for the Bandcamp source-locator retry / fail-loud behavior.

The locator must distinguish a genuine "nothing found" (empty candidates,
safe to cache as a miss) from a "call failed" (must raise, so the run does
not poison the attempt ledger with false misses).
"""

from __future__ import annotations

import json

import pytest

from src.ai_genre_enrichment import bandcamp_enrichment
from src.ai_genre_enrichment.client import OpenAIEnrichmentClient


class _Resp:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


def test_locate_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}
    payload = {"candidate_sources": [], "warnings": []}

    def flaky(self, prompt, response_format, *, instructions):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient 503")
        return _Resp(payload)

    monkeypatch.setattr(OpenAIEnrichmentClient, "_call_openai", flaky)
    monkeypatch.setattr(bandcamp_enrichment.time, "sleep", lambda *_: None)

    result = bandcamp_enrichment._locate_bandcamp_url(
        artist="a", album="b", model="m", api_key="k"
    )
    assert result == payload
    assert calls["n"] == 2  # retried once


def test_locate_raises_on_persistent_failure(monkeypatch):
    def always_fail(self, prompt, response_format, *, instructions):
        raise RuntimeError("api down")

    monkeypatch.setattr(OpenAIEnrichmentClient, "_call_openai", always_fail)
    monkeypatch.setattr(bandcamp_enrichment.time, "sleep", lambda *_: None)

    # Must raise (NOT return empty) so the caller records a retryable failure,
    # not a permanent miss.
    with pytest.raises(RuntimeError):
        bandcamp_enrichment._locate_bandcamp_url(
            artist="a", album="b", model="m", api_key="k", max_retries=2
        )


def test_locate_empty_result_is_returned_not_raised(monkeypatch):
    # A successful call that finds nothing is a genuine miss — returned, not raised.
    monkeypatch.setattr(
        OpenAIEnrichmentClient, "_call_openai",
        lambda self, p, rf, *, instructions: _Resp({"candidate_sources": [], "warnings": []}),
    )
    result = bandcamp_enrichment._locate_bandcamp_url(
        artist="a", album="b", model="m", api_key="k"
    )
    assert result == {"candidate_sources": [], "warnings": []}
