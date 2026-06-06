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
from src.ai_genre_enrichment.routing import WebMode


class _Resp:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


def test_locator_enables_web_search_by_default(monkeypatch):
    """Regression: the locator MUST run with web search, or it hallucinates
    URLs from memory that 404 (the original full-library failure)."""
    import src.ai_genre_enrichment.client as client_mod

    captured = {}

    class _FakeClient:
        def __init__(self, *, model, api_key, web_mode):
            captured["web_mode"] = web_mode

        def _call_openai(self, prompt, response_format, *, instructions):
            return _Resp({"candidate_sources": [], "warnings": []})

    monkeypatch.setattr(client_mod, "OpenAIEnrichmentClient", _FakeClient)
    bandcamp_enrichment._locate_bandcamp_url(artist="a", album="b", model="m", api_key="k")
    assert captured["web_mode"] == WebMode.REQUIRED


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


def test_fetch_bandcamp_tags_handles_html_404_as_empty(monkeypatch):
    """A located-but-unresolvable URL (e.g. hallucinated → 404) yields empty
    tags without raising, so the caller records a miss rather than crashing."""
    import urllib.error

    monkeypatch.setattr(
        bandcamp_enrichment, "_locate_bandcamp_url",
        lambda **_kw: {
            "candidate_sources": [{
                "source_url": "https://caribou.bandcamp.com/album/mixtape-2020",
                "source_type": "bandcamp_release",
                "source_name": "Bandcamp",
                "identity_status": "confirmed",
                "identity_confidence": 0.9,
                "release_specific": True,
                "reason": "x",
            }],
            "warnings": [],
        },
    )

    def boom(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    url, tags, confidence = bandcamp_enrichment.fetch_bandcamp_tags(
        artist="caribou", album="mixtape 2020", api_key="k", fetch_html=boom,
    )
    assert url == "https://caribou.bandcamp.com/album/mixtape-2020"
    assert tags == []
    assert confidence == 0.9
