"""Tests for the Last.fm tag fetcher: retry/backoff, junk-artist skip."""

from __future__ import annotations

import requests

from src.ai_genre_enrichment import lastfm_enrichment as lf


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


_OK_PAYLOAD = {"toptags": {"tag": [{"name": "shoegaze"}, {"name": "dream pop"}]}}


def test_transient_504_then_success_retries(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(504)
        return _Resp(200, _OK_PAYLOAD)

    monkeypatch.setattr(lf.time, "sleep", lambda *_: None)
    monkeypatch.setattr(lf.requests, "get", fake_get)
    tags = lf._fetch_toptags("artist.gettoptags", "k", artist="acetone")
    assert tags == ["shoegaze", "dream pop"]
    assert calls["n"] == 2  # retried once


def test_persistent_504_returns_empty_no_raise(monkeypatch):
    monkeypatch.setattr(lf.time, "sleep", lambda *_: None)
    monkeypatch.setattr(lf.requests, "get", lambda *a, **k: _Resp(504))
    assert lf._fetch_toptags("artist.gettoptags", "k", artist="x", max_retries=3) == []


def test_network_error_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(lf.time, "sleep", lambda *_: None)
    monkeypatch.setattr(lf.requests, "get", boom)
    assert lf._fetch_toptags("artist.gettoptags", "k", artist="x") == []


def test_junk_artist_skipped_without_api_call(monkeypatch):
    def must_not_call(*a, **k):
        raise AssertionError("API should not be called for junk artist")

    monkeypatch.setattr(lf.requests, "get", must_not_call)
    assert lf.fetch_lastfm_tags(artist="@", api_key="k") == []
    assert lf.fetch_lastfm_tags(artist="   ", api_key="k") == []


def test_non_latin_artist_is_not_skipped(monkeypatch):
    seen = {"called": False}

    def fake_get(url, params=None, timeout=None):
        seen["called"] = True
        return _Resp(200, _OK_PAYLOAD)

    monkeypatch.setattr(lf.requests, "get", fake_get)
    tags = lf.fetch_lastfm_tags(artist="電気グルーヴ", api_key="k")
    assert seen["called"] is True
    assert "shoegaze" in tags
