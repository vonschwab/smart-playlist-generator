"""Service probes: not-configured→pass, configured+ok→pass, configured+broken→fail; never throw.

Every probe is exercised without touching the network — either the
config is deliberately absent (the probe returns before any I/O) or the
`_no_real_network` autouse fixture below replaces `requests.get` with a
stub, so `test_all_services_dispatch` (which calls every SERVICES id,
including the keyless `musicbrainz` probe) can't escape to a real host.
"""
import pytest
import requests

from src.setup.result import CheckResult
from src.setup import services


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Block any probe that falls through to a real `requests.get` call."""
    def _blocked_get(*a, **k):
        raise RuntimeError("network access blocked in tests")
    monkeypatch.setattr(requests, "get", _blocked_get)


def test_lastfm_not_configured_is_pass():
    r = services.test_connection("lastfm", {"lastfm": {}})
    assert r.status == "pass" and "not configured" in r.summary.lower()


def test_lastfm_ok(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def get_user_info(self): return {"name": "dylan"}
    monkeypatch.setattr(services, "LastFMClient", FakeClient, raising=False)
    r = services.test_connection("lastfm", {"lastfm": {"api_key": "k", "username": "dylan"}})
    assert r.status == "pass"


def test_lastfm_bad_key_is_fail(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def get_user_info(self): raise RuntimeError("Invalid API key")
    monkeypatch.setattr(services, "LastFMClient", FakeClient, raising=False)
    r = services.test_connection("lastfm", {"lastfm": {"api_key": "bad", "username": "x"}})
    assert r.status == "fail" and r.fix_hint is not None


def test_openai_key_presence(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert services.test_connection("openai", {}).status == "fail"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert services.test_connection("openai", {}).status == "pass"


def test_probe_never_throws(monkeypatch):
    """A probe that raises unexpectedly becomes a fail result, not an exception."""
    def boom(*a, **k): raise ValueError("kaboom")
    monkeypatch.setattr(services, "_probe_discogs", boom, raising=False)
    r = services.test_connection("discogs", {"discogs": {"token": "t"}})
    assert isinstance(r, CheckResult) and r.status == "fail"


def test_all_services_dispatch():
    for s in services.SERVICES:
        r = services.test_connection(s, {})  # all not-configured → pass/warn, never raise
        assert isinstance(r, CheckResult)
