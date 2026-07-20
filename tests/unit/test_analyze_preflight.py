# tests/unit/test_analyze_preflight.py
"""Pre-flight: configured+broken aborts early; passing proceeds; escape hatch bypasses."""
import argparse
import pytest

import scripts.analyze_library as al
from src.setup.result import CheckResult


def _args(**kw):
    ns = argparse.Namespace(config="config.yaml", no_preflight=False)
    ns.__dict__.update(kw)
    return ns


def _cfg(preflight=True):
    class C:
        def get(self, *keys, default=None):
            if keys == ("analyze", "preflight"):
                return preflight
            return default
    return C()


def test_preflight_aborts_on_configured_failure(monkeypatch):
    monkeypatch.setattr(al, "test_connection",
                        lambda svc, cfg: CheckResult(svc, "fail", "unreachable", "fix it"))
    with pytest.raises(RuntimeError, match="pre-flight"):
        al._run_preflight(_cfg(), _args(), ["lastfm", "sonic"])


def test_preflight_passes_when_ok(monkeypatch):
    monkeypatch.setattr(al, "test_connection",
                        lambda svc, cfg: CheckResult(svc, "pass", "ok"))
    al._run_preflight(_cfg(), _args(), ["lastfm", "sonic"])  # no raise


def test_preflight_ignores_unscheduled_services(monkeypatch):
    seen = []
    monkeypatch.setattr(al, "test_connection",
                        lambda svc, cfg: seen.append(svc) or CheckResult(svc, "pass", "ok"))
    al._run_preflight(_cfg(), _args(), ["sonic", "muq"])  # no network stages
    assert seen == []  # nothing pre-flighted


def test_no_preflight_flag_bypasses(monkeypatch):
    monkeypatch.setattr(al, "test_connection",
                        lambda svc, cfg: CheckResult(svc, "fail", "unreachable", "fix"))
    al._run_preflight(_cfg(), _args(no_preflight=True), ["lastfm"])  # no raise


def test_config_preflight_false_bypasses(monkeypatch):
    monkeypatch.setattr(al, "test_connection",
                        lambda svc, cfg: CheckResult(svc, "fail", "unreachable", "fix"))
    al._run_preflight(_cfg(preflight=False), _args(), ["lastfm"])  # no raise
