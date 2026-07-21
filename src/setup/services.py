# src/setup/services.py
"""Uniform `test_connection()` probes for each optional external service.

One place to construct-and-probe each integration MixArc can
talk to, reusing the client's cheapest existing identity/reachability call
(no new API surface on the wire). A later pre-flight task wires these into
`analyze_library.py`.

Semantics (mirrors `src/setup/checks.py` / SP-1's `CheckResult` contract):
  - Not configured (missing key/token/URL) -> `pass`, "not configured —
    optional". These are all optional enrichment integrations (design
    principle 14: local-first; external APIs enrich, they never gate).
    Exception: `openai` / `anthropic_api` check only an env var and have no
    "unconfigured" state distinct from "key absent" -> that's a `fail`.
  - Configured but the probe fails or raises -> `fail`, with a `fix_hint`.
  - Configured and reachable -> `pass`, "connected as <identity>".
  - `musicbrainz` is informational only (keyless, never gates anything): a
    failed ping is `warn`, never `fail`.

`test_connection()` NEVER raises: every probe body is either self-contained
(catches its own exceptions, e.g. musicbrainz) or is invoked through a
wrapper here that turns any exception into a `fail` CheckResult.

Every probe also runs under a hard ~5s wall-clock budget (`_run_with_timeout`,
below), not just the ones that happen to make a raw `requests.get` call.
`LastFMClient.get_user_info()` in particular has no timeout/retry-count seam
of its own to reuse -- its `_make_request` retries up to 5x with exponential
backoff on top of a 10s per-attempt timeout, i.e. 65+ seconds worst case --
so the bound is enforced generically at the probe-dispatch level instead,
in a background thread. A later pre-flight task wires these into
`analyze_library.py` and assumes each call returns in ~5s; this is what
makes that assumption true regardless of which client is misbehaving.

Service clients are imported lazily, inside each probe, so a missing
optional dependency (e.g. no `claude_agent_sdk` installed) can never break
`import src.setup.services` itself. `LastFMClient` is additionally exposed
as a module-level name (defaulting to `None`, resolved on first real use)
so tests can `monkeypatch.setattr(services, "LastFMClient", FakeClient)`
without needing the real client/its dependency chain importable.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import requests

from src.setup.result import CheckResult

SERVICES: tuple[str, ...] = (
    "lastfm",
    "discogs",
    "plex",
    "claude_code",
    "openai",
    "anthropic_api",
    "musicbrainz",
)

_TIMEOUT_SECONDS = 5

# Hard wall-clock budget for an entire probe call (see `_run_with_timeout`).
# Deliberately a module global (not a default-arg binding) so tests can
# `monkeypatch.setattr(services, "_PROBE_TIMEOUT_SECONDS", ...)` to shrink it.
_PROBE_TIMEOUT_SECONDS = 5.0

# Test seam: left None here (no eager import), resolved lazily in
# _probe_lastfm on first real use. Tests monkeypatch this attribute
# directly; production code falls back to the real `src.lastfm_client`.
LastFMClient = None

_FIX_HINTS: dict[str, str] = {
    "lastfm": "check lastfm.api_key / username",
    "discogs": "check discogs.token",
    "plex": "check plex.base_url / plex.token",
    "claude_code": "authenticate Claude Code / `claude login`",
    "openai": "set OPENAI_API_KEY",
    "anthropic_api": "set ANTHROPIC_API_KEY",
}


def _cfg(config: Any, section: str, key: str, default: Any = None) -> Any:
    """Read `section.key` off a plain dict or a `Config`-like object.

    `Config.get(section, key, default)` (src/config_loader.py) is a 3-arg
    section/key getter, NOT dict's 2-arg `.get(key, default)` — this
    bridges both call shapes so `test_connection` accepts either, per the
    Task 3 contract ("config is a plain dict OR a Config object").
    """
    if isinstance(config, dict):
        return (config.get(section) or {}).get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(section, key, default)
        except TypeError:
            pass
    sub = getattr(config, section, None)
    if isinstance(sub, dict):
        return sub.get(key, default)
    return default


def _user_agent() -> str:
    try:
        from importlib.metadata import version
        v = version("mixarc")
    except Exception:
        v = "dev"
    return f"mixarc/{v} (+https://github.com/vonschwab/mixarc)"


def _run_with_timeout(service: str, fn: Any) -> CheckResult:
    """Run a probe callable with a hard ~5s wall-clock budget.

    Some clients (Last.fm's in particular) retry internally well past what
    any single `requests` `timeout=` kwarg bounds -- up to 65+ seconds in
    the worst case. Rather than let one slow/hung service stall an entire
    pre-flight sweep, this returns a `fail` result once the budget elapses
    instead of waiting on the client. The orphaned daemon thread cannot be
    force-killed and is simply abandoned; it will not block process exit.

    This is also the single place every probe's exceptions are caught, so
    `test_connection` never raises regardless of which probe is dispatched
    (mirrors the old per-call `try/except` this replaces, generically).
    """
    result: list[CheckResult] = []

    def _target() -> None:
        try:
            result.append(fn())
        except Exception as exc:  # any probe error -> fail (never propagate)
            # musicbrainz is informational-only and never fails, even on an
            # unexpected exception -- matches its own internal contract.
            status = "warn" if service == "musicbrainz" else "fail"
            result.append(CheckResult(
                service, status, f"{service} probe error: {exc}",
                fix_hint=None if status == "warn" else _FIX_HINTS.get(service),
            ))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(_PROBE_TIMEOUT_SECONDS)
    if t.is_alive() or not result:
        return CheckResult(
            service, "fail", f"{service} timed out after {_PROBE_TIMEOUT_SECONDS:.0f}s",
            fix_hint="service slow or unreachable; check network/credentials",
        )
    return result[0]


def _lastfm_credentials(config: Any) -> tuple[str, str] | None:
    api_key = _cfg(config, "lastfm", "api_key")
    username = _cfg(config, "lastfm", "username")
    if not api_key or not username:
        return None
    return api_key, username


def _discogs_token(config: Any) -> str | None:
    return _cfg(config, "discogs", "token") or None


def _plex_credentials(config: Any) -> tuple[str, str] | None:
    enabled = _cfg(config, "plex", "enabled", False)
    base_url = _cfg(config, "plex", "base_url")
    token = _cfg(config, "plex", "token")
    if not enabled or not base_url or not token:
        return None
    return base_url, token


def _not_configured_result(service: str, config: Any) -> CheckResult | None:
    """Fast, synchronous "not configured -> pass" check for the optional
    services that have one, run BEFORE `_run_with_timeout` would spawn a
    thread -- so an unconfigured service returns instantly, no thread
    involved. Returns `None` when the service has no such fast-path (it
    always goes through the timed probe)."""
    if service == "lastfm" and _lastfm_credentials(config) is None:
        return CheckResult("lastfm", "pass", "not configured — optional")
    if service == "discogs" and _discogs_token(config) is None:
        return CheckResult("discogs", "pass", "not configured — optional")
    if service == "plex" and _plex_credentials(config) is None:
        return CheckResult("plex", "pass", "not configured — optional")
    return None


def test_connection(service: str, config: Any) -> CheckResult:
    """Probe one service by id. Dispatches by name; never raises."""
    probe = globals().get(f"_probe_{service}")
    if probe is None:
        return CheckResult(service, "fail", f"unknown service: {service}")
    early = _not_configured_result(service, config)
    if early is not None:
        return early
    return _run_with_timeout(service, lambda: probe(config))


def _probe_lastfm(config: Any) -> CheckResult:
    creds = _lastfm_credentials(config)
    if creds is None:
        return CheckResult("lastfm", "pass", "not configured — optional")
    api_key, username = creds

    client_cls = LastFMClient
    if client_cls is None:
        from src.lastfm_client import LastFMClient as client_cls  # lazy: optional dep chain

    client = client_cls(api_key, username)
    info = client.get_user_info()
    if not info:
        return CheckResult(
            "lastfm", "fail", "lastfm unreachable: get_user_info() returned no data",
            fix_hint=_FIX_HINTS["lastfm"],
        )
    who = info.get("username") if isinstance(info, dict) else None
    return CheckResult("lastfm", "pass", f"connected as {who or username}")


def _probe_discogs(config: Any) -> CheckResult:
    token = _discogs_token(config)
    if token is None:
        return CheckResult("discogs", "pass", "not configured — optional")

    resp = requests.get(
        "https://api.discogs.com/oauth/identity",
        headers={"Authorization": f"Discogs token={token}", "User-Agent": _user_agent()},
        timeout=_TIMEOUT_SECONDS,
    )
    if resp.status_code == 200:
        data = resp.json()
        return CheckResult("discogs", "pass", f"connected as {data.get('username', 'unknown')}")
    return CheckResult(
        "discogs", "fail", f"discogs unreachable: HTTP {resp.status_code}",
        fix_hint=_FIX_HINTS["discogs"],
    )


def _probe_plex(config: Any) -> CheckResult:
    creds = _plex_credentials(config)
    if creds is None:
        return CheckResult("plex", "pass", "not configured — optional")
    base_url, token = creds

    from src.plex_exporter import PlexExporter  # lazy: optional integration

    exporter = PlexExporter(base_url, token, timeout=_TIMEOUT_SECONDS)
    machine_id = exporter.test_connection()
    return CheckResult("plex", "pass", f"connected (machineIdentifier={machine_id})")


def _probe_claude_code(config: Any) -> CheckResult:
    try:
        from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient  # lazy: optional dep

        ClaudeCodeEnrichmentClient._ensure_sdk()
    except Exception as exc:
        return CheckResult(
            "claude_code", "fail", f"claude_code unreachable: {exc}",
            fix_hint=_FIX_HINTS["claude_code"],
        )
    return CheckResult("claude_code", "pass", "claude-agent-sdk available")


def _probe_openai(config: Any) -> CheckResult:
    if os.environ.get("OPENAI_API_KEY"):
        return CheckResult("openai", "pass", "key present")
    return CheckResult(
        "openai", "fail", "openai unreachable: no API key",
        fix_hint=_FIX_HINTS["openai"],
    )


def _probe_anthropic_api(config: Any) -> CheckResult:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult("anthropic_api", "pass", "key present")
    return CheckResult(
        "anthropic_api", "fail", "anthropic_api unreachable: no API key",
        fix_hint=_FIX_HINTS["anthropic_api"],
    )


def _probe_musicbrainz(config: Any) -> CheckResult:
    """Keyless reachability ping. Informational only: never `fail`, at worst `warn`."""
    try:
        resp = requests.get(
            "https://musicbrainz.org/ws/2/",
            headers={"User-Agent": _user_agent()},
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return CheckResult("musicbrainz", "warn", f"musicbrainz unreachable: {exc}")
    if resp.status_code < 500:
        return CheckResult("musicbrainz", "pass", f"reachable (HTTP {resp.status_code})")
    return CheckResult("musicbrainz", "warn", f"musicbrainz unreachable: HTTP {resp.status_code}")
