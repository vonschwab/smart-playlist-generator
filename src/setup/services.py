# src/setup/services.py
"""Uniform `test_connection()` probes for each optional external service.

One place to construct-and-probe each integration Playlist Generator can
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

Service clients are imported lazily, inside each probe, so a missing
optional dependency (e.g. no `claude_agent_sdk` installed) can never break
`import src.setup.services` itself. `LastFMClient` is additionally exposed
as a module-level name (defaulting to `None`, resolved on first real use)
so tests can `monkeypatch.setattr(services, "LastFMClient", FakeClient)`
without needing the real client/its dependency chain importable.
"""
from __future__ import annotations

import os
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
    return f"mixarc/{v} (+https://github.com/vonschwab/smart-playlist-generator)"


def test_connection(service: str, config: Any) -> CheckResult:
    """Probe one service by id. Dispatches by name; never raises."""
    probe = globals().get(f"_probe_{service}")
    if probe is None:
        return CheckResult(service, "fail", f"unknown service: {service}")
    try:
        return probe(config)
    except Exception as exc:
        if service == "musicbrainz":
            return CheckResult(service, "warn", f"musicbrainz unreachable: {exc}")
        return CheckResult(
            service, "fail", f"{service} unreachable: {exc}",
            fix_hint=_FIX_HINTS.get(service),
        )


def _probe_lastfm(config: Any) -> CheckResult:
    api_key = _cfg(config, "lastfm", "api_key")
    username = _cfg(config, "lastfm", "username")
    if not api_key or not username:
        return CheckResult("lastfm", "pass", "not configured — optional")

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
    token = _cfg(config, "discogs", "token")
    if not token:
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
    enabled = _cfg(config, "plex", "enabled", False)
    base_url = _cfg(config, "plex", "base_url")
    token = _cfg(config, "plex", "token")
    if not enabled or not base_url or not token:
        return CheckResult("plex", "pass", "not configured — optional")

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
