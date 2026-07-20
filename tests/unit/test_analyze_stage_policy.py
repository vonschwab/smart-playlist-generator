"""Not-configured stages skip loudly; configured-but-broken still raises.

Uniform policy across analyze_library.py's optional-service stages
(lastfm/discogs/popularity/energy): a missing key/token/env is a *skip*
(``{"skipped": True, "reason": "<service> not configured — <consequence>"}``),
never a RuntimeError — a fresh install without Last.fm/Discogs keys or a
WSL+Essentia energy setup must not abort analysis. A *configured* service that
is actually broken (bad key, unreachable network, failed WSL preflight) must
keep raising exactly as before — never silently swallowed into a skip.
"""
import sys
import types

import numpy as np
import pytest

import scripts.analyze_library as al


def _ctx(tmp_path, config_text):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config_text, encoding="utf-8")
    # Match the real ctx shape (see analyze_library.py run loop ~L2662-2673):
    # conn is always a real sqlite3.Connection in production, never None —
    # stage_discogs queries it before its own token check even runs.
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    return {"config_path": str(cfg), "db_path": str(tmp_path / "m.db"),
            "out_dir": str(tmp_path), "args": None, "conn": conn}


# ---------------------------------------------------------------------------
# lastfm
# ---------------------------------------------------------------------------

def test_lastfm_skips_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    from scripts.analyze_library import stage_lastfm
    res = stage_lastfm(_ctx(tmp_path, "library: {}\n"))
    assert res["skipped"] is True
    assert "not configured" in res["reason"]


def test_lastfm_configured_but_broken_still_raises(tmp_path, monkeypatch):
    """Key present + broken service must NOT silently skip.

    Avoids a real network call (and touching the real production
    data/ai_genre_enrichment.db, which SidecarStore(ENRICHMENT_DB_PATH) would
    otherwise open) by monkeypatching ENRICHMENT_DB_PATH to a throwaway path
    and SidecarStore's construction to raise — the point is the stage
    propagates a configured failure, not that the network fails.
    """
    monkeypatch.setenv("LASTFM_API_KEY", "not-a-real-key")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", tmp_path / "enrichment.db")

    class _BoomStore:
        def __init__(self, *a, **k):
            raise RuntimeError("lastfm service unreachable")

    monkeypatch.setattr(al, "SidecarStore", _BoomStore)
    with pytest.raises(Exception):
        al.stage_lastfm(_ctx(tmp_path, "lastfm:\n  api_key: not-a-real-key\n  username: nobody\n"))


# ---------------------------------------------------------------------------
# discogs
# ---------------------------------------------------------------------------

def test_discogs_skips_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCOGS_TOKEN", raising=False)
    from scripts.analyze_library import stage_discogs
    res = stage_discogs(_ctx(tmp_path, "library: {}\n"))
    assert res["skipped"] is True
    assert "not configured" in res["reason"]


def test_discogs_configured_but_broken_still_raises(tmp_path, monkeypatch):
    """Token present + client init failure must NOT silently skip."""
    monkeypatch.setenv("DISCOGS_TOKEN", "not-a-real-token")

    class _BoomClient:
        def __init__(self, *a, **k):
            raise RuntimeError("discogs client init failed")

    monkeypatch.setattr(al, "DiscogsClient", _BoomClient)
    with pytest.raises(Exception):
        al.stage_discogs(_ctx(tmp_path, "discogs:\n  token: not-a-real-token\n"))


# ---------------------------------------------------------------------------
# popularity (shares the lastfm key)
# ---------------------------------------------------------------------------

def test_popularity_skips_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    # stage_popularity's own "no_artifact" pre-check would mask the key check
    # if the artifact doesn't exist — create a stub so we reach the key check.
    np.savez(tmp_path / "data_matrices_step1.npz", track_ids=np.array(["a"], dtype=object))
    res = al.stage_popularity(_ctx(tmp_path, "library: {}\n"))
    assert res["skipped"] is True
    assert "not configured" in res["reason"]


# ---------------------------------------------------------------------------
# energy — non-regression critical: Dylan's canonical config.yaml has NO
# `analyze.energy` section at all (confirmed by reading it directly) and
# relies entirely on the EnergyConfig dataclass defaults, which describe his
# real WSL setup. So "section absent" must NOT skip — only an explicit
# `enabled: false`, or a non-Windows host (where WSL fundamentally can't
# exist — the actual fresh/non-Windows-user case), skips.
# ---------------------------------------------------------------------------

def _energy_ctx(tmp_path, config_text="library: {}\n", force=False):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config_text, encoding="utf-8")
    args = types.SimpleNamespace(force=force, energy_workers=None)
    np.savez(tmp_path / "data_matrices_step1.npz", track_ids=np.array(["a", "b"], dtype=object))
    return {"args": args, "config_path": str(cfg), "out_dir": str(tmp_path),
            "db_path": str(tmp_path / "m.db"), "cancellation_check": None}


def test_energy_skips_when_disabled_in_config(tmp_path, monkeypatch):
    called = {"preflight": False}
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: called.__setitem__("preflight", True))
    ctx = _energy_ctx(tmp_path, "analyze:\n  energy:\n    enabled: false\n")
    res = al.stage_energy(ctx)
    assert res["skipped"] is True
    assert "not configured" in res["reason"]
    assert called["preflight"] is False


def test_energy_skips_on_non_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    called = {"preflight": False}
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: called.__setitem__("preflight", True))
    ctx = _energy_ctx(tmp_path)  # no analyze.energy section — same as Dylan's config
    res = al.stage_energy(ctx)
    assert res["skipped"] is True
    assert "not configured" in res["reason"]
    assert called["preflight"] is False


def test_energy_absent_section_still_preflights_on_windows(tmp_path, monkeypatch):
    """Pins Dylan's non-regression case: no `analyze.energy` section, on
    Windows, must still preflight and raise on WSL failure exactly as today."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (2, 2))

    def boom(cfg):
        raise RuntimeError("WSL not available")

    monkeypatch.setattr(al, "_energy_preflight", boom)
    ctx = _energy_ctx(tmp_path)  # no analyze.energy section at all
    with pytest.raises(RuntimeError, match="WSL"):
        al.stage_energy(ctx)
