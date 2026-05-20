"""
PlaylistGenerator smoke golden tests — safety net for Tier-3.2 extraction.

Strategy: bypass __init__ via __new__, monkeypatch _maybe_generate_ds_playlist
to return synthetic tracks, and snapshot the result structure in JSON goldens
under tests/unit/goldens/playlist_generator/.

Any extraction in PR-1 through PR-6 that silently changes the shape or content
of the returned dict will be caught here.

To re-baseline after an INTENTIONAL behaviour change, delete the relevant
golden file and re-run (the test will write a new baseline and skip).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Fixture helpers live in the top-level tests/ package so they can be reused
# by future Tier-3.2 PRs without going through conftest.py injection.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.conftest_playlist_generator import SYNTHETIC_TRACKS, make_synthetic_generator

GOLDEN_DIR = Path(__file__).parent / "goldens" / "playlist_generator"

# ---------------------------------------------------------------------------
# Synthetic DS result — what _maybe_generate_ds_playlist returns
# ---------------------------------------------------------------------------

SYNTHETIC_DS_TRACKS: List[Dict[str, Any]] = list(SYNTHETIC_TRACKS[:8])


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def smoke_generator(monkeypatch):
    """
    Return a PlaylistGenerator with all I/O stubbed out.

    _maybe_generate_ds_playlist → SYNTHETIC_DS_TRACKS
    _post_order_validate_ds_output → no-op (we trust the golden to catch regressions)
    _compute_edge_scores_from_artifact → [] (no artifact on disk)
    _print_playlist_report → no-op (suppress console output)
    """
    gen = make_synthetic_generator()

    monkeypatch.setattr(
        gen,
        "_maybe_generate_ds_playlist",
        lambda *args, **kwargs: list(SYNTHETIC_DS_TRACKS),
        raising=False,
    )
    monkeypatch.setattr(
        gen,
        "_post_order_validate_ds_output",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        gen,
        "_compute_edge_scores_from_artifact",
        lambda *args, **kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(
        gen,
        "_print_playlist_report",
        lambda *args, **kwargs: None,
        raising=False,
    )

    return gen


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def _call_artist_basic(gen) -> Optional[Dict[str, Any]]:
    """artist_basic: 'Artist 0' has 3 tracks per artist × 4 artists = 12 tracks total;
    Artist 0 has t0..t2 (3 tracks).  With collaborations disabled and < 4 exact matches
    the method falls back to collaboration search — but Artist 0 has exactly 3 solo tracks,
    so it will trigger the collaboration-search path and, finding none, return None.

    To guarantee a non-None return we request include_collaborations=False and use
    artist_only=False (library-wide DS scope), ensuring the 12-track library has ≥ 4
    tracks for the artist check (library wide).  We override artist_name to one that
    has ≥ 4 tracks: Artist 0 only has 3.  Instead, pass the full SYNTHETIC_TRACKS
    (all 12 tracks are "artist 0..3") — we want any result, even None, as the golden.
    """
    return gen.create_playlist_for_artist(
        artist_name="Artist 0",
        track_count=8,
        include_collaborations=False,
    )


def _call_genre_basic(gen) -> Optional[Dict[str, Any]]:
    """genre_basic: library mock returns all 12 SYNTHETIC_TRACKS for genre 'ambient'.
    All 12 have duration=180s which passes is_valid_duration(47s..720s) check.
    """
    return gen.create_playlist_for_genre(
        genre_name="ambient",
        track_count=8,
    )


def _call_seed_list_basic(gen) -> Optional[Dict[str, Any]]:
    """seed_list_basic: provide explicit track IDs so the exact-match path is taken."""
    return gen.create_playlist_from_seed_tracks(
        seed_tracks=["Track 0 - Artist 0", "Track 3 - Artist 1"],
        seed_track_ids=["t0", "t3"],
        track_count=8,
    )


SMOKE_SCENARIOS: Dict[str, Any] = {
    "artist_basic": _call_artist_basic,
    "genre_basic": _call_genre_basic,
    "seed_list_basic": _call_seed_list_basic,
}


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


def _serialize_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce the orchestrator return value to a JSON-comparable snapshot.

    We capture: whether the result is None, the top-level keys present,
    the track_ids in order, and the track count.  We deliberately omit
    ds_report (internal metrics) and artists tuple (set-ordering is
    non-deterministic) to keep the golden stable.
    """
    if result is None:
        return {"result": None}

    tracks = result.get("tracks") or []
    track_ids = [
        str(t.get("rating_key") or t.get("track_id") or "")
        for t in tracks
    ]
    # Stable key listing (sorted) so golden doesn't depend on dict ordering
    top_keys = sorted(k for k in result if k != "ds_report")

    return {
        "result": "dict",
        "top_keys": top_keys,
        "track_ids": track_ids,
        "track_count": len(tracks),
    }


# ---------------------------------------------------------------------------
# The parametrized smoke golden test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", sorted(SMOKE_SCENARIOS.keys()))
def test_playlist_generator_smoke_golden(scenario_name, smoke_generator, request):
    call_fn = SMOKE_SCENARIOS[scenario_name]

    result = call_fn(smoke_generator)
    snapshot = _serialize_result(result)

    golden_path = GOLDEN_DIR / f"{scenario_name}.json"

    generate = request.config.getoption("--generate-golden", default=False)

    if generate or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if not generate:
            pytest.skip(
                f"Created golden baseline at {golden_path}; rerun to verify."
            )
        return

    expected = json.loads(golden_path.read_text(encoding="utf-8"))

    assert snapshot["result"] == expected["result"], (
        f"[{scenario_name}] result presence changed: "
        f"got {snapshot['result']!r}, expected {expected['result']!r}"
    )

    if expected["result"] is None:
        # Both None — done.
        return

    assert snapshot["top_keys"] == expected["top_keys"], (
        f"[{scenario_name}] returned dict keys changed.\n"
        f"  Got:      {snapshot['top_keys']}\n"
        f"  Expected: {expected['top_keys']}"
    )
    assert snapshot["track_ids"] == expected["track_ids"], (
        f"[{scenario_name}] track ordering or content changed.\n"
        f"  Got:      {snapshot['track_ids']}\n"
        f"  Expected: {expected['track_ids']}"
    )
    assert snapshot["track_count"] == expected["track_count"], (
        f"[{scenario_name}] track_count changed: "
        f"{snapshot['track_count']} vs {expected['track_count']}"
    )
