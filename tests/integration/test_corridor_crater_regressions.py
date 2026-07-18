"""Phase 1 Task 9 (spec section 6): permanent crater regressions.

The corridor-pooling design spec's own acceptance section names two cells --
SADE+home and Swirlies+home -- as the "crater" cells the whole corridor body
of work exists to fix (docs/superpowers/specs/2026-07-12-corridor-first-
pooling-design.md section 1: "SADE+home and Swirlies+home craters traced
end-to-end" at T=0.028/T=0.018 under the pre-corridor legacy pool). This test
pins the floor permanently, independent of golden-file churn (spec section 6:
"permanent crater regressions are golden-independent" -- goldens get
re-baselined every phase; this test's bar does not move with them).

Multi-pier seeds via ``generate_like_gui`` (CLAUDE.md session discipline:
"Multi-pier seeds through the gui_fidelity harness -- never hand-built
overrides, never single-seed topology"), never a fabricated single-seed
fixture. The seed track_ids below are 5 real SADE / Swirlies tracks spanning
every studio album in the library (hand-picked from ``tracks`` via
Artist/Title, not derived from the artist-clustering medoid selection the
production Artist-mode corpus captures use) -- a faithful multi-pier proxy
for the corpus's "artist name -> clustered piers" cells, not a byte-identical
replay of them. Consequently this test's own min_T numbers will NOT match the
corpus's SADE/home (currently plateaued at 0.454, see Concerns in
.superpowers/sdd/p1-permode-width-report.md and the known-issues section of
docs/corridor_baseline/phase1_contract_report.md) or Swirlies/home values --
this test only pins the FLOOR (min_T >= transition_floor, below_floor == 0)
for its own hand-picked multi-pier topology, not the corpus's specific
plateau number.

"home" detent axes (sonic_mode=strict, genre_mode=strict, cohesion_mode=
dynamic, pace_mode=dynamic) match scripts/corridor_baseline/runner.py's
``DETENTS["home"]`` exactly -- the same slider combination the corpus/probe
harness uses for every "home" cell.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.gui_fidelity import gui_ui_state, generate_like_gui, resolved_artifact_path

ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

HOME_UI_KWARGS = dict(
    sonic_mode="strict", genre_mode="strict", cohesion_mode="dynamic", pace_mode="dynamic",
)

# 5 real SADE tracks, one per studio album (Diamond Life -> Love Deluxe ->
# Lovers Rock -> Promise -> Stronger Than Pride), spanning the full sonic/era
# range of the artist's catalog -- a faithful multi-pier stand-in for what
# artist-mode medoid clustering would select across clusters.
SADE_SEEDS = [
    "85e1a8a201f3e207a3183add4ef883bc",  # Smooth Operator (Diamond Life)
    "df832f354e8709c1d50db50af3f09598",  # No Ordinary Love (Love Deluxe)
    "045d42de29c67c608e1a5c4d614d31f5",  # King Of Sorrow (Lovers Rock)
    "625542465149f8b807073ef7c30870c2",  # The Sweetest Taboo (Promise)
    "76ba6c587d5863a299d22995dfd9d5ca",  # Love is Stronger Than Pride (Stronger Than Pride)
]

# 5 real Swirlies tracks spanning all 3 studio albums (Blonder Tongue Audio
# Baton x2, They Spent Their Wild Youthful Days... x2, What to Do About Them
# x1) -- the sentinel cell the pool-starvation research (M1-M4,
# docs/POOL_STARVATION_RESEARCH_2026-07-12.md) traced end-to-end.
SWIRLIES_SEEDS = [
    "6ed03ec90deef8e359c87af5f23f2143",  # Bell (Blonder Tongue Audio Baton)
    "61dd49c4f4674caa7bdbcc84be487d42",  # Park the Car by the Side of the Road (Blonder Tongue Audio Baton)
    "229154e1a1c83142e74b11a09ac49ba1",  # San Cristobal de Las Casas (They Spent...)
    "8901900b93b70017d98704054dd08209",  # Sunn (They Spent...)
    "e16a08426b5fa32959160d2ae20f43a4",  # Cousteau (What to Do About Them)
]


def _assert_no_crater(seeds: list[str], label: str) -> None:
    ui = gui_ui_state(**HOME_UI_KWARGS)
    res = generate_like_gui(seeds=seeds, ui=ui, length=20, random_seed=0)
    playlist_stats = res.playlist_stats.get("playlist", {})

    min_t = playlist_stats.get("min_transition")
    below_floor = playlist_stats.get("below_floor_count")
    transition_floor = playlist_stats.get("transition_floor")

    assert min_t is not None, f"{label}: min_transition missing from playlist_stats"
    assert transition_floor is not None, f"{label}: transition_floor missing from playlist_stats"
    assert below_floor is not None, f"{label}: below_floor_count missing from playlist_stats"

    assert min_t >= transition_floor, (
        f"{label}: CRATER — min_T={min_t:.4f} below transition_floor={transition_floor:.4f} "
        f"(track_ids={res.track_ids})"
    )
    assert below_floor == 0, (
        f"{label}: {below_floor} edge(s) below the transition floor "
        f"(min_T={min_t:.4f}, floor={transition_floor:.4f})"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_sade_home_does_not_crater():
    """SADE+home: pre-corridor legacy pooling produced T=0.028 (spec section 1).
    This hand-picked multi-pier proxy currently holds well clear of the floor
    (min_T ~0.40 as of this task's writing) -- the corpus's own clustered-pier
    SADE/home cell plateaus at 0.454 under every tested strict width (KNOWN
    ISSUE, not a regression this test is meant to catch; see
    docs/corridor_baseline/phase1_contract_report.md's known-issues section).
    This test's job is narrower and permanent: never again let SADE/home fall
    BELOW the transition floor, regardless of which phase's width/golden
    churn is in flight."""
    _assert_no_crater(SADE_SEEDS, "SADE+home")


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_swirlies_home_does_not_crater():
    """Swirlies+home: pre-corridor legacy pooling produced T=0.018, below_floor=4
    (spec section 1; the M1 "amputated manifold" pool-starvation sentinel).
    Corridor pooling fixed this cell to below_floor=0 by Task 6 and it has
    held at every subsequent gate (Task 8 flip, width re-pin, per-mode
    widths). This test pins that recovery permanently, golden-file-
    independent, per spec section 6."""
    _assert_no_crater(SWIRLIES_SEEDS, "Swirlies+home")
