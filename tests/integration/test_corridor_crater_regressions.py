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


# ─────────────────────────────────────────────────────────────────────────
# Phase 2 Task 2 headline outcomes (RELATIVE repair triggers, commit fea3e8c
# and its parent 74d8add). These pin the *specific* measured improvements
# the relative-epsilon trigger (tail-DP + edge repair, `*_relative_epsilon`
# default 0.25) produced, not just "no crater" -- a much tighter bar than
# the transition_floor=0.2 the tests above check against, since Task 2's own
# job is to lift weak-but-not-broken edges, not merely keep them off the
# floor. Thresholds (0.55 / 0.55 / 0.339) are the corpus's own measured
# values minus generous slack -- but PC and Swirlies reproduce comfortably
# above their thresholds on THIS file's hand-picked seed topology (0.6851
# and 0.8127 respectively, both measured 2026-07-18), while SADE's proxy
# lands much closer to its floor (0.5531 vs 0.55) than the corpus's own
# 0.6676 would suggest -- see that test's docstring for why (same "proxy
# topology != corpus topology" caveat test_sade_home_does_not_crater already
# documents above). Measured numbers are recorded in each docstring so a
# future re-measurement can tell drift from noise.
# ─────────────────────────────────────────────────────────────────────────

# 6 real Parquet Courts tracks -- the exact pier set from the Phase 2 Task 1
# probe (docs/corridor_baseline/phase2_mechanism_probes.md, "Probe 1") and
# its Task 2 deep-dive reproduction (.superpowers/sdd/p2-task-2-report.md),
# recovered from logs/playlists/2026-07-18_174422_Parquet_Courts_b06d13.log
# and re-confirmed identical in the reproduction log
# logs/playlists/2026-07-18_203637_Parquet_Courts_000001.log. This cell is
# NOT one of the 6 artists in the 12-cell docs/corridor_baseline/
# phase2_task2_corpus.json corpus file -- it is the Task 1/2 "deep-dived"
# regression case reported separately in p2-task-2-report.md, run at
# production defaults (cohesion_mode=dynamic, pace_mode=dynamic; no mode
# flags passed in the original `main_app.py --artist "Parquet Courts"
# --anchor-seed-ids ...` reproduction), hence no HOME_UI_KWARGS below.
PARQUET_COURTS_SEEDS = [
    "13e7fd470c7ee9d802e2e05d1205b04f",  # Bodies made of
    "44c065f0f201fb843a8b1a82bfbb9f61",  # Freebird II
    "da6dc06b3bdb2c303eb5fc969b43adac",  # Mardi Gras Beads
    "d83034823de937da82d855995501f60a",  # Borrowed Time
    "7a6d3bae11fa410feb97f5d26cbe02fc",  # Human Performance
    "f86a9b71d2e35855ffc91f0488297915",  # Into the garden
]


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_parquet_courts_task2_min_transition():
    """PC segment 4's Human Performance -> ... -> Into the garden landing
    edge was the playlist's global weakest edge pre-Task-2 (T=0.394, Probe 1
    of docs/corridor_baseline/phase2_mechanism_probes.md -- corridor
    admission was refuted as the cause; the actual mechanism was the
    tail-DP/edge-repair floor gates under-triggering on a "clears 0.3, not
    close to achievable" edge). With the relative trigger active
    (tail_dp_relative_epsilon default 0.25), the Task 2 reproduction
    (.superpowers/sdd/p2-task-2-report.md, 2026-07-18) measured the landing
    edge at T=0.805 and the playlist's own min_transition at 0.6851,
    below_floor=0. Threshold 0.55 sits well below either number -- generous
    slack under the measured 0.805/0.6851, tight enough to catch a real
    reversion of the relative-trigger fix."""
    ui = gui_ui_state()
    res = generate_like_gui(seeds=PARQUET_COURTS_SEEDS, ui=ui, length=30, random_seed=0)
    playlist_stats = res.playlist_stats.get("playlist", {})
    min_t = playlist_stats.get("min_transition")

    assert min_t is not None, "Parquet Courts: min_transition missing from playlist_stats"
    assert min_t >= 0.55, (
        f"Parquet Courts: min_T={min_t:.4f} fell below the Task 2 regression floor 0.55 "
        f"(measured 0.805 landing edge / 0.6851 playlist min_T on 2026-07-18; "
        f"track_ids={res.track_ids})"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_sade_home_task2_min_transition():
    """SADE+home's segment-0 first edge (Sade "Siempre Hay Esperanza" ->
    Isaac Hayes "Let's Stay Together") was the run's weakest at T=0.454
    pre-Task-2 (Probe 2 of phase2_mechanism_probes.md -- a materially better,
    fully-admitted connector (Plunky & Oneness of Juju, min_sim=0.697) sat in
    the same pool unused; tail-DP structurally cannot reach a segment's
    *first* edge, only edge repair can). With the relative trigger active,
    edge repair fired (pos=1 swap worst-T 0.452 -> 0.730) and the corpus's
    OWN clustered-pier SADE/home cell (docs/corridor_baseline/
    phase2_task2_corpus.json, 2026-07-18) measured min_transition=0.6676
    (up from 0.454), below_floor=0.

    This test's hand-picked SADE_SEEDS proxy (same topology as
    test_sade_home_does_not_crater above, per that test's own docstring
    caveat that this proxy's numbers do NOT match the corpus's clustered-pier
    cell) measures noticeably lower on this harness: min_transition=0.5531
    as of 2026-07-18, only ~0.003 above the 0.55 threshold below -- NOT the
    generous margin the corpus number would suggest. 0.55 is used anyway
    (matches the corpus-derived floor the fix is meant to guard, and this
    proxy did clear it on measurement), but a future re-run landing between
    0.50 and 0.55 is much more likely to be normal topology-proxy noise than
    an actual relative-trigger regression -- cross-check against
    test_sade_home_does_not_crater (which never craters, i.e. never drops
    below transition_floor=0.2) before concluding the fix broke."""
    ui = gui_ui_state(**HOME_UI_KWARGS)
    res = generate_like_gui(seeds=SADE_SEEDS, ui=ui, length=20, random_seed=0)
    playlist_stats = res.playlist_stats.get("playlist", {})
    min_t = playlist_stats.get("min_transition")

    assert min_t is not None, "SADE+home: min_transition missing from playlist_stats"
    assert min_t >= 0.55, (
        f"SADE+home: min_T={min_t:.4f} fell below the Task 2 regression floor 0.55 "
        f"(this proxy measured 0.5531 on 2026-07-18; corpus clustered-pier cell "
        f"measured 0.6676, phase2_task2_corpus.json; track_ids={res.track_ids})"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_swirlies_home_task2_regression():
    """Swirlies+home (the M1 pool-starvation sentinel, see
    test_swirlies_home_does_not_crater above) also improved under the Task 2
    relative trigger: tail-DP fired on segments 2 and 3 (window min
    0.353->0.801 and 0.351->0.872). The corpus's OWN clustered-pier
    Swirlies/home cell (docs/corridor_baseline/phase2_task2_corpus.json,
    2026-07-18) measured min_transition=0.5955 (up from 0.359 pre-Task-2),
    below_floor=0 held.

    This test's hand-picked SWIRLIES_SEEDS proxy (same topology as
    test_swirlies_home_does_not_crater above) measures higher on this
    harness: min_transition=0.8127 as of 2026-07-18, below_floor=0.
    Threshold 0.339 is "within -0.02 of the pre-Task-2 0.359" per the Task 2
    report's own acceptance bar -- comfortable slack under either the corpus's
    0.5955 or this proxy's own 0.8127, while still catching a reversion back
    toward the pre-Task-2 number."""
    ui = gui_ui_state(**HOME_UI_KWARGS)
    res = generate_like_gui(seeds=SWIRLIES_SEEDS, ui=ui, length=20, random_seed=0)
    playlist_stats = res.playlist_stats.get("playlist", {})
    min_t = playlist_stats.get("min_transition")
    below_floor = playlist_stats.get("below_floor_count")

    assert min_t is not None, "Swirlies+home: min_transition missing from playlist_stats"
    assert below_floor is not None, "Swirlies+home: below_floor_count missing from playlist_stats"
    assert below_floor == 0, (
        f"Swirlies+home: {below_floor} edge(s) below the transition floor "
        f"(min_T={min_t:.4f}; track_ids={res.track_ids})"
    )
    assert min_t >= 0.339, (
        f"Swirlies+home: min_T={min_t:.4f} fell below the Task 2 regression floor 0.339 "
        f"(this proxy measured 0.8127 on 2026-07-18; corpus clustered-pier cell measured "
        f"0.5955, phase2_task2_corpus.json; track_ids={res.track_ids})"
    )
