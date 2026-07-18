"""End-to-end regression catalog, driven through the GUI-fidelity harness.

Each test here reproduces a real bug fixed during the 2026-06 genre/diversity work
by generating EXACTLY as the GUI would (production config resolved from config.yaml).
Add a case here whenever a generation bug is fixed — that is the "update regularly"
half of the playlist-testing skill.

Marked @integration @slow: they need the live artifact bundle + sidecar.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import sqlite3
from pathlib import Path

import pytest

import src.playlist.pipeline.core as _pipeline_core
from tests.support.gui_fidelity import (
    generate_like_gui,
    gui_ui_state,
    artist_at_positions,
    find_min_gap_violations,
    resolved_artifact_path,
)
from src.features.artifacts import load_artifact_bundle
from src.config_loader import Config, resolve_database_path
from src.genre.authority import on_tag_track_ids_for_artist
from src.local_library_client import LocalLibraryClient
from src.playlist.tag_steering import _canonical_genre_ids_for_tags
from src.playlist_generator import PlaylistGenerator
from src.playlist_gui.policy import derive_runtime_config, merge_overrides as _merge_overrides
from src.playlist_gui.ui_state import UIStateModel

# Resolved via config.yaml (not a hardcoded repo-relative path): in a satellite
# workspace (docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md)
# data/ holds only a 0-byte metadata.db stub and no artifacts/ dir at all --
# config.yaml's playlists.ds_pipeline.artifact_path is the absolute canonical
# path. A hardcoded "data/artifacts/..." literal here silently skipped every
# test in this file (and generate_like_gui's own artifact_path fallback uses
# this exact same resolver, so this also guarantees load_artifact_bundle(ART)
# and generate_like_gui() share one artifact instance).
ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

# Five seeds from the seeds-mode run where Smog clustered at 14/17 under min_gap=9.
SEEDS = [
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler - Howling at the Second Moon
    "b8f8aa0e86f977f9fcb26f615e130ac9",  # Hayden Pedigo - Nearer, Nearer
    "42473b911cef5674e56b8e2ce87df7cb",  # Steve Hiett - Are These My Memories?
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia - Hold On Magnolia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan - Let's Move to the Country
]


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_strong_artist_gap_enforced_across_segments():
    """Regression for commit 8101ac1: cross-segment min_gap was hardcoded to 1, so
    'Artist Gap: strong' (min_gap=9) still produced same-artist pairs 3-5 apart that
    straddled segment boundaries (Smog 14/17, Hayden Pedigo 6/11, William Ackerman 21/24).

    Regression for the Phase 1 min_gap fix (2026-07-18, root-caused via this same
    fixture): CONFIRMED, deterministic violations under artist_spacing='strong'
    (min_gap=9) -- [(7,13,'Golden Brown',6), (11,17,'Hayden Pedigo',6),
    (16,21,'Dylan Golden Aycock',5)] (1-indexed positions). Traced to a
    pre-existing hole in cross-segment min_gap enforcement, exposed (not caused)
    by mini-pier waypoint subdivision: ``recent_boundary_artists``
    (pier_bridge_builder.py, fixed 2026-06-01) only ever blocked artists BACKWARD
    (already-placed tracks). ``ordered_seeds`` -- the full pier sequence including
    mini-pier waypoints (SP3, 2026-06-30) -- is fully fixed before the segment
    loop starts, so an early segment could freely place an interior track by the
    same artist as a not-yet-built pier/waypoint a few positions later; nothing
    re-validated that placement once the later pier committed, since piers are
    placed unconditionally (never gated on artist novelty). Fixed by adding a
    FORWARD half to the same mechanism: ``_forward_pier_gap_block_indices`` +
    ``_pier_nominal_positions`` (pier_bridge_builder.py) compute, once per
    generation, which upcoming piers land within ``min_gap`` of each segment and
    fold their artist keys into ``_recent_artists_for_segment`` -- the single
    choke point every placement mechanism (corridor pool artist-gate, beam,
    micro-pier, terminal-greedy fallback) already reads. See
    .superpowers/sdd/p1-mingap-fix-report.md.
    """
    load_artifact_bundle.cache_clear()
    # sonic_variant_override passed explicitly (not relied on as a generate_like_gui
    # side effect, see test_dense_genre_integration.py's live_bundle fixture for the
    # established precedent): tests/conftest.py's autouse _reset_sonic_variant_override
    # resets the process-wide override to None before every test, so a direct load
    # here (before any generate_like_gui call in THIS test) would otherwise fail with
    # "Artifact missing required keys: ['X_sonic']" -- pre-existing bug found running
    # -m slow in isolation for the first time (Phase 1 Task 9, 2026-07-18).
    bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow",
        pace_mode="narrow", artist_spacing="strong",
    )
    res = generate_like_gui(seeds=SEEDS, ui=ui, length=30, random_seed=0)

    assert len(res.track_ids) == 30, f"expected 30 tracks, got {len(res.track_ids)}"
    artists = artist_at_positions(bundle, res.track_ids)
    violations = find_min_gap_violations(artists, min_gap=9)
    assert not violations, f"cross-segment min_gap=9 violated: {violations}"


# Green-House track IDs from the failing run (beatless/ambient artist).
GH_SEEDS = [
    "dc7a45bf0c0dbf6ebd574343df4e0159",  # Produce Aisle
    "1d73b404fc6e0de8e4628e64ae9dc982",  # Dragline Silk
    "1e86f3e9cae613f43ece846b71c9f7d5",  # Sanibel
    "981e59a511d15e23109f5a3bcf8f4f8c",  # Hinterland I
    "37dc61f3c8f6ba0c3742979deef6af96",  # Farewell, Little Island
]


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_pace_narrow_feasible_for_ambient_piers():
    """Regression: pace_mode=narrow threw 'Segment infeasible under bridge_floor
    backoff' for Green-House (beatless) because the rhythm-cosine hard gate was
    unsatisfiable (random-pair rhythm cosine p50 ≈ -0.01; narrow floor was 0.45).
    After the BPM+onset band retune, the hard rhythm-cosine gate is removed and
    replaced by BPM/onset bands + a soft penalty that gracefully handles beatless
    artists. Fixing commit: pace-gate-retune (2026-06-12).
    """
    load_artifact_bundle.cache_clear()
    # sonic_variant_override passed explicitly -- see the identical comment on
    # test_strong_artist_gap_enforced_across_segments above (same pre-existing bug,
    # same fix).
    bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")
    ti = bundle.track_id_to_index
    seeds = [t for t in GH_SEEDS if t in ti]
    if len(seeds) < 4:
        pytest.skip("Green-House piers not in this artifact build")

    res = generate_like_gui(
        seeds=seeds,
        cohesion_mode="dynamic", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow",
        length=30, random_seed=0,
    )
    assert res is not None
    assert len(res.track_ids) == 30, f"expected 30 tracks, got {len(res.track_ids)}"


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_hard_seed_pair_never_fails():
    """Regression for greedy terminal guarantee: a seed combination that exhausts
    every relaxation tier (bridge_floor backoff, micro-pier) must still produce a
    full 30-track playlist instead of raising ValueError('Segment N infeasible…').
    The terminal greedy path kicks in when guarantee_feasible=True (set by default
    via the infeasible_handling config block).
    """
    res = generate_like_gui(
        seeds=[
            "29a6637c9ba785f6270b114b37e59594",
            "afd9ee94229bde6f31c853bfbe754730",
            "a7cf50c432f58d0df81fcbb22c4bd674",
            "8539afd5d87ff30c3180863dced469c8",
            "631f693758a8c5de622d750a08cbf6ee",
        ],
        cohesion_mode="dynamic", genre_mode="narrow", sonic_mode="narrow",
        pace_mode="dynamic", artist_spacing="strong", length=30, random_seed=0,
    )
    tids = getattr(res, "track_ids", res)
    # >= 30 (not == 30): the Phase 1 min_gap fix (2026-07-18, forward pier-gap
    # blocking -- see test_strong_artist_gap_enforced_across_segments) removes
    # one more candidate from segment 5's pool here, so variable_bridge_length's
    # add-only bottleneck-maximizing choice now legitimately flexes that segment
    # nominal=2 -> chosen=3 (31 tracks total, confirmed zero min_gap violations).
    # Same relaxation precedent as test_corridor_pooling.py's `in (20, 21)`.
    assert len(tids) >= 30, f"expected >= 30 tracks, got {len(tids)} — greedy terminal did not fire"


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: tag-first pier selection
# (docs/superpowers/plans/2026-07-08-tag-first-pier-selection.md)
#
# Pier selection is DB-clustering / artist-mode (PlaylistGenerator.create_playlist_
# for_artist), which is explicitly out of generate_like_gui's scope (seeds-mode only
# — see the playlist-testing skill's "What this harness does NOT cover"). These
# cases instead build a real PlaylistGenerator through the SAME production config
# chain generate_like_gui uses for seeds mode (derive_runtime_config ->
# merge_overrides -> load_config_with_overrides), so tag_steering_tags is set
# exactly as the GUI would set it — never a hand-built override dict — then call
# the real create_playlist_for_artist against the live DB + artifact.
# ─────────────────────────────────────────────────────────────────────────────

BOC = "Boards of Canada"
REAL_ESTATE = "Real Estate"


class _PiersCaptured(Exception):
    """Raised by the ``_maybe_generate_ds_playlist`` stub the instant the real
    ``create_playlist_for_artist`` hands off the piers it selected. Lets
    pier-SELECTION tests skip the (expensive) beam search while still running
    every line of the real production pier-selection code — clustering, the
    bridgeability veto, and the tag-first dispatch — unmodified.
    """

    def __init__(self, seed_track_id, anchor_seed_ids):
        self.pier_ids = [str(seed_track_id)] + [str(a) for a in (anchor_seed_ids or [])]
        super().__init__(f"piers captured: {self.pier_ids}")


def _artist_mode_config(steering_tags, *, tag_first_pier_selection=None, anchor_max=None,
                        config_path="config.yaml"):
    """A real Config() (full config.yaml fidelity, presets baked as normal) with
    playlists.ds_pipeline.pier_bridge.tag_steering_tags set via the production
    derive_runtime_config -> merge_overrides -> load_config_with_overrides chain —
    the same one resolve_gui_overrides() uses. tag_first_pier_selection and
    tag_steering_anchor_max have no GUI slider (both are engineering rollback knobs,
    not user controls), so they are set directly on the loaded pier_bridge dict,
    mirroring how the plan describes them.
    """
    from src.playlist_gui.worker import load_config_with_overrides

    ui = UIStateModel(mode="artist", steering_tags=list(steering_tags))
    decisions = derive_runtime_config(ui)
    overrides = _merge_overrides({}, decisions.overrides)
    merged = load_config_with_overrides(config_path, overrides)
    pb_merged = ((merged.get("playlists") or {}).get("ds_pipeline") or {}).get("pier_bridge") or {}

    cfg = Config(config_path)
    pb = cfg.config.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    pb.update(pb_merged)
    if tag_first_pier_selection is not None:
        pb["tag_first_pier_selection"] = bool(tag_first_pier_selection)
    if anchor_max is not None:
        pb["tag_steering_anchor_max"] = int(anchor_max)
    return cfg


def _artist_generator(steering_tags, *, tag_first_pier_selection=None, anchor_max=None,
                      config_path="config.yaml"):
    cfg = _artist_mode_config(
        steering_tags, tag_first_pier_selection=tag_first_pier_selection,
        anchor_max=anchor_max, config_path=config_path,
    )
    library = LocalLibraryClient(db_path=resolve_database_path(cfg))
    return PlaylistGenerator(library_client=library, config=cfg)


def _select_piers(artist_name, steering_tags, *, popular_seeds_mode="off",
                   tag_first_pier_selection=None, anchor_max=None, track_count=30):
    """The exact piers create_playlist_for_artist hands to the DS pipeline, without
    paying for the beam search. Includes any Phase-B on-tag anchors, since those are
    injected into ordered_medoids -> pier_ids BEFORE the DS handoff."""
    generator = _artist_generator(
        steering_tags, tag_first_pier_selection=tag_first_pier_selection, anchor_max=anchor_max,
    )
    generator._maybe_generate_ds_playlist = (
        lambda **kwargs: (_ for _ in ()).throw(
            _PiersCaptured(kwargs.get("seed_track_id"), kwargs.get("anchor_seed_ids"))
        )
    )
    with pytest.raises(_PiersCaptured) as excinfo:
        generator.create_playlist_for_artist(
            artist_name=artist_name, track_count=track_count,
            popular_seeds_mode=popular_seeds_mode, random_seed=0,
        )
    return excinfo.value.pier_ids


def _authority_on_tag_ids(artist_name: str, tags, db_path: str) -> dict:
    """Mirrors the manual pier_check.py logic: re-reads the published genre
    authority (release_effective_genres via tracks.album_id, observed_leaf/legacy
    only) for ``artist_name``, using the SAME authority helpers the production
    tag-first dispatch itself reads — not a reimplementation of the SQL.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        gids = _canonical_genre_ids_for_tags(con, tags)
        if not gids:
            return {}
        return on_tag_track_ids_for_artist(con, artist_name, gids)
    finally:
        con.close()


def _nonseed_authority_on_tag_ids(tags, exclude_artist: str, db_path: str) -> set:
    """Library-wide published-authority (release_effective_genres, non-inferred layers)
    track_ids carrying ANY of ``tags``, with ``exclude_artist`` removed — i.e. exactly
    the universe Phase-B draws on-tag ANCHORS from (``_on_tag_track_ids`` in
    playlist_generator, built via resolve_tag_sonic_prototype_rows). Mirrors that
    function's SQL so this is an INDEPENDENT authority re-read of the realized piers,
    not a call back into the production selection path. {} if no tag maps.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        gids = _canonical_genre_ids_for_tags(con, tags)
        if not gids:
            return set()
        gph = ",".join("?" for _ in gids)
        rows = con.execute(
            f"SELECT DISTINCT t.track_id FROM tracks t "
            f"JOIN release_effective_genres reg ON reg.album_id = t.album_id "
            f"WHERE reg.genre_id IN ({gph}) "
            f"AND reg.assignment_layer NOT LIKE 'inferred%' "
            f"AND LOWER(TRIM(t.artist)) != LOWER(TRIM(?))",
            list(gids) + [str(exclude_artist)],
        ).fetchall()
        return {str(r[0]) for r in rows}
    finally:
        con.close()


_DB_PATH = None


def _db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = resolve_database_path(Config("config.yaml"))
    return _DB_PATH


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_tag_first_piers_boc_hauntology_on_tag():
    """Regression for the tag-first pier selection fix (2026-07-08 plan): piers used
    to be sonic-cluster-first (measured: BoC+hauntology 1/4 on-tag at default
    knobs). Tag-first draws piers from the artist's authority on-tag member set
    first, so >=3/4 realized SEED-ARTIST piers should now be authority-hauntology.

    Phase B (2026-07-09) additionally injects non-BoC on-tag ANCHORS into the pier
    set; the tag-FIRST guarantee is about the seed artist's own piers, so the anchor
    piers are excluded here (they get their own coverage below).
    """
    piers = _select_piers(BOC, ["hauntology"], popular_seeds_mode="off")
    on_tag = _authority_on_tag_ids(BOC, ["hauntology"], _db_path())
    assert len(piers) >= 3, f"too few piers realized: {piers}"
    anchors = _nonseed_authority_on_tag_ids(["hauntology"], BOC, _db_path())
    seed_piers = [p for p in piers if p not in anchors]
    assert seed_piers, f"no seed-artist piers realized (all anchors?): {piers}"
    hits = sum(1 for p in seed_piers if p in on_tag)
    assert hits / len(seed_piers) >= 0.75, (
        f"only {hits}/{len(seed_piers)} BoC (seed-artist) piers are authority-hauntology "
        f"(piers={piers})"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_tag_first_piers_boc_hauntology_rollback_reproduces_legacy():
    """tag_first_pier_selection: false must reproduce the pre-fix behavior (<=2/4
    on-tag, per the plan's measured baseline) — guards the rollback knob AND that
    the fix (not e.g. an artifact rebuild) is what moved the on-tag fraction.
    """
    piers = _select_piers(
        BOC, ["hauntology"], popular_seeds_mode="off", tag_first_pier_selection=False,
    )
    on_tag = _authority_on_tag_ids(BOC, ["hauntology"], _db_path())
    assert len(piers) >= 3, f"too few piers realized: {piers}"
    hits = sum(1 for p in piers if p in on_tag)
    assert hits / len(piers) <= 0.5, (
        f"legacy (tag_first_pier_selection=false) path unexpectedly on-tag-heavy: "
        f"{hits}/{len(piers)} (piers={piers}) — did the fix silently become the "
        f"only reachable code path?"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_tag_first_piers_boc_multi_tag_union():
    """Multi-tag union (hauntology + kosmische — the latter contributing zero BoC
    tracks in the live DB, measured 2026-07-08) must not break tag-first pier
    selection: members = union over both tags, degrading gracefully to the
    single-tag result rather than crashing or reverting to legacy.

    As in the single-tag case, Phase-B non-BoC anchor piers are excluded so the
    fraction measures the seed-artist tag-first guarantee.
    """
    piers = _select_piers(BOC, ["hauntology", "kosmische"], popular_seeds_mode="off")
    on_tag = _authority_on_tag_ids(BOC, ["hauntology", "kosmische"], _db_path())
    assert len(piers) >= 3, f"too few piers realized: {piers}"
    anchors = _nonseed_authority_on_tag_ids(["hauntology", "kosmische"], BOC, _db_path())
    seed_piers = [p for p in piers if p not in anchors]
    assert seed_piers, f"no seed-artist piers realized (all anchors?): {piers}"
    hits = sum(1 for p in seed_piers if p in on_tag)
    assert hits / len(seed_piers) >= 0.75, (
        f"multi-tag union: only {hits}/{len(seed_piers)} BoC piers on-tag (piers={piers})"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_tag_first_piers_boc_zero_on_tag_falls_back_to_legacy(caplog):
    """An artist with zero authority on-tag tracks for the requested tag must take
    the legacy fallback path (never crash, never fabricate membership). BoC has
    zero jangle-pop authority tracks in the live DB (measured 2026-07-08).
    """
    on_tag = _authority_on_tag_ids(BOC, ["jangle pop"], _db_path())
    assert not on_tag, "test fixture assumption broken: BoC now has jangle-pop tracks"

    with caplog.at_level(logging.INFO):
        _select_piers(BOC, ["jangle pop"], popular_seeds_mode="off")

    assert any("no authority on-tag tracks" in r.message for r in caplog.records), (
        "expected the tag-first legacy-fallback INFO log line to fire"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_tag_first_piers_real_estate_jangle_pop_no_regression():
    """Real Estate + jangle pop restricts piers to the authority on-tag subset
    (21/55 tracks in the live DB, measured 2026-07-08 — NOT '~all tracks' as the
    design doc estimated). Guard against a regression in distinct-artist count /
    worst-edge quality vs. the untagged baseline (plan acceptance: 'within a
    notch, no regression'). Runs the REAL DS pipeline (not pier-capture) because
    both metrics come from the recomputed final-playlist edge scores.
    """
    with_tag = _artist_generator(["jangle pop"]).create_playlist_for_artist(
        artist_name=REAL_ESTATE, track_count=30, popular_seeds_mode="off",
    )
    without_tag = _artist_generator([]).create_playlist_for_artist(
        artist_name=REAL_ESTATE, track_count=30, popular_seeds_mode="off",
    )
    assert with_tag is not None and without_tag is not None

    m_with = (with_tag.get("ds_report") or {}).get("metrics") or {}
    m_without = (without_tag.get("ds_report") or {}).get("metrics") or {}

    da_with = m_with.get("distinct_artists")
    da_without = m_without.get("distinct_artists")
    t_with = m_with.get("min_transition")
    t_without = m_without.get("min_transition")
    assert da_with is not None and da_without is not None, (
        f"missing distinct_artists metric: with={m_with} without={m_without}"
    )
    assert t_with is not None and t_without is not None, (
        f"missing min_transition metric: with={m_with} without={m_without}"
    )

    assert da_with >= da_without - 1, (
        f"distinct-artist regression: with-tag={da_with} without-tag={da_without}"
    )
    assert t_with >= t_without - 0.05, (
        f"worst-edge regression: with-tag min_transition={t_with:.3f} "
        f"without-tag={t_without:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase B: on-tag ANCHOR injection
# (docs/superpowers/plans/2026-07-09-bridge-side-phase-b.md, Task 3)
#
# Phase A (surface on-tag tracks as bridges) could not GUARANTEE a sonically-
# peripheral on-tag clique appears: a Ghost Box record is authority-hauntology but
# sonically ~0.1 genre-sim to Boards of Canada, so it fell below the seed-artist
# genre gate and the bridges never placed it. Phase B injects up to K representative
# on-tag tracks as PIERS (ungated, un-droppable anchors), selected to be bridgeable
# to a seed pier + tag-central + cross-artist-diverse. These cases drive the real
# create_playlist_for_artist and re-read the published authority
# (release_effective_genres) for the REALIZED piers — the same pier-fix precedent
# the Task-6 cases above use. Measured baselines (live DB + artifact, random_seed=0,
# 2026-07-09): BoC+hauntology injects 3 anchors (The Focus Group / Plone / Belbury
# Poly), worst-edge 0.537; anchor_max=0 -> 4 BoC-only piers, worst-edge 0.327.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_phase_b_anchors_boc_hauntology_injects_ontag_piers(caplog):
    """The Phase-B payoff: BoC + hauntology now GUARANTEES non-BoC authority-hauntology
    tracks appear AS PIERS (the long-standing Ghost Box goal Phase A could not hit).

    Two assertions off the realized run: (1) >=2 of the realized piers are non-BoC
    authority-hauntology tracks (pier-capture + INDEPENDENT authority re-read); (2) the
    "on-tag anchors: injected N" log fired with N>=2; and worst-edge min-T stays healthy
    (>= ~0.4; measured 0.537 vs the 0.327 Phase-A-only baseline — anchors IMPROVE it here).
    """
    # (1) Realized piers include the injected non-BoC on-tag anchors (cheap capture).
    piers = _select_piers(BOC, ["hauntology"], popular_seeds_mode="off")
    anchor_universe = _nonseed_authority_on_tag_ids(["hauntology"], BOC, _db_path())
    assert anchor_universe, "fixture assumption broken: no non-BoC hauntology tracks in authority"
    anchor_piers = [p for p in piers if p in anchor_universe]
    assert len(anchor_piers) >= 2, (
        f"expected >=2 non-BoC authority-hauntology anchor piers, got {len(anchor_piers)} "
        f"(realized piers={piers})"
    )

    # (2) Full run: the injection log fires (N>=2) and the worst edge stays healthy.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        res = _artist_generator(["hauntology"]).create_playlist_for_artist(
            artist_name=BOC, track_count=30, popular_seeds_mode="off", random_seed=0,
        )
    assert res is not None and res.get("track_ids"), "BoC+hauntology generation produced nothing"

    injected = [
        m.group(1) for r in caplog.records
        for m in [re.search(r"on-tag anchors: injected (\d+)", r.getMessage())] if m
    ]
    assert injected, "expected the 'Tag steering on-tag anchors: injected N' INFO log to fire"
    assert max(int(n) for n in injected) >= 2, (
        f"anchor injection log fired but with N<2: {injected}"
    )

    t_min = ((res.get("ds_report") or {}).get("metrics") or {}).get("min_transition")
    assert t_min is not None, "missing min_transition metric"
    # ~0.4 floor per the plan; measured 0.537 gives comfortable margin.
    assert t_min >= 0.40, f"worst-edge regressed below ~0.4 with anchors: min_transition={t_min:.3f}"


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_phase_b_anchors_no_regression_real_estate_jangle_pop():
    """No-regression guard: injecting on-tag anchors must not blow up the worst edge on
    an already-on-genre artist. Real Estate + jangle pop (default anchor_max) vs. the
    anchor_max=0 baseline — worst-edge min-T not worse by >~1.5 notch (0.075), and no
    edge below the transition floor.

    Measured 2026-07-09 (pre-corridor-flip, legacy pooling): with-anchor 0.849 vs
    baseline 0.790 (anchors slightly HELP), below_floor=0 both.

    RE-CALIBRATED Phase 1 Task 9 (2026-07-18): with-anchor 0.610 vs baseline 0.684
    (anchors now cost ~0.074, deterministic across repeated runs at random_seed=0) --
    below_floor=0 still holds for both (the actual quality-floor bar this test's second
    assertion pins). The absolute drop from ~0.79-0.85 to ~0.61-0.68 matches the
    corridor Task 8 pool-starvation fix exactly (.superpowers/sdd/p1-task-8-report.md
    section 3+6): deleting build_balanced_candidate_pool's call left Artist mode
    hard-clamped to a small pre-restricted slice until that fix, so EVERY Artist-mode
    generation now genuinely scans the ~43k-track library instead of a few hundred/
    thousand pre-filtered candidates -- Task 8's own 12-cell corpus already measured
    this same magnitude of across-the-board min_T shift (up to -0.35) as an accepted,
    documented consequence of the fix, not a new bug. The relative tolerance widened
    from 0.05 to 0.075 to accommodate the larger noise floor of the now-much-wider
    admission universe (still well inside "anchors are roughly neutral", not silently
    uncapped) -- verified deterministic, not a blind loosen-to-pass. Flagged for
    Dylan in .superpowers/sdd/p1-task-9-report.md: anchors flipped from "slightly
    help" to "slightly cost" under the wider universe; below_floor==0 (the hard
    quality bar) is unaffected.

    NB: the plan also lists Eno + neoclassical here, but that case genuinely REGRESSES
    under Phase B (with-anchor worst-edge 0.133 / 2 edges below floor vs 0.528 baseline,
    reproducible 2026-07-09) — the injected neoclassical anchors are sonically distant
    from the Eno pier chain once they must connect to their SEQUENTIAL pier neighbours
    (bridgeability is only checked against the nearest seed pier, not the ordered chain).
    Real Estate is used as the no-regression case; the Eno regression is a known Phase-B
    limitation flagged for follow-up, not asserted here.
    """
    with_anchors = _artist_generator(["jangle pop"], anchor_max=3).create_playlist_for_artist(
        artist_name=REAL_ESTATE, track_count=30, popular_seeds_mode="off", random_seed=0,
    )
    baseline = _artist_generator(["jangle pop"], anchor_max=0).create_playlist_for_artist(
        artist_name=REAL_ESTATE, track_count=30, popular_seeds_mode="off", random_seed=0,
    )
    assert with_anchors is not None and baseline is not None

    m_with = (with_anchors.get("ds_report") or {}).get("metrics") or {}
    m_base = (baseline.get("ds_report") or {}).get("metrics") or {}
    t_with = m_with.get("min_transition")
    t_base = m_base.get("min_transition")
    assert t_with is not None and t_base is not None, (
        f"missing min_transition metric: with={m_with} base={m_base}"
    )
    assert t_with >= t_base - 0.075, (
        f"anchor injection regressed the worst edge by >1.5 notch: with-anchors "
        f"min_transition={t_with:.3f} baseline={t_base:.3f}"
    )
    # Still on-genre / no broken edge introduced by the anchors -- the bar that
    # actually matters and is unaffected by the corridor-wide-universe recalibration
    # above.
    assert (m_with.get("below_floor") or 0) == 0, (
        f"anchors introduced {m_with.get('below_floor')} below-floor edge(s) (worst-edge "
        f"broken) for Real Estate + jangle pop"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_phase_b_anchors_rollback_seed_only_piers():
    """Rollback guard (#22): tag_steering_anchor_max=0 must inject ZERO anchors, so the
    realized piers are seed-artist-only (Phase-A-only behaviour). BoC + hauntology at
    anchor_max=0 -> all piers are Boards of Canada, none in the non-BoC on-tag anchor
    universe. Measured 2026-07-09: 4 BoC piers, 0 non-seed.
    """
    piers = _select_piers(BOC, ["hauntology"], popular_seeds_mode="off", anchor_max=0)
    assert len(piers) >= 3, f"too few piers realized: {piers}"
    anchor_universe = _nonseed_authority_on_tag_ids(["hauntology"], BOC, _db_path())
    assert anchor_universe, "fixture assumption broken: no non-BoC hauntology tracks in authority"
    intruders = [p for p in piers if p in anchor_universe]
    assert not intruders, (
        f"anchor_max=0 (rollback) still injected non-seed anchor pier(s): {intruders} "
        f"(realized piers={piers})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 Task 1: retire the pool-level artist cap + never-starve backstop
# (docs/superpowers/plans/2026-07-12-corridor-phase0-subtraction.md, Task 1)
#
# The cap (CandidatePoolConfig.candidates_per_artist, walked in
# src/playlist/candidate_pool.py's per-artist rank walk) and the backstop
# (CandidatePoolConfig.min_pool_size) both go: the beam already enforces
# per-segment/cross-segment artist diversity, and the pool cap alone was
# independently deleting 14-60% of post-floor candidates (M2, docs/
# POOL_STARVATION_RESEARCH_2026-07-12.md).
#
# Neither field has the config.yaml seam the original task brief assumed:
#   - candidates_per_artist: default_ds_config (src/playlist/config.py:544-560)
#     computes it PURELY from `mode` + `playlist_len` via a mode-keyed formula.
#     overrides.get("candidate_pool") is never consulted for it at all -- no
#     yaml key reaches it under ANY path. Confirmed independently two ways:
#     scripts/corridor_baseline/perturb.py's _CANDIDATE_POOL_FIELD_MAP maps it
#     to None with the comment "no yaml key ... reaches them under any path --
#     confirmed by reading default_ds_config top to bottom", and
#     docs/corridor_baseline/knob_sweep.json's real sweep run recorded
#     status="unmapped" for both live corridor cells. A config_overrides
#     splice therefore CANNOT move this field -- see
#     _force_candidates_per_artist below for the (only) way to isolate it.
#   - min_pool_size: default_ds_config never populates CandidatePoolConfig.
#     min_pool_size from ANY override at all (it only reaches its dataclass
#     default of 0 there). The live value comes from a DIFFERENT family --
#     src/playlist/pipeline/core.py:540-552 reads pb_overrides["min_pool_size"]
#     (i.e. playlists.ds_pipeline.pier_bridge.min_pool_size, populated by
#     sonic_mode presets in mode_presets.py) and injects it post-hoc via
#     `replace(cfg.candidate, min_pool_size=...)`. That path IS reachable
#     through config_overrides (verified below).
# ─────────────────────────────────────────────────────────────────────────────

BET_ARTIST = "Bill Evans Trio"


def _load_bundle_for_seed_lookup():
    """load_artifact_bundle(ART) directly, with the process-wide sonic-variant
    override freshly re-resolved from config.yaml first.

    tests/conftest.py's autouse ``_reset_sonic_variant_override`` fixture resets
    the global to None before every test (so a synthetic-artifact test elsewhere
    never inherits it); a bare ``load_artifact_bundle(str(ART))`` here would then
    fail on the MuQ-only live artifact ("Artifact missing required keys:
    ['X_sonic']"). ``resolved_artifact_path()`` re-runs the same
    ``load_config_with_overrides`` call generate_like_gui uses internally, so
    this bundle and the one generate_like_gui loads later in the same test are
    guaranteed to resolve the identical sonic variant.
    """
    resolved_artifact_path()
    load_artifact_bundle.cache_clear()
    return load_artifact_bundle(str(ART))


def _dense_artist_seeds(bundle, artist_name: str, n: int) -> list[str]:
    """Deterministic n pier track_ids for a dense corpus artist.

    Sorted lexicographically by track_id (stable across machines/runs -- no
    similarity ranking involved), so seed selection itself introduces zero
    randomness into the RED/GREEN comparison. Bill Evans Trio is used
    throughout the corridor baselines (scripts/corridor_baseline/runner.py's
    CORPUS/SWEEP_CELLS) -- dense (66 tracks in the live artifact, verified) and
    known-bridgeable, so it needs no fixture beyond the shared artifact bundle.
    """
    arts = [str(a) for a in bundle.track_artists]
    idxs = [i for i, a in enumerate(arts) if a.strip().lower() == artist_name.strip().lower()]
    assert len(idxs) >= n, f"{artist_name} has only {len(idxs)} tracks in the artifact, need >= {n}"
    tids = sorted(str(bundle.track_ids[i]) for i in idxs)
    return tids[:n]


def _force_candidates_per_artist(monkeypatch: pytest.MonkeyPatch, cap: int) -> None:
    """Monkeypatch default_ds_config so the resolved CandidatePoolConfig carries
    candidates_per_artist=cap, regardless of mode/playlist_len -- the only way
    to isolate this field (see the module-docstring-comment above for why no
    config_overrides path reaches it). Patches the name in
    src.playlist.pipeline.core's namespace, which is where
    generate_playlist_ds -> _generate_playlist_ds_impl actually calls it
    (src/playlist/pipeline/core.py:361), so every other resolved value (floors,
    target_artists, min_gap, ...) stays exactly what production would compute
    for this mode/playlist_len -- only the per-artist cap differs.
    """
    real = _pipeline_core.default_ds_config

    def _wrapped(mode, *, playlist_len, overrides=None):
        cfg = real(mode, playlist_len=playlist_len, overrides=overrides)
        return dataclasses.replace(
            cfg, candidate=dataclasses.replace(cfg.candidate, candidates_per_artist=cap)
        )

    monkeypatch.setattr(_pipeline_core, "default_ds_config", _wrapped)


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_artist_cap_knob_is_inert_after_phase0(monkeypatch):
    """Phase 0 Task 1: the pool-level per-artist cap must no longer gate
    build_candidate_pool. Same seeds+modes at candidates_per_artist=2 vs =12
    (forced directly onto the resolved CandidatePoolConfig, holding every other
    resolved value fixed) must produce the IDENTICAL candidate pool once the
    cap walk is deleted.

    Primary assertions are on playlist_stats["candidate_pool"] (pool_size /
    artist_cap_excluded), NOT track_ids equality. Measured on this seed set:
    cap=2 vs cap=12 already converge to the SAME final 30-track playlist even
    PRE-fix (the beam never needed >2-per-artist here) -- so track_ids
    equality alone is a VACUOUS RED/GREEN signal for this corpus. The cap's
    actual, provable-pre-fix effect is on pool COMPOSITION (measured:
    pool_size 884 (cap=2) vs 3291 (cap=12) out of eligible_count=5619,
    artist_cap_excluded 4735 vs 2328) -- exactly the mechanism Task 1's
    Produces contract targets ("returns every post-floor-eligible index ...
    without per-artist truncation"; "artist_cap_excluded (now always 0)").
    track_ids equality is kept as a secondary no-regression check.
    """
    bundle = _load_bundle_for_seed_lookup()
    seeds = _dense_artist_seeds(bundle, BET_ARTIST, n=4)
    common = dict(
        seeds=seeds, cohesion_mode="dynamic", genre_mode="dynamic",
        sonic_mode="dynamic", pace_mode="dynamic", length=30, random_seed=0,
    )

    _force_candidates_per_artist(monkeypatch, 2)
    a = generate_like_gui(**common)
    _force_candidates_per_artist(monkeypatch, 12)
    b = generate_like_gui(**common)

    # Non-vacuity: prove the forced values actually landed on the resolved
    # config before trusting any equality/inequality below.
    # DsRunResult.effective nests pool-family params under "candidate_pool"
    # (verified empirically -- not the flat shape the original brief assumed).
    assert a.effective.get("candidate_pool", {}).get("candidates_per_artist") == 2, a.effective
    assert b.effective.get("candidate_pool", {}).get("candidates_per_artist") == 12, b.effective

    a_cp = a.playlist_stats["candidate_pool"]
    b_cp = b.playlist_stats["candidate_pool"]
    assert a_cp["eligible_count"] == b_cp["eligible_count"], (a_cp, b_cp)

    # After Phase 0 the cap must not truncate the pool at all: pool_size ==
    # eligible_count for BOTH forced cap values, and artist_cap_excluded == 0
    # regardless of what candidates_per_artist was forced to.
    assert a_cp["pool_size"] == b_cp["pool_size"] == a_cp["eligible_count"], (
        f"candidates_per_artist=2 vs =12 produced DIFFERENT pool sizes after "
        f"Phase 0 (the cap should be fully inert): a_pool={a_cp['pool_size']} "
        f"b_pool={b_cp['pool_size']} eligible={a_cp['eligible_count']}"
    )
    assert a_cp["artist_cap_excluded"] == 0, a_cp
    assert b_cp["artist_cap_excluded"] == 0, b_cp

    assert a.track_ids == b.track_ids, (
        "candidates_per_artist=2 vs =12 produced DIFFERENT final playlists: "
        f"a={a.track_ids} b={b.track_ids}"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_min_pool_size_backstop_is_inert_after_phase0():
    """Phase 0 Task 1: the never-starve backstop must no longer gate the pool.
    Reachable via playlists.ds_pipeline.pier_bridge.min_pool_size (NOT
    candidate_pool.min_pool_size -- see module-docstring-comment above).

    Primary assertions are on playlist_stats["candidate_pool"]["pool_size"],
    NOT track_ids equality -- same rationale as
    test_artist_cap_knob_is_inert_after_phase0: measured on this seed set,
    min_pool_size=0 vs =5000 already produce the SAME final 30-track playlist
    even PRE-fix, but the backstop's actual, provable-pre-fix effect is a
    massive pool-composition swing (measured: pool_size 2146 (off) vs 5000
    (on), distinct_artists 509 vs 1519, out of eligible_count=5619).
    track_ids equality is kept as a secondary no-regression check.
    """
    bundle = _load_bundle_for_seed_lookup()
    seeds = _dense_artist_seeds(bundle, BET_ARTIST, n=4)
    common = dict(
        seeds=seeds, cohesion_mode="dynamic", genre_mode="dynamic",
        sonic_mode="dynamic", pace_mode="dynamic", length=30, random_seed=0,
    )

    off = generate_like_gui(
        **common,
        config_overrides={"playlists": {"ds_pipeline": {"pier_bridge": {"min_pool_size": 0}}}},
    )
    on = generate_like_gui(
        **common,
        config_overrides={"playlists": {"ds_pipeline": {"pier_bridge": {"min_pool_size": 5000}}}},
    )

    off_cp = off.playlist_stats["candidate_pool"]
    on_cp = on.playlist_stats["candidate_pool"]
    assert off_cp["eligible_count"] == on_cp["eligible_count"], (off_cp, on_cp)

    # After Phase 0 the backstop must not act at all: pool_size == pool_size
    # regardless of min_pool_size (the field becomes a pure no-op).
    assert off_cp["pool_size"] == on_cp["pool_size"], (
        f"min_pool_size=0 vs =5000 produced DIFFERENT pool sizes after Phase 0 "
        f"(the backstop should be fully inert): off_pool={off_cp['pool_size']} "
        f"on_pool={on_cp['pool_size']}"
    )

    assert off.track_ids == on.track_ids, (
        "min_pool_size=0 vs =5000 produced DIFFERENT final playlists: "
        f"off={off.track_ids} on={on.track_ids}"
    )
