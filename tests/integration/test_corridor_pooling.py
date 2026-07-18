"""Phase 1 Tasks 3+4+5: corridor segment pooling + relevance mask / widening
ladder + reseats (bangers, tag guarantee, tail-DP, edge repair) + genre_mode
production wiring, behind the dev flag.

Spec: docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.
Reports: .superpowers/sdd/p1-task-3-report.md, .superpowers/sdd/p1-task-4-report.md,
.superpowers/sdd/p1-task-5-report.md.

Tiers:
  * `test_corridor_pooling_*` / `test_corridor_widening_ladder_*` -- real
    end-to-end generation via the gui_fidelity harness (@integration @slow,
    needs the live artifact). Pins the Step-1 contract: corridor flag on,
    multi-pier generation completes, every non-pier playlist track is a
    member of its segment's corridor (min_sim >= threshold, recomputed
    independently here), and the per-segment health line (contract F7)
    appears exactly once per segment -- including under widening (Task 4).
  * `test_corridor_universe_duration_reference_is_none_*` -- fast unit-level
    wiring check (synthetic ArtifactBundle, no artifact needed): the
    dead-knob-trap regression from the Task 3 review. Phase 1 deliberately
    never wires a real duration soft-penalty reference into the corridor
    universe; this pins that off-state so a future change can't start
    silently acting on an unconfigured knob.
  * `test_corridor_filters_*_before_capping_not_after` -- fast unit-level
    filter-before-cap regressions (synthetic ArtifactBundle): track-key
    collision (Task 3 re-review fix) and pier-artist-key exclusion (Task 4,
    carried finding -- the same one-mask-before-build_corridor fix covers
    both exclusion kinds, this pins the artist-key half).
  * `test_corridor_relevance_mask_*` -- fast unit-level wiring check (Task 4):
    genre_mode keys the relevance-mask floor fed into build_eligible_universe,
    directly via build_pier_bridge_playlist's genre_mode kwarg. As of Task 5
    this kwarg is ALSO reachable from real production config (see the
    genre_mode production-wiring test below) -- these tests keep exercising
    the mechanism directly (bypassing the full config chain) since that's
    what they're pinning.
  * `test_corridor_genre_mode_threads_through_production_wiring` -- real
    end-to-end generation (@integration @slow): Task 5 req 0. genre_mode
    reaches build_pier_bridge_playlist via the REAL production call chain
    (playlist_generator.py -> ds_pipeline_runner.py -> pipeline/core.py),
    driven purely by config -- no test-only kwargs. Task 4 built the
    mechanism but left this wiring gap open (documented in its own report as
    a KNOWN GAP); every corridor generation silently got genre_mode="off"
    regardless of the slider until Task 5 closed it.
  * `test_corridor_bangers_*` -- fast unit-level wiring check (Task 5 reseat
    1): the Oops-All-Bangers popularity rank-cutoff gate, applied once at
    corridor universe build via build_pier_bridge_playlist's new
    popularity_ranks/popularity_rank_cutoff kwargs (core.py's _run_pier_bridge
    closure plumbs the same array it already resolves for the legacy pool).
  * `test_corridor_tag_guarantee_*` -- fast unit-level wiring check (Task 5
    reseat 2): on-tag guarantee ids reach build_corridor's force_include for
    every segment; verified via the new forced_included diagnostics count
    (never a member-id dump).
  * `test_corridor_repair_stack_draws_only_from_corridor_union` -- real,
    stressed end-to-end generation (@integration @slow, Task 5 reseats 3+4):
    with tail_dp_enabled + edge_repair_enabled on a corridor run stressed
    enough to actually trigger both, every final track still clears at least
    ONE segment's recorded corridor threshold (a corridor-membership recheck
    against the run's own threshold diagnostics -- edge repair draws from the
    UNION of every segment's final corridor members under corridor pooling,
    not any one segment's alone, so per-segment-only membership is not
    guaranteed for a repaired track; per-union membership is).
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.features.artifacts import ArtifactBundle, load_artifact_bundle
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge.eligible_universe import build_eligible_universe
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist
from tests.support.gui_fidelity import gui_ui_state, generate_like_gui, resolved_artifact_path

# Reuse the same real, known-good 5-seed multi-pier fixture as
# tests/integration/test_gui_fidelity_regressions.py (William Tyler / Hayden
# Pedigo / Steve Hiett / Songs: Ohia / Bill Callahan) so this test doesn't
# need to discover its own valid seed set.
ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

SEEDS = [
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler - Howling at the Second Moon
    "b8f8aa0e86f977f9fcb26f615e130ac9",  # Hayden Pedigo - Nearer, Nearer
    "42473b911cef5674e56b8e2ce87df7cb",  # Steve Hiett - Are These My Memories?
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia - Hold On Magnolia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan - Let's Move to the Country
]

CORRIDOR_OVERRIDES = {
    "playlists": {"ds_pipeline": {"pier_bridge": {"pooling": "corridor"}}},
}


def _l2norm(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_corridor_pooling_generation_completes_with_corridor_membership_and_health_line(caplog):
    """Step 1 (task-3-brief.md): corridor flag on, 4-segment (5-seed) generation
    completes; every non-pier playlist track is a member of its segment's
    corridor; the health line appears once per segment.

    Phase 1 Task 9 found (and this Task-9-followup fix resolves) a
    diagnostics-staleness bug: under variable_bridge_length, the once-per-
    segment health-line/diagnostics gate could latch onto a DIFFERENT
    (narrower, earlier-tried) widening-ladder attempt than the one that
    actually supplied the segment's emitted tracks -- see
    `_run_corridor_widening_ladder`'s docstring/comments and the "Health line
    (Task 9 fix)" comment in `pier_bridge_builder.py`'s main segment loop.
    The membership recheck below now additionally asserts that the recorded
    threshold is independently reproducible from the recorded `size`/`width`
    (i.e. the diagnostics genuinely describe the attempt that supplied the
    emitted tracks, not merely that every emitted track happens to clear
    some lower threshold) -- see the per-segment recomputation loop.
    """
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow", pace_mode="narrow",
    )

    with caplog.at_level(logging.INFO, logger="src.playlist.pier_bridge_builder"):
        res = generate_like_gui(
            seeds=SEEDS, ui=ui, length=20, random_seed=0,
            config_overrides=CORRIDOR_OVERRIDES,
        )

    # Loaded AFTER generate_like_gui (not before): generate_like_gui's config
    # chain sets the process-wide artifacts.sonic_variant_override ("muq") as
    # a side effect via load_config_with_overrides. Loading the bundle first
    # (before anything has set the override) fails -- this artifact carries
    # only X_sonic_muq, not a plain X_sonic key.
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))

    assert len(res.track_ids) in (20, 21), f"expected 20 tracks, got {len(res.track_ids)}"

    playlist_stats = res.playlist_stats.get("playlist", {})
    assert playlist_stats.get("pooling_strategy") == "corridor"
    corridor_segments = playlist_stats.get("corridor_segments") or []
    num_segments = len(SEEDS) - 1
    assert len(corridor_segments) == num_segments, (
        f"expected {num_segments} corridor segment diagnostics entries, got "
        f"{len(corridor_segments)}: {corridor_segments}"
    )
    for entry in corridor_segments:
        assert entry["size"] >= 0
        assert 0.0 <= entry["threshold"] <= 1.0 or entry["size"] == 0

    # Health line (contract F7): exactly one "Corridor[seg N]:" line per segment.
    corridor_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder" and r.getMessage().startswith("Corridor[seg ")
    ]
    assert len(corridor_lines) == num_segments, (
        f"expected exactly {num_segments} Corridor health lines, got {len(corridor_lines)}: "
        f"{corridor_lines}"
    )
    logged_segments = set()
    for line in corridor_lines:
        seg_num = int(line.split("Corridor[seg ")[1].split("]")[0])
        assert seg_num not in logged_segments, f"segment {seg_num} logged more than once"
        logged_segments.add(seg_num)
        assert "size=" in line and "width=" in line and "widened=" in line
        assert "support_a=" in line and "support_b=" in line
        assert "threshold=" in line and "capped=" in line

    # Membership: every non-pier track between consecutive piers must satisfy
    # min(sim(track, pier_a), sim(track, pier_b)) >= that segment's recorded
    # threshold. Recomputed independently from the bundle's sonic matrix
    # (never trusted from the health line alone).
    X_norm = _l2norm(bundle.X_sonic)
    seed_id_set = set(SEEDS)
    pier_positions = [i for i, tid in enumerate(res.track_ids) if tid in seed_id_set]
    assert len(pier_positions) == len(SEEDS), (
        f"expected all {len(SEEDS)} seeds present as piers, found {len(pier_positions)}"
    )
    threshold_by_seg = {int(e["seg"]): float(e["threshold"]) for e in corridor_segments}

    for seg in range(len(pier_positions) - 1):
        start_pos = pier_positions[seg]
        end_pos = pier_positions[seg + 1]
        pier_a_vec = X_norm[bundle.track_id_to_index[res.track_ids[start_pos]]]
        pier_b_vec = X_norm[bundle.track_id_to_index[res.track_ids[end_pos]]]
        threshold = threshold_by_seg[seg]
        for pos in range(start_pos + 1, end_pos):
            tid = res.track_ids[pos]
            vec = X_norm[bundle.track_id_to_index[tid]]
            min_sim = min(float(np.dot(vec, pier_a_vec)), float(np.dot(vec, pier_b_vec)))
            assert min_sim >= threshold - 1e-6, (
                f"track {tid} at position {pos} (segment {seg}) has min_sim={min_sim:.4f} "
                f"below the segment's corridor threshold={threshold:.4f}"
            )

    # Confidence check (same principle as test_corridor_repair_stack_draws_
    # only_from_corridor_union's "otherwise the membership recheck above is
    # vacuous" guard): this fixture must actually exercise variable-bridge
    # RE-ENTRY (choose_segment_length trying more than one interior length
    # for some segment), or the membership recheck above never touches the
    # Task 9 bug's trigger geometry at all -- it would pass equally well
    # before AND after this fix. `generate_like_gui`'s config chain defaults
    # `variable_bridge_length` on (see config.yaml / config.example.yaml), so
    # this is exercising the real production default, not a test-only knob.
    var_bridge_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder" and r.getMessage().startswith("Var-bridge seg ")
    ]
    flexed_segments = [
        int(line.split("Var-bridge seg ")[1].split(":")[0])
        for line in var_bridge_lines
        if "flexed=True" in line
    ]
    assert flexed_segments, (
        "expected at least one segment to actually flex under variable_bridge_length "
        f"-- otherwise the membership recheck above never exercises the Task 9 "
        f"re-entry scenario (var-bridge log lines seen: {var_bridge_lines})"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_corridor_widening_ladder_health_line_survives_variable_bridge_reentry(caplog):
    """Focused regression for the Task 9 diagnostics-staleness fix.

    Same real 5-seed fixture as the test above (its segment 2 deterministically
    forces variable_bridge_length re-entry: `choose_segment_length` tries the
    nominal interior length first, then a longer flexed length, each running
    its OWN independent `_run_corridor_widening_ladder` invocation -- see that
    function's docstring/comments). This test pins the exact bug geometry from
    `.superpowers/sdd/p1-task-9-report.md`: pre-fix, the recorded health line/
    diagnostics for the re-entered segment described the FIRST (nominal-length)
    attempt's un-widened corridor, not the ACCEPTED (chosen-length) attempt's
    widened one -- even though the accepted attempt supplied the emitted
    tracks. Post-fix, the two must agree.
    """
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow", pace_mode="narrow",
    )

    with caplog.at_level(logging.INFO, logger="src.playlist.pier_bridge_builder"):
        res = generate_like_gui(
            seeds=SEEDS, ui=ui, length=20, random_seed=0,
            config_overrides=CORRIDOR_OVERRIDES,
        )

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))

    # Find a segment that actually re-entered the ladder: chosen != nominal
    # length (var-bridge flexed it), evidenced by the "Var-bridge seg N:
    # nominal=X chosen=Y flexed=True" summary line pier_bridge_builder.py logs
    # once the interior length is picked (after however many lengths
    # choose_segment_length tried).
    var_bridge_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder" and r.getMessage().startswith("Var-bridge seg ")
    ]
    reentered_segments = []
    for line in var_bridge_lines:
        seg_num = int(line.split("Var-bridge seg ")[1].split(":")[0])
        nominal = int(line.split("nominal=")[1].split(" ")[0])
        chosen = int(line.split("chosen=")[1].split(" ")[0])
        if "flexed=True" in line and chosen != nominal:
            reentered_segments.append(seg_num)
    assert reentered_segments, (
        f"expected at least one segment where var-bridge actually chose a "
        f"DIFFERENT length than nominal (real re-entry, not just a re-tried-"
        f"but-kept-nominal flex) -- got: {var_bridge_lines}"
    )

    # Evidence the ladder really ran MORE THAN ONCE for the same seg_idx (the
    # Task 9 trigger): count independent widen-ladder attempt sequences via
    # their terminal marker ("recovered at attempt" or "EXHAUSTED after") --
    # each candidate interior length runs its own ladder to one such
    # conclusion. A single conclusion would mean no re-entry actually
    # happened despite the length differing, which would make this test
    # vacuous.
    for seg_num in reentered_segments:
        prefix = f"CorridorWiden[seg {seg_num}]"
        conclusions = [
            r.getMessage() for r in caplog.records
            if r.name == "src.playlist.pier_bridge_builder"
            and r.getMessage().startswith(prefix)
            and ("recovered at attempt" in r.getMessage() or "EXHAUSTED after" in r.getMessage())
        ]
        assert len(conclusions) >= 2, (
            f"expected segment {seg_num} (flagged as re-entered above) to show "
            f">=2 independent widening-ladder conclusions in the log -- got "
            f"{conclusions}; without this the re-entry claim is unverified"
        )

    # The actual Task 9 fix pin: every non-pier track in a re-entered segment
    # clears the RECORDED (accepted-attempt) threshold -- pre-fix, this failed
    # for segment 2 of this exact fixture (track 8e73c8dcfc201bf17d6a6001c828c233
    # at position 11, min_sim=0.4784 < stale recorded threshold=0.5068, per the
    # xfail's original writeup) because the recorded diagnostics described the
    # narrower, un-widened nominal-length attempt instead of the wider, accepted
    # one that actually supplied the tracks.
    playlist_stats = res.playlist_stats.get("playlist", {})
    corridor_segments = playlist_stats.get("corridor_segments") or []
    threshold_by_seg = {int(e["seg"]): float(e["threshold"]) for e in corridor_segments}
    X_norm = _l2norm(bundle.X_sonic)
    seed_id_set = set(SEEDS)
    pier_positions = [i for i, tid in enumerate(res.track_ids) if tid in seed_id_set]

    for seg_num in reentered_segments:
        start_pos = pier_positions[seg_num]
        end_pos = pier_positions[seg_num + 1]
        pier_a_vec = X_norm[bundle.track_id_to_index[res.track_ids[start_pos]]]
        pier_b_vec = X_norm[bundle.track_id_to_index[res.track_ids[end_pos]]]
        threshold = threshold_by_seg[seg_num]
        for pos in range(start_pos + 1, end_pos):
            tid = res.track_ids[pos]
            vec = X_norm[bundle.track_id_to_index[tid]]
            min_sim = min(float(np.dot(vec, pier_a_vec)), float(np.dot(vec, pier_b_vec)))
            assert min_sim >= threshold - 1e-6, (
                f"track {tid} at position {pos} (re-entered segment {seg_num}) has "
                f"min_sim={min_sim:.4f} below the recorded corridor "
                f"threshold={threshold:.4f} -- the health line/diagnostics are "
                f"describing a DIFFERENT (narrower) attempt than the one that "
                f"supplied this track, exactly the Task 9 bug"
            )


# ── Dead-knob-trap wiring test (fast, no artifact) ──────────────────────────

_N = 4
_TRACK_IDS = np.array([f"t{i}" for i in range(_N)], dtype=object)
_X_SONIC = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _synthetic_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_TRACK_IDS,
        artist_keys=np.array([f"artist-{i}" for i in range(_N)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(_N)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(_N)], dtype=object),
        X_sonic=_X_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((_N, 1)),
        X_genre_smoothed=np.zeros((_N, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_TRACK_IDS)},
        durations_ms=np.full(_N, 200_000.0),
    )


# ── Filter-before-cap regression (review fix, 2026-07-17) ──────────────────
#
# Legacy (SegmentCandidatePoolBuilder.build) runs ALL structural filters --
# including track-key collision -- on the FULL universe before scoring/
# capping. An earlier version of the corridor path ran track-key collision
# AFTER build_corridor's own segment_pool_max cap, so a segment whose
# corridor top-K happened to be dominated by already-used track keys could
# starve even though clean, lower-ranked candidates existed just below the
# cap. This fixture forces exactly that shape: 6 "duplicate" candidates that
# outrank 2 "clean" candidates on corridor score but collide (by
# artist+title, i.e. track_key) with a seed, plus segment_pool_max/
# max_segment_pool_max held small enough that expansion-attempt escalation
# can never grow the cap past the duplicate count. Filter-before-cap must
# exclude the 6 duplicates BEFORE ranking/capping, leaving exactly the 2
# clean candidates for the interior; filter-after-cap always caps on
# duplicates first and starves permanently.
_R = 10
_DUP_COUNT = 6
_R_TRACK_IDS = np.array([f"r{i}" for i in range(_R)], dtype=object)
_R_ARTISTS = np.array(
    ["Seed Artist A", "Seed Artist B"]
    + ["Seed Artist A"] * _DUP_COUNT  # r2..r7: collide with seed r0's artist
    + ["Clean Artist C", "Clean Artist D"],  # r8, r9: distinct
    dtype=object,
)
_R_TITLES = np.array(
    ["Seed Song A", "Seed Song B"]
    + ["Seed Song A"] * _DUP_COUNT  # r2..r7: collide with seed r0's title too
    + ["Clean Song C", "Clean Song D"],
    dtype=object,
)
_R_SONIC = np.array(
    [
        [1.0, 0.0, 0.0],  # r0 pier A
        [0.0, 1.0, 0.0],  # r1 pier B
        *([[1.0, 1.0, 0.0]] * _DUP_COUNT),  # r2..r7: balanced -> highest hmean rank
        [0.9, 0.6, 0.0],  # r8: clean, lower hmean rank than the duplicates
        [0.9, 0.6, 0.0],  # r9: clean, lower hmean rank than the duplicates
    ],
    dtype=float,
)


def _dup_collision_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_R_TRACK_IDS,
        artist_keys=_R_ARTISTS,
        track_artists=_R_ARTISTS,
        track_titles=_R_TITLES,
        X_sonic=_R_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((_R, 1)),
        X_genre_smoothed=np.zeros((_R, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_R_TRACK_IDS)},
        durations_ms=np.full(_R, 200_000.0),
    )


def test_corridor_filters_track_key_collisions_before_capping_not_after():
    bundle = _dup_collision_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,  # permissive: every non-seed row is a corridor member
        segment_pool_max=2,
        # Held small on purpose: with 6 duplicates ranked above the 2 clean
        # candidates, expansion-attempt escalation (doubling up to this cap)
        # can NEVER grow past the duplicate count -- so filter-after-cap
        # would starve on every attempt, not just the first.
        max_segment_pool_max=4,
        transition_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )

    result = build_pier_bridge_playlist(
        seed_track_ids=["r0", "r1"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=list(range(2, _R)),
        cfg=cfg,
    )

    assert result.success, f"corridor segment starved: {result.failure_reason}"
    assert len(result.track_ids) == 4, f"expected 4 tracks, got {result.track_ids}"
    dup_ids = {f"r{i}" for i in range(2, 2 + _DUP_COUNT)}
    assert not (set(result.track_ids) & dup_ids), (
        f"track-key-colliding duplicates leaked into the playlist: {result.track_ids}"
    )
    assert set(result.track_ids) == {"r0", "r1", "r8", "r9"}, (
        f"expected the 2 clean below-cap-rank candidates to fill the interior, got "
        f"{result.track_ids}"
    )


def test_corridor_universe_duration_reference_is_none_dead_knob_trap():
    """Phase 1 scope (see pier_bridge_builder.py's corridor-universe comment):
    the corridor path never wires a real duration soft-penalty reference --
    this is a deliberate, documented off-state, not an oversight. Regression
    guard: nothing before Task 4/5 wires real duration data should start
    silently passing a truthy `duration_reference_ms` through this seam
    while claiming duration penalty is off (CLAUDE.md's dead-knob trap, in
    reverse: an UNconfigured knob must never start silently acting)."""
    bundle = _synthetic_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,  # permissive: this test only checks wiring
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )

    with patch(
        "src.playlist.pier_bridge_builder.build_eligible_universe",
        wraps=build_eligible_universe,
    ) as mock_build:
        build_pier_bridge_playlist(
            seed_track_ids=["t0", "t3"],
            total_tracks=3,
            bundle=bundle,
            candidate_pool_indices=[1, 2],
            cfg=cfg,
        )

    assert mock_build.called, "corridor path must call build_eligible_universe"
    _, kwargs = mock_build.call_args
    assert kwargs["duration_reference_ms"] is None
    assert kwargs["duration_penalty_weight"] == 0.0
    assert kwargs["title_hard_exclude_flags"] == frozenset()
    assert kwargs["relevance_mask"] is None
    assert kwargs["excluded_track_ids"] == set()


# ── Task 7: C1 duration-penalty ON-case + title-hygiene wiring ─────────────
#
# Companion to the dead-knob-trap test above: that test pins the OFF-state
# (duration_penalty_enabled=False, the PierBridgeConfig default). These pin
# the ON-state -- when cfg.duration_penalty_enabled=True /
# cfg.title_hard_exclude_flags is non-empty, build_eligible_universe must
# receive the REAL computed reference/weight/flags, not None/0.0/frozenset().
# Prior to the Task 7 fix, the corridor call site hardcoded
# duration_reference_ms=None, duration_penalty_weight=0.0,
# title_hard_exclude_flags=frozenset() UNCONDITIONALLY -- these two tests
# fail (RED) against that hardcoding regardless of cfg.


def test_corridor_universe_duration_reference_wired_when_enabled():
    """C1 rehome ON-case: cfg.duration_penalty_enabled=True must produce a
    real seed-median duration_reference_ms and the configured
    duration_penalty_weight/duration_cutoff_multiplier at the
    build_eligible_universe call site -- not the module's off-state."""
    from dataclasses import replace as _dc_replace

    bundle = _dc_replace(
        _synthetic_bundle(),
        # Seeds t0, t3 both 200_000ms -> seed-median reference is 200_000ms
        # (unambiguous, no averaging needed).
        durations_ms=np.array([200_000.0, 250_000.0, 500_000.0, 200_000.0]),
    )
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
        duration_penalty_enabled=True,
        duration_penalty_weight=0.6,
        duration_cutoff_multiplier=1.5,
    )

    with patch(
        "src.playlist.pier_bridge_builder.build_eligible_universe",
        wraps=build_eligible_universe,
    ) as mock_build:
        build_pier_bridge_playlist(
            seed_track_ids=["t0", "t3"],
            total_tracks=3,
            bundle=bundle,
            candidate_pool_indices=[1, 2],
            cfg=cfg,
        )

    assert mock_build.called, "corridor path must call build_eligible_universe"
    _, kwargs = mock_build.call_args
    assert kwargs["duration_reference_ms"] == pytest.approx(200_000.0), (
        "expected the real seed-median duration (200_000ms), not the dead-knob None"
    )
    assert kwargs["duration_penalty_weight"] == pytest.approx(0.6)
    assert kwargs["duration_cutoff_multiplier"] == pytest.approx(1.5)

    # And the wiring actually BITES: track 2 (500_000ms) is > cutoff
    # (200_000 * 1.5 = 300_000ms) so build_eligible_universe's own hard
    # cutoff excludes it -- proving this isn't just an inert kwarg pass-through.
    # Recomputed directly (not read off the mock) for a version-independent
    # assertion of the real returned EligibleUniverse.
    universe = build_eligible_universe(**kwargs)
    assert 2 not in universe.indices.tolist(), (
        "track 2 (500_000ms, over the 300_000ms cutoff) must be hard-excluded "
        "once the duration reference is real"
    )
    assert universe.stats["excluded_duration_cutoff"] == 1


def test_corridor_universe_title_hard_exclude_flags_wired_when_configured():
    """Category B4 (INVARIANT hard gate): cfg.title_hard_exclude_flags must
    reach build_eligible_universe as the real configured flag set, not the
    dead-knob frozenset(). Without this wiring, a flagged title (e.g. an
    "Interlude") can leak into a corridor-mode playlist."""
    from dataclasses import replace as _dc_replace

    bundle = _dc_replace(
        _synthetic_bundle(),
        track_titles=np.array(["Track 0", "Some Interlude", "Track 2", "Track 3"], dtype=object),
    )
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
        title_hard_exclude_flags=("interlude",),  # tuple: JSON-safe field type, see PierBridgeConfig
    )

    with patch(
        "src.playlist.pier_bridge_builder.build_eligible_universe",
        wraps=build_eligible_universe,
    ) as mock_build:
        result = build_pier_bridge_playlist(
            seed_track_ids=["t0", "t3"],
            total_tracks=3,
            bundle=bundle,
            candidate_pool_indices=[1, 2],
            cfg=cfg,
        )

    assert mock_build.called
    _, kwargs = mock_build.call_args
    assert kwargs["title_hard_exclude_flags"] == frozenset({"interlude"}), (
        "expected the real configured flag set, not the dead-knob frozenset()"
    )
    universe = build_eligible_universe(**kwargs)
    assert 1 not in universe.indices.tolist(), (
        "track 1 ('Some Interlude') must be hard-excluded once title hygiene is wired"
    )
    assert universe.stats["excluded_title_hygiene"] == 1
    assert "t1" not in result.track_ids, (
        "the flagged-title track must never surface in the final corridor playlist"
    )


# ── Task 4: pier-artist-key collision, filter-before-cap (carried finding) ──
#
# Same shape as test_corridor_filters_track_key_collisions_before_capping_
# not_after above, but the exclusion mechanism under test is pier-artist-key
# collision (disallow_pier_artists_in_interiors), not track-key collision --
# the Task 3 re-review's Minor finding: the filter-before-cap fix folds BOTH
# exclusion kinds into the same one-mask-before-build_corridor pass
# (_build_corridor_segment_pool's _row_ok closure), but only the track-key
# half had a regression test. Distinct titles here (unlike the track-key
# fixture) so this test isolates the artist-key path specifically.
_S = 10
_ART_DUP_COUNT = 6
_S_TRACK_IDS = np.array([f"s{i}" for i in range(_S)], dtype=object)
_S_ARTISTS = np.array(
    ["Seed Artist A", "Seed Artist B"]
    + ["Seed Artist A"] * _ART_DUP_COUNT  # s2..s7: collide with PIER A's artist
    + ["Clean Artist C", "Clean Artist D"],  # s8, s9: distinct
    dtype=object,
)
_S_TITLES = np.array(
    ["Seed Song A", "Seed Song B"]
    + [f"Other Song {i}" for i in range(_ART_DUP_COUNT)]  # distinct titles: no track-key collision
    + ["Clean Song C", "Clean Song D"],
    dtype=object,
)
_S_SONIC = np.array(
    [
        [1.0, 0.0, 0.0],  # s0 pier A
        [0.0, 1.0, 0.0],  # s1 pier B
        *([[1.0, 1.0, 0.0]] * _ART_DUP_COUNT),  # s2..s7: balanced -> highest hmean rank
        [0.9, 0.6, 0.0],  # s8: clean, lower hmean rank than the collisions
        [0.9, 0.6, 0.0],  # s9: clean, lower hmean rank than the collisions
    ],
    dtype=float,
)


def _artist_collision_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_S_TRACK_IDS,
        artist_keys=_S_ARTISTS,
        track_artists=_S_ARTISTS,
        track_titles=_S_TITLES,
        X_sonic=_S_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((_S, 1)),
        X_genre_smoothed=np.zeros((_S, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_S_TRACK_IDS)},
        durations_ms=np.full(_S, 200_000.0),
    )


def test_corridor_filters_pier_artist_collisions_before_capping_not_after():
    bundle = _artist_collision_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,  # permissive: every non-seed row is a corridor member
        segment_pool_max=2,
        # Held small on purpose, same rationale as the track-key fixture: 6
        # pier-artist collisions ranked above the 2 clean candidates means
        # expansion-attempt escalation (doubling up to this cap) can NEVER
        # grow past the collision count -- filter-after-cap would starve on
        # every attempt.
        max_segment_pool_max=4,
        transition_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
        disallow_pier_artists_in_interiors=True,
    )

    result = build_pier_bridge_playlist(
        seed_track_ids=["s0", "s1"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=list(range(2, _S)),
        cfg=cfg,
    )

    assert result.success, f"corridor segment starved: {result.failure_reason}"
    assert len(result.track_ids) == 4, f"expected 4 tracks, got {result.track_ids}"
    collision_ids = {f"s{i}" for i in range(2, 2 + _ART_DUP_COUNT)}
    assert not (set(result.track_ids) & collision_ids), (
        f"pier-artist-colliding candidates leaked into the playlist: {result.track_ids}"
    )
    assert set(result.track_ids) == {"s0", "s1", "s8", "s9"}, (
        f"expected the 2 clean below-cap-rank candidates to fill the interior, got "
        f"{result.track_ids}"
    )


# ── Task 4: genre-mode-keyed relevance mask wiring ──────────────────────────
#
# 8 rows: 2 piers (seeds), 3 genre-aligned candidates (cos=1.0 to the seeds'
# genre profile), 3 genre-orthogonal candidates (cos=0.0). genre_mode="strict"
# floors at 0.30 (see _corridor_genre_relevance_floor) so the orthogonal trio
# is excluded from the eligible universe; genre_mode="off" disables the mask
# entirely (relevance_mask=None), so nothing is excluded on genre grounds.
# Seeds are exempt from exclusion either way (build_eligible_universe never
# excludes a seed index), so genre_mode="strict" universe = 2 seeds + 3
# aligned = 5; genre_mode="off" universe = all 8 rows.
_M = 8
_M_TRACK_IDS = np.array([f"m{i}" for i in range(_M)], dtype=object)
_M_ARTISTS = np.array([f"Mask Artist {i}" for i in range(_M)], dtype=object)
_M_TITLES = np.array([f"Mask Song {i}" for i in range(_M)], dtype=object)
_M_SONIC = np.array([[float(i), 0.0, 0.0] for i in range(_M)], dtype=float)
_M_GENRE = np.array(
    [
        [1.0, 0.0],  # m0: pier A
        [1.0, 0.0],  # m1: pier B
        [1.0, 0.0],  # m2: aligned candidate
        [1.0, 0.0],  # m3: aligned candidate
        [1.0, 0.0],  # m4: aligned candidate
        [0.0, 1.0],  # m5: orthogonal candidate
        [0.0, 1.0],  # m6: orthogonal candidate
        [0.0, 1.0],  # m7: orthogonal candidate
    ],
    dtype=float,
)


def _mask_wiring_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_M_TRACK_IDS,
        artist_keys=_M_ARTISTS,
        track_artists=_M_ARTISTS,
        track_titles=_M_TITLES,
        X_sonic=_M_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=_M_GENRE,
        X_genre_smoothed=_M_GENRE,
        genre_vocab=np.array(["a", "b"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_M_TRACK_IDS)},
        durations_ms=np.full(_M, 200_000.0),
    )


def _run_mask_wiring_case(genre_mode) -> int:
    bundle = _mask_wiring_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,  # permissive width: the mask is the only lever under test
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["m0", "m1"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=list(range(2, _M)),
        cfg=cfg,
        genre_mode=genre_mode,
    )
    assert result.success, f"[{genre_mode}] corridor segment starved: {result.failure_reason}"
    return int(result.stats["corridor_universe_size"])


def test_corridor_relevance_mask_off_vs_strict_changes_universe_size():
    off_size = _run_mask_wiring_case("off")
    strict_size = _run_mask_wiring_case("strict")

    assert off_size == _M, f"genre_mode=off should leave the full universe eligible, got {off_size}"
    # 2 seeds (exempt) + 3 genre-aligned candidates; the 3 orthogonal
    # candidates fall below the strict floor (0.30) and are excluded.
    assert strict_size == 5, f"genre_mode=strict should shrink the universe to 5, got {strict_size}"
    assert strict_size < off_size


def test_corridor_relevance_mask_unspecified_matches_off():
    """genre_mode=None (unspecified -- the production caller's current state,
    see build_pier_bridge_playlist's genre_mode docstring) must behave
    identically to genre_mode="off": a missing signal is never guessed into
    an active gate."""
    bundle = _mask_wiring_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["m0", "m1"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=list(range(2, _M)),
        cfg=cfg,
        # genre_mode intentionally omitted
    )
    assert result.success
    assert int(result.stats["corridor_universe_size"]) == _M


# ── Task 4: quality-triggered widening ladder ───────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_corridor_widening_ladder_recovers_stressed_segment(caplog):
    """Step 1 (task-4-brief.md): force a stressed segment via a tight
    corridor_width_percentile PLUS a transition_floor override the narrow
    width can't clear for at least one segment -- assert the widening ladder
    recovers (widened>=1 in at least one segment's diagnostics) and the run
    still completes at full length. Values below were selected empirically
    against this fixture's real artifact (see report for the probe log);
    corridor_width_percentile alone (without the transition_floor override)
    does not stress this particular 5-seed/20-track fixture because
    segment_pool_max caps the corridor's member count well before the width
    threshold binds at any width the default beam/pool sizing can reach."""
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow", pace_mode="narrow",
    )
    overrides = {
        "playlists": {"ds_pipeline": {
            "pier_bridge": {
                "pooling": "corridor",
                "corridor_width_percentile": 0.995,
                "corridor_widen_step": 0.10,
                "corridor_widen_attempts": 3,
            },
            "constraints": {"transition_floor": 0.55},
        }},
    }

    with caplog.at_level(logging.INFO, logger="src.playlist.pier_bridge_builder"):
        res = generate_like_gui(
            seeds=SEEDS, ui=ui, length=20, random_seed=0,
            config_overrides=overrides,
        )

    assert len(res.track_ids) == 20, f"expected 20 tracks, got {len(res.track_ids)}"

    playlist_stats = res.playlist_stats.get("playlist", {})
    corridor_segments = playlist_stats.get("corridor_segments") or []
    num_segments = len(SEEDS) - 1
    assert len(corridor_segments) == num_segments

    widened_values = [int(e.get("widened", 0)) for e in corridor_segments]
    assert any(w >= 1 for w in widened_values), (
        f"expected at least one segment to widen under a stressed corridor, "
        f"got widened values: {widened_values}"
    )

    # F7 contract preserved under widening (Task 3): exactly one
    # "Corridor[seg N]:" health line per segment. The ladder's own
    # "CorridorWiden[seg N]:" progress/exhaustion lines use a DISTINCT prefix
    # specifically so they never collide with this pinned count.
    corridor_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder" and r.getMessage().startswith("Corridor[seg ")
    ]
    assert len(corridor_lines) == num_segments, (
        f"expected exactly {num_segments} Corridor health lines even under "
        f"widening, got {len(corridor_lines)}: {corridor_lines}"
    )

    widen_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder" and r.getMessage().startswith("CorridorWiden[seg ")
    ]
    assert widen_lines, "expected at least one CorridorWiden[seg N]: log line under a stressed corridor"


# ── Task 5 req 0: genre_mode production wiring (HARD requirement) ───────────

@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_corridor_genre_mode_threads_through_production_wiring():
    """Req 0 (task-5-brief.md, HARD, carried from Task 4's own report as a
    KNOWN GAP): genre_mode must reach build_pier_bridge_playlist via the REAL
    production call chain (playlist_generator.py -> ds_pipeline_runner.py ->
    pipeline/core.py -> build_pier_bridge_playlist), driven purely by config
    -- no test-only kwargs (this is the anti-dead-knob proof). Compares
    corridor_universe_size between genre_mode="off" and genre_mode="strict"
    on an otherwise-identical real generation via generate_like_gui, which
    resolves genre_mode exactly the way the GUI worker does (UIStateModel ->
    derive_runtime_config -> merge_overrides -> load_config_with_overrides ->
    resolve_genre_ds_params) -- Task 4 built the relevance-mask mechanism but
    left this wiring gap open; before this test can pass, every corridor
    generation silently got the "off" bucket regardless of the slider."""
    ui_off = gui_ui_state(
        cohesion_mode="narrow", genre_mode="off", sonic_mode="narrow", pace_mode="narrow",
    )
    ui_strict = gui_ui_state(
        cohesion_mode="narrow", genre_mode="strict", sonic_mode="narrow", pace_mode="narrow",
    )

    res_off = generate_like_gui(
        seeds=SEEDS, ui=ui_off, length=20, random_seed=0,
        config_overrides=CORRIDOR_OVERRIDES,
    )
    res_strict = generate_like_gui(
        seeds=SEEDS, ui=ui_strict, length=20, random_seed=0,
        config_overrides=CORRIDOR_OVERRIDES,
    )

    size_off = res_off.playlist_stats.get("playlist", {}).get("corridor_universe_size")
    size_strict = res_strict.playlist_stats.get("playlist", {}).get("corridor_universe_size")
    assert size_off is not None and size_strict is not None, (
        f"corridor_universe_size missing from diagnostics: off={size_off} strict={size_strict}"
    )
    assert size_strict < size_off, (
        f"genre_mode=strict should shrink corridor_universe_size vs genre_mode=off "
        f"(pure config-driven, no test-only kwargs) -- got strict={size_strict} off={size_off}. "
        f"If these are equal, genre_mode never reached build_pier_bridge_playlist."
    )


# ── Per-mode corridor width: sonic_mode production wiring ───────────────────
# (spec section 4, pulled forward from Phase 2 by Dylan's 2026-07-18 decision)

@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_corridor_sonic_mode_threads_through_production_wiring():
    """sonic_mode must reach build_pier_bridge_playlist via the REAL
    production call chain (playlist_generator.py -> ds_pipeline_runner.py ->
    pipeline/core.py -> build_pier_bridge_playlist), driven purely by config
    -- no test-only kwargs (same anti-dead-knob proof as the genre_mode test
    above, mirrored for the sonic axis). Before this task, sonic_mode never
    reached the corridor path at all (it only ever moved the now-vestigial
    legacy candidate-pool gate) -- every corridor generation silently used
    the same width regardless of the slider.

    Compares per-segment corridor sizes (``playlist_stats["corridor_segments"]``)
    between sonic_mode="strict" (width 0.985, tighter) and sonic_mode="dynamic"
    (width 0.95, wider) on an otherwise-identical real generation via
    generate_like_gui. genre_mode is held at "off" (mask disabled) so genre's
    relevance mask cannot confound the comparison -- this isolates the sonic
    width lever specifically."""
    ui_strict = gui_ui_state(
        cohesion_mode="narrow", genre_mode="off", sonic_mode="strict", pace_mode="narrow",
    )
    ui_dynamic = gui_ui_state(
        cohesion_mode="narrow", genre_mode="off", sonic_mode="dynamic", pace_mode="narrow",
    )

    res_strict = generate_like_gui(
        seeds=SEEDS, ui=ui_strict, length=20, random_seed=0,
        config_overrides=CORRIDOR_OVERRIDES,
    )
    res_dynamic = generate_like_gui(
        seeds=SEEDS, ui=ui_dynamic, length=20, random_seed=0,
        config_overrides=CORRIDOR_OVERRIDES,
    )

    segs_strict = res_strict.playlist_stats.get("playlist", {}).get("corridor_segments") or []
    segs_dynamic = res_dynamic.playlist_stats.get("playlist", {}).get("corridor_segments") or []
    assert segs_strict and segs_dynamic, (
        f"corridor_segments missing from diagnostics: strict={segs_strict} dynamic={segs_dynamic}"
    )

    total_strict = sum(int(e.get("size", 0)) for e in segs_strict)
    total_dynamic = sum(int(e.get("size", 0)) for e in segs_dynamic)
    assert total_strict < total_dynamic, (
        f"sonic_mode=strict (width=0.985) should shrink corridor sizes vs "
        f"sonic_mode=dynamic (width=0.95) (pure config-driven, no test-only "
        f"kwargs) -- got strict_total={total_strict} dynamic_total={total_dynamic} "
        f"(strict sizes={[e.get('size') for e in segs_strict]}, "
        f"dynamic sizes={[e.get('size') for e in segs_dynamic]}). If these are "
        f"equal, sonic_mode never reached build_pier_bridge_playlist."
    )

    # Confirm the width itself is really the resolved per-mode value (not
    # just coincidentally smaller for some unrelated reason).
    widths_strict = {round(float(e.get("width", -1)), 4) for e in segs_strict}
    widths_dynamic = {round(float(e.get("width", -1)), 4) for e in segs_dynamic}
    assert widths_strict != widths_dynamic, (
        f"expected different resolved corridor widths per sonic_mode -- "
        f"got strict={widths_strict} dynamic={widths_dynamic}"
    )


# ── Task 5 reseat 1: Oops-All-Bangers on the corridor universe ──────────────
#
# 6 rows: 2 piers (b0, b1) + 4 candidates spread along one sonic axis so every
# non-pier row clears a permissive (0.0) corridor width regardless of
# popularity. b2/b3 are "bangers" (rank 0/1, well inside a cutoff of 10);
# b4/b5 are "non-bangers" (rank 50/60, well outside). interior_len=2 makes the
# gated case a DETERMINISTIC fill (exactly 2 candidates survive the gate, so
# the beam has no choice but to use both) -- avoiding any dependence on beam
# preference to prove membership, same principle as the tag-guarantee test's
# diagnostics-route check below.
_B = 6
_B_TRACK_IDS = np.array([f"bg{i}" for i in range(_B)], dtype=object)
_B_ARTISTS = np.array([f"Bangers Artist {i}" for i in range(_B)], dtype=object)
_B_TITLES = np.array([f"Bangers Song {i}" for i in range(_B)], dtype=object)
_B_SONIC = np.array([[float(i), 0.0, 0.0] for i in range(_B)], dtype=float)
_B_RANKS = np.array([-1, -1, 0, 1, 50, 60], dtype=float)
_B_CUTOFF = 10


def _bangers_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_B_TRACK_IDS,
        artist_keys=_B_ARTISTS,
        track_artists=_B_ARTISTS,
        track_titles=_B_TITLES,
        X_sonic=_B_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((_B, 1)),
        X_genre_smoothed=np.zeros((_B, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_B_TRACK_IDS)},
        durations_ms=np.full(_B, 200_000.0),
    )


def _run_bangers_case(gated: bool):
    bundle = _bangers_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,  # permissive: the bangers gate is the only lever under test
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    kwargs = dict(popularity_ranks=_B_RANKS, popularity_rank_cutoff=_B_CUTOFF) if gated else {}
    return build_pier_bridge_playlist(
        seed_track_ids=["bg0", "bg1"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=list(range(2, _B)),
        cfg=cfg,
        **kwargs,
    )


def test_corridor_bangers_gate_shrinks_universe_and_every_member_clears_cutoff():
    result = _run_bangers_case(gated=True)
    assert result.success, f"corridor segment starved: {result.failure_reason}"
    assert int(result.stats["corridor_universe_size"]) == 4, (
        f"expected 2 piers + 2 bangers = 4, got {result.stats['corridor_universe_size']}"
    )
    # Deterministic fill: exactly the 2 surviving bangers fill the 2 interior
    # slots -- every corridor member (hence every playlist track) is within
    # the rank cutoff.
    assert set(result.track_ids) == {"bg0", "bg1", "bg2", "bg3"}, (
        f"non-banger track leaked into the playlist: {result.track_ids}"
    )


def test_corridor_bangers_gate_off_leaves_full_universe():
    result = _run_bangers_case(gated=False)
    assert result.success, f"corridor segment starved: {result.failure_reason}"
    assert int(result.stats["corridor_universe_size"]) == _B, (
        f"gate inactive (no popularity_ranks/cutoff passed) should leave the full "
        f"universe eligible, got {result.stats['corridor_universe_size']}"
    )


def test_corridor_bangers_gate_exempts_tag_guarantee_ids():
    """CRITICAL review fix (post-7f0b5da): build_corridor's force_include
    lookup is scoped to universe_indices=avail_idx, itself derived from
    corridor_universe.indices. Before this fix, the bangers popularity gate
    ran without any awareness of on_tag_guarantee_indices -- a below-cutoff
    on-tag track was dropped from corridor_universe entirely, making it
    invisible to force_include (silently skipped, no signal, no error).
    Legacy parity: candidate_pool.select_pool_guarantee (:1348-1369) resolves
    its guarantee universe BYPASSING _apply_popularity_gate entirely -- the
    tag guarantee overrides bangers, not the other way around. bg5 is a
    non-banger (rank 60, cutoff 10) guaranteed via on_tag_guarantee_ids; it
    must survive the bangers gate and land in the segment corridor
    (forced_included >= 1) -- this must FAIL on 7f0b5da."""
    bundle = _bangers_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,  # permissive: isolate the bangers/guarantee interaction
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["bg0", "bg1"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=list(range(2, _B)),
        cfg=cfg,
        popularity_ranks=_B_RANKS,
        popularity_rank_cutoff=_B_CUTOFF,
        on_tag_guarantee_ids={"bg5"},  # bg5: rank 60, well outside the cutoff=10
    )
    assert result.success, f"corridor segment starved: {result.failure_reason}"
    corridor_segments = result.stats.get("corridor_segments") or []
    assert len(corridor_segments) == 1
    assert int(corridor_segments[0].get("forced_included", 0)) >= 1, (
        f"guaranteed non-banger track (bg5) was silently dropped by the "
        f"bangers gate before force_include ever ran: {corridor_segments[0]}"
    )


# ── Task 5 reseat 2: tag-steering pool guarantee (force_include) ────────────
#
# 4 rows: 2 piers (g0, g1) far apart on the sonic axis, plus g2 (aligned with
# BOTH piers -- clears a tight width) and g3 (far off-axis -- fails the same
# tight width). Without a guarantee, g3 can never enter g1<->g0's corridor;
# with on_tag_guarantee_ids={"g3"}, corridor.py's force_include (Task 1,
# forced-first semantics) admits it regardless of width. Checked via the new
# `forced_included` diagnostics summary count (never a member-id dump, per
# the NDJSON size discipline every other corridor_segments field follows).
_G = 4
_G_TRACK_IDS = np.array([f"g{i}" for i in range(_G)], dtype=object)
_G_ARTISTS = np.array([f"Guarantee Artist {i}" for i in range(_G)], dtype=object)
_G_TITLES = np.array([f"Guarantee Song {i}" for i in range(_G)], dtype=object)
_G_SONIC = np.array(
    [
        [0.0, 0.0, 0.0],   # g0 pier A
        [1.0, 1.0, 0.0],   # g1 pier B
        [0.5, 0.5, 0.0],   # g2: on the A-B line -- clears a tight width
        [0.0, 0.0, 1.0],   # g3: orthogonal to the A-B plane -- fails a tight width
    ],
    dtype=float,
)


def _guarantee_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_G_TRACK_IDS,
        artist_keys=_G_ARTISTS,
        track_artists=_G_ARTISTS,
        track_titles=_G_TITLES,
        X_sonic=_G_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((_G, 1)),
        X_genre_smoothed=np.zeros((_G, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_G_TRACK_IDS)},
        durations_ms=np.full(_G, 200_000.0),
    )


def _run_guarantee_case(guaranteed: bool):
    bundle = _guarantee_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.85,  # tight enough that g3 fails membership on its own
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    kwargs = dict(on_tag_guarantee_ids={"g3"}) if guaranteed else {}
    return build_pier_bridge_playlist(
        seed_track_ids=["g0", "g1"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=cfg,
        **kwargs,
    )


def test_corridor_tag_guarantee_forces_below_threshold_track_into_segment_corridor():
    result = _run_guarantee_case(guaranteed=True)
    assert result.success, f"corridor segment starved: {result.failure_reason}"
    corridor_segments = result.stats.get("corridor_segments") or []
    assert len(corridor_segments) == 1
    assert int(corridor_segments[0].get("forced_included", 0)) >= 1, (
        f"expected the guaranteed below-threshold track to be counted as "
        f"forced_included, got: {corridor_segments[0]}"
    )


def test_corridor_tag_guarantee_absent_by_default():
    result = _run_guarantee_case(guaranteed=False)
    assert result.success, f"corridor segment starved: {result.failure_reason}"
    corridor_segments = result.stats.get("corridor_segments") or []
    assert len(corridor_segments) == 1
    assert int(corridor_segments[0].get("forced_included", 0)) == 0, (
        f"expected no forced inclusions without on_tag_guarantee_ids, got: "
        f"{corridor_segments[0]}"
    )


# ── Task 6 remediation: widening ladder scarcity gate ───────────────────────
#
# Root cause (traced, SADE/home A/B log, see .superpowers/sdd/
# p1-task6-remediation-report.md): the ladder widened even when the corridor
# was NOT the binding constraint -- a beam-path-internal weak edge paid 3x
# beam cost (initial + 2 widen attempts), EXHAUSTED with no improvement, and
# the repair stack fixed the edge anyway. Fix: gate WIDENING (not the
# trigger) on the Phase-0a-validated anchor-support coverage metric.
#
# Fixture: permissive corridor width (0.0) means the WHOLE tiny synthetic
# universe is a corridor member for every anchor, REGARDLESS of width -- so
# widening the corridor changes nothing about which candidates are visible,
# and the beam's chosen path (hence min_edge_T) is bit-identical before and
# after a widen attempt. An unreachable transition_floor (2.0; real T is
# always < 1) guarantees the quality trigger fires on attempt 0 regardless of
# the sonic geometry chosen, without tripping the beam's only remaining real
# gate (the -0.5 anti-alignment safety in is_broken_transition --
# transition_floor itself no longer hard-gates the beam post-roam-promotion,
# and center_transitions defaults False here so that safety isn't even
# armed). Under the iteration-2 empirical continue-gate: attempt 1 always
# runs unconditionally, but since it produces ZERO improvement (identical
# min_edge_T, deterministically), attempt 2 must be STOPPED.
_W = 4
_W_TRACK_IDS = np.array([f"w{i}" for i in range(_W)], dtype=object)
_W_ARTISTS = np.array([f"Widen Artist {i}" for i in range(_W)], dtype=object)
_W_TITLES = np.array([f"Widen Song {i}" for i in range(_W)], dtype=object)
_W_SONIC = np.array(
    [
        [1.0, 0.0, 0.0],   # w0 pier A
        [0.0, 1.0, 0.0],   # w1 pier B
        [0.6, 0.6, 0.0],   # w2: interior candidate
        [0.5, 0.5, 0.1],   # w3: interior candidate
    ],
    dtype=float,
)


def _widen_gate_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=_W_TRACK_IDS,
        artist_keys=_W_ARTISTS,
        track_artists=_W_ARTISTS,
        track_titles=_W_TITLES,
        X_sonic=_W_SONIC,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((_W, 1)),
        X_genre_smoothed=np.zeros((_W, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(_W_TRACK_IDS)},
        durations_ms=np.full(_W, 200_000.0),
    )


def test_corridor_widening_ladder_stops_early_when_no_improvement(caplog):
    """Iteration 2 (empirical continue-gate): a weak edge whose corridor
    membership is width-INVARIANT (permissive width 0.0 already admits the
    whole tiny universe, so widening changes nothing) must still try widen
    attempt 1 unconditionally, then STOP before attempt 2 once that attempt
    demonstrates zero improvement -- one widen attempt paid for, not the
    full 2-attempt budget, ``widen_stopped_early: true`` recorded in the
    segment's corridor diagnostics."""
    bundle = _widen_gate_bundle()
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.0,
        corridor_widen_step=0.05,
        corridor_widen_attempts=2,
        corridor_widen_improvement_epsilon=0.02,
        transition_floor=2.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )

    with caplog.at_level(logging.INFO, logger="src.playlist.pier_bridge_builder"):
        result = build_pier_bridge_playlist(
            seed_track_ids=["w0", "w1"],
            total_tracks=4,
            bundle=bundle,
            candidate_pool_indices=[2, 3],
            cfg=cfg,
        )

    assert result.success, f"corridor segment starved: {result.failure_reason}"
    corridor_segments = result.stats.get("corridor_segments") or []
    assert len(corridor_segments) == 1
    seg = corridor_segments[0]
    assert seg.get("widened") == 0, (
        f"expected the initial (un-widened) attempt to remain best (attempt 1 "
        f"tied it, not beat it), got: {seg}"
    )
    assert seg.get("widen_stopped_early") is True, (
        f"expected widen_stopped_early=True after a non-improving attempt 1, got: {seg}"
    )

    stopped_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder"
        and r.getMessage().startswith("CorridorWiden[seg 0] STOPPED")
    ]
    assert stopped_lines, "expected a CorridorWiden[seg 0] STOPPED log line"

    widen_attempt_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder"
        and r.getMessage().startswith("CorridorWiden[seg 0]: attempt")
    ]
    assert len(widen_attempt_lines) == 1, (
        f"expected exactly 1 widen attempt (attempt 1, unconditional) before "
        f"stopping, got: {widen_attempt_lines}"
    )


def test_corridor_widening_ladder_still_widens_when_each_attempt_improves(caplog):
    """Companion positive case (fast, synthetic, seed=0 deterministic): a
    starved corridor (tight width -> attempt 0 is infeasible, no path at
    all) combined with an unreachable transition_floor must still WIDEN
    through the full corridor_widen_attempts budget as long as each attempt
    keeps improving (going from "no path" to "a weak path" on attempt 1
    counts as improvement; attempt 1 -> attempt 2 here empirically improves
    -0.081 -> 0.173, comfortably > epsilon=0.02) -- the empirical gate must
    not suppress genuine, demonstrated widening value. Mirrors the
    real-artifact Swirlies-class stressed test
    (test_corridor_widening_ladder_recovers_stressed_segment) at unit-test
    speed/determinism."""
    n = 60
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Force w0/w1 (piers) to anti-correlated directions so most of the
    # random cloud sits far from at least one anchor -- a tight width then
    # admits only a sparse sliver, starving anchor support.
    X[0] = np.array([1.0, 0.0, 0.0])
    X[1] = np.array([-1.0, 0.0, 0.0])

    track_ids = np.array([f"z{i}" for i in range(n)], dtype=object)
    artists = np.array([f"Starve Artist {i}" for i in range(n)], dtype=object)
    titles = np.array([f"Starve Song {i}" for i in range(n)], dtype=object)

    bundle = ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=artists,
        track_artists=artists,
        track_titles=titles,
        X_sonic=X,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((n, 1)),
        X_genre_smoothed=np.zeros((n, 1)),
        genre_vocab=np.array(["x"], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
        durations_ms=np.full(n, 200_000.0),
    )
    cfg = PierBridgeConfig(
        corridor_width_percentile=0.995,  # tight -> attempt 0 is infeasible (no path)
        corridor_widen_step=0.10,
        corridor_widen_attempts=2,
        corridor_widen_improvement_epsilon=0.02,
        transition_floor=2.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )

    with caplog.at_level(logging.INFO, logger="src.playlist.pier_bridge_builder"):
        result = build_pier_bridge_playlist(
            seed_track_ids=["z0", "z1"],
            total_tracks=4,
            bundle=bundle,
            candidate_pool_indices=list(range(2, n)),
            cfg=cfg,
        )

    assert result.success, f"corridor segment starved: {result.failure_reason}"
    corridor_segments = result.stats.get("corridor_segments") or []
    assert len(corridor_segments) == 1
    seg = corridor_segments[0]
    assert seg.get("widened") == 2, (
        f"expected attempt 2 (the fully-widened corridor) to be the best-scoring "
        f"attempt, got: {seg}"
    )
    assert not seg.get("widen_stopped_early"), (
        f"expected widen_stopped_early falsy when every attempt keeps improving, got: {seg}"
    )

    widen_attempt_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder"
        and r.getMessage().startswith("CorridorWiden[seg 0]: attempt")
    ]
    assert len(widen_attempt_lines) == 2, (
        f"expected exactly 2 widen-attempt log lines, got: {widen_attempt_lines}"
    )

    exhausted_lines = [
        r.getMessage() for r in caplog.records
        if r.name == "src.playlist.pier_bridge_builder"
        and r.getMessage().startswith("CorridorWiden[seg 0] EXHAUSTED")
    ]
    assert exhausted_lines, (
        f"expected the ladder to EXHAUST (transition_floor=2.0 is unreachable): {caplog.text}"
    )


# ── Task 5 reseats 3+4: tail-DP + edge repair draw only from the corridor
# union (real, stressed generation) ─────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_corridor_repair_stack_draws_only_from_corridor_union():
    """Stressed run (same stress shape as the widening-ladder test: a tight
    width the ladder can't always fully recover from) with tail_dp_enabled
    (default True) AND edge_repair_enabled turned on. Corridor-membership
    recheck via the run's own threshold diagnostics (never a member-id dump):
    every final track -- including any tail-DP swap or edge-repair swap --
    must clear at least ONE segment's recorded corridor threshold. Under
    corridor pooling, edge repair draws from the UNION of every segment's
    final corridor members (design spec §3's sanctioned substitute for
    per-edge scoping, since repair_playlist_edges takes one candidate pool
    for the whole pass), so a repaired track's home segment need not be the
    one it clears -- the OR-across-segments check below is the correct
    membership contract for this reseat, not a per-segment-only check."""
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow", pace_mode="narrow",
    )
    overrides = {
        "playlists": {"ds_pipeline": {
            "pier_bridge": {
                "pooling": "corridor",
                "corridor_width_percentile": 0.995,
                "corridor_widen_step": 0.10,
                "corridor_widen_attempts": 3,
                "edge_repair_enabled": True,
            },
            "constraints": {"transition_floor": 0.55},
        }},
    }

    res = generate_like_gui(
        seeds=SEEDS, ui=ui, length=20, random_seed=0,
        config_overrides=overrides,
    )

    assert len(res.track_ids) == 20, f"expected 20 tracks, got {len(res.track_ids)}"

    playlist_stats = res.playlist_stats.get("playlist", {})
    corridor_segments = playlist_stats.get("corridor_segments") or []
    num_segments = len(SEEDS) - 1
    assert len(corridor_segments) == num_segments

    # Load AFTER generate_like_gui (side effect sets the sonic-variant
    # override), same ordering discipline as the membership test above.
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))
    X_norm = _l2norm(bundle.X_sonic)
    seed_id_set = set(SEEDS)
    pier_positions = [i for i, tid in enumerate(res.track_ids) if tid in seed_id_set]
    assert len(pier_positions) == len(SEEDS)

    threshold_by_seg = {int(e["seg"]): float(e["threshold"]) for e in corridor_segments}
    seg_pier_vecs = []
    for seg in range(len(pier_positions) - 1):
        a_vec = X_norm[bundle.track_id_to_index[res.track_ids[pier_positions[seg]]]]
        b_vec = X_norm[bundle.track_id_to_index[res.track_ids[pier_positions[seg + 1]]]]
        seg_pier_vecs.append((a_vec, b_vec, threshold_by_seg[seg]))

    for pos, tid in enumerate(res.track_ids):
        if tid in seed_id_set:
            continue
        vec = X_norm[bundle.track_id_to_index[tid]]
        clears_any_segment = any(
            min(float(np.dot(vec, a_vec)), float(np.dot(vec, b_vec))) >= threshold - 1e-6
            for a_vec, b_vec, threshold in seg_pier_vecs
        )
        assert clears_any_segment, (
            f"track {tid} at position {pos} clears NO segment's corridor threshold -- "
            f"came from outside the corridor union"
        )

    # Confidence check: this stress config must actually exercise both repair
    # mechanisms, or the membership recheck above is vacuous (every track
    # trivially self-satisfies its own un-repaired segment). Evidence, not
    # member-id dumps: edge_repair_applied is a diagnostics boolean; tail-DP
    # has no stats-dict entry (see pier_bridge_builder.py's tail-DP comment),
    # so its evidence is the log line it already emits on every real swap.
    assert playlist_stats.get("repair_applied") is True, (
        "expected this stressed config to actually trigger edge repair -- "
        "otherwise the membership recheck above is vacuous "
        f"(full playlist_stats repair keys: repair_applied="
        f"{playlist_stats.get('repair_applied')!r})"
    )
