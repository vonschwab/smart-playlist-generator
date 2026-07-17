"""Phase 1 Task 3: corridor segment pooling behind the dev flag.

Spec: docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.
Report: .superpowers/sdd/p1-task-3-report.md.

Two tiers:
  * `test_corridor_pooling_*` -- real end-to-end generation via the
    gui_fidelity harness (@integration @slow, needs the live artifact).
    Pins the Step-1 contract: corridor flag on, multi-pier generation
    completes, every non-pier playlist track is a member of its segment's
    corridor (min_sim >= threshold, recomputed independently here), and the
    per-segment health line (contract F7) appears exactly once per segment.
  * `test_corridor_universe_duration_reference_is_none_*` -- fast unit-level
    wiring check (synthetic ArtifactBundle, no artifact needed): the
    dead-knob-trap regression from the Task 3 review. Phase 1 deliberately
    never wires a real duration soft-penalty reference into the corridor
    universe; this pins that off-state so a future change can't start
    silently acting on an unconfigured knob.
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
    corridor; the health line appears once per segment."""
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

    assert len(res.track_ids) == 20, f"expected 20 tracks, got {len(res.track_ids)}"

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
        pooling="corridor",
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
        pooling="corridor",
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
