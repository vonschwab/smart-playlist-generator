# tests/unit/test_var_bridge_add_only.py
"""Variable-bridge length must be ADD-only (Task 3, 2026-07-02 cascade reorder
plan): a segment's chosen interior length may only be >= its nominal
even-split length, never shorter. Shortening a segment (deleting a track) is
now the exclusive job of the remove-only edge_delete pass that runs later,
after break-glass repair.

Drives the real ``build_pier_bridge_playlist`` entry point directly (same
pattern as tests/unit/test_edge_delete.py's outlier-bundle test and
tests/unit/test_builder_pair_floor_wiring.py) with a hand-built two-pier
ArtifactBundle -- NOT a single-seed arc (num_seeds=1 hits the special-cased
``is_single_seed_arc`` path). This is the harness pattern the multi-pier
tests in this repo use when the live artifact isn't required (see
tests/unit/test_edge_delete.py's comment for why gui_fidelity isn't used
here: engineering a deterministic length-flex trigger against the live
artifact's real sonic vectors is not reliable).

Fixture geometry: pier A (0deg) and pier B (20deg) are a CLOSE pier pair
bridged by a candidate pool of exactly nominal-length (4) interior tracks --
three "good" tracks evenly spaced between the piers (5/10/15deg) and one bad
OUTLIER (200deg). Because the pool has exactly 4 usable candidates, the
nominal-length (4) build is FORCED to include the outlier, producing a badly
broken bottleneck edge (~-0.99, far below the 0.30 variable_bridge_min_edge
"good enough" threshold) that triggers the length-flex evaluation. Under the
pre-Task-3 bidirectional selector this reliably SHRINKS the segment (to 2 or
3, excluding the outlier, bottleneck ~0.98-0.99) -- the genuine RED this test
pins. Under the Task-3 add-only selector, shrinking is no longer an option
(lo == nominal): lengths above nominal have no additional usable candidates
in the pool (len(pool) < interior_len at 5/6), so the selector must fall back
to the forced nominal(4) build every time -- chosen == nominal, never below.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist

_VAR_BRIDGE_LOG_RE = re.compile(r"Var-bridge seg (\d+): nominal=(\d+) chosen=(\d+)")


def _vec(deg: float) -> list[float]:
    r = math.radians(deg)
    return [math.cos(r), math.sin(r)]


def _close_pier_pair_with_outlier_bundle() -> ArtifactBundle:
    # t0 = pier A (0deg), t1/t2/t3 = good interior chain (5/10/15deg), t4 = bad
    # outlier (200deg), t5 = pier B (20deg). Pool = exactly 4 candidates
    # (t1..t4) == nominal interior length, so the outlier is UNAVOIDABLE at
    # the nominal length but excludable at any shorter length.
    degs = [0, 5, 10, 15, 200, 20]
    X = np.array([_vec(d) for d in degs], dtype=float)
    n = X.shape[0]
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X,
        X_sonic_start=X,
        X_sonic_mid=X,
        X_sonic_end=X,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _build_with_var_bridge(caplog):
    cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        center_transitions=False,
        collapse_segment_pool_by_artist=False,
        genre_steering_enabled=False,
        edge_repair_enabled=False,
        edge_delete_enabled=False,
        variable_bridge_length=True,
    )
    bundle = _close_pier_pair_with_outlier_bundle()
    with caplog.at_level(logging.INFO, logger="src.playlist.pier_bridge_builder"):
        result = build_pier_bridge_playlist(
            seed_track_ids=["t0", "t5"],
            total_tracks=6,  # 2 piers + 4 interior == nominal segment length 4
            bundle=bundle,
            candidate_pool_indices=[1, 2, 3, 4],
            cfg=cfg,
            X_genre_smoothed=bundle.X_genre_smoothed,
        )
    return result, caplog


def _parse_var_bridge_lines(caplog) -> list[tuple[int, int, int]]:
    hits = []
    for record in caplog.records:
        m = _VAR_BRIDGE_LOG_RE.search(record.getMessage())
        if m:
            hits.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return hits


def test_variable_bridge_never_shrinks_below_nominal(caplog):
    result, caplog = _build_with_var_bridge(caplog)
    assert result.success

    hits = _parse_var_bridge_lines(caplog)
    assert hits, "expected at least one 'Var-bridge seg ...' log line"
    for seg_idx, nominal, chosen in hits:
        assert chosen >= nominal, (
            f"segment {seg_idx} shrank below nominal: nominal={nominal} chosen={chosen} "
            "(variable bridge must be ADD-only per the 2026-07-02 cascade reorder)"
        )
