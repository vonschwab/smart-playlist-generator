"""Pace-gate perceptual-BPM fallback for no-tower variants (MERT plan Phase 5, item 4).

With a no-tower sonic variant the builder cannot derive ``rhythm_matrix`` from
``tower_dims``, so a configured ``pace_bridge_floor > 0`` must fall back to the
perceptual-BPM gate (``bpm_bridge_max_log_distance`` machinery) with exactly
one warning — never a silent no-op.

Fixture mirrors tests/unit/test_builder_pace_gate_wiring.py: the off-pace
candidate (t2) is engineered to win on full-vector similarity, so it is chosen
when no gate runs and rejected when the BPM fallback gate is live.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist

_BUILDER_LOGGER = "src.playlist.pier_bridge_builder"


def _bundle(X_sonic: np.ndarray) -> ArtifactBundle:
    n = int(X_sonic.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_sonic,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
        sonic_variant="mert",
        sonic_pre_scaled=True,
        tower_dims=None,  # no-tower variant: rhythm axis cannot be sliced
    )


# t2 is identical to the piers (wins on full-vector similarity); t1 is the
# slightly-off but on-BPM alternative.
_X = np.array(
    [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # t0: pier A
        [1.0, 0.0, 0.6, 0.8, 0.0, 1.0],  # t1: weaker similarity, on-BPM
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # t2: best similarity, off-BPM
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # t3: pier B
    ],
    dtype=float,
)

# t2 at double tempo: |log2(240/120)| = 1.0, beyond any finite fallback cap.
_BPM = np.array([120.0, 120.0, 240.0, 120.0], dtype=float)


def _build(*, pace_bridge_floor: float, perceptual_bpm):
    bundle = _bundle(_X)
    cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        pace_bridge_floor=pace_bridge_floor,
        progress_enabled=False,
        center_transitions=False,
        collapse_segment_pool_by_artist=False,
    )
    return build_pier_bridge_playlist(
        seed_track_ids=["t0", "t3"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[1, 2],
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=bundle.X_genre_smoothed,
        perceptual_bpm=perceptual_bpm,
    )


def _pace_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Pace bridge gate" in r.getMessage()
    ]


def test_off_bpm_candidate_wins_when_gate_disabled(caplog):
    """Baseline: floor 0 → no pace gating, t2 wins on similarity, no warning."""
    with caplog.at_level(logging.WARNING, logger=_BUILDER_LOGGER):
        result = _build(pace_bridge_floor=0.0, perceptual_bpm=_BPM)
    assert result.success
    assert result.track_ids == ["t0", "t2", "t3"]
    assert _pace_warnings(caplog) == []


def test_bpm_fallback_gates_when_rhythm_dims_unavailable(caplog):
    """No tower dims + pace_bridge_floor > 0 → the perceptual-BPM gate must
    actually gate (t2 rejected at double tempo) and exactly one warning logs."""
    with caplog.at_level(logging.WARNING, logger=_BUILDER_LOGGER):
        result = _build(pace_bridge_floor=0.5, perceptual_bpm=_BPM)
    assert result.success
    assert result.track_ids == ["t0", "t1", "t3"]
    warnings = _pace_warnings(caplog)
    assert len(warnings) == 1, f"expected exactly one pace warning, got: {warnings}"
    assert "BPM" in warnings[0]


def test_no_bpm_data_warns_loudly_instead_of_silent_noop(caplog):
    """Without BPM data the fallback cannot run either — the builder must say
    so loudly (configured gate with no data is a loud warning, not silence)."""
    with caplog.at_level(logging.WARNING, logger=_BUILDER_LOGGER):
        result = _build(pace_bridge_floor=0.5, perceptual_bpm=None)
    assert result.success
    assert result.track_ids == ["t0", "t2", "t3"]  # nothing gated
    warnings = _pace_warnings(caplog)
    assert len(warnings) == 1
    assert "INACTIVE" in warnings[0]
