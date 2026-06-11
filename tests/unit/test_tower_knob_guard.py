"""Tower-knob guard for no-tower sonic variants (MERT plan Phase 5, item 3).

A no-tower variant (e.g. ``mert``) is a single space: ``transition_weights`` /
``tower_weights`` cannot act on it. Per the configured-knob-must-act rule:

- default weights (0.20/0.50/0.30) or no weights + no-tower variant
  → INFO log that the tower knobs are inert, no raise;
- NON-default weights + no-tower variant → startup error (raise);
- non-default weights + tower_weighted variant → unchanged current behavior;
- legacy artifacts with no variant declared are out of scope (unchanged).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from src.features.artifacts import (
    ArtifactBundle,
    DEFAULT_TOWER_TRANSITION_WEIGHTS,
    validate_tower_knobs,
)

NON_DEFAULT_WEIGHTS = (0.40, 0.35, 0.25)


def _bundle(
    *,
    dim: int,
    tower_dims: Optional[Tuple[int, int, int]],
    sonic_variant: Optional[str],
    sonic_pre_scaled: bool,
) -> ArtifactBundle:
    n = 3
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=None,
        track_titles=None,
        X_sonic=np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32),
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.ones((n, 2), np.float32),
        X_genre_smoothed=np.ones((n, 2), np.float32),
        genre_vocab=np.array(["x", "y"], dtype=object),
        track_id_to_index={str(t): i for i, t in enumerate(track_ids)},
        sonic_variant=sonic_variant,
        sonic_pre_scaled=sonic_pre_scaled,
        tower_dims=tower_dims,
    )


def _mert_bundle() -> ArtifactBundle:
    return _bundle(dim=8, tower_dims=None, sonic_variant="mert", sonic_pre_scaled=True)


def test_default_weights_with_no_tower_variant_logs_info_and_passes(caplog):
    with caplog.at_level(logging.INFO, logger="src.features.artifacts"):
        validate_tower_knobs(_mert_bundle(), DEFAULT_TOWER_TRANSITION_WEIGHTS)
    inert_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "inert" in r.getMessage()
    ]
    assert inert_msgs, "expected an INFO log saying tower knobs are inert for this variant"
    assert any("mert" in m for m in inert_msgs)


def test_unset_weights_with_no_tower_variant_logs_info_and_passes(caplog):
    with caplog.at_level(logging.INFO, logger="src.features.artifacts"):
        validate_tower_knobs(_mert_bundle(), None)
    assert any(
        r.levelno == logging.INFO and "inert" in r.getMessage() for r in caplog.records
    )


def test_non_default_weights_with_no_tower_variant_raises():
    with pytest.raises(ValueError, match="transition_weights"):
        validate_tower_knobs(_mert_bundle(), NON_DEFAULT_WEIGHTS)


def test_non_default_weights_with_tower_variant_is_fine():
    bundle = _bundle(
        dim=6, tower_dims=(2, 2, 2), sonic_variant="tower_weighted", sonic_pre_scaled=True
    )
    validate_tower_knobs(bundle, NON_DEFAULT_WEIGHTS)  # must not raise


def test_legacy_artifact_without_variant_is_unchanged():
    """No variant declared (legacy/raw artifact): guard must not fire even with
    non-default weights — that path keeps its current behavior (the golden
    synthetic artifacts run exactly this combination)."""
    bundle = _bundle(dim=32, tower_dims=None, sonic_variant=None, sonic_pre_scaled=False)
    validate_tower_knobs(bundle, NON_DEFAULT_WEIGHTS)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline hook: the guard must run in the live generate_playlist_ds path
# ─────────────────────────────────────────────────────────────────────────────

def _write_mert_artifact(path) -> None:
    """Tiny synthetic artifact with a declared no-tower variant (mert)."""
    n = 8
    # Sonic ramp keeps every track reachable from the seed (mirrors the
    # layered ds-pipeline smoke fixture).
    X_mert = np.array(
        [[1.0 - 0.05 * i, 0.05 * i, 0.0, 0.0] for i in range(n)], dtype=np.float32
    )
    np.savez(
        path,
        track_ids=np.array([f"t{i}" for i in range(n)], dtype=object),
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=np.zeros((n, 4), np.float32),
        X_sonic_mert=X_mert,
        X_sonic_variant=np.array("mert"),
        X_genre_raw=np.ones((n, 2), np.float32),
        X_genre_smoothed=np.ones((n, 2), np.float32),
        genre_vocab=np.array(["x", "y"], dtype=object),
    )


def _run_pipeline(artifact_path, transition_weights):
    from src.features.artifacts import load_artifact_bundle
    from src.playlist.pier_bridge_builder import PierBridgeConfig
    from src.playlist.pipeline import generate_playlist_ds

    load_artifact_bundle.cache_clear()
    pb_cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        initial_neighbors_m=10,
        initial_bridge_helpers=5,
        max_neighbors_m=10,
        max_bridge_helpers=5,
        initial_beam_width=4,
        max_beam_width=4,
        max_expansion_attempts=1,
        segment_pool_strategy="segment_scored",
        segment_pool_max=8,
        max_segment_pool_max=8,
        progress_enabled=False,
        edge_repair_enabled=False,
        genre_steering_enabled=False,
        weight_genre=0.0,
        dj_bridging_enabled=False,
    )
    return generate_playlist_ds(
        artifact_path=str(artifact_path),
        seed_track_id="t0",
        num_tracks=4,
        mode="dynamic",
        pace_mode="off",
        random_seed=0,
        overrides={
            "transition_weights": transition_weights,
            "candidate": {
                "similarity_floor": -1.0,
                "min_sonic_similarity": None,
                "max_pool_size": 8,
                "target_artists": 4,
                "candidates_per_artist": 2,
                "seed_artist_bonus": 0,
                "title_hard_exclude_flags": [],
            },
            "pier_bridge": {
                "audit_run": {"enabled": False},
                "infeasible_handling": {"enabled": False},
            },
        },
        pier_bridge_config=pb_cfg,
        dry_run=True,
    )


def test_pipeline_raises_on_non_default_weights_with_mert_variant(tmp_path):
    artifact = tmp_path / "mert-artifact.npz"
    _write_mert_artifact(artifact)
    with pytest.raises(ValueError, match="transition_weights"):
        _run_pipeline(artifact, {"rhythm": 0.40, "timbre": 0.35, "harmony": 0.25})


def test_pipeline_accepts_default_weights_with_mert_variant(tmp_path):
    artifact = tmp_path / "mert-artifact-ok.npz"
    _write_mert_artifact(artifact)
    result = _run_pipeline(
        artifact, {"rhythm": 0.20, "timbre": 0.50, "harmony": 0.30}
    )
    assert len(result.track_ids) == 4
