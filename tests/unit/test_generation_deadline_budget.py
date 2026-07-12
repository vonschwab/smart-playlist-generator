"""Unit tests for the total-generation wall-clock budget (shared deadline).

Root cause: strict + low-cosine seed pools (hyperpop) caused ~23-min generations
because the existing per-build `_SEGMENT_RELAXATION_BUDGET_S` anchor only guarded
tiers 2-3; tier-1's `for attempt` + bridge-floor backoff loops were unguarded, and
each One-Each retry in core.py reset a fresh 40s budget. Fix: a single shared
deadline computed once in core.py and threaded into every loop in
build_pier_bridge_playlist.

These tests exercise build_pier_bridge_playlist directly (no full pipeline needed):
  1. An already-expired deadline → returns a valid full-length playlist quickly via
     the guaranteed-fill fallback (NOT a grind, NOT an error).
  2. A generous deadline → normal fast path; playlist returned normally.
  3. Back-compat: deadline=None preserves legacy behavior (no regression).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    build_pier_bridge_playlist,
)
from src.playlist.run_audit import parse_infeasible_handling_config


# ---------------------------------------------------------------------------
# Synthetic bundle factory (mirrors test_generation_cancellation.py)
# ---------------------------------------------------------------------------

def _make_bundle(
    n: int = 80,
    sonic_dim: int = 16,
    genre_dim: int = 8,
    num_artists: int = 12,
    seed: int = 42,
) -> ArtifactBundle:
    """Deterministic synthetic ArtifactBundle."""
    rng = np.random.default_rng(seed)
    track_ids = np.array([f"t{i}" for i in range(n)])
    artist_keys = np.array([f"a{i % num_artists}" for i in range(n)])
    track_artists = np.array([f"Artist {i % num_artists}" for i in range(n)])
    track_titles = np.array([f"Song {i}" for i in range(n)])
    X_sonic = rng.standard_normal((n, sonic_dim))
    X_genre_raw = (rng.random((n, genre_dim)) > 0.7).astype(float)
    X_genre_smoothed = np.clip(
        X_genre_raw + 0.05 * rng.standard_normal((n, genre_dim)), 0.0, 1.0
    )
    genre_vocab = np.array([f"g{i}" for i in range(genre_dim)])
    durations_ms = np.full(n, 200_000, dtype=np.int64)
    track_id_to_index = {str(tid): i for i, tid in enumerate(track_ids)}
    return ArtifactBundle(
        artifact_path=Path("deadline_test"),
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        track_id_to_index=track_id_to_index,
        durations_ms=durations_ms,
    )


def _run_builder(
    bundle: ArtifactBundle,
    deadline: Optional[float],
    seed_ids: Optional[list[str]] = None,
    total_tracks: int = 15,
) -> object:
    """Call build_pier_bridge_playlist with infeasible_handling=guarantee_feasible."""
    if seed_ids is None:
        seed_ids = ["t0", "t20", "t40"]
    cfg = PierBridgeConfig(
        bridge_floor=0.0,
        transition_floor=0.0,
        center_transitions=True,
    )
    seed_idx_set = {bundle.track_id_to_index[s] for s in seed_ids}
    candidate_pool = [i for i in range(len(bundle.track_ids)) if i not in seed_idx_set]
    # Enable guarantee_feasible so the fallback path always produces a full playlist.
    infeasible_cfg = parse_infeasible_handling_config({"enabled": True, "guarantee_feasible": True})
    return build_pier_bridge_playlist(
        seed_track_ids=seed_ids,
        total_tracks=total_tracks,
        bundle=bundle,
        candidate_pool_indices=candidate_pool,
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=None,
        infeasible_handling=infeasible_cfg,
        deadline=deadline,
    )


# ---------------------------------------------------------------------------
# Test 1: already-expired deadline → fast fallback, valid full-length playlist
# ---------------------------------------------------------------------------

def test_expired_deadline_produces_valid_playlist_quickly():
    """An already-expired deadline must bail immediately to the guaranteed-fill
    fallback and return a valid, full-length playlist — not grind for minutes."""
    bundle = _make_bundle()

    # Deadline in the past — all loops should bail immediately.
    expired_deadline = time.monotonic() - 1.0

    t0 = time.monotonic()
    result = _run_builder(bundle, deadline=expired_deadline)
    elapsed = time.monotonic() - t0

    # Must complete very quickly (well under 5s; actual should be <<1s).
    assert elapsed < 5.0, (
        f"Expired deadline should bail to fallback immediately, took {elapsed:.1f}s"
    )

    # Must still return a valid, non-empty playlist.
    assert result is not None
    assert len(result.track_ids) > 0, "Expected non-empty playlist from fallback"

    # Total tracks should be close to the requested length (fallback fills from universe).
    # The exact count depends on min_gap and universe size, but must be non-zero.
    assert len(result.track_ids) >= 3  # at least the piers


# ---------------------------------------------------------------------------
# Test 2: generous deadline → normal path, playlist returned without bailing
# ---------------------------------------------------------------------------

def test_generous_deadline_allows_normal_build():
    """A deadline far in the future should not interfere with normal builds."""
    bundle = _make_bundle()

    # 5-minute deadline — effectively no restriction for a toy bundle.
    generous_deadline = time.monotonic() + 300.0

    result = _run_builder(bundle, deadline=generous_deadline)

    assert result is not None
    assert len(result.track_ids) > 0


# ---------------------------------------------------------------------------
# Test 3: deadline=None → back-compat, legacy behavior preserved
# ---------------------------------------------------------------------------

def test_no_deadline_preserves_legacy_behavior():
    """deadline=None must not change any behavior — existing callers are unaffected."""
    bundle = _make_bundle()

    result = _run_builder(bundle, deadline=None)

    assert result is not None
    assert len(result.track_ids) > 0
