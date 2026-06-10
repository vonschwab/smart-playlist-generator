"""
Per-segment candidate pool tests — Layer 1 + Layer 2.

Layer 1  genre_admission_aggregate in build_candidate_pool
         "per_seed"  = union of per-seed genre neighborhoods (new)
         "centroid"  = legacy centroid average (unchanged default)

Layer 2  genre_bridge_weight in SegmentPoolConfig
         blends genre score into bridge re-ranking (new, re-rank only, no hard gate)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.artifacts import ArtifactBundle
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig
from src.playlist.segment_pool_builder import SegmentCandidatePoolBuilder, SegmentPoolConfig


# ── shared helpers ─────────────────────────────────────────────────────────────

def _pool_cfg(**overrides):
    """Minimal CandidatePoolConfig: all non-genre gates disabled."""
    base = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=50,
        target_artists=20,
        candidates_per_artist=10,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
    )
    base.update(overrides)
    return CandidatePoolConfig(**base)


def _make_bundle(n: int, *, rng_seed: int = 0) -> ArtifactBundle:
    """Minimal ArtifactBundle with n tracks (no audio; identity sonic matrix)."""
    rng = np.random.default_rng(rng_seed)
    X_sonic = rng.standard_normal((n, 4))
    X_sonic /= np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12
    X_genre = np.zeros((n, 2), dtype=np.float64)
    genre_vocab = np.array(["g0", "g1"])
    track_ids = np.array([f"t{i}" for i in range(n)])
    artist_keys = np.array([f"a{i}" for i in range(n)])
    return ArtifactBundle(
        artifact_path=None,  # type: ignore[arg-type]
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=None,
        track_titles=None,
        X_sonic=X_sonic,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        genre_vocab=genre_vocab,
        track_id_to_index={tid: i for i, tid in enumerate(track_ids)},
    )


# ── Layer 1 fixture ────────────────────────────────────────────────────────────

def _indie_rock_setup():
    """
    2D L2-normalized genre embedding. Five indie seeds, one rock seed.

    Tracks 0–4:  indie seeds   [1, 0]
    Track  5:    rock seed     [0, 1]
    Tracks 6–10: indie non-seed  [1, 0]
    Track  11:   rock non-seed   [0, 1]
    Track  12:   outlier         [-1, 0]
    """
    indie = np.array([1.0, 0.0])
    rock  = np.array([0.0, 1.0])
    out   = np.array([-1.0, 0.0])

    X = np.array(
        [indie] * 5 + [rock] + [indie] * 5 + [rock] + [out],
        dtype=np.float64,
    )  # shape (13, 2), already L2-normalised (unit vectors)

    # Flat embedding — all tracks equally similar in the hybrid space so the
    # similarity_floor gate doesn't remove anything.
    emb = np.ones((13, 2), dtype=np.float64)

    artist_keys = np.array([f"a{i}" for i in range(13)])

    return X, 0, [1, 2, 3, 4, 5], emb, artist_keys


# ── Layer 1 tests ──────────────────────────────────────────────────────────────

class TestGenreAdmissionAggregate:

    def test_centroid_rejects_rock_track_baseline(self):
        """Centroid mode: 5:1 indie weighting → rock track (idx 11) should be rejected."""
        X, seed_idx, seed_indices, emb, artist_keys = _indie_rock_setup()

        result = build_candidate_pool(
            seed_idx=seed_idx,
            seed_indices=seed_indices,
            embedding=emb,
            artist_keys=artist_keys,
            cfg=_pool_cfg(),
            random_seed=0,
            X_genre_dense=X,
            genre_admission_percentile=0.5,
            genre_admission_aggregate="centroid",
            mode="dynamic",
        )

        admitted = {int(i) for i in result.pool_indices}
        assert 11 not in admitted, (
            "centroid mode must reject rock track (idx 11) when 5:1 indie weighting"
            " drives centroid floor above 0.197"
        )
        for i in range(6, 11):
            assert i in admitted, f"centroid mode must still admit indie track {i}"

    def test_per_seed_admits_rock_track(self):
        """per_seed mode: rock track admitted via its nearest seed (union semantics)."""
        X, seed_idx, seed_indices, emb, artist_keys = _indie_rock_setup()

        result = build_candidate_pool(
            seed_idx=seed_idx,
            seed_indices=seed_indices,
            embedding=emb,
            artist_keys=artist_keys,
            cfg=_pool_cfg(),
            random_seed=0,
            X_genre_dense=X,
            genre_admission_percentile=0.5,
            genre_admission_aggregate="per_seed",
            mode="dynamic",
        )

        admitted = {int(i) for i in result.pool_indices}
        assert 11 in admitted, (
            "per_seed mode must admit rock track (idx 11): it is in the top-50% of"
            " the rock seed's genre neighborhood even though centroid rejects it"
        )
        for i in range(6, 11):
            assert i in admitted, f"per_seed mode must also admit indie tracks"

    def test_default_aggregate_matches_explicit_centroid(self):
        """Omitting genre_admission_aggregate produces identical results to 'centroid'."""
        X, seed_idx, seed_indices, emb, artist_keys = _indie_rock_setup()

        kwargs = dict(
            seed_idx=seed_idx,
            seed_indices=seed_indices,
            embedding=emb,
            artist_keys=artist_keys,
            cfg=_pool_cfg(),
            random_seed=0,
            X_genre_dense=X,
            genre_admission_percentile=0.5,
            mode="dynamic",
        )

        result_default  = build_candidate_pool(**kwargs)
        result_explicit = build_candidate_pool(**kwargs, genre_admission_aggregate="centroid")

        assert set(result_default.pool_indices) == set(result_explicit.pool_indices), (
            "default (no genre_admission_aggregate) must be identical to explicit centroid"
        )

    def test_per_seed_fixed_floor_admits_rock_track(self):
        """per_seed with a fixed min_genre_similarity (no percentile): union of per-seed max."""
        X, seed_idx, seed_indices, emb, artist_keys = _indie_rock_setup()

        # Rock track's max cosine to any seed = 1.0 (matches rock seed exactly) ≥ 0.5
        # Centroid cosine would be ~0.197 → rejected.
        result = build_candidate_pool(
            seed_idx=seed_idx,
            seed_indices=seed_indices,
            embedding=emb,
            artist_keys=artist_keys,
            cfg=_pool_cfg(),
            random_seed=0,
            X_genre_dense=X,
            min_genre_similarity=0.5,
            genre_admission_aggregate="per_seed",
            mode="dynamic",
        )

        admitted = {int(i) for i in result.pool_indices}
        assert 11 in admitted, (
            "per_seed with fixed floor must admit rock track: max(cosines) = 1.0 ≥ 0.5"
        )

    def test_centroid_fixed_floor_rejects_rock_track(self):
        """Regression: centroid+fixed floor = 0.5 still rejects rock track (~0.197 centroid sim)."""
        X, seed_idx, seed_indices, emb, artist_keys = _indie_rock_setup()

        result = build_candidate_pool(
            seed_idx=seed_idx,
            seed_indices=seed_indices,
            embedding=emb,
            artist_keys=artist_keys,
            cfg=_pool_cfg(),
            random_seed=0,
            X_genre_dense=X,
            min_genre_similarity=0.5,
            genre_admission_aggregate="centroid",
            mode="dynamic",
        )

        admitted = {int(i) for i in result.pool_indices}
        assert 11 not in admitted, (
            "centroid mode with fixed floor 0.5 must reject rock track (centroid cos ≈ 0.197)"
        )


# ── Layer 2 helpers ────────────────────────────────────────────────────────────

def _segment_setup():
    """
    4 tracks. Pier 0 is indie [1,0], pier 1 is rock [0,1].
    Candidate 2: genre between both piers → high genre_hmean.
    Candidate 3: genre only matches pier 0 → lower genre_hmean.
    All tracks identical in sonic space → equal sonic bridge scores.
    """
    X_sonic = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=np.float64,
    )  # identical sonic → identical hmean(simA, simB)

    v = 1.0 / np.sqrt(2)
    X_genre = np.array(
        [
            [1.0, 0.0],  # pier 0 (indie)
            [0.0, 1.0],  # pier 1 (rock)
            [v,   v  ],  # candidate 2: genre bridge (equal to both)
            [1.0, 0.0],  # candidate 3: genre matches only pier 0
        ],
        dtype=np.float64,
    )

    bundle = _make_bundle(4)

    return X_sonic, X_genre, bundle


# ── Layer 2 tests ──────────────────────────────────────────────────────────────

class TestSegmentPoolGenreBlend:

    def test_genre_blend_promotes_bridge_candidate(self):
        """genre_bridge_weight > 0 must rank the genre-bridge candidate above the one-sided match."""
        X_sonic, X_genre, bundle = _segment_setup()

        cfg = SegmentPoolConfig(
            pier_a=0,
            pier_b=1,
            X_full_norm=X_sonic,
            universe_indices=[2, 3],
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=4,
            X_genre_dense=X_genre,
            genre_bridge_weight=0.5,
        )
        result = SegmentCandidatePoolBuilder().build(cfg)

        # Both candidates must be present
        assert 2 in result.candidates
        assert 3 in result.candidates
        idx2 = result.candidates.index(2)
        idx3 = result.candidates.index(3)
        assert idx2 < idx3, (
            "genre_bridge_weight=0.5 must rank genre-bridge candidate (2) "
            "before one-sided candidate (3)"
        )

    def test_genre_weight_zero_ordering_unchanged(self):
        """genre_bridge_weight=0.0 must produce byte-identical ordering to no X_genre_dense."""
        X_sonic, X_genre, bundle = _segment_setup()

        base_kwargs: dict = dict(
            pier_a=0,
            pier_b=1,
            X_full_norm=X_sonic,
            universe_indices=[2, 3],
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=4,
        )

        result_no_genre = SegmentCandidatePoolBuilder().build(
            SegmentPoolConfig(**base_kwargs, X_genre_dense=None, genre_bridge_weight=0.0)
        )
        result_zero_weight = SegmentCandidatePoolBuilder().build(
            SegmentPoolConfig(**base_kwargs, X_genre_dense=X_genre, genre_bridge_weight=0.0)
        )

        assert result_no_genre.candidates == result_zero_weight.candidates, (
            "genre_bridge_weight=0.0 must not change candidate ordering"
        )

    def test_genre_blend_requires_x_genre_dense(self):
        """genre_bridge_weight > 0 with X_genre_dense=None must not raise and falls back to sonic."""
        X_sonic, _, bundle = _segment_setup()

        cfg = SegmentPoolConfig(
            pier_a=0,
            pier_b=1,
            X_full_norm=X_sonic,
            universe_indices=[2, 3],
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=4,
            X_genre_dense=None,
            genre_bridge_weight=0.5,  # non-zero but no matrix → fallback silently
        )

        result = SegmentCandidatePoolBuilder().build(cfg)
        assert len(result.candidates) == 2, "fallback must still return both candidates"
