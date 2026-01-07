import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.segment_pool_builder import SegmentCandidatePoolBuilder, SegmentPoolConfig


def _bundle_for_pool_test() -> ArtifactBundle:
    track_ids = np.array(["t0", "t1", "t2", "t3"])
    artist_keys = np.array(["a0", "a1", "a2", "a3"])
    track_artists = np.array(["A0", "A1", "A2", "A3"])
    track_titles = np.array(["S0", "S1", "S2", "S3"])

    X_full = np.array(
        [
            [1.0, 0.0],  # pier A
            [0.0, 1.0],  # pier B
            [0.9, 0.1],  # local to A
            [0.1, 0.9],  # local to B / genre target
        ],
        dtype=float,
    )
    X_full = X_full / (np.linalg.norm(X_full, axis=1, keepdims=True) + 1e-12)
    X_genre = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.1],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    X_genre = X_genre / (np.linalg.norm(X_genre, axis=1, keepdims=True) + 1e-12)
    genre_vocab = np.array(["g0", "g1"])

    return ArtifactBundle(
        artifact_path=None,  # type: ignore[arg-type]
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_full,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        genre_vocab=genre_vocab,
        track_id_to_index={tid: idx for idx, tid in enumerate(track_ids)},
    )


def test_segment_pool_baseline_unchanged_when_strategy_segment_scored():
    bundle = _bundle_for_pool_test()
    base_cfg = SegmentPoolConfig(
        pier_a=0,
        pier_b=1,
        X_full_norm=bundle.X_sonic,
        universe_indices=[0, 1, 2, 3],
        used_track_ids=set(),
        bundle=bundle,
        bridge_floor=0.0,
        segment_pool_max=10,
        pool_strategy="segment_scored",
    )
    cfg_with_pooling = SegmentPoolConfig(
        **{**base_cfg.__dict__, "pool_k_local": 2, "pool_k_genre": 2},
    )
    base = SegmentCandidatePoolBuilder().build(base_cfg)
    with_pooling = SegmentCandidatePoolBuilder().build(cfg_with_pooling)
    assert base.candidates == with_pooling.candidates


def test_dj_union_pool_includes_local_and_genre_sources():
    bundle = _bundle_for_pool_test()
    cfg = SegmentPoolConfig(
        pier_a=0,
        pier_b=1,
        X_full_norm=bundle.X_sonic,
        universe_indices=[2, 3],
        used_track_ids=set(),
        bundle=bundle,
        bridge_floor=0.0,
        segment_pool_max=10,
        pool_strategy="dj_union",
        pool_k_local=1,
        pool_k_toward=0,
        pool_k_genre=1,
        pool_k_union_max=10,
        interior_length=1,
        X_genre_norm=bundle.X_genre_smoothed,
        genre_targets=[bundle.X_genre_smoothed[1]],
    )
    result = SegmentCandidatePoolBuilder().build(cfg)
    assert 2 in result.candidates
    assert 3 in result.candidates
    assert result.diagnostics.get("dj_pool_strategy") == "dj_union"
