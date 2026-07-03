"""Pool lever: steering target blends into the dense genre-admission centroid."""
import numpy as np

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _pool(steering_target=None, steering_blend=0.5):
    n = 5  # 0 = seed; 1,2 aligned with seed genre; 3,4 aligned with target genre
    embedding = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))  # sonically identical
    embedding += np.arange(n).reshape(-1, 1) * 1e-6         # break exact ties
    artist_keys = np.array([f"artist{i}" for i in range(n)])
    x_genre_dense = np.array([
        [1.0, 0.0],   # seed
        [1.0, 0.0],   # near-seed genre
        [0.9, 0.1],
        [0.0, 1.0],   # on-target genre
        [0.1, 0.9],
    ])
    x_genre_dense /= np.linalg.norm(x_genre_dense, axis=1, keepdims=True)
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=3,
        seed_artist_bonus=1,
        max_artist_fraction_final=1.0,
        duration_penalty_enabled=False,
        title_exclusion_enabled=False,
    )
    return build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_dense=x_genre_dense,
        genre_admission_percentile=0.5,   # floor at the median non-seed sim
        mode="dynamic",
        steering_target=steering_target,
        steering_blend=steering_blend,
    )


def test_unsteered_pool_prefers_seed_genre_neighbors():
    res = _pool(steering_target=None)
    assert {1, 2} <= set(res.pool_indices.tolist())
    assert not {3, 4} <= set(res.pool_indices.tolist())


def test_full_blend_flips_admission_toward_target():
    res = _pool(steering_target=np.array([0.0, 1.0]), steering_blend=1.0)
    assert {3, 4} <= set(res.pool_indices.tolist())
    assert not {1, 2} <= set(res.pool_indices.tolist())


def test_zero_blend_equals_unsteered():
    base = _pool(steering_target=None)
    zero = _pool(steering_target=np.array([0.0, 1.0]), steering_blend=0.0)
    assert base.pool_indices.tolist() == zero.pool_indices.tolist()
