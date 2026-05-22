import numpy as np

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(**overrides):
    base = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=20,
        target_artists=2,
        candidates_per_artist=20,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        duration_penalty_enabled=False,
        genre_compatibility_enabled=False,
        title_hard_exclude_flags=frozenset(),
        genre_idf_enabled=False,
        pace_admission_floor=0.0,
    )
    base.update(overrides)
    return CandidatePoolConfig(**base)


def test_pace_floor_rejects_opposite_rhythm_candidate():
    N = 8
    embedding = np.ones((N, 4), dtype=float)
    X_sonic = np.zeros((N, 32), dtype=float)
    X_sonic[:, 0] = 1.0
    X_sonic[5, :8] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    artist_keys = np.array([f"a{i}" for i in range(N)])

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=_make_cfg(pace_admission_floor=0.80),
        random_seed=0,
        X_sonic=X_sonic,
        tower_pca_dims=(8, 16, 8),
    )

    assert 5 not in set(result.pool_indices)
    assert result.stats["below_pace_floor"] >= 1


def test_pace_floor_uses_max_over_seeds():
    N = 10
    embedding = np.ones((N, 4), dtype=float)
    X_sonic = np.zeros((N, 32), dtype=float)
    X_sonic[:, 0] = 1.0
    X_sonic[1, :8] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_sonic[5, :8] = X_sonic[1, :8]
    artist_keys = np.array([f"a{i}" for i in range(N)])

    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[1],
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=_make_cfg(pace_admission_floor=0.80),
        random_seed=0,
        X_sonic=X_sonic,
        tower_pca_dims=(8, 16, 8),
    )

    assert 5 in set(result.pool_indices)
