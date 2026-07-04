"""Genre-rescue: re-admit the top-K sonic-nearest tracks rejected ONLY by the
genre hard gate (Fix 3, 2026-07-04).

The slider-differentiation eval showed tight genre gating craters the worst
live edge on niche seeds (Codeine minT 0.030-0.079): the gate strips the sonic
connectors the beam needs. Additive rescue mirrors the energy admission-rescue
pattern — never removes, only re-admits. The lever for keeping a great sonic
neighbor is sonic admission, not the genre metric (Embassy lesson).
"""
import numpy as np

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(**kwargs):
    base = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        duration_penalty_enabled=False,
        duration_penalty_weight=0.0,
        duration_cutoff_multiplier=2.5,
        genre_compatibility_enabled=False,
        genre_compatibility_penalty_strength=0.0,
        genre_compatibility_compatible_threshold=0.35,
        genre_compatibility_conflict_threshold=0.15,
        title_hard_exclude_flags=frozenset(),
        genre_idf_enabled=False,
    )
    base.update(kwargs)
    return CandidatePoolConfig(**base)


def _fixture():
    """idx0 seed. idx1 genre-off + sonic-near (the connector). idx2 genre-off +
    sonic-far. idx3/idx4 genre-match."""
    N = 5
    embedding = np.eye(N, 8) * 0.2 + 0.8  # all mutually similar in hybrid space
    artist_keys = np.array([f"a{i}" for i in range(N)])
    X_sonic = np.zeros((N, 2))
    X_sonic[0] = [1.0, 0.0]                       # seed
    X_sonic[1] = [0.95, np.sqrt(1 - 0.95**2)]     # sonic-near, genre-off
    X_sonic[2] = [0.10, np.sqrt(1 - 0.10**2)]     # sonic-far, genre-off
    X_sonic[3] = [0.60, np.sqrt(1 - 0.60**2)]
    X_sonic[4] = [0.55, np.sqrt(1 - 0.55**2)]
    genre_vocab = ["slowcore", "jazz"]
    X_genre_raw = np.array([
        [1.0, 0.0],  # seed
        [0.0, 1.0],  # genre-off
        [0.0, 1.0],  # genre-off
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    return embedding, artist_keys, X_sonic, X_genre_raw, genre_vocab


def _pool(k_rescue: int):
    embedding, artist_keys, X_sonic, X_genre_raw, genre_vocab = _fixture()
    cfg = _make_cfg(genre_rescue_k=k_rescue)
    return build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_raw,
        genre_vocab=genre_vocab,
        min_genre_similarity=0.5,
        mode="dynamic",
    )


def test_rescue_readmits_sonic_nearest_genre_reject():
    result = _pool(k_rescue=1)
    pool = set(result.pool_indices.tolist())
    assert 1 in pool          # sonic-near connector rescued
    assert 2 not in pool      # sonic-far genre-reject NOT rescued (k=1)
    assert 3 in pool and 4 in pool
    assert result.stats.get("genre_rescue_admitted") == 1


def test_rescue_off_preserves_hard_gate():
    result = _pool(k_rescue=0)
    pool = set(result.pool_indices.tolist())
    assert 1 not in pool and 2 not in pool
    assert result.stats.get("genre_rescue_admitted", 0) == 0
