import numpy as np
import pytest
from src.playlist.candidate_pool import build_candidate_pool, CandidatePoolConfig


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
        pace_admission_floor=0.0,
        pace_bridge_floor=0.0,
        bpm_admission_max_log_distance=float("inf"),
        bpm_stability_min=0.5,
    )
    base.update(kwargs)
    return CandidatePoolConfig(**base)


def test_bpm_floor_inf_admits_all():
    rng = np.random.default_rng(0)
    N = 5
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    perceptual_bpm = np.array([60.0, 120.0, 240.0, 70.0, 180.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=float("inf"))
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    pool_ids = set(result.pool_indices.tolist())
    assert len(pool_ids) >= 3


def test_bpm_floor_rejects_octave_mismatch():
    rng = np.random.default_rng(0)
    N = 5
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    # seed (idx 0) at 70 BPM; idx 2 at 140 (1 octave away)
    perceptual_bpm = np.array([70.0, 80.0, 140.0, 72.0, 60.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30)
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    pool_ids = set(result.pool_indices.tolist())
    assert 2 not in pool_ids  # 140 BPM rejected
    assert 1 in pool_ids      # 80 BPM close to 70
    assert result.stats.get("below_bpm_floor", 0) >= 1


def test_bpm_floor_max_over_seeds():
    """Candidate compatible with any seed BPM should pass."""
    rng = np.random.default_rng(0)
    N = 5
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    perceptual_bpm = np.array([70.0, 140.0, 140.0, 60.0, 100.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30)
    result = build_candidate_pool(
        seed_idx=0, seed_indices=[1],
        embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    # idx 2 at 140 BPM matches seed idx 1 at 140 BPM → should pass
    assert 2 in set(result.pool_indices.tolist())


def test_bpm_floor_skips_low_stability():
    """Candidate with low tempo_stability bypasses BPM gate."""
    rng = np.random.default_rng(0)
    N = 3
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    perceptual_bpm = np.array([70.0, 140.0, 75.0])
    tempo_stability = np.array([0.9, 0.3, 0.9])  # idx 1 unreliable
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30, bpm_stability_min=0.5)
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
    )
    assert 1 in set(result.pool_indices.tolist())


def test_bpm_floor_skips_nan():
    """Candidate with NaN BPM bypasses the gate."""
    rng = np.random.default_rng(0)
    N = 3
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    perceptual_bpm = np.array([70.0, np.nan, 75.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30)
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    assert 1 in set(result.pool_indices.tolist())
