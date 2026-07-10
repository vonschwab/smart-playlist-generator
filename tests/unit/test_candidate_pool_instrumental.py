"""Task 8: candidate-pool admission demotion for vocal-classified candidates.

Mirrors the duration-penalty test pattern (test_candidate_pool_pace_floor.py):
build a tiny pool where every candidate is otherwise tied on seed_sim, then
verify a high-voice_prob non-seed candidate is demoted in the returned
seed_sim ranking while the same run with the guard off (or voice_prob=None)
leaves it untouched. Soft signal — never excludes the candidate outright.
"""
import numpy as np
import pytest

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(**overrides):
    base = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=20,
        target_artists=4,
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


def _pool():
    # 4 tracks, all identical embeddings -> tied seed_sim=1.0 pre-penalty.
    N = 4
    embedding = np.ones((N, 4), dtype=float)
    artist_keys = np.array([f"a{i}" for i in range(N)])
    return embedding, artist_keys


def test_instrumental_demotion_lowers_seed_sim_for_vocal_candidate():
    embedding, artist_keys = _pool()
    # Track 1 is confidently vocal; tracks 2/3 are confidently instrumental.
    voice_prob = np.array([0.0, 0.9, 0.0, 0.05])

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=_make_cfg(instrumental_enabled=True, instrumental_penalty_weight=0.5),
        random_seed=0,
        voice_prob=voice_prob,
    )

    seed_sim_by_index = dict(zip(result.pool_indices.tolist(), result.seed_sim.tolist()))
    assert seed_sim_by_index[2] == pytest.approx(1.0)  # untouched (voice_prob=0.0)
    assert seed_sim_by_index[1] == pytest.approx(1.0 - 0.5 * 0.9)
    assert seed_sim_by_index[1] < seed_sim_by_index[2]
    # Soft penalty: the demoted vocal track is still present in the pool.
    assert 1 in result.pool_indices
    assert result.stats.get("instrumental_penalty_applied", 0) >= 1


def test_instrumental_demotion_inert_when_disabled():
    embedding, artist_keys = _pool()
    voice_prob = np.array([0.0, 0.9, 0.0, 0.05])

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        # instrumental_enabled defaults to False; voice_prob supplied but must be a no-op.
        cfg=_make_cfg(),
        random_seed=0,
        voice_prob=voice_prob,
    )

    assert np.allclose(result.seed_sim, 1.0)
    assert "instrumental_penalty_applied" not in result.stats


def test_instrumental_demotion_inert_when_voice_prob_none():
    embedding, artist_keys = _pool()

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=_make_cfg(instrumental_enabled=True, instrumental_penalty_weight=0.5),
        random_seed=0,
        voice_prob=None,
    )

    assert np.allclose(result.seed_sim, 1.0)


def test_instrumental_params_effective_absent_when_voice_prob_all_nan():
    # Review fix: load_voice_prob never returns None -- a missing/corrupt sidecar
    # yields an all-NaN array. params_effective must not report the guard as
    # in-effect ("instrumental_penalty_weight") when it had no finite data to
    # act on, even though voice_prob is not None.
    embedding, artist_keys = _pool()
    voice_prob = np.full(4, np.nan)

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=_make_cfg(instrumental_enabled=True, instrumental_penalty_weight=0.5),
        random_seed=0,
        voice_prob=voice_prob,
    )

    assert "instrumental_penalty_weight" not in result.params_effective
    assert np.allclose(result.seed_sim, 1.0)
