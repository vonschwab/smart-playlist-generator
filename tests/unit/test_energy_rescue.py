import numpy as np
from dataclasses import replace as _replace
from src.playlist.energy_rescue import select_energy_rescue
from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig

def test_spans_arousal_range():
    arousal = np.array([0.0, -2.0, 2.0, 1.0, -1.0, 0.5])
    src = [0, 1, 2, 3, 4, 5]
    picked = select_energy_rescue(arousal, src, k_energy=3)
    vals = sorted(arousal[i] for i in picked)
    assert len(picked) == 3
    assert vals[0] == -2.0 and vals[-1] == 2.0   # endpoints of the range are represented

def test_returns_all_when_source_small():
    arousal = np.array([0.0, 1.0, -1.0])
    assert sorted(select_energy_rescue(arousal, [0, 1, 2], k_energy=5)) == [0, 1, 2]

def test_zero_k_and_empty_source():
    arousal = np.array([0.0, 1.0])
    assert select_energy_rescue(arousal, [0, 1], k_energy=0) == []
    assert select_energy_rescue(arousal, [], k_energy=3) == []

def test_skips_nan_arousal():
    arousal = np.array([0.0, np.nan, 2.0])
    picked = select_energy_rescue(arousal, [0, 1, 2], k_energy=3)
    assert 1 not in picked and set(picked) == {0, 2}


def _toy(n=40):
    rng = np.random.default_rng(0)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    track_ids = [f"t{i}" for i in range(n)]
    artist_keys = [f"a{i}" for i in range(n)]
    return X_sonic, track_ids, artist_keys


def test_rescue_admits_rhythm_rejected_but_sonic_ok():
    # Two seeds with high onset; make most tracks fail the onset band, but keep
    # them sonically close to a seed; rescue should re-admit arousal-spanning ones.
    X_sonic, track_ids, artist_keys = _toy()
    n = len(track_ids)
    onset = np.full(n, 5.0); onset[:2] = 0.1            # seeds slow, others fast -> onset band rejects others
    arousal = np.linspace(-2, 2, n)
    base = CandidateConfig(
        similarity_floor=-1.0, min_sonic_similarity=None, max_pool_size=1000,
        target_artists=1000, onset_admission_max_log_distance=0.30,
        candidates_per_artist=100, seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
    )
    seeds = [0, 1]
    no_rescue = build_candidate_pool(
        seed_idx=0, seed_indices=seeds, embedding=X_sonic, artist_keys=artist_keys,
        track_ids=track_ids, cfg=base, random_seed=0, X_sonic=X_sonic, onset_rate=onset,
        X_energy=arousal,
    )
    with_rescue = build_candidate_pool(
        seed_idx=0, seed_indices=seeds, embedding=X_sonic, artist_keys=artist_keys,
        track_ids=track_ids, cfg=_replace(base, pace_rescue_k_energy=6),
        random_seed=0, X_sonic=X_sonic, onset_rate=onset, X_energy=arousal,
    )
    n0 = len(no_rescue.pool_indices); n1 = len(with_rescue.pool_indices)
    assert n1 >= n0                                   # additive: never shrinks
    assert n1 > n0                                    # rescue actually admitted some
    # rescued set spans arousal (min and max arousal both present among rescued)
    rescued = set(int(i) for i in with_rescue.pool_indices) - set(int(i) for i in no_rescue.pool_indices)
    a_res = sorted(arousal[i] for i in rescued)
    assert a_res and a_res[0] < -0.5 and a_res[-1] > 0.5
