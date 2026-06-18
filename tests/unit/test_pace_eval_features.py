import numpy as np
from scripts.research import pace_eval_features as fe


def test_candidates_cover_ratified_menu():
    assert set(fe.CANDIDATES) == {
        "rhythm_tower", "perceptual_bpm", "onset_rate", "beat_strength",
        "arousal_p50", "danceability", "energy_pair", "energy_dist",
        "energy_onset", "pace_full",
    }


def test_zscore_and_candidate_vector():
    raw_scalars = {k: np.array([0.0, 2.0, 4.0]) for k in fe.SCALAR_KEYS}
    raw_tower = np.array([[0.0] * 9, [2.0] * 9, [4.0] * 9])
    zs, zt = fe.zscore_features(raw_scalars, raw_tower)
    # idx 1 is the mean -> zscore 0 for every scalar
    v = fe.candidate_vector("pace_full", 1, zs, zt)
    assert v.shape == (5,) and np.allclose(v, 0.0)
    assert fe.candidate_vector("energy_pair", 0, zs, zt).shape == (2,)
    assert fe.candidate_vector("rhythm_tower", 0, zs, zt).shape == (9,)


def test_candidate_vector_nan_when_feature_missing():
    raw_scalars = {k: np.array([np.nan, 1.0]) for k in fe.SCALAR_KEYS}
    raw_tower = np.zeros((2, 9))
    zs, zt = fe.zscore_features(raw_scalars, raw_tower)
    v = fe.candidate_vector("arousal_p50", 0, zs, zt)
    assert np.isnan(v).any()
