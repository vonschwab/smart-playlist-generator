"""Onset admission band rejects far-density candidates; rhythm-cosine no longer gates."""
import numpy as np
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _cfg():
    return CandidatePoolConfig(
        similarity_floor=-1.0, min_sonic_similarity=None, max_pool_size=50,
        target_artists=50, candidates_per_artist=5, seed_artist_bonus=0,
        max_artist_fraction_final=1.0, onset_admission_max_log_distance=0.6,
    )


def test_onset_band_rejects_far_density():
    # 4 tracks; seed idx0 onset=2.0; idx1 close (2.5), idx2 far (16.0), idx3 NaN bypass
    N = 4
    emb = np.eye(N, 8, dtype=float)  # arbitrary, similarity_floor=-1 admits all sonically
    onset = np.array([2.0, 2.5, 16.0, np.nan])
    result = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=emb,
        artist_keys=np.array(["s", "a", "b", "c"]),
        track_ids=np.array(["s", "a", "b", "c"]),
        track_titles=np.array(["s", "a", "b", "c"]),
        track_artists=np.array(["s", "a", "b", "c"]),
        durations_ms=np.array([200000] * N),
        cfg=_cfg(), random_seed=0, X_sonic=emb,
        onset_rate=onset,
    )
    members = set(result.pool_indices.tolist())
    assert 1 in members        # close density admitted
    assert 2 not in members    # far density rejected
    assert 3 in members        # NaN bypassed
