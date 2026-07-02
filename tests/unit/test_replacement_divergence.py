"""SP-B: replacement divergences are BPM (pace) and full-sonic cosine (sound)."""
import numpy as np

from src.playlist.replacement import ReplacementContext, _pace_divergence, _sound_divergence


def _ctx(X_sonic, perceptual_bpm=None):
    return ReplacementContext(
        X_sonic=X_sonic,
        X_full=X_sonic,
        X_start=None,
        X_end=None,
        X_mid=None,
        X_genre_smoothed=None,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=None,
        track_ids=np.array(["a", "b"], dtype=object),
        artist_keys=np.array(["x", "y"], dtype=object),
        candidate_pool_indices=np.array([0, 1]),
        idf_weights=None,
    )


def test_sound_divergence_is_full_sonic_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # orthogonal
    ctx = _ctx(X)
    assert abs(_sound_divergence(ctx, cand_idx=1, current_idx=0) - 1.0) < 1e-6
    ctx2 = _ctx(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))  # identical
    assert abs(_sound_divergence(ctx2, cand_idx=1, current_idx=0) - 0.0) < 1e-6


def test_pace_divergence_uses_bpm_and_degrades_to_zero():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ctx = _ctx(X, perceptual_bpm=np.array([120.0, 60.0]))
    assert _pace_divergence(ctx, cand_idx=1, current_idx=0) > 0.0
    ctx_nobpm = _ctx(X, perceptual_bpm=None)
    # No BPM data: no pace signal (the tower rhythm axis is gone) — 0.0, not garbage.
    assert _pace_divergence(ctx_nobpm, cand_idx=1, current_idx=0) == 0.0
