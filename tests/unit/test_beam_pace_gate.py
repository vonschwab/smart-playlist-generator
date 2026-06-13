import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def test_beam_soft_penalty_demotes_off_rhythm_candidate():
    # rhythm-cosine hard gate removed (2026-06-12 pace-gate retune); the
    # equivalent is rhythm_soft_penalty_strength=1.0 which zeros the score of
    # any candidate whose rhythm cosine falls below the threshold.
    X = np.ones((4, 4), dtype=float)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    rhythm = np.array(
        [
            [1.0, 0.0],  # pier A
            [1.0, 0.0],  # aligned candidate
            [0.0, 1.0],  # orthogonal candidate (pace_sim=0.0 < threshold)
            [1.0, 0.0],  # pier B
        ],
        dtype=float,
    )

    path, _hits, _edges, err = _beam_search_segment(
        0,
        3,
        1,
        [2, 1],
        X_norm,
        X_norm,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            bridge_floor=-1.0,
            transition_floor=-1.0,
            pace_bridge_floor=0.0,
            rhythm_soft_penalty_threshold=0.5,
            rhythm_soft_penalty_strength=1.0,
            progress_enabled=False,
        ),
        5,
        rhythm_matrix=rhythm,
    )

    assert err is None
    assert path == [1]
