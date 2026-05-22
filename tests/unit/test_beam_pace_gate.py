import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def test_beam_filters_candidates_failing_pace_gate():
    X = np.ones((4, 4), dtype=float)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    rhythm = np.array(
        [
            [1.0, 0.0],  # pier A
            [1.0, 0.0],  # aligned candidate
            [0.0, 1.0],  # orthogonal candidate
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
            pace_bridge_floor=0.50,
            progress_enabled=False,
        ),
        5,
        rhythm_matrix=rhythm,
    )

    assert err is None
    assert path == [1]
