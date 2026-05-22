import numpy as np

from src.playlist.pier_bridge_builder import _segment_far_stats, _select_connector_candidates


def test_segment_far_stats_basic():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    stats = _segment_far_stats(
        pier_a=0,
        pier_b=1,
        X_full_norm=X,
        X_genre_norm=None,
        universe=[0, 1, 2],
        used_track_ids=set(),
        bridge_floor=0.1,
    )
    assert stats["sonic_sim"] is not None
    assert stats["sonic_sim"] < 0.1
    assert stats["connector_scarcity"] is not None


def test_select_connector_candidates_ranks_by_bridgeability():
    X = np.array(
        [
            [1.0, 0.0],  # cand 0
            [0.0, 1.0],  # cand 1
            [0.7, 0.7],  # cand 2 (best min sim)
            [1.0, 0.0],  # pier A
            [0.0, 1.0],  # pier B
        ],
        dtype=float,
    )
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    available = [0, 1, 2]
    picked = _select_connector_candidates(available, X, pier_a=3, pier_b=4, cap=2)
    assert picked[0] == 2
