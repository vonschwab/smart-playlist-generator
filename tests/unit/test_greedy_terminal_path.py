import numpy as np
from src.playlist.pier_bridge_builder import _greedy_terminal_path


def _rows(vectors):
    X = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def test_interior_len_zero_returns_empty():
    X = _rows([[1, 0], [0, 1], [1, 1]])
    assert _greedy_terminal_path([0, 1, 2], set(), 0, 2, 0, X) == []


def test_returns_none_when_pool_too_small():
    X = _rows([[1, 0], [0, 1], [1, 1]])
    # piers 0 and 2 excluded, only idx 1 usable, need 2
    assert _greedy_terminal_path([0, 1, 2], set(), 0, 2, 2, X) is None


def test_excludes_used_piers_and_dedups():
    X = _rows([[1, 0], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0, 1]])
    pier_a, pier_b = 0, 4
    # candidates with a duplicate (1 appears twice); used={3}
    path = _greedy_terminal_path([1, 1, 2, 3], {3}, pier_a, pier_b, 2, X)
    assert path is not None
    assert len(path) == 2 and len(set(path)) == 2          # no duplicates
    assert pier_a not in path and pier_b not in path and 3 not in path


def test_nan_rows_are_filtered_not_raised():
    X = _rows([[1, 0], [0, 1], [1, 1], [1, 1]])
    X[2] = np.nan  # poisoned row must be skipped, sort must not raise
    path = _greedy_terminal_path([1, 2, 3], set(), 0, 1, 1, X)
    assert path == [3]  # idx 2 (NaN) filtered; idx 3 chosen (idx 1 is pier_b)
