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


# --- genre-aware terminal placement (band-aid for genre-blind fallback) -----
# idx 0=pier_a, 1=pier_b, 2=cand_S (sonic-best, genre-incoherent),
#               3=cand_G (sonic-poor, genre-coherent with both piers).
_SONIC = _rows([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
_GENRE = _rows([[1, 0], [1, 0], [0, 1], [1, 0]])


def test_genre_weight_zero_is_legacy_sonic_behavior():
    # No genre args -> picks the sonic-best track even though it is genre-incoherent.
    assert _greedy_terminal_path([2, 3], set(), 0, 1, 1, _SONIC) == [2]
    # Passing genre vectors but weight 0 must be identical to legacy.
    assert _greedy_terminal_path(
        [2, 3], set(), 0, 1, 1, _SONIC, X_genre_norm=_GENRE, genre_weight=0.0
    ) == [2]


def test_genre_weight_flips_to_genre_coherent_pick():
    # With genre weighted in, the sonic-best-but-genre-incoherent track loses to
    # the genre-coherent one.
    assert _greedy_terminal_path(
        [2, 3], set(), 0, 1, 1, _SONIC, X_genre_norm=_GENRE, genre_weight=0.7
    ) == [3]


def test_genre_weight_without_vectors_falls_back_to_sonic():
    # Never-fail: genre weight set but no genre matrix -> sonic behavior, no crash.
    assert _greedy_terminal_path(
        [2, 3], set(), 0, 1, 1, _SONIC, X_genre_norm=None, genre_weight=0.7
    ) == [2]


# --- artist diversity in terminal placement (the hard constraint the fallback
#     was violating: 3 same-artist tracks placed adjacent) ----------------------
# pier_a=0, pier_b=1 (orthogonal). idx 2,3,4 = artist "edm" (highest sonic blend),
# idx 5="a", 6="b" (clearly lower blend). interior_len=3.
_DIV = _rows([
    [1.0, 0.0],   # 0 pier_a
    [0.0, 1.0],   # 1 pier_b
    [1.0, 1.00],  # 2 edm (best)
    [1.0, 0.95],  # 3 edm
    [1.0, 0.90],  # 4 edm
    [1.0, 0.20],  # 5 a  (lower blend)
    [0.9, 0.20],  # 6 b  (lower blend)
])
_DIV_KEYS = {2: {"edm"}, 3: {"edm"}, 4: {"edm"}, 5: {"a"}, 6: {"b"}}


def _keys(idx):
    return _DIV_KEYS.get(int(idx), set())


def test_legacy_clusters_same_artist_without_keys():
    # No artist_key_fn -> legacy behavior picks the 3 top-scored, all "edm" (the bug).
    path = _greedy_terminal_path([2, 3, 4, 5, 6], set(), 0, 1, 3, _DIV)
    assert sorted(path) == [2, 3, 4]


def test_artist_key_fn_prevents_same_artist_clustering():
    path = _greedy_terminal_path(
        [2, 3, 4, 5, 6], set(), 0, 1, 3, _DIV, artist_key_fn=_keys
    )
    assert path is not None and len(path) == 3
    for x, y in zip(path, path[1:]):
        assert not (_keys(x) & _keys(y)), f"adjacent same-artist at {x},{y}"
    # 3 distinct artists available for 3 slots -> "edm" appears at most once
    assert sum(1 for i in path if _keys(i) == {"edm"}) == 1


def test_never_fail_when_only_one_artist():
    # All candidates one artist -> still fills interior_len (best-effort, never-fail).
    path = _greedy_terminal_path(
        [2, 3, 4], set(), 0, 1, 3, _DIV, artist_key_fn=lambda idx: {"edm"}
    )
    assert path is not None and sorted(path) == [2, 3, 4]


def test_blocked_boundary_artist_not_placed_first():
    # Previous segment ended on "edm" -> the fallback must not lead with edm.
    path = _greedy_terminal_path(
        [2, 3, 4, 5, 6], set(), 0, 1, 3, _DIV,
        artist_key_fn=_keys, blocked_artist_keys={"edm"},
    )
    assert path is not None and len(path) == 3
    assert _keys(path[0]) != {"edm"}
    for x, y in zip(path, path[1:]):
        assert not (_keys(x) & _keys(y))
