"""Unit test for the edge_repair per-pass identity memo: cached lookups must
equal the uncached functions and must actually memoize (return the same object)."""
import types
from typing import Any

from src.playlist.repair.edge_repair import _IdentityMemo, _cap_artist_keys_for_idx
from src.playlist.identity_keys import identity_keys_for_index


def _fake_bundle() -> Any:
    # identity_keys_for_index / _cap_artist_keys_for_idx read only these four
    # attributes (each under try/except), so a duck-typed namespace suffices.
    return types.SimpleNamespace(
        track_ids=["t0", "t1", "t2"],
        track_artists=["Miles Davis Quintet", "", "Bill Evans Trio"],
        artist_keys=["miles", "solo", "bill"],
        track_titles=["So What", "Untitled", "Waltz"],
    )


def test_memo_keys_for_index_equal_uncached_and_memoized():
    b = _fake_bundle()
    memo = _IdentityMemo()
    for i in range(3):
        assert memo.keys_for_index(b, i) == identity_keys_for_index(b, i)
    first = memo.keys_for_index(b, 0)
    assert memo.keys_for_index(b, 0) is first  # cached: identical object on re-lookup


def test_memo_cap_keys_equal_uncached_and_memoized():
    b = _fake_bundle()
    memo = _IdentityMemo()
    for i in range(3):
        assert memo.cap_keys(b, i, None) == _cap_artist_keys_for_idx(b, i, None)
    first = memo.cap_keys(b, 0, None)
    assert memo.cap_keys(b, 0, None) is first  # cached: identical object (incl. empty set)
