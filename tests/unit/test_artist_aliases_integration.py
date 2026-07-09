import types
import numpy as np
from src.playlist.artist_aliases import set_artist_link_map_for_testing


def _ns_bundle(artist_keys, track_artists=None):
    return types.SimpleNamespace(
        artist_keys=np.array(artist_keys, dtype=object),
        track_artists=np.array(track_artists if track_artists is not None else artist_keys, dtype=object),
    )


def test_artist_indices_gathers_alias_members():
    from src.playlist.artist_style import _artist_indices_in_bundle
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    b = _ns_bundle(["Alex G", "(Sandy) Alex G", "Other Band"])
    assert _artist_indices_in_bundle(b, "Alex G") == [0, 1]
    assert _artist_indices_in_bundle(b, "(Sandy) Alex G") == [0, 1]


def test_artist_indices_unlinked_unchanged():
    from src.playlist.artist_style import _artist_indices_in_bundle
    set_artist_link_map_for_testing(None)  # empty
    b = _ns_bundle(["Alex G", "(Sandy) Alex G", "Other Band"])
    assert _artist_indices_in_bundle(b, "Alex G") == [0]


def test_normalize_primary_artist_key_merges_aliases():
    from src.playlist.identity_keys import normalize_primary_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert normalize_primary_artist_key("Alex G") == normalize_primary_artist_key("(Sandy) Alex G")


def test_identity_keys_for_index_merges_aliases():
    from src.playlist.identity_keys import identity_keys_for_index
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    b = types.SimpleNamespace(
        track_ids=np.array(["t0", "t1"], dtype=object),
        track_artists=np.array(["Alex G", "(Sandy) Alex G"], dtype=object),
        artist_keys=np.array(["Alex G", "(Sandy) Alex G"], dtype=object),
        track_titles=np.array(["S0", "S1"], dtype=object),
    )
    assert identity_keys_for_index(b, 0).artist_key == identity_keys_for_index(b, 1).artist_key


def test_candidate_pool_normalize_key_merges_aliases():
    from src.playlist.candidate_pool import _normalize_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert _normalize_artist_key("Alex G") == _normalize_artist_key("(Sandy) Alex G")
    set_artist_link_map_for_testing(None)
    assert _normalize_artist_key("Alex G") != _normalize_artist_key("(Sandy) Alex G")
