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
