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


def test_fire_popularity_merges_alias_catalogs(tmp_path):
    from src.analyze.popularity_runner import init_top_tracks_cache, upsert_artist_top_tracks, load_artist_popularity_values
    from unittest.mock import MagicMock
    from src.string_utils import normalize_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])

    b = types.SimpleNamespace(
        track_ids=np.array(["early", "late", "other"], dtype=object),
        track_titles=np.array(["Early Song", "Late Song", "Nope"], dtype=object),
        artist_keys=np.array(["Alex G", "(Sandy) Alex G", "Other"], dtype=object),
        track_artists=np.array(["Alex G", "(Sandy) Alex G", "Other"], dtype=object),
        durations_ms=None,
    )
    db = str(tmp_path / "pop.db")
    init_top_tracks_cache(db)
    # Each name's catalog has its own Last.fm hit, cached under its own key.
    upsert_artist_top_tracks(db, normalize_artist_key("Alex G"), "2026-06-24T00:00:00+00:00",
                             [{"name": "Early Song", "mbid": "early-song-mbid", "rank": 0}])
    upsert_artist_top_tracks(db, normalize_artist_key("(Sandy) Alex G"), "2026-06-24T00:00:00+00:00",
                             [{"name": "Late Song", "mbid": "late-song-mbid", "rank": 0}])
    client = MagicMock()  # fresh cache -> no network
    vec = load_artist_popularity_values(
        b, "Alex G", client=client, db_path=db, limit=50, max_age_days=30,
        now_iso="2026-06-24T00:00:00+00:00")
    assert vec is not None and vec.shape == (3,)
    assert vec[0] == 1.0 and vec[1] == 1.0   # BOTH catalogs' hits landed
    assert np.isnan(vec[2])
    client.get_artist_top_tracks.assert_not_called()
