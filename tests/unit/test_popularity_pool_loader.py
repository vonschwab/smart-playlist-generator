import numpy as np
import types
from unittest.mock import MagicMock
from src.analyze.popularity_runner import (
    load_pool_popularity_values,
    init_top_tracks_cache,
    upsert_artist_top_tracks,
)


def _bundle(ids, titles, keys):
    return types.SimpleNamespace(
        track_ids=np.array(ids, dtype=object),
        track_titles=np.array(titles, dtype=object),
        artist_keys=np.array(keys, dtype=object),
    )


def test_pool_loader_scores_multiple_artists_aligned(tmp_path):
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-25T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    upsert_artist_top_tracks(db, "the smiths", "2026-06-25T00:00:00+00:00",
                             [{"name": "This Charming Man", "mbid": "", "rank": 0}])
    b = _bundle(
        ["n_hit", "n_deep", "s_hit", "other"],
        ["In Bloom", "Endless Nameless", "This Charming Man", "Song"],
        ["nirvana", "nirvana", "the smiths", "unknownband"],
    )
    client = MagicMock()  # both scanned artists cached -> no fetch
    # 'unknownband' is deliberately NOT in the scan map (not a pool artist) -> its track stays NaN
    vec = load_pool_popularity_values(
        b, {"nirvana": "Nirvana", "the smiths": "The Smiths"},
        client=client, db_path=db, now_iso="2026-06-25T00:00:00+00:00")
    assert vec.shape == (4,)
    assert vec[0] == 1.0          # Nirvana hit
    assert np.isnan(vec[1])       # Nirvana deep cut, not on top list
    assert vec[2] == 1.0          # Smiths hit (different artist, aligned)
    assert np.isnan(vec[3])       # artist not in the scan map -> untouched
    client.get_artist_top_tracks.assert_not_called()


def test_pool_loader_none_without_client(tmp_path):
    b = _bundle(["a"], ["T"], ["x"])
    assert load_pool_popularity_values(
        b, {"x": "X"}, client=None, db_path=str(tmp_path / "e.db")) is None


def test_pool_loader_fetch_failure_is_graceful(tmp_path):
    # an uncached artist whose fetch raises must not break the whole vector
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-25T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    b = _bundle(["n_hit", "boom"], ["In Bloom", "X"], ["nirvana", "flaky"])
    client = MagicMock()
    client.get_artist_top_tracks.side_effect = RuntimeError("net")  # only fires for uncached 'flaky'
    vec = load_pool_popularity_values(
        b, {"nirvana": "Nirvana", "flaky": "Flaky"},
        client=client, db_path=db, now_iso="2026-06-25T00:00:00+00:00")
    assert vec[0] == 1.0          # cached artist still resolved
    assert np.isnan(vec[1])       # failed-fetch artist -> NaN, no crash


def test_pool_loader_cached_reads_cache_only(tmp_path):
    from src.analyze.popularity_runner import load_pool_popularity_values_cached
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-25T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    b = _bundle(
        ["n_hit", "n_deep", "u_track"],
        ["In Bloom", "Endless Nameless", "X"],
        ["nirvana", "nirvana", "uncached"],
    )
    vec = load_pool_popularity_values_cached(b, [0, 1, 2], db_path=db)
    assert vec[0] == 1.0          # cached hit
    assert np.isnan(vec[1])       # cached artist, not a top track
    assert np.isnan(vec[2])       # uncached artist -> NaN (no fetch)


def test_pool_loader_cached_only_scans_given_indices(tmp_path):
    from src.analyze.popularity_runner import load_pool_popularity_values_cached
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-25T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    b = _bundle(["n_hit", "n_hit2"], ["In Bloom", "In Bloom"], ["nirvana", "nirvana"])
    vec = load_pool_popularity_values_cached(b, [0], db_path=db)  # only index 0 in the pool
    assert vec[0] == 1.0
    assert np.isnan(vec[1])       # index 1 not in pool_indices -> untouched
