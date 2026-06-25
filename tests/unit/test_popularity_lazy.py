import numpy as np
import types
from unittest.mock import MagicMock
from src.analyze.popularity_runner import (
    init_top_tracks_cache, upsert_artist_top_tracks,
    get_artist_top_tracks_cached_or_fetch,
    load_artist_popularity_values,
)

ROWS = [{"name": "Hit", "playcount": 9, "mbid": "", "rank": 0}]


def test_cache_hit_skips_fetch(tmp_path):
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-20T00:00:00+00:00", ROWS)
    client = MagicMock()
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00",
    )
    assert out == ROWS
    client.get_artist_top_tracks.assert_not_called()   # fresh cache -> no network


def test_miss_fetches_and_caches(tmp_path):
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    client = MagicMock()
    client.get_artist_top_tracks.return_value = ROWS
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db, now_iso="2026-06-24T00:00:00+00:00",
    )
    assert out == ROWS
    client.get_artist_top_tracks.assert_called_once()
    # now cached -> second call doesn't fetch
    client2 = MagicMock()
    again = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client2, db_path=db, now_iso="2026-06-24T00:00:00+00:00")
    client2.get_artist_top_tracks.assert_not_called()
    assert again == ROWS


def test_stale_refetches(tmp_path):
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-01-01T00:00:00+00:00", [])  # old + empty
    fresh = [{"name": "New", "playcount": 1, "mbid": "", "rank": 0}]
    client = MagicMock()
    client.get_artist_top_tracks.return_value = fresh
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00")
    client.get_artist_top_tracks.assert_called_once()
    assert out == fresh


def test_fetch_failure_falls_back_to_stale_then_empty(tmp_path):
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-01-01T00:00:00+00:00", ROWS)  # stale
    client = MagicMock()
    client.get_artist_top_tracks.side_effect = RuntimeError("net")
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00")
    assert out == ROWS    # stale cache used on failure (graceful)
    # no cache at all + failure -> []
    out2 = get_artist_top_tracks_cached_or_fetch(
        "absent", "Absent", client=client, db_path=db, now_iso="2026-06-24T00:00:00+00:00")
    assert out2 == []


def test_aware_vs_naive_now_iso_does_not_crash_treats_as_stale(tmp_path):
    # Stored timestamp is timezone-AWARE; a naive now_iso would raise TypeError on
    # subtraction. The TTL guard must swallow it (never gate generation) -> stale -> refetch.
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-24T00:00:00+00:00", ROWS)  # aware
    fresh_rows = [{"name": "New", "playcount": 1, "mbid": "", "rank": 0}]
    client = MagicMock()
    client.get_artist_top_tracks.return_value = fresh_rows
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-25T00:00:00")  # NAIVE (no offset)
    client.get_artist_top_tracks.assert_called_once()    # treated as stale -> refetched
    assert out == fresh_rows


def _bundle(track_ids, titles, artist_keys):
    return types.SimpleNamespace(
        track_ids=np.array(track_ids, dtype=object),
        track_titles=np.array(titles, dtype=object),
        artist_keys=np.array(artist_keys, dtype=object),
        track_artists=None,
    )


def test_load_artist_popularity_aligns_seed_artist(tmp_path):
    b = _bundle(
        ["n_studio", "n_live", "other"],
        ["In Bloom", "In Bloom (Live)", "Some Song"],
        ["nirvana", "nirvana", "other"],
    )
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-24T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    client = MagicMock()
    vec = load_artist_popularity_values(
        b, "Nirvana", client=client, db_path=db, limit=50, max_age_days=30,
        now_iso="2026-06-24T00:00:00+00:00")
    assert vec is not None and vec.shape == (3,)
    assert vec[0] == 1.0            # studio In Bloom got the hit popularity
    assert np.isnan(vec[1])         # live lost
    assert np.isnan(vec[2])         # other artist untouched
    client.get_artist_top_tracks.assert_not_called()   # used the fresh cache


def test_load_artist_popularity_none_without_client(tmp_path):
    b = _bundle(["a"], ["T"], ["x"])
    assert load_artist_popularity_values(
        b, "X", client=None, db_path=str(tmp_path / "e.db"),
        limit=50, max_age_days=30, now_iso="2026-06-24T00:00:00+00:00") is None
