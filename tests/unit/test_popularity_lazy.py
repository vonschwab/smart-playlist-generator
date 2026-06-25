from unittest.mock import MagicMock
from src.analyze.popularity_runner import (
    init_top_tracks_cache, upsert_artist_top_tracks,
    get_artist_top_tracks_cached_or_fetch,
)

ROWS = [{"name": "Hit", "playcount": 9, "mbid": "", "rank": 0}]


def test_cache_hit_skips_fetch(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-20T00:00:00+00:00", ROWS)
    client = MagicMock()
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00",
    )
    assert out == ROWS
    client.get_artist_top_tracks.assert_not_called()   # fresh cache -> no network


def test_miss_fetches_and_caches(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    client = MagicMock(); client.get_artist_top_tracks.return_value = ROWS
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
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-01-01T00:00:00+00:00", [])  # old + empty
    fresh = [{"name": "New", "playcount": 1, "mbid": "", "rank": 0}]
    client = MagicMock(); client.get_artist_top_tracks.return_value = fresh
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00")
    client.get_artist_top_tracks.assert_called_once()
    assert out == fresh


def test_fetch_failure_falls_back_to_stale_then_empty(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-01-01T00:00:00+00:00", ROWS)  # stale
    client = MagicMock(); client.get_artist_top_tracks.side_effect = RuntimeError("net")
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00")
    assert out == ROWS    # stale cache used on failure (graceful)
    # no cache at all + failure -> []
    out2 = get_artist_top_tracks_cached_or_fetch(
        "absent", "Absent", client=client, db_path=db, now_iso="2026-06-24T00:00:00+00:00")
    assert out2 == []
