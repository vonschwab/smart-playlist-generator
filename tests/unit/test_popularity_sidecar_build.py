from src.analyze.popularity_runner import (
    init_top_tracks_cache, cached_artist_keys,
    upsert_artist_top_tracks, get_artist_top_tracks_cached,
)


def test_cache_roundtrip_and_skip_set(tmp_path):
    db = str(tmp_path / "enrich.db")
    init_top_tracks_cache(db)
    assert cached_artist_keys(db) == set()
    rows = [{"name": "X", "playcount": 5, "mbid": "", "rank": 0}]
    upsert_artist_top_tracks(db, "nirvana", "2026-06-24T00:00:00Z", rows)
    assert cached_artist_keys(db) == {"nirvana"}
    assert get_artist_top_tracks_cached(db, "nirvana") == rows
    # upsert replaces
    upsert_artist_top_tracks(db, "nirvana", "later", [])
    assert get_artist_top_tracks_cached(db, "nirvana") == []
    assert get_artist_top_tracks_cached(db, "absent") == []
