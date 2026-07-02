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
                             [{"name": "In Bloom", "mbid": "in-bloom-mbid", "rank": 0}])
    upsert_artist_top_tracks(db, "the smiths", "2026-06-25T00:00:00+00:00",
                             [{"name": "This Charming Man", "mbid": "charming-man-mbid", "rank": 0}])
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


def test_load_pool_popularity_ranks_cached_returns_rank_not_score(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    import src.analyze.popularity_runner as pr

    bundle = SimpleNamespace(
        track_ids=np.array(["t0", "t1", "t2", "t3"], dtype=object),
        track_titles=np.array(["Hit A", "Hit B", "Deep Cut", "Other"], dtype=object),
        artist_keys=np.array(["nirvana", "nirvana", "nirvana", "uncached"], dtype=object),
    )
    # nirvana top tracks: Hit A rank 0, Hit B rank 1 (Deep Cut absent)
    def fake_cached(db_path, key):
        if key == "nirvana":
            return [{"name": "Hit A", "rank": 0, "mbid": ""},
                    {"name": "Hit B", "rank": 1, "mbid": ""}]
        return []
    monkeypatch.setattr(pr, "get_artist_top_tracks_cached", fake_cached)

    ranks = pr.load_pool_popularity_ranks_cached(bundle, [0, 1, 2, 3], db_path=":memory:")
    assert ranks.tolist() == [0, 1, -1, -1]   # rank, rank, not-in-top-N, uncached
    assert ranks.dtype.kind == "i"


def test_annotate_and_log_playlist_popularity(tmp_path, caplog):
    import logging
    from src.analyze.popularity_runner import annotate_and_log_playlist_popularity
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-26T00:00:00+00:00", [
        {"name": "In Bloom", "mbid": "", "rank": 0},
        {"name": "Polly", "mbid": "", "rank": 4},
    ])
    tracks = [
        {"artist": "Nirvana", "title": "In Bloom"},
        {"artist": "Nirvana", "title": "Endless Nameless"},  # real deep cut, not on top list
        {"artist": "Unknown Band", "title": "Whatever"},     # uncached artist
    ]
    with caplog.at_level(logging.INFO):
        annotate_and_log_playlist_popularity(tracks, db_path=db)
    assert tracks[0]["popularity_rank"] == 1      # rank 0 -> #1 (1-based)
    assert tracks[1]["popularity_rank"] is None   # not on the artist's top-50
    assert tracks[2]["popularity_rank"] is None   # uncached -> None
    msgs = " ".join(r.message for r in caplog.records)
    assert "Last.fm #1" in msgs and "In Bloom" in msgs
