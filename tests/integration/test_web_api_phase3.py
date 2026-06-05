import sys

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_playlist_out_maps_transition_score_and_percentiles():
    from src.playlist_web.schemas import PlaylistOut

    raw = {
        "name": "X", "track_count": 2,
        "tracks": [
            {"position": 0, "rating_key": "k0", "artist": "A", "title": "T0",
             "album": "Al", "duration_ms": 1, "file_path": "/0", "genres": [],
             "transition_score": 0.62},
            {"position": 1, "rating_key": "k1", "artist": "B", "title": "T1",
             "album": "Al", "duration_ms": 1, "file_path": "/1", "genres": [],
             "transition_score": None},
        ],
        "metrics": {"mean_transition": 0.7, "min_transition": 0.5,
                    "p10_transition": 0.55, "p90_transition": 0.8, "distinct_artists": 2},
    }
    pl = PlaylistOut.from_worker(raw)
    assert pl.tracks[0].transition_score == 0.62
    assert pl.tracks[1].transition_score is None
    assert pl.metrics.p10_transition == 0.55
    assert pl.metrics.p90_transition == 0.8
