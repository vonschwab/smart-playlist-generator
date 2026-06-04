"""Pydantic schemas for web API request/response serialization."""

import pytest

from src.playlist_web.schemas import (
    GenerateRequestBody,
    PlaylistOut,
    TrackOut,
)


def test_request_body_maps_to_generate_request():
    body = GenerateRequestBody(
        mode="artist", artist="Acetone", tracks=20,
        genre_mode="narrow", sonic_mode="strict", pace_mode="dynamic",
    )
    req = body.to_request()
    assert req.mode == "artist"
    assert req.artist == "Acetone"
    assert req.tracks == 20
    assert req.genre_mode == "narrow"
    assert req.validation_error() is None
    args = req.to_worker_args()
    assert args["mode"] == "artist"
    assert args["artist"] == "Acetone"


def test_request_body_validation_error_surfaces():
    body = GenerateRequestBody(mode="artist", artist="", tracks=10)
    assert body.to_request().validation_error() == "Enter an artist before generating."


def test_playlist_out_parses_worker_result():
    raw = {
        "name": "Generated Playlist",
        "track_count": 1,
        "tracks": [{
            "position": 0, "rating_key": "k1", "artist": "Acetone",
            "title": "Sundown", "album": "Cindy", "duration_ms": 200000,
            "file_path": "/x.flac", "sonic_similarity": 0.91,
            "genre_similarity": 0.8, "genres": ["slowcore"],
        }],
        "metrics": {"mean_transition": 0.88, "min_transition": 0.81, "distinct_artists": 18},
    }
    out = PlaylistOut.from_worker(raw)
    assert out.track_count == 1
    assert out.tracks[0].title == "Sundown"
    assert out.metrics.distinct_artists == 18
