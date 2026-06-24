from unittest.mock import patch
from src.lastfm_client import LastFMClient


def _client():
    return LastFMClient(api_key="k", username="u")


def test_get_artist_top_tracks_parses_ranked_list():
    fake = {
        "toptracks": {
            "track": [
                {"name": "Smells Like Teen Spirit", "playcount": "9000000",
                 "listeners": "2000000", "mbid": "mbid-slts"},
                {"name": "Come as You Are", "playcount": "7000000",
                 "listeners": "1800000", "mbid": ""},
            ]
        }
    }
    with patch.object(LastFMClient, "_make_request", return_value=fake):
        out = _client().get_artist_top_tracks("Nirvana", limit=50)
    assert [t["name"] for t in out] == ["Smells Like Teen Spirit", "Come as You Are"]
    assert out[0]["playcount"] == 9000000 and out[0]["mbid"] == "mbid-slts"
    assert out[0]["rank"] == 0 and out[1]["rank"] == 1


def test_get_artist_top_tracks_handles_single_and_empty():
    single = {"toptracks": {"track": {"name": "X", "playcount": "5", "mbid": ""}}}
    with patch.object(LastFMClient, "_make_request", return_value=single):
        out = _client().get_artist_top_tracks("A")
    assert len(out) == 1 and out[0]["name"] == "X" and out[0]["rank"] == 0
    with patch.object(LastFMClient, "_make_request", return_value=None):
        assert _client().get_artist_top_tracks("A") == []
    with patch.object(LastFMClient, "_make_request", return_value={"toptracks": {}}):
        assert _client().get_artist_top_tracks("A") == []
