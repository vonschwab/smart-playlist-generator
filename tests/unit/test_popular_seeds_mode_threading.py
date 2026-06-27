from src.playlist.request_models import GeneratePlaylistRequest


def test_popular_seeds_mode_round_trips_through_worker_args():
    req = GeneratePlaylistRequest.from_worker_args(
        {"mode": "artist", "artist": "Stereolab", "popular_seeds_mode": "fire"}
    )
    assert req.popular_seeds_mode == "fire"
    args = req.to_worker_args()
    assert args.get("popular_seeds_mode") == "fire"


def test_popular_seeds_mode_defaults_off():
    req = GeneratePlaylistRequest.from_worker_args({"mode": "artist", "artist": "X"})
    assert req.popular_seeds_mode == "off"
