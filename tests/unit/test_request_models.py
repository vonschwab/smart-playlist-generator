"""Tests for GeneratePlaylistRequest roundtrip through worker args."""

from src.playlist.request_models import GeneratePlaylistRequest


def test_popular_seeds_and_seed_epoch_roundtrip_worker_args():
    req = GeneratePlaylistRequest(mode="artist", artist="Nirvana", popular_seeds=True, seed_epoch=3)
    args = req.to_worker_args()
    assert args.get("popular_seeds") is True and args.get("seed_epoch") == 3
    back = GeneratePlaylistRequest.from_worker_args(args)
    assert back.popular_seeds is True and back.seed_epoch == 3
    # defaults sparse
    base = GeneratePlaylistRequest(mode="artist", artist="X")
    a2 = base.to_worker_args()
    assert "popular_seeds" not in a2 and "seed_epoch" not in a2


def test_popularity_mode_roundtrip_worker_args():
    req = GeneratePlaylistRequest(mode="artist", artist="Nirvana", popularity_mode="oops")
    args = req.to_worker_args()
    assert args.get("popularity_mode") == "oops"
    back = GeneratePlaylistRequest.from_worker_args(args)
    assert back.popularity_mode == "oops"
    # default "off" is sparse + parses back to off
    base = GeneratePlaylistRequest(mode="artist", artist="X")
    assert base.popularity_mode == "off"
    assert "popularity_mode" not in base.to_worker_args()
    assert GeneratePlaylistRequest.from_worker_args({}).popularity_mode == "off"
