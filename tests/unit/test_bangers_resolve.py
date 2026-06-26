from src.playlist_generator import (
    _resolve_popularity_rank_cutoff,
    _resolve_popular_seeds,
)


def test_cutoff_off_is_none():
    assert _resolve_popularity_rank_cutoff("off", {}) is None


def test_cutoff_on_defaults_50_oops_defaults_10():
    assert _resolve_popularity_rank_cutoff("on", {}) == 50
    assert _resolve_popularity_rank_cutoff("oops", {}) == 10


def test_cutoff_reads_config_overrides():
    cfg = {"rank_cutoff_on": 40, "rank_cutoff_oops": 15}
    assert _resolve_popularity_rank_cutoff("on", cfg) == 40
    assert _resolve_popularity_rank_cutoff("oops", cfg) == 15


def test_popular_seeds_forced_only_by_oops():
    assert _resolve_popular_seeds(False, "oops") is True     # OOPS forces it
    assert _resolve_popular_seeds(False, "on") is False      # ON does not
    assert _resolve_popular_seeds(False, "off") is False
    assert _resolve_popular_seeds(True, "on") is True        # user's own choice respected
