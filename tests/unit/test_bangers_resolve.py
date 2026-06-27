from src.playlist_generator import (
    _resolve_popularity_rank_cutoff,
    _resolve_popular_seeds_mode,
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


def test_oops_forces_fire_regardless_of_user_mode():
    assert _resolve_popular_seeds_mode("off", "oops") == "fire"
    assert _resolve_popular_seeds_mode("on", "oops") == "fire"
    assert _resolve_popular_seeds_mode("fire", "oops") == "fire"


def test_non_oops_passes_through_user_mode():
    assert _resolve_popular_seeds_mode("off", "off") == "off"
    assert _resolve_popular_seeds_mode("on", "on") == "on"
    assert _resolve_popular_seeds_mode("fire", "on") == "fire"
    assert _resolve_popular_seeds_mode("", "off") == "off"
