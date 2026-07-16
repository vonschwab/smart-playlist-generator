"""Light unit coverage for the one pure piece of the knob-sweep orchestration:
the dead-outlets enabling-parent-flag annotation. Everything else in
capture_knob_sweep.py is engine-calling orchestration, exercised by the
smoke run (see task-5-report.md), not unit tests."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cb_knob_sweep", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "capture_knob_sweep.py")
knob_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(knob_sweep)


def test_finds_family_matched_disabled_flag():
    ref_flat = {
        "playlist.pier_config.dj_bridging_enabled": False,
        "playlist.pier_config.dj_genre_coverage_weight": 0.15,
        "playlist.pier_config.genre_steering_enabled": True,
    }
    assert (knob_sweep.find_enabling_parent("dj_genre_coverage_weight", ref_flat)
            == "playlist.pier_config.dj_bridging_enabled")


def test_prefers_longest_common_prefix_among_multiple_disabled_families():
    ref_flat = {
        "candidate_pool.genre_compatibility_enabled": False,
        "playlist.pier_config.genre_steering_enabled": False,
        "candidate_pool.genre_compatibility_penalty_strength": 0.5,
    }
    assert (knob_sweep.find_enabling_parent("genre_compatibility_penalty_strength", ref_flat)
            == "candidate_pool.genre_compatibility_enabled")


def test_no_match_when_no_disabled_sibling():
    ref_flat = {"playlist.pier_config.dj_bridging_enabled": True}
    assert knob_sweep.find_enabling_parent("dj_genre_coverage_weight", ref_flat) is None


def test_flag_itself_has_no_parent():
    ref_flat = {"playlist.pier_config.dj_bridging_enabled": False}
    assert knob_sweep.find_enabling_parent("dj_bridging_enabled", ref_flat) is None
