"""Unit tests for Phase 2 Task 4 pace-plumb fix: core._resolve_pace_overrides.

resolve_pace_mode() was called WITHOUT its `overrides` param at pipeline/core.py's
pace_settings call site (pre-existing dead-outlet, documented in
scripts/corridor_baseline/perturb.py's genuinely-dead-outlets comment), so no
yaml key could ever influence the per-mode pace band knobs (bpm_bridge_max_log_
distance etc. -- preset-only). These tests pin the pure helper that builds the
override dict now threaded into that call, and prove resolve_pace_mode's
resolved band actually changes when the override is applied.
"""
from __future__ import annotations

import src.playlist.pipeline.core as core
from src.playlist.mode_presets import resolve_pace_mode


def test_no_overrides_returns_empty_dict():
    assert core._resolve_pace_overrides(None, {}) == {}
    assert core._resolve_pace_overrides({}, {}) == {}


def test_candidate_pool_admission_side_keys_forwarded_with_rename():
    overrides = {
        "candidate_pool": {
            "pace_admission_floor": 0.12,
            "pace_bridge_floor": 0.34,
            "bpm_admission_max_log_distance": 0.55,
            "onset_admission_max_log_distance": 0.66,
        }
    }
    result = core._resolve_pace_overrides(overrides, {})
    assert result == {
        "admission_floor": 0.12,
        "bridge_floor": 0.34,
        "bpm_admission_max_log_distance": 0.55,
        "onset_admission_max_log_distance": 0.66,
    }


def test_pier_bridge_bridge_side_and_energy_keys_forwarded_verbatim():
    pb_overrides = {
        "bpm_bridge_max_log_distance": 0.71,
        "bpm_trust_min_onset_rate": 0.4,
        "onset_bridge_max_log_distance": 0.61,
        "bpm_bridge_soft_penalty_strength": 0.9,
        "onset_bridge_soft_penalty_strength": 0.8,
        "energy_step_cap": 2.0,
        "energy_step_strength": 0.25,
        "energy_arc_band": 0.6,
        "energy_arc_strength": 0.35,
        "pace_rescue_k_energy": 15,
    }
    result = core._resolve_pace_overrides(None, pb_overrides)
    assert result["bpm_bridge_max_log_distance"] == 0.71
    assert result["pace_rescue_k_energy"] == 15.0


def test_non_numeric_and_bool_values_ignored():
    overrides = {"candidate_pool": {"pace_admission_floor": "not-a-number"}}
    pb_overrides = {"bpm_bridge_max_log_distance": True}
    result = core._resolve_pace_overrides(overrides, pb_overrides)
    assert result == {}


def test_unrelated_pier_bridge_keys_do_not_leak_into_pace_overrides():
    # A totally unrelated pier_bridge.* knob (e.g. mini_pier_enabled) must never
    # show up in the pace override dict -- only the verified pace-preset keys.
    pb_overrides = {"mini_pier_enabled": True, "weight_bridge": 0.6}
    assert core._resolve_pace_overrides(None, pb_overrides) == {}


def test_override_changes_resolve_pace_mode_resolved_band():
    """The wiring test the brief asks for: an override actually flips the
    resolved value resolve_pace_mode returns, same contract as resolve_pace_
    mode's own documented overrides= behavior (test_pace_mode_presets.py::
    test_overrides_apply)."""
    overrides = {"pier_bridge": {"bpm_bridge_max_log_distance": 0.12345}}
    pb_overrides = overrides["pier_bridge"]
    pace_overrides = core._resolve_pace_overrides(overrides, pb_overrides)
    baseline = resolve_pace_mode("narrow")
    overridden = resolve_pace_mode("narrow", overrides=pace_overrides)
    assert baseline["bpm_bridge_max_log_distance"] != 0.12345
    assert overridden["bpm_bridge_max_log_distance"] == 0.12345
    # Untouched keys still come from the narrow preset, unaffected.
    assert overridden["bridge_floor"] == baseline["bridge_floor"]
