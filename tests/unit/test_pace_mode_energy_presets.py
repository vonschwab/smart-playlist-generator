"""Tests for pace_mode energy preset configuration."""
from src.playlist.mode_presets import PACE_MODE_PRESETS
from src.playlist.pier_bridge.config import PierBridgeConfig

KEYS = ["energy_step_cap", "energy_step_strength", "energy_arc_band", "energy_arc_strength"]


def test_config_defaults_off():
    c = PierBridgeConfig()
    for k in KEYS:
        assert getattr(c, k) == 0.0


def test_presets_have_energy_keys_and_off_disables_arc():
    for mode in ("strict", "narrow", "dynamic", "off"):
        for k in KEYS:
            assert k in PACE_MODE_PRESETS[mode], f"{mode} missing {k}"
    # always-on step cap (anti-whiplash) even at off
    assert PACE_MODE_PRESETS["off"]["energy_step_strength"] > 0.0
    # off disables the arc term
    assert PACE_MODE_PRESETS["off"]["energy_arc_strength"] == 0.0
    # strict is the tightest step cap
    assert PACE_MODE_PRESETS["strict"]["energy_step_cap"] < PACE_MODE_PRESETS["off"]["energy_step_cap"]
