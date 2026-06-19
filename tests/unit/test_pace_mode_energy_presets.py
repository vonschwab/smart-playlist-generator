"""Tests for pace_mode energy preset configuration."""
from src.playlist.mode_presets import PACE_MODE_PRESETS
from src.playlist.pier_bridge.config import PierBridgeConfig

KEYS = ["energy_step_cap", "energy_step_strength", "energy_arc_band", "energy_arc_strength"]


def test_config_defaults_off():
    c = PierBridgeConfig()
    for k in KEYS:
        assert getattr(c, k) == 0.0


def test_presets_have_energy_keys_all_zero_by_default():
    # Keys must exist in all modes; all values are 0.0 (feature off, opt-in via config.yaml).
    for mode in ("strict", "narrow", "dynamic", "off"):
        for k in KEYS:
            assert k in PACE_MODE_PRESETS[mode], f"{mode} missing {k}"
            assert PACE_MODE_PRESETS[mode][k] == 0.0, f"{mode}.{k} must be 0.0 until calibrated"
