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


def test_presets_have_rescue_k_default_zero():
    # pace_rescue_k_energy must be PRESENT in all pace modes and default to 0
    # (no-op until calibrated; Task 3).
    for mode in ("strict", "narrow", "dynamic", "off"):
        assert "pace_rescue_k_energy" in PACE_MODE_PRESETS[mode], \
            f"{mode} missing pace_rescue_k_energy key"
        assert PACE_MODE_PRESETS[mode]["pace_rescue_k_energy"] == 0, \
            f"{mode}.pace_rescue_k_energy must default 0 (no-op until calibrated)"
