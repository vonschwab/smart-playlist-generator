"""Tests for pace_mode energy preset configuration.

Pins the calibrated values shipped by Task 5 (2026-06-20).
Worst-edge-sonic gate: on >= off - 0.05; wall < 90s.

Calibration results:
  strict:  k=20 PASS (2 feasible seed sets; HIGH_AROUSAL infeasible at strict BPM bands)
  narrow:  k=5 PASS (k=10 FAIL: WIDE_SWING wes dropped -0.064, beyond -0.05 gate)
  dynamic: k=0 (pool not starved at dynamic BPM bands; arc strengths from task brief)
  off:     k=0 (BPM gate off; no rescue needed; arc disabled)
"""
from src.playlist.mode_presets import PACE_MODE_PRESETS
from src.playlist.pier_bridge.config import PierBridgeConfig

KEYS = ["energy_step_cap", "energy_step_strength", "energy_arc_band", "energy_arc_strength"]


def test_config_defaults_off():
    c = PierBridgeConfig()
    for k in KEYS:
        assert getattr(c, k) == 0.0


def test_presets_have_energy_keys():
    # All energy keys must be present in all modes.
    for mode in ("strict", "narrow", "dynamic", "off"):
        for k in KEYS:
            assert k in PACE_MODE_PRESETS[mode], f"{mode} missing {k}"
        assert "pace_rescue_k_energy" in PACE_MODE_PRESETS[mode], \
            f"{mode} missing pace_rescue_k_energy"


def test_strict_energy_values_calibrated():
    """strict: k=20 PASS (Task 5 calibration 2026-06-20)."""
    p = PACE_MODE_PRESETS["strict"]
    assert p["pace_rescue_k_energy"] == 20
    assert p["energy_arc_band"] == 0.5
    assert p["energy_arc_strength"] == 0.3
    assert p["energy_step_cap"] == 1.0
    assert p["energy_step_strength"] == 0.2


def test_narrow_energy_values_calibrated():
    """narrow: k=5 PASS (k=10 FAIL on WIDE_SWING; Task 5 calibration 2026-06-20)."""
    p = PACE_MODE_PRESETS["narrow"]
    assert p["pace_rescue_k_energy"] == 5
    assert p["energy_arc_band"] == 0.4
    assert p["energy_arc_strength"] == 0.2
    assert p["energy_step_cap"] == 1.5
    assert p["energy_step_strength"] == 0.15


def test_dynamic_energy_values():
    """dynamic: all arc values 0.0 (unevaluated; reverted pending full eval-gate run)."""
    p = PACE_MODE_PRESETS["dynamic"]
    assert p["pace_rescue_k_energy"] == 0
    assert p["energy_arc_band"] == 0.0
    assert p["energy_arc_strength"] == 0.0
    assert p["energy_step_cap"] == 0.0
    assert p["energy_step_strength"] == 0.0


def test_off_energy_values_disabled():
    """off: all energy disabled (pace=off means no pace constraint at all)."""
    p = PACE_MODE_PRESETS["off"]
    assert p["pace_rescue_k_energy"] == 0
    for k in KEYS:
        assert p[k] == 0.0, f"off.{k} must be 0.0 (pace disabled)"
