import pytest
from src.playlist.mode_presets import (
    PACE_MODE_PRESETS,
    resolve_pace_mode,
)


def test_pace_mode_presets_has_four_modes():
    assert set(PACE_MODE_PRESETS.keys()) == {"strict", "narrow", "dynamic", "off"}


def test_off_disables_all_gates():
    settings = resolve_pace_mode("off")
    assert settings["admission_floor"] == 0.0
    assert settings["bridge_floor"] == 0.0
    assert settings["bpm_admission_max_log_distance"] == float("inf")
    assert settings["bpm_bridge_max_log_distance"] == float("inf")


def test_dynamic_is_middle_ground_not_disabled():
    settings = resolve_pace_mode("dynamic")
    # rhythm-cosine hard floors removed (now soft penalty); onset bands carry the gate
    assert settings["admission_floor"] == 0.0
    assert settings["bridge_floor"] == 0.0
    assert settings["onset_admission_max_log_distance"] < float("inf")
    assert settings["onset_bridge_max_log_distance"] < float("inf")
    assert settings["bpm_admission_max_log_distance"] < float("inf")
    assert settings["bpm_bridge_max_log_distance"] < float("inf")


def test_dynamic_catches_double_time():
    settings = resolve_pace_mode("dynamic")
    assert settings["bpm_admission_max_log_distance"] < 1.0
    assert settings["bpm_bridge_max_log_distance"] < 1.0


def test_pace_mode_monotonic_strict_to_off():
    modes = ["strict", "narrow", "dynamic", "off"]
    settings = [resolve_pace_mode(m) for m in modes]
    for i in range(len(modes) - 1):
        assert settings[i]["admission_floor"] >= settings[i + 1]["admission_floor"]
        assert settings[i]["bridge_floor"] >= settings[i + 1]["bridge_floor"]
        assert settings[i]["bpm_admission_max_log_distance"] <= settings[i + 1]["bpm_admission_max_log_distance"]
        assert settings[i]["bpm_bridge_max_log_distance"] <= settings[i + 1]["bpm_bridge_max_log_distance"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown pace mode"):
        resolve_pace_mode("turbo")


def test_overrides_apply():
    settings = resolve_pace_mode("narrow", {"admission_floor": 0.10})
    assert settings["admission_floor"] == 0.10
    assert settings["bridge_floor"] == PACE_MODE_PRESETS["narrow"]["bridge_floor"]
