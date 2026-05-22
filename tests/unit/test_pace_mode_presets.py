import pytest

from src.playlist.mode_presets import PACE_MODE_PRESETS, resolve_pace_mode


def test_pace_mode_presets_has_three_modes():
    assert set(PACE_MODE_PRESETS) == {"strict", "narrow", "dynamic"}


def test_dynamic_is_no_op():
    settings = resolve_pace_mode("dynamic")

    assert settings["admission_floor"] == 0.0
    assert settings["bridge_floor"] == 0.0


def test_strict_is_tightest():
    strict = resolve_pace_mode("strict")
    narrow = resolve_pace_mode("narrow")
    dynamic = resolve_pace_mode("dynamic")

    assert strict["admission_floor"] > narrow["admission_floor"] > dynamic["admission_floor"]
    assert strict["bridge_floor"] > narrow["bridge_floor"] > dynamic["bridge_floor"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown pace mode"):
        resolve_pace_mode("turbo")


def test_overrides_apply():
    settings = resolve_pace_mode("narrow", {"admission_floor": 0.10})

    assert settings["admission_floor"] == 0.10
    assert settings["bridge_floor"] == PACE_MODE_PRESETS["narrow"]["bridge_floor"]
