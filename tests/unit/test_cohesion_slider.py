"""Tests for CohesionSlider widget."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from src.playlist_gui.widgets.cohesion_slider import (
    COHESION_MODE_LEVELS,
    CohesionSlider,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestCohesionSlider:
    def test_default_value_is_dynamic(self, qapp):
        slider = CohesionSlider()
        assert slider.get_cohesion_mode() == "dynamic"

    @pytest.mark.parametrize("mode", ["strict", "narrow", "dynamic", "discover"])
    def test_set_and_get_roundtrip(self, qapp, mode):
        slider = CohesionSlider()
        slider.set_cohesion_mode(mode)
        assert slider.get_cohesion_mode() == mode

    def test_invalid_value_is_ignored(self, qapp):
        slider = CohesionSlider()
        slider.set_cohesion_mode("dynamic")
        slider.set_cohesion_mode("not_a_mode")  # should silently ignore
        assert slider.get_cohesion_mode() == "dynamic"

    def test_signal_emitted_on_change(self, qapp):
        slider = CohesionSlider()
        received: list[str] = []
        slider.cohesion_mode_changed.connect(received.append)
        slider.set_cohesion_mode("strict")
        assert received == ["strict"]

    def test_signal_not_emitted_when_value_unchanged(self, qapp):
        slider = CohesionSlider()
        received: list[str] = []
        slider.cohesion_mode_changed.connect(received.append)
        slider.set_cohesion_mode("dynamic")  # already dynamic by default
        assert received == []

    def test_levels_ordered_strict_to_discover(self):
        # Slider position 0 = strict (leftmost, tightest)
        # Slider position 3 = discover (rightmost, loosest)
        assert COHESION_MODE_LEVELS == ["strict", "narrow", "dynamic", "discover"]
