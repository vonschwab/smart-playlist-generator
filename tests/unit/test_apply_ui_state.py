"""Tests for programmatic UI state restoration."""

import pytest

from src.playlist_gui.widgets.generate_panel import GeneratePanel


def test_artist_panel_set_presence(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._artist_panel.set_presence("very_high")
    assert panel._artist_panel.get_presence() == "very_high"

    panel._artist_panel.set_presence("very_low")
    assert panel._artist_panel.get_presence() == "very_low"


def test_artist_panel_set_variety(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._artist_panel.set_variety("sprawling")
    assert panel._artist_panel.get_variety() == "sprawling"

    panel._artist_panel.set_variety("focused")
    assert panel._artist_panel.get_variety() == "focused"


def test_seeds_panel_set_auto_order(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._seeds_panel.set_auto_order(False)
    assert panel._seeds_panel.get_auto_order() is False

    panel._seeds_panel.set_auto_order(True)
    assert panel._seeds_panel.get_auto_order() is True
