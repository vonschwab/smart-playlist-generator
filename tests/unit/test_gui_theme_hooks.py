"""Tests for shared QSS hooks replacing inline widget styles."""

from PySide6.QtWidgets import QLabel

from src.playlist_gui.blacklist_window import BlacklistWindow
from src.playlist_gui.main_window import MainWindow
from src.playlist_gui.ui_state import UIStateModel
from src.playlist_gui.widgets.generate_panel import GeneratePanel
from src.playlist_gui.widgets.mode_panels import ArtistModePanel
from src.playlist_gui.widgets.seed_chips import SeedChipsList


class _ConfigModel:
    def __init__(self, count=0):
        self._count = count

    def override_count(self):
        return self._count


def test_generate_panel_validation_banner_uses_theme(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    assert panel._validation_label.objectName() == "validationBanner"
    assert panel._validation_label.styleSheet() == ""


def test_artist_mode_note_uses_theme(qtbot):
    panel = ArtistModePanel()
    qtbot.addWidget(panel)

    assert panel._multi_note.objectName() == "modeInlineNote"
    assert panel._multi_note.styleSheet() == ""


def test_seed_chip_list_uses_theme_hooks(qtbot):
    widget = SeedChipsList()
    qtbot.addWidget(widget)

    assert widget._count_label.objectName() == "seedCountLabel"
    assert widget._count_label.styleSheet() == ""
    assert widget._clear_btn.objectName() == "compactActionButton"
    assert widget._clear_btn.styleSheet() == ""
    assert widget._info_label.objectName() == "seedInfoLabel"
    assert widget._info_label.styleSheet() == ""


def test_blacklist_header_uses_theme(monkeypatch, qtbot):
    monkeypatch.setattr(BlacklistWindow, "resize", lambda *args: None)
    window = BlacklistWindow(
        worker_client=None,
        config_path_provider=lambda: "config.yaml",
        overrides_provider=lambda: {},
    )
    qtbot.addWidget(window)

    header = window.findChild(QLabel, "dialogHeaderLabel")

    assert header is not None
    assert header.styleSheet() == ""


def test_override_status_no_preset(qtbot):
    """When no preset is active, the status label is cleared."""
    window = MainWindow.__new__(MainWindow)
    window._override_status_label = QLabel("Base config")
    window._active_preset_name = None
    window._preset_ui_state_snapshot = None
    window._generate_panel = None
    qtbot.addWidget(window._override_status_label)

    MainWindow._update_override_status(window)

    assert window._override_status_label.text() == ""
    assert window._override_status_label.property("state") == "none"
    assert window._override_status_label.styleSheet() == ""


def test_override_status_preset_clean(qtbot):
    """When a preset is active and unchanged, the label shows the preset name."""
    panel = GeneratePanel()
    qtbot.addWidget(panel)
    snapshot = panel.build_ui_state()

    window = MainWindow.__new__(MainWindow)
    window._override_status_label = QLabel()
    window._active_preset_name = "MyPreset"
    window._preset_ui_state_snapshot = snapshot
    window._generate_panel = panel
    qtbot.addWidget(window._override_status_label)

    MainWindow._update_override_status(window)

    assert window._override_status_label.text() == "Preset: MyPreset"
    assert window._override_status_label.property("state") == "preset"
    assert window._override_status_label.styleSheet() == ""


def test_override_status_preset_modified(qtbot):
    """When a preset is active but the state has changed, the label shows (modified)."""
    panel = GeneratePanel()
    qtbot.addWidget(panel)
    snapshot = panel.build_ui_state()
    # Mutate snapshot so it no longer matches the panel's live state
    from dataclasses import replace
    different_snapshot = replace(snapshot, track_count=snapshot.track_count + 1)

    window = MainWindow.__new__(MainWindow)
    window._override_status_label = QLabel()
    window._active_preset_name = "MyPreset"
    window._preset_ui_state_snapshot = different_snapshot
    window._generate_panel = panel
    qtbot.addWidget(window._override_status_label)

    MainWindow._update_override_status(window)

    assert window._override_status_label.text() == "Preset: MyPreset (modified)"
    assert window._override_status_label.property("state") == "preset_modified"
    assert window._override_status_label.styleSheet() == ""
