"""Tests for shared QSS hooks replacing inline widget styles."""

from PySide6.QtWidgets import QLabel

from src.playlist_gui.blacklist_window import BlacklistWindow
from src.playlist_gui.main_window import MainWindow
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


def test_override_status_uses_state_property(qtbot):
    window = MainWindow.__new__(MainWindow)
    window._override_status_label = QLabel("Base config")
    window._config_model = _ConfigModel(count=2)
    window._active_preset_name = None
    window._dirty_overrides = False
    qtbot.addWidget(window._override_status_label)

    MainWindow._update_override_status(window)

    assert window._override_status_label.objectName() == "overrideStatusLabel"
    assert window._override_status_label.property("state") == "overrides"
    assert window._override_status_label.styleSheet() == ""
