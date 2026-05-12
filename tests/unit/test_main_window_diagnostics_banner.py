"""Tests for main-window diagnostics banner theming."""

from PySide6.QtWidgets import QFrame, QLabel

from src.playlist_gui.main_window import MainWindow


def test_diagnostics_banner_uses_theme_object_names(qtbot):
    window = MainWindow.__new__(MainWindow)
    window._banner_frame = QFrame()
    window._banner_time_label = QLabel()
    qtbot.addWidget(window._banner_frame)
    qtbot.addWidget(window._banner_time_label)

    MainWindow._style_diagnostics_banner(window)

    assert window._banner_frame.objectName() == "diagnosticsBanner"
    assert window._banner_frame.styleSheet() == ""
    assert window._banner_time_label.objectName() == "diagnosticsBannerTime"
    assert window._banner_time_label.styleSheet() == ""
