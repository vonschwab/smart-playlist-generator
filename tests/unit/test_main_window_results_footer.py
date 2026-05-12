"""Tests for the main-window results footer."""

from PySide6.QtWidgets import QLabel, QPushButton

from src.playlist_gui.main_window import MainWindow


def _window_with_footer_widgets(qtbot):
    window = MainWindow.__new__(MainWindow)
    window._results_summary_label = QLabel()
    window._results_summary_detail = QLabel()
    window._export_local_btn = QPushButton("Export to Local (M3U8)")
    window._export_plex_btn = QPushButton("Export to Plex")
    for widget in [
        window._results_summary_label,
        window._results_summary_detail,
        window._export_local_btn,
        window._export_plex_btn,
    ]:
        qtbot.addWidget(widget)
    return window


def test_results_footer_disables_export_until_playlist_exists(qtbot):
    window = _window_with_footer_widgets(qtbot)

    MainWindow._update_results_footer(window, playlist_name="", tracks=[])

    assert window._results_summary_label.text() == "No playlist loaded"
    assert window._results_summary_detail.text() == "Generate a playlist to enable export."
    assert window._export_local_btn.isEnabled() is False
    assert window._export_plex_btn.isEnabled() is False
    assert window._export_local_btn.toolTip() == "Generate a playlist before exporting."
    assert window._export_plex_btn.toolTip() == "Generate a playlist before exporting."


def test_results_footer_enables_export_with_playlist_summary(qtbot):
    window = _window_with_footer_widgets(qtbot)
    tracks = [{"title": "One"}, {"title": "Two"}]

    MainWindow._update_results_footer(window, playlist_name="Late Night", tracks=tracks)

    assert window._results_summary_label.text() == "Late Night"
    assert window._results_summary_detail.text() == "2 tracks ready to export."
    assert window._export_local_btn.isEnabled() is True
    assert window._export_plex_btn.isEnabled() is True
    assert window._export_local_btn.toolTip() == "Export 2 tracks to an M3U8 file."
    assert window._export_plex_btn.toolTip() == "Export 2 tracks to Plex."
