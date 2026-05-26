"""Tests for MainWindow generation state ownership."""

from src.playlist_gui.main_window import MainWindow
from src.playlist_gui.ui_state import UIStateModel


class _FakeGeneratePanel:
    def __init__(self, state=None):
        self._state = state or UIStateModel()
        self.applied = []

    def build_ui_state(self):
        return self._state


def test_debug_report_uses_generate_panel_state_not_hidden_widgets():
    window = MainWindow.__new__(MainWindow)
    window._generate_panel = _FakeGeneratePanel(
        UIStateModel(mode="artist", artist_queries=["Slowdive"])
    )
    window._active_preset_name = None
    window._config_path = "config.yaml"
    window._worker_client = None
    window._job_manager = None
    window._last_diagnostics = []
    window._log_path = None

    args = MainWindow._collect_debug_report_args(window)

    assert args["mode"] == "artist"
    assert args["artist"] == "Slowdive"
