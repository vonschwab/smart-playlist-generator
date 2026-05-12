import logging

from PySide6.QtWidgets import QDialog

from src.playlist.request_models import LibraryPipelineRequest
from src.playlist_gui.main_window import MainWindow


class _FakeLogPanel:
    def __init__(self):
        self.entries = []

    def append_log(self, level, message, is_verbose=False):
        self.entries.append((level, message, is_verbose))


def test_worker_log_handler_preserves_worker_level():
    window = MainWindow.__new__(MainWindow)
    window._log_panel = _FakeLogPanel()
    window._logger = logging.getLogger("test.worker_logs")

    MainWindow._on_worker_log(window, "WARNING", "Analyze Library warning", None)

    assert window._log_panel.entries == [("WARNING", "Analyze Library warning", False)]


class _FakeProgressBar:
    def __init__(self):
        self.value = None

    def setValue(self, value):
        self.value = value


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


def test_worker_progress_does_not_emit_visible_info_log(caplog):
    window = MainWindow.__new__(MainWindow)
    window._progress_bar = _FakeProgressBar()
    window._stage_label = _FakeLabel()
    window._logger = logging.getLogger("test.worker_progress")

    caplog.set_level(logging.INFO, logger="test.worker_progress")

    MainWindow._on_worker_progress(window, "scan", 50, 100, "Scanning", None)

    assert window._progress_bar.value == 50
    assert window._stage_label.text == "Scanning"
    assert "worker progress" not in caplog.text


class _FakeConfigModel:
    def get_overrides(self):
        return {"library": {"database_path": "metadata.db"}}


class _FakeJobManager:
    def __init__(self):
        self.calls = []

    def enqueue_pipeline(self, config_path, overrides, request=None):
        self.calls.append((config_path, overrides, request))


class _FakeStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(message)


def test_run_pipeline_logs_human_readable_stage_labels(monkeypatch):
    import src.playlist_gui.main_window as main_window_mod

    class _AcceptedDialog:
        def __init__(self, parent=None):
            self.parent = parent

        def exec(self):
            return QDialog.Accepted

        def build_request(self, config_path, overrides):
            return LibraryPipelineRequest(
                config_path=config_path,
                overrides=overrides,
                stages=["sonic", "genre-sim", "artifacts"],
            )

    monkeypatch.setattr(
        main_window_mod,
        "AnalyzeLibraryOptionsDialog",
        _AcceptedDialog,
    )

    window = MainWindow.__new__(MainWindow)
    window._config_model = _FakeConfigModel()
    window._job_manager = _FakeJobManager()
    window._config_path = "config.yaml"
    window._log_panel = _FakeLogPanel()
    window._status_bar = _FakeStatusBar()

    MainWindow._on_run_pipeline(window)

    expected = "Update sonic features -> Build genre similarity -> Build DS artifacts"
    assert window._job_manager.calls
    assert window._log_panel.entries == [
        ("INFO", f"Queued Analyze Library: {expected}", False)
    ]
    assert window._status_bar.messages == [f"Analyze Library queued: {expected}"]
