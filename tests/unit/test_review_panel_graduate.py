"""Test the graduate-to-YAML and CLI launcher buttons on ReviewPanel."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_graduate_button_runs_graduate_ai_and_graduate_reviewed(tmp_path, app):
    from src.playlist_gui.widgets.review_panel import ReviewPanel

    sidecar = tmp_path / "sidecar.db"
    panel = ReviewPanel(sidecar_db_path=str(sidecar))
    panel.show()

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.playlist_gui.widgets.review_panel.subprocess.run", return_value=completed) as mock_run:
        panel.graduate_button.click()

    commands = [call.args[0][2] for call in mock_run.call_args_list]
    assert "graduate-ai" in commands
    assert "graduate-reviewed" in commands


def test_cli_review_button_spawns_terminal_with_review_command(tmp_path, app):
    from src.playlist_gui.widgets.review_panel import ReviewPanel

    sidecar = tmp_path / "sidecar.db"
    panel = ReviewPanel(sidecar_db_path=str(sidecar))
    panel.show()

    with patch("src.playlist_gui.widgets.review_panel.subprocess.Popen") as mock_popen:
        panel.cli_review_button.click()

    assert mock_popen.call_count == 1
    argv = mock_popen.call_args.args[0]
    assert "review" in argv


def test_graduate_emits_signal_after_success(tmp_path, app):
    from src.playlist_gui.widgets.review_panel import ReviewPanel

    sidecar = tmp_path / "sidecar.db"
    panel = ReviewPanel(sidecar_db_path=str(sidecar))
    panel.show()

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    fired: list[bool] = []
    panel.vocab_graduated.connect(lambda: fired.append(True))

    with patch("src.playlist_gui.widgets.review_panel.subprocess.run", return_value=completed):
        panel.graduate_button.click()

    assert fired == [True]
