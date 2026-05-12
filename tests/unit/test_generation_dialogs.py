"""Tests for generation result dialogs."""

from PySide6.QtWidgets import QMessageBox

from src.playlist_gui.main_window import MainWindow


def test_generation_failure_dialog_has_actionable_details(qtbot):
    window = MainWindow.__new__(MainWindow)

    dialog = MainWindow._build_generation_failure_dialog(window, "No candidates found")
    qtbot.addWidget(dialog)

    assert dialog.objectName() == "generationFailureDialog"
    assert dialog.icon() == QMessageBox.Warning
    assert dialog.windowTitle() == "Generation Failed"
    assert dialog.text() == "Playlist generation failed:\nNo candidates found"
    assert "Build Artifacts" in dialog.detailedText()
    assert "Dynamic" in dialog.detailedText()


def test_generation_incomplete_dialog_has_actionable_details(qtbot):
    window = MainWindow.__new__(MainWindow)

    dialog = MainWindow._build_generation_incomplete_dialog(window, actual=18, requested=30)
    qtbot.addWidget(dialog)

    assert dialog.objectName() == "generationIncompleteDialog"
    assert dialog.icon() == QMessageBox.Information
    assert dialog.windowTitle() == "Generation Incomplete"
    assert "Generated 18 tracks instead of 30 requested." in dialog.text()
    assert "Reduce track count (currently 30)" in dialog.detailedText()
    assert "Relax recency filter" in dialog.detailedText()
