"""Tests for export dialog presentation and state."""

from src.playlist_gui.widgets.export_dialog import ExportLocalDialog, ExportPlexDialog


def test_export_local_dialog_uses_stylable_dark_theme_sections(qtbot, tmp_path):
    dialog = ExportLocalDialog(
        default_name="Late Night",
        default_directory=str(tmp_path),
        artist_name="",
    )
    qtbot.addWidget(dialog)

    assert dialog.objectName() == "exportLocalDialog"
    assert dialog._name_group.objectName() == "dialogSection"
    assert dialog._directory_group.objectName() == "dialogSection"
    assert dialog._preview_group.objectName() == "dialogSection"
    assert dialog._preview_label.objectName() == "dialogPreviewCard"
    assert dialog._browse_btn.objectName() == "dialogSecondaryButton"
    assert dialog._export_btn.objectName() == "dialogPrimaryButton"
    assert dialog._export_btn.styleSheet() == ""


def test_export_plex_dialog_uses_stylable_warning_and_actions(qtbot):
    dialog = ExportPlexDialog(
        default_name="Late Night",
        artist_name="",
        plex_configured=False,
    )
    qtbot.addWidget(dialog)

    assert dialog.objectName() == "exportPlexDialog"
    assert dialog._name_group.objectName() == "dialogSection"
    assert dialog._warning_label.objectName() == "dialogWarningBanner"
    assert dialog._info_label.objectName() == "dialogSummary"
    assert dialog._export_btn.objectName() == "exportPlexButton"
    assert dialog._export_btn.styleSheet() == ""
    assert dialog._export_btn.isEnabled() is False
