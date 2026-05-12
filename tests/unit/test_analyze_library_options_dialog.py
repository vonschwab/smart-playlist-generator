from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER
from src.playlist_gui.widgets.analyze_library_options_dialog import AnalyzeLibraryOptionsDialog


def test_analyze_library_options_dialog_defaults_to_full_analyze(qtbot):
    dialog = AnalyzeLibraryOptionsDialog()
    qtbot.addWidget(dialog)

    request = dialog.build_request("config.yaml", {"library": {"database_path": "metadata.db"}})

    assert dialog.objectName() == "analyzeLibraryOptionsDialog"
    assert dialog._title_label.objectName() == "dialogTitle"
    assert dialog._summary_label.objectName() == "dialogSummary"
    assert dialog._preset_frame.objectName() == "dialogControlFrame"
    assert dialog._preview_label.objectName() == "dialogPreviewCard"
    assert dialog._force_check.objectName() == "dialogOptionToggle"
    assert dialog._dry_run_check.objectName() == "dialogOptionToggle"
    assert request.config_path == "config.yaml"
    assert request.overrides == {"library": {"database_path": "metadata.db"}}
    assert request.stages == list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert request.force is False
    assert request.dry_run is False
    assert dialog._stages_group.isVisibleTo(dialog) is False
    assert "Will run: Scan library -> Update genres" in dialog._preview_label.text()


def test_analyze_library_options_dialog_builds_custom_request(qtbot):
    dialog = AnalyzeLibraryOptionsDialog()
    qtbot.addWidget(dialog)

    dialog.set_preset_for_testing("Custom")
    dialog.set_stage_checked_for_testing("scan", False)
    dialog.set_stage_checked_for_testing("genres", False)
    dialog.set_stage_checked_for_testing("discogs", False)
    dialog.set_stage_checked_for_testing("verify", False)
    dialog.set_force_for_testing(True)
    dialog.set_dry_run_for_testing(True)

    request = dialog.build_request("config.yaml", {})

    assert request.stages == ["sonic", "genre-sim", "artifacts"]
    assert request.force is True
    assert request.dry_run is True
    assert dialog._stages_group.isVisibleTo(dialog) is True
    assert "Will run: Update sonic features -> Build genre similarity -> Build DS artifacts" in dialog._preview_label.text()
    assert "Options: force rebuild, dry run" in dialog._preview_label.text()


def test_analyze_library_options_dialog_preview_updates_for_quick_verify(qtbot):
    dialog = AnalyzeLibraryOptionsDialog()
    qtbot.addWidget(dialog)

    dialog.set_preset_for_testing("Quick Verify")

    assert dialog._stages_group.isVisibleTo(dialog) is False
    assert dialog.selected_stages() == ["verify"]
    assert dialog._preview_label.text() == "Will run: Verify outputs"
