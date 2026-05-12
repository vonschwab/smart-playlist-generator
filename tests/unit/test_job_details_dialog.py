"""Tests for job details dialog result presentation."""

from PySide6.QtWidgets import QLabel

from src.playlist_gui.jobs.job_model import Job, JobStatus
from src.playlist_gui.jobs.job_types import JobType
from src.playlist_gui.widgets.job_details_dialog import JobDetailsDialog


def test_job_details_dialog_shows_analyze_library_results_tab(qtbot):
    job = Job(
        job_id="job-1",
        job_type=JobType.ANALYZE_LIBRARY,
        status=JobStatus.SUCCESS,
        summary="2 stages: 1 ran, 1 skipped, 0 verify issues, 1.5s",
        result_data={
            "report_path": "data/artifacts/beat3tower_32k/analyze_run_report.json",
            "out_dir": "data/artifacts/beat3tower_32k",
            "verify_issues": [],
            "stages": [
                {
                    "name": "scan",
                    "decision": "ran",
                    "processed_count": 20,
                    "errors_count": 0,
                    "duration_sec": 1.2,
                },
                {
                    "name": "genres",
                    "decision": "skipped",
                    "processed_count": 0,
                    "errors_count": 0,
                    "duration_sec": 0.1,
                },
            ],
        },
    )

    dialog = JobDetailsDialog(job)
    qtbot.addWidget(dialog)

    tabs = dialog.findChild(type(dialog._tabs))
    tab_names = [tabs.tabText(index) for index in range(tabs.count())]

    assert dialog.objectName() == "jobDetailsDialog"
    assert dialog._summary_group.objectName() == "jobSummaryCard"
    assert dialog._tabs.objectName() == "jobDetailsTabs"
    assert dialog._results_headline_label.objectName() == "analyzeResultsHeadline"
    assert dialog._results_table.objectName() == "analyzeStageTable"
    assert dialog._results_table.showGrid() is False
    assert "Results" in tab_names
    assert dialog._results_table.rowCount() == 2
    assert dialog._results_table.item(0, 0).text() == "Scan library"
    assert dialog._results_table.item(0, 1).text() == "ran"
    assert dialog._readout_table.item(0, 0).text() == "Status"
    assert dialog._readout_table.item(0, 1).text() == "Success"


def test_job_details_dialog_shows_analyze_library_failure_readout(qtbot):
    job = Job(
        job_id="job-2",
        job_type=JobType.ANALYZE_LIBRARY,
        status=JobStatus.FAILED,
        summary="Analyze Library failed",
        error_message="Worker exited unexpectedly",
        traceback="Traceback detail",
    )

    dialog = JobDetailsDialog(job)
    qtbot.addWidget(dialog)

    tab_names = [dialog._tabs.tabText(index) for index in range(dialog._tabs.count())]

    assert "Results" in tab_names
    assert "Error Details" in tab_names
    assert (
        dialog._attention_summary_label.text()
        == "Needs Attention: Worker exited unexpectedly"
    )
    assert dialog._attention_summary_label.objectName() == "analyzeAttentionBanner"
    assert dialog._readout_table.item(0, 0).text() == "Status"
    assert dialog._readout_table.item(0, 1).text() == "Failed"
    assert "Worker exited unexpectedly" in dialog._attention_text.toPlainText()


def test_job_details_dialog_exposes_selectable_analyze_library_paths(qtbot):
    job = Job(
        job_id="job-3",
        job_type=JobType.ANALYZE_LIBRARY,
        status=JobStatus.SUCCESS,
        result_data={
            "report_path": "data/artifacts/beat3tower_32k/analyze_run_report.json",
            "out_dir": "data/artifacts/beat3tower_32k",
            "stages": [],
        },
    )

    dialog = JobDetailsDialog(job)
    qtbot.addWidget(dialog)

    assert dialog._report_path_edit.isReadOnly() is True
    assert (
        dialog._report_path_edit.text()
        == "data/artifacts/beat3tower_32k/analyze_run_report.json"
    )
    assert dialog._report_path_copy_btn.text() == "Copy"
    assert dialog._report_path_open_btn.text() == "Open"
    assert dialog._output_path_edit.isReadOnly() is True
    assert dialog._output_path_edit.text() == "data/artifacts/beat3tower_32k"


def test_job_details_dialog_disables_missing_analyze_library_paths(qtbot):
    job = Job(
        job_id="job-4",
        job_type=JobType.ANALYZE_LIBRARY,
        status=JobStatus.FAILED,
        summary="Analyze Library failed",
        error_message="Worker exited unexpectedly",
        result_data={},
    )

    dialog = JobDetailsDialog(job)
    qtbot.addWidget(dialog)

    assert dialog._report_path_edit.text() == "Not available"
    assert dialog._report_path_copy_btn.isEnabled() is False
    assert dialog._report_path_open_btn.isEnabled() is False
    assert dialog._output_path_edit.text() == "Not available"
    assert dialog._output_path_copy_btn.isEnabled() is False
    assert dialog._output_path_open_btn.isEnabled() is False


def test_job_details_resume_notice_uses_theme(qtbot):
    job = Job(
        job_id="job-5",
        job_type=JobType.BUILD_ARTIFACTS,
        status=JobStatus.FAILED,
        can_resume=True,
    )

    dialog = JobDetailsDialog(job)
    qtbot.addWidget(dialog)

    resume_label = dialog.findChild(QLabel, "resumeNotice")
    assert resume_label is not None
    assert resume_label.styleSheet() == ""
