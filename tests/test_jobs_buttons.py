from PySide6.QtCore import QObject, Signal

from src.playlist_gui.jobs.job_manager import JobManager
from src.playlist_gui.jobs.job_model import Job, JobStatus, JobTableModel
from src.playlist_gui.jobs.job_types import JobType
from src.playlist_gui.widgets.jobs_panel import JobsPanel


class FakeWorkerClient(QObject):
    progress_received = Signal(str, int, int, str, object)
    result_received = Signal(str, dict, object)
    error_received = Signal(str, str, object)
    done_received = Signal(str, bool, str, bool, object, str)
    busy_changed = Signal(bool)
    worker_stopped = Signal(int, str)
    checkpoint_received = Signal(dict, object)

    def __init__(self):
        super().__init__()
        self.cancel_request_id = None

    def is_busy(self):
        return False

    def is_running(self):
        return True

    def scan_library(self, *args, **kwargs):
        return "req-scan"

    def update_genres(self, *args, **kwargs):
        return "req-genres"

    def update_sonic(self, *args, **kwargs):
        return "req-sonic"

    def build_artifacts(self, *args, **kwargs):
        return "req-art"

    def analyze_library(self, *args, **kwargs):
        return "req-analyze"

    def cancel(self, request_id=None):
        self.cancel_request_id = request_id
        return True

    def get_pid(self):
        return None

    def was_busy_on_last_exit(self):
        return False


def test_pending_count_excludes_skipped():
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    mgr._jobs = [
        Job(job_id="a", job_type=JobType.SCAN_LIBRARY, status=JobStatus.PENDING),
        Job(job_id="b", job_type=JobType.SCAN_LIBRARY, status=JobStatus.SKIPPED),
        Job(job_id="c", job_type=JobType.SCAN_LIBRARY, status=JobStatus.SUCCESS),
    ]
    assert mgr.pending_count() == 1


def test_clear_pending_removes_queue_items():
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    mgr._jobs = [
        Job(job_id="a", job_type=JobType.SCAN_LIBRARY, status=JobStatus.PENDING),
        Job(job_id="b", job_type=JobType.UPDATE_GENRES, status=JobStatus.PENDING),
    ]
    mgr.cancel_pending()
    assert mgr.pending_count() == 0
    assert not any(j.status == JobStatus.PENDING for j in mgr.jobs())
    model = JobTableModel(mgr.jobs())
    assert model.rowCount() == 0  # queue filter default shows none after clear


def test_cancel_active_sends_worker_cancel():
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    job = Job(job_id="x", job_type=JobType.SCAN_LIBRARY, status=JobStatus.RUNNING, request_id="req123")
    mgr._jobs = [job]
    mgr._active_job_id = job.job_id

    mgr.cancel_active_job()

    assert fake.cancel_request_id == "req123"
    assert job.status == JobStatus.CANCELLING


def test_cancel_no_active_no_worker_call():
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    mgr.cancel_active_job()
    assert fake.cancel_request_id is None


def test_jobs_panel_disables_run_all_when_queue_active(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    panel = JobsPanel(mgr)
    qtbot.addWidget(panel)

    assert panel._run_all_btn.isEnabled() is True

    mgr.enqueue_pipeline("config.yaml", {})

    assert panel._run_all_btn.isEnabled() is False
    assert "Active: Analyze Library" in panel._status_label.text()
    assert "Pending: 0" in panel._status_label.text()


def test_jobs_panel_uses_analyze_library_language(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    panel = JobsPanel(mgr)
    qtbot.addWidget(panel)

    assert panel._run_all_btn.text() == "Analyze Library"
    assert "Analyze Library" in panel._run_all_btn.toolTip()
    assert panel._latest_details_btn.text() == "Latest Details"
    assert "Analyze Library results" in panel._latest_details_btn.toolTip()
    assert panel._latest_details_btn.isEnabled() is False


def test_jobs_panel_uses_polished_queue_surface(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    panel = JobsPanel(mgr)
    qtbot.addWidget(panel)

    assert panel._toolbar_frame.objectName() == "jobsToolbar"
    assert panel._status_label.objectName() == "jobsStatusLabel"
    assert panel._latest_summary_label.objectName() == "jobsLatestSummary"
    assert panel._table.objectName() == "jobsTable"
    assert panel._table.showGrid() is False
    assert panel._table.verticalHeader().defaultSectionSize() == 28
    assert panel._run_all_btn.objectName() == "jobsPrimaryButton"


def test_job_manager_stores_analyze_library_result_summary(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)

    job = mgr.enqueue_pipeline("config.yaml", {})[0]

    fake.result_received.emit(
        "analyze_library",
        {
            "summary": "7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s",
            "stages": [{"name": "scan", "decision": "ran"}],
        },
        job.job_id,
    )

    assert job.result_data["summary"] == "7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s"
    assert job.summary == "7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s"


def test_job_table_model_formats_analyze_library_stage_ids():
    model = JobTableModel(
        [
            Job(
                job_id="analyze-running",
                job_type=JobType.ANALYZE_LIBRARY,
                status=JobStatus.RUNNING,
                stage="genre-sim",
            )
        ]
    )

    assert model.data(model.index(0, 3)) == "Build genre similarity"


def test_job_table_model_surfaces_analyze_library_attention_in_summary():
    model = JobTableModel(
        [
            Job(
                job_id="analyze-success",
                job_type=JobType.ANALYZE_LIBRARY,
                status=JobStatus.SUCCESS,
                summary="7 stages: 3 ran, 4 skipped, 2 verify issues, 31.2s",
                result_data={
                    "verify_issues": ["missing_manifest", "stale_artifact"],
                    "errors_count": 0,
                },
            )
        ]
    )
    model.set_filter_mode("history")

    assert (
        model.data(model.index(0, 6))
        == "Needs Attention: 2 verify issues - missing_manifest; stale_artifact"
    )


def test_job_table_model_surfaces_analyze_library_failure_in_summary():
    model = JobTableModel(
        [
            Job(
                job_id="analyze-failed",
                job_type=JobType.ANALYZE_LIBRARY,
                status=JobStatus.FAILED,
                summary="Analyze Library failed",
                error_message="Worker exited unexpectedly",
            )
        ]
    )
    model.set_filter_mode("history")

    assert (
        model.data(model.index(0, 6))
        == "Needs Attention: Worker exited unexpectedly"
    )


def test_jobs_panel_exposes_latest_completed_details_under_queue_filter(qtbot, monkeypatch):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    panel = JobsPanel(mgr)
    qtbot.addWidget(panel)
    opened = []

    monkeypatch.setattr(panel, "_open_job_details", lambda job: opened.append(job.job_id))

    job = mgr.enqueue_pipeline("config.yaml", {})[0]
    fake.result_received.emit(
        "analyze_library",
        {"summary": "7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s"},
        job.job_id,
    )
    fake.done_received.emit(
        "analyze_library",
        True,
        "Analyze Library complete",
        False,
        job.job_id,
        job.summary,
    )

    assert panel._view_filter.currentText() == "Queue"
    assert panel._model.rowCount() == 0
    assert panel._latest_details_btn.isEnabled() is True
    assert panel._latest_completed_job().job_id == job.job_id

    panel._latest_details_btn.click()

    assert opened == [job.job_id]


def test_jobs_panel_shows_latest_analyze_library_summary_under_queue_filter(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    panel = JobsPanel(mgr)
    qtbot.addWidget(panel)

    job = mgr.enqueue_pipeline("config.yaml", {})[0]
    fake.result_received.emit(
        "analyze_library",
        {"summary": "7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s"},
        job.job_id,
    )
    fake.done_received.emit(
        "analyze_library",
        True,
        "Analyze Library complete",
        False,
        job.job_id,
        job.summary,
    )

    assert panel._view_filter.currentText() == "Queue"
    assert panel._model.rowCount() == 0
    assert panel._latest_summary_label.isVisibleTo(panel) is True
    assert (
        panel._latest_summary_label.text()
        == "Latest: Analyze Library - 7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s"
    )


def test_jobs_panel_surfaces_latest_analyze_library_failure_attention(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    panel = JobsPanel(mgr)
    qtbot.addWidget(panel)

    job = mgr.enqueue_pipeline("config.yaml", {})[0]
    fake.error_received.emit("Worker exited unexpectedly", "Traceback detail", job.job_id)
    fake.done_received.emit(
        "analyze_library",
        False,
        "Analyze Library failed",
        False,
        job.job_id,
        "Analyze Library failed",
    )

    assert panel._latest_summary_label.isVisibleTo(panel) is True
    assert (
        panel._latest_summary_label.text()
        == "Latest: Analyze Library - Needs Attention: Worker exited unexpectedly"
    )
