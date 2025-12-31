from PySide6.QtCore import QObject, Signal

from src.playlist_gui.jobs.job_manager import JobManager
from src.playlist_gui.jobs.job_model import Job, JobStatus, JobTableModel
from src.playlist_gui.jobs.job_types import JobType


class FakeWorkerClient(QObject):
    progress_received = Signal(str, int, int, str, object)
    error_received = Signal(str, str, object)
    done_received = Signal(str, bool, str, bool, object, str)
    busy_changed = Signal(bool)
    worker_stopped = Signal(int, str)

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
