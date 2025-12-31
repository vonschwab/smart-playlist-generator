from PySide6.QtCore import QObject, Signal

from src.playlist_gui.jobs.job_manager import JobManager
from src.playlist_gui.jobs.job_types import JobType


class FakeWorkerClient(QObject):
    progress_received = Signal(str, int, int, str, object)
    error_received = Signal(str, str, object)
    done_received = Signal(str, bool, str, bool, object, str)
    busy_changed = Signal(bool)
    worker_stopped = Signal(int, str)

    def __init__(self):
        super().__init__()
        self._running = True

    def is_busy(self):
        return False

    def is_running(self):
        return self._running

    def scan_library(self, *args, **kwargs):
        return "req-scan"

    def update_genres(self, *args, **kwargs):
        return "req-genres"

    def update_sonic(self, *args, **kwargs):
        return "req-sonic"

    def build_artifacts(self, *args, **kwargs):
        return "req-art"

    def get_pid(self):
        return None

    def was_busy_on_last_exit(self):
        return False


def test_worker_crash_marks_active_failed_and_pending_skipped(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)

    job1 = mgr.enqueue_job(JobType.SCAN_LIBRARY, "config.yaml", {})
    job2 = mgr.enqueue_job(JobType.UPDATE_GENRES, "config.yaml", {})

    assert job1.status == "RUNNING"
    assert job2.status == "PENDING"

    fake.worker_stopped.emit(1, "crashed")

    assert job1.status == "FAILED"
    assert job2.status == "SKIPPED"


def test_retry_queue_reenqueues_skipped(qtbot):
    fake = FakeWorkerClient()
    mgr = JobManager(fake)
    mgr.enqueue_job(JobType.SCAN_LIBRARY, "config.yaml", {})
    mgr.enqueue_job(JobType.UPDATE_GENRES, "config.yaml", {})

    fake.worker_stopped.emit(1, "crashed")
    skipped = [j for j in mgr.jobs() if j.status == "SKIPPED"]
    assert skipped

    mgr.retry_skipped()
    new_pending = [j for j in mgr.jobs() if j.status == "PENDING" or j.status == "RUNNING"]
    # Should include re-enqueued jobs for each skipped
    assert len(new_pending) >= len(skipped)
    assert new_pending[-1].job_type == skipped[-1].job_type
