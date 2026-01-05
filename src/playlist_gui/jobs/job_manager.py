"""
JobManager coordinates queued jobs and worker communication.
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal, Slot

from ..worker_client import WorkerClient
from .job_model import Job, JobStatus
from .job_store import JobStore
from .job_types import JobType


class JobManager(QObject):
    """Queue and track library jobs."""

    job_added = Signal(object)  # Job
    job_updated = Signal(object)  # Job
    active_changed = Signal(object)  # Job or None
    queue_changed = Signal()

    def __init__(self, worker_client: WorkerClient, job_store: Optional[JobStore] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._worker_client = worker_client
        self._store = job_store or JobStore()
        self._jobs: List[Job] = self._store.load_history()
        self._active_job_id: Optional[str] = None
        self._crashed = False
        self._auto_start_blocked = False
        self._logger = logging.getLogger("playlist_gui.jobs")

        # Clean up any orphaned RUNNING jobs from a previous session
        for job in self._jobs:
            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(timezone.utc)
                if not job.summary:
                    job.summary = "Interrupted - worker restarted"

        # Connect worker signals
        self._worker_client.progress_received.connect(self._on_progress_received)
        self._worker_client.error_received.connect(self._on_error_received)
        self._worker_client.done_received.connect(self._on_done_received)
        self._worker_client.busy_changed.connect(self._on_busy_changed)
        self._worker_client.worker_stopped.connect(self._on_worker_stopped)

    # Public API ---------------------------------------------------------
    def jobs(self) -> List[Job]:
        return list(self._jobs)

    def active_job(self) -> Optional[Job]:
        return self._find_job(self._active_job_id)

    def enqueue_job(self, job_type: JobType, base_config_path: str, overrides: Optional[Dict] = None) -> Job:
        job = Job(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            status=JobStatus.PENDING,
            base_config_path=base_config_path,
        )
        job.overrides = dict(overrides or {})

        self._jobs.append(job)
        self.job_added.emit(job)
        self.queue_changed.emit()
        self._auto_start_blocked = False
        self._start_next_job()
        return job

    def enqueue_pipeline(self, base_config_path: str, overrides: Optional[Dict] = None) -> List[Job]:
        """Enqueue the default pipeline (scan -> genres -> sonic -> artifacts)."""
        created: List[Job] = []
        for job_type in JobType.ordered_pipeline():
            created.append(self.enqueue_job(job_type, base_config_path, overrides))
        return created

    def cancel_active_job(self) -> None:
        """Cancel the currently running job (if any)."""
        if not self._active_job_id:
            self._logger.info("Cancel requested but no active job")
            return
        active = self._find_job(self._active_job_id)
        if not active:
            return
        request_id = active.request_id if active else None
        self._logger.info("Cancel requested for job %s (%s)", active.job_id, request_id)
        active.status = JobStatus.CANCELLING
        active.stage = "Cancelling"
        active.summary = active.summary or "Cancellation requested"
        self.job_updated.emit(active)
        if not self._worker_client.cancel(request_id=request_id):
            active.status = JobStatus.FAILED
            active.finished_at = datetime.now(timezone.utc)
            active.summary = "Cancel request failed"
            self.job_updated.emit(active)
            self._store.save_history(self._jobs)
            self._active_job_id = None
            self.active_changed.emit(None)
            self.queue_changed.emit()

    def cancel_pending(self) -> None:
        """Mark pending/unfinished jobs as skipped without starting new work."""
        self._auto_start_blocked = True
        updated = False
        removed_ids: List[str] = []
        for job in list(self._jobs):
            if job.status == JobStatus.PENDING:
                removed_ids.append(job.job_id)
                updated = True
        if removed_ids:
            self._jobs = [j for j in self._jobs if j.job_id not in removed_ids]
            self._logger.info("Cleared %s pending jobs", len(removed_ids))
        if updated:
            self._store.save_history(self._jobs)
            self.queue_changed.emit()

    def retry_skipped(self) -> List[Job]:
        """Re-enqueue skipped jobs in original order."""
        skipped = [j for j in self._jobs if j.status == JobStatus.SKIPPED]
        new_jobs: List[Job] = []
        for job in skipped:
            new_jobs.append(self.enqueue_job(job.job_type, job.base_config_path or "", getattr(job, "overrides", {})))
        self._crashed = False
        self._auto_start_blocked = False
        return new_jobs

    def has_pending(self) -> bool:
        return any(job.status == JobStatus.PENDING for job in self._jobs)

    def pending_jobs(self) -> List[Job]:
        return [j for j in self._jobs if j.status == JobStatus.PENDING]

    def pending_count(self) -> int:
        return len(self.pending_jobs())

    # Internal helpers ---------------------------------------------------
    def _find_job(self, job_id: Optional[str]) -> Optional[Job]:
        if not job_id:
            return None
        for job in self._jobs:
            if job.job_id == job_id:
                return job
        return None

    def _start_next_job(self) -> None:
        if self._active_job_id or self._auto_start_blocked:
            return
        if self._worker_client.is_busy():
            return

        # Ensure worker is running before dispatching jobs
        if not self._worker_client.is_running():
            self._logger.info("Starting worker for job execution")
            if not self._worker_client.start():
                self._logger.error("Failed to start worker process")
                return

        next_job = next((job for job in self._jobs if job.status == JobStatus.PENDING), None)
        if not next_job:
            return

        self._start_job(next_job)

    def _start_job(self, job: Job) -> None:
        job.started_at = datetime.now(timezone.utc)
        job.status = JobStatus.RUNNING
        job.progress_current = 0
        job.progress_total = 100
        job.stage = "Starting"
        self._active_job_id = job.job_id
        self.job_updated.emit(job)
        self.active_changed.emit(job)

        # Dispatch to worker
        request_id = self._dispatch_job(job)
        if request_id:
            job.request_id = request_id
        else:
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(timezone.utc)
            job.summary = "Failed to start job"
            self._active_job_id = None
            self.job_updated.emit(job)
            self.active_changed.emit(None)
            self._store.save_history(self._jobs)
            self._start_next_job()

    def _dispatch_job(self, job: Job) -> Optional[str]:
        overrides = getattr(job, "overrides", {}) or {}
        if job.job_type == JobType.SCAN_LIBRARY:
            return self._worker_client.scan_library(job.base_config_path, overrides, job_id=job.job_id)
        if job.job_type == JobType.UPDATE_GENRES:
            return self._worker_client.update_genres(job.base_config_path, overrides, job_id=job.job_id)
        if job.job_type == JobType.UPDATE_SONIC:
            return self._worker_client.update_sonic(job.base_config_path, overrides, job_id=job.job_id)
        if job.job_type == JobType.BUILD_ARTIFACTS:
            return self._worker_client.build_artifacts(job.base_config_path, overrides, job_id=job.job_id)
        return None

    # Slots --------------------------------------------------------------
    @Slot(str, int, int, str, object)
    def _on_progress_received(self, stage: str, current: int, total: int, detail: str, job_id: Optional[str]) -> None:
        job = self._find_job(job_id or self._active_job_id)
        if not job:
            return
        job.progress_current = current
        job.progress_total = total or 100
        job.stage = detail or stage or job.stage
        self.job_updated.emit(job)

    @Slot(str, str, object)
    def _on_error_received(self, message: str, tb: str, job_id: Optional[str]) -> None:
        job = self._find_job(job_id or self._active_job_id)
        if not job:
            return
        job.error_message = message
        job.traceback = tb
        self.job_updated.emit(job)

    @Slot(str, bool, str, bool, object, str)
    def _on_done_received(
        self,
        cmd: str,
        ok: bool,
        detail: str,
        cancelled: bool,
        job_id: Optional[str],
        summary: str,
    ) -> None:
        job = self._find_job(job_id or self._active_job_id)
        if not job:
            return

        job.finished_at = datetime.now(timezone.utc)
        job.stage = detail or job.stage
        job.summary = summary or detail or job.summary

        if cancelled:
            job.status = JobStatus.CANCELLED
            job.stage = "Cancelled"
        elif ok:
            job.status = JobStatus.SUCCESS
        else:
            job.status = JobStatus.FAILED
            if detail and not job.error_message:
                job.error_message = detail

        if job.job_id == self._active_job_id:
            self._active_job_id = None
            self.active_changed.emit(None)

        self.job_updated.emit(job)
        self._store.save_history(self._jobs)
        self.queue_changed.emit()
        self._start_next_job()

    @Slot(bool)
    def _on_busy_changed(self, is_busy: bool) -> None:
        if not is_busy:
            self._start_next_job()

    @Slot(int, str)
    def _on_worker_stopped(self, exit_code: int, status: str) -> None:
        """Mark active job failed if worker died mid-run."""
        if self._active_job_id:
            job = self._find_job(self._active_job_id)
            if job:
                if job.status == JobStatus.CANCELLING:
                    job.status = JobStatus.CANCELLED
                    job.summary = "Worker terminated after cancel"
                else:
                    job.status = JobStatus.FAILED
                job.finished_at = datetime.now(timezone.utc)
                job.summary = job.summary or f"Worker exited unexpectedly (code={exit_code}, status={status})"
                self.job_updated.emit(job)
                self._store.save_history(self._jobs)
            self._active_job_id = None
            self.active_changed.emit(None)
        # Skip remaining pending jobs until user retries
        pending = [j for j in self._jobs if j.status == JobStatus.PENDING]
        if pending:
            for job in pending:
                job.status = JobStatus.SKIPPED
                job.finished_at = datetime.now(timezone.utc)
                job.summary = "Skipped due to worker crash"
                self.job_updated.emit(job)
            self._store.save_history(self._jobs)
            self.queue_changed.emit()
        self._crashed = True
