"""In-memory job registry for tracking playlist generation tasks."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict, deque
from typing import Optional

from .schemas import JobOut, PlaylistOut


class _JobState:
    """Internal mutable state for a job."""

    def __init__(self, job_id: str, max_log_lines: int, request_params: Optional[dict] = None):
        self.job_id = job_id
        self.status = "pending"
        self.stage = ""
        self.error: Optional[str] = None
        self.playlist: Optional[PlaylistOut] = None
        self.tool_result: Optional[dict] = None
        self.logs: deque[str] = deque(maxlen=max_log_lines)
        self.created_at: float = time.time()
        self.request_params: dict = request_params or {}


class JobRegistry:
    """In-memory registry for tracking playlist generation jobs."""

    def __init__(self, max_log_lines: int = 500, max_jobs: int = 50):
        """Initialize the registry.

        Args:
            max_log_lines: Maximum log lines to retain per job.
            max_jobs: Maximum number of jobs to keep in memory (LRU).
        """
        self._jobs: OrderedDict[str, _JobState] = OrderedDict()
        self._max_log_lines = max_log_lines
        self._max_jobs = max_jobs

    def create(self, request_params: Optional[dict] = None) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = _JobState(job_id, self._max_log_lines, request_params)
        self._jobs[job_id].status = "running"
        # Evict oldest job if we exceed max_jobs
        while len(self._jobs) > self._max_jobs:
            self._jobs.popitem(last=False)
        return job_id

    def apply_event(self, event: dict) -> None:
        """Apply an event to a job, updating its state.

        Handles: log, progress, result, error, done, cancelled events.
        """
        job = self._jobs.get(event.get("job_id"))
        if not job:
            return
        etype = event.get("type")
        if etype == "log":
            job.logs.append(f"{event.get('level', 'INFO')}: {event.get('msg', '')}")
        elif etype == "progress":
            job.stage = event.get("detail") or event.get("stage") or job.stage
        elif etype == "result" and event.get("result_type") == "playlist":
            job.playlist = PlaylistOut.from_worker(event.get("playlist", {}))
        elif etype == "result":
            job.tool_result = dict(event)
        elif etype == "error":
            job.error = event.get("message", "Unknown error")
        elif etype == "done":
            if event.get("cancelled"):
                job.status = "cancelled"
            elif event.get("ok"):
                job.status = "success"
            else:
                job.status = "failed"
                if event.get("detail") and not job.error:
                    job.error = event["detail"]

    def _to_out(self, job: _JobState) -> JobOut:
        """Convert internal state to response model."""
        return JobOut(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            error=job.error,
            playlist=job.playlist,
            tool_result=job.tool_result,
            created_at=job.created_at,
            request_params=job.request_params,
        )

    def get(self, job_id: str) -> Optional[JobOut]:
        """Get a job by ID, or None if not found."""
        job = self._jobs.get(job_id)
        return self._to_out(job) if job else None

    def logs(self, job_id: str) -> list[str]:
        """Get all logs for a job, or empty list if not found."""
        job = self._jobs.get(job_id)
        return list(job.logs) if job else []

    def recent(self) -> list[JobOut]:
        """Get all jobs, newest first."""
        return [self._to_out(j) for j in reversed(self._jobs.values())]
