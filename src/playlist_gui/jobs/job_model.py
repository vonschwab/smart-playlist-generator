"""
Job model definitions and Qt table model for displaying jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from .job_types import JobType


class JobStatus(str, Enum):
    """State machine for jobs."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


def _safe_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_dt(val: Optional[datetime]) -> Optional[datetime]:
    if not val:
        return None
    if isinstance(val, datetime) and val.tzinfo is None:
        return val.replace(tzinfo=timezone.utc)
    return val


@dataclass
class Job:
    """Represents a single job run."""

    job_id: str
    job_type: JobType
    status: str = JobStatus.PENDING
    request_id: Optional[str] = None
    created_at: datetime = field(default_factory=_safe_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_current: int = 0
    progress_total: int = 100
    stage: str = ""
    summary: str = ""
    error_message: str = ""
    traceback: str = ""
    base_config_path: Optional[str] = None
    overrides: dict = field(default_factory=dict, repr=False, compare=False)

    def progress_percent(self) -> int:
        if self.progress_total <= 0:
            return 0
        pct = int((self.progress_current / self.progress_total) * 100)
        return max(0, min(100, pct))

    def elapsed(self) -> Optional[timedelta]:
        start = _normalize_dt(self.started_at)
        end = _normalize_dt(self.finished_at)
        if start and end:
            return end - start
        if start and self.status == JobStatus.RUNNING:
            return _safe_now() - start
        return None

    def to_dict(self) -> Dict:
        """Serialize to a JSON-safe dict."""

        def _dt(val: Optional[datetime]) -> Optional[str]:
            return val.isoformat() if isinstance(val, datetime) else None

        status_value = self.status.value if isinstance(self.status, Enum) else self.status

        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value if isinstance(self.job_type, JobType) else str(self.job_type),
            "status": status_value,
            "request_id": self.request_id,
            "created_at": _dt(self.created_at),
            "started_at": _dt(self.started_at),
            "finished_at": _dt(self.finished_at),
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
            "stage": self.stage,
            "summary": self.summary,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "base_config_path": self.base_config_path,
        }

    @staticmethod
    def from_dict(payload: Dict) -> "Job":
        """Deserialize from dict."""

        def _parse_dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            try:
                dt = datetime.fromisoformat(val)
                return _normalize_dt(dt)
            except Exception:
                return None

        job_type = payload.get("job_type") or JobType.SCAN_LIBRARY
        if isinstance(job_type, str):
            try:
                job_type = JobType(job_type)
            except ValueError:
                job_type = JobType.SCAN_LIBRARY

        status = payload.get("status", JobStatus.PENDING)
        if isinstance(status, str):
            try:
                status = JobStatus(status)
            except ValueError:
                status = JobStatus.PENDING

        return Job(
            job_id=payload.get("job_id", ""),
            job_type=job_type,
            status=status,
            request_id=payload.get("request_id"),
            created_at=_parse_dt(payload.get("created_at")) or _safe_now(),
            started_at=_parse_dt(payload.get("started_at")),
            finished_at=_parse_dt(payload.get("finished_at")),
            progress_current=int(payload.get("progress_current", 0)),
            progress_total=int(payload.get("progress_total", 100)),
            stage=payload.get("stage", ""),
            summary=payload.get("summary", ""),
            error_message=payload.get("error_message", ""),
            traceback=payload.get("traceback", ""),
            base_config_path=payload.get("base_config_path"),
        )


class JobTableModel(QAbstractTableModel):
    """Table model for listing jobs in a dock."""

    HEADERS = ["Type", "Status", "Progress", "Stage", "Elapsed", "Last Run", "Summary"]

    def __init__(self, jobs: Optional[List[Job]] = None, parent=None):
        super().__init__(parent)
        self._jobs: List[Job] = list(jobs or [])
        self._filter_mode: str = "queue"  # queue | history | all
        self._filtered: List[Job] = self._apply_filter()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._filtered)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._filtered)):
            return None

        job = self._filtered[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            status_text = job.status.value if isinstance(job.status, JobStatus) else job.status
            if col == 0:
                return job.job_type.label()
            if col == 1:
                return status_text.title()
            if col == 2:
                pct = job.progress_percent()
                return f"{pct}% ({job.progress_current}/{job.progress_total})"
            if col == 3:
                return job.stage or "-"
            if col == 4:
                elapsed = job.elapsed()
                if elapsed:
                    minutes = int(elapsed.total_seconds() // 60)
                    seconds = int(elapsed.total_seconds() % 60)
                    return f"{minutes}m {seconds}s"
                return "-"
            if col == 5:
                ts = job.finished_at or job.started_at or job.created_at
                return ts.strftime("%Y-%m-%d %H:%M") if ts else "-"
            if col == 6:
                return job.summary or job.error_message or "-"

        if role == Qt.UserRole:
            return job

        if role == Qt.ForegroundRole:
            status_key = job.status
            if not isinstance(status_key, JobStatus):
                try:
                    status_key = JobStatus(str(status_key))
                except Exception:
                    status_key = None
            palette = {
                JobStatus.SUCCESS: QColor("#155724"),
                JobStatus.FAILED: QColor("#8b0000"),
                JobStatus.CANCELLED: QColor("#8b5500"),
                JobStatus.CANCELLING: QColor("#8b5500"),
                JobStatus.RUNNING: QColor("#0f4c81"),
                JobStatus.SKIPPED: QColor("#666666"),
            }
            return palette.get(status_key)

        return None

    # Public helpers -----------------------------------------------------
    def set_jobs(self, jobs: List[Job]) -> None:
        self.beginResetModel()
        self._jobs = list(jobs)
        self._filtered = self._apply_filter()
        self.endResetModel()

    def add_job(self, job: Job) -> None:
        self._jobs.append(job)
        self._rebuild_filter()

    def update_job(self, job: Job) -> None:
        for idx, existing in enumerate(self._jobs):
            if existing.job_id == job.job_id:
                self._jobs[idx] = job
                break
        self._rebuild_filter()

    def remove_jobs(self, job_ids: List[str]) -> None:
        if not job_ids:
            return
        self.beginResetModel()
        self._jobs = [j for j in self._jobs if j.job_id not in job_ids]
        self._filtered = self._apply_filter()
        self.endResetModel()

    def jobs(self) -> List[Job]:
        return list(self._jobs)

    def set_filter_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in {"queue", "history", "all"}:
            mode = "queue"
        if mode == self._filter_mode:
            return
        self._filter_mode = mode
        self._rebuild_filter()

    def _rebuild_filter(self) -> None:
        self.beginResetModel()
        self._filtered = self._apply_filter()
        self.endResetModel()

    def _apply_filter(self) -> List[Job]:
        def _status_str(job: Job) -> str:
            return job.status.value if isinstance(job.status, JobStatus) else str(job.status).upper()

        if self._filter_mode == "history":
            return [
                j
                for j in self._jobs
                if _status_str(j) in {"SUCCESS", "FAILED", "CANCELLED", "SKIPPED"}
            ]
        if self._filter_mode == "queue":
            return [j for j in self._jobs if _status_str(j) in {"PENDING", "RUNNING", "CANCELLING"}]
        return list(self._jobs)
