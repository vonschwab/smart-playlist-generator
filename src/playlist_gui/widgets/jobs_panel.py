"""
Jobs panel - shows queued/running/completed jobs with progress and summary.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..jobs import Job, JobManager, JobStatus, JobTableModel
from ..jobs.job_types import JobType
from src.playlist.analyze_library_results import format_analyze_library_attention_summary
from .job_details_dialog import JobDetailsDialog


class JobsPanel(QWidget):
    """Dockable panel for job status and queue controls."""

    def __init__(
        self,
        job_manager: JobManager,
        on_run_pipeline: Optional[Callable[[], None]] = None,
        on_cancel_active: Optional[Callable[[], None]] = None,
        on_cancel_pending: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._job_manager = job_manager
        self._on_run_pipeline = on_run_pipeline
        self._on_cancel_active = on_cancel_active
        self._on_cancel_pending = on_cancel_pending
        self._model = JobTableModel(job_manager.jobs())

        self._setup_ui()
        self._connect_signals()
        self._refresh_status()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Toolbar
        self._toolbar_frame = QFrame()
        self._toolbar_frame.setObjectName("jobsToolbar")
        toolbar = QHBoxLayout(self._toolbar_frame)
        toolbar.setContentsMargins(8, 6, 8, 6)
        toolbar.setSpacing(8)

        self._run_all_btn = QPushButton("Analyze Library")
        self._run_all_btn.setObjectName("jobsPrimaryButton")
        self._run_all_btn.setToolTip(
            "Analyze Library: scan, genres, Discogs, sonic analysis, genre similarity, artifacts, and verify"
        )
        self._run_all_btn.clicked.connect(self._handle_run_pipeline)
        toolbar.addWidget(self._run_all_btn)

        self._cancel_active_btn = QPushButton("Cancel Active")
        self._cancel_active_btn.setToolTip("Cancel the currently running operation")
        self._cancel_active_btn.clicked.connect(self._handle_cancel_active)
        toolbar.addWidget(self._cancel_active_btn)

        self._cancel_pending_btn = QPushButton("Clear Pending")
        self._cancel_pending_btn.setToolTip("Remove queued operations that have not started")
        self._cancel_pending_btn.clicked.connect(self._handle_cancel_pending)
        toolbar.addWidget(self._cancel_pending_btn)

        self._latest_details_btn = QPushButton("Latest Details")
        self._latest_details_btn.setToolTip("Open the latest job details, including Analyze Library results")
        self._latest_details_btn.clicked.connect(self._handle_latest_details)
        toolbar.addWidget(self._latest_details_btn)

        self._view_filter = QComboBox()
        self._view_filter.addItems(["Queue", "History", "All"])
        self._view_filter.setCurrentText("Queue")
        self._view_filter.currentTextChanged.connect(self._on_filter_changed)
        show_label = QLabel("Show:")
        show_label.setObjectName("jobsToolbarLabel")
        toolbar.addWidget(show_label)
        toolbar.addWidget(self._view_filter)

        toolbar.addStretch()

        self._status_label = QLabel("")
        self._status_label.setObjectName("jobsStatusLabel")
        toolbar.addWidget(self._status_label)

        layout.addWidget(self._toolbar_frame)

        self._latest_summary_label = QLabel("")
        self._latest_summary_label.setObjectName("jobsLatestSummary")
        self._latest_summary_label.setWordWrap(True)
        self._latest_summary_label.setVisible(False)
        layout.addWidget(self._latest_summary_label)

        # Table
        self._table = QTableView()
        self._table.setObjectName("jobsTable")
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.setEditTriggers(QTableView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.setTextElideMode(Qt.ElideRight)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(28)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(5, Qt.DescendingOrder)  # Last run
        self._table.doubleClicked.connect(self._on_job_double_clicked)
        layout.addWidget(self._table)

    def _connect_signals(self) -> None:
        self._job_manager.job_added.connect(self._on_job_added)
        self._job_manager.job_updated.connect(self._on_job_updated)
        self._job_manager.active_changed.connect(self._on_active_changed)
        self._job_manager.queue_changed.connect(self._on_queue_changed)

    # Slots --------------------------------------------------------------
    def _on_job_added(self, job: Job) -> None:
        self._model.add_job(job)
        self._refresh_status()

    def _on_job_updated(self, job: Job) -> None:
        self._model.update_job(job)
        self._refresh_status()

    def _on_active_changed(self, job: Optional[Job]) -> None:
        self._refresh_status(job)

    # Actions ------------------------------------------------------------
    def _handle_run_pipeline(self) -> None:
        if self._on_run_pipeline:
            self._on_run_pipeline()

    def _handle_cancel_active(self) -> None:
        if self._on_cancel_active:
            self._on_cancel_active()

    def _handle_cancel_pending(self) -> None:
        if self._on_cancel_pending:
            self._on_cancel_pending()

    def _handle_latest_details(self) -> None:
        job = self._latest_completed_job()
        if job:
            self._open_job_details(job)

    def _on_filter_changed(self, text: str) -> None:
        mode = text.lower()
        if mode == "queue":
            self._model.set_filter_mode("queue")
        elif mode == "history":
            self._model.set_filter_mode("history")
        else:
            self._model.set_filter_mode("all")
        self._refresh_status()

    # Helpers ------------------------------------------------------------
    def _on_queue_changed(self) -> None:
        # Reload model from manager when queue changes (clears, retries)
        self._model.set_jobs(self._job_manager.jobs())
        self._refresh_status()

    def _refresh_status(self, active_job: Optional[Job] = None) -> None:
        active = active_job or self._job_manager.active_job()

        pending_count = self._job_manager.pending_count()
        running_text = ""
        if active:
            running_text = f"Active: {active.job_type.label()} ({active.status.title()})"

        status_parts = []
        if running_text:
            status_parts.append(running_text)
        status_parts.append(f"Pending: {pending_count}")

        self._status_label.setText(" | ".join(status_parts))

        # Enable/disable buttons based on queue state
        self._cancel_active_btn.setEnabled(
            active is not None and active.status in (JobStatus.RUNNING, JobStatus.CANCELLING)
        )
        self._cancel_pending_btn.setEnabled(pending_count > 0)
        self._run_all_btn.setEnabled(active is None and pending_count == 0)
        latest = self._latest_completed_job()
        self._latest_details_btn.setEnabled(latest is not None)
        self._refresh_latest_summary(latest)

    def _on_job_double_clicked(self, index) -> None:
        """Open job details dialog when a job is double-clicked."""
        if not index.isValid():
            return

        # Get the job from the model
        job = self._model.data(index, Qt.UserRole)
        if not job:
            return

        self._open_job_details(job)

    def _latest_completed_job(self) -> Optional[Job]:
        """Return the newest completed job, independent of current table filter."""
        terminal_statuses = {
            JobStatus.SUCCESS,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.SKIPPED,
        }
        candidates = []
        for job in self._job_manager.jobs():
            status = job.status
            if isinstance(status, str):
                try:
                    status = JobStatus(status)
                except ValueError:
                    continue
            if status in terminal_statuses:
                candidates.append(job)
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda job: job.finished_at or job.started_at or job.created_at,
        )

    def _refresh_latest_summary(self, job: Optional[Job]) -> None:
        """Show a compact readout for the latest completed job."""
        if not job:
            self._latest_summary_label.setText("")
            self._latest_summary_label.setVisible(False)
            return

        summary = self._latest_summary_text(job)
        self._latest_summary_label.setText(f"Latest: {job.job_type.label()} - {summary}")
        self._latest_summary_label.setVisible(True)

    def _latest_summary_text(self, job: Job) -> str:
        if job.job_type == JobType.ANALYZE_LIBRARY:
            status = job.status.value if hasattr(job.status, "value") else str(job.status)
            attention = format_analyze_library_attention_summary(
                job.result_data or {},
                status=status,
                error_message=job.error_message,
            )
            if attention:
                return attention
        return job.summary or job.error_message or job.stage or "No summary available"

    def _open_job_details(self, job: Job) -> None:
        """Open details dialog for a job."""
        dialog = JobDetailsDialog(job, parent=self)
        dialog.exec()
