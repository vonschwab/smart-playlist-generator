"""
Jobs panel - shows queued/running/completed jobs with progress and summary.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..jobs import Job, JobManager, JobStatus, JobTableModel


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
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._run_all_btn = QPushButton("Run All (Scan -> Genres -> Sonic -> Artifacts)")
        self._run_all_btn.clicked.connect(self._handle_run_pipeline)
        toolbar.addWidget(self._run_all_btn)

        self._cancel_active_btn = QPushButton("Cancel Active")
        self._cancel_active_btn.clicked.connect(self._handle_cancel_active)
        toolbar.addWidget(self._cancel_active_btn)

        self._cancel_pending_btn = QPushButton("Clear Pending")
        self._cancel_pending_btn.clicked.connect(self._handle_cancel_pending)
        toolbar.addWidget(self._cancel_pending_btn)

        self._view_filter = QComboBox()
        self._view_filter.addItems(["Queue", "History", "All"])
        self._view_filter.setCurrentText("Queue")
        self._view_filter.currentTextChanged.connect(self._on_filter_changed)
        toolbar.addWidget(QLabel("Show:"))
        toolbar.addWidget(self._view_filter)

        toolbar.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #555; font-size: 11px;")
        toolbar.addWidget(self._status_label)

        layout.addLayout(toolbar)

        # Table
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.setEditTriggers(QTableView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(5, Qt.DescendingOrder)  # Last run
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
