"""
Job Details Dialog - Shows comprehensive job information with tabs.

Features:
- Job summary and status
- Performance metrics with timing and throughput
- Checkpoint data for resumable jobs
- Error details with traceback
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..jobs import Job
from src.playlist.analyze_library_results import (
    build_analyze_library_readout,
    format_analyze_library_action_label,
    format_analyze_library_attention_summary,
)


class JobDetailsDialog(QDialog):
    """
    Dialog displaying comprehensive job information.
    """

    def __init__(self, job: Job, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._job = job
        self.setObjectName("jobDetailsDialog")
        self.setWindowTitle(f"Job Details - {job.job_type.label()}")
        self.resize(800, 600)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Summary section
        summary_group = self._create_summary_section()
        self._summary_group = summary_group
        layout.addWidget(summary_group)

        # Tabs for detailed information
        tabs = QTabWidget()
        self._tabs = tabs
        self._tabs.setObjectName("jobDetailsTabs")

        # Performance metrics tab
        perf_tab = self._create_performance_tab()
        tabs.addTab(perf_tab, "Performance")

        if self._job.job_type.value == "analyze_library":
            results_tab = self._create_analyze_library_results_tab()
            tabs.addTab(results_tab, "Results")

        # Checkpoint data tab (if available)
        if self._job.checkpoint_data:
            checkpoint_tab = self._create_checkpoint_tab()
            tabs.addTab(checkpoint_tab, "Checkpoint")

        # Error details tab (if failed)
        if self._job.error_message or self._job.traceback:
            error_tab = self._create_error_tab()
            tabs.addTab(error_tab, "Error Details")

        layout.addWidget(tabs)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self._button_box = button_box
        self._button_box.setObjectName("jobDetailsButtons")
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_summary_section(self) -> QGroupBox:
        """Create the summary information section."""
        group = QGroupBox("Job Summary")
        group.setObjectName("jobSummaryCard")
        layout = QVBoxLayout(group)

        # Job type and status
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"<b>Type:</b> {self._job.job_type.label()}"))
        info_layout.addWidget(QLabel(f"<b>Status:</b> {self._job.status.title()}"))
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Progress
        if self._job.progress_total > 0:
            progress_text = f"{self._job.progress_current}/{self._job.progress_total} ({self._job.progress_percent()}%)"
            layout.addWidget(QLabel(f"<b>Progress:</b> {progress_text}"))

        # Timing
        elapsed = self._job.elapsed()
        if elapsed:
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            layout.addWidget(QLabel(f"<b>Duration:</b> {minutes}m {seconds}s"))

        # Summary/Stage
        if self._job.summary:
            layout.addWidget(QLabel(f"<b>Summary:</b> {self._job.summary}"))
        if self._job.stage:
            layout.addWidget(QLabel(f"<b>Stage:</b> {self._job.stage}"))

        # Resume capability
        if self._job.can_resume:
            resume_label = QLabel("✓ <b>This job can be resumed from checkpoint</b>")
            resume_label.setObjectName("resumeNotice")
            layout.addWidget(resume_label)

        return group

    def _create_analyze_library_results_tab(self) -> QWidget:
        """Create a structured Analyze Library results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        result = self._job.result_data or {}
        readout = build_analyze_library_readout(
            result,
            status=str(self._job.status.value if hasattr(self._job.status, "value") else self._job.status),
            error_message=self._job.error_message,
        )
        self._results_headline_label = QLabel(f"<b>{readout['headline']}</b>")
        self._results_headline_label.setObjectName("analyzeResultsHeadline")
        layout.addWidget(self._results_headline_label)
        attention_summary = format_analyze_library_attention_summary(
            result,
            status=str(self._job.status.value if hasattr(self._job.status, "value") else self._job.status),
            error_message=self._job.error_message,
        )
        self._attention_summary_label = QLabel(attention_summary)
        self._attention_summary_label.setObjectName("analyzeAttentionBanner")
        if attention_summary:
            self._attention_summary_label.setWordWrap(True)
            layout.addWidget(self._attention_summary_label)

        readout_table = QTableWidget()
        self._readout_table = readout_table
        self._readout_table.setObjectName("jobReadoutTable")
        metrics = list(readout.get("metrics") or [])
        readout_table.setColumnCount(2)
        readout_table.setHorizontalHeaderLabels(["Metric", "Value"])
        readout_table.horizontalHeader().setStretchLastSection(True)
        readout_table.setShowGrid(False)
        readout_table.setRowCount(len(metrics))
        readout_table.setMaximumHeight(140)
        for row, (label, value) in enumerate(metrics):
            readout_table.setItem(row, 0, QTableWidgetItem(str(label)))
            readout_table.setItem(row, 1, QTableWidgetItem(str(value)))
        layout.addWidget(readout_table)

        attention = list(readout.get("attention") or [])
        if attention:
            layout.addWidget(QLabel("<b>Needs Attention:</b>"))
            attention_text = QTextEdit()
            self._attention_text = attention_text
            attention_text.setReadOnly(True)
            attention_text.setObjectName("analyzeAttentionText")
            attention_text.setPlainText("\n".join(f"- {item}" for item in attention))
            attention_text.setMaximumHeight(100)
            layout.addWidget(attention_text)
        else:
            self._attention_text = QTextEdit()
            self._attention_text.setReadOnly(True)

        self._add_path_row(layout, "Report", str(result.get("report_path") or ""), "report")
        self._add_path_row(layout, "Output", str(result.get("out_dir") or ""), "output")

        verify_issues = result.get("verify_issues") or []
        if verify_issues:
            layout.addWidget(QLabel(f"<b>Verify issues:</b> {', '.join(map(str, verify_issues))}"))

        stages = list(result.get("stages") or [])
        table = QTableWidget()
        self._results_table = table
        self._results_table.setObjectName("analyzeStageTable")
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Stage", "Decision", "Processed", "Errors", "Duration"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setShowGrid(False)
        table.setRowCount(len(stages))

        for row, stage in enumerate(stages):
            processed = stage.get("processed_count")
            if processed is None:
                processed = "-"
            duration = stage.get("duration_sec")
            duration_text = "-" if duration is None else f"{float(duration):.1f}s"
            stage_name = str(stage.get("name") or "-")
            values = [
                format_analyze_library_action_label(stage_name),
                stage.get("decision") or "-",
                str(processed),
                str(stage.get("errors_count", 0)),
                duration_text,
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))

        if stages:
            layout.addWidget(table)
        else:
            layout.addWidget(QLabel("<i>No completed stage report is available.</i>"))
        layout.addStretch()
        return widget

    def _add_path_row(
        self,
        layout: QVBoxLayout,
        label: str,
        path_value: str,
        attr_prefix: str,
    ) -> None:
        """Add a readable path row with copy/open actions."""
        has_path = bool(path_value)
        display_value = path_value if has_path else "Not available"
        row = QHBoxLayout()
        row.addWidget(QLabel(f"<b>{label}:</b>"))

        path_edit = QLineEdit(display_value)
        path_edit.setReadOnly(True)
        path_edit.setToolTip(path_value if has_path else f"{label} path is not available for this job")
        path_edit.setCursorPosition(0)
        row.addWidget(path_edit, 1)

        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip(f"Copy {label.lower()} path")
        copy_btn.clicked.connect(
            lambda _checked=False, value=path_value: QApplication.clipboard().setText(value)
        )
        copy_btn.setEnabled(has_path)
        row.addWidget(copy_btn)

        open_btn = QPushButton("Open")
        open_btn.setToolTip(f"Open {label.lower()} path")
        open_btn.clicked.connect(
            lambda _checked=False, value=path_value: self._open_path(value)
        )
        open_btn.setEnabled(has_path)
        row.addWidget(open_btn)

        setattr(self, f"_{attr_prefix}_path_edit", path_edit)
        setattr(self, f"_{attr_prefix}_path_copy_btn", copy_btn)
        setattr(self, f"_{attr_prefix}_path_open_btn", open_btn)
        layout.addLayout(row)

    def _open_path(self, path_value: str) -> None:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _create_performance_tab(self) -> QWidget:
        """Create the performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Check if we have performance data
        perf_data = getattr(self._job, "performance_data", None)

        if not perf_data:
            # Show basic metrics from job progress
            layout.addWidget(QLabel("<i>Detailed performance metrics not available</i>"))

            # Show what we have
            metrics_table = QTableWidget()
            metrics_table.setColumnCount(2)
            metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
            metrics_table.horizontalHeader().setStretchLastSection(True)

            rows = []

            # Elapsed time
            elapsed = self._job.elapsed()
            if elapsed:
                minutes = int(elapsed.total_seconds() // 60)
                seconds = int(elapsed.total_seconds() % 60)
                rows.append(("Duration", f"{minutes}m {seconds}s"))

            # Progress
            if self._job.progress_total > 0:
                rows.append(("Items Processed", f"{self._job.progress_current}/{self._job.progress_total}"))

                # Calculate throughput
                if elapsed and elapsed.total_seconds() > 0:
                    throughput = self._job.progress_current / elapsed.total_seconds()
                    rows.append(("Throughput", f"{throughput:.2f} items/s"))

            metrics_table.setRowCount(len(rows))
            for i, (metric, value) in enumerate(rows):
                metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))

            layout.addWidget(metrics_table)

        else:
            # Show detailed stage-level performance metrics
            layout.addWidget(QLabel("<b>Stage Performance Metrics</b>"))

            perf_table = QTableWidget()
            perf_table.setColumnCount(4)
            perf_table.setHorizontalHeaderLabels(["Stage", "Duration", "Throughput", "Items"])
            perf_table.horizontalHeader().setStretchLastSection(True)

            stages = list(perf_data.items())
            perf_table.setRowCount(len(stages))

            for i, (stage_name, metrics) in enumerate(stages):
                perf_table.setItem(i, 0, QTableWidgetItem(stage_name))
                perf_table.setItem(i, 1, QTableWidgetItem(f"{metrics['duration']:.1f}s"))
                perf_table.setItem(i, 2, QTableWidgetItem(f"{metrics['throughput']:.1f}/s"))
                perf_table.setItem(i, 3, QTableWidgetItem(str(metrics['items'])))

            layout.addWidget(perf_table)

        layout.addStretch()
        return widget

    def _create_checkpoint_tab(self) -> QWidget:
        """Create the checkpoint data tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("<b>Checkpoint Information</b>"))
        layout.addWidget(QLabel("This job was cancelled and saved a checkpoint for resumption."))

        # Display checkpoint data
        checkpoint_table = QTableWidget()
        checkpoint_table.setColumnCount(2)
        checkpoint_table.setHorizontalHeaderLabels(["Field", "Value"])
        checkpoint_table.horizontalHeader().setStretchLastSection(True)

        checkpoint_data = self._job.checkpoint_data or {}

        rows = [
            ("Items Completed", str(checkpoint_data.get("items_completed", "N/A"))),
            ("Total Items", str(checkpoint_data.get("total_items", "N/A"))),
            ("Stage", str(checkpoint_data.get("stage", "N/A"))),
        ]

        # Add resumable state info if present
        resumable_state = checkpoint_data.get("resumable_state", {})
        if resumable_state:
            rows.append(("Last File Index", str(resumable_state.get("last_file_index", "N/A"))))

            # Stats if available
            stats = resumable_state.get("stats", {})
            if stats:
                rows.append(("New Tracks", str(stats.get("new", "N/A"))))
                rows.append(("Updated Tracks", str(stats.get("updated", "N/A"))))
                rows.append(("Failed", str(stats.get("failed", "N/A"))))

        checkpoint_table.setRowCount(len(rows))
        for i, (field, value) in enumerate(rows):
            checkpoint_table.setItem(i, 0, QTableWidgetItem(field))
            checkpoint_table.setItem(i, 1, QTableWidgetItem(value))

        layout.addWidget(checkpoint_table)
        layout.addStretch()
        return widget

    def _create_error_tab(self) -> QWidget:
        """Create the error details tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Error message
        if self._job.error_message:
            layout.addWidget(QLabel("<b>Error Message:</b>"))
            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setPlainText(self._job.error_message)
            error_text.setMaximumHeight(100)
            layout.addWidget(error_text)

        # Traceback
        if self._job.traceback:
            layout.addWidget(QLabel("<b>Traceback:</b>"))
            traceback_text = QTextEdit()
            traceback_text.setReadOnly(True)
            traceback_text.setPlainText(self._job.traceback)
            traceback_text.setFontFamily("Consolas, Courier New, monospace")
            layout.addWidget(traceback_text)

        return widget
