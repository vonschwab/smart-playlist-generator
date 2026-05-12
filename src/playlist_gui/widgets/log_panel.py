"""
Log Panel - Displays structured logs from the worker process and GUI.

Features:
- Append structured logs with level coloring
- Filter by log level
- Search within logs
- Copy selected or visible logs
- Open logs folder
- Clear button (view only)
- Auto-scroll to bottom
"""
from __future__ import annotations

from typing import Optional, List, Tuple
from pathlib import Path
from PySide6.QtCore import Qt, Slot, QUrl
from PySide6.QtGui import QColor, QTextCursor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from platformdirs import user_log_dir
except ImportError:  # pragma: no cover - optional
    user_log_dir = None

from ..utils.redaction import redact_text

# Log level colors
LEVEL_COLORS = {
    "DEBUG": QColor(140, 140, 140),
    "INFO": QColor(224, 224, 224),
    "WARNING": QColor(255, 190, 80),
    "ERROR": QColor(255, 105, 105),
    "CRITICAL": QColor(210, 120, 255),
}


class LogPanel(QWidget):
    """
    Log display panel with filtering, search, and clipboard helpers.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._auto_scroll = True
        self._entries: List[Tuple[str, str, bool]] = []  # (level, message, is_verbose)
        self._entries_bytes = 0
        self._max_entries = 2000
        self._max_bytes = 2 * 1024 * 1024

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        self._toolbar_frame = QFrame()
        self._toolbar_frame.setObjectName("logToolbar")
        toolbar = QHBoxLayout(self._toolbar_frame)
        toolbar.setContentsMargins(8, 6, 8, 6)
        toolbar.setSpacing(8)

        self._title_label = QLabel("Logs")
        self._title_label.setObjectName("logTitle")
        toolbar.addWidget(self._title_label)

        self._status_label = QLabel("0 entries")
        self._status_label.setObjectName("logStatusLabel")
        toolbar.addWidget(self._status_label)

        toolbar.addSpacing(8)

        # Level filter label
        level_label = QLabel("Level:")
        level_label.setObjectName("logToolbarLabel")
        toolbar.addWidget(level_label)

        # Level filter checkboxes
        self._level_checks = {}
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            check = QCheckBox(level)
            check.setObjectName("logLevelToggle")
            check.setChecked(level != "DEBUG")  # DEBUG off by default
            check.stateChanged.connect(self._on_filter_changed)
            toolbar.addWidget(check)
            self._level_checks[level] = check

        # Show verbose logs toggle (for worker verbose_log events)
        self._show_verbose_check = QCheckBox("Show Verbose")
        self._show_verbose_check.setObjectName("logLevelToggle")
        self._show_verbose_check.setChecked(True)
        self._show_verbose_check.setToolTip("Show detailed verbose logs from worker operations")
        self._show_verbose_check.stateChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self._show_verbose_check)

        toolbar.addStretch()

        search_label = QLabel("Search:")
        search_label.setObjectName("logToolbarLabel")
        toolbar.addWidget(search_label)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Find text...")
        self._search_edit.textChanged.connect(self._refresh_view)
        self._search_edit.setClearButtonEnabled(True)
        toolbar.addWidget(self._search_edit, stretch=1)

        # Auto-scroll toggle
        self._auto_scroll_check = QCheckBox("Auto-scroll")
        self._auto_scroll_check.setObjectName("logLevelToggle")
        self._auto_scroll_check.setChecked(True)
        self._auto_scroll_check.stateChanged.connect(self._on_auto_scroll_changed)
        toolbar.addWidget(self._auto_scroll_check)

        # Copy button
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setObjectName("logActionButton")
        self._copy_btn.clicked.connect(self._copy_selected)
        toolbar.addWidget(self._copy_btn)

        # Open logs folder
        self._open_btn = QPushButton("Open Logs")
        self._open_btn.setObjectName("logActionButton")
        self._open_btn.clicked.connect(self._open_logs_folder)
        toolbar.addWidget(self._open_btn)

        # Clear button (view only)
        self._clear_btn = QPushButton("Clear View")
        self._clear_btn.setObjectName("logActionButton")
        self._clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(self._clear_btn)

        layout.addWidget(self._toolbar_frame)

        # Log text display
        self._text_edit = QTextEdit()
        self._text_edit.setObjectName("logText")
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self._text_edit.setFontFamily("Consolas, Courier New, monospace")
        self._text_edit.document().setMaximumBlockCount(self._max_entries + 1)
        layout.addWidget(self._text_edit)
        self._update_status()

    def _get_enabled_levels(self) -> set:
        """Get the set of enabled log levels."""
        enabled = set()
        for level, check in self._level_checks.items():
            if check.isChecked():
                enabled.add(level)
        return enabled

    def _level_priority(self, level: str) -> int:
        """Get numeric priority for a log level."""
        priorities = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        return priorities.get(level.upper(), 1)

    @Slot(str, str)
    def append_log(self, level: str, message: str, is_verbose: bool = False) -> None:
        """
        Append a log message to the display.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message text
            is_verbose: Whether this is a verbose worker log
        """
        level = level.upper()
        message = redact_text(message)

        self._entries.append((level, message, is_verbose))
        self._entries_bytes += self._entry_size(level, message)
        trimmed_for_bytes = self._trim_entries()
        if trimmed_for_bytes:
            self._refresh_view()
        elif self._entry_is_visible(level, message, is_verbose):
            self._append_to_view(level, message)
            if self._auto_scroll:
                self._text_edit.moveCursor(QTextCursor.End)
        self._update_status()

    @Slot(str, str, dict, object)
    def append_verbose_log(self, level: str, message: str, context: dict, job_id: Optional[str]) -> None:
        """
        Append a verbose log event from the worker.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message text
            context: Additional context data
            job_id: Associated job ID (optional)
        """
        # Format message with context if present
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items() if v)
            message = f"{message} [{context_str}]"

        if job_id:
            message = f"[Job {job_id[:8]}] {message}"

        self.append_log(level, message, is_verbose=True)

    def _refresh_view(self) -> None:
        """Rebuild the visible log view based on filters/search."""
        if not hasattr(self, "_text_edit"):
            return

        enabled_levels = self._get_enabled_levels()
        show_verbose = self._show_verbose_check.isChecked() if hasattr(self, "_show_verbose_check") else True
        query = (self._search_edit.text() or "").lower()

        self._text_edit.clear()

        for entry in self._entries:
            # Handle both old and new tuple formats for backward compatibility
            if len(entry) == 3:
                level, msg, is_verbose = entry
            else:
                level, msg = entry
                is_verbose = False

            if not self._entry_is_visible(
                level,
                msg,
                is_verbose,
                enabled_levels=enabled_levels,
                show_verbose=show_verbose,
                query=query,
            ):
                continue

            self._append_to_view(level, msg)

        if self._auto_scroll:
            self._text_edit.moveCursor(QTextCursor.End)
        self._update_status()

    def _append_to_view(self, level: str, message: str) -> None:
        """Append a single line with color styling."""
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        color = LEVEL_COLORS.get(level.upper(), LEVEL_COLORS["INFO"])
        self._text_edit.setTextColor(color)
        self._text_edit.insertPlainText(f"{level}: {message}\n")

    @Slot()
    def _on_filter_changed(self) -> None:
        """Handle level filter changes."""
        self._refresh_view()

    @Slot(int)
    def _on_auto_scroll_changed(self, state: int) -> None:
        """Handle auto-scroll toggle."""
        self._auto_scroll = state == Qt.Checked

    @Slot()
    def clear(self) -> None:  # type: ignore[override]
        """Clear the visible log view (keeps history list)."""
        self._text_edit.clear()
        self._update_status()

    def _entry_size(self, level: str, message: str) -> int:
        return len(f"{level}:{message}".encode("utf-8", "ignore"))

    def _entry_is_visible(
        self,
        level: str,
        message: str,
        is_verbose: bool,
        *,
        enabled_levels: Optional[set] = None,
        show_verbose: Optional[bool] = None,
        query: Optional[str] = None,
    ) -> bool:
        """Return whether an entry should be rendered under current filters."""
        if enabled_levels is None:
            enabled_levels = self._get_enabled_levels()
        if show_verbose is None:
            show_verbose = self._show_verbose_check.isChecked() if hasattr(self, "_show_verbose_check") else True
        if query is None:
            query = (self._search_edit.text() or "").lower()

        if level not in enabled_levels:
            return False
        if is_verbose and not show_verbose:
            return False
        if query and query not in f"{level} {message}".lower():
            return False
        return True

    def _visible_entry_count(self) -> int:
        """Return the count of entries currently visible under filters/search."""
        enabled_levels = self._get_enabled_levels()
        show_verbose = self._show_verbose_check.isChecked() if hasattr(self, "_show_verbose_check") else True
        query = (self._search_edit.text() or "").lower()
        count = 0
        for entry in self._entries:
            if len(entry) == 3:
                level, msg, is_verbose = entry
            else:
                level, msg = entry
                is_verbose = False
            if self._entry_is_visible(
                level,
                msg,
                is_verbose,
                enabled_levels=enabled_levels,
                show_verbose=show_verbose,
                query=query,
            ):
                count += 1
        return count

    def _update_status(self) -> None:
        """Update the log count readout."""
        if not hasattr(self, "_status_label"):
            return
        total = len(self._entries)
        visible = self._visible_entry_count()
        total_noun = "entry" if total == 1 else "entries"
        if total == visible:
            self._status_label.setText(f"{total} {total_noun}")
        else:
            self._status_label.setText(f"{total} {total_noun} ({visible} visible)")

    def _trim_entries(self) -> bool:
        """Ensure the in-memory buffer stays within bounds."""
        trimmed_for_bytes = False
        while self._entries and (len(self._entries) > self._max_entries or self._entries_bytes > self._max_bytes):
            if self._entries_bytes > self._max_bytes:
                trimmed_for_bytes = True
            entry = self._entries.pop(0)
            # Handle both old and new tuple formats
            old_level = entry[0]
            old_msg = entry[1]
            self._entries_bytes -= self._entry_size(old_level, old_msg)
            if self._entries_bytes < 0:
                self._entries_bytes = 0
        return trimmed_for_bytes

    def _copy_selected(self) -> None:
        """Copy selected text or all visible text if none selected."""
        cursor = self._text_edit.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText().replace("\u2029", "\n")
        else:
            text = self._text_edit.toPlainText()
        if text:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(text)

    def _open_logs_folder(self) -> None:
        """Open the logs directory."""
        if user_log_dir:
            log_dir = Path(user_log_dir("PlaylistGenerator", "PlaylistGenerator"))
        else:
            log_dir = Path.home() / ".PlaylistGenerator" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))
