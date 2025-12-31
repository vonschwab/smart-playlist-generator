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
    "DEBUG": QColor(128, 128, 128),     # Gray
    "INFO": QColor(0, 0, 0),             # Black
    "WARNING": QColor(200, 150, 0),      # Orange
    "ERROR": QColor(200, 0, 0),          # Red
    "CRITICAL": QColor(150, 0, 150),     # Purple
}


class LogPanel(QWidget):
    """
    Log display panel with filtering, search, and clipboard helpers.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._auto_scroll = True
        self._entries: List[Tuple[str, str]] = []  # (level, message)
        self._entries_bytes = 0
        self._max_entries = 2000
        self._max_bytes = 2 * 1024 * 1024

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Level filter label
        toolbar.addWidget(QLabel("Level:"))

        # Level filter checkboxes
        self._level_checks = {}
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            check = QCheckBox(level)
            check.setChecked(level != "DEBUG")  # DEBUG off by default
            check.stateChanged.connect(self._on_filter_changed)
            toolbar.addWidget(check)
            self._level_checks[level] = check

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Search:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Find text...")
        self._search_edit.textChanged.connect(self._refresh_view)
        self._search_edit.setClearButtonEnabled(True)
        toolbar.addWidget(self._search_edit, stretch=1)

        # Auto-scroll toggle
        self._auto_scroll_check = QCheckBox("Auto-scroll")
        self._auto_scroll_check.setChecked(True)
        self._auto_scroll_check.stateChanged.connect(self._on_auto_scroll_changed)
        toolbar.addWidget(self._auto_scroll_check)

        # Copy button
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(60)
        copy_btn.clicked.connect(self._copy_selected)
        toolbar.addWidget(copy_btn)

        # Open logs folder
        open_btn = QPushButton("Open Logs")
        open_btn.setFixedWidth(90)
        open_btn.clicked.connect(self._open_logs_folder)
        toolbar.addWidget(open_btn)

        # Clear button (view only)
        clear_btn = QPushButton("Clear View")
        clear_btn.setFixedWidth(90)
        clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(clear_btn)

        layout.addLayout(toolbar)

        # Log text display
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self._text_edit.setFontFamily("Consolas, Courier New, monospace")
        layout.addWidget(self._text_edit)

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
    def append_log(self, level: str, message: str) -> None:
        """
        Append a log message to the display.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message text
        """
        level = level.upper()
        message = redact_text(message)

        self._entries.append((level, message))
        self._entries_bytes += self._entry_size(level, message)
        self._trim_entries()
        self._refresh_view()

    def _refresh_view(self) -> None:
        """Rebuild the visible log view based on filters/search."""
        if not hasattr(self, "_text_edit"):
            return

        enabled_levels = self._get_enabled_levels()
        query = (self._search_edit.text() or "").lower()

        self._text_edit.clear()

        for level, msg in self._entries:
            if level not in enabled_levels:
                continue
            if query and query not in f"{level} {msg}".lower():
                continue
            self._append_to_view(level, msg)

        if self._auto_scroll:
            self._text_edit.moveCursor(QTextCursor.End)

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

    def _entry_size(self, level: str, message: str) -> int:
        return len(f"{level}:{message}".encode("utf-8", "ignore"))

    def _trim_entries(self) -> None:
        """Ensure the in-memory buffer stays within bounds."""
        while self._entries and (len(self._entries) > self._max_entries or self._entries_bytes > self._max_bytes):
            old_level, old_msg = self._entries.pop(0)
            self._entries_bytes -= self._entry_size(old_level, old_msg)
            if self._entries_bytes < 0:
                self._entries_bytes = 0

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
