"""
Log Panel - Displays structured logs from the worker process

Features:
- Append structured logs with level coloring
- Filter by log level
- Clear button
- Auto-scroll to bottom
"""
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


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
    Log display panel with filtering and clear functionality.

    Usage:
        panel = LogPanel()
        panel.append_log("INFO", "Application started")
        panel.append_log("ERROR", "Something went wrong")
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min_level = "INFO"  # Minimum level to display
        self._auto_scroll = True

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

        # Auto-scroll toggle
        self._auto_scroll_check = QCheckBox("Auto-scroll")
        self._auto_scroll_check.setChecked(True)
        self._auto_scroll_check.stateChanged.connect(self._on_auto_scroll_changed)
        toolbar.addWidget(self._auto_scroll_check)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
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

        # Check if this level is enabled
        enabled = self._get_enabled_levels()
        if level not in enabled and level != "CRITICAL":
            return

        # Format the log line
        color = LEVEL_COLORS.get(level, QColor(0, 0, 0))

        # Move cursor to end
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._text_edit.setTextCursor(cursor)

        # Insert formatted text
        # Use HTML for coloring
        level_padded = f"[{level}]".ljust(10)
        html = f'<span style="color: {color.name()};">{level_padded}</span>{self._escape_html(message)}<br>'
        cursor.insertHtml(html)

        # Auto-scroll
        if self._auto_scroll:
            scrollbar = self._text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "&nbsp;")
        )

    @Slot()
    def clear(self) -> None:
        """Clear all log messages."""
        self._text_edit.clear()

    @Slot()
    def _on_filter_changed(self) -> None:
        """Handle filter checkbox change."""
        # Just affects future messages, not a retroactive filter
        pass

    @Slot(int)
    def _on_auto_scroll_changed(self, state: int) -> None:
        """Handle auto-scroll toggle."""
        self._auto_scroll = state == Qt.Checked

    def get_text(self) -> str:
        """Get all log text as plain text."""
        return self._text_edit.toPlainText()

    def set_font_size(self, size: int) -> None:
        """Set the log font size."""
        font = self._text_edit.font()
        font.setPointSize(size)
        self._text_edit.setFont(font)
