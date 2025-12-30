"""
Export Dialog - Save As style dialog for exporting playlists

Provides dialogs for:
- Local M3U8 export with directory selection
- Plex export with playlist naming
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QMessageBox,
)


class ExportLocalDialog(QDialog):
    """
    Save As dialog for exporting playlist to local M3U8 file.

    Features:
    - Playlist name input with default "Auto - <artist> <date>"
    - Directory browser with address bar
    - Preview of full file path
    """

    def __init__(
        self,
        parent=None,
        default_name: str = "",
        default_directory: str = "",
        artist_name: str = ""
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Playlist to M3U8")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._artist_name = artist_name
        self._default_directory = default_directory or "E:\\PLAYLISTS"

        # Generate default name if not provided
        if not default_name:
            date_str = datetime.now().strftime("%Y-%m-%d")
            if artist_name:
                default_name = f"Auto - {artist_name} {date_str}"
            else:
                default_name = f"Auto - Playlist {date_str}"

        self._setup_ui(default_name)

    def _setup_ui(self, default_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ─────────────────────────────────────────────────────────────────────
        # Playlist Name
        # ─────────────────────────────────────────────────────────────────────
        name_group = QGroupBox("Playlist Name")
        name_layout = QVBoxLayout(name_group)

        self._name_edit = QLineEdit(default_name)
        self._name_edit.setPlaceholderText("Enter playlist name...")
        self._name_edit.textChanged.connect(self._update_preview)
        name_layout.addWidget(self._name_edit)

        layout.addWidget(name_group)

        # ─────────────────────────────────────────────────────────────────────
        # Export Directory
        # ─────────────────────────────────────────────────────────────────────
        dir_group = QGroupBox("Export Directory")
        dir_layout = QVBoxLayout(dir_group)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(self._default_directory)
        self._dir_edit.setPlaceholderText("Select export directory...")
        self._dir_edit.textChanged.connect(self._update_preview)
        dir_row.addWidget(self._dir_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(browse_btn)

        dir_layout.addLayout(dir_row)
        layout.addWidget(dir_group)

        # ─────────────────────────────────────────────────────────────────────
        # File Preview
        # ─────────────────────────────────────────────────────────────────────
        preview_group = QGroupBox("File Path Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_label = QLabel()
        self._preview_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 8px;
                font-family: monospace;
                color: #333;
            }
        """)
        self._preview_label.setWordWrap(True)
        preview_layout.addWidget(self._preview_label)

        layout.addWidget(preview_group)

        # ─────────────────────────────────────────────────────────────────────
        # Buttons
        # ─────────────────────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.setDefault(True)
        self._export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self._export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self._export_btn)

        layout.addLayout(btn_layout)

        # Initial preview update
        self._update_preview()

    def _browse_directory(self) -> None:
        """Open directory browser."""
        current_dir = self._dir_edit.text() or self._default_directory

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            current_dir,
            QFileDialog.ShowDirsOnly
        )

        if directory:
            self._dir_edit.setText(directory)

    def _update_preview(self) -> None:
        """Update the file path preview."""
        name = self._name_edit.text().strip()
        directory = self._dir_edit.text().strip()

        if name and directory:
            # Sanitize filename
            safe_name = self._sanitize_filename(name)
            full_path = Path(directory) / f"{safe_name}.m3u8"
            self._preview_label.setText(str(full_path))
            self._export_btn.setEnabled(True)
        else:
            self._preview_label.setText("(Enter name and directory)")
            self._export_btn.setEnabled(False)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove invalid filename characters."""
        invalid_chars = '<>:"/\\|?*'
        result = name
        for char in invalid_chars:
            result = result.replace(char, '_')
        return result.strip()

    def _on_export(self) -> None:
        """Validate and accept the dialog."""
        directory = self._dir_edit.text().strip()

        # Check if directory exists
        if not Path(directory).exists():
            reply = QMessageBox.question(
                self,
                "Create Directory?",
                f"Directory does not exist:\n{directory}\n\nCreate it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to create directory:\n{e}"
                    )
                    return
            else:
                return

        self.accept()

    def get_export_path(self) -> Optional[Path]:
        """Get the full export file path."""
        name = self._name_edit.text().strip()
        directory = self._dir_edit.text().strip()

        if name and directory:
            safe_name = self._sanitize_filename(name)
            return Path(directory) / f"{safe_name}.m3u8"
        return None

    def get_playlist_name(self) -> str:
        """Get the playlist name."""
        return self._name_edit.text().strip()


class ExportPlexDialog(QDialog):
    """
    Dialog for exporting playlist to Plex.

    Features:
    - Playlist name input with default "Auto - <artist> <date>"
    - Option to replace existing playlist with same name
    """

    def __init__(
        self,
        parent=None,
        default_name: str = "",
        artist_name: str = "",
        plex_configured: bool = True
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Playlist to Plex")
        self.setMinimumWidth(400)
        self.setModal(True)

        self._artist_name = artist_name
        self._plex_configured = plex_configured

        # Generate default name if not provided
        if not default_name:
            date_str = datetime.now().strftime("%Y-%m-%d")
            if artist_name:
                default_name = f"Auto - {artist_name} {date_str}"
            else:
                default_name = f"Auto - Playlist {date_str}"

        self._setup_ui(default_name)

    def _setup_ui(self, default_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Warning if Plex not configured
        if not self._plex_configured:
            warning_label = QLabel(
                "Plex is not configured. Please set plex.enabled=true and "
                "configure plex.base_url and PLEX_TOKEN in your config."
            )
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3cd;
                    border: 1px solid #ffc107;
                    border-radius: 4px;
                    padding: 10px;
                    color: #856404;
                }
            """)
            layout.addWidget(warning_label)

        # ─────────────────────────────────────────────────────────────────────
        # Playlist Name
        # ─────────────────────────────────────────────────────────────────────
        name_group = QGroupBox("Playlist Name")
        name_layout = QVBoxLayout(name_group)

        self._name_edit = QLineEdit(default_name)
        self._name_edit.setPlaceholderText("Enter playlist name...")
        self._name_edit.textChanged.connect(self._update_state)
        name_layout.addWidget(self._name_edit)

        info_label = QLabel(
            "If a playlist with this name already exists, it will be replaced."
        )
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        name_layout.addWidget(info_label)

        layout.addWidget(name_group)

        # ─────────────────────────────────────────────────────────────────────
        # Buttons
        # ─────────────────────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._export_btn = QPushButton("Export to Plex")
        self._export_btn.setDefault(True)
        self._export_btn.setStyleSheet("""
            QPushButton {
                background-color: #e5a00d;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cc8a00;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self._export_btn.clicked.connect(self.accept)
        self._export_btn.setEnabled(self._plex_configured)
        btn_layout.addWidget(self._export_btn)

        layout.addLayout(btn_layout)

        self._update_state()

    def _update_state(self) -> None:
        """Update button enabled state."""
        name = self._name_edit.text().strip()
        self._export_btn.setEnabled(bool(name) and self._plex_configured)

    def get_playlist_name(self) -> str:
        """Get the playlist name."""
        return self._name_edit.text().strip()
