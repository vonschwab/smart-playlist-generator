"""
Track Table - Power table for playlist tracks (Foobar2000-style)

Features:
- QTableView + model/view architecture for performance
- Sorting by clicking column headers
- Quick filter/search box
- Right-click context menu (copy, open, export)
- Keyboard shortcuts (Ctrl+F, Ctrl+C, Esc)
- Double-click to open file
"""
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal, Slot, QRegularExpression
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QClipboard
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .track_table_model import Column, TrackTableModel, TrackFilterProxyModel


class TrackTable(QWidget):
    """
    Power table widget for displaying playlist tracks.

    Features:
        - Sortable columns (click header)
        - Quick filter across artist/title/album
        - Right-click context menu with copy/open/export actions
        - Keyboard shortcuts (Ctrl+F, Ctrl+C, double-click)

    Columns:
        #        - Track position
        Artist   - Artist name
        Title    - Track title
        Album    - Album name
        Duration - Track duration (M:SS)
        File Path - Full file path

    Signals:
        track_selected: Emitted when a track is selected (position, track_data)
        track_double_clicked: Emitted on double-click (position, track_data)
        status_changed: Emitted when filter status changes (visible_count, total_count)
    """

    track_selected = Signal(int, dict)
    track_double_clicked = Signal(int, dict)
    status_changed = Signal(int, int)  # visible_count, total_count
    blacklist_requested = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        # Ensure a QApplication exists (tests may construct this widget outside
        # the normal GUI boot path).
        self._owned_app = None
        if QApplication.instance() is None:
            self._owned_app = QApplication([])

        super().__init__(parent)

        # Settings
        self._double_click_opens_file = True
        self._playlist_name = "Playlist"
        # Optional column visibility (default hidden to keep layout compact)
        self._column_visibility = {
            Column.SONIC_SIM: False,
            Column.GENRE_SIM: False,
            Column.GENRES: False,
            Column.FILE_PATH: True,  # still shown by default but toggleable
        }

        # Setup models
        self._model = TrackTableModel(self)
        self._proxy = TrackFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ─────────────────────────────────────────────────────────────────────
        # Filter bar
        # ─────────────────────────────────────────────────────────────────────
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(4)

        # Filter icon/label
        filter_label = QLabel("Filter:")
        filter_layout.addWidget(filter_label)

        # Filter input
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Type to filter tracks...")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._filter_edit, stretch=1)

        # Search path checkbox
        self._search_path_check = QCheckBox("Include path")
        self._search_path_check.setToolTip("Include file path in filter matching")
        self._search_path_check.toggled.connect(self._on_search_path_toggled)
        filter_layout.addWidget(self._search_path_check)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        filter_layout.addWidget(self._status_label)

        layout.addLayout(filter_layout)

        # ─────────────────────────────────────────────────────────────────────
        # Table view
        # ─────────────────────────────────────────────────────────────────────
        self._table = QTableView()
        self._table.setModel(self._proxy)

        # Configure selection
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setAlternatingRowColors(True)

        # Configure editing
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Configure sorting
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(Column.INDEX, Qt.AscendingOrder)

        # Configure header
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)

        # Column resize modes
        header.setSectionResizeMode(Column.INDEX, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(Column.ARTIST, QHeaderView.Interactive)
        header.setSectionResizeMode(Column.TITLE, QHeaderView.Stretch)
        header.setSectionResizeMode(Column.ALBUM, QHeaderView.Interactive)
        header.setSectionResizeMode(Column.DURATION, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(Column.SONIC_SIM, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(Column.GENRE_SIM, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(Column.GENRES, QHeaderView.Interactive)
        header.setSectionResizeMode(Column.FILE_PATH, QHeaderView.Interactive)

        # Set default column widths
        self._table.setColumnWidth(Column.ARTIST, 150)
        self._table.setColumnWidth(Column.ALBUM, 150)
        self._table.setColumnWidth(Column.SONIC_SIM, 90)
        self._table.setColumnWidth(Column.GENRE_SIM, 90)
        self._table.setColumnWidth(Column.GENRES, 220)
        self._table.setColumnWidth(Column.FILE_PATH, 300)
        self._apply_column_visibility()

        # Vertical header (row numbers) hidden
        self._table.verticalHeader().setVisible(False)

        # Connect signals
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._table.doubleClicked.connect(self._on_double_click)

        # Context menu
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self._table)

    # Persistence helpers
    def get_filter_text(self) -> str:
        """Return current filter text."""
        return self._filter_edit.text()

    def set_filter_text(self, text: str) -> None:
        """Restore filter text."""
        self._filter_edit.setText(text or "")

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        # Ctrl+F focuses filter
        shortcut_filter = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut_filter.activated.connect(self._focus_filter)

        # Ctrl+C copies selected
        shortcut_copy = QShortcut(QKeySequence("Ctrl+C"), self)
        shortcut_copy.activated.connect(self._copy_selected_artist_title)

        # Esc clears filter (when filter focused)
        self._filter_edit.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        """Handle events for filter edit (Esc to clear)."""
        if obj == self._filter_edit:
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.KeyPress:
                from PySide6.QtGui import QKeyEvent
                key_event: QKeyEvent = event
                if key_event.key() == Qt.Key_Escape:
                    self._filter_edit.clear()
                    return True
        return super().eventFilter(obj, event)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_tracks(self, tracks: List[Dict[str, Any]], playlist_name: str = "") -> None:
        """
        Set the tracks to display.

        Args:
            tracks: List of track dicts with keys: position, artist, title, album, duration_ms, file_path
            playlist_name: Optional playlist name for export filename
        """
        if playlist_name:
            self._playlist_name = playlist_name

        # Preserve sort column and order
        sort_col = self._proxy.sortColumn()
        sort_order = self._proxy.sortOrder()

        self._model.set_tracks(tracks)

        # Restore sort
        if sort_col >= 0:
            self._proxy.sort(sort_col, sort_order)

        self._update_status()

    def clear(self) -> None:
        """Clear all tracks."""
        self._model.clear()
        self._update_status()

    def get_tracks(self) -> List[Dict[str, Any]]:
        """Get all tracks (unfiltered)."""
        return self._model.get_tracks()

    def mark_blacklisted(self, track_ids: set[str], value: bool) -> int:
        """Update blacklisted flag in the table model."""
        updated = self._model.mark_blacklisted(track_ids, value)
        if updated:
            self._update_status()
        return updated

    def get_visible_tracks(self) -> List[Dict[str, Any]]:
        """Get currently visible (filtered) tracks."""
        return self._proxy.get_visible_tracks()

    def get_selected_tracks(self) -> List[Dict[str, Any]]:
        """Get all selected tracks."""
        tracks = []
        indexes = self._table.selectionModel().selectedRows()
        for index in indexes:
            track = self._proxy.get_track(index.row())
            if track:
                tracks.append(track)
        return tracks

    def get_selected_track(self) -> Optional[Dict[str, Any]]:
        """Get the first selected track, or None."""
        tracks = self.get_selected_tracks()
        return tracks[0] if tracks else None

    def get_track_count(self) -> int:
        """Get the total number of tracks."""
        return self._model.rowCount()

    def scroll_to_top(self) -> None:
        """Scroll to the top of the table."""
        self._table.scrollToTop()

    def select_row(self, row: int) -> None:
        """Select a specific row."""
        if 0 <= row < self._proxy.rowCount():
            index = self._proxy.index(row, 0)
            self._table.selectionModel().select(
                index,
                self._table.selectionModel().ClearAndSelect | self._table.selectionModel().Rows
            )

    def focus_filter(self) -> None:
        """Focus the filter input."""
        self._filter_edit.setFocus()
        self._filter_edit.selectAll()

    def set_double_click_opens_file(self, enabled: bool) -> None:
        """Enable/disable double-click to open file."""
        self._double_click_opens_file = enabled

    # ─────────────────────────────────────────────────────────────────────────
    # Filter handlers
    # ─────────────────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_filter_changed(self, text: str) -> None:
        """Handle filter text change."""
        # Escape regex special characters for safety
        escaped = re.escape(text)
        self._proxy.setFilterRegularExpression(
            QRegularExpression(escaped, QRegularExpression.CaseInsensitiveOption)
        )
        self._update_status()

    @Slot(bool)
    def _on_search_path_toggled(self, checked: bool) -> None:
        """Handle search path checkbox toggle."""
        self._proxy.set_include_path_in_search(checked)
        self._update_status()

    def _update_status(self) -> None:
        """Update the status label."""
        visible = self._proxy.get_visible_count()
        total = self._proxy.get_total_count()

        if total == 0:
            self._status_label.setText("")
        elif visible == total:
            self._status_label.setText(f"{total} tracks")
        else:
            self._status_label.setText(f"Showing {visible} of {total} tracks")

        self.status_changed.emit(visible, total)

    @Slot()
    def _focus_filter(self) -> None:
        """Focus the filter input."""
        self.focus_filter()

    # ─────────────────────────────────────────────────────────────────────────
    # Selection handlers
    # ─────────────────────────────────────────────────────────────────────────

    @Slot()
    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        track = self.get_selected_track()
        if track:
            self.track_selected.emit(track.get("position", 0), track)

    @Slot()
    def _on_double_click(self, index) -> None:
        """Handle double-click on a row."""
        track = self._proxy.get_track(index.row())
        if track:
            self.track_double_clicked.emit(track.get("position", 0), track)

            if self._double_click_opens_file:
                self._open_file(track)

    # ─────────────────────────────────────────────────────────────────────────
    # Context menu
    # ─────────────────────────────────────────────────────────────────────────

    @Slot()
    def _show_context_menu(self, pos) -> None:
        """Show context menu at position."""
        global_pos = self._table.viewport().mapToGlobal(pos)
        menu = QMenu(self)

        selected = self.get_selected_tracks()
        has_selection = len(selected) > 0
        single_selection = len(selected) == 1

        # Copy submenu
        copy_menu = menu.addMenu("Copy")

        if single_selection:
            track = selected[0]
            copy_menu.addAction("Copy Artist", lambda: self._copy_text(track.get("artist", "")))
            copy_menu.addAction("Copy Title", lambda: self._copy_text(track.get("title", "")))
            copy_menu.addAction("Copy Album", lambda: self._copy_text(track.get("album", "")))
            copy_menu.addAction("Copy File Path", lambda: self._copy_text(track.get("file_path", "")))
            copy_menu.addSeparator()

        if has_selection:
            copy_menu.addAction(
                f"Copy {len(selected)} Rows as Artist - Title",
                self._copy_selected_artist_title
            )
            copy_menu.addAction(
                f"Copy {len(selected)} File Paths",
                self._copy_selected_paths
            )

        # Open submenu
        if single_selection:
            open_menu = menu.addMenu("Open")
            open_menu.addAction("Open File", lambda: self._open_file(selected[0]))
            open_menu.addAction("Open Containing Folder", lambda: self._open_folder(selected[0]))

        menu.addSeparator()

        # Export submenu
        export_menu = menu.addMenu("Export")
        if has_selection:
            export_menu.addAction(
                f"Export Selection ({len(selected)} tracks) as M3U8",
                lambda: self._export_m3u8(selected)
            )
        export_menu.addAction(
            "Export Whole Playlist as M3U8",
            lambda: self._export_m3u8(self.get_tracks())
        )

        if has_selection:
            menu.addSeparator()
            menu.addAction(
                f"Blacklist {len(selected)} Track(s)",
                lambda: self._confirm_blacklist(selected),
            )

        # Column visibility submenu
        columns_menu = menu.addMenu("Columns")
        for col, label in [
            (Column.SONIC_SIM, "Sonic Similarity"),
            (Column.GENRE_SIM, "Genre Similarity"),
            (Column.GENRES, "Genres"),
            (Column.FILE_PATH, "File Path"),
        ]:
            action = columns_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._column_visibility.get(col, True))
            action.triggered.connect(lambda checked, c=col: self._toggle_column(c, checked))

        menu.exec(global_pos)

    def _confirm_blacklist(self, selected: List[Dict[str, Any]]) -> None:
        """Confirm blacklist action and emit request."""
        if not selected:
            return
        reply = QMessageBox.question(
            self,
            "Blacklist Tracks",
            f"Blacklist {len(selected)} track(s)? They will no longer appear in generated playlists.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.blacklist_requested.emit(selected)

    def _apply_column_visibility(self) -> None:
        """Hide or show optional columns based on current settings."""
        for col, visible in self._column_visibility.items():
            self._table.setColumnHidden(col, not visible)

    def _toggle_column(self, column: int, visible: bool) -> None:
        """Toggle a column visibility and persist state."""
        self._column_visibility[column] = visible
        self._apply_column_visibility()

    # ─────────────────────────────────────────────────────────────────────────
    # Copy actions
    # ─────────────────────────────────────────────────────────────────────────

    def _copy_text(self, text: str) -> None:
        """Copy text to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    @Slot()
    def _copy_selected_artist_title(self) -> None:
        """Copy selected rows as 'Artist - Title' (newline separated)."""
        selected = self.get_selected_tracks()
        if not selected:
            return

        lines = []
        for track in selected:
            artist = track.get("artist", "Unknown")
            title = track.get("title", "Unknown")
            lines.append(f"{artist} - {title}")

        self._copy_text("\n".join(lines))

    @Slot()
    def _copy_selected_paths(self) -> None:
        """Copy selected file paths (newline separated)."""
        selected = self.get_selected_tracks()
        if not selected:
            return

        paths = [track.get("file_path", "") for track in selected if track.get("file_path")]
        self._copy_text("\n".join(paths))

    # ─────────────────────────────────────────────────────────────────────────
    # Open actions
    # ─────────────────────────────────────────────────────────────────────────

    def _open_file(self, track: Dict[str, Any]) -> None:
        """Open the track file with the default application."""
        file_path = track.get("file_path", "")
        if not file_path:
            return

        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"File does not exist:\n{file_path}")
            return

        # Use Qt's cross-platform open
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_folder(self, track: Dict[str, Any]) -> None:
        """Open the containing folder and select the file (Windows)."""
        file_path = track.get("file_path", "")
        if not file_path:
            return

        path = Path(file_path)
        if not path.exists():
            # Try to open parent folder at least
            parent = path.parent
            if parent.exists():
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(parent)))
            else:
                QMessageBox.warning(self, "Folder Not Found", f"Folder does not exist:\n{path.parent}")
            return

        # Windows: use explorer /select to highlight the file
        if os.name == 'nt':
            try:
                # explorer /select,"path" - quotes handle spaces
                subprocess.run(['explorer', '/select,', str(path)], check=False)
            except Exception:
                # Fallback to opening folder
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        else:
            # Non-Windows: just open the folder
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    # ─────────────────────────────────────────────────────────────────────────
    # Export actions
    # ─────────────────────────────────────────────────────────────────────────

    def _export_m3u8(self, tracks: List[Dict[str, Any]]) -> None:
        """Export tracks as M3U8 playlist file."""
        if not tracks:
            QMessageBox.information(self, "No Tracks", "No tracks to export.")
            return

        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', self._playlist_name)
        default_name = f"{safe_name}_{timestamp}.m3u8"

        # Show save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Playlist",
            default_name,
            "M3U8 Playlist (*.m3u8);;M3U Playlist (*.m3u);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self._write_m3u8(file_path, tracks)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(tracks)} tracks to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export playlist:\n{e}"
            )

    def _write_m3u8(self, file_path: str, tracks: List[Dict[str, Any]]) -> None:
        """
        Write tracks to M3U8 file.

        Args:
            file_path: Output file path
            tracks: List of track dicts
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")

            for track in tracks:
                duration_ms = track.get("duration_ms", 0)
                try:
                    duration_sec = int(duration_ms) // 1000
                except (ValueError, TypeError):
                    duration_sec = 0

                artist = track.get("artist", "Unknown")
                title = track.get("title", "Unknown")
                path = track.get("file_path", "")

                # Write EXTINF line
                f.write(f"#EXTINF:{duration_sec},{artist} - {title}\n")

                # Write path
                f.write(f"{path}\n")
