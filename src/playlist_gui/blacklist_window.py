"""Blacklist window for viewing and managing blacklisted tracks."""
from __future__ import annotations

from typing import Callable, List, Dict, Any, Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .widgets.track_table import TrackTable
from .worker_client import WorkerClient


class BlacklistWindow(QDialog):
    """Dialog for viewing and removing blacklisted tracks."""

    def __init__(
        self,
        worker_client: WorkerClient,
        config_path_provider: Callable[[], str],
        overrides_provider: Callable[[], dict],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._worker_client = worker_client
        self._config_path_provider = config_path_provider
        self._overrides_provider = overrides_provider
        self._track_table = TrackTable(self)

        self.setWindowTitle("Blacklisted Tracks")
        self.resize(900, 600)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel("Blacklisted tracks (removed from all playlists)")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        layout.addWidget(self._track_table)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(self._refresh_btn)

        self._remove_btn = QPushButton("Remove from blacklist")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch(1)

        layout.addLayout(btn_row)

    def refresh(self) -> None:
        """Request a refresh of the blacklist table."""
        if not self._worker_client:
            return
        config_path = self._config_path_provider()
        overrides = self._overrides_provider()
        self._worker_client.fetch_blacklist(config_path, overrides)

    @Slot()
    def _on_remove_clicked(self) -> None:
        selected = self._track_table.get_selected_tracks()
        if not selected:
            QMessageBox.information(self, "Remove from blacklist", "No tracks selected.")
            return
        track_ids = []
        for track in selected:
            tid = track.get("rating_key") or track.get("track_id") or track.get("id")
            if tid:
                track_ids.append(str(tid))
        if not track_ids:
            QMessageBox.information(self, "Remove from blacklist", "No valid track ids found.")
            return
        config_path = self._config_path_provider()
        overrides = self._overrides_provider()
        self._worker_client.set_blacklisted(config_path, track_ids, False, overrides)

    def handle_blacklist_result(self, data: Dict[str, Any]) -> None:
        tracks = data.get("tracks", [])
        self._track_table.set_tracks(tracks, playlist_name="Blacklist")

    def handle_blacklist_set_result(self, data: Dict[str, Any]) -> None:
        value = bool(data.get("value", True))
        if value:
            return
        self.refresh()
