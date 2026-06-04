from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class GenreEnrichmentWindow(QDialog):
    """Dedicated GUI surface for hybrid genre enrichment and direct edits."""

    def __init__(
        self,
        worker_client: Any,
        *,
        sidecar_db_path: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self._worker_client = worker_client
        self._sidecar_db_path = Path(sidecar_db_path)
        self._rows: list[dict[str, Any]] = []

        self.setWindowTitle("Enrich Genres")
        self.setMinimumSize(820, 560)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        controls = QGroupBox("Run Enrichment")
        controls_layout = QGridLayout(controls)
        controls_layout.addWidget(QLabel("Artist"), 0, 0)
        self.artist_edit = QLineEdit()
        self.artist_edit.setObjectName("genreEnrichmentArtistEdit")
        controls_layout.addWidget(self.artist_edit, 0, 1)
        controls_layout.addWidget(QLabel("Album"), 1, 0)
        self.album_edit = QLineEdit()
        self.album_edit.setObjectName("genreEnrichmentAlbumEdit")
        controls_layout.addWidget(self.album_edit, 1, 1)

        self.full_scan_button = QPushButton("Full Scan Unenriched")
        self.full_scan_button.setObjectName("genreEnrichmentFullScanButton")
        self.artist_button = QPushButton("Enrich Artist")
        self.artist_button.setObjectName("genreEnrichmentArtistButton")
        self.album_button = QPushButton("Enrich Album")
        self.album_button.setObjectName("genreEnrichmentAlbumButton")
        button_row = QHBoxLayout()
        button_row.addWidget(self.full_scan_button)
        button_row.addWidget(self.artist_button)
        button_row.addWidget(self.album_button)
        button_row.addStretch(1)
        controls_layout.addLayout(button_row, 2, 0, 1, 2)
        layout.addWidget(controls)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("genreEnrichmentStatusLabel")
        layout.addWidget(self.status_label)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setObjectName("genreEnrichmentResultsTable")
        self.results_table.setHorizontalHeaderLabels(["Artist", "Album", "Genres"])
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, stretch=1)

        editor = QGroupBox("Edit Selected Release")
        editor_layout = QVBoxLayout(editor)
        self.editor_label = QLabel("Select a release to edit its resolved genres.")
        self.editor_label.setWordWrap(True)
        editor_layout.addWidget(self.editor_label)
        self.genre_text = QPlainTextEdit()
        self.genre_text.setObjectName("genreEnrichmentGenreText")
        self.genre_text.setPlaceholderText("One genre per line")
        editor_layout.addWidget(self.genre_text, stretch=1)
        save_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.save_button = QPushButton("Save Genres")
        self.save_button.setObjectName("genreEnrichmentSaveButton")
        save_row.addStretch(1)
        save_row.addWidget(self.refresh_button)
        save_row.addWidget(self.save_button)
        editor_layout.addLayout(save_row)
        layout.addWidget(editor, stretch=1)

        self.full_scan_button.clicked.connect(self._run_full_scan)
        self.artist_button.clicked.connect(self._run_artist)
        self.album_button.clicked.connect(self._run_album)
        self.refresh_button.clicked.connect(self.refresh_results)
        self.save_button.clicked.connect(self._save_selected_genres)
        self.results_table.itemSelectionChanged.connect(self._on_selection_changed)

        if hasattr(worker_client, "busy_changed"):
            worker_client.busy_changed.connect(self._set_busy)
        if hasattr(worker_client, "result_received"):
            worker_client.result_received.connect(self._on_worker_result)
        if hasattr(worker_client, "done_received"):
            worker_client.done_received.connect(self._on_worker_done)

        self.refresh_results()

    def set_artist(self, artist: str) -> None:
        self.artist_edit.setText(artist or "")

    @Slot()
    def refresh_results(self) -> None:
        self._rows = self._load_enriched_rows()
        self.results_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = [
                row["artist"],
                row["album"],
                ", ".join(row["genres"]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row_index)
                self.results_table.setItem(row_index, column, item)
        self.results_table.resizeColumnsToContents()
        self.status_label.setText(f"{len(self._rows)} enriched release(s) loaded")

    @Slot()
    def _run_full_scan(self) -> None:
        if not self._ensure_worker_running():
            return
        self.status_label.setText("Starting full scan for unenriched albums...")
        self._worker_client.enrich_genres(scope="all_unenriched")

    @Slot()
    def _run_artist(self) -> None:
        artist = self.artist_edit.text().strip()
        if not artist:
            QMessageBox.warning(self, "Artist Required", "Enter an artist to enrich.")
            return
        if not self._ensure_worker_running():
            return
        self.status_label.setText(f"Starting enrichment for artist: {artist}")
        self._worker_client.enrich_genres(scope="artist", artist=artist)

    @Slot()
    def _run_album(self) -> None:
        artist = self.artist_edit.text().strip()
        album = self.album_edit.text().strip()
        if not artist or not album:
            QMessageBox.warning(self, "Artist and Album Required", "Enter both artist and album.")
            return
        if not self._ensure_worker_running():
            return
        self.status_label.setText(f"Starting enrichment for: {artist} / {album}")
        self._worker_client.enrich_genres(scope="album", artist=artist, album=album)

    @Slot()
    def _save_selected_genres(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.warning(self, "No Release Selected", "Select a release before saving genres.")
            return
        genres = self._clean_genres(self.genre_text.toPlainText().splitlines())
        if not genres:
            QMessageBox.warning(self, "No Genres", "At least one genre is required.")
            return
        if not self._ensure_worker_running():
            return
        self.status_label.setText(f"Saving genres for: {row['artist']} / {row['album']}")
        self._worker_client.edit_genres(row["artist"], row["album"], genres)

    @Slot()
    def _on_selection_changed(self) -> None:
        row = self._selected_row()
        if row is None:
            self.editor_label.setText("Select a release to edit its resolved genres.")
            self.genre_text.clear()
            return
        self.artist_edit.setText(row["artist"])
        self.album_edit.setText(row["album"])
        self.editor_label.setText(f"<b>{row['artist']} / {row['album']}</b>")
        self.genre_text.setPlainText("\n".join(row["genres"]))

    @Slot(bool)
    def _set_busy(self, is_busy: bool) -> None:
        for button in (self.full_scan_button, self.artist_button, self.album_button, self.save_button):
            button.setEnabled(not is_busy)

    @Slot(str, dict, object)
    def _on_worker_result(self, result_type: str, data: dict, job_id: object = None) -> None:
        if result_type in {"enrich_genres", "enrich_artist", "edit_genres"}:
            self.refresh_results()
            if result_type == "enrich_genres":
                applied = data.get("applied", 0)
                releases = data.get("releases", 0)
                self.status_label.setText(f"Enriched {releases} release(s); applied {applied} genre(s).")

    @Slot(str, bool, str, bool, object, str)
    def _on_worker_done(
        self,
        cmd: str,
        ok: bool,
        detail: str,
        cancelled: bool,
        job_id: object = None,
        summary: str = "",
    ) -> None:
        if cmd not in {"enrich_genres", "enrich_artist", "edit_genres"}:
            return
        if cancelled:
            self.status_label.setText("Cancelled")
        elif ok:
            self.refresh_results()
            self.status_label.setText(summary or detail or "Complete")
        else:
            self.status_label.setText(f"Failed: {detail}")

    def _ensure_worker_running(self) -> bool:
        if hasattr(self._worker_client, "is_running") and not self._worker_client.is_running():
            if not self._worker_client.start():
                QMessageBox.critical(self, "Worker Error", "Failed to start worker process.")
                return False
        return True

    def _selected_row(self) -> dict[str, Any] | None:
        selected = self.results_table.selectedItems()
        if not selected:
            return None
        row_index = selected[0].data(Qt.UserRole)
        if row_index is None:
            row_index = selected[0].row()
        try:
            return self._rows[int(row_index)]
        except (IndexError, TypeError, ValueError):
            return None

    def _load_enriched_rows(self) -> list[dict[str, Any]]:
        if not self._sidecar_db_path.exists():
            return []
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

        resolver = EnrichedGenreResolver(self._sidecar_db_path)
        uri = self._sidecar_db_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT normalized_artist, normalized_album, signature_json "
                    "FROM enriched_genre_signatures "
                    "ORDER BY normalized_artist, normalized_album"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["signature_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            artist = row["normalized_artist"] or ""
            album = row["normalized_album"] or ""
            genres = resolver.get_enriched_genres(artist=artist, album=album)
            if genres is None:
                genres = self._clean_genres(payload.get("genres") or [])
            result.append(
                {
                    "artist": artist,
                    "album": album,
                    "genres": list(genres),
                }
            )
        return result

    @staticmethod
    def _clean_genres(genres: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for genre in genres:
            value = str(genre).strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
        return cleaned
