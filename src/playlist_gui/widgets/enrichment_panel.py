"""Panel showing per-artist genre enrichment status and triggering enrichment runs."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


class EnrichmentPanel(QWidget):
    """Compact panel: shows artist + enrichment status + Enrich button.

    Signals:
        enrich_requested(str): User clicked Enrich. Argument is the artist name.
    """

    enrich_requested = Signal(str)

    def __init__(self, *, sidecar_db_path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._sidecar_db_path = sidecar_db_path
        self._artist: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.artist_label = QLabel("(no artist)")
        self.artist_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.artist_label)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.enrich_button = QPushButton("Enrich genres")
        self.enrich_button.clicked.connect(self._on_enrich_clicked)
        button_row.addStretch(1)
        button_row.addWidget(self.enrich_button)
        layout.addLayout(button_row)

    def set_artist(self, artist: str) -> None:
        self._artist = artist
        self.artist_label.setText(artist or "(no artist)")
        self._refresh_status()

    def set_running(self, running: bool) -> None:
        self.enrich_button.setEnabled(not running)
        if running:
            self.status_label.setText("Enriching...")

    def refresh(self) -> None:
        """Re-read enrichment status from the sidecar."""
        self._refresh_status()

    def _refresh_status(self) -> None:
        if not self._artist:
            self.status_label.setText("")
            return
        resolver = EnrichedGenreResolver(self._sidecar_db_path)
        status = resolver.get_artist_enrichment_status(self._artist)
        count = status["enriched_count"]
        if count == 0:
            self.status_label.setText("Not enriched")
        else:
            self.status_label.setText(f"{count} album(s) enriched")

    def _on_enrich_clicked(self) -> None:
        if self._artist:
            self.enrich_requested.emit(self._artist)
