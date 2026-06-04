from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeWorkerClient(QObject):
    busy_changed = Signal(bool)
    result_received = Signal(str, dict, object)
    done_received = Signal(str, bool, str, bool, object, str)

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        self._running = True
        return True

    def enrich_genres(self, *, scope: str, artist: str = "", album: str = "", job_id=None):
        self.calls.append(("enrich_genres", scope, artist, album))
        return "req-1"

    def edit_genres(self, artist: str, album: str, genres: list, job_id=None):
        self.calls.append(("edit_genres", artist, album, genres))
        return "req-2"


def _seed_sidecar(sidecar_path: Path) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "duster::stratosphere",
                "duster",
                "stratosphere",
                None,
                json.dumps({"genres": ["slowcore", "space rock"], "sources": []}),
                "2026-06-03T00:00:00",
            ),
        )
        conn.commit()


def test_window_runs_scoped_enrichment_commands(tmp_path, app):
    from src.playlist_gui.widgets.genre_enrichment_window import GenreEnrichmentWindow

    worker = FakeWorkerClient()
    window = GenreEnrichmentWindow(worker, sidecar_db_path=str(tmp_path / "sidecar.db"))
    window.artist_edit.setText("Duster")
    window.album_edit.setText("Stratosphere")

    window.full_scan_button.click()
    window.artist_button.click()
    window.album_button.click()

    assert worker.calls == [
        ("enrich_genres", "all_unenriched", "", ""),
        ("enrich_genres", "artist", "Duster", ""),
        ("enrich_genres", "album", "Duster", "Stratosphere"),
    ]


def test_window_loads_and_saves_selected_release_genres(tmp_path, app):
    from src.playlist_gui.widgets.genre_enrichment_window import GenreEnrichmentWindow

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar)
    worker = FakeWorkerClient()
    window = GenreEnrichmentWindow(worker, sidecar_db_path=str(sidecar))

    assert window.results_table.rowCount() == 1
    window.results_table.selectRow(0)
    assert "slowcore" in window.genre_text.toPlainText()

    window.genre_text.setPlainText("slowcore\nshoegaze\nslowcore\n")
    window.save_button.click()

    assert worker.calls[-1] == (
        "edit_genres",
        "duster",
        "stratosphere",
        ["slowcore", "shoegaze"],
    )
