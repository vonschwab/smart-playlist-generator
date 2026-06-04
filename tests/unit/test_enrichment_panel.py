"""Test EnrichmentPanel: status display and enrich button."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def _seed_sidecar(sidecar_path: Path, artist: str, albums: list[str]) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as conn:
        for album in albums:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    f"{artist}::{album}",
                    artist,
                    album,
                    None,
                    json.dumps({"genres": ["slowcore"], "sources": []}),
                    "2026-05-28T00:00:00",
                ),
            )
        conn.commit()


def test_panel_shows_enrichment_status_for_artist(tmp_path, app):
    from src.playlist_gui.widgets.enrichment_panel import EnrichmentPanel

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar, "duster", ["stratosphere", "together"])

    panel = EnrichmentPanel(sidecar_db_path=str(sidecar))
    panel.set_artist("Duster")

    assert panel.artist_label.text() == "Duster"
    assert "2 album" in panel.status_label.text().lower()


def test_panel_emits_enrich_requested_when_button_clicked(tmp_path, app):
    from src.playlist_gui.widgets.enrichment_panel import EnrichmentPanel

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar, "duster", [])

    panel = EnrichmentPanel(sidecar_db_path=str(sidecar))
    panel.set_artist("Duster")

    received: list[str] = []
    panel.enrich_requested.connect(lambda artist: received.append(artist))
    panel.enrich_button.click()

    assert received == ["Duster"]


def test_panel_disables_button_while_running(tmp_path, app):
    from src.playlist_gui.widgets.enrichment_panel import EnrichmentPanel

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar, "duster", [])

    panel = EnrichmentPanel(sidecar_db_path=str(sidecar))
    panel.set_artist("Duster")
    panel.set_running(True)
    assert not panel.enrich_button.isEnabled()
    assert "hybrid" in panel.status_label.text().lower()
    panel.set_running(False)
    assert panel.enrich_button.isEnabled()
