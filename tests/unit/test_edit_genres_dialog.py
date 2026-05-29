"""Tests for EditGenresDialog and TrackTable edit-genres wiring."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.gui


def test_dialog_emits_committed_with_normalized_genres(qtbot):
    from PySide6.QtWidgets import QDialogButtonBox
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog

    dialog = EditGenresDialog(
        artist="Autechre",
        album="Amber",
        current_genres=["idm", "glitch", "warp"],
    )
    qtbot.addWidget(dialog)

    dialog.set_text("idm\nglitch\nmodular synthesizer\n")

    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append((artist, album, genres)))

    dialog._on_save_clicked()
    assert captured == [("Autechre", "Amber", ["idm", "glitch", "modular synthesizer"])]


def test_dialog_strips_empty_lines_and_whitespace(qtbot):
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog

    dialog = EditGenresDialog(artist="X", album="Y", current_genres=["a"])
    qtbot.addWidget(dialog)
    dialog.set_text("  ambient  \n\n  drone \n\n")

    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append(genres))
    dialog._on_save_clicked()
    assert captured == [["ambient", "drone"]]


def test_dialog_deduplicates(qtbot):
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog

    dialog = EditGenresDialog(artist="X", album="Y", current_genres=["a"])
    qtbot.addWidget(dialog)
    dialog.set_text("idm\nIDM\nidm")
    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append(genres))
    dialog._on_save_clicked()
    assert captured == [["idm"]]


def test_dialog_rejects_empty_save(qtbot, monkeypatch):
    """Saving with all lines blank must NOT emit genres_committed."""
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
    dialog = EditGenresDialog(artist="X", album="Y", current_genres=["a"])
    qtbot.addWidget(dialog)
    dialog.set_text("   \n\n\n")

    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append(genres))
    dialog._on_save_clicked()
    assert captured == []


def test_track_table_emits_edit_genres_for_single_album_selection(qtbot):
    """Single-album selection exposes Edit genres action that emits the signal."""
    from src.playlist_gui.widgets.track_table import TrackTable

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks([
        {"position": 1, "artist": "Autechre", "album": "Amber", "title": "Foil"},
    ])
    table.select_row(0)

    captured = []
    table.edit_genres_requested.connect(lambda payload: captured.append(payload))

    action = table._build_edit_genres_action_for_selection()
    assert action is not None
    action.trigger()
    assert captured == [{"artist": "Autechre", "album": "Amber"}]


def test_track_table_no_edit_genres_when_mixed_albums(qtbot):
    """Mixed-album selection returns None from the helper."""
    from src.playlist_gui.widgets.track_table import TrackTable

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks([
        {"position": 1, "artist": "Autechre", "album": "Amber", "title": "Foil"},
        {"position": 2, "artist": "Stereolab", "album": "Emperor", "title": "Cybele"},
    ])
    table._table.selectAll()

    action = table._build_edit_genres_action_for_selection()
    assert action is None


def test_refresh_genres_for_album_updates_matching_rows(qtbot, tmp_path, monkeypatch):
    """After committing edits, the table Genres column reflects the new genres."""
    from src.playlist_gui.widgets.track_table import TrackTable
    from src.ai_genre_enrichment.storage import SidecarStore

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("autechre::amber", "autechre", "amber", None,
             json.dumps({"genres": ["idm"], "sources": []}), "2026-05-28"),
        )
        conn.commit()

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks([
        {"position": 1, "artist": "Autechre", "album": "Amber",
         "title": "Foil", "genres": ["idm"]},
        {"position": 2, "artist": "Other", "album": "Different",
         "title": "X", "genres": ["jazz"]},
    ])

    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber",
        genres_add=["modular synthesizer"], genres_remove=[],
    )
    monkeypatch.setattr(
        "src.playlist_gui.widgets.track_table.SIDECAR_DB_PATH",
        str(sidecar),
    )

    table.refresh_genres_for_album(artist="Autechre", album="Amber")

    rows = table.get_tracks()
    assert set(rows[0]["genres"]) == {"idm", "modular synthesizer"}
    assert rows[1]["genres"] == ["jazz"]
