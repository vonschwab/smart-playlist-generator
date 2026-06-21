"""Tests for the handle_edit_genres worker command."""

from __future__ import annotations

import json


from src.playlist_gui.worker import handle_edit_genres
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


def _seed_signature(store, release_key, na, nb, genres):
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (release_key, na, nb, None,
             json.dumps({"genres": genres, "sources": []}), "2026-05-28"),
        )
        conn.commit()


def test_edit_genres_handler_writes_override(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber",
                    ["idm", "glitch", "warp"])

    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))

    cmd = {
        "cmd": "edit_genres",
        "request_id": "r1",
        "artist": "Autechre",
        "album": "Amber",
        "genres": ["idm", "glitch", "modular synthesizer"],
        "base_genres": ["idm", "glitch", "warp"],
    }
    handle_edit_genres(cmd)

    override = store.get_user_override("autechre::amber")
    assert override is not None
    assert "modular synthesizer" in override["genres_add"]
    assert "warp" in override["genres_remove"]

    resolver = EnrichedGenreResolver(str(sidecar))
    new_genres = set(resolver.get_enriched_genres(artist="Autechre", album="Amber"))
    assert new_genres == {"idm", "glitch", "modular synthesizer"}


def test_edit_genres_removes_graph_genre_against_client_baseline(tmp_path, monkeypatch):
    """Regression: a genre shown only via the graph authority (no sidecar
    signature) must still be removable. The diff uses the client-sent
    base_genres — what the GUI actually displayed — not the sidecar resolver,
    which returns nothing for a graph-only album and would silently drop the
    removal."""
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    # Deliberately NO enriched_genre_signatures row: the resolver sees nothing,
    # mirroring an album whose displayed chips come from release_effective_genres.
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))

    cmd = {
        "cmd": "edit_genres",
        "request_id": "r1",
        "artist": "Low",
        "album": "Hey What",
        "genres": ["slowcore"],                     # user removed "dream pop"
        "base_genres": ["slowcore", "dream pop"],   # the displayed graph genres
    }
    handle_edit_genres(cmd)

    override = store.get_user_override("low::hey what")
    assert override is not None
    assert "dream pop" in override["genres_remove"]
    assert override["genres_add"] == []


def test_edit_genres_no_artist_is_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(tmp_path / "x.db"))
    handle_edit_genres({"cmd": "edit_genres", "artist": "", "album": "Amber", "genres": []})
    out = capsys.readouterr().out
    events = [json.loads(line) for line in out.strip().splitlines() if line.strip()]
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is False
