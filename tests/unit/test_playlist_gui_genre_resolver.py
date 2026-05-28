"""Test that the GUI worker uses EnrichedGenreResolver to populate playlist genres."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


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
                "2026-05-28T00:00:00",
            ),
        )
        conn.commit()


def test_resolve_track_genres_prefers_enriched(tmp_path):
    """When an album is enriched, _resolve_track_genres returns enriched genres."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar_path)

    track = {"artist": "Duster", "album": "Stratosphere", "rating_key": "t1"}
    fallback = lambda: ["indie rock", "rock"]

    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=fallback,
    )
    assert result == ["slowcore", "space rock"]


def test_resolve_track_genres_falls_back_when_unenriched(tmp_path):
    """When an album is not enriched, falls back to the provided source."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar_path)

    track = {"artist": "Unknown", "album": "Album", "rating_key": "t1"}
    fallback = lambda: ["indie rock", "rock"]

    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=fallback,
    )
    assert result == ["indie rock", "rock"]


def test_resolve_track_genres_no_sidecar_uses_fallback(tmp_path):
    """When sidecar DB doesn't exist, always falls back."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "nonexistent.db"

    track = {"artist": "Duster", "album": "Stratosphere", "rating_key": "t1"}
    fallback = lambda: ["indie rock", "rock"]

    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=fallback,
    )
    assert result == ["indie rock", "rock"]
