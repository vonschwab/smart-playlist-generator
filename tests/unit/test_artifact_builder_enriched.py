"""Test that the artifact builder uses enriched genres when a resolver is provided."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_metadata_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            album TEXT,
            album_id TEXT,
            is_blacklisted INTEGER DEFAULT 0,
            file_path TEXT
        );
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
    """)
    conn.execute("INSERT INTO tracks VALUES('t1', 'Duster', 'Stratosphere', 'a1', 0, '/music/t1.mp3')")
    conn.execute("INSERT INTO track_genres VALUES('t1', 'indie rock', 'file', 1.0)")
    conn.commit()
    conn.close()


def _make_sidecar(path: Path, signatures: list[tuple]) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore
    store = SidecarStore(str(path))
    store.initialize()
    with store.connect() as conn:
        for release_key, normalized_artist, normalized_album, genres in signatures:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (release_key, normalized_artist, normalized_album, None,
                 json.dumps({"genres": genres, "sources": []}), "2026-05-28T00:00:00"),
            )
        conn.commit()


def test_artifact_builder_uses_enriched_genres(tmp_path):
    """When a resolver is provided, genre_lists reflect enriched signatures."""
    from src.analyze.artifact_builder import collect_track_genres
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    genres = collect_track_genres(
        track_id="t1",
        metadata_db_path=str(metadata),
        enriched_resolver=resolver,
    )
    names = [g[0] for g in genres]
    assert sorted(names) == ["slowcore", "space rock"]


def test_artifact_builder_falls_back_when_no_signature(tmp_path):
    from src.analyze.artifact_builder import collect_track_genres
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [])

    resolver = EnrichedGenreResolver(str(sidecar))
    genres = collect_track_genres(
        track_id="t1",
        metadata_db_path=str(metadata),
        enriched_resolver=resolver,
    )
    names = [g[0] for g in genres]
    assert "indie rock" in names
