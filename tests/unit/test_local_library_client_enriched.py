"""Test that LocalLibraryClient.get_tracks_for_genre honors enriched signatures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path



def _make_metadata_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            artist_key TEXT,
            title TEXT,
            album TEXT,
            album_id TEXT,
            duration_ms INTEGER,
            file_path TEXT,
            musicbrainz_id TEXT,
            is_blacklisted INTEGER DEFAULT 0
        );
        CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO tracks(track_id, artist, artist_key, title, album, album_id, "
            "duration_ms, file_path, musicbrainz_id, is_blacklisted) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (r["track_id"], r["artist"], r.get("artist_key", r["artist"].lower()),
             r["title"], r["album"], r["album_id"], 200000,
             f"/music/{r['track_id']}.mp3", None),
        )
        for g in r.get("album_genres", []):
            conn.execute("INSERT INTO album_genres(album_id, genre) VALUES(?, ?)", (r["album_id"], g))
        for g in r.get("artist_genres", []):
            conn.execute("INSERT INTO artist_genres(artist, genre) VALUES(?, ?)", (r["artist"], g))
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


def test_get_tracks_for_genre_excludes_enriched_release_when_genre_not_in_signature(tmp_path):
    """Stratosphere has raw 'indie rock' but enriched signature lacks it → exclude."""
    from src.local_library_client import LocalLibraryClient
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "Duster", "title": "Topical Solution",
         "album": "Stratosphere", "album_id": "a1", "album_genres": ["indie rock"]},
    ])

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    client = LocalLibraryClient(db_path=str(metadata), enriched_resolver=resolver)
    tracks = client.get_tracks_for_genre("indie rock")
    assert tracks == []


def test_get_tracks_for_genre_includes_enriched_release_when_genre_in_signature(tmp_path):
    """Stratosphere's raw lacks 'slowcore' but enriched signature has it → include."""
    from src.local_library_client import LocalLibraryClient
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "Duster", "title": "Topical Solution",
         "album": "Stratosphere", "album_id": "a1", "album_genres": ["indie rock"]},
    ])

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    client = LocalLibraryClient(db_path=str(metadata), enriched_resolver=resolver)
    tracks = client.get_tracks_for_genre("slowcore")
    assert len(tracks) == 1
    assert tracks[0]["title"] == "Topical Solution"


def test_get_tracks_for_genre_unenriched_release_uses_raw_query(tmp_path):
    """An album with no enriched signature falls back to the existing UNION query."""
    from src.local_library_client import LocalLibraryClient
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "Random", "title": "Song",
         "album": "Album", "album_id": "a1", "album_genres": ["indie rock"]},
    ])

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [])  # no enrichment

    resolver = EnrichedGenreResolver(str(sidecar))
    client = LocalLibraryClient(db_path=str(metadata), enriched_resolver=resolver)
    tracks = client.get_tracks_for_genre("indie rock")
    assert len(tracks) == 1
    assert tracks[0]["title"] == "Song"


def test_get_tracks_for_genre_no_resolver_uses_raw_only(tmp_path):
    """Backwards compatible: no resolver → existing behavior."""
    from src.local_library_client import LocalLibraryClient

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "X", "title": "T",
         "album": "A", "album_id": "a1", "album_genres": ["rock"]},
    ])

    client = LocalLibraryClient(db_path=str(metadata))  # no resolver
    tracks = client.get_tracks_for_genre("rock")
    assert len(tracks) == 1
