"""Test that the GUI worker uses EnrichedGenreResolver to populate playlist genres."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path



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


def _seed_published_metadata(db_path: Path) -> None:
    """metadata.db slice with publish-maintained authority tables (rich)."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (
            album_id TEXT, genre_id TEXT, assignment_layer TEXT,
            confidence REAL, source TEXT,
            PRIMARY KEY (album_id, genre_id, assignment_layer)
        );
        CREATE TABLE genre_graph_canonical_genres (
            genre_id TEXT PRIMARY KEY, name TEXT, kind TEXT,
            specificity_score REAL, status TEXT, taxonomy_version TEXT
        );
        INSERT INTO tracks VALUES ('t1', 'Blood Orange', 'Angels Pulse', 'a1');
        INSERT INTO release_effective_genres VALUES
            ('a1', 'neo_soul', 'observed_leaf', 1.0, 'graph'),
            ('a1', 'art_pop', 'observed_leaf', 0.9, 'graph'),
            ('a1', 'alternative', 'inferred_family', 0.95, 'graph');
        INSERT INTO genre_graph_canonical_genres VALUES
            ('neo_soul', 'neo-soul', 'subgenre', 0.8, 'active', 'v-test'),
            ('art_pop', 'art pop', 'subgenre', 0.8, 'active', 'v-test'),
            ('alternative', 'alternative', 'family', 0.1, 'active', 'v-test');
        """
    )
    conn.commit()
    conn.close()


def test_resolve_track_genres_prefers_published_authority(tmp_path):
    """The published authority (release_effective_genres) outranks the sidecar
    signature — signatures are the old bandcamp-era layer and can be stale/sparse
    while the authority is rich (the Blood Orange bug)."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar_path)  # duster::stratosphere with a 2-tag signature
    metadata_path = tmp_path / "metadata.db"
    _seed_published_metadata(metadata_path)

    track = {"artist": "Blood Orange", "album": "Angels Pulse", "rating_key": "t1"}
    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=lambda: ["raw tag"],
        db_path=str(metadata_path),
    )
    assert sorted(result) == ["alternative", "art pop", "neo-soul"]


def test_resolve_track_genres_authority_miss_falls_to_signature(tmp_path):
    """Unpublished album: the signature chain still works as before."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar_path)
    metadata_path = tmp_path / "metadata.db"
    _seed_published_metadata(metadata_path)  # has no duster rows

    track = {"artist": "Duster", "album": "Stratosphere", "rating_key": "t9"}
    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=lambda: ["raw tag"],
        db_path=str(metadata_path),
    )
    assert result == ["slowcore", "space rock"]


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


def _seed_sidecar_for_ordering(sidecar_path: Path) -> None:
    """Release whose enriched genres are stored broad-first with one noise tag."""
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "acetone::york blvd",
                "acetone",
                "york blvd",
                None,
                json.dumps({"genres": ["rock", "slowcore", "seen live"], "sources": []}),
                "2026-06-11T00:00:00",
            ),
        )
        conn.commit()


def test_resolve_display_genres_orders_sub_to_broad(tmp_path):
    """Enriched genres come back canonicalized and most-specific first; noise dropped.

    Uses the real taxonomy: slowcore (0.88) > rock (0.05); 'seen live' rejected.
    """
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar_for_ordering(sidecar_path)

    track = {"artist": "Acetone", "album": "York Blvd", "rating_key": "t1"}
    result = worker._resolve_display_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=lambda: [],
    )
    assert result == ["slowcore", "rock"]


def test_resolve_display_genres_raw_fallback_when_uncovered(tmp_path):
    """All-uncovered genres fall back to raw unordered — chips never go blank."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "nonexistent.db"
    track = {"artist": "Unknown", "album": "Album", "rating_key": "t1"}
    raw = ["totally-not-a-genre", "also-not-one"]
    result = worker._resolve_display_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=lambda: list(raw),
    )
    assert result == raw
