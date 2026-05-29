"""Test SimilarityCalculator routes through EnrichedGenreResolver when configured."""

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
            is_blacklisted INTEGER DEFAULT 0
        );
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
    """)
    conn.execute("INSERT INTO tracks VALUES('t1', 'Duster', 'Stratosphere', 'a1', 0)")
    conn.execute("INSERT INTO track_genres VALUES('t1', 'indie rock', 'file', 1.0)")
    conn.execute("INSERT INTO album_genres VALUES('a1', 'rock')")
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


def test_combined_genres_uses_enriched_when_present(tmp_path):
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock", "shoegaze"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)
    genres = calc.get_filtered_combined_genres_for_track("t1")
    assert sorted(genres) == ["shoegaze", "slowcore", "space rock"]


def test_combined_genres_falls_back_when_unenriched(tmp_path):
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [])  # nothing enriched

    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)
    genres = calc.get_filtered_combined_genres_for_track("t1")
    assert "indie rock" in genres or "rock" in genres  # at least some raw genre present


def test_weighted_genres_uses_enriched_with_uniform_weight(tmp_path):
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "shoegaze"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)
    weighted = calc.get_weighted_genres_for_track("t1")
    weights = {g: w for g, w in weighted}
    assert weights == {"slowcore": 1.0, "shoegaze": 1.0}


def test_no_resolver_preserves_existing_behavior(tmp_path):
    from src.similarity_calculator import SimilarityCalculator

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    calc = SimilarityCalculator(db_path=str(metadata))  # no resolver
    genres = calc.get_filtered_combined_genres_for_track("t1")
    assert "indie rock" in genres  # raw behavior preserved


def test_bulk_preload_uses_enriched_signatures(tmp_path):
    """Regression: _preload_combined_genres_for_library must honor the resolver.

    The bulk preload is used by find_similar_tracks for candidate scoring. Before
    the fix, it bypassed the resolver and used raw genres for all candidates,
    while the single-track path used enriched. Mixing the two caused inconsistent
    similarity scoring between seed and candidates.
    """
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    conn = sqlite3.connect(str(metadata))
    conn.execute("INSERT INTO tracks VALUES('t2', 'Foo Bar', 'Baz', 'a2', 0)")
    conn.execute("INSERT INTO track_genres VALUES('t2', 'jazz', 'file', 1.0)")
    conn.commit()
    conn.close()

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock"]),
    ])
    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)

    bulk = calc._preload_combined_genres_for_library()

    # t1 is enriched: must use enriched signature, NOT raw 'indie rock' or 'rock'
    assert set(bulk["t1"]) == {"slowcore", "space rock"}, (
        f"Enriched release t1 should use signature only, got {bulk['t1']}"
    )
    # t2 is unenriched: must fall back to raw DB lookup
    assert "jazz" in bulk["t2"], f"Unenriched release t2 should keep raw, got {bulk['t2']}"


def test_bulk_preload_matches_single_track_path(tmp_path):
    """The bulk and single-track paths must return the same genres per track."""
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    conn = sqlite3.connect(str(metadata))
    conn.execute("INSERT INTO tracks VALUES('t2', 'Foo Bar', 'Baz', 'a2', 0)")
    conn.execute("INSERT INTO track_genres VALUES('t2', 'jazz', 'file', 1.0)")
    conn.commit()
    conn.close()

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock"]),
    ])
    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)

    bulk = calc._preload_combined_genres_for_library()
    for tid in ("t1", "t2"):
        single = calc._get_combined_genres(tid)
        assert set(bulk[tid]) == set(single), (
            f"Bulk vs single divergence for {tid}: bulk={bulk[tid]} single={single}"
        )
