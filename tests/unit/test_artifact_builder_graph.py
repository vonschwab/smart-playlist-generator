"""Tests for the `graph` artifact genre source.

The graph source feeds the artifact's genre content from the published
authority (`release_effective_genres` + `genre_graph_canonical_genres` in
metadata.db, both maintained by the publish stage) instead of the raw
track/album/artist genre tables. Replacement semantics, like the enriched
source: covered albums use graph genres exclusively; uncovered tracks fall
back to legacy lookups.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _make_published_db(path: Path) -> None:
    """Minimal metadata.db slice with publish-maintained authority tables."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            album TEXT,
            album_id TEXT
        );
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
        CREATE TABLE release_effective_genres (
            album_id TEXT NOT NULL,
            genre_id TEXT NOT NULL,
            assignment_layer TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (album_id, genre_id, assignment_layer)
        );
        CREATE TABLE genre_graph_canonical_genres (
            genre_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT,
            specificity_score REAL,
            status TEXT,
            taxonomy_version TEXT
        );
    """)
    conn.executemany(
        "INSERT INTO tracks VALUES(?, ?, ?, ?)",
        [
            ("t1", "ATCQ", "Midnight Marauders", "a1"),
            ("t2", "Duster", "Stratosphere", "a2"),   # album NOT in effective table
            ("t3", "Unknown", "Single", None),          # no album_id at all
        ],
    )
    conn.executemany(
        "INSERT INTO track_genres VALUES(?, ?, 'file', 1.0)",
        [
            ("t1", "legacy noise"),   # must be ignored for covered album a1
            ("t2", "indie rock"),     # legacy fallback for uncovered album a2
        ],
    )
    conn.executemany(
        "INSERT INTO release_effective_genres VALUES(?, ?, ?, ?, ?)",
        [
            ("a1", "boom_bap", "observed_leaf", 0.8, "graph"),
            ("a1", "jazz", "inferred_family", 0.9, "graph"),
            ("a1", "hip_hop", "observed_leaf", 0.95, "graph"),
            ("a1", "hip_hop", "inferred_family", 0.95, "graph"),  # dup across layers
            ("a1", "indie_alternative", "inferred_family", 0.75, "graph"),
            ("a1", "mystery_id", "observed_leaf", 1.0, "user"),   # no canonical row
        ],
    )
    conn.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES(?, ?, ?, 0.5, 'active', 'v-test')",
        [
            ("boom_bap", "boom bap", "subgenre"),
            ("jazz", "jazz", "family"),
            ("hip_hop", "hip hop", "family"),
            ("indie_alternative", "indie/alternative", "family"),
        ],
    )
    conn.commit()
    conn.close()


class TestAuthorityBatchReaders:
    def test_resolved_genres_by_album_batches_all_albums(self, tmp_path):
        from src.genre.authority import resolved_genres_by_album

        db = tmp_path / "metadata.db"
        _make_published_db(db)
        conn = sqlite3.connect(str(db))
        try:
            by_album = resolved_genres_by_album(conn)
        finally:
            conn.close()
        assert set(by_album) == {"a1"}
        rows = by_album["a1"]
        assert {(r.genre_id, r.assignment_layer) for r in rows} == {
            ("boom_bap", "observed_leaf"),
            ("jazz", "inferred_family"),
            ("hip_hop", "observed_leaf"),
            ("hip_hop", "inferred_family"),
            ("indie_alternative", "inferred_family"),
            ("mystery_id", "observed_leaf"),
        }

    def test_canonical_genre_names_maps_id_to_display_name(self, tmp_path):
        from src.genre.authority import canonical_genre_names

        db = tmp_path / "metadata.db"
        _make_published_db(db)
        conn = sqlite3.connect(str(db))
        try:
            names = canonical_genre_names(conn)
        finally:
            conn.close()
        assert names["boom_bap"] == "boom bap"
        assert names["hip_hop"] == "hip hop"
        assert "mystery_id" not in names


def _load(db: Path, track_ids, **kwargs):
    from scripts.build_beat3tower_artifacts import load_genres_for_tracks

    genre_lists, vocab, stats = load_genres_for_tracks(
        str(db), track_ids, use_graph_genres=True, **kwargs
    )
    return [dict(gl) for gl in genre_lists], vocab, stats


class TestGraphGenreLoading:
    def test_covered_album_uses_graph_genres_as_replacement(self, tmp_path):
        """Covered album: published graph genres only — raw tiers ignored."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, stats = _load(db, ["t1"], normalize_genres=False)
        t1 = genres[0]
        assert "legacy noise" not in t1          # replacement, not supplement
        assert t1["boom bap"] == pytest.approx(0.8)   # id->name, conf x leaf 1.0
        assert stats["graph_tracks"] == 1
        assert stats["graph_albums"] == 1

    def test_inferred_family_weight_is_damped(self, tmp_path):
        """Families are hubs: weight = confidence x 0.5."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, _stats = _load(db, ["t1"], normalize_genres=False)
        assert genres[0]["jazz"] == pytest.approx(0.9 * 0.5)

    def test_duplicate_genre_across_layers_keeps_max_weight(self, tmp_path):
        """hip_hop appears as leaf (0.95) and family (0.95x0.5): leaf wins."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, _stats = _load(db, ["t1"], normalize_genres=False)
        assert genres[0]["hip hop"] == pytest.approx(0.95)

    def test_graph_names_bypass_legacy_normalization(self, tmp_path):
        """Canonical names are already final: 'indie/alternative' must not be
        split by the legacy normalizer even with normalization enabled."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, vocab, _stats = _load(db, ["t1"], normalize_genres=True)
        assert genres[0]["indie/alternative"] == pytest.approx(0.75 * 0.5)
        assert "indie/alternative" in vocab

    def test_uncovered_album_falls_back_to_legacy(self, tmp_path):
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, _stats = _load(db, ["t2", "t3"], normalize_genres=False)
        assert "indie rock" in genres[0]   # t2: raw track_genres retained
        assert genres[1] == {}             # t3: no album, no raw rows

    def test_unknown_genre_id_kept_with_loud_warning(self, tmp_path, caplog):
        import logging

        db = tmp_path / "metadata.db"
        _make_published_db(db)
        with caplog.at_level(logging.WARNING):
            genres, _vocab, _stats = _load(db, ["t1"], normalize_genres=False)
        assert genres[0]["mystery_id"] == pytest.approx(1.0)
        assert any("mystery_id" in rec.message for rec in caplog.records)

    def test_missing_authority_tables_raises(self, tmp_path):
        """Config says graph but publish never ran: loud error, not silent legacy."""
        db = tmp_path / "metadata.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT, album_id TEXT);
            CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        """)
        conn.execute("INSERT INTO tracks VALUES('t1', 'A', 'B', 'a1')")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="publish"):
            _load(db, ["t1"], normalize_genres=False)


class TestGenreArtifactSourceGraph:
    def test_resolve_graph_value(self):
        from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource

        assert GenreArtifactSource.resolve("graph") is GenreArtifactSource.GRAPH

    def test_make_resolver_returns_none_for_graph(self):
        """Graph mode reads metadata.db directly — no sidecar resolver."""
        from src.ai_genre_enrichment.artifact_modes import (
            GenreArtifactSource,
            make_resolver,
        )

        assert make_resolver(GenreArtifactSource.GRAPH, "ignored.db") is None
