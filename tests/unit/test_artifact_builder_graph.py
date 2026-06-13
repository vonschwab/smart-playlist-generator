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
            ("t4", "Patricia Wolf", "Smoking Mountain", "a3"),  # inferred-only album
            ("t5", "VV Torso", "LPVVII", "a4"),                 # legacy-layer album
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
            ("a1", "rnb_soul", "observed_leaf", 0.9, "graph"),    # leaf with '/'
            ("a1", "mystery_id", "observed_leaf", 1.0, "user"),   # no canonical row
            # a3: no observed/legacy rows at all — inferred fallback case
            ("a3", "ambient", "inferred_family", 0.8, "graph"),
            ("a3", "electronic", "inferred_parent", 0.6, "graph"),
            # a4: un-enriched album absorbed as legacy observations
            ("a4", "hardcore punk", "legacy", 1.0, "legacy"),
            ("a4", "post-punk", "legacy", 0.8, "legacy"),
        ],
    )
    conn.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES(?, ?, ?, 0.5, 'active', 'v-test')",
        [
            ("boom_bap", "boom bap", "subgenre"),
            ("jazz", "jazz", "family"),
            ("hip_hop", "hip hop", "family"),
            ("indie_alternative", "indie/alternative", "family"),
            ("rnb_soul", "r&b/soul", "family"),
            ("ambient", "ambient", "family"),
            ("electronic", "electronic", "family"),
            ("hardcore punk", "hardcore punk", "subgenre"),
            ("post-punk", "post-punk", "subgenre"),
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
        assert set(by_album) == {"a1", "a3", "a4"}
        rows = by_album["a1"]
        assert {(r.genre_id, r.assignment_layer) for r in rows} == {
            ("boom_bap", "observed_leaf"),
            ("jazz", "inferred_family"),
            ("hip_hop", "observed_leaf"),
            ("hip_hop", "inferred_family"),
            ("indie_alternative", "inferred_family"),
            ("rnb_soul", "observed_leaf"),
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

    def test_display_genre_names_for_covered_track(self, tmp_path):
        """Published genres as display names, deduped across layers."""
        from src.genre.authority import display_genre_names_for_track

        db = tmp_path / "metadata.db"
        _make_published_db(db)
        conn = sqlite3.connect(str(db))
        try:
            names = display_genre_names_for_track(conn, "t1")
        finally:
            conn.close()
        assert "boom bap" in names            # id -> display name
        assert "indie/alternative" in names   # family included
        assert "mystery_id" in names          # unmapped id passes through
        assert names.count("hip hop") == 1    # observed+inferred deduped

    def test_display_genre_names_uncovered_or_missing(self, tmp_path):
        """Uncovered track -> []; missing authority tables -> [] (display
        paths fall back, they don't crash)."""
        from src.genre.authority import display_genre_names_for_track

        db = tmp_path / "metadata.db"
        _make_published_db(db)
        conn = sqlite3.connect(str(db))
        try:
            assert display_genre_names_for_track(conn, "t2") == []  # album uncovered
            assert display_genre_names_for_track(conn, "t3") == []  # no album_id
            assert display_genre_names_for_track(conn, "nope") == []
        finally:
            conn.close()

        bare = tmp_path / "bare.db"
        conn = sqlite3.connect(str(bare))
        conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT)")
        try:
            assert display_genre_names_for_track(conn, "t1") == []
        finally:
            conn.close()


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

    def test_inferred_layers_excluded_from_similarity_vectors(self, tmp_path):
        """Inferred taxonomy ancestors saturate genre cosine (hub mass on every
        release) — they stay in the authority for display but out of X_genre_*."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, vocab, _stats = _load(db, ["t1"], normalize_genres=False)
        assert "jazz" not in genres[0]                 # inferred_family dropped
        assert "indie/alternative" not in genres[0]    # inferred_family dropped
        assert "indie/alternative" not in vocab

    def test_duplicate_genre_across_layers_keeps_observed_weight(self, tmp_path):
        """hip_hop appears as leaf (0.95) and family (excluded): leaf weight wins."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, _stats = _load(db, ["t1"], normalize_genres=False)
        assert genres[0]["hip hop"] == pytest.approx(0.95)

    def test_graph_names_bypass_legacy_normalization(self, tmp_path):
        """Canonical names are already final: 'r&b/soul' must not be split by
        the legacy normalizer even with normalization enabled."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, vocab, _stats = _load(db, ["t1"], normalize_genres=True)
        assert genres[0]["r&b/soul"] == pytest.approx(0.9)
        assert "r&b/soul" in vocab

    def test_legacy_layer_gets_full_weight(self, tmp_path):
        """Legacy rows are absorbed raw observations of un-enriched albums —
        full weight, not the old silent 0.5 default for unlisted layers."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, _stats = _load(db, ["t5"], normalize_genres=False)
        assert genres[0]["hardcore punk"] == pytest.approx(1.0)
        assert genres[0]["post-punk"] == pytest.approx(0.8)

    def test_inferred_only_album_keeps_damped_fallback(self, tmp_path):
        """An album with no observed/legacy rows keeps its inferred rows at
        0.5x confidence — a sparse vector beats silently zeroing the release
        out of every genre gate."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        genres, _vocab, stats = _load(db, ["t4"], normalize_genres=False)
        assert genres[0]["ambient"] == pytest.approx(0.8 * 0.5)
        assert genres[0]["electronic"] == pytest.approx(0.6 * 0.5)
        assert stats["graph_inferred_only_albums"] == 1

    def test_unknown_assignment_layer_raises(self, tmp_path):
        """A layer the builder doesn't recognize means the publish schema moved:
        loud error, not a silent default weight."""
        db = tmp_path / "metadata.db"
        _make_published_db(db)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO release_effective_genres VALUES('a1', 'jazz', 'experimental_layer', 0.5, 'graph')"
        )
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="experimental_layer"):
            _load(db, ["t1"], normalize_genres=False)

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
