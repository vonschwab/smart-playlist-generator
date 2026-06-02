import inspect
import sqlite3
import sys
from pathlib import Path

from scripts import consolidate_enrichment_dbs
from src.ai_genre_enrichment.storage import SidecarStore


def _create_canonical_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enriched_genres (
                enriched_genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
                release_key TEXT NOT NULL,
                normalized_artist TEXT NOT NULL,
                normalized_album TEXT NOT NULL,
                album_id TEXT,
                genre TEXT NOT NULL,
                basis TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                source_tag_id INTEGER,
                source_page_id INTEGER,
                source_ref TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'accepted',
                enrichment_policy_version TEXT,
                added_at TEXT NOT NULL
            );

            CREATE TABLE enriched_genre_signatures (
                release_key TEXT PRIMARY KEY,
                normalized_artist TEXT NOT NULL,
                normalized_album TEXT NOT NULL,
                album_id TEXT,
                signature_json TEXT NOT NULL,
                enrichment_policy_version TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )


def _create_legacy_source_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enriched_genres (
                enriched_genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
                release_key TEXT NOT NULL,
                normalized_artist TEXT NOT NULL,
                normalized_album TEXT NOT NULL,
                album_id TEXT,
                genre TEXT NOT NULL,
                basis TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                source_tag_id INTEGER,
                source_page_id INTEGER,
                source_ref TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'accepted',
                added_at TEXT NOT NULL
            );

            CREATE TABLE enriched_genre_signatures (
                release_key TEXT PRIMARY KEY,
                normalized_artist TEXT NOT NULL,
                normalized_album TEXT NOT NULL,
                album_id TEXT,
                signature_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            INSERT INTO enriched_genres (
                release_key, normalized_artist, normalized_album, album_id,
                genre, basis, confidence, source_ref, status, added_at
            ) VALUES (
                'artist::album', 'artist', 'album', 'a1',
                'shoegaze', 'authoritative_source', 0.9, 'source_tag:1', 'accepted', '2026-01-01'
            );

            INSERT INTO enriched_genre_signatures (
                release_key, normalized_artist, normalized_album, album_id,
                signature_json, updated_at
            ) VALUES (
                'artist::album', 'artist', 'album', 'a1',
                '{"genres":["shoegaze"],"sources":[]}', '2026-01-01'
            );
            """
        )


def _create_partial_source_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enriched_genre_signatures (
                release_key TEXT PRIMARY KEY,
                normalized_artist TEXT NOT NULL,
                normalized_album TEXT NOT NULL,
                album_id TEXT,
                signature_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            INSERT INTO enriched_genre_signatures (
                release_key, normalized_artist, normalized_album, album_id,
                signature_json, updated_at
            ) VALUES (
                'partial::album', 'partial', 'album', 'p1',
                '{"genres":["ambient"],"sources":[]}', '2026-01-01'
            );
            """
        )


def test_consolidation_copies_missing_legacy_policy_columns_as_null(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical_db = data_dir / "canonical.db"
    _create_canonical_db(canonical_db)
    _create_legacy_source_db(data_dir / "ai_genre_legacy.db")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["consolidate_enrichment_dbs.py", "--canon", str(canonical_db), "--apply"],
    )

    assert consolidate_enrichment_dbs.main() == 0

    with sqlite3.connect(canonical_db) as conn:
        assert conn.execute(
            "SELECT enrichment_policy_version FROM enriched_genres"
        ).fetchone() == (None,)
        assert conn.execute(
            "SELECT enrichment_policy_version FROM enriched_genre_signatures"
        ).fetchone() == (None,)


def test_consolidation_skips_partial_source_db_and_continues(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical_db = data_dir / "canonical.db"
    _create_canonical_db(canonical_db)
    _create_partial_source_db(data_dir / "ai_genre_partial.db")
    _create_legacy_source_db(data_dir / "ai_genre_valid.db")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["consolidate_enrichment_dbs.py", "--canon", str(canonical_db), "--apply"],
    )

    assert consolidate_enrichment_dbs.main() == 0
    assert (
        "skipping ai_genre_partial.db: missing required table enriched_genres"
        in capsys.readouterr().out
    )
    with sqlite3.connect(canonical_db) as conn:
        assert conn.execute(
            "SELECT release_key FROM enriched_genre_signatures"
        ).fetchall() == [("artist::album",)]


def test_rebuild_policy_version_is_not_publicly_overridable() -> None:
    parameters = inspect.signature(
        SidecarStore.rebuild_enriched_genres_for_release
    ).parameters

    assert list(parameters) == ["self", "release_key"]
