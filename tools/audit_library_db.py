#!/usr/bin/env python3
"""
Read-only audit of metadata.db invariants.

Checks:
- Table presence and row counts
- Null/duplicate file paths
- Orphaned genre rows
- Sonic feature coverage
- Marker counts (MusicBrainz/genre empties)

Safe to run without tokens; no writes are performed. Uses SQLite read-only mode when available.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
    except sqlite3.OperationalError:
        return False
    return any(row[1] == column for row in cur.fetchall())


def scalar(conn: sqlite3.Connection, query: str, params: Tuple = ()) -> Optional[int]:
    try:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return None


def resolve_db_path(args: argparse.Namespace) -> Path:
    if args.db:
        return Path(args.db)
    config_path = Path(args.config)
    if config_path.exists():
        try:
            from src.config_loader import Config  # type: ignore
        except Exception:
            pass
        else:
            try:
                cfg = Config(config_path)
                return Path(cfg.library_database_path)
            except Exception:
                pass
    return Path("data/metadata.db")


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(db_path)


def print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def audit_tracks(conn: sqlite3.Connection) -> Dict[str, Optional[int]]:
    stats: Dict[str, Optional[int]] = {}
    if not table_exists(conn, "tracks"):
        print("tracks table missing")
        return stats
    stats["tracks_total"] = scalar(conn, "SELECT COUNT(*) FROM tracks")
    stats["tracks_with_path"] = scalar(
        conn, "SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL AND file_path != ''"
    )
    stats["tracks_missing_path"] = scalar(
        conn, "SELECT COUNT(*) FROM tracks WHERE file_path IS NULL OR file_path = ''"
    )
    stats["duplicate_paths"] = scalar(
        conn,
        """
        SELECT COUNT(*) FROM (
            SELECT file_path FROM tracks
            WHERE file_path IS NOT NULL AND file_path != ''
            GROUP BY file_path
            HAVING COUNT(*) > 1
        )
        """,
    )
    stats["sonic_present"] = scalar(conn, "SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL")
    stats["sonic_failed"] = scalar(conn, "SELECT COUNT(*) FROM tracks WHERE sonic_failed_at IS NOT NULL")
    stats["mbid_markers"] = scalar(
        conn,
        "SELECT COUNT(*) FROM tracks WHERE musicbrainz_id IN ('__NO_MATCH__','__ERROR__','__REJECT__')",
    )
    stats["mbid_real"] = scalar(
        conn,
        "SELECT COUNT(*) FROM tracks WHERE musicbrainz_id IS NOT NULL AND musicbrainz_id NOT IN "
        "('__NO_MATCH__','__ERROR__','__REJECT__')",
    )
    if column_exists(conn, "tracks", "album_id"):
        stats["tracks_with_album_id"] = scalar(
            conn,
            "SELECT COUNT(*) FROM tracks WHERE album_id IS NOT NULL AND album_id != ''",
        )
    return stats


def audit_genres(conn: sqlite3.Connection) -> Dict[str, Optional[int]]:
    stats: Dict[str, Optional[int]] = {}
    if table_exists(conn, "track_genres"):
        stats["track_genres_total"] = scalar(conn, "SELECT COUNT(*) FROM track_genres")
        stats["track_genres_file"] = scalar(
            conn, "SELECT COUNT(*) FROM track_genres WHERE source = 'file'"
        )
        stats["track_genres_orphans"] = scalar(
            conn,
            """
            SELECT COUNT(*) FROM track_genres tg
            LEFT JOIN tracks t ON tg.track_id = t.track_id
            WHERE t.track_id IS NULL
            """,
        )
    if table_exists(conn, "artist_genres"):
        stats["artist_genres_total"] = scalar(conn, "SELECT COUNT(*) FROM artist_genres")
        stats["artist_empty"] = scalar(
            conn, "SELECT COUNT(*) FROM artist_genres WHERE genre = '__EMPTY__'"
        )
    if table_exists(conn, "album_genres"):
        stats["album_genres_total"] = scalar(conn, "SELECT COUNT(*) FROM album_genres")
        stats["album_empty"] = scalar(
            conn, "SELECT COUNT(*) FROM album_genres WHERE genre = '__EMPTY__'"
        )
        stats["album_discogs"] = scalar(
            conn,
            "SELECT COUNT(*) FROM album_genres WHERE source IN ('discogs_release','discogs_master')",
        )
        stats["album_musicbrainz"] = scalar(
            conn, "SELECT COUNT(*) FROM album_genres WHERE source = 'musicbrainz_release'"
        )
        stats["album_genres_orphans"] = scalar(
            conn,
            """
            SELECT COUNT(*) FROM album_genres ag
            LEFT JOIN tracks t ON ag.album_id = t.album_id
            WHERE t.album_id IS NULL
            """,
        )
    if table_exists(conn, "track_effective_genres"):
        stats["effective_orphans"] = scalar(
            conn,
            """
            SELECT COUNT(*) FROM track_effective_genres teg
            LEFT JOIN tracks t ON teg.track_id = t.track_id
            WHERE t.track_id IS NULL
            """,
        )
    return stats


def audit_track_id_orphans(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Report orphaned track_id references for every table that has a track_id column.
    """
    results: Dict[str, int] = {}
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    for table in tables:
        if table == "tracks":
            continue
        if not column_exists(conn, table, "track_id"):
            continue
        count = scalar(
            conn,
            f"""
            SELECT COUNT(*) FROM {table} x
            WHERE x.track_id NOT IN (SELECT track_id FROM tracks)
            """,
        )
        if count is not None:
            results[table] = count
    return results


def audit_database(db_path: Path) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Convenience wrapper to audit a database programmatically.
    """
    conn = connect_readonly(db_path)
    conn.row_factory = sqlite3.Row
    report = {
        "tracks": audit_tracks(conn),
        "genres": audit_genres(conn),
        "albums": audit_albums(conn),
        "orphans": audit_track_id_orphans(conn),
    }
    conn.close()
    return report


def audit_albums(conn: sqlite3.Connection) -> Dict[str, Optional[int]]:
    stats: Dict[str, Optional[int]] = {}
    if not table_exists(conn, "albums"):
        return stats
    stats["albums_total"] = scalar(conn, "SELECT COUNT(*) FROM albums")
    stats["albums_no_tracks"] = scalar(
        conn,
        """
        SELECT COUNT(*) FROM albums a
        LEFT JOIN tracks t ON a.album_id = t.album_id
        WHERE t.album_id IS NULL
        """,
    )
    return stats


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Read-only audit of metadata.db invariants.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config path to resolve DB path (default: config.yaml).",
    )
    parser.add_argument(
        "--db",
        help="Explicit path to metadata.db (overrides config).",
    )
    args = parser.parse_args(argv)

    db_path = resolve_db_path(args)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    try:
        conn = connect_readonly(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to open database {db_path}: {exc}")
        return 1

    print(f"Auditing database: {db_path}")

    print_header("Tracks")
    for key, val in audit_tracks(conn).items():
        if val is not None:
            print(f"{key:22}: {val:,}")

    print_header("Genres")
    for key, val in audit_genres(conn).items():
        if val is not None:
            print(f"{key:22}: {val:,}")

    print_header("Albums")
    for key, val in audit_albums(conn).items():
        if val is not None:
            print(f"{key:22}: {val:,}")

    print_header("Track ID Orphans")
    orphan_stats = audit_track_id_orphans(conn)
    if not orphan_stats:
        print("None found")
    else:
        for table, count in orphan_stats.items():
            print(f"{table:22}: {count:,}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
