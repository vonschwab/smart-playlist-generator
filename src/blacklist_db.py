"""Database helpers for track blacklist schema."""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
    except sqlite3.OperationalError:
        return False
    for row in cursor.fetchall():
        try:
            name = row["name"]
        except Exception:
            name = row[1]
        if name == column_name:
            return True
    return False


def ensure_blacklist_schema(
    conn: sqlite3.Connection,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Ensure blacklist column + index exist on tracks table."""
    log = logger or logging.getLogger(__name__)
    if not _table_exists(conn, "tracks"):
        return
    if not _column_exists(conn, "tracks", "is_blacklisted"):
        try:
            conn.execute(
                "ALTER TABLE tracks ADD COLUMN is_blacklisted INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
            log.info("Added tracks.is_blacklisted column")
        except sqlite3.OperationalError:
            conn.rollback()
            if not _column_exists(conn, "tracks", "is_blacklisted"):
                raise
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tracks_is_blacklisted ON tracks(is_blacklisted)"
        )
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()
