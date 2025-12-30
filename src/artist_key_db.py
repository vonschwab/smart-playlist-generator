"""
Artist key database helpers for normalized artist matching.
"""
from __future__ import annotations

import logging
from typing import Optional

from .string_utils import normalize_artist_key


def ensure_artist_key_schema(
    conn,
    *,
    logger: Optional[logging.Logger] = None,
    batch_size: int = 1000,
) -> None:
    """
    Ensure tracks.artist_key exists, backfill values, and add an index.
    """
    log = logger or logging.getLogger(__name__)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(tracks)")
    cols = set()
    for row in cur.fetchall():
        try:
            cols.add(row["name"])
        except Exception:
            cols.add(row[1])

    if "artist_key" not in cols:
        cur.execute("ALTER TABLE tracks ADD COLUMN artist_key TEXT")
        conn.commit()

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracks_artist_key ON tracks(artist_key)"
    )
    conn.commit()

    cur.execute(
        "SELECT COUNT(1) FROM tracks WHERE artist_key IS NULL OR artist_key = ''"
    )
    row = cur.fetchone()
    total = row[0] if row else 0
    if total <= 0:
        return

    log.info("Backfilling artist_key for %d tracks", total)
    processed = 0
    while True:
        cur.execute(
            "SELECT rowid, artist FROM tracks "
            "WHERE artist_key IS NULL OR artist_key = '' "
            "LIMIT ?",
            (batch_size,),
        )
        rows = cur.fetchall()
        if not rows:
            break
        updates = []
        for row in rows:
            try:
                rowid = row["rowid"]
                artist = row["artist"]
            except Exception:
                rowid = row[0]
                artist = row[1]
            updates.append((normalize_artist_key(artist or ""), rowid))
        cur.executemany("UPDATE tracks SET artist_key = ? WHERE rowid = ?", updates)
        conn.commit()
        processed += len(updates)
        log.debug("Backfilled artist_key: %d/%d", processed, total)
    log.info("Artist key backfill complete: %d tracks", processed)
