"""Build side of the Last.fm popularity sidecar.

Fetches each artist's top tracks (cached in ai_genre_enrichment.db so re-runs
skip), resolves each Last.fm track to the *canonical* local track per song
(mbid-first, then loose-title + version-preference), and writes
data/artifacts/beat3tower_32k/popularity/popularity_sidecar.npz aligned to the
artifact's track_ids. Mirrors the energy sidecar. Reads metadata.db read-only.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

ENRICHMENT_DB_DEFAULT = "data/ai_genre_enrichment.db"


def init_top_tracks_cache(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS artist_top_tracks_cache ("
            "artist_key TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, "
            "track_count INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL)"
        )


def cached_artist_keys(db_path: str) -> set:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT artist_key FROM artist_top_tracks_cache")
        return {r[0] for r in rows}


def upsert_artist_top_tracks(
    db_path: str, artist_key: str, fetched_at: str, top_tracks: List[dict]
) -> None:
    payload = json.dumps(top_tracks)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO artist_top_tracks_cache (artist_key, fetched_at, track_count, payload_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(artist_key) DO UPDATE SET "
            "fetched_at=excluded.fetched_at, track_count=excluded.track_count, "
            "payload_json=excluded.payload_json",
            (artist_key, fetched_at, len(top_tracks), payload),
        )


def get_artist_top_tracks_cached(db_path: str, artist_key: str) -> List[dict]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM artist_top_tracks_cache WHERE artist_key = ?",
            (artist_key,),
        ).fetchone()
    return json.loads(row[0]) if row else []
