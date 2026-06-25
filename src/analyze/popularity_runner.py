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
from typing import Dict, List, Optional

import numpy as np

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


def resolve_top_tracks_to_popularity(
    top_tracks: List[dict], local_tracks: List[dict]
) -> Dict[str, float]:
    """Map one artist's ranked Last.fm top tracks to local track_ids + popularity.

    mbid-first, else loose-normalized-title grouping with version-preference
    (studio/remaster beat live/demo/alt). Score = 1 - rank/N. On collision keep
    the higher score. Returns {track_id: popularity in [0,1]}.
    """
    if not top_tracks or not local_tracks:
        return {}
    from src.title_dedupe import (
        calculate_version_preference_score,
        normalize_title_for_dedupe,
    )

    by_mbid: Dict[str, str] = {}
    by_norm: Dict[str, List[dict]] = {}
    for lt in local_tracks:
        mbid = str(lt.get("musicbrainz_id") or "")
        if mbid:
            # first local track wins if two share an mbid (rare)
            by_mbid.setdefault(mbid, str(lt["track_id"]))
        norm = normalize_title_for_dedupe(str(lt.get("title") or ""), mode="loose")
        if norm:
            by_norm.setdefault(norm, []).append(lt)

    n = len(top_tracks)
    out: Dict[str, float] = {}
    for t in top_tracks:
        rank = int(t.get("rank", 0))
        score = 1.0 - rank / n
        tid: Optional[str] = None
        mbid = str(t.get("mbid") or "")
        if mbid and mbid in by_mbid:
            tid = by_mbid[mbid]
        else:
            norm = normalize_title_for_dedupe(str(t.get("name") or ""), mode="loose")
            cands = by_norm.get(norm, [])
            if cands:
                best = max(
                    cands,
                    key=lambda lt: (
                        calculate_version_preference_score(str(lt.get("title") or "")),
                        str(lt["track_id"]),
                    ),
                )
                tid = str(best["track_id"])
        if tid is not None and score > out.get(tid, -1.0):
            out[tid] = score
    return out


def _local_tracks_by_artist(metadata_db: str, min_artist_tracks: int) -> Dict[str, List[dict]]:
    by_artist: Dict[str, List[dict]] = {}
    with sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT track_id, title, musicbrainz_id, artist_key FROM tracks "
            "WHERE artist_key IS NOT NULL AND artist_key <> ''"
        ):
            by_artist.setdefault(str(r["artist_key"]), []).append({
                "track_id": str(r["track_id"]),
                "title": str(r["title"] or ""),
                "musicbrainz_id": str(r["musicbrainz_id"] or ""),
            })
    return {k: v for k, v in by_artist.items() if len(v) >= min_artist_tracks}


def build_popularity_sidecar(
    *, artifact_npz: str, metadata_db: str, enrichment_db: str,
    out_path: str, min_artist_tracks: int,
) -> dict:
    """Resolve cached Last.fm top tracks to local track_ids and write the sidecar."""
    tids = [str(t) for t in np.load(artifact_npz, allow_pickle=True)["track_ids"]]
    pos = {t: i for i, t in enumerate(tids)}
    popularity = np.full(len(tids), np.nan, dtype=np.float32)

    by_artist = _local_tracks_by_artist(metadata_db, min_artist_tracks)
    cached = cached_artist_keys(enrichment_db)
    matched = artists_resolved = 0
    for artist_key, local_tracks in by_artist.items():
        if artist_key not in cached:
            continue
        top = get_artist_top_tracks_cached(enrichment_db, artist_key)
        if not top:
            continue
        artists_resolved += 1
        for tid, score in resolve_top_tracks_to_popularity(top, local_tracks).items():
            j = pos.get(tid)
            if j is not None:
                popularity[j] = score
                matched += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, track_ids=np.array(tids, dtype=object), popularity=popularity,
    )
    logger.info("popularity sidecar: %d tracks, %d matched, %d artists resolved -> %s",
                len(tids), matched, artists_resolved, out_path)
    return {"tracks": len(tids), "matched": matched, "artists_resolved": artists_resolved}
