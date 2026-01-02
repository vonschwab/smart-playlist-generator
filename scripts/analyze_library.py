#!/usr/bin/env python3
"""
Unified Analyze Library pipeline:
- Genres (normalized)
- Discogs genres (optional)
- Sonic analysis
- Genre similarity matrix
- DS artifact build
- Verification
"""
import argparse
import json
import logging
import os
import sys
import time
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import sqlite3

from src.config_loader import Config
from src.analyze.genre_similarity import build_genre_similarity_matrix
from src.analyze.artifact_builder import build_ds_artifacts
from src.features.artifacts import load_artifact_bundle
from scripts.update_genres_v3_normalized import NormalizedGenreUpdater
from scripts.update_sonic import SonicFeaturePipeline
from scripts.scan_library import LibraryScanner
from scripts.update_discogs_genres import DiscogsClient, iter_albums, upsert_album_genres, best_match, fetch_genres, normalize_tag, discogs_status, load_config_token
from src.logging_utils import ProgressLogger

DEFAULT_OUT_DIR = ROOT_DIR / "data" / "artifacts" / "beat3tower_32k"
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "sonic", "genre-sim", "artifacts", "verify"]

logger = logging.getLogger("analyze_library")


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint and state helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _safe_count(conn: sqlite3.Connection, query: str, params: tuple = ()) -> int:
    try:
        row = conn.execute(query, params).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except Exception:
            return int(row["c"])
    except Exception:
        return 0


def ensure_analyze_state_schema(conn: sqlite3.Connection) -> None:
    """Create a small table to persist last-success fingerprints per stage."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyze_state (
            stage_name TEXT PRIMARY KEY,
            last_success_fingerprint TEXT,
            last_success_at TEXT
        )
        """
    )
    conn.commit()


def get_last_fingerprint(conn: sqlite3.Connection, stage: str) -> Optional[str]:
    try:
        row = conn.execute(
            "SELECT last_success_fingerprint FROM analyze_state WHERE stage_name=?",
            (stage,),
        ).fetchone()
        if not row:
            return None
        try:
            return row["last_success_fingerprint"]
        except Exception:
            return row[0]
    except Exception:
        return None


def set_last_fingerprint(conn: sqlite3.Connection, stage: str, fingerprint: str) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO analyze_state(stage_name, last_success_fingerprint, last_success_at)
        VALUES (?, ?, ?)
        ON CONFLICT(stage_name) DO UPDATE SET
            last_success_fingerprint=excluded.last_success_fingerprint,
            last_success_at=excluded.last_success_at
        """,
        (stage, fingerprint, now),
    )
    conn.commit()


def compute_config_hash(cfg: Config, args: argparse.Namespace) -> str:
    config_data = getattr(cfg, "config", {}) or {}
    relevant = {
        "library": config_data.get("library", {}),
        "artifacts": config_data.get("artifacts", {}),
        "cli": {
            "stages": args.stages,
            "max_tracks": args.max_tracks,
            "limit": args.limit,
            "out_dir": args.out_dir,
        },
    }
    return _hash_obj(relevant)


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return []
    cols = []
    for row in rows:
        try:
            cols.append(row["name"])
        except Exception:
            cols.append(row[1])
    cols.sort()
    return cols


def compute_stage_fingerprint(ctx: Dict, stage: str) -> str:
    """
    Build a small stable fingerprint for the inputs a stage depends on.
    Does not read files outside the DB/config.
    """
    conn = ctx["conn"]
    cfg_hash = ctx.get("config_hash", "")
    if stage == "scan":
        total_tracks = _safe_count(conn, "SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL")
        max_mtime = _safe_count(conn, "SELECT MAX(COALESCE(file_mtime_ns, file_modified)) FROM tracks")
        key = {
            "stage": stage,
            "library_root": ctx.get("library_root"),
            "total_tracks": total_tracks,
            "max_mtime": max_mtime,
            "cols": _table_columns(conn, "tracks"),
        }
        return _hash_obj(key)

    if stage == "mbid":
        unknown = _safe_count(
            conn,
            "SELECT COUNT(*) FROM tracks WHERE COALESCE(mbid_status,'unknown') IN ('unknown','failed')"
        )
        markers = _safe_count(
            conn,
            "SELECT COUNT(*) FROM tracks WHERE musicbrainz_id IN ('__NO_MATCH__','__ERROR__','__REJECT__')"
        )
        key = {"stage": stage, "pending": unknown, "markers": markers, "cols": _table_columns(conn, "tracks")}
        return _hash_obj(key)

    if stage == "genres":
        missing_artists = _safe_count(
            conn,
            """
            SELECT COUNT(DISTINCT t.artist)
            FROM tracks t
            LEFT JOIN artist_genres g
              ON t.artist = g.artist
             AND g.source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
            WHERE t.artist IS NOT NULL AND TRIM(t.artist) != ''
              AND t.file_path IS NOT NULL AND t.file_path != ''
              AND g.artist IS NULL
            """,
        )
        missing_albums = _safe_count(
            conn,
            """
            SELECT COUNT(*)
            FROM albums a
            LEFT JOIN (
                SELECT DISTINCT album_id FROM album_genres WHERE source = 'musicbrainz_release'
            ) g ON a.album_id = g.album_id
            WHERE g.album_id IS NULL
              AND a.album_id IN (
                  SELECT DISTINCT album_id
                  FROM tracks
                  WHERE file_path IS NOT NULL AND file_path != ''
                    AND album_id IS NOT NULL AND album != ''
              )
            """,
        )
        key = {
            "stage": stage,
            "missing_artists": missing_artists,
            "missing_albums": missing_albums,
            "artist_cols": _table_columns(conn, "artist_genres"),
            "album_cols": _table_columns(conn, "album_genres"),
        }
        return _hash_obj(key)

    if stage == "discogs":
        pending = _safe_count(
            conn,
            """
            SELECT COUNT(DISTINCT t.album_id)
            FROM tracks t
            LEFT JOIN albums a ON a.album_id = t.album_id
            WHERE t.album IS NOT NULL AND t.album != ''
              AND t.file_path IS NOT NULL AND t.file_path != ''
              AND COALESCE(a.discogs_status,'unknown') IN ('unknown','failed','no_match')
            """,
        )
        key = {
            "stage": stage,
            "pending": pending,
            "album_cols": _table_columns(conn, "albums"),
            "album_genre_cols": _table_columns(conn, "album_genres"),
        }
        return _hash_obj(key)

    if stage == "sonic":
        pending = _safe_count(
            conn,
            "SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL AND sonic_features IS NULL AND COALESCE(sonic_failed_at,0)=0"
        )
        has_features = _safe_count(
            conn,
            "SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL"
        )
        key = {
            "stage": stage,
            "pending": pending,
            "has_features": has_features,
            "cols": _table_columns(conn, "tracks"),
        }
        return _hash_obj(key)

    if stage == "genre-sim":
        genre_rows = _safe_count(conn, "SELECT COUNT(*) FROM track_genres WHERE genre != '__EMPTY__'")
        artist_rows = _safe_count(conn, "SELECT COUNT(*) FROM artist_genres WHERE genre != '__EMPTY__'")
        album_rows = _safe_count(conn, "SELECT COUNT(*) FROM album_genres WHERE genre != '__EMPTY__'")
        out_path = Path(ctx["out_dir"]) / "genre_similarity_matrix.npz"
        out_mtime = int(out_path.stat().st_mtime) if out_path.exists() else 0
        key = {
            "stage": stage,
            "track_genres": genre_rows,
            "artist_genres": artist_rows,
            "album_genres": album_rows,
            "config": cfg_hash,
            "artifact_exists": out_path.exists(),
            "artifact_mtime": out_mtime,
        }
        return _hash_obj(key)

    if stage == "artifacts":
        tracks_with_features = _safe_count(conn, "SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL")
        genre_rows = _safe_count(conn, "SELECT COUNT(*) FROM track_genres WHERE genre != '__EMPTY__'")
        out_path = Path(ctx["out_dir"]) / "data_matrices_step1.npz"
        manifest_path = Path(ctx["out_dir"]) / "artifact_manifest.json"
        out_mtime = int(out_path.stat().st_mtime) if out_path.exists() else 0
        manifest_mtime = int(manifest_path.stat().st_mtime) if manifest_path.exists() else 0
        key = {
            "stage": stage,
            "tracks_with_features": tracks_with_features,
            "track_genres": genre_rows,
            "config": cfg_hash,
            "artifact_exists": out_path.exists(),
            "artifact_mtime": out_mtime,
            "manifest_mtime": manifest_mtime,
        }
        return _hash_obj(key)

    if stage == "verify":
        artifact_mtime = 0
        try:
            artifact_path = ctx["out_dir"] / "data_matrices_step1.npz"
            if artifact_path.exists():
                artifact_mtime = int(artifact_path.stat().st_mtime)
        except Exception:
            artifact_mtime = 0
        key = {"stage": stage, "artifact_mtime": artifact_mtime}
    return _hash_obj(key)

    return _hash_obj({"stage": stage})


def estimate_stage_units(ctx: Dict, stage: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Estimate how many work items a stage will process for progress/ETA reporting.
    Returns (count, label) where count may be None if unknown.
    """
    conn = ctx["conn"]
    try:
        if stage == "scan":
            count = _safe_count(conn, "SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL AND file_path != ''")
            return count, "tracks (existing footprint; filesystem discovery may differ)"
        if stage == "mbid":
            pending = _safe_count(
                conn,
                "SELECT COUNT(*) FROM tracks WHERE COALESCE(mbid_status,'unknown') IN ('unknown','failed')",
            )
            return pending, "tracks needing MBID"
        if stage == "genres":
            missing_artists = _safe_count(
                conn,
                """
                SELECT COUNT(DISTINCT t.artist)
                FROM tracks t
                LEFT JOIN artist_genres g
                  ON t.artist = g.artist
                 AND g.source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
                WHERE t.artist IS NOT NULL AND TRIM(t.artist) != ''
                  AND t.file_path IS NOT NULL AND t.file_path != ''
                  AND g.artist IS NULL
                """,
            )
            missing_albums = _safe_count(
                conn,
                """
                SELECT COUNT(*)
                FROM albums a
                LEFT JOIN (
                    SELECT DISTINCT album_id FROM album_genres WHERE source = 'musicbrainz_release'
                ) g ON a.album_id = g.album_id
                WHERE g.album_id IS NULL
                  AND a.album_id IN (
                      SELECT DISTINCT album_id
                      FROM tracks
                      WHERE file_path IS NOT NULL AND file_path != ''
                        AND album_id IS NOT NULL AND album != ''
                  )
                """,
            )
            return missing_artists + missing_albums, "artist/album genre lookups"
        if stage == "discogs":
            pending = _safe_count(
                conn,
                """
                SELECT COUNT(DISTINCT t.album_id)
                FROM tracks t
                LEFT JOIN albums a ON a.album_id = t.album_id
                WHERE t.album IS NOT NULL AND t.album != ''
                  AND t.file_path IS NOT NULL AND t.file_path != ''
                  AND COALESCE(a.discogs_status,'unknown') IN ('unknown','failed','no_match')
                """,
            )
            return pending, "albums needing Discogs genres"
        if stage == "sonic":
            pending = _safe_count(
                conn,
                "SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL AND sonic_features IS NULL AND COALESCE(sonic_failed_at,0)=0",
            )
            return pending, "tracks needing sonic features"
        if stage == "genre-sim":
            rows = _safe_count(conn, "SELECT COUNT(*) FROM track_genres WHERE genre != '__EMPTY__'")
            return rows, "genre rows feeding similarity"
        if stage == "artifacts":
            tracks_with_features = _safe_count(
                conn, "SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL"
            )
            return tracks_with_features, "tracks with sonic features for artifacts"
    except Exception:
        return None, None
    return None, None


def summarize_items(result: Any) -> Optional[str]:
    """Render a human-friendly item count from a stage result."""
    if not isinstance(result, dict):
        return None
    preferred_keys = [
        "total",
        "scan_total",
        "total_tracks",
        "total_albums",
        "pending",
        "updated",
        "hits",
        "tracks",
    ]
    for key in preferred_keys:
        if key in result and result.get(key) is not None:
            try:
                return f"{int(result.get(key)):,}"
            except Exception:
                return str(result.get(key))
    return None


def _extract_processed_and_errors(stage: str, result: Any) -> Tuple[Optional[int], int, Optional[str]]:
    processed = None
    errors = 0
    top_err = None
    if isinstance(result, dict):
        for key in ("total", "scan_total", "pending", "hits", "updated"):
            if result.get(key) is not None:
                try:
                    processed = int(result.get(key))
                    break
                except Exception:
                    processed = None
        if stage == "verify":
            issues = result.get("issues") or []
            errors = len(issues)
            if issues:
                top_err = ",".join(issues[:3])
        elif "errors" in result and result.get("errors") is not None:
            try:
                errors = int(result.get("errors"))
            except Exception:
                errors = 0
        elif "misses" in result and result.get("misses") is not None:
            try:
                errors = len(result.get("misses") or [])
                top_err = "misses"
            except Exception:
                errors = 0
    return processed, errors, top_err


def _get_git_commit() -> Optional[str]:
    head = ROOT_DIR / ".git" / "HEAD"
    try:
        if not head.exists():
            return None
        content = head.read_text().strip()
        if content.startswith("ref:"):
            ref_path = content.split(" ", 1)[1].strip()
            ref_file = ROOT_DIR / ".git" / ref_path
            if ref_file.exists():
                return ref_file.read_text().strip()
        return content
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MusicBrainz MBID enrichment (optional)
# ─────────────────────────────────────────────────────────────────────────────

def stage_mbid(ctx: Dict) -> Dict:
    """
    Optional stage: fetch MusicBrainz recording MBIDs into metadata.db.
    Uses scripts/fetch_mbids_musicbrainz.py logic in-process to avoid file writes.
    Respects limit/force flags; default limit=200 to avoid hammering MB.
    """
    args = ctx["args"]
    db_path = ctx["db_path"]
    limit = args.limit if args.limit and args.limit > 0 else 200
    force = args.force
    force_no_match = getattr(args, "force_no_match", False)
    force_error = getattr(args, "force_error", False)
    force_reject = getattr(args, "force_reject", False)
    force_all = getattr(args, "force_all", False)

    # Import fetcher utilities
    from scripts.fetch_mbids_musicbrainz import (
        get_session,
        load_candidates,
        search_recording_relaxed,
        best_artist_similarity,
        similarity,
        NO_MATCH_MARKER,
        ERROR_MARKER,
        REJECT_MARKER,
    )

    conn = ctx["conn"]
    candidates = load_candidates(
        conn,
        limit,
        force,
        force_no_match,
        force_error,
        force_reject,
        force_all,
        artist_like=None,
    )
    logger.info("MBID stage: loaded %d candidates (force=%s)", len(candidates), force)

    prog = (
        ProgressLogger(
            logger,
            total=len(candidates),
            label="mbid",
            unit="tracks",
            interval_s=getattr(args, "progress_interval", 15.0),
            every_n=getattr(args, "progress_every", 500),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
        if getattr(args, "progress", True)
        else None
    )
    session = get_session(timeout=10.0, max_retries=3)
    cur = conn.cursor()
    updated = 0
    skipped = 0
    errors = 0
    processed = 0

    last_call = 0.0
    for track_id, title, artist, duration_ms in candidates:
        processed += 1
        if prog:
            prog.update(detail=f"{artist} - {title}")
        elapsed = time.perf_counter() - last_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        try:
            rec = search_recording_relaxed(session, artist, title, duration_ms, tolerance_ms=4000)
            last_call = time.perf_counter()
            if rec:
                mbid = rec.get("id")
                mb_title = rec.get("title", "")
                mb_score = rec.get("score", 0)
                artist_credit = rec.get("artist-credit") or []
                credit_names = [ac.get("name") for ac in artist_credit if isinstance(ac, dict) and ac.get("name")]
                mb_artist = " & ".join(credit_names)
                mb_len = rec.get("length")
                dur_diff = None
                if duration_ms is not None and mb_len is not None:
                    dur_diff = abs(int(mb_len) - int(duration_ms))
                title_sim = similarity(title, mb_title)
                artist_sim = best_artist_similarity(artist, mb_artist or "")

                if artist_sim < 0.40 or title_sim < 0.55:
                    skipped += 1
                    cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (REJECT_MARKER, track_id))
                    continue
                if duration_ms is not None and mb_len is not None and dur_diff is not None and dur_diff > 4000:
                    skipped += 1
                    cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (REJECT_MARKER, track_id))
                    continue
                elif mb_len is None and not (mb_score >= 80 and artist_sim >= 0.7 and title_sim >= 0.7):
                    skipped += 1
                    cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (REJECT_MARKER, track_id))
                    continue

                cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (mbid, track_id))
                updated += cur.rowcount
            else:
                skipped += 1
                cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (NO_MATCH_MARKER, track_id))
        except Exception as exc:
            errors += 1
            logger.debug("MBID error for %s - %s: %s", artist, title, exc)
            cur.execute("UPDATE tracks SET musicbrainz_id = ? WHERE track_id = ?", (ERROR_MARKER, track_id))
            last_call = time.perf_counter()

        if processed % 200 == 0:
            conn.commit()

    conn.commit()
    if prog:
        prog.finish(detail=f"MBID stage processed {processed:,} tracks")
    logger.info("MBID stage: updated=%d skipped=%d errors=%d", updated, skipped, errors)
    return {"updated": updated, "skipped": skipped, "errors": errors, "total": len(candidates)}


def stage_scan(ctx: Dict) -> Dict:
    """
    Run filesystem scan (incremental by default) and report counts.
    Uses LibraryScanner to pull new/modified files into the DB.
    """
    args = ctx["args"]
    quick = not args.force  # default: incremental; --force triggers full scan
    limit = args.limit if args.limit and args.limit > 0 else None

    cur = ctx["conn"].cursor()
    file_genres_before = None
    try:
        cur.execute("SELECT COUNT(*) AS c FROM track_genres WHERE source = 'file'")
        file_genres_before = cur.fetchone()["c"]
    except sqlite3.OperationalError:
        file_genres_before = None

    scanner = LibraryScanner(config_path=ctx["config_path"])
    scan_stats = scanner.run(
        quick=quick,
        limit=limit,
        progress=getattr(args, "progress", True),
        progress_interval=getattr(args, "progress_interval", 15.0),
        progress_every=getattr(args, "progress_every", 500),
        verbose_each=bool(getattr(args, "verbose", False)),
    ) or {"total": 0, "new": 0, "updated": 0, "failed": 0}
    scanner.close()

    cur.execute("SELECT COUNT(*) AS c FROM tracks WHERE file_path IS NOT NULL")
    total = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM tracks WHERE sonic_features IS NULL AND file_path IS NOT NULL")
    pending_sonic = cur.fetchone()["c"]
    total_artists = None
    artists_with = None
    try:
        cur.execute("SELECT COUNT(DISTINCT artist) AS c FROM tracks WHERE artist IS NOT NULL")
        total_artists = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(DISTINCT artist) AS c FROM artist_genres")
        artists_with = cur.fetchone()["c"]
    except sqlite3.OperationalError:
        logger.warning("Normalized genre tables missing; scan will report sonic stats only.")
    file_genres_after = None
    file_genres_delta = None
    if file_genres_before is not None:
        try:
            cur.execute("SELECT COUNT(*) AS c FROM track_genres WHERE source = 'file'")
            file_genres_after = cur.fetchone()["c"]
            file_genres_delta = file_genres_after - file_genres_before
        except sqlite3.OperationalError:
            file_genres_after = None
            file_genres_delta = None

    return {
        "total_tracks": total,
        "pending_sonic": pending_sonic,
        "artists_with_genres": artists_with,
        "total_artists": total_artists,
        "scan_total": scan_stats.get("total", 0),
        "scan_new": scan_stats.get("new", 0),
        "scan_updated": scan_stats.get("updated", 0),
        "scan_failed": scan_stats.get("failed", 0),
        "file_genres_before": file_genres_before,
        "file_genres_after": file_genres_after,
        "file_genres_delta": file_genres_delta,
        "orphaned": scan_stats.get("orphaned", {}),
        "skipped": False,
    }


def stage_genres(ctx: Dict) -> Dict:
    db_path = ctx["db_path"]
    args = ctx["args"]
    limit = args.limit
    force = args.force

    conn = ctx["conn"]
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT COUNT(DISTINCT t.artist) AS c
            FROM tracks t
            LEFT JOIN artist_genres g
              ON t.artist = g.artist
             AND g.source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
            WHERE t.artist IS NOT NULL AND TRIM(t.artist) != ''
              AND t.file_path IS NOT NULL AND t.file_path != ''
              AND g.artist IS NULL
            """
        )
        missing_artists = cur.fetchone()["c"]
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM albums a
            LEFT JOIN (
                SELECT DISTINCT album_id FROM album_genres WHERE source = 'musicbrainz_release'
            ) g ON a.album_id = g.album_id
            WHERE g.album_id IS NULL
              AND a.album_id IN (
                  SELECT DISTINCT album_id
                  FROM tracks
                  WHERE file_path IS NOT NULL AND file_path != ''
                    AND album_id IS NOT NULL AND album != ''
              )
            """
        )
        missing_albums = cur.fetchone()["c"]
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM artist_genres
            WHERE source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
              AND genre != '__EMPTY__'
            """
        )
        artist_genres_before = cur.fetchone()["c"]
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM album_genres
            WHERE source = 'musicbrainz_release'
              AND genre != '__EMPTY__'
            """
        )
        album_genres_before = cur.fetchone()["c"]
    except sqlite3.OperationalError:
        logger.warning("Normalized genre tables missing; skipping genres stage.")
        return {"skipped": True, "reason": "missing_normalized_tables"}

    if not force and missing_artists == 0 and missing_albums == 0:
        logger.info("Skipping genres stage (no missing artists/albums; use --force to re-run)")
        return {
            "missing_artists": missing_artists,
            "missing_albums": missing_albums,
            "added_artist_genres": 0,
            "added_album_genres": 0,
            "skipped": True,
        }

    updater = NormalizedGenreUpdater(config_path=ctx["config_path"], db_path=db_path)
    updater.update_artist_genres(
        limit=limit,
        progress=getattr(args, "progress", True),
        progress_interval=getattr(args, "progress_interval", 15.0),
        progress_every=getattr(args, "progress_every", 500),
        verbose_each=bool(getattr(args, "verbose", False)),
    )
    updater.update_album_genres(
        limit=limit,
        progress=getattr(args, "progress", True),
        progress_interval=getattr(args, "progress_interval", 15.0),
        progress_every=getattr(args, "progress_every", 500),
        verbose_each=bool(getattr(args, "verbose", False)),
    )
    updater.close()

    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM artist_genres
        WHERE source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
          AND genre != '__EMPTY__'
        """
    )
    artist_genres_after = cur.fetchone()["c"]
    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM album_genres
        WHERE source = 'musicbrainz_release'
          AND genre != '__EMPTY__'
        """
    )
    album_genres_after = cur.fetchone()["c"]
    cur.execute(
        """
        SELECT COUNT(DISTINCT t.artist) AS c
        FROM tracks t
        LEFT JOIN artist_genres g
          ON t.artist = g.artist
         AND g.source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
        WHERE t.artist IS NOT NULL AND TRIM(t.artist) != ''
          AND t.file_path IS NOT NULL AND t.file_path != ''
          AND g.artist IS NULL
        """
    )
    missing_artists_after = cur.fetchone()["c"]
    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM albums a
        LEFT JOIN (
            SELECT DISTINCT album_id FROM album_genres WHERE source = 'musicbrainz_release'
        ) g ON a.album_id = g.album_id
        WHERE g.album_id IS NULL
          AND a.album_id IN (
              SELECT DISTINCT album_id
              FROM tracks
              WHERE file_path IS NOT NULL AND file_path != ''
                AND album_id IS NOT NULL AND album != ''
          )
        """
    )
    missing_albums_after = cur.fetchone()["c"]

    return {
        "missing_artists": missing_artists_after,
        "missing_albums": missing_albums_after,
        "added_artist_genres": max(0, artist_genres_after - artist_genres_before),
        "added_album_genres": max(0, album_genres_after - album_genres_before),
        "skipped": False,
    }


def stage_discogs(ctx: Dict) -> Dict:
    """
    Fetch Discogs genres/styles for library albums (PRODUCTION REQUIRED).
    Complements MusicBrainz data with additional genre sources.

    Requires: DISCOGS_TOKEN environment variable or discogs.token in config.yaml
    Get token from: https://www.discogs.com/settings/developers (personal user token)
    """
    db_path = ctx["db_path"]
    config_path = ctx["config_path"]
    args = ctx["args"]

    conn = ctx["conn"]
    discogs_before = None
    try:
        discogs_before = conn.execute(
            """
            SELECT COUNT(*) FROM album_genres
            WHERE source IN ('discogs_release', 'discogs_master')
              AND genre != '__EMPTY__'
            """
        ).fetchone()[0]
    except sqlite3.OperationalError:
        discogs_before = None

    # Check if Discogs token is available - REQUIRED for production
    token = os.getenv("DISCOGS_TOKEN") or load_config_token(Path(config_path) if config_path else None)
    if not token:
        msg = (
            "Discogs token required for production pipeline. "
            "Set DISCOGS_TOKEN environment variable or add discogs.token to config.yaml. "
            "Get token from: https://www.discogs.com/settings/developers"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Try to create Discogs client
    try:
        client = DiscogsClient(token)
    except Exception as exc:
        msg = f"Failed to initialize Discogs client: {exc}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Check album_genres table exists
    try:
        already_processed = conn.execute(
            "SELECT COUNT(DISTINCT album_id) FROM album_genres WHERE source IN ('discogs_release', 'discogs_master')"
        ).fetchone()[0]
    except sqlite3.OperationalError as exc:
        msg = f"album_genres table missing or inaccessible: {exc}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Check how many albums need Discogs processing
    try:
        total_albums_count = conn.execute(
            """
            SELECT COUNT(DISTINCT album_id)
            FROM tracks
            WHERE album IS NOT NULL AND album != ''
              AND file_path IS NOT NULL AND file_path != ''
            """
        ).fetchone()[0]

        # Count albums that DON'T have Discogs data yet
        albums_needing_discogs = conn.execute(
            """
            SELECT COUNT(DISTINCT t.album_id)
            FROM tracks t
            WHERE t.album IS NOT NULL AND t.album != ''
              AND t.file_path IS NOT NULL AND t.file_path != ''
            AND t.album_id NOT IN (
                SELECT DISTINCT album_id FROM album_genres
                WHERE source IN ('discogs_release','discogs_master')
            )
            """
        ).fetchone()[0]
    except sqlite3.OperationalError:
        total_albums_count = 0
        albums_needing_discogs = 0

    # If all albums already processed and not forcing, skip this stage
    if albums_needing_discogs == 0 and not args.force:
        logger.info(
            "Skipping Discogs stage: all %d albums already have Discogs data (use --force to recheck)",
            total_albums_count
        )
        return {
            "skipped": False,
            "total_albums": total_albums_count,
            "hits": already_processed,
            "misses": 0,
            "added_discogs_genres": 0,
            "reason": "all_processed",
        }

    logger.info("Starting Discogs genre fetcher (already processed: %d/%d albums)...", already_processed, total_albums_count)

    misses = []
    total_hits = 0
    total_albums = 0

    try:
        # Get ALL albums, but we'll filter in the loop
        albums_list = list(iter_albums(conn, None, args.limit, None))
        total_to_process = len(albums_list)

        if total_to_process == 0:
            logger.info("No albums found in library to process")
            return {
                "skipped": False,
                "total_albums": 0,
                "hits": 0,
                "misses": 0,
                "added_discogs_genres": 0,
                "reason": "no_albums",
            }

        # Only process albums that need Discogs data
        albums_to_fetch = []
        for album in albums_list:
            has_data, has_empty = discogs_status(conn, album.album_id)

            # Process if: no data and (not empty marker OR force)
            if not has_data and (not has_empty or args.force):
                albums_to_fetch.append(album)
            # Also process empty markers if forced
            elif has_empty and args.force:
                albums_to_fetch.append(album)

        if len(albums_to_fetch) == 0:
            logger.info("No new albums to fetch from Discogs (all %d have data or are marked empty)", total_to_process)
            return {
                "skipped": False,
                "total_albums": total_to_process,
                "hits": 0,
                "misses": 0,
                "added_discogs_genres": 0,
                "reason": "none_to_process",
            }

        logger.info("Processing %d albums needing Discogs data (of %d total)...", len(albums_to_fetch), total_to_process)

        prog = (
            ProgressLogger(
                logger,
                total=len(albums_to_fetch),
                label="discogs",
                unit="albums",
                interval_s=getattr(args, "progress_interval", 15.0),
                every_n=getattr(args, "progress_every", 500),
                verbose_each=bool(getattr(args, "verbose", False)),
            )
            if getattr(args, "progress", True)
            else None
        )

        for idx, album in enumerate(albums_to_fetch, start=1):
            total_albums += 1
            if prog:
                prog.update(detail=f"{album.artist} - {album.title}")

            try:
                match = best_match(client, album, threshold=0.55, strict_artist=False)
            except Exception as exc:
                logger.debug("Discogs search failed for %s - %s: %s", album.artist, album.title, exc)
                misses.append((album, "search_error", str(exc)))
                continue

            if not match:
                logger.debug("Discogs no match for %s - %s", album.artist, album.title)
                upsert_album_genres(conn, album.album_id, ["__EMPTY__"], "discogs_release", args.dry_run)
                misses.append((album, "no_match", ""))
                continue

            release_id = match.get("id")
            master_id = match.get("master_id")

            try:
                genres, styles = fetch_genres(client, release_id, master_id)
            except Exception as exc:
                logger.debug("Discogs fetch failed for %s - %s: %s", album.artist, album.title, exc)
                misses.append((album, "fetch_error", str(exc)))
                continue

            norm_genres = [normalize_tag(g) for g in genres if g]
            norm_styles = [normalize_tag(s) for s in styles if s]

            # Write to database
            upsert_album_genres(conn, album.album_id, norm_genres, "discogs_release", args.dry_run)
            if master_id and norm_styles:
                upsert_album_genres(conn, album.album_id, norm_styles, "discogs_master", args.dry_run)

            total_hits += 1

    except Exception as exc:
        logger.error("Discogs stage failed: %s", exc)
        raise RuntimeError(f"Discogs stage failed: {exc}") from exc

    if prog:
        prog.finish(detail=f"Discogs processed {total_albums:,} albums")

    logger.info("Discogs stage complete: %d albums processed, %d hits, %d misses", total_albums, total_hits, len(misses))

    discogs_after = None
    added_discogs_genres = 0
    if discogs_before is not None:
        try:
            discogs_after = conn.execute(
                """
                SELECT COUNT(*) FROM album_genres
                WHERE source IN ('discogs_release', 'discogs_master')
                  AND genre != '__EMPTY__'
                """
            ).fetchone()[0]
            added_discogs_genres = max(0, discogs_after - discogs_before)
        except sqlite3.OperationalError:
            added_discogs_genres = 0

    return {
        "skipped": False,
        "total_albums": total_albums,
        "hits": total_hits,
        "misses": len(misses),
        "added_discogs_genres": added_discogs_genres,
    }


def stage_sonic(ctx: Dict) -> Dict:
    args = ctx["args"]
    force = args.force
    limit = args.limit
    workers_arg = args.workers
    if isinstance(workers_arg, str) and workers_arg.lower() == "auto":
        workers = None
    else:
        workers = int(workers_arg)
    pipeline = SonicFeaturePipeline(
        db_path=ctx["db_path"],
        use_beat_sync=False,
        use_beat3tower=True,
    )
    try:
        pending = pipeline.get_pending_tracks(limit, force=force)
    except sqlite3.OperationalError as exc:
        logger.warning("Skipping sonic stage; schema missing required columns (%s)", exc)
        return {"pending": None, "skipped": True, "reason": "schema_missing"}
    if not force and len(pending) == 0:
        logger.info("Skipping sonic stage (no pending tracks; use --force to re-run)")
        return {"pending": 0, "skipped": True}
    mode = "beat3tower"
    logger.info(f"Running sonic stage in {mode} mode")
    start_ts = int(time.time())
    pipeline.run(
        limit=limit,
        workers=workers,
        force=force,
        progress=getattr(args, "progress", True),
        progress_interval=getattr(args, "progress_interval", 15.0),
        progress_every=getattr(args, "progress_every", 500),
        verbose_each=bool(getattr(args, "verbose", False)),
    )
    updated = 0
    try:
        cursor = ctx["conn"].cursor()
        cursor.execute(
            """
            SELECT COUNT(*) AS c
            FROM tracks
            WHERE sonic_analyzed_at IS NOT NULL
              AND sonic_analyzed_at >= ?
            """,
            (start_ts,),
        )
        updated = cursor.fetchone()["c"]
    except sqlite3.OperationalError:
        updated = len(pending)
    return {"pending": len(pending), "skipped": False, "mode": mode, "updated": updated}


def stage_genre_sim(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    out_path = out_dir / "genre_similarity_matrix.npz"
    force_rebuild = ctx["args"].force or bool(ctx.get("genres_dirty")) or bool(ctx.get("force_stage"))
    if out_path.exists() and not force_rebuild:
        logger.info("Skipping genre-sim stage (exists: %s; use --force to rebuild)", out_path)
        return {"path": str(out_path), "skipped": True}
    if out_path.exists() and force_rebuild and not ctx["args"].force:
        logger.info("Rebuilding genre-sim (new genres detected since last build)")
    try:
        result = build_genre_similarity_matrix(
            db_path=ctx["db_path"],
            config_path=ctx["config_path"],
            out_path=str(out_path),
            min_count=2,
            max_genres=0,
        )
    except RuntimeError as exc:
        logger.warning("Skipping genre-sim stage: %s", exc)
        return {"path": str(out_path), "skipped": True, "reason": str(exc)}
    return {"path": str(out_path), "skipped": False, "stats": result.stats}     


def _write_artifact_manifest(out_dir: Path, fingerprint: str, config_hash: str, stats: Optional[Dict]) -> Path:
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "config_hash": config_hash,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats or {},
    }
    manifest_path = out_dir / "artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def stage_artifacts(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    out_path = out_dir / "data_matrices_step1.npz"
    genre_sim_path = out_dir / "genre_similarity_matrix.npz"
    genre_sim_use = genre_sim_path if genre_sim_path.exists() else None
    force_rebuild = (
        ctx["args"].force
        or bool(ctx.get("genres_dirty"))
        or bool(ctx.get("sonic_dirty"))
        or bool(ctx.get("force_stage"))
    )
    if out_path.exists() and not force_rebuild:
        logger.info("Skipping artifacts stage (exists: %s; use --force to rebuild)", out_path)
        return {"path": str(out_path), "skipped": True}
    if out_path.exists() and force_rebuild and not ctx["args"].force:
        logger.info("Rebuilding artifacts (new genres or sonic updates detected since last build)")
    try:
        result = build_ds_artifacts(
            db_path=ctx["db_path"],
            config_path=ctx["config_path"],
            out_path=out_path,
            genre_sim_path=genre_sim_use,
            max_tracks=ctx["args"].max_tracks or ctx["args"].limit or 0,        
        )
    except RuntimeError as exc:
        logger.warning("Skipping artifacts stage: %s", exc)
        return {"path": str(out_path), "skipped": True, "reason": str(exc)}     
    fingerprint = compute_stage_fingerprint(ctx, "artifacts")
    manifest_path = _write_artifact_manifest(out_dir, fingerprint, ctx.get("config_hash", ""), getattr(result, "stats", {}))
    return {
        "path": str(out_path),
        "skipped": False,
        "stats": getattr(result, "stats", {}),
        "fingerprint": fingerprint,
        "manifest": str(manifest_path),
    }


def stage_verify(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    artifact_path = out_dir / "data_matrices_step1.npz"
    if not artifact_path.exists():
        logger.warning("Verify stage skipped: artifact not found at %s", artifact_path)
        return {"skipped": True}
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_fp = None
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                manifest_fp = manifest_data.get("fingerprint")
        except Exception:
            manifest_fp = None
    current_fp = compute_stage_fingerprint(ctx, "artifacts")
    bundle = load_artifact_bundle(artifact_path)
    issues = []
    if manifest_fp is None:
        issues.append("missing_manifest")
    elif manifest_fp != current_fp:
        issues.append("stale_artifact")
    if bundle.track_ids.size == 0:
        issues.append("no_tracks")
    if len(bundle.track_id_to_index) != bundle.track_ids.size:
        issues.append("duplicate_track_ids")
    if bundle.X_sonic.shape[0] != bundle.X_genre_raw.shape[0]:
        issues.append("row_mismatch")
    if np.any([k == "" for k in bundle.artist_keys]):
        issues.append("empty_artist_keys")
    return {
        "skipped": False,
        "tracks": int(bundle.track_ids.size),
        "genres": int(bundle.genre_vocab.size),
        "issues": issues,
        "artifact_fingerprint": current_fp,
    }


STAGE_FUNCS = {
    "scan": stage_scan,
    "mbid": stage_mbid,
    "genres": stage_genres,
    "discogs": stage_discogs,
    "sonic": stage_sonic,
    "genre-sim": stage_genre_sim,
    "artifacts": stage_artifacts,
    "verify": stage_verify,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from src.logging_utils import add_logging_args

    parser = argparse.ArgumentParser(description="Analyze library pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--db-path", help="Override DB path (default from config)")
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER_DEFAULT),
        help="Comma-separated stages to run (default: scan,genres,discogs,sonic,genre-sim,artifacts,verify)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Workers for sonic stage (int or 'auto'; default: auto)",
    )
    parser.add_argument("--limit", type=int, help="Limit tracks for sonic/artifacts")
    parser.add_argument("--max-tracks", type=int, default=0, help="Cap tracks for artifact build (0=all)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    parser.add_argument("--force-no-match", action="store_true", help="MBID stage: reprocess __NO_MATCH__ tracks")
    parser.add_argument("--force-error", action="store_true", help="MBID stage: reprocess __ERROR__ tracks")
    parser.add_argument("--force-reject", action="store_true", help="MBID stage: reprocess __REJECT__ tracks")
    parser.add_argument("--force-all", action="store_true", help="MBID stage: process all tracks regardless of existing musicbrainz_id")
    parser.add_argument("--out-dir", help="Output directory for artifacts")
    parser.add_argument("--beat-sync", action="store_true", help="DEPRECATED: legacy sonic mode is disabled")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True,
                        help="Enable progress logging (default)")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable progress logging")
    parser.add_argument("--progress-interval", type=float, default=15.0,
                        help="Seconds between progress updates (default: 15)")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="Items between progress updates (default: 500)")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-item progress (DEBUG)")
    add_logging_args(parser)
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    global logger
    stages_requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages_requested:
        if s not in STAGE_FUNCS:
            raise ValueError(f"Unknown stage: {s}")

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("Plan (dry run):")
        logger.info("  stages: %s", ", ".join(stages_requested))
        logger.info("  out_dir: %s", out_dir)
        return 0

    from src.logging_utils import configure_logging, resolve_log_level
    run_id = str(uuid.uuid4())
    log_level = resolve_log_level(args)
    if getattr(args, "verbose", False) and not getattr(args, "debug", False) and not getattr(args, "quiet", False) and getattr(args, "log_level", "INFO").upper() == "INFO":
        log_level = "DEBUG"
    log_file = getattr(args, 'log_file', None) or 'logs/analyze_library.log'
    configure_logging(level=log_level, log_file=log_file, run_id=run_id, show_run_id=getattr(args, "show_run_id", False))

    # Re-get logger after configuration
    logger = logging.getLogger("analyze_library")

    if args.beat_sync:
        logger.error("Legacy sonic mode (--beat-sync) is deprecated and disabled. Beat3tower is always used.")
        return 2

    cfg = Config(args.config)
    db_path = args.db_path or cfg.library_database_path
    config_hash = compute_config_hash(cfg, args)
    git_commit = _get_git_commit()

    # Shared DB connection for quick checks
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError as exc:
        logger.warning("Failed to apply SQLite pragmas (%s)", exc)

    ctx = {
        "config_path": args.config,
        "db_path": db_path,
        "out_dir": out_dir,
        "args": args,
        "conn": conn,
        "genres_dirty": False,
        "sonic_dirty": False,
        "config_hash": config_hash,
        "library_root": cfg.library_music_directory,
    }

    ensure_analyze_state_schema(conn)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {
        "run_id": run_id,
        "started_at": started_at,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "stages": {},
        "out_dir": str(out_dir),
    }
    logger.info("Analyze run start | run_id=%s | db=%s | out_dir=%s | stages=%s", run_id, db_path, out_dir, ", ".join(stages_requested))
    logger.info("  config_hash=%s | git=%s", config_hash, git_commit or "-")
    logger.info(
        "  progress=%s interval=%.1fs every=%d verbose_each=%s",
        "on" if getattr(args, "progress", True) else "off",
        float(getattr(args, "progress_interval", 15.0)),
        int(getattr(args, "progress_every", 500)),
        bool(getattr(args, "verbose", False)),
    )
    pending_snapshot = []
    for st in stages_requested:
        pc, pl = estimate_stage_units(ctx, st)
        if pc is not None:
            pending_snapshot.append(f"{st}={pc:,}" + (f" {pl}" if pl else ""))
    if pending_snapshot:
        logger.info("  pending_estimates: %s", "; ".join(pending_snapshot))

    start_total = time.time()
    for stage in stages_requested:
        func = STAGE_FUNCS[stage]
        stage_start = time.time()
        fingerprint_before = compute_stage_fingerprint(ctx, stage)
        last_fp = get_last_fingerprint(conn, stage)
        allow_skip = stage not in ("scan",)
        ctx["force_stage"] = bool(last_fp and fingerprint_before != last_fp)
        pending_count, pending_label = estimate_stage_units(ctx, stage)
        pending_msg = ""
        if pending_label:
            pending_val = "?" if pending_count is None else f"{pending_count:,}"
            pending_msg = f" | pending={pending_val} {pending_label}"
        else:
            pending_val = "unknown"
        if (not args.force) and allow_skip and last_fp and fingerprint_before == last_fp:
            duration = time.time() - stage_start
            logger.info(
                "run_id=%s | stage=%s | decision=skipped | reason=fingerprint_same | pending=%s",
                run_id,
                stage,
                pending_val if pending_msg else "unknown",
            )
            report["stages"][stage] = {
                "decision": "skipped",
                "reason": "fingerprint_unchanged",
                "duration_sec": duration,
                "fingerprint_before": fingerprint_before,
                "last_success_fingerprint": last_fp,
                "pending_estimate": pending_count,
                "pending_label": pending_label,
                "processed_count": 0,
                "errors_count": 0,
                "throughput": None,
            }
            ctx["force_stage"] = False
            continue

        run_reason = "forced" if args.force else ("fingerprint_changed" if last_fp and fingerprint_before != last_fp else "required")
        logger.info(
            "run_id=%s | stage=%s | decision=%s | reason=%s | pending=%s",
            run_id,
            stage,
            "forced" if args.force else "ran",
            run_reason,
            pending_val if pending_msg else "unknown",
        )
        result = func(ctx)
        duration = time.time() - stage_start
        fingerprint_after = compute_stage_fingerprint(ctx, stage)
        set_last_fingerprint(conn, stage, fingerprint_after)
        items = summarize_items(result)
        processed_count, errors_count, top_err = _extract_processed_and_errors(stage, result)
        rate = None
        if processed_count is not None and duration > 0:
            try:
                rate = processed_count / duration
            except Exception:
                rate = None
        logger.info(
            "run_id=%s | stage=%s | decision=%s | reason=%s | processed=%s | elapsed_s=%.2f | throughput=%s | errors=%d | top_error_categories=%s",
            run_id,
            stage,
            "forced" if args.force else "ran",
            run_reason,
            processed_count if processed_count is not None else (items or "-"),
            duration,
            f"{rate:.1f}/s" if rate is not None else "-",
            errors_count,
            top_err or "-",
        )
        if stage == "scan" and isinstance(result, dict):
            mod_reasons = result.get("modified_reasons") or {}
            if mod_reasons:
                reason_bits = ", ".join(f"{k}={v}" for k, v in sorted(mod_reasons.items()))
                logger.info("  scan modified breakdown: %s", reason_bits)
                if args.verbose:
                    examples = result.get("modified_examples") or {}
                    for reason, paths in examples.items():
                        if paths:
                            logger.debug("    %s examples: %s", reason, "; ".join(paths))
        report["stages"][stage] = {
            "decision": "ran",
            "result": result,
            "duration_sec": duration,
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
            "reason": run_reason,
            "pending_estimate": pending_count,
            "pending_label": pending_label,
            "processed_count": processed_count,
            "errors_count": errors_count,
            "throughput": rate,
        }
        ctx["force_stage"] = False
        if stage == "scan":
            scan_total = result.get("scan_total", 0) if isinstance(result, dict) else 0
            file_genres_delta = result.get("file_genres_delta", 0) if isinstance(result, dict) else 0
            orphaned = result.get("orphaned", {}) if isinstance(result, dict) else {}
            orphaned_removed = any((v or 0) > 0 for v in (orphaned or {}).values())
            if scan_total > 0 or (file_genres_delta is not None and file_genres_delta != 0) or orphaned_removed:
                ctx["genres_dirty"] = True
        elif stage == "genres":
            if result.get("added_artist_genres", 0) > 0 or result.get("added_album_genres", 0) > 0:
                ctx["genres_dirty"] = True
        elif stage == "discogs":
            if result.get("added_discogs_genres", 0) > 0:
                ctx["genres_dirty"] = True
        elif stage == "sonic":
            if result.get("updated", 0) > 0:
                ctx["sonic_dirty"] = True

    report["total_duration_sec"] = time.time() - start_total
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report_path = out_dir / "analyze_run_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote run report to %s", report_path)
    logger.info("RUN RECAP | run_id=%s | config_hash=%s | report=%s", run_id, config_hash, report_path)
    verify_issues = []
    if "verify" in report["stages"]:
        verify_result = report["stages"]["verify"].get("result") or {}
        verify_issues = verify_result.get("issues") or []
    logger.info("  verify_issues=%s", ",".join(verify_issues) if verify_issues else "none")
    for stage in stages_requested:
        stage_report = report["stages"].get(stage, {})
        decision = stage_report.get("decision", "-")
        reason = stage_report.get("reason", "-")
        pending_before = stage_report.get("pending_estimate")
        processed = stage_report.get("processed_count")
        elapsed = stage_report.get("duration_sec")
        rate = stage_report.get("throughput")
        errors_count = stage_report.get("errors_count", 0)
        top_err = "-"
        result_obj = stage_report.get("result") or {}
        if isinstance(result_obj, dict) and result_obj.get("issues"):
            top_err = ",".join((result_obj.get("issues") or [])[:3])
        logger.info(
            "  stage=%s | decision=%s | reason=%s | pending_before=%s | processed=%s | elapsed=%.2fs | rate=%s | errors=%s | top_error_categories=%s",
            stage,
            decision,
            reason,
            pending_before if pending_before is not None else "-",
            processed if processed is not None else "-",
            elapsed if elapsed is not None else 0.0,
            f"{rate:.2f}/s" if rate else "-",
            errors_count,
            top_err,
        )
    logger.info("Total elapsed: %.2fs", report["total_duration_sec"])
    conn.close()
    return 0


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
