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
from pathlib import Path
from typing import Dict, List, Optional

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

DEFAULT_OUT_DIR = ROOT_DIR / "data" / "artifacts" / "beat3tower_32k"
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "sonic", "genre-sim", "artifacts", "verify"]

logger = logging.getLogger("analyze_library")

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
    )

    conn = ctx["conn"]
    candidates = load_candidates(
        conn,
        limit,
        force,
        force_no_match,
        force_error,
        force_all,
        artist_like=None,
    )
    logger.info("MBID stage: loaded %d candidates (force=%s)", len(candidates), force)

    session = get_session(timeout=10.0, max_retries=3)
    cur = conn.cursor()
    updated = 0
    skipped = 0
    errors = 0
    processed = 0

    last_call = 0.0
    for track_id, title, artist, duration_ms in candidates:
        processed += 1
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
                    continue
                if duration_ms is not None and mb_len is not None and dur_diff is not None and dur_diff > 4000:
                    skipped += 1
                    continue
                elif mb_len is None and not (mb_score >= 80 and artist_sim >= 0.7 and title_sim >= 0.7):
                    skipped += 1
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
    scan_stats = scanner.run(quick=quick, limit=limit) or {"total": 0, "new": 0, "updated": 0, "failed": 0}
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
    limit = ctx["args"].limit
    force = ctx["args"].force

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
    updater.update_artist_genres(limit=limit)
    updater.update_album_genres(limit=limit)
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

        start_time = time.monotonic()

        for idx, album in enumerate(albums_to_fetch, start=1):
            total_albums += 1

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

            if idx % 10 == 0 or idx == len(albums_to_fetch):
                elapsed = time.monotonic() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = len(albums_to_fetch) - idx
                est_sec = remaining / rate if rate > 0 else 0
                logger.info("Discogs: %d/%d processed (%d hits, est %.0fs remaining)", idx, len(albums_to_fetch), total_hits, est_sec)

    except Exception as exc:
        logger.error("Discogs stage failed: %s", exc)
        raise RuntimeError(f"Discogs stage failed: {exc}") from exc

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
    pipeline.run(limit=limit, workers=workers, force=force)
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
    force_rebuild = ctx["args"].force or bool(ctx.get("genres_dirty"))
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


def stage_artifacts(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    out_path = out_dir / "data_matrices_step1.npz"
    genre_sim_path = out_dir / "genre_similarity_matrix.npz"
    genre_sim_use = genre_sim_path if genre_sim_path.exists() else None
    force_rebuild = ctx["args"].force or bool(ctx.get("genres_dirty")) or bool(ctx.get("sonic_dirty"))
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
    return {"path": str(out_path), "skipped": False, "stats": result.stats}


def stage_verify(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    artifact_path = out_dir / "data_matrices_step1.npz"
    if not artifact_path.exists():
        logger.warning("Verify stage skipped: artifact not found at %s", artifact_path)
        return {"skipped": True}
    bundle = load_artifact_bundle(artifact_path)
    issues = []
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
    parser.add_argument("--force-all", action="store_true", help="MBID stage: process all tracks regardless of existing musicbrainz_id")
    parser.add_argument("--out-dir", help="Output directory for artifacts")
    parser.add_argument("--beat-sync", action="store_true", help="DEPRECATED: legacy sonic mode is disabled")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    add_logging_args(parser)
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    stages_requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages_requested:
        if s not in STAGE_FUNCS:
            raise ValueError(f"Unknown stage: {s}")

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Plan (dry run):")
        print(f"  stages: {', '.join(stages_requested)}")
        print(f"  out_dir: {out_dir}")
        return 0

    from src.logging_utils import configure_logging, resolve_log_level
    log_level = resolve_log_level(args)
    log_file = getattr(args, 'log_file', None) or 'analyze_library.log'
    configure_logging(level=log_level, log_file=log_file)

    # Re-get logger after configuration
    global logger
    logger = logging.getLogger("analyze_library")

    if args.beat_sync:
        logger.error("Legacy sonic mode (--beat-sync) is deprecated and disabled. Beat3tower is always used.")
        return 2

    cfg = Config(args.config)
    db_path = args.db_path or cfg.library_database_path

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
    }

    report = {"stages": {}, "out_dir": str(out_dir)}
    start_total = time.time()
    for stage in stages_requested:
        func = STAGE_FUNCS[stage]
        stage_start = time.time()
        logger.info("Stage start: %s", stage)
        result = func(ctx)
        duration = time.time() - stage_start
        logger.info("Stage end: %s (%.2fs)", stage, duration)
        report["stages"][stage] = {"result": result, "duration_sec": duration}
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
    report_path = out_dir / "analyze_run_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote run report to %s", report_path)
    conn.close()
    return 0


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
