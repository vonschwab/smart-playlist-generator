#!/usr/bin/env python3
"""
Unified Analyze Library pipeline:
- Genres (normalized)
- Sonic analysis
- Genre similarity matrix
- DS artifact build
- Verification
"""
import argparse
import json
import logging
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

DEFAULT_OUT_DIR = ROOT_DIR / "experiments" / "genre_similarity_lab" / "artifacts"
STAGE_ORDER_DEFAULT = ["scan", "genres", "sonic", "genre-sim", "artifacts", "verify"]

logger = logging.getLogger("analyze_library")


def stage_scan(ctx: Dict) -> Dict:
    """
    Run filesystem scan (incremental by default) and report counts.
    Uses LibraryScanner to pull new/modified files into the DB.
    """
    args = ctx["args"]
    quick = not args.force  # default: incremental; --force triggers full scan
    limit = args.limit if args.limit and args.limit > 0 else None

    scanner = LibraryScanner(config_path=ctx["config_path"])
    scanner.run(quick=quick, limit=limit)
    scanner.close()

    cur = ctx["conn"].cursor()
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
    return {
        "total_tracks": total,
        "pending_sonic": pending_sonic,
        "artists_with_genres": artists_with,
        "total_artists": total_artists,
        "skipped": False,
    }


def stage_genres(ctx: Dict) -> Dict:
    db_path = ctx["db_path"]
    limit = ctx["args"].limit
    force = ctx["args"].force

    conn = ctx["conn"]
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(DISTINCT artist) AS c FROM tracks WHERE artist IS NOT NULL")
        total_artists = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(DISTINCT artist) AS c FROM artist_genres")
        artists_with = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM albums")
        total_albums = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(DISTINCT album_id) AS c FROM album_genres")
        albums_with = cur.fetchone()["c"]
    except sqlite3.OperationalError:
        logger.warning("Normalized genre tables missing; skipping genres stage.")
        return {"skipped": True, "reason": "missing_normalized_tables"}

    if not force and artists_with >= total_artists and albums_with >= total_albums:
        logger.info("Skipping genres stage (already populated; use --force to re-run)")
        return {"artists_with": artists_with, "albums_with": albums_with, "skipped": True}

    updater = NormalizedGenreUpdater(config_path=ctx["config_path"], db_path=db_path)
    updater.update_artist_genres(limit=limit)
    updater.update_album_genres(limit=limit)
    updater.close()
    return {"artists_with": artists_with, "albums_with": albums_with, "skipped": False}


def stage_sonic(ctx: Dict) -> Dict:
    args = ctx["args"]
    force = args.force
    limit = args.limit
    use_beat_sync = args.beat_sync
    workers_arg = args.workers
    if isinstance(workers_arg, str) and workers_arg.lower() == "auto":
        workers = None
    else:
        workers = int(workers_arg)
    pipeline = SonicFeaturePipeline(db_path=ctx["db_path"], use_beat_sync=use_beat_sync)
    try:
        pending = pipeline.get_pending_tracks(limit, force=force)
    except sqlite3.OperationalError as exc:
        logger.warning("Skipping sonic stage; schema missing required columns (%s)", exc)
        return {"pending": None, "skipped": True, "reason": "schema_missing"}
    if not force and len(pending) == 0:
        logger.info("Skipping sonic stage (no pending tracks; use --force to re-run)")
        return {"pending": 0, "skipped": True}
    mode = "beat-sync" if use_beat_sync else "windowed"
    logger.info(f"Running sonic stage in {mode} mode")
    pipeline.run(limit=limit, workers=workers, force=force)
    return {"pending": len(pending), "skipped": False, "mode": mode}


def stage_genre_sim(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    out_path = out_dir / "genre_similarity_matrix.npz"
    if out_path.exists() and not ctx["args"].force:
        logger.info("Skipping genre-sim stage (exists: %s; use --force to rebuild)", out_path)
        return {"path": str(out_path), "skipped": True}
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
    if out_path.exists() and not ctx["args"].force:
        logger.info("Skipping artifacts stage (exists: %s; use --force to rebuild)", out_path)
        return {"path": str(out_path), "skipped": True}
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
    "genres": stage_genres,
    "sonic": stage_sonic,
    "genre-sim": stage_genre_sim,
    "artifacts": stage_artifacts,
    "verify": stage_verify,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze library pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--db-path", help="Override DB path (default from config)")
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER_DEFAULT),
        help="Comma-separated stages to run (genres,sonic,genre-sim,artifacts,verify)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Workers for sonic stage (int or 'auto'; default: auto)",
    )
    parser.add_argument("--limit", type=int, help="Limit tracks for sonic/artifacts")
    parser.add_argument("--max-tracks", type=int, default=0, help="Cap tracks for artifact build (0=all)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if outputs exist")
    parser.add_argument("--out-dir", help="Output directory for artifacts")
    parser.add_argument("--beat-sync", action="store_true", help="Use beat-synchronized feature extraction (Phase 2)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
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

    from src.logging_config import setup_logging
    logger = setup_logging(name='analyze_library', log_file='analyze_library.log')
    cfg = Config(args.config)
    db_path = args.db_path or cfg.library_database_path

    # Shared DB connection for quick checks
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    ctx = {
        "config_path": args.config,
        "db_path": db_path,
        "out_dir": out_dir,
        "args": args,
        "conn": conn,
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
