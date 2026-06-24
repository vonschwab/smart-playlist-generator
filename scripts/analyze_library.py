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
import contextlib
import json
import logging
import os
import sys
import time
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import sqlite3

from src.config_loader import Config
from src.analyze.genre_similarity import build_genre_similarity_matrix
from src.features.artifacts import load_artifact_bundle
from src.genre.artifact_identity import dense_sidecar_mismatch_reason_from_paths
from scripts.update_genres_v3_normalized import NormalizedGenreUpdater
from scripts.update_sonic import SonicFeaturePipeline
from scripts.scan_library import LibraryScanner
from src.playlist.subprocess_stream import run_streaming_subprocess
from scripts.update_discogs_genres import DiscogsClient, iter_albums, upsert_album_genres, best_match, fetch_genres, normalize_tag, discogs_status, load_config_token
from src.logging_utils import ProgressLogger
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.discovery import discover_releases
from src.ai_genre_enrichment.lastfm_enrichment import fetch_lastfm_tags
from src.ai_genre_enrichment.tag_adjudicator import adjudicate_tags
from src.ai_genre_enrichment.provider import resolve_enrichment_model
from src.ai_genre_enrichment.adjudication_store import AdjudicationStore
from src.ai_genre_enrichment.adjudication_runner import build_todo, run_adjudication
from src.ai_genre_enrichment.adjudication_apply import apply_adjudications
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.album_adjudicator import (
    ADJUDICATOR_INSTRUCTIONS,
    ADJUDICATOR_INSTRUCTIONS_THOROUGH,
    ADJUDICATOR_PROMPT_VERSION,
    ADJUDICATOR_PROMPT_VERSION_THOROUGH,
)
from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.genre.graph_adapter import load_graph_adapter
from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER

DEFAULT_OUT_DIR = ROOT_DIR / "data" / "artifacts" / "beat3tower_32k"
ENRICHMENT_DB_PATH = ROOT_DIR / "data" / "ai_genre_enrichment.db"
# Single source of truth for the default stage order, shared with the GUI/web
# worker via request_models. `enrich` stays runnable via `--stages enrich` but is
# not in the default run (the Sonnet `adjudicate`+`apply` path replaced it).
STAGE_ORDER_DEFAULT = list(ANALYZE_LIBRARY_STAGE_ORDER)

# Substrings that identify a TRANSIENT Claude failure (rate limit / usage window).
# These are resumable: the cache already persists whatever was adjudicated before
# the error, so a re-run will pick up where it left off.
# Hard config errors (SDK not installed, unauthenticated, bad model name) do NOT
# match these patterns and must still propagate loudly.
_TRANSIENT_CLAUDE_PATTERNS: tuple[str, ...] = (
    "Claude Code returned an error result: success",
    "Claude Code request failed after retries",
)

logger = logging.getLogger("analyze_library")


def effective_prompt_version(thorough: bool = False) -> str:
    instructions = ADJUDICATOR_INSTRUCTIONS_THOROUGH if thorough else ADJUDICATOR_INSTRUCTIONS
    base = ADJUDICATOR_PROMPT_VERSION_THOROUGH if thorough else ADJUDICATOR_PROMPT_VERSION
    h = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]
    return f"{base}+{h}"


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


def _configured_genre_source(ctx: Dict) -> str:
    """playlists.ds_pipeline.genre_source from config ('legacy' default).

    Reads the YAML directly rather than via Config — Config validates whole
    required sections, which is irrelevant (and fragile) for reading one key.
    """
    import yaml

    try:
        with open(ctx["config_path"], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return "legacy"
    value = ((cfg.get("playlists") or {}).get("ds_pipeline") or {}).get("genre_source")
    return str(value or "legacy").strip().lower()


def _mert_fold_settings(config_path: str) -> Tuple[bool, str]:
    """(fold_enabled, active_variant) for the post-artifact MERT fold.

    - ``analyze.mert.fold_into_artifact`` (default True) toggles the auto-fold
      that runs at the end of the artifacts stage. Set it False to keep the tower
      rollback active without a code change.
    - ``artifacts.sonic_variant_override`` (default 'mert') chooses the active
      variant the fold writes — set to 'tower_weighted' for the documented rollback.
    """
    import yaml

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    mert_cfg = (cfg.get("analyze") or {}).get("mert") or {}
    enabled = bool(mert_cfg.get("fold_into_artifact", True))
    override = ((cfg.get("artifacts") or {}).get("sonic_variant_override") or "mert")
    variant = str(override).strip() or "mert"
    return enabled, variant


def _assert_dbs_not_aliased(*db_paths: str) -> None:
    """Refuse to run if a SQLite DB (or its parent dir) is reached via a symlink/junction.

    Opening ONE SQLite DB through two different path strings (e.g. the real path AND a
    worktree symlink) creates two independent WAL/SHM sets that checkpoint over each
    other and corrupt the B-tree. This corrupted the genre sidecar on 2026-06-22. The
    pipeline hardcodes its data paths to ROOT_DIR, so a symlinked DB only happens when
    someone aliased it to run from a git worktree — fail loudly here instead of silently
    corrupting. Run analyze from the MAIN checkout. See the project memory
    `feedback_worktree_sqlite_wal_aliasing`.
    """
    import stat as _stat

    def _is_reparse(p: Path) -> bool:
        try:
            st = os.lstat(p)
        except OSError:
            return False
        if _stat.S_ISLNK(st.st_mode):
            return True
        attrs = getattr(st, "st_file_attributes", 0)  # Windows junction / dir-symlink
        return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT

    offenders: List[str] = []
    for raw in db_paths:
        p = Path(raw)
        if _is_reparse(p):
            offenders.append(str(p))
        elif _is_reparse(p.parent):
            offenders.append(f"{p}  (via symlinked dir {p.parent})")
    if offenders:
        raise RuntimeError(
            "Refusing to run analyze: SQLite DB reached through a symlink/junction -> "
            f"{'; '.join(offenders)}. This causes WAL-aliasing corruption (two -wal files "
            "for one DB). Run the analyze pipeline from the MAIN checkout, not a git "
            "worktree. See memory feedback_worktree_sqlite_wal_aliasing."
        )


def _sidecar_count(query: str, params: tuple = ()) -> int:
    """COUNT(*) against the enrichment sidecar, 0 if absent/unreadable."""
    try:
        if not ENRICHMENT_DB_PATH.exists():
            return 0
        conn = sqlite3.connect(f"file:{ENRICHMENT_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
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

    if stage == "lastfm":
        total_albums = _safe_count(conn, "SELECT COUNT(DISTINCT album_id) FROM albums "
                                         "WHERE album_id IS NOT NULL AND album_id != ''")
        lastfm_pages = _sidecar_count(
            "SELECT COUNT(*) FROM ai_genre_source_pages WHERE source_type='lastfm_tags'")
        key = {"stage": stage, "total_albums": total_albums, "lastfm_pages": lastfm_pages}
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

    if stage == "mert":
        # Universe = DB tracks with a file_path (what stage_mert actually embeds).
        # Keyed off the DB, not the artifact, so a freshly-scanned track flips this
        # fingerprint and the orchestrator re-runs MERT in the same pass — before
        # the artifact rebuild that would otherwise hide the new track.
        try:
            track_ids = sorted(
                str(r[0]) for r in conn.execute(
                    "SELECT track_id FROM tracks "
                    "WHERE file_path IS NOT NULL AND file_path != ''"
                ).fetchall()
            )
        except Exception:
            track_ids = []
        key = {"stage": stage, "track_ids": track_ids}
        return _hash_obj(key)

    if stage == "energy":
        from src.analyze.energy_runner import pending_energy, load_energy_config
        pending, total = pending_energy(Path(ctx["out_dir"]))
        cfg = load_energy_config(ctx["config_path"])
        key = {"stage": stage, "pending": pending, "total": total,
               "workers": cfg.workers, "distro": cfg.distro, "python": cfg.python}
        return _hash_obj(key)

    if stage == "enrich":
        source_pages = _sidecar_count("SELECT COUNT(*) FROM ai_genre_source_pages")
        signatures = _sidecar_count("SELECT COUNT(*) FROM enriched_genre_signatures")
        assignments = _sidecar_count(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments")
        key = {"stage": stage, "source_pages": source_pages,
               "signatures": signatures, "assignments": assignments}
        return _hash_obj(key)

    if stage == "adjudicate":
        total_albums = _safe_count(conn, "SELECT COUNT(*) FROM albums")
        done = _sidecar_count(
            "SELECT COUNT(DISTINCT album_id) FROM adjudications WHERE status='complete'")
        return _hash_obj({"stage": stage, "total_albums": total_albums, "done": done})
    if stage == "apply":
        complete = _sidecar_count(
            "SELECT COUNT(*) FROM adjudications WHERE status='complete'")
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        tax_version = load_default_layered_taxonomy().version
        return _hash_obj({"stage": stage, "complete": complete, "taxonomy": tax_version})

    if stage == "publish":
        side_assignments = _sidecar_count(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments")
        published_rows = _safe_count(conn, "SELECT COUNT(*) FROM release_effective_genres")
        key = {"stage": stage, "side_assignments": side_assignments,
               "published_rows": published_rows}
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
        # When artifacts consume the published graph genres, a re-publish must
        # change this fingerprint (legacy artifacts never read that table).
        effective_genre_rows = None
        if _configured_genre_source(ctx) == "graph":
            effective_genre_rows = _safe_count(
                conn, "SELECT COUNT(*) FROM release_effective_genres"
            )
        out_path = Path(ctx["out_dir"]) / "data_matrices_step1.npz"
        manifest_path = Path(ctx["out_dir"]) / "artifact_manifest.json"
        out_mtime = int(out_path.stat().st_mtime) if out_path.exists() else 0
        manifest_mtime = int(manifest_path.stat().st_mtime) if manifest_path.exists() else 0
        key = {
            "stage": stage,
            "tracks_with_features": tracks_with_features,
            "track_genres": genre_rows,
            "effective_genre_rows": effective_genre_rows,
            "config": cfg_hash,
            "artifact_exists": out_path.exists(),
            "artifact_mtime": out_mtime,
            "manifest_mtime": manifest_mtime,
        }
        return _hash_obj(key)

    if stage == "genre-embedding":
        art = Path(ctx["out_dir"]) / "data_matrices_step1.npz"
        sc = Path(ctx["out_dir"]) / "data_matrices_step1_genre_emb_dim64.npz"
        key = {
            "stage": stage,
            "artifact_exists": art.exists(),
            "artifact_mtime": int(art.stat().st_mtime) if art.exists() else 0,
            "sidecar_exists": sc.exists(),
            "sidecar_mtime": int(sc.stat().st_mtime) if sc.exists() else 0,
            "config": cfg_hash,
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
        if stage == "lastfm":
            total_albums = _safe_count(conn, "SELECT COUNT(DISTINCT album_id) FROM albums "
                                            "WHERE album_id IS NOT NULL AND album_id != ''")
            scraped = _sidecar_count(
                "SELECT COUNT(*) FROM ai_genre_source_pages WHERE source_type='lastfm_tags'")
            return max(0, total_albums - scraped), "releases needing Last.fm tags"
        if stage == "enrich":
            source_pages = _sidecar_count("SELECT COUNT(DISTINCT release_key) "
                                          "FROM ai_genre_source_pages")
            return source_pages, "releases with source pages to enrich"
        if stage == "adjudicate":
            total = _safe_count(conn, "SELECT COUNT(*) FROM albums")
            done = _sidecar_count(
                "SELECT COUNT(DISTINCT album_id) FROM adjudications WHERE status='complete'")
            return max(0, total - done), "albums to adjudicate"
        if stage == "apply":
            complete = _sidecar_count("SELECT COUNT(*) FROM adjudications WHERE status='complete'")
            return complete, "adjudications to apply"
        if stage == "publish":
            total_albums = _safe_count(conn, "SELECT COUNT(*) FROM albums "
                                            "WHERE album_id IS NOT NULL AND album_id != ''")
            return total_albums, "albums to resolve into release_effective_genres"
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
        if stage == "energy":
            from src.analyze.energy_runner import pending_energy
            pending, _total = pending_energy(Path(ctx["out_dir"]))
            return pending, "tracks needing energy descriptors"
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

    # Honor the orchestrator's resolved db_path (from --db-path / config) instead of
    # letting LibraryScanner fall back to its ROOT_DIR/data/metadata.db default —
    # that default ignored --db-path entirely (a silent "looks-wired-but-isn't" gap).
    scanner = LibraryScanner(config_path=ctx["config_path"], db_path=ctx["db_path"])
    scan_stats = scanner.run(
        quick=quick,
        limit=limit,
        cleanup=True,  # Always cleanup missing files during analyze pipeline
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


def _resolve_lastfm_api_key(ctx: Dict) -> Optional[str]:
    """Resolve the Last.fm API key from args, env, or config (in that order)."""
    args = ctx["args"]
    key = getattr(args, "lastfm_api_key", None) or os.environ.get("LASTFM_API_KEY")
    if key:
        return key
    try:
        return Config(ctx["config_path"]).lastfm_api_key or None
    except Exception:
        return None


def stage_lastfm(ctx: Dict) -> Dict:
    """Fetch Last.fm top tags into the sidecar for releases that lack them.

    No LLM. Deterministic classification only (adjudicate=False); the `enrich`
    stage owns AI adjudication. Missing API key raises (production-required,
    like the discogs stage).
    """
    import time

    args = ctx["args"]
    api_key = _resolve_lastfm_api_key(ctx)
    if not api_key:
        raise RuntimeError(
            "Last.fm API key required for the lastfm stage. Set LASTFM_API_KEY, "
            "pass --lastfm-api-key, or add lastfm.api_key to config.yaml."
        )

    store = SidecarStore(str(ENRICHMENT_DB_PATH))
    store.initialize()

    limit = args.limit if args.limit and args.limit > 0 else None
    releases = discover_releases(ctx["db_path"], limit=limit)
    if not releases:
        return {"skipped": True, "reason": "no_releases", "extracted": 0}

    already = store.release_keys_with_source_type("lastfm_tags")
    pending = [r for r in releases if args.force or r.release_key not in already]
    skipped_existing = len(releases) - len(pending)
    if not pending:
        logger.info("Skipping lastfm stage (all %d releases already scraped)", len(releases))
        return {"skipped": True, "reason": "all_scraped", "extracted": 0,
                "skipped_existing": skipped_existing}

    prog = (
        ProgressLogger(
            logger, total=len(pending), label="lastfm", unit="releases",
            interval_s=getattr(args, "progress_interval", 15.0),
            every_n=getattr(args, "progress_every", 500),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
        if getattr(args, "progress", True) else None
    )

    extracted = empty = failed = 0
    for release in pending:
        if prog:
            prog.update(detail=release.release_key)
        try:
            tags = fetch_lastfm_tags(
                artist=release.normalized_artist,
                album=release.normalized_album or None,
                api_key=api_key,
                limit=20,
            )
            if not tags:
                empty += 1
                time.sleep(0.25)
                continue
            album_segment = f"/album/{release.normalized_album}" if release.normalized_album else ""
            page_id = store.upsert_source_page(
                release_key=release.release_key,
                normalized_artist=release.normalized_artist,
                normalized_album=release.normalized_album,
                album_id=release.album_id,
                source_url=f"lastfm://artist/{release.normalized_artist}{album_segment}",
                source_type="lastfm_tags",
                identity_status="confirmed",
                identity_confidence=0.9,
                evidence_summary="Last.fm top tags via API.",
            )
            store.replace_source_tags(page_id, tags)
            store.classify_source_tags(page_id, adjudicate=False, model=None)
            extracted += 1
        except Exception as exc:  # network blip / API error — log and continue
            failed += 1
            logger.warning("Last.fm failed for %s: %s", release.release_key, exc)
        time.sleep(0.25)  # ~5 req/s courtesy limit

    if prog:
        prog.finish(detail=f"lastfm extracted {extracted:,} of {len(pending):,}")
    logger.info("lastfm stage: extracted=%d empty=%d failed=%d skipped_existing=%d",
                extracted, empty, failed, skipped_existing)
    return {"skipped": False, "extracted": extracted, "empty": empty,
            "failed": failed, "skipped_existing": skipped_existing,
            "total": len(pending), "errors": failed}


def _pending_pages_for_releases(store: SidecarStore, release_keys: list[str]) -> List[Tuple[str, int]]:
    """[(release_key, source_page_id)] for all source pages of the given releases."""
    if not release_keys:
        return []
    pairs: List[Tuple[str, int]] = []
    with store.connect() as conn:
        placeholders = ",".join("?" for _ in release_keys)
        rows = conn.execute(
            f"SELECT release_key, source_page_id FROM ai_genre_source_pages "
            f"WHERE release_key IN ({placeholders}) ORDER BY source_page_id",
            release_keys,
        ).fetchall()
    for row in rows:
        pairs.append((row[0], int(row[1])))
    return pairs


def _uncached_review_only_tags(store: SidecarStore, page_ids: List[int]) -> List[Tuple[str, str]]:
    """Distinct (raw_tag, normalized_tag) review_only tags not yet in the adjudication cache.

    Re-derives classification from the raw tag via the canonical classifier (the
    same classify_source_tag that classify_source_tags uses), so this depends
    only on ai_genre_source_tags (columns: source_tag_id, raw_tag, source_page_id).
    """
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    if not page_ids:
        return []
    seen: dict[str, str] = {}
    with store.connect() as conn:
        placeholders = ",".join("?" for _ in page_ids)
        rows = conn.execute(
            f"SELECT raw_tag FROM ai_genre_source_tags "
            f"WHERE source_page_id IN ({placeholders})",
            page_ids,
        ).fetchall()
    for (raw_tag,) in rows:
        c = classify_source_tag(raw_tag)
        norm = c.normalized_tag
        if (
            c.classification == "review_only"
            and norm
            and norm not in seen
            and store.lookup_cached_adjudication(norm) is None
        ):
            seen[norm] = raw_tag
    return [(raw, norm) for norm, raw in seen.items()]


def _enrich_materialization_action(
    *, force: bool, has_assignments: bool, stored_fp: Optional[str], current_fp: str
) -> str:
    """Decide whether to (re)materialize a release in the incremental enrich.

    Returns "materialize" | "adopt" | "skip":
      - force                       -> materialize (deliberate full re-derive)
      - no existing assignments     -> materialize (new music; new fusion applies)
      - assignments, no stored fp   -> adopt (first incremental run: keep the
                                       existing/surgical state, record baseline)
      - stored fp == current fp     -> skip (evidence unchanged)
      - stored fp != current fp     -> materialize (evidence genuinely changed)
    """
    if force or not has_assignments:
        return "materialize"
    if stored_fp is None:
        return "adopt"
    return "skip" if stored_fp == current_fp else "materialize"


def _release_evidence_fingerprint(store: Any, release: Any) -> str:
    """Stable hash of everything that feeds fuse_release_evidence for a release.

    Covers sidecar source tags + their classification/domain, accepted enriched
    genres, model-prior terms, and the release payload's existing file/MB/Discogs
    genres. Unchanged inputs -> unchanged hash -> enrich skips the release; any
    real evidence change flips the hash and triggers re-materialization through
    the current fusion policy.
    """
    rk = release.release_key
    terms = sorted(
        (
            str(r.get("source_type")),
            str(r.get("source_domain") or ""),
            str(r.get("term")),
            str(r.get("mapping_status") or ""),
            round(float(r.get("confidence") or 0.0), 4),
            str(r.get("classifier") or ""),
        )
        for r in store.hybrid_source_terms_for_release(rk)
    )
    enriched = sorted(
        str(r.get("genre")) for r in store.accepted_enriched_genres_for_release(rk)
    )
    priors = sorted(
        (
            str(r.get("normalized_term")),
            str(r.get("mapping_status") or ""),
            round(float(r.get("confidence") or 0.0), 4),
        )
        for r in store.latest_model_prior_terms_for_release(rk)
    )
    existing = {
        k: sorted(v)
        for k, v in sorted((getattr(release, "existing_genres_by_source", {}) or {}).items())
    }
    payload = json.dumps(
        {"terms": terms, "enriched": enriched, "priors": priors, "existing": existing},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def stage_enrich(ctx: Dict) -> Dict:
    """Adjudicate unknown tags (chunked Claude calls) and materialize graph genres.

    Writes only to the enrichment sidecar. De-duplicates unknown tags library-wide:
    a tag is sent to Claude once, cached, then reused across every release.
    Raises if the LLM backend fails (explicitly-requested work that cannot run is
    an error, not a silent skip).
    """
    args = ctx["args"]
    store = SidecarStore(str(ENRICHMENT_DB_PATH))
    store.initialize()

    limit = args.limit if args.limit and args.limit > 0 else None
    releases = discover_releases(ctx["db_path"], limit=limit)
    by_key = {r.release_key: r for r in releases}
    page_pairs = _pending_pages_for_releases(store, list(by_key.keys()))
    if not page_pairs:
        logger.info("Skipping enrich stage (no source pages to enrich)")
        return {"skipped": True, "reason": "no_source_pages", "releases_enriched": 0}

    page_ids = [pid for _, pid in page_pairs]
    pending_keys = sorted({rk for rk, _ in page_pairs})

    # Pre-pass: deterministic classification populates known tags + cache hits.
    for _, page_id in page_pairs:
        store.classify_source_tags(page_id, adjudicate=False, model=None)

    # Collect distinct uncached review_only tags across ALL pending pages, adjudicate
    # in chunks, and cache definitive results so per-release classification is cache-only.
    unknown = _uncached_review_only_tags(store, page_ids)
    model = getattr(args, "model", None) or resolve_enrichment_model(None)
    chunk_size = max(1, int(getattr(args, "enrich_chunk_size", 50)))
    injected_client = getattr(args, "enrich_client", None)
    tags_adjudicated = 0
    chunks_used = 0
    enriched = 0
    skipped = 0
    assignments = 0
    for i in range(0, len(unknown), chunk_size):
        chunk = unknown[i:i + chunk_size]
        try:
            results = adjudicate_tags(chunk, model=model, client=injected_client)
        except RuntimeError as exc:
            exc_str = str(exc)
            if any(pat in exc_str for pat in _TRANSIENT_CLAUDE_PATTERNS):
                logger.warning(
                    "enrich stage: transient Claude failure — pausing cleanly "
                    "(cache preserved, re-run will resume). Reason: %s", exc_str,
                )
                return {
                    "skipped": False,
                    "paused": True,
                    "pause_reason": exc_str,
                    "releases_enriched": enriched,
                    "releases_skipped_unchanged": skipped,
                    "tags_adjudicated": tags_adjudicated,
                    "chunks_used": chunks_used,
                    "genre_assignments": assignments,
                    "total": enriched,
                }
            raise
        chunks_used += 1
        for norm, decision in results.items():
            classification = decision.get("classification")
            if classification and classification != "review_only":
                store.cache_adjudication(
                    normalized_tag=norm,
                    classification=classification,
                    confidence=float(decision.get("confidence", 0.0)),
                    classifier="ai",
                )
                tags_adjudicated += 1

    # Per-release: re-classify (cache hits now), rebuild signatures, fuse, materialize.
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.hybrid_evidence import fuse_release_evidence

    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)

    pages_by_release: dict[str, List[int]] = {}
    for rk, pid in page_pairs:
        pages_by_release.setdefault(rk, []).append(pid)

    prog = (
        ProgressLogger(
            logger, total=len(pending_keys), label="enrich", unit="releases",
            interval_s=getattr(args, "progress_interval", 15.0),
            every_n=getattr(args, "progress_every", 500),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
        if getattr(args, "progress", True) else None
    )

    # Incremental: only (re)materialize NEW music or releases whose evidence
    # actually changed. Existing releases are left exactly as they are (the
    # surgical/published state) unless --force. This stops a routine analyze
    # run from wholesale re-deriving the library and undoing targeted genre
    # fixes (2026-06-13 incident). The current fusion policy still applies to
    # every release that IS materialized — new music gets the better rules.
    force = bool(getattr(args, "force", False))
    for rk in pending_keys:
        release = by_key.get(rk)
        if release is None:
            continue
        if prog:
            prog.update(detail=rk)
        for pid in pages_by_release.get(rk, []):
            store.classify_source_tags(pid, adjudicate=False, model=None)
        store.rebuild_enriched_genres_for_release(rk)
        current_fp = _release_evidence_fingerprint(store, release)
        action = _enrich_materialization_action(
            force=force,
            has_assignments=store.has_genre_assignments(rk),
            stored_fp=store.materialization_fingerprint(rk),
            current_fp=current_fp,
        )
        if action != "materialize":
            if action == "adopt":
                store.set_materialization_fingerprint(rk, current_fp)
            skipped += 1
            continue
        fused = fuse_release_evidence(store, release)
        summary = materialize_layered_assignments(
            store, release_id=rk, artist=release.normalized_artist,
            album=release.normalized_album, report=fused, taxonomy=taxonomy,
        )
        store.set_materialization_fingerprint(rk, current_fp)
        assignments += summary.genre_assignment_count
        enriched += 1

    if prog:
        prog.finish(detail=f"materialized {enriched:,} releases ({skipped:,} unchanged)")
    logger.info(
        "enrich stage: materialized=%d skipped_unchanged=%d tags_adjudicated=%d chunks=%d assignments=%d",
        enriched, skipped, tags_adjudicated, chunks_used, assignments,
    )
    return {"skipped": False, "releases_enriched": enriched,
            "releases_skipped_unchanged": skipped,
            "tags_adjudicated": tags_adjudicated, "chunks_used": chunks_used,
            "genre_assignments": assignments, "total": enriched}


def stage_adjudicate(ctx: Dict) -> Dict:
    """Album-grain Sonnet adjudication. One call per new album -> sidecar checkpoint.

    Scan-each-release-once: skips albums already complete under the current prompt_version
    (NOT keyed on input_hash — apply/publish drift the evidence and would otherwise re-run
    the whole backlog every cycle). A prompt-contract change bumps prompt_version and re-runs;
    `--force` re-adjudicates everything. Returns a `paused` result on the rate wall
    (resumable). Writes only the sidecar `adjudications` table.
    """
    args = ctx["args"]
    conn = sqlite3.connect(ctx["db_path"])
    store = AdjudicationStore(str(ENRICHMENT_DB_PATH))
    pv = effective_prompt_version(thorough=False)
    try:
        id2name = {r[0]: r[1] for r in conn.execute(
            "SELECT genre_id, name FROM genre_graph_canonical_genres")}
        album_ids = [r[0] for r in conn.execute("SELECT album_id FROM albums ORDER BY album_id")]
        limit = getattr(args, "limit", None)
        if limit and limit > 0:
            album_ids = album_ids[:limit]
        todo = build_todo(store, conn, id2name, album_ids, prompt_version=pv,
                          force=bool(getattr(args, "force", False)))
        if not todo:
            logger.info("Skipping adjudicate stage (no new/changed albums)")
            return {"skipped": True, "reason": "nothing_pending", "adjudicated": 0}
        model = getattr(args, "adjudicate_model", None) or "sonnet"
        client = getattr(args, "adjudicate_client", None) or ClaudeCodeEnrichmentClient(model=model)
        adapter = load_graph_adapter()
        summary = run_adjudication(
            store, todo, model=model, instructions=ADJUDICATOR_INSTRUCTIONS,
            prompt_version=pv, adapter=adapter, client=client)
    finally:
        conn.close()
        store.close()
    if summary.paused:
        return {"paused": True, "pause_reason": summary.pause_reason,
                "adjudicated": summary.adjudicated, "failed": summary.failed}
    return {"adjudicated": summary.adjudicated, "failed": summary.failed,
            "total": summary.adjudicated}


def stage_apply(ctx: Dict) -> Dict:
    """Deterministic apply of checkpointed adjudications: materialize non-escalated, queue escalated."""
    args = ctx["args"]
    std_pv = effective_prompt_version(thorough=False)
    tho_pv = effective_prompt_version(thorough=True)
    rows = []
    with contextlib.closing(sqlite3.connect(str(ENRICHMENT_DB_PATH))) as side:
        for album_id, pv, rj, ih in side.execute(
            "SELECT album_id, prompt_version, response_json, input_hash "
            "FROM adjudications WHERE status='complete'"
        ):
            resp = json.loads(rj) if rj else None
            if resp is None:
                continue
            resp["input_hash"] = ih
            rows.append((album_id, pv, resp))
    if not rows:
        logger.info("Skipping apply stage (no complete adjudications)")
        return {"skipped": True, "reason": "no_adjudications", "materialized": 0, "escalated": 0}
    conn = sqlite3.connect(ctx["db_path"])
    # SidecarStore has no close() — it opens a fresh connection per operation
    # (context-managed) and closes it automatically after each call.
    store = SidecarStore(str(ENRICHMENT_DB_PATH))
    store.initialize()
    queue = EscalationQueue(ENRICHMENT_DB_PATH)
    try:
        id2name = {r[0]: r[1] for r in conn.execute(
            "SELECT genre_id, name FROM genre_graph_canonical_genres")}
        taxonomy = load_default_layered_taxonomy()
        adapter = load_graph_adapter()
        summary = apply_adjudications(
            rows=rows, thorough_pv=tho_pv, std_pv=std_pv, meta_conn=conn, id2name=id2name,
            taxonomy=taxonomy, adapter=adapter, sidecar_store=store, queue=queue,
            model=getattr(args, "adjudicate_model", None) or "sonnet")
    finally:
        conn.close()
        queue.close()
    return {"materialized": summary.materialized, "escalated": summary.escalated,
            "total": summary.materialized}


def _release_effective_genres_exists(db_path: str) -> bool:
    """True if metadata.db already has the published release_effective_genres table."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='release_effective_genres'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def stage_publish(ctx: Dict) -> Dict:
    """Publish authoritative genres into metadata.db (release_effective_genres).

    First publish (table absent) takes a timestamped metadata.db backup; later
    runs write directly (publish is atomic + idempotent and scoped to derived
    genre tables — it never touches tracks/sonic/track_genres). Dry-run rolls back.
    """
    import shutil
    from src.genre.genre_publish import publish as publish_genres
    from scripts.validate_published_genres import validate as validate_published

    args = ctx["args"]
    db_path = ctx["db_path"]
    sidecar = str(ENRICHMENT_DB_PATH)
    if not ENRICHMENT_DB_PATH.exists():
        logger.info("Skipping publish stage (no enrichment sidecar at %s)", sidecar)
        return {"skipped": True, "reason": "no_sidecar"}

    dry_run = bool(getattr(args, "dry_run", False))
    backed_up = False
    if not dry_run and not _release_effective_genres_exists(db_path):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.bak.{ts}"
        shutil.copy2(db_path, backup_path)
        backed_up = True
        logger.info("First publish — backed up metadata.db to %s", backup_path)

    stats = publish_genres(db_path, sidecar, dry_run=dry_run)
    stats_dict = stats.as_dict()

    validation_ok = True
    if not dry_run:
        rc = validate_published(db_path)
        validation_ok = (rc == 0)
        if not validation_ok:
            logger.warning("publish stage: validation reported problems (rc=%d)", rc)

    logger.info(
        "publish stage: graph_albums=%d legacy_albums=%d total=%d backed_up=%s dry_run=%s",
        stats_dict.get("graph_albums", 0), stats_dict.get("legacy_albums", 0),
        stats_dict.get("total_albums", 0), backed_up, dry_run,
    )
    # SP4: when the artifact builder consumes the published graph genres
    # (genre_source=graph), a real publish dirties the genre inputs so the
    # artifacts stage rebuilds this run. Legacy artifacts read raw tables that
    # publish does not touch, so they deliberately stay clean.
    if not dry_run and _configured_genre_source(ctx) == "graph":
        ctx["genres_dirty"] = True
    return {"skipped": False, "backed_up": backed_up, "dry_run": dry_run,
            "validation_ok": validation_ok, "stats": stats_dict,
            "total": stats_dict.get("total_albums", 0),
            "errors": 0 if validation_ok else 1}


def _run_sonic_in_process(ctx: Dict, *, workers, force, limit) -> None:
    """Run the sonic pool in-process (fallback). The pool's watchdog in
    ``update_sonic.run`` still guarantees completion (serial fallback) even if
    the in-process pool stalls."""
    args = ctx["args"]
    pipeline = SonicFeaturePipeline(
        db_path=ctx["db_path"],
        use_beat_sync=False,
        use_beat3tower=True,
    )
    try:
        pipeline.run(
            limit=limit,
            workers=workers,
            force=force,
            progress=getattr(args, "progress", True),
            progress_interval=getattr(args, "progress_interval", 15.0),
            progress_every=getattr(args, "progress_every", 500),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
    finally:
        pipeline.close()


def _run_sonic_analysis(ctx: Dict, *, workers, force, limit) -> None:
    """Run the sonic pool in a separate, import-light process.

    The pool's children re-import scripts/run_sonic_pool.py (numpy-free at module
    scope), so the Windows spawn ``prepare()`` re-import can't deadlock loading
    numpy's C extension -- the root cause of the Analyze Library hang
    (project_analyze_pool_deadlock). Output is streamed to the analyze logger;
    cancellation propagates (kills the child). A non-zero exit falls back to
    in-process analysis (whose own watchdog still guarantees completion)."""
    args = ctx["args"]
    cancel = ctx.get("cancellation_check")
    entry = str(ROOT_DIR / "scripts" / "run_sonic_pool.py")
    argv = [sys.executable, entry, "--db-path", str(ctx["db_path"])]
    if workers is not None:
        argv += ["--workers", str(workers)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if force:
        argv += ["--force"]
    argv += [
        "--progress-interval", str(getattr(args, "progress_interval", 15.0)),
        "--progress-every", str(getattr(args, "progress_every", 500)),
    ]
    if bool(getattr(args, "verbose", False)):
        argv += ["--verbose"]

    # A raised exception here is cancellation -> propagate (do NOT fall back).
    rc = run_streaming_subprocess(
        argv,
        on_line=lambda line: logger.info(line) if line else None,
        cancellation_check=cancel,
    )
    if rc != 0:
        logger.error(
            "Sonic subprocess exited with code %s; falling back to in-process analysis.",
            rc,
        )
        _run_sonic_in_process(ctx, workers=workers, force=force, limit=limit)


def stage_sonic(ctx: Dict) -> Dict:
    args = ctx["args"]
    force = args.force
    limit = args.limit
    workers_arg = args.workers
    if isinstance(workers_arg, str) and workers_arg.lower() == "auto":
        workers = None
    else:
        workers = int(workers_arg)
    # Short-lived read-only pre-check; release the connection before analysis runs.
    pre = SonicFeaturePipeline(
        db_path=ctx["db_path"],
        use_beat_sync=False,
        use_beat3tower=True,
    )
    try:
        try:
            pending = pre.get_pending_tracks(limit, force=force)
        except sqlite3.OperationalError as exc:
            logger.warning("Skipping sonic stage; schema missing required columns (%s)", exc)
            return {"pending": None, "skipped": True, "reason": "schema_missing"}
        if not force and len(pending) == 0:
            logger.info("Skipping sonic stage (no pending tracks; use --force to re-run)")
            return {"pending": 0, "skipped": True}
        pending_count = len(pending)
    finally:
        pre.close()
    mode = "beat3tower"
    logger.info(f"Running sonic stage in {mode} mode")
    start_ts = int(time.time())
    _run_sonic_analysis(ctx, workers=workers, force=force, limit=limit)
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
        updated = pending_count
    return {"pending": pending_count, "skipped": False, "mode": mode, "updated": updated}


def stage_genre_sim(ctx: Dict) -> Dict:
    out_dir = ctx["out_dir"]
    out_path = out_dir / "genre_similarity_matrix.npz"

    from src.config_loader import Config as _Config
    from src.genre.graph_similarity import npz_similarity_source
    sim_source = _Config(ctx["config_path"]).get_ds_genre_similarity_source()
    existing_source = npz_similarity_source(out_path)
    source_mismatch = existing_source is not None and existing_source != sim_source

    force_rebuild = (
        ctx["args"].force
        or bool(ctx.get("genres_dirty"))
        or bool(ctx.get("force_stage"))
        or source_mismatch
    )
    if out_path.exists() and not force_rebuild:
        logger.info("Skipping genre-sim stage (exists: %s; use --force to rebuild)", out_path)
        return {"path": str(out_path), "skipped": True}
    if source_mismatch:
        logger.info(
            "Rebuilding genre-sim (existing matrix source=%s, config wants %s)",
            existing_source, sim_source,
        )
    elif out_path.exists() and force_rebuild and not ctx["args"].force:
        logger.info("Rebuilding genre-sim (new genres detected since last build)")

    if sim_source == "graph":
        try:
            from src.genre.graph_adapter import load_graph_adapter
            from src.genre.graph_similarity import build_graph_similarity, save_graph_similarity_npz
            graph_result = build_graph_similarity(load_graph_adapter())
            save_graph_similarity_npz(graph_result, out_path)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping genre-sim stage (graph source failed): %s", exc)
            return {"path": str(out_path), "skipped": True, "reason": str(exc)}
        return {"path": str(out_path), "skipped": False, "stats": graph_result.stats}

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
    # Unified on the same beat3tower builder the GUI "Build Artifacts" button uses.
    # genre_source=None defers to playlists.ds_pipeline.genre_source in config
    # (resolved inside build_artifacts) so the analyze pipeline and the GUI button
    # honor the same knob. A hardcoded "legacy" here previously made the config
    # key a silent no-op for analyze runs.
    from argparse import Namespace
    from scripts.build_beat3tower_artifacts import build_artifacts as build_beat3tower_artifacts
    args_ns = Namespace(
        db_path=ctx["db_path"],
        config=ctx["config_path"],
        output=str(out_path),
        genre_sim_path=str(genre_sim_use) if genre_sim_use else None,
        max_tracks=ctx["args"].max_tracks or ctx["args"].limit or 0,
        no_pca=False,
        pca_variance=0.95,
        clip_sigma=3.0,
        random_seed=42,
        no_genre_normalization=False,
        sidecar_db=str(ENRICHMENT_DB_PATH),
        genre_source=None,
        verbose=bool(getattr(ctx["args"], "verbose", False)),
    )
    try:
        build_beat3tower_artifacts(args_ns)
    except RuntimeError as exc:
        logger.warning("Skipping artifacts stage: %s", exc)
        return {"path": str(out_path), "skipped": True, "reason": str(exc)}

    # If the 2DFTM harmony sidecar exists, fold it into the freshly built artifact.
    # New tracks without sidecar entries get zero harmony vectors (gracefully degraded
    # until the user runs extract_harmony_2dftm_sidecar.py for the new tracks).
    _sidecar = out_dir / "harmony_2dftm_sidecar.npz"
    if _sidecar.exists():
        logger.info("2DFTM sidecar found; folding harmony into rebuilt artifact...")
        try:
            from scripts.fold_2dftm_into_artifact import fold_harmony
            # Pass the message as a plain arg (%s), not as a logging format string,
            # so any literal % in a fold message can't raise or mangle output.
            fold_harmony(
                out_path, _sidecar, no_backup=True,
                log_fn=lambda msg="", **kw: logger.info("%s", msg),
            )
            logger.info("2DFTM harmony fold complete")
        except Exception as exc:
            logger.warning("2DFTM fold failed (artifact left as-is): %s", exc)

    # Fold the MERT sidecar back in. A fresh build (and the 2DFTM fold above) leave
    # X_sonic_variant on the tower blend; without this, every artifacts rebuild
    # silently reverts the production sonic space from the learned MERT embedding to
    # the tower rollback. This is the auto-fold that replaces the manual
    # `fold_mert_into_artifact.py` step the old pipeline relied on.
    fold_enabled, active_variant = _mert_fold_settings(ctx["config_path"])
    mert_sidecar = out_dir / "mert_sidecar.npz"
    if fold_enabled and mert_sidecar.exists():
        logger.info("Folding MERT sidecar into rebuilt artifact (X_sonic_variant=%s)...", active_variant)
        try:
            from scripts.fold_mert_into_artifact import fold_mert
            fold_mert(
                out_path, mert_sidecar, set_active=active_variant, no_backup=True,
                log_fn=lambda msg="", **kw: logger.info("%s", msg),
            )
            logger.info("MERT fold complete; X_sonic_variant=%s", active_variant)
        except Exception as exc:
            # Loud: a failed fold leaves generation on the tower rollback space.
            logger.error(
                "MERT fold FAILED after artifact rebuild — generation will use the WRONG "
                "sonic space (tower rollback) until re-folded: %s", exc
            )
    elif fold_enabled:
        logger.warning(
            "MERT sidecar not found (%s); artifact left on the tower variant. "
            "Run the mert stage to build the sidecar.", mert_sidecar
        )

    fingerprint = compute_stage_fingerprint(ctx, "artifacts")
    manifest_path = _write_artifact_manifest(out_dir, fingerprint, ctx.get("config_hash", ""), {})
    # Signal the genre-embedding stage that the artifact changed and its dense
    # sidecar must be rebuilt (otherwise the dense genre vectors go stale).
    ctx["artifacts_dirty"] = True
    return {
        "path": str(out_path),
        "skipped": False,
        "fingerprint": fingerprint,
        "manifest": str(manifest_path),
    }


def stage_genre_embedding(ctx: Dict) -> Dict:
    """Rebuild the dense PMI-SVD genre embedding sidecar (the 'new genre system').

    Corpus-only (skip_prior=True) — no API calls. Reruns whenever the main
    artifact was rebuilt this run (artifacts_dirty) or its mtime changed.
    """
    out_dir = ctx["out_dir"]
    artifact_path = out_dir / "data_matrices_step1.npz"
    sidecar = artifact_path.parent / f"{artifact_path.stem}_genre_emb_dim64.npz"
    if not artifact_path.exists():
        logger.warning("Skipping genre-embedding stage: artifact not found at %s", artifact_path)
        return {"skipped": True, "reason": "artifact_missing"}
    force_rebuild = (
        ctx["args"].force
        or bool(ctx.get("artifacts_dirty"))
        or bool(ctx.get("force_stage"))
    )
    if sidecar.exists() and not force_rebuild:
        logger.info("Skipping genre-embedding stage (exists: %s; use --force to rebuild)", sidecar)
        return {"path": str(sidecar), "skipped": True}
    from scripts.build_genre_embedding import build_genre_embedding_sidecar
    try:
        out = build_genre_embedding_sidecar(artifact_path, skip_prior=True)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Skipping genre-embedding stage: %s", exc)
        return {"path": str(sidecar), "skipped": True, "reason": str(exc)}
    return {"path": str(out), "skipped": False}


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
    # Dense genre embedding sidecar (the "new genre system") must exist and be
    # aligned to the exact sparse genre inputs used to build it.
    sidecar_path = artifact_path.parent / f"{artifact_path.stem}_genre_emb_dim64.npz"
    if not sidecar_path.exists():
        issues.append("genre_embedding_missing")
    else:
        try:
            reason = dense_sidecar_mismatch_reason_from_paths(
                artifact_path=artifact_path,
                sidecar_path=sidecar_path,
            )
            if reason is not None:
                logger.warning("Verify: genre embedding sidecar %s", reason)
                issue_by_reason = {
                    "track_ids mismatch": "genre_embedding_track_mismatch",
                    "vocabulary mismatch": "genre_embedding_vocab_mismatch",
                    "schema version mismatch": "genre_embedding_schema_mismatch",
                    "sparse genre identity mismatch": "genre_embedding_sparse_identity_mismatch",
                }
                issues.append(issue_by_reason.get(reason, "genre_embedding_invalid"))
        except Exception as exc:
            logger.warning("Verify: could not read genre embedding sidecar: %s", exc)
            issues.append("genre_embedding_unreadable")
    # Sonic-variant guard: when the MERT sidecar exists and folding is enabled, the
    # active sonic variant MUST be the learned embedding. A mismatch means an
    # artifacts rebuild clobbered the variant and the fold did not restore it —
    # generation would silently run on the tower rollback space. Fail loudly.
    try:
        fold_enabled, active_variant = _mert_fold_settings(ctx["config_path"])
        mert_sidecar = artifact_path.parent / "mert_sidecar.npz"
        if fold_enabled and mert_sidecar.exists():
            with np.load(artifact_path, allow_pickle=True) as _z:
                current_variant = (
                    str(_z["X_sonic_variant"]) if "X_sonic_variant" in _z else ""
                )
            if current_variant != active_variant:
                logger.error(
                    "Verify: X_sonic_variant=%r but expected %r — MERT fold missing or "
                    "clobbered; generation would use the wrong sonic space.",
                    current_variant, active_variant,
                )
                issues.append("sonic_variant_mismatch")
    except Exception as exc:
        logger.warning("Verify: could not check X_sonic_variant: %s", exc)
    return {
        "skipped": False,
        "tracks": int(bundle.track_ids.size),
        "genres": int(bundle.genre_vocab.size),
        "issues": issues,
        "artifact_fingerprint": current_fp,
    }


def _build_mert_embedder(device: str, torch_threads: int):
    """Build and return the real MERT embedder.

    Isolated so tests can monkeypatch ``al._build_mert_embedder`` without
    loading torch/transformers.  Sets torch CPU thread count when requested.
    """
    if torch_threads > 0:
        import torch
        torch.set_num_threads(torch_threads)
    from scripts.extract_mert_sidecar import build_real_embedder
    return build_real_embedder(device)


def stage_mert(ctx: Dict) -> Dict:
    """Extract MERT sonic embeddings into resumable shards and a merged sidecar npz.

    Pending = artifact track_ids minus whatever the shard manifest already marks
    done-or-failed.  Returns immediately (skipped=True) when pending==0 and
    force is False so the real embedder is never loaded in that case.
    """
    import yaml
    from scripts.extract_mert_sidecar import (
        ShardStore,
        load_paths,
        merge_shards,
        MODEL_NAME,
        DEFAULT_REVISION,
        EMB_DIM,
        run_extraction,
    )

    args = ctx["args"]
    force: bool = bool(args.force)
    limit: Optional[int] = args.limit if args.limit else None
    out_dir: Path = Path(ctx["out_dir"])

    # Read MERT config block (device / torch_threads / shard_size).
    device = "cpu"
    torch_threads = 0
    shard_size = 500
    try:
        with open(ctx["config_path"], "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        _mert_cfg = (_cfg.get("analyze") or {}).get("mert") or {}
        device = str(_mert_cfg.get("device", "cpu"))
        torch_threads = int(_mert_cfg.get("torch_threads", 0))
        shard_size = int(_mert_cfg.get("shard_size", 500))
    except Exception:
        pass

    # Determine the shard directory and merged-sidecar path.
    shard_dir = out_dir / "mert_shards"
    sidecar_path = out_dir / "mert_sidecar.npz"

    # Universe = every track with a file_path in the DB (read-only). Keyed off the
    # DB rather than the existing artifact so newly-scanned tracks are embedded in
    # THIS run, before the artifact is rebuilt — otherwise a fresh file would not
    # enter the MERT sidecar until a second pass (it is not yet in the stale
    # artifact's track_ids). stage_artifacts folds the sidecar into the rebuilt
    # artifact afterwards, aligning to the canonical track order.
    db_paths: Dict[str, str] = {}
    try:
        db_paths = load_paths(ctx["db_path"])
    except Exception as exc:
        logger.warning("stage_mert: cannot load file paths from db: %s", exc)
        return {"skipped": True, "pending": 0, "reason": str(exc)}
    universe_ids: List[str] = list(db_paths.keys())

    # Build the skip set from the existing manifest (if any).
    skip_ids: set = set()
    if not force and shard_dir.exists() and (shard_dir / "manifest.json").exists():
        try:
            store_probe = ShardStore(
                shard_dir,
                model_name=MODEL_NAME,
                model_revision=DEFAULT_REVISION,
                emb_dim=EMB_DIM,
                shard_size=shard_size,
            )
            skip_ids = store_probe.skip_ids()
        except Exception as exc:
            logger.warning("stage_mert: manifest read failed, treating as empty: %s", exc)

    # Force re-extracts the whole DB universe; otherwise only what isn't done yet.
    pending_ids = list(universe_ids) if force else [t for t in universe_ids if t not in skip_ids]
    if limit is not None:
        pending_ids = pending_ids[:limit]

    if not pending_ids:
        logger.info("stage_mert: nothing pending (manifest complete); skipping")
        return {"skipped": True, "pending": 0}

    logger.info("stage_mert: %d track(s) pending (device=%s)", len(pending_ids), device)

    items: List[Tuple[str, Optional[str]]] = [
        (tid, db_paths.get(tid)) for tid in pending_ids
    ]

    embedder = _build_mert_embedder(device, torch_threads)
    store = ShardStore(
        shard_dir,
        model_name=MODEL_NAME,
        model_revision=DEFAULT_REVISION,
        emb_dim=EMB_DIM,
        shard_size=shard_size,
    )

    result = run_extraction(items, embedder, store)
    n_ok: int = result["ok"]
    n_fail: int = result["failed"]

    # Merge all shards into the sidecar npz.
    try:
        merge_shards(shard_dir, sidecar_path)
    except Exception as exc:
        logger.warning("stage_mert: merge_shards failed: %s", exc)
        sidecar_path = None  # type: ignore[assignment]

    return {
        "skipped": False,
        "pending": len(pending_ids),
        "ok": n_ok,
        "failed": n_fail,
        "sidecar": str(sidecar_path) if sidecar_path else None,
    }


def _energy_pending(out_dir):
    from src.analyze.energy_runner import pending_energy
    return pending_energy(out_dir)


def _energy_preflight(cfg):
    from src.analyze.energy_runner import preflight_wsl
    preflight_wsl(cfg)


def _energy_run(cfg, *, force, cancellation_check):
    from src.analyze.energy_runner import run_energy_scan
    return run_energy_scan(
        cfg, repo_root=ROOT_DIR, force=force, logger=logger,
        cancellation_check=cancellation_check,
    )


def _checkpoint_metadata_for_wsl(db_path) -> None:
    """Fold metadata.db's WAL into the main file before the WSL energy extractor reads it.

    The extractor opens metadata.db as an immutable snapshot over /mnt/c; WAL's -shm
    index can't be coordinated across the Windows<->WSL boundary, so data still only in
    the WAL would be invisible (and mode=ro raised "disk I/O error"). Checkpointing here
    makes the main file complete and current for that snapshot. Best-effort: even a busy
    checkpoint flushes committed frames, which is all the immutable read needs.
    """
    if not db_path:
        return
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        logger.info("stage_energy: checkpointed metadata.db (WAL flushed for WSL immutable read)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("stage_energy: metadata.db checkpoint failed (continuing): %s", exc)


def stage_energy(ctx: Dict) -> Dict:
    """Run the WSL-only Essentia energy scan into <artifact>/energy/energy_sidecar.npz.

    Default-on; skip-fast when up-to-date (like MERT). Hard-fail (RuntimeError)
    if WSL/Essentia is unreachable. Standalone pace-axis sidecar — never folded
    into the sonic blend, never writes metadata.db.
    """
    from src.analyze.energy_runner import load_energy_config, energy_paths

    args = ctx["args"]
    out_dir = Path(ctx["out_dir"])
    cfg = load_energy_config(ctx["config_path"])
    energy_workers = getattr(args, "energy_workers", None)
    if energy_workers is not None:
        cfg.workers = int(energy_workers)

    artifact_npz, _ckpt, _sidecar = energy_paths(out_dir)
    if not artifact_npz.exists():
        logger.info("stage_energy: artifact missing; skipping (build artifacts first)")
        return {"skipped": True, "pending": 0, "reason": "no_artifact"}

    pending, total = _energy_pending(out_dir)
    if pending == 0 and not args.force:
        logger.info("stage_energy: nothing pending (sidecar complete); skipping")
        return {"skipped": True, "pending": 0}

    logger.info("stage_energy: %d/%d track(s) pending (workers=%d, distro=%s)",
                pending, total, cfg.workers, cfg.distro)
    _energy_preflight(cfg)  # raises RuntimeError if WSL/venv/models missing
    # Flush metadata.db's WAL so the WSL extractor's immutable snapshot read is complete.
    _checkpoint_metadata_for_wsl(ctx.get("db_path"))
    res = _energy_run(cfg, force=bool(args.force),
                      cancellation_check=ctx.get("cancellation_check"))
    return {"skipped": False, "pending": pending, **res}


STAGE_FUNCS = {
    "scan": stage_scan,
    "mbid": stage_mbid,
    "genres": stage_genres,
    "discogs": stage_discogs,
    "lastfm": stage_lastfm,
    "enrich": stage_enrich,
    "adjudicate": stage_adjudicate,
    "apply": stage_apply,
    "publish": stage_publish,
    "sonic": stage_sonic,
    "mert": stage_mert,
    "genre-sim": stage_genre_sim,
    "artifacts": stage_artifacts,
    "energy": stage_energy,
    "genre-embedding": stage_genre_embedding,
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
    parser.add_argument(
        "--energy-workers",
        type=int,
        default=None,
        help="Workers for the WSL energy stage (overrides analyze.energy.workers)",
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
    parser.add_argument("--lastfm-api-key", default=None,
                        help="Last.fm API key for the lastfm stage (else env LASTFM_API_KEY / config)")
    parser.add_argument("--model", default=None,
                        help="LLM model override for the enrich stage (default: provider default)")
    parser.add_argument("--adjudicate-model", default=None,
                        help="Model for the adjudicate stage (default: sonnet)")
    parser.add_argument("--enrich-chunk-size", type=int, default=50,
                        help="Tags per adjudication chunk in the enrich stage (default: 50)")
    add_logging_args(parser)
    return parser.parse_args(argv)


def run_pipeline(
    args: argparse.Namespace,
    cancellation_check: Optional[Callable[[], None]] = None,
    console_logging: bool = True,
) -> int:
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
    configure_logging(
        level=log_level,
        log_file=log_file,
        run_id=run_id,
        show_run_id=getattr(args, "show_run_id", False),
        console=console_logging,
        force=not console_logging,
    )

    # Re-get logger after configuration
    logger = logging.getLogger("analyze_library")

    if args.beat_sync:
        logger.error("Legacy sonic mode (--beat-sync) is deprecated and disabled. Beat3tower is always used.")
        return 2

    cfg = Config(args.config)
    db_path = args.db_path or cfg.library_database_path
    # Hard stop before opening anything: a symlinked/aliased DB means a worktree run
    # that would WAL-corrupt the DB (the 2026-06-22 incident). Fail loudly instead.
    _assert_dbs_not_aliased(db_path, str(ENRICHMENT_DB_PATH))
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
        "artifacts_dirty": False,
        "config_hash": config_hash,
        "library_root": cfg.library_music_directory,
    }

    def _check_cancelled() -> None:
        if cancellation_check is not None:
            cancellation_check()

    ctx["cancellation_check"] = _check_cancelled

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
    try:
        for stage in stages_requested:
            _check_cancelled()
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
                _check_cancelled()
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
            _check_cancelled()
            duration = time.time() - stage_start
            if isinstance(result, dict) and result.get("paused"):
                report["paused"] = True
                report["paused_stage"] = stage
                report["stages"][stage] = {
                    "decision": "paused",
                    "pause_reason": result.get("pause_reason"),
                    "duration_sec": duration,
                    "fingerprint_before": fingerprint_before,
                    "pending_estimate": pending_count,
                    "pending_label": pending_label,
                }
                logger.warning(
                    "run_id=%s | stage=%s | decision=paused | reason=%s",
                    run_id, stage, result.get("pause_reason"),
                )
                break
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
    except Exception:
        conn.close()
        raise

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
