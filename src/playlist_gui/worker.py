#!/usr/bin/env python3
"""
GUI Worker Process - NDJSON Protocol Handler (v1)

This worker runs as a separate process, receiving commands via stdin and
emitting events via stdout using newline-delimited JSON (NDJSON).

Protocol Version: 1
  - All commands include request_id for correlation
  - All events include request_id to match the originating command
  - Cancel command for cooperative cancellation

Commands (GUI -> Worker, one JSON per line):
  {"cmd":"ping","request_id":"<uuid>","protocol_version":1}
  {"cmd":"generate_playlist","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{...},"args":{...}}
  {"cmd":"cancel","request_id":"<uuid-to-cancel>"}
  {"cmd":"scan_library","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"update_genres","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"update_sonic","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"build_artifacts","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}

Events (Worker -> GUI, one JSON per line):
  {"type":"log","request_id":"<uuid>","level":"INFO","msg":"..."}
  {"type":"progress","request_id":"<uuid>","stage":"...","current":...,"total":...}
  {"type":"result","request_id":"<uuid>","result_type":"playlist","playlist":{...}}
  {"type":"error","request_id":"<uuid>","message":"...","traceback":"..."}
  {"type":"done","request_id":"<uuid>","cmd":"...","ok":true}
  {"type":"done","request_id":"<uuid>","cmd":"...","ok":false,"cancelled":true}

Security:
  - Never emit secrets (API keys, tokens) in events
  - Redact secret values from error tracebacks
"""
import json
import logging
import os
import sys
import sqlite3
import threading
import traceback
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .utils.redaction import redact_text

try:
    from ..cancellation import CancellationToken, CancelledException
except ImportError:
    try:
        from src.cancellation import CancellationToken, CancelledException
    except ImportError:
        CancellationToken = None

        class CancelledException(Exception):
            """Fallback cancellation type; never aliases broad Exception."""

            pass


# Protocol version
PROTOCOL_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Worker State Management
# ─────────────────────────────────────────────────────────────────────────────

class CancellationError(Exception):
    """Raised when a command is cancelled."""
    pass


@dataclass
class WorkerState:
    """
    Manages worker state for request tracking and cancellation.

    Thread-safe for cancel requests that may arrive during command execution.
    """
    current_request_id: Optional[str] = None
    current_job_id: Optional[str] = None
    current_cmd: Optional[str] = None
    cancel_requested: bool = False
    _lock: threading.Lock = None

    def __post_init__(self):
        self._lock = threading.Lock()

    def start_request(self, request_id: str, cmd: str, job_id: Optional[str]) -> None:
        """Mark a new request as active."""
        with self._lock:
            self.current_request_id = request_id
            self.current_cmd = cmd
            self.current_job_id = job_id
            self.cancel_requested = False

    def end_request(self) -> None:
        """Clear the active request."""
        with self._lock:
            self.current_request_id = None
            self.current_cmd = None
            self.current_job_id = None
            self.cancel_requested = False

    def request_cancel(self, request_id: str) -> bool:
        """
        Request cancellation for a specific request_id.

        Returns True if the request_id matches the active request.
        """
        with self._lock:
            if self.current_request_id == request_id:
                self.cancel_requested = True
                return True
            return False

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        with self._lock:
            return self.cancel_requested

    def check_cancelled(self) -> None:
        """
        Check if cancelled and raise CancellationError if so.

        Call this at stage boundaries in command handlers.
        """
        if self.is_cancelled():
            raise CancellationError("Operation cancelled by user")

    def get_request_id(self) -> Optional[str]:
        """Get the current request_id."""
        with self._lock:
            return self.current_request_id

    def get_job_id(self) -> Optional[str]:
        """Get the current job_id, if any."""
        with self._lock:
            return self.current_job_id


# Global worker state
_worker_state = WorkerState()

# Global cancellation token for current operation
_current_cancellation_token: Optional[CancellationToken] = None
_token_lock = threading.Lock()


def set_cancellation_token(token: Optional[CancellationToken]) -> None:
    """Set the cancellation token for the current operation."""
    global _current_cancellation_token
    with _token_lock:
        _current_cancellation_token = token


def get_cancellation_token() -> Optional[CancellationToken]:
    """Get the current cancellation token."""
    with _token_lock:
        return _current_cancellation_token


def check_cancelled() -> None:
    """Convenience function to check cancellation at stage boundaries."""
    _worker_state.check_cancelled()


# ─────────────────────────────────────────────────────────────────────────────
# Secret Redaction
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Event Emission
# ─────────────────────────────────────────────────────────────────────────────

def emit_event(event: Dict[str, Any]) -> None:
    """
    Emit an event to stdout as NDJSON (one JSON object per line).

    Automatically includes request_id from current worker state.
    """
    # Include request_id if we have an active request
    request_id = _worker_state.get_request_id()
    if request_id and "request_id" not in event:
        event["request_id"] = request_id

    # Include job_id if available
    job_id = _worker_state.get_job_id()
    if job_id and "job_id" not in event:
        event["job_id"] = job_id

    try:
        line = json.dumps(event, default=str)
        print(line, flush=True)
    except Exception as e:
        # Fallback for serialization errors
        fallback = {"type": "error", "message": f"Serialization error: {e}"}
        if request_id:
            fallback["request_id"] = request_id
        print(json.dumps(fallback), flush=True)


def emit_log(level: str, msg: str, request_id: Optional[str] = None) -> None:
    """Emit a log event."""
    event = {"type": "log", "level": level, "msg": redact_text(msg)}
    if request_id:
        event["request_id"] = request_id
    emit_event(event)


def emit_progress(stage: str, current: int, total: int, detail: Optional[str] = None) -> None:
    """Emit a progress event."""
    event = {"type": "progress", "stage": stage, "current": current, "total": total}
    if detail:
        event["detail"] = detail
    emit_event(event)


def emit_error(message: str, tb: Optional[str] = None) -> None:
    """Emit an error event with redacted traceback."""
    event = {"type": "error", "message": redact_text(message)}
    if tb:
        event["traceback"] = redact_text(tb)
        # Also emit full traceback to the log stream so GUI output isn't truncated
        emit_log("DEBUG", f"Traceback detail:\n{tb}")
    emit_event(event)


def emit_result(result_type: str, data: Dict[str, Any]) -> None:
    """Emit a result event."""
    emit_event({"type": "result", "result_type": result_type, **data})


# Backwards-compatible alias expected by legacy tests
def redact_secrets_in_text(text: str) -> str:
    return redact_text(text)


def emit_done(
    cmd: str,
    ok: bool,
    detail: Optional[str] = None,
    cancelled: bool = False,
    summary: Optional[str] = None,
) -> None:
    """Emit a done event indicating command completion."""
    event = {"type": "done", "cmd": cmd, "ok": ok}
    if detail:
        event["detail"] = detail
    if summary:
        event["summary"] = summary
    if cancelled:
        event["cancelled"] = True
    emit_event(event)


def emit_checkpoint(stage: str, items_completed: int, resumable_state: Optional[Dict[str, Any]] = None) -> None:
    """
    Emit a checkpoint event for resumable operations.

    Args:
        stage: Stage name (e.g., 'scan', 'genres')
        items_completed: Number of items completed so far
        resumable_state: Optional state data for resumption (e.g., last file processed)
    """
    event = {
        "type": "checkpoint",
        "stage": stage,
        "items_completed": items_completed,
    }
    if resumable_state:
        event["resumable_state"] = resumable_state
    emit_event(event)


def emit_verbose_log(level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Emit a verbose log event (only sent when verbose mode is enabled).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        message: Log message
        context: Optional context data (file path, item name, etc.)
    """
    import datetime
    event = {
        "type": "verbose_log",
        "level": level,
        "message": redact_text(message),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if context:
        event["context"] = context
    emit_event(event)


def emit_performance(stage: str, elapsed_seconds: float, items_processed: int, throughput: Optional[float] = None) -> None:
    """
    Emit a performance metrics event.

    Args:
        stage: Stage name (e.g., 'scan', 'genres')
        elapsed_seconds: Time elapsed for this stage
        items_processed: Number of items processed
        throughput: Optional throughput (items/sec), calculated if not provided
    """
    if throughput is None and elapsed_seconds > 0:
        throughput = items_processed / elapsed_seconds

    event = {
        "type": "performance",
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "items_processed": items_processed,
        "throughput": round(throughput, 2) if throughput else 0.0,
    }
    emit_event(event)


# ─────────────────────────────────────────────────────────────────────────────
# Logging Integration
# ─────────────────────────────────────────────────────────────────────────────

class WorkerLogHandler(logging.Handler):
    """Logging handler that emits logs as NDJSON events."""

    def emit(self, record):
        try:
            msg = self.format(record)
            emit_log(record.levelname, msg)
        except Exception:
            pass  # Avoid recursion if logging fails


def setup_worker_logging():
    """Configure logging to emit via NDJSON."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Add our NDJSON handler
    handler = WorkerLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)


# ─────────────────────────────────────────────────────────────────────────────
# Config Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config_with_overrides(base_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Load base config and merge with overrides."""
    if not Path(base_path).exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Deep merge overrides
    def deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(config, overrides)

    # Apply mode presets (genre_mode/sonic_mode) to resolve weights and thresholds
    _apply_mode_presets(merged)

    return merged


def _apply_mode_presets(config: Dict[str, Any]) -> None:
    """
    Apply genre_mode and sonic_mode presets if specified in config.
    Modifies config in-place.
    """
    playlists_cfg = config.get('playlists', {})
    if not playlists_cfg:
        return
    try:
        from src.playlist.mode_presets import apply_mode_presets
    except ImportError:
        return

    apply_mode_presets(playlists_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _build_seed_similarity_components(
    *,
    tracks: list[dict[str, Any]],
    ds_report: dict[str, Any],
    edge_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Optional[float]]]]:
    """
    Build T/S1/S2 similarity components for sonic + genre.

    T: similarity to immediately preceding track (transition edge score)
    S1: similarity to the most recent seed track before this track
    S2: similarity to the next seed track after this track
    """
    if not ds_report:
        return {}

    artifact_path = ds_report.get("artifact_path")
    seed_track_id = ds_report.get("seed_track_id")
    anchor_seed_ids = ds_report.get("anchor_seed_ids") or []
    if not artifact_path or not seed_track_id:
        return {}

    track_ids: list[str] = []
    for track in tracks:
        tid = track.get("rating_key") or track.get("id") or track.get("track_id")
        track_ids.append(str(tid) if tid is not None else "")

    if not track_ids:
        return {}

    seed_ids = [str(seed_track_id)] + [str(sid) for sid in anchor_seed_ids if sid]
    seed_set = set(seed_ids)

    try:
        from src.features.artifacts import load_artifact_bundle
        from src.similarity.sonic_variant import compute_sonic_variant_norm

        bundle = load_artifact_bundle(artifact_path)
    except Exception:
        return {}

    track_idx_by_id = bundle.track_id_to_index
    track_indices = [track_idx_by_id.get(tid) for tid in track_ids]
    seed_positions = [
        idx for idx, tid in enumerate(track_ids)
        if tid in seed_set and track_indices[idx] is not None
    ]
    if not seed_positions:
        return {}

    sonic_variant = ds_report.get("sonic_variant") or "raw"
    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    if X_sonic is None:
        return {}

    X_sonic_norm, _ = compute_sonic_variant_norm(X_sonic, sonic_variant)
    X_genre_norm = None
    if X_genre is not None:
        denom = np.linalg.norm(X_genre, axis=1, keepdims=True) + 1e-12
        X_genre_norm = X_genre / denom

    def _dot(mat: Optional[np.ndarray], a: Optional[int], b: Optional[int]) -> Optional[float]:
        if mat is None or a is None or b is None:
            return None
        return float(np.dot(mat[a], mat[b]))

    components: dict[str, dict[str, dict[str, Optional[float]]]] = {}
    for idx, track_id in enumerate(track_ids):
        if not track_id:
            continue
        cur_idx = track_indices[idx]
        if cur_idx is None:
            continue

        prev_seed_pos_idx = bisect_right(seed_positions, idx) - 1
        prev_seed_pos = seed_positions[prev_seed_pos_idx] if prev_seed_pos_idx >= 0 else None
        next_seed_pos_idx = bisect_right(seed_positions, idx)
        next_seed_pos = seed_positions[next_seed_pos_idx] if next_seed_pos_idx < len(seed_positions) else None

        prev_seed_idx = track_indices[prev_seed_pos] if prev_seed_pos is not None else None
        next_seed_idx = track_indices[next_seed_pos] if next_seed_pos is not None else None

        s1_sonic = _dot(X_sonic_norm, cur_idx, prev_seed_idx)
        s2_sonic = _dot(X_sonic_norm, cur_idx, next_seed_idx)
        s1_genre = _dot(X_genre_norm, cur_idx, prev_seed_idx)
        s2_genre = _dot(X_genre_norm, cur_idx, next_seed_idx)

        if prev_seed_pos is not None and prev_seed_pos == idx:
            s1_sonic = 1.0
            if X_genre_norm is not None:
                s1_genre = 1.0

        edge = edge_map.get(track_id, {})
        t_sonic = edge.get("S")
        t_genre = edge.get("G")

        components[track_id] = {
            "sonic": {"t": t_sonic, "s1": s1_sonic, "s2": s2_sonic},
            "genre": {"t": t_genre, "s1": s1_genre, "s2": s2_genre},
        }

    return components
def handle_doctor(cmd_data: Dict[str, Any]) -> None:
    """Quick diagnostics."""
    checks = []
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    cfg_path = Path(base_path)

    if cfg_path.exists():
        checks.append({"name": "config_path", "ok": True, "detail": str(cfg_path)})
    else:
        checks.append({"name": "config_path", "ok": False, "detail": f"Missing {cfg_path}"})

    try:
        cfg = load_config_with_overrides(base_path, overrides)
        checks.append({"name": "config_load", "ok": True, "detail": "Loaded"})
    except Exception as e:
        checks.append({"name": "config_load", "ok": False, "detail": str(e)})
        emit_result("doctor", {"checks": checks})
        emit_done("doctor", False, "Config failed")
        return

    db_path = Path(cfg.get("library", {}).get("database_path", "data/metadata.db"))
    if not db_path.is_absolute():
        db_path = cfg_path.parent / db_path
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            conn.close()
            checks.append({"name": "database", "ok": count > 0, "detail": f"{count} tracks"})
        except Exception as e:
            checks.append({"name": "database", "ok": False, "detail": str(e)})
    else:
        checks.append({"name": "database", "ok": False, "detail": f"Missing {db_path}"})

    artifact_path = Path(cfg.get("playlists", {}).get("ds_pipeline", {}).get("artifact_path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz"))
    if not artifact_path.is_absolute():
        artifact_path = cfg_path.parent / artifact_path
    checks.append({"name": "artifacts", "ok": artifact_path.exists(), "detail": str(artifact_path)})

    emit_result("doctor", {"checks": checks})
    emit_done("doctor", True, "Doctor complete")

def handle_ping(cmd_data: Dict[str, Any]) -> None:
    """Handle ping command - simple health check with protocol version."""
    emit_log("INFO", "Pong!")
    # Include protocol version in ping response
    emit_event({
        "type": "result",
        "result_type": "pong",
        "protocol_version": PROTOCOL_VERSION
    })
    emit_done("ping", True)


def handle_cancel(cmd_data: Dict[str, Any]) -> None:
    """
    Handle cancel command - request cancellation of an active command.

    The cancel command itself does not have a request_id context;
    it specifies the request_id to cancel.
    """
    target_request_id = cmd_data.get("request_id")
    if not target_request_id:
        emit_log("WARNING", "Cancel command missing request_id", request_id=None)
        return

    if _worker_state.request_cancel(target_request_id):
        emit_log("WARNING", f"Cancellation requested for {target_request_id}", request_id=target_request_id)

        # Signal the cancellation token if available
        token = get_cancellation_token()
        if token:
            token.cancel(f"User requested cancellation for {target_request_id}")
    else:
        current = _worker_state.get_request_id()
        if current:
            emit_log("INFO", f"Cancel request_id {target_request_id} does not match active {current}")
        else:
            emit_log("INFO", f"Cancel request_id {target_request_id} - no active request")


def handle_generate_playlist(cmd_data: Dict[str, Any]) -> None:
    """Handle playlist generation command with cancellation support."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    args = cmd_data.get("args", {})

    mode = args.get("mode", "artist")
    artist = args.get("artist")
    genre = args.get("genre")
    track_title = args.get("track")
    seed_tracks = args.get("seed_tracks")
    if isinstance(seed_tracks, list):
        seed_tracks = [str(t).strip() for t in seed_tracks if str(t).strip()]
        if not seed_tracks:
            seed_tracks = None
    else:
        seed_tracks = None
    seed_track_ids = args.get("seed_track_ids")
    if isinstance(seed_track_ids, list):
        seed_track_ids = [str(t).strip() for t in seed_track_ids if str(t).strip()]
        if not seed_track_ids:
            seed_track_ids = None
    else:
        seed_track_ids = None
    track_count = args.get("tracks", 30)
    include_collaborations = bool(args.get("include_collaborations", False))

    emit_log("INFO", f"Starting playlist generation (mode={mode})")
    emit_progress("init", 0, 100, "Loading configuration")

    try:
        # Load and merge config
        config = load_config_with_overrides(base_path, overrides)
        emit_progress("init", 10, 100, "Configuration loaded")

        # Cancellation check after config load
        check_cancelled()

        # Import application modules (heavy imports deferred)
        emit_log("INFO", "Loading application modules...")
        emit_progress("init", 20, 100, "Loading modules")

        # Add project root to path if needed
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.config_loader import Config
        from src.local_library_client import LocalLibraryClient
        from src.playlist_generator import PlaylistGenerator
        from src.track_matcher import TrackMatcher
        from src.metadata_client import MetadataClient
        from src.lastfm_client import LastFMClient

        # Cancellation check after imports
        check_cancelled()

        # Apply overrides to config
        # Write merged config to temp or use Config directly
        class MergedConfig:
            """Wrapper that uses merged config dict instead of loading from file."""
            def __init__(self, config_dict: dict, original_path: str):
                self.config = config_dict
                self.config_path = original_path
                self._validate()

            def _validate(self):
                # Minimal validation
                if 'library' not in self.config:
                    raise ValueError("Missing 'library' section in config")

            def get(self, section: str, key: str = None, default: Any = None) -> Any:
                if section not in self.config:
                    return default
                if key is None:
                    return self.config.get(section, default)
                return self.config[section].get(key, default)

            @property
            def library_database_path(self) -> str:
                return self.config['library']['database_path']

            @property
            def library_music_directory(self) -> str:
                return self.config['library'].get('music_directory', 'E:\\MUSIC')

            @property
            def openai_api_key(self) -> str:
                import os
                return os.getenv('OPENAI_API_KEY') or self.config.get('openai', {}).get('api_key', '')

            @property
            def openai_model(self) -> str:
                return self.config.get('openai', {}).get('model', 'gpt-4o-mini')

            @property
            def lastfm_api_key(self) -> str:
                import os
                return os.getenv('LASTFM_API_KEY') or self.config.get('lastfm', {}).get('api_key', '')

            @property
            def lastfm_username(self) -> str:
                import os
                return os.getenv('LASTFM_USERNAME') or self.config.get('lastfm', {}).get('username', '')

            @property
            def lastfm_history_days(self) -> int:
                return self.config.get('lastfm', {}).get('history_days', 90)

            @property
            def min_duration_minutes(self) -> int:
                return self.config.get('playlists', {}).get('min_duration_minutes', 90)

            @property
            def min_track_duration_seconds(self) -> int:
                return self.config.get('playlists', {}).get('min_track_duration_seconds', 90)

            @property
            def max_track_duration_seconds(self) -> int:
                return self.config.get('playlists', {}).get('max_track_duration_seconds', 720)

            @property
            def recently_played_filter_enabled(self) -> bool:
                return self.config.get('playlists', {}).get('recently_played_filter', {}).get('enabled', True)

            @property
            def recently_played_lookback_days(self) -> int:
                return self.config.get('playlists', {}).get('recently_played_filter', {}).get('lookback_days', 0)

            @property
            def recently_played_min_playcount(self) -> int:
                return self.config.get('playlists', {}).get('recently_played_filter', {}).get('min_playcount_threshold', 0)

            @property
            def max_tracks_per_artist(self) -> int:
                return self.config.get('playlists', {}).get('max_tracks_per_artist', 3)

            @property
            def artist_window_size(self) -> int:
                return self.config.get('playlists', {}).get('artist_window_size', 8)

            @property
            def max_artist_per_window(self) -> int:
                return self.config.get('playlists', {}).get('max_artist_per_window', 1)

            @property
            def min_seed_artist_ratio(self) -> float:
                return self.config.get('playlists', {}).get('min_seed_artist_ratio', 0.125)

            @property
            def dynamic_sonic_ratio(self) -> float:
                return self.config.get('playlists', {}).get('dynamic_mode', {}).get('sonic_ratio', 0.6)

            @property
            def dynamic_genre_ratio(self) -> float:
                return self.config.get('playlists', {}).get('dynamic_mode', {}).get('genre_ratio', 0.4)

            @property
            def similarity_min_threshold(self) -> float:
                return self.config.get('playlists', {}).get('similarity', {}).get('min_threshold', 0.5)

            @property
            def limit_similar_tracks(self) -> int:
                return self.config.get('playlists', {}).get('limits', {}).get('similar_tracks', 50)

            @property
            def title_dedupe_enabled(self) -> bool:
                return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('enabled', True)

            @property
            def title_dedupe_threshold(self) -> int:
                return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('threshold', 92)

            @property
            def title_dedupe_mode(self) -> str:
                return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('mode', 'loose')

            @property
            def title_dedupe_short_title_min_len(self) -> int:
                return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('short_title_min_len', 6)

        emit_progress("init", 30, 100, "Initializing library client")
        merged_config = MergedConfig(config, base_path)
        library = LocalLibraryClient(db_path=merged_config.library_database_path)

        # Cancellation check after library init
        check_cancelled()

        emit_progress("init", 40, 100, "Initializing track matcher")
        matcher = TrackMatcher(library, library_id=None, db_path=merged_config.library_database_path)

        # Metadata client
        metadata = None
        try:
            metadata = MetadataClient(merged_config.library_database_path)
        except Exception:
            pass

        # Last.FM client (optional)
        lastfm = None
        if merged_config.lastfm_api_key and merged_config.lastfm_username:
            try:
                lastfm = LastFMClient(
                    api_key=merged_config.lastfm_api_key,
                    username=merged_config.lastfm_username
                )
            except Exception:
                pass

        # Cancellation check before playlist generator init
        check_cancelled()

        emit_progress("init", 50, 100, "Initializing playlist generator")
        generator = PlaylistGenerator(
            library,
            merged_config,
            lastfm_client=lastfm,
            track_matcher=matcher,
            metadata_client=metadata
        )

        # Get DS pipeline mode from config
        # Note: genre_mode and sonic_mode are SEPARATE settings that control weighting,
        # not the pipeline algorithm. They are already applied to the config via overrides.
        ds_mode = config.get('playlists', {}).get('ds_pipeline', {}).get('mode', 'dynamic')

        # Log which modes are active
        genre_mode = args.get("genre_mode") or config.get('playlists', {}).get(
            'genre_mode'
        )
        sonic_mode = args.get("sonic_mode") or config.get('playlists', {}).get(
            'sonic_mode'
        )
        if genre_mode:
            emit_log("INFO", f"Genre mode: {genre_mode}")
        if sonic_mode:
            emit_log("INFO", f"Sonic mode: {sonic_mode}")
        emit_log("INFO", f"DS pipeline mode: {ds_mode}")

        # Cancellation check before generation
        check_cancelled()

        emit_progress("generate", 60, 100, "Generating playlist")
        emit_log("INFO", f"Running playlist generation with mode={ds_mode}")

        if mode == "artist" and artist:
            # Single artist mode
            playlist_data = generator.create_playlist_for_artist(
                artist,
                track_count,
                track_title=track_title,
                track_titles=seed_tracks,
                dynamic=(ds_mode == "dynamic"),
                ds_mode_override=ds_mode,
                include_collaborations=include_collaborations,
            )
        elif mode == "seeds" and seed_tracks:
            # Seeds mode (Phase 2 UI)
            playlist_data = generator.create_playlist_from_seed_tracks(
                seed_tracks,
                track_count=track_count,
                dynamic=(ds_mode == "dynamic"),
                ds_mode_override=ds_mode,
                seed_track_ids=seed_track_ids,
            )
        elif mode == "artist" and seed_tracks:
            # Legacy seeds mode (old UI sent mode="artist" with seed_tracks)
            playlist_data = generator.create_playlist_from_seed_tracks(
                seed_tracks,
                track_count=track_count,
                dynamic=(ds_mode == "dynamic"),
                ds_mode_override=ds_mode,
                seed_track_ids=seed_track_ids,
            )
        elif mode == "genre" and genre:
            # Genre mode
            playlist_data = generator.create_playlist_for_genre(
                genre,
                track_count,
                dynamic=(ds_mode == "dynamic"),
                ds_mode_override=ds_mode,
            )
        else:
            # Fallback for unknown modes or missing required parameters
            raise ValueError(
                "Invalid mode or missing parameters: "
                f"mode={mode}, artist={artist}, seed_tracks={seed_tracks}"
            )

        # Cancellation check after generation, before output
        check_cancelled()

        emit_progress("generate", 90, 100, "Formatting results")

        if playlist_data:
            tracks = playlist_data.get('tracks', [])
            ds_report = playlist_data.get('ds_report', {}) or {}
            edge_scores = (
                (ds_report.get("playlist_stats") or {}).get("playlist", {}).get("edge_scores")
                or ds_report.get("edge_scores")
                or []
            )
            edge_map = {str(edge.get("cur_id")): edge for edge in edge_scores if edge.get("cur_id")}

            similarity_components = _build_seed_similarity_components(
                tracks=tracks,
                ds_report=ds_report,
                edge_map=edge_map,
            )

            formatted_tracks = []

            for i, track in enumerate(tracks, 1):
                genres = track.get('genres', [])
                rating_key = track.get('rating_key') or track.get('id') or track.get('track_id')
                # Fill genres lazily from similarity calculator if missing
                if (not genres) and rating_key and getattr(generator, "similarity_calc", None):
                    try:
                        genres = generator.similarity_calc.get_filtered_combined_genres_for_track(str(rating_key)) or []
                    except Exception:
                        genres = track.get('genres', []) or []

                # Prefer explicit similarity fields; fall back to edge scores from DS report
                edge = edge_map.get(str(rating_key), {})
                sonic_sim = (
                    track.get('similarity_score')
                    or track.get('hybrid_score')
                    or track.get('score')
                    or track.get('sonic_similarity')
                    or edge.get('S')
                )
                genre_sim = (
                    track.get('genre_sim')
                    or track.get('genre_similarity')
                    or track.get('G')
                    or edge.get('G')
                )
                components = similarity_components.get(str(rating_key)) if rating_key else None
                if components:
                    sonic_comp = components.get("sonic")
                    genre_comp = components.get("genre")
                    if sonic_comp and sonic_comp.get("t") is not None:
                        sonic_sim = sonic_comp.get("t")
                    if genre_comp and genre_comp.get("t") is not None:
                        genre_sim = genre_comp.get("t")
                else:
                    sonic_comp = None
                    genre_comp = None

                formatted_tracks.append({
                    "position": i,
                    "rating_key": rating_key,
                    "artist": track.get('artist', 'Unknown'),
                    "title": track.get('title', 'Unknown'),
                    "album": track.get('album', ''),
                    "duration_ms": track.get('duration', 0),
                    "file_path": track.get('file_path', ''),
                    "sonic_similarity": sonic_sim,
                    "genre_similarity": genre_sim,
                    "sonic_similarity_components": sonic_comp,
                    "genre_similarity_components": genre_comp,
                    "genres": genres,
                })

            playlist_result = {
                "name": playlist_data.get('title', 'Generated Playlist'),
                "tracks": formatted_tracks,
                "track_count": len(formatted_tracks),
            }

            # Include DS report metrics if available
            if ds_report:
                metrics = ds_report.get('metrics', {})
                playlist_result["metrics"] = {
                    "mean_transition": metrics.get('mean_transition'),
                    "min_transition": metrics.get('min_transition'),
                    "distinct_artists": metrics.get('distinct_artists'),
                }

            emit_result("playlist", {"playlist": playlist_result})
            emit_progress("complete", 100, 100, "Done")
            emit_done("generate_playlist", True, f"Generated {len(formatted_tracks)} tracks")
        else:
            emit_error("No playlist generated")
            emit_done("generate_playlist", False, "No playlist generated")

    except CancellationError:
        emit_log("INFO", "Playlist generation cancelled")
        emit_done("generate_playlist", False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("generate_playlist", False, str(e))


def handle_scan_library(cmd_data: Dict[str, Any]) -> None:
    """Handle library scan command with cancellation support."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})

    # Check if force flag is set for full scan
    force = overrides.get('force', False)
    resume_from_checkpoint = overrides.get("resume_from_checkpoint") or {}
    scan_type = "full scan" if force else "quick scan (incremental)"
    if resume_from_checkpoint:
        scan_type = f"{scan_type}, resuming"
    emit_log("INFO", f"Starting library scan ({scan_type})")
    emit_progress("scan", 0, 100, "Initializing")

    # Create cancellation token for this operation
    cancellation_token = CancellationToken() if CancellationToken else None
    set_cancellation_token(cancellation_token)

    try:
        config = load_config_with_overrides(base_path, overrides)
        check_cancelled()

        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.scan_library import LibraryScanner

        music_dir = config.get('library', {}).get('music_directory', 'E:\\MUSIC')

        check_cancelled()
        emit_progress("scan", 10, 100, f"Scanning {music_dir}")

        # LibraryScanner reads music_dir and db_path from config
        # Default to quick=True (incremental) unless force=True in overrides
        quick = not force
        scanner = LibraryScanner(config_path=base_path, cancellation_token=cancellation_token)
        stats = scanner.run(
            quick=quick,
            cleanup=True,  # Always cleanup missing files during pipeline
            resume_from_checkpoint=resume_from_checkpoint,
        )

        check_cancelled()
        emit_result("scan", {"stats": stats})
        total = stats.get('total', 0) if isinstance(stats, dict) else 0
        new_count = stats.get('new', 0) if isinstance(stats, dict) else 0
        updated = stats.get('updated', 0) if isinstance(stats, dict) else 0
        summary = f"Indexed {total} files (new {new_count}, updated {updated})"
        emit_done("scan_library", True, f"Scanned {total} files", summary=summary)

    except CancellationError:
        emit_log("INFO", "Library scan cancelled by user")
        emit_done("scan_library", False, "Cancelled by user", cancelled=True, summary="Scan cancelled")
    except CancelledException as e:
        # Graceful cancellation with checkpoint
        progress_data = getattr(e, 'progress', {})
        items_completed = progress_data.get('items_completed', 0)
        total_items = progress_data.get('total_items', 0)
        stats = progress_data.get('stats', {})

        emit_log("INFO", f"Library scan cancelled: {e}")

        # Emit checkpoint for potential resumption
        emit_checkpoint(
            stage="scan",
            items_completed=items_completed,
            resumable_state={
                "last_file_index": progress_data.get('last_file_index', 0),
                "stats": stats,
                "base_config_path": base_path,
                "overrides": overrides,
            }
        )

        # Emit partial results
        if stats:
            emit_result("scan", {"stats": stats})

        summary = f"Scan cancelled: {items_completed}/{total_items} files processed"
        emit_done("scan_library", False, str(e), cancelled=True, summary=summary)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("scan_library", False, str(e), summary="Scan failed")
    finally:
        # Clear cancellation token
        set_cancellation_token(None)


def handle_update_genres(cmd_data: Dict[str, Any]) -> None:
    """Handle genre update command with cancellation support."""
    emit_log("INFO", "Starting genre update")
    emit_progress("genres", 0, 100, "Initializing")

    try:
        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        check_cancelled()

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import and run genre updater
        from scripts.update_genres_v3_normalized import NormalizedGenreUpdater

        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        check_cancelled()
        emit_progress("genres", 20, 100, "Fetching genres")

        # This runs the full genre update (artists + albums)
        updater = NormalizedGenreUpdater(db_path=db_path)
        emit_progress("genres", 30, 100, "Updating artist genres")
        updater.update_artist_genres(progress=False)
        check_cancelled()
        emit_progress("genres", 60, 100, "Updating album genres")
        updater.update_album_genres(progress=False)
        check_cancelled()
        emit_progress("genres", 80, 100, "Updating file tag genres")
        updater.update_track_genres()
        check_cancelled()
        stats = {"artists_updated": 0, "albums_updated": 0}  # Updater doesn't return detailed stats
        updater.close()

        check_cancelled()
        emit_result("genres", {"stats": stats})
        summary = "Updated genres"
        if isinstance(stats, dict):
            artists = stats.get("artists_updated") or stats.get("artists_processed")
            albums = stats.get("albums_updated") or stats.get("albums_processed")
            parts = []
            if artists:
                parts.append(f"{artists} artists")
            if albums:
                parts.append(f"{albums} albums")
            if parts:
                summary = "Updated genres for " + ", ".join(parts)
        emit_done("update_genres", True, summary, summary=summary)

    except CancellationError:
        emit_log("INFO", "Genre update cancelled")
        emit_done("update_genres", False, "Cancelled by user", cancelled=True, summary="Genre update cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("update_genres", False, str(e), summary="Genre update failed")


def handle_update_sonic(cmd_data: Dict[str, Any]) -> None:
    """Handle sonic feature extraction command with cancellation support."""
    emit_log("INFO", "Starting sonic feature extraction")
    emit_progress("sonic", 0, 100, "Initializing")

    try:
        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        check_cancelled()

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.update_sonic import SonicFeaturePipeline

        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        check_cancelled()
        emit_progress("sonic", 20, 100, "Extracting features")

        # Run sonic analysis with beat3tower
        pipeline = SonicFeaturePipeline(db_path=db_path, use_beat_sync=False, use_beat3tower=True)
        emit_progress("sonic", 30, 100, "Analyzing tracks")
        pipeline.run(limit=None, force=False, progress=False)
        check_cancelled()
        stats = pipeline.get_stats()
        pipeline.close()

        check_cancelled()
        emit_result("sonic", {"stats": stats})
        summary = "Sonic analysis complete"
        if isinstance(stats, dict):
            analyzed = stats.get("analyzed") or stats.get("total") or stats.get("total_tracks")
            if analyzed is not None:
                summary = f"Analyzed {analyzed} tracks"
        emit_done("update_sonic", True, summary, summary=summary)

    except CancellationError:
        emit_log("INFO", "Sonic feature extraction cancelled")
        emit_done("update_sonic", False, "Cancelled by user", cancelled=True, summary="Sonic analysis cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("update_sonic", False, str(e), summary="Sonic analysis failed")


def handle_build_artifacts(cmd_data: Dict[str, Any]) -> None:
    """Handle artifact building command with cancellation support."""
    emit_log("INFO", "Starting artifact build")
    emit_progress("artifacts", 0, 100, "Initializing")

    try:
        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        check_cancelled()

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.build_beat3tower_artifacts import build_artifacts

        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        output_path = config.get('playlists', {}).get('ds_pipeline', {}).get(
            'artifact_path', 'data/artifacts/beat3tower_32k/data_matrices_step1.npz'
        )

        check_cancelled()
        emit_progress("artifacts", 20, 100, "Building matrices")

        # Create argparse-like namespace for build_artifacts
        from argparse import Namespace
        args = Namespace(
            db_path=db_path,
            config=base_path,
            output=output_path,
            genre_sim_path=None,
            max_tracks=0,
            no_pca=False,
            pca_variance=0.95,
            clip_sigma=3.0,
            random_seed=42,
            no_genre_normalization=False,
            verbose=False
        )
        build_artifacts(args)

        check_cancelled()
        emit_result("artifacts", {"output_path": output_path})
        summary = f"Built artifacts at {output_path}"
        emit_done("build_artifacts", True, summary, summary=summary)

    except CancellationError:
        emit_log("INFO", "Artifact build cancelled")
        emit_done("build_artifacts", False, "Cancelled by user", cancelled=True, summary="Artifact build cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("build_artifacts", False, str(e), summary="Artifact build failed")


def handle_blacklist_fetch(cmd_data: Dict[str, Any]) -> None:
    """Fetch blacklisted tracks from metadata DB."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        tracks = metadata.fetch_blacklisted_tracks()
        emit_result("blacklist", {"tracks": tracks, "count": len(tracks)})
        emit_done("blacklist_fetch", True, f"Fetched {len(tracks)} blacklisted tracks")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_fetch", False, str(e))


def handle_blacklist_set(cmd_data: Dict[str, Any]) -> None:
    """Set blacklisted flag for track ids."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    track_ids = cmd_data.get("track_ids", []) or []
    value = bool(cmd_data.get("value", True))
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        updated = metadata.set_blacklisted([str(t) for t in track_ids], value)
        emit_result(
            "blacklist_set",
            {"track_ids": track_ids, "value": value, "updated": updated},
        )
        emit_done("blacklist_set", True, f"Updated {updated} track(s)")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_set", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Command Router
# ─────────────────────────────────────────────────────────────────────────────

# Commands that run with request tracking (long-running operations)
TRACKED_COMMAND_HANDLERS = {
    "ping": handle_ping,
    "generate_playlist": handle_generate_playlist,
    "scan_library": handle_scan_library,
    "update_genres": handle_update_genres,
    "update_sonic": handle_update_sonic,
    "build_artifacts": handle_build_artifacts,
    "doctor": handle_doctor,
    "blacklist_fetch": handle_blacklist_fetch,
    "blacklist_set": handle_blacklist_set,
}

# Commands that don't have their own request context
def handle_set_logging_level(cmd_data: Dict[str, Any]) -> None:
    """
    Handle set_logging_level command - update worker logging level.

    This is an untracked command that doesn't produce tracked events.
    """
    level = cmd_data.get("level", "INFO").upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    if level not in valid_levels:
        emit_log("WARNING", f"Invalid log level: {level}, using INFO")
        level = "INFO"

    try:
        from . import logging_utils
        logging_utils.set_log_level(level)
        emit_log("INFO", f"Logging level changed to {level}")
    except Exception as e:
        emit_log("ERROR", f"Failed to change logging level: {e}")


UNTRACKED_COMMAND_HANDLERS = {
    "cancel": handle_cancel,
    "set_logging_level": handle_set_logging_level,
}


def process_command(line: str) -> None:
    """
    Parse and dispatch a command line.

    Handles request_id tracking for correlated events.
    """
    line = line.strip()
    if not line:
        return

    try:
        cmd_data = json.loads(line)
    except json.JSONDecodeError as e:
        emit_error(f"Invalid JSON: {e}")
        return

    cmd = cmd_data.get("cmd")
    if not cmd:
        emit_error("Missing 'cmd' field in command")
        return

    # Handle untracked commands (like cancel) first
    if cmd in UNTRACKED_COMMAND_HANDLERS:
        try:
            UNTRACKED_COMMAND_HANDLERS[cmd](cmd_data)
        except Exception as e:
            emit_log("ERROR", f"Untracked command {cmd} failed: {e}")
        return

    # For tracked commands, get identifiers
    request_id = cmd_data.get("request_id")
    job_id = cmd_data.get("job_id")
    if not request_id:
        # Legacy support: generate a warning but continue
        emit_log("WARNING", f"Command {cmd} missing request_id (legacy mode)")

    handler = TRACKED_COMMAND_HANDLERS.get(cmd)
    if not handler:
        emit_error(f"Unknown command: {cmd}")
        if request_id:
            _worker_state.start_request(request_id, cmd, job_id)
        emit_done(cmd, False, "Unknown command")
        if request_id:
            _worker_state.end_request()
        return

    # Start tracking this request
    if request_id:
        _worker_state.start_request(request_id, cmd, job_id)

    try:
        handler(cmd_data)
    except CancellationError:
        emit_log("INFO", f"Command {cmd} cancelled")
        emit_done(cmd, False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(f"Command {cmd} failed: {e}", tb)
        emit_done(cmd, False, str(e))
    finally:
        # Always clear request state after command completes
        if request_id:
            _worker_state.end_request()


def main():
    """Main worker loop - read commands from stdin, emit events to stdout."""
    setup_worker_logging()
    emit_log("INFO", "Worker started, ready for commands")

    try:
        for line in sys.stdin:
            process_command(line)
    except KeyboardInterrupt:
        emit_log("INFO", "Worker interrupted")
    except Exception as e:
        emit_error(f"Worker fatal error: {e}", traceback.format_exc())
        sys.exit(1)

    emit_log("INFO", "Worker shutdown")


if __name__ == "__main__":
    main()
