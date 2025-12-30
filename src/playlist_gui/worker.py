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
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
    current_cmd: Optional[str] = None
    cancel_requested: bool = False
    _lock: threading.Lock = None

    def __post_init__(self):
        self._lock = threading.Lock()

    def start_request(self, request_id: str, cmd: str) -> None:
        """Mark a new request as active."""
        with self._lock:
            self.current_request_id = request_id
            self.current_cmd = cmd
            self.cancel_requested = False

    def end_request(self) -> None:
        """Clear the active request."""
        with self._lock:
            self.current_request_id = None
            self.current_cmd = None
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


# Global worker state
_worker_state = WorkerState()


def check_cancelled() -> None:
    """Convenience function to check cancellation at stage boundaries."""
    _worker_state.check_cancelled()


# ─────────────────────────────────────────────────────────────────────────────
# Secret Redaction
# ─────────────────────────────────────────────────────────────────────────────

SECRET_PATTERNS = re.compile(
    r"(api_key|token|secret|password|credential|bearer)[\s]*[=:]\s*['\"]?([^'\"\s,}]+)",
    re.IGNORECASE
)


def redact_secrets_in_text(text: str) -> str:
    """Redact any secret values from text (for logs, tracebacks)."""
    return SECRET_PATTERNS.sub(r"\1=***REDACTED***", text)


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
    event = {"type": "log", "level": level, "msg": redact_secrets_in_text(msg)}
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
    event = {"type": "error", "message": redact_secrets_in_text(message)}
    if tb:
        event["traceback"] = redact_secrets_in_text(tb)
    emit_event(event)


def emit_result(result_type: str, data: Dict[str, Any]) -> None:
    """Emit a result event."""
    emit_event({"type": "result", "result_type": result_type, **data})


def emit_done(cmd: str, ok: bool, detail: Optional[str] = None, cancelled: bool = False) -> None:
    """Emit a done event indicating command completion."""
    event = {"type": "done", "cmd": cmd, "ok": ok}
    if detail:
        event["detail"] = detail
    if cancelled:
        event["cancelled"] = True
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

    return deep_merge(config, overrides)


# ─────────────────────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────────────────────

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

    mode = args.get("mode", "history")
    artist = args.get("artist")
    track_title = args.get("track")
    track_count = args.get("tracks", 30)

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

        # Get DS mode override from config
        ds_mode = config.get('playlists', {}).get('ds_pipeline', {}).get('mode', 'dynamic')

        # Cancellation check before generation
        check_cancelled()

        emit_progress("generate", 60, 100, "Generating playlist")
        emit_log("INFO", f"Running pipeline with mode={ds_mode}")

        if mode == "artist" and artist:
            # Single artist mode
            playlist_data = generator.create_playlist_for_artist(
                artist,
                track_count,
                track_title=track_title,
                dynamic=(ds_mode == "dynamic"),
                ds_mode_override=ds_mode,
            )
        else:
            # History mode - generate batch
            playlist_count = config.get('playlists', {}).get('count', 8)
            playlists = generator.create_playlist_batch(
                playlist_count,
                dynamic=(ds_mode == "dynamic"),
                ds_mode_override=ds_mode,
            )
            # Use first playlist
            playlist_data = playlists[0] if playlists else None

        # Cancellation check after generation, before output
        check_cancelled()

        emit_progress("generate", 90, 100, "Formatting results")

        if playlist_data:
            tracks = playlist_data.get('tracks', [])
            formatted_tracks = []

            for i, track in enumerate(tracks, 1):
                formatted_tracks.append({
                    "position": i,
                    "artist": track.get('artist', 'Unknown'),
                    "title": track.get('title', 'Unknown'),
                    "album": track.get('album', ''),
                    "duration_ms": track.get('duration', 0),
                    "file_path": track.get('file_path', ''),
                })

            playlist_result = {
                "name": playlist_data.get('title', 'Generated Playlist'),
                "tracks": formatted_tracks,
                "track_count": len(formatted_tracks),
            }

            # Include DS report metrics if available
            ds_report = playlist_data.get('ds_report', {})
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

    emit_log("INFO", "Starting library scan")
    emit_progress("scan", 0, 100, "Initializing")

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
        scanner = LibraryScanner(config_path=base_path)
        stats = scanner.scan(quick=False)

        check_cancelled()
        emit_result("scan", {"stats": stats})
        emit_done("scan_library", True, f"Scanned {stats.get('total', 0)} files")

    except CancellationError:
        emit_log("INFO", "Library scan cancelled")
        emit_done("scan_library", False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("scan_library", False, str(e))


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
        from scripts.update_genres_v3_normalized import update_genres_main

        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        check_cancelled()
        emit_progress("genres", 20, 100, "Fetching genres")

        # This runs the full genre update
        stats = update_genres_main(db_path=db_path, config_path=base_path)

        check_cancelled()
        emit_result("genres", {"stats": stats})
        emit_done("update_genres", True)

    except CancellationError:
        emit_log("INFO", "Genre update cancelled")
        emit_done("update_genres", False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("update_genres", False, str(e))


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

        from scripts.update_sonic import main as update_sonic_main

        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        check_cancelled()
        emit_progress("sonic", 20, 100, "Extracting features")

        stats = update_sonic_main(db_path=db_path)

        check_cancelled()
        emit_result("sonic", {"stats": stats})
        emit_done("update_sonic", True)

    except CancellationError:
        emit_log("INFO", "Sonic feature extraction cancelled")
        emit_done("update_sonic", False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("update_sonic", False, str(e))


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

        from scripts.build_beat3tower_artifacts import main as build_artifacts_main

        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        output_path = config.get('playlists', {}).get('ds_pipeline', {}).get(
            'artifact_path', 'data/artifacts/beat3tower_32k/data_matrices_step1.npz'
        )

        check_cancelled()
        emit_progress("artifacts", 20, 100, "Building matrices")

        build_artifacts_main(
            db_path=db_path,
            config_path=base_path,
            output_path=output_path
        )

        check_cancelled()
        emit_result("artifacts", {"output_path": output_path})
        emit_done("build_artifacts", True)

    except CancellationError:
        emit_log("INFO", "Artifact build cancelled")
        emit_done("build_artifacts", False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("build_artifacts", False, str(e))


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
}

# Commands that don't have their own request context
UNTRACKED_COMMAND_HANDLERS = {
    "cancel": handle_cancel,
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

    # For tracked commands, get request_id
    request_id = cmd_data.get("request_id")
    if not request_id:
        # Legacy support: generate a warning but continue
        emit_log("WARNING", f"Command {cmd} missing request_id (legacy mode)")

    handler = TRACKED_COMMAND_HANDLERS.get(cmd)
    if not handler:
        emit_error(f"Unknown command: {cmd}")
        if request_id:
            _worker_state.start_request(request_id, cmd)
        emit_done(cmd, False, "Unknown command")
        if request_id:
            _worker_state.end_request()
        return

    # Start tracking this request
    if request_id:
        _worker_state.start_request(request_id, cmd)

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
