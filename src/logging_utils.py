"""
Unified logging utilities for MixArc.

All entrypoints should call configure_logging() once at startup.
"""
import itertools
import logging
import sys
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, List, Union, Callable

# Track whether logging has been configured
_logging_configured = False
_run_id: Optional[str] = None
_HANDLER_TAG = "_pg_handler"
_CONSOLE_FMT_NO_RUN_ID = '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s'
_CONSOLE_FMT_WITH_RUN_ID = '%(asctime)s | %(levelname)-5s | %(name)s | run_id=%(run_id)s | %(message)s'
_FILE_FMT_WITH_RUN_ID = '%(asctime)s | %(levelname)-5s | %(name)s | %(funcName)s:%(lineno)d | run_id=%(run_id)s | %(message)s'

# Per-playlist DEBUG log files (see
# docs/superpowers/specs/2026-07-02-per-playlist-logging-design.md).
# Distinct from _HANDLER_TAG so console-level controls (set_log_level) and
# configure_logging's tagged-handler cleanup never touch these handlers.
_PLAYLIST_HANDLER_TAG = "_pg_playlist_handler"
_PLAYLIST_LOG_SHORTID_COUNTER = itertools.count(1)
_ARTIST_UNSAFE_RE = re.compile(r'[^A-Za-z0-9_-]')
_ARTIST_MAX_LEN = 40


class RunIdFilter(logging.Filter):
    """Inject run_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id or "-"
        return True


def set_run_id(run_id: Optional[str]) -> None:
    """Set a global run_id used by log records."""
    global _run_id
    _run_id = run_id


def set_log_level(level: Union[str, int]) -> None:
    """
    Dynamically change the log level for all console handlers.

    Args:
        level: Log level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR') or int (logging.DEBUG, etc.)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, _HANDLER_TAG, False):
            handler.setLevel(level)


def get_log_level() -> str:
    """
    Get the current log level of console handlers.

    Returns:
        Log level name as string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, _HANDLER_TAG, False):
            return logging.getLevelName(handler.level)
    return 'INFO'


def _build_formatter(fmt: str, datefmt: Optional[str] = None) -> logging.Formatter:
    return logging.Formatter(fmt, datefmt=datefmt)


def configure_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    file_level: str = 'DEBUG',
    force: bool = False,
    run_id: Optional[str] = None,
    console: bool = True,
    show_run_id: bool = False,
    json_logs: bool = False,
) -> None:
    """
    Configure logging for the entire application.

    Should be called once at application startup (in main entrypoint).
    Subsequent calls are ignored unless force=True.

    Args:
        level: Console log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        file_level: Log level for file output (default DEBUG)
        force: If True, reconfigure even if already configured
        run_id: Optional run identifier to inject into log records
        console: Whether to add a console handler

    Environment variable overrides:
        LOG_LEVEL: Override the level parameter
        LOG_FILE: Override the log_file parameter
    """
    global _logging_configured

    if run_id:
        set_run_id(run_id)

    if _logging_configured and not force:
        return

    # Environment overrides
    level = os.getenv('LOG_LEVEL', level).upper()
    if log_file is None:
        log_file = os.getenv('LOG_FILE')

    # Get root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Remove handlers we previously installed (tagged)
    for handler in root.handlers[:]:
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)

    # Ensure a single run_id filter on root
    root.filters = [f for f in root.filters if not isinstance(f, RunIdFilter)]
    root.addFilter(RunIdFilter())

    use_run_id_console = show_run_id or level.upper() == "DEBUG" or json_logs

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level, logging.INFO))
        console_fmt = _build_formatter(
            _CONSOLE_FMT_WITH_RUN_ID if use_run_id_console else _CONSOLE_FMT_NO_RUN_ID,
            datefmt='%H:%M:%S',
        )
        console_handler.setFormatter(console_fmt)
        console_handler.addFilter(RunIdFilter())
        setattr(console_handler, _HANDLER_TAG, True)
        root.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level, logging.DEBUG))

        file_fmt = _build_formatter(
            _FILE_FMT_WITH_RUN_ID,
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        file_handler.setFormatter(file_fmt)
        file_handler.addFilter(RunIdFilter())
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ['urllib3', 'requests', 'musicbrainzngs', 'httpx']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True

    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={level}, file={log_file or 'none'}, run_id={_run_id or '-'}")


@contextmanager
def stage_timer(stage_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing pipeline stages.

    Logs stage start at DEBUG, completion with timing at INFO.

    Args:
        stage_name: Human-readable name for the stage
        logger: Logger to use (defaults to caller's module logger)

    Usage:
        with stage_timer("Candidate generation"):
            candidates = generate_candidates()
        # Logs: "Candidate generation completed in 2.3s"
    """
    if logger is None:
        # Get caller's logger based on module
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_module = frame.f_back.f_back.f_globals.get('__name__', __name__)
            logger = logging.getLogger(caller_module)
        else:
            logger = logging.getLogger(__name__)

    logger.debug(f"{stage_name} starting...")
    start = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed < 1:
            logger.info(f"{stage_name} completed in {elapsed*1000:.0f}ms")
        elif elapsed < 60:
            logger.info(f"{stage_name} completed in {elapsed:.1f}s")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            logger.info(f"{stage_name} completed in {minutes}m {seconds:.0f}s")


def redact(
    value: Any,
    keys: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
) -> str:
    """
    Redact sensitive information from a value before logging.

    Args:
        value: Value to redact (string, path, or dict)
        keys: Dict keys to redact (for dict values)
        patterns: Additional regex patterns to redact

    Returns:
        Redacted string representation

    Usage:
        logger.info(f"Config: {redact(config_path)}")
        logger.debug(f"Response: {redact(response, keys=['token', 'api_key'])}")
    """
    if value is None:
        return "None"

    # Handle Path objects - redact user directory
    if isinstance(value, Path):
        value = str(value)

    text = str(value)

    # Redact common patterns
    default_patterns = [
        # API keys and tokens
        (r'(["\']?(?:api[_-]?key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?)([^"\'\s]+)(["\']?)', r'\1***REDACTED***\3'),
        # User home directories (Windows and Unix)
        (r'C:\\Users\\[^\\]+', r'C:\\Users\\***'),
        (r'/home/[^/]+', r'/home/***'),
        (r'/Users/[^/]+', r'/Users/***'),
        # Email addresses
        (r'[\w.-]+@[\w.-]+\.\w+', r'***@***.***'),
    ]

    for pattern, replacement in default_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Apply custom patterns
    if patterns:
        for pattern in patterns:
            text = re.sub(pattern, '***REDACTED***', text)

    # Handle dict-like structures (redact specific keys)
    if keys:
        for key in keys:
            text = re.sub(
                rf'(["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']?)([^"\'\s,}}]+)(["\']?)',
                r'\1***REDACTED***\3',
                text,
                flags=re.IGNORECASE
            )

    return text


def format_count(n: int, singular: str, plural: Optional[str] = None) -> str:
    """
    Format a count with proper singular/plural form.

    Args:
        n: The count
        singular: Singular form (e.g., "track")
        plural: Plural form (defaults to singular + "s")

    Returns:
        Formatted string like "1 track" or "5 tracks"
    """
    if plural is None:
        plural = singular + 's'
    return f"{n:,} {singular if n == 1 else plural}"


def truncate_list(items: List[Any], max_items: int = 3, format_fn=str) -> str:
    """
    Format a list for logging, truncating if needed.

    Args:
        items: List to format
        max_items: Maximum items to show before truncating
        format_fn: Function to format each item

    Returns:
        Formatted string like "rock, pop, jazz (+5 more)"
    """
    if not items:
        return "(none)"

    formatted = [format_fn(item) for item in items[:max_items]]
    result = ', '.join(formatted)

    if len(items) > max_items:
        result += f" (+{len(items) - max_items} more)"

    return result


def _human_time(seconds: float) -> str:
    """Render a human-friendly duration."""
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours}h"


class ProgressLogger:
    """
    Emit periodic progress updates with optional per-item verbose logging.

    Default: INFO summaries every interval_s seconds or every_n items.
    Verbose: DEBUG per-item + INFO summaries.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: Optional[int],
        label: str,
        unit: str = "items",
        interval_s: float = 15.0,
        every_n: int = 500,
        verbose_each: bool = False,
        level_summary: int = logging.INFO,
        level_each: int = logging.DEBUG,
    ) -> None:
        self.logger = logger
        self.total = total if total and total > 0 else None
        self.label = label
        self.unit = unit
        self.interval_s = interval_s
        self.every_n = every_n
        self.verbose_each = verbose_each
        self.level_summary = level_summary
        self.level_each = level_each
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.last_count = 0
        self.processed = 0

    def _should_emit(self) -> bool:
        now = time.perf_counter()
        if (now - self.last_log_time) >= self.interval_s:
            return True
        if (self.processed - self.last_count) >= self.every_n:
            return True
        if self.total and self.processed >= self.total:
            return True
        return False

    def _progress_msg(self) -> str:
        elapsed = time.perf_counter() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0.0
        percent = f"{(self.processed / self.total * 100):.1f}%" if self.total else "?"
        eta = ""
        if self.total and rate > 0:
            remaining = max(self.total - self.processed, 0)
            eta_val = remaining / rate
            eta = f" | ETA {_human_time(eta_val)}"
        return (
            f"{self.label}: {self.processed:,}"
            + (f"/{self.total:,}" if self.total else "")
            + (f" ({percent})" if self.total else "")
            + f" | {rate:.1f} {self.unit}/s{eta}"
        )

    def update(self, n: int = 1, detail: Optional[str] = None) -> None:
        self.processed += n
        if self.verbose_each and detail:
            safe_detail = Path(detail).name if os.path.isabs(detail) else detail
            self.logger.log(
                self.level_each,
                f"{self.label} item {self.processed}" + (f"/{self.total}" if self.total else "") + f": {safe_detail}",
            )
        if self.verbose_each:
            # Still emit summaries periodically
            if self._should_emit():
                self._emit_summary()
            return
        if self._should_emit():
            self._emit_summary()

    def _emit_summary(self) -> None:
        self.logger.log(self.level_summary, self._progress_msg())
        self.last_log_time = time.perf_counter()
        self.last_count = self.processed

    def finish(self, detail: Optional[str] = None) -> None:
        # Always emit final summary
        if detail:
            self.logger.log(self.level_each if self.verbose_each else self.level_summary, detail)
        elapsed = time.perf_counter() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0.0
        suffix = f" | elapsed {_human_time(elapsed)} | avg {rate:.1f} {self.unit}/s"
        self.logger.log(self.level_summary, f"{self.label} complete: {self.processed:,} {self.unit}{suffix}")
def add_logging_args(parser) -> None:
    """
    Add standard logging CLI arguments to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance

    Usage:
        parser = argparse.ArgumentParser()
        add_logging_args(parser)
        args = parser.parse_args()

        level = resolve_log_level(args)
        configure_logging(level=level, log_file=args.log_file)
    """
    group = parser.add_argument_group('logging')
    group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging (shortcut for --log-level DEBUG)'
    )
    group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress most output (shortcut for --log-level WARNING)'
    )
    group.add_argument(
        '--log-file',
        type=str,
        metavar='PATH',
        help='Write logs to file'
    )
    group.add_argument(
        '--show-run-id',
        action='store_true',
        help='Include run_id in console logs (always included in file logs)',
    )


def resolve_log_level(args) -> str:
    """
    Resolve log level from parsed arguments.

    Priority: --debug > --quiet > --log-level

    Args:
        args: Parsed argparse namespace with logging arguments

    Returns:
        Log level string (DEBUG, INFO, WARNING, ERROR)
    """
    if getattr(args, 'debug', False):
        return 'DEBUG'
    if getattr(args, 'quiet', False):
        return 'WARNING'
    return getattr(args, 'log_level', 'INFO')


class RunSummary:
    """
    Collect metrics during a run and log a summary at the end.

    Usage:
        summary = RunSummary("Genre Update")
        summary.add("artists_processed", 150)
        summary.add("api_calls", 145)
        summary.add("failures", 5)
        summary.set_timing(elapsed_seconds)
        summary.log()
    """

    def __init__(self, title: str, logger: Optional[logging.Logger] = None):
        self.title = title
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: dict = {}
        self.timing: Optional[float] = None
        self.start_time = time.perf_counter()

    def add(self, key: str, value: Union[int, float, str]) -> None:
        """Add a metric to the summary."""
        self.metrics[key] = value

    def increment(self, key: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self.metrics[key] = self.metrics.get(key, 0) + amount

    def set_timing(self, seconds: float) -> None:
        """Set explicit timing (otherwise uses time since init)."""
        self.timing = seconds

    def log(self, level: int = logging.INFO) -> None:
        """Log the summary."""
        elapsed = self.timing if self.timing is not None else (time.perf_counter() - self.start_time)

        self.logger.log(level, "=" * 60)
        self.logger.log(level, f"{self.title.upper()} SUMMARY")

        for key, value in self.metrics.items():
            display_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                self.logger.log(level, f"  {display_key}: {value:.2f}")
            else:
                self.logger.log(level, f"  {display_key}: {value}")

        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.0f}s"

        self.logger.log(level, f"  Total Time: {time_str}")
        self.logger.log(level, "=" * 60)


def playlist_log_dir() -> Path:
    """ROOT-anchored directory for per-playlist DEBUG log files.

    Resolves to <repo_root>/logs/playlists, independent of cwd.
    """
    return Path(__file__).resolve().parents[1] / "logs" / "playlists"


def _sanitize_artist(artist: Optional[Any]) -> str:
    """Replace filesystem-unsafe characters and cap length for a log filename."""
    text = str(artist) if artist else "unknown"
    safe = _ARTIST_UNSAFE_RE.sub('_', text)
    safe = safe[:_ARTIST_MAX_LEN]
    return safe or "unknown"


def make_playlist_log_path(
    artist: Optional[Any],
    request_id: Optional[Any],
    *,
    dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Build a unique, sortable per-playlist log path.

    Shape: <dir>/<YYYY-MM-DD_HHMMSS>_<safe_artist>_<shortid>.log
    """
    base_dir = Path(dir) if dir is not None else playlist_log_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_artist = _sanitize_artist(artist)
    if request_id:
        shortid = str(request_id)[:6]
    else:
        shortid = f"{next(_PLAYLIST_LOG_SHORTID_COUNTER):06d}"
    return base_dir / f"{timestamp}_{safe_artist}_{shortid}.log"


@contextmanager
def playlist_log_file(
    artist: Optional[Any],
    request_id: Optional[Any],
    *,
    enabled: bool = True,
    dir: Optional[Union[str, Path]] = None,
    level: int = logging.DEBUG,
):
    """Attach a per-playlist DEBUG FileHandler to the root logger for the
    duration of the block, then detach and close it.

    When disabled, yields None and attaches nothing (byte-identical to
    not having this feature). Never raises out of setup or teardown -- a
    logging failure must never break a playlist generation.
    """
    if not enabled:
        yield None
        return

    handler: Optional[logging.FileHandler] = None
    path: Optional[Path] = None
    try:
        base_dir = Path(dir) if dir is not None else playlist_log_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        path = make_playlist_log_path(artist, request_id, dir=base_dir)

        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(_build_formatter(_FILE_FMT_WITH_RUN_ID, datefmt='%Y-%m-%d %H:%M:%S'))
        handler.addFilter(RunIdFilter())
        setattr(handler, _PLAYLIST_HANDLER_TAG, True)

        logging.getLogger().addHandler(handler)
    except Exception:
        if handler is not None:
            try:
                handler.close()
            except Exception:
                pass
        yield None
        return

    try:
        yield path
    finally:
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass


def cleanup_old_playlist_logs(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> int:
    """Delete *.log files under dir older than retention_days. Never raises."""
    return _cleanup_logs_older_than(dir, playlist_log_dir, retention_days)


def cleanup_old_playlist_logs_async(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> None:
    """Run cleanup_old_playlist_logs in a daemon thread. Never raises."""
    try:
        thread = threading.Thread(
            target=cleanup_old_playlist_logs,
            args=(dir, retention_days),
            daemon=True,
            name="playlist-log-cleanup",
        )
        thread.start()
    except Exception:
        pass


def _cleanup_logs_older_than(
    dir: Optional[Union[str, Path]],
    default_dir: Callable[[], Path],
    retention_days: int,
) -> int:
    """Delete *.log files under the resolved dir older than retention_days. Never raises."""
    try:
        base_dir = Path(dir) if dir is not None else default_dir()
        if not base_dir.exists():
            return 0
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for log_path in base_dir.glob("*.log"):
            try:
                if log_path.is_file() and log_path.stat().st_mtime < cutoff:
                    log_path.unlink()
                    deleted += 1
            except OSError:
                continue
        return deleted
    except Exception:
        return 0


def analyze_log_dir() -> Path:
    """ROOT-anchored directory for per-run Analyze Library log files."""
    return Path(__file__).resolve().parents[1] / "logs" / "analyze"


def make_analyze_log_path(run_id, *, dir: Optional[Union[str, Path]] = None) -> Path:
    """Build a unique, sortable per-run analyze log path.

    Shape: <dir>/<YYYY-MM-DD_HHMMSS>_<run_id[:6]>.log
    """
    base_dir = Path(dir) if dir is not None else analyze_log_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    shortid = str(run_id)[:6] if run_id else "000000"
    return base_dir / f"{timestamp}_{shortid}.log"


def cleanup_old_analyze_logs(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> int:
    """Delete analyze *.log files older than retention_days. Never raises."""
    return _cleanup_logs_older_than(dir, analyze_log_dir, retention_days)


def cleanup_old_analyze_logs_async(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> None:
    """Run cleanup_old_analyze_logs in a daemon thread. Never raises."""
    try:
        thread = threading.Thread(
            target=cleanup_old_analyze_logs,
            args=(dir, retention_days),
            daemon=True,
            name="analyze-log-cleanup",
        )
        thread.start()
    except Exception:
        pass
