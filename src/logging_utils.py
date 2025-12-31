"""
Unified logging utilities for Playlist Generator.

All entrypoints should call configure_logging() once at startup.
"""
import logging
import sys
import os
import re
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, List, Union

# Track whether logging has been configured
_logging_configured = False
_run_id: Optional[str] = None
_HANDLER_TAG = "_pg_handler"
_CONSOLE_FMT_NO_RUN_ID = '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s'
_CONSOLE_FMT_WITH_RUN_ID = '%(asctime)s | %(levelname)-5s | %(name)s | run_id=%(run_id)s | %(message)s'
_FILE_FMT_WITH_RUN_ID = '%(asctime)s | %(levelname)-5s | %(name)s | %(funcName)s:%(lineno)d | run_id=%(run_id)s | %(message)s'


class RunIdFilter(logging.Filter):
    """Inject run_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id or "-"
        return True


def set_run_id(run_id: Optional[str]) -> None:
    """Set a global run_id used by log records."""
    global _run_id
    _run_id = run_id


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
        if self.verbose_each:
            return False
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
