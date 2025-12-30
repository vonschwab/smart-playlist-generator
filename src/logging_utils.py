"""
Unified logging utilities for Playlist Generator

This module provides the single source of truth for logging configuration.
All entrypoints should call configure_logging() once at startup.

Usage:
    from src.logging_utils import configure_logging, stage_timer, redact

    configure_logging(level='INFO', log_file='run.log')

    with stage_timer("Processing tracks"):
        process_tracks()
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


def configure_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    file_level: str = 'DEBUG',
    force: bool = False,
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

    Environment variable overrides:
        LOG_LEVEL: Override the level parameter
        LOG_FILE: Override the log_file parameter
    """
    global _logging_configured

    if _logging_configured and not force:
        return

    # Environment overrides
    level = os.getenv('LOG_LEVEL', level).upper()
    if log_file is None:
        log_file = os.getenv('LOG_FILE')

    # Get root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level, logging.INFO))

    # Use simpler format for console
    console_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level, logging.DEBUG))

        # More detailed format for file
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ['urllib3', 'requests', 'musicbrainzngs', 'httpx']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True

    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={level}, file={log_file or 'none'}")


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
            # Match key: value or key=value patterns
            text = re.sub(
                rf'(["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']?)([^"\'\\s,}}]+)(["\']?)',
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
            # Format key nicely
            display_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                self.logger.log(level, f"  {display_key}: {value:.2f}")
            else:
                self.logger.log(level, f"  {display_key}: {value}")

        # Format timing
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.0f}s"

        self.logger.log(level, f"  Total Time: {time_str}")
        self.logger.log(level, "=" * 60)
