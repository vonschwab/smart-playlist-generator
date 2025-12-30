"""
Logging configuration for Playlist Generator

This module provides backwards-compatible wrappers around logging_utils.
New code should import directly from src.logging_utils.

Deprecated:
    - setup_logging() - use configure_logging() instead
    - setup_root_logging() - use configure_logging() instead
"""
import logging
import os
from typing import Optional

from src.logging_utils import configure_logging, stage_timer, redact, RunSummary


def setup_logging(
    name: str = 'playlist_generator',
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging and return a named logger.

    DEPRECATED: Use configure_logging() from src.logging_utils instead.

    This function is kept for backwards compatibility with existing scripts.
    It now configures root logging and returns a named logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Named logger instance
    """
    # Resolve level
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')

    # Configure root logging (idempotent)
    configure_logging(level=level, log_file=log_file)

    # Return named logger
    return logging.getLogger(name)


def setup_root_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Set up root logger for all modules.

    DEPRECATED: Use configure_logging() from src.logging_utils instead.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')

    if log_file is None:
        log_file = os.getenv('LOG_FILE')

    configure_logging(level=level, log_file=log_file)


# Re-export utilities for convenience
__all__ = [
    'setup_logging',
    'setup_root_logging',
    'configure_logging',
    'stage_timer',
    'redact',
    'RunSummary',
]
