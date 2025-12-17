"""
Centralized logging configuration for Playlist Generator

Provides unified logging setup across all scripts and modules.
Supports environment variable overrides and consistent formatting.
"""
import logging
import sys
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = 'playlist_generator',
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up centralized logging for scripts and applications.

    Args:
        name: Logger name (typically __name__ or 'playlist_generator')
        level: Log level (DEBUG, INFO, WARNING, ERROR) - defaults to config or INFO
        log_file: Optional log file path

    Returns:
        Configured logger instance

    Examples:
        # In a script
        from src.logging_config import setup_logging
        logger = setup_logging(level='INFO', log_file='my_script.log')
        logger.info("Script started")

        # With environment variable override
        export LOG_LEVEL=DEBUG
        logger = setup_logging()  # Will use DEBUG level
    """
    # Determine log level (environment > argument > default)
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')

    # Determine log file (environment > argument > default)
    if log_file is None:
        log_file = os.getenv('LOG_FILE')

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create parent directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG to file

        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def setup_root_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Set up root logger for all modules.

    This should be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Examples:
        # In main_app.py or API startup
        from src.logging_config import setup_root_logging
        setup_root_logging(level='INFO', log_file='playlist_generator.log')
    """
    # Determine log level
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')

    if log_file is None:
        log_file = os.getenv('LOG_FILE', 'playlist_generator.log')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)


# Example usage / test
if __name__ == "__main__":
    # Test script logging
    logger = setup_logging(level='DEBUG', log_file='test_logging.log')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("\nLogging configured successfully!")
    print("Check test_logging.log for file output")
