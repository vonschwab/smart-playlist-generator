"""
GUI logging setup with rotating file handler and Qt signal bridge.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple

from PySide6.QtCore import QObject, Signal

try:
    from platformdirs import user_log_dir
except ImportError:  # pragma: no cover
    user_log_dir = None

from .utils.redaction import redact_text
from .utils.bounded_buffer import BoundedBuffer


class LogEmitter(QObject):
    """Qt signal emitter for log lines."""

    log_ready = Signal(str, str)  # level, message


class QtLogHandler(logging.Handler):
    """Logging handler that emits records to Qt and stores a ring buffer."""

    def __init__(self, emitter: LogEmitter, buffer: BoundedBuffer):
        super().__init__()
        self._emitter = emitter
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return

        redacted = redact_text(msg)
        line = f"{record.levelname}: {redacted}"
        self._buffer.append(line)
        self._emitter.log_ready.emit(record.levelname.upper(), redacted)


def _build_log_dir() -> Path:
    if user_log_dir:
        base = Path(user_log_dir("PlaylistGenerator", "PlaylistGenerator"))
    else:
        base = Path.home() / ".PlaylistGenerator" / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def setup_gui_logging() -> Tuple[LogEmitter, BoundedBuffer, Path]:
    """
    Configure application-wide logging with rotation and Qt bridge.

    Returns:
        (emitter, buffer, log_path)
    """
    emitter = LogEmitter()
    buffer = BoundedBuffer()
    log_dir = _build_log_dir()
    log_path = log_dir / "playlist_gui.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplication on restart
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    def _console_filter(rec: logging.LogRecord) -> bool:
        rec.msg = redact_text(rec.getMessage())
        rec.args = ()
        return True

    console.addFilter(_console_filter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    class RedactFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.msg = redact_text(record.getMessage())
            record.args = ()
            return True

    file_handler.addFilter(RedactFilter())
    root.addHandler(file_handler)

    qt_handler = QtLogHandler(emitter, buffer)
    qt_handler.setLevel(logging.INFO)
    qt_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(qt_handler)

    logging.getLogger(__name__).info("GUI logging initialized at %s", log_path)
    return emitter, buffer, log_path
