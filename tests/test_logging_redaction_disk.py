import logging
from pathlib import Path

import types

from src.playlist_gui import gui_logging


def test_redaction_before_disk(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(gui_logging, "user_log_dir", lambda *args, **kwargs: str(log_dir))

    emitter, buffer, log_path = gui_logging.setup_gui_logging()

    logger = logging.getLogger("redaction_test")
    secret_lines = [
        "token=SHOULD_NOT_APPEAR",
        "Authorization: Bearer SHOULD_NOT_APPEAR",
        "--api-key=SHOULD_NOT_APPEAR",
    ]
    for line in secret_lines:
        logger.warning(line)

    logging.shutdown()

    content = log_path.read_text(encoding="utf-8")
    for line in secret_lines:
        assert "SHOULD_NOT_APPEAR" not in content
