import io
import logging
import sys
from pathlib import Path

import src.logging_utils as logging_utils


def _reset_logging():
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging_utils._logging_configured = False  # type: ignore[attr-defined]
    logging_utils.set_run_id(None)


def test_configure_logging_idempotent():
    _reset_logging()
    logging_utils.configure_logging(level="INFO", force=True)
    root = logging.getLogger()
    count = len(root.handlers)

    logging_utils.configure_logging(level="INFO")
    assert len(root.handlers) == count
    _reset_logging()


def test_run_id_in_output(monkeypatch):
    _reset_logging()
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    logging_utils.configure_logging(level="INFO", force=True, run_id="test-run")
    logging.getLogger(__name__).info("hello")
    output = buf.getvalue()
    assert "run_id=test-run" not in output  # console default omits

    _reset_logging()
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    logging_utils.configure_logging(level="INFO", force=True, run_id="test-run", show_run_id=True)
    logging.getLogger(__name__).info("hello")
    output = buf.getvalue()
    assert "run_id=test-run" in output

    _reset_logging()
    tmp_file = Path("test_log.txt")
    if tmp_file.exists():
        tmp_file.unlink()
    logging_utils.configure_logging(level="INFO", force=True, run_id="test-run", log_file=str(tmp_file))
    logging.getLogger(__name__).info("hello-file")
    file_text = tmp_file.read_text(encoding="utf-8")
    assert "run_id=test-run" in file_text
    _reset_logging()
    tmp_file.unlink(missing_ok=True)


def test_no_basicconfig_in_src_scripts():
    base_paths = [Path("src"), Path("scripts")]
    offending = []
    for base in base_paths:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "basicConfig(" in text:
                offending.append(path)
    assert offending == []
