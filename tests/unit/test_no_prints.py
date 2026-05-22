import ast
import io
import logging
from pathlib import Path

import src.logging_utils as logging_utils


def test_no_print_statements():
    """Library code in src/ must not call print() — use the logging module instead.

    Scripts under scripts/ are user-facing CLIs and may print freely.
    The lone src/ exception is playlist_gui/worker.py, which deliberately
    writes NDJSON events to stdout as its IPC protocol with the GUI.
    """
    root = Path(__file__).resolve().parent.parent
    sources = list((root.parent / "src").rglob("*.py"))
    exempt = {
        (root.parent / "src" / "playlist_gui" / "worker.py").resolve(),
    }
    offenders = []
    for path in sources:
        if path.resolve() in exempt:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"print() calls remain in src/: {offenders}"


def test_quiet_suppresses_info(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(logging_utils, "_logging_configured", False)
    monkeypatch.setattr(logging_utils.sys, "stdout", buf)
    logging_utils.configure_logging(level="WARNING", console=True, force=True)

    logger = logging.getLogger("quiet_test")
    logger.info("info hidden")
    logger.warning("warn shown")

    output = buf.getvalue()
    assert "info hidden" not in output
    assert "warn shown" in output
