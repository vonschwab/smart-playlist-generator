"""
Debug report builder for support bundles.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import List
from importlib import metadata

from ..utils.redaction import redact_text
from ..diagnostics.checks import CheckResult


def _tail_lines(path: Path, n: int) -> List[str]:
    if not path or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return [line.rstrip("\n") for line in lines[-n:]]
    except Exception:
        return []


def _get_version() -> str:
    try:
        return metadata.version("playlist-generator")
    except Exception:
        return "unknown"


def build_debug_report(
    base_config_path: str,
    preset_label: str,
    mode: str,
    artist: str,
    worker_status: str,
    last_job_summary: str,
    last_job_error: str,
    readiness: List[CheckResult],
    gui_log_path: Path,
    worker_events: List[str],
) -> str:
    lines: List[str] = []
    lines.append("== Playlist Generator Debug Report ==")
    lines.append(f"App version: {_get_version()}")
    lines.append(f"Worker version: {_get_version()}")
    lines.append(f"OS: {platform.platform()}")
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"Base config path: {redact_text(base_config_path)}")
    lines.append(f"Preset: {preset_label}")
    lines.append(f"Mode: {mode}")
    if mode.lower() == "artist":
        lines.append(f"Artist query: {redact_text(artist)}")
    lines.append(f"Worker status: {worker_status}")
    lines.append(f"Last job: {redact_text(last_job_summary)}")
    if last_job_error:
        lines.append(f"Last job error: {redact_text(last_job_error)}")

    lines.append("\nReadiness:")
    if readiness:
        for r in readiness:
            lines.append(f"- {r.name}: {'OK' if r.ok else 'FAIL'} ({redact_text(r.detail)})")
    else:
        lines.append("- No diagnostics run")

    lines.append("\nGUI log (last 100 lines):")
    for line in _tail_lines(gui_log_path, 100):
        lines.append(redact_text(line))

    lines.append("\nWorker events (last 100):")
    for line in worker_events[-100:]:
        lines.append(redact_text(line))

    return "\n".join(lines)
