"""Workspace detection shared by guard hooks and doctor: canonical vs satellite.

A SATELLITE is a standing full clone of the canonical checkout whose git
`origin` is a LOCAL path (see docs/superpowers/specs/
2026-07-06-simultaneous-sessions-design.md). The canonical checkout's origin
is GitHub. Detection reads `.git/config` directly (no subprocess — this runs
inside PreToolUse hooks on every tool call).

Unknown states (no .git, no origin, unreadable config) report CANONICAL:
that resolves toward the strictest guard behavior, never toward allowing a
satellite-only action in the wrong place.
"""

import os
import re
from pathlib import Path

_ORIGIN_SECTION = re.compile(r'^\s*\[remote\s+"origin"\]\s*$')
_ANY_SECTION = re.compile(r"^\s*\[")
_URL_LINE = re.compile(r"^\s*url\s*=\s*(.+?)\s*$")
_NONLOCAL_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")
_LOCAL_PATH = re.compile(r"^[a-zA-Z]:[\\/]|^\\\\|^/")  # drive, UNC, posix root


def origin_url(project_dir):
    """The `[remote "origin"] url` from .git/config, or None."""
    cfg = Path(project_dir) / ".git" / "config"
    try:
        text = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_origin = False
    for line in text.splitlines():
        if _ANY_SECTION.match(line):
            in_origin = bool(_ORIGIN_SECTION.match(line))
            continue
        if in_origin:
            m = _URL_LINE.match(line)
            if m:
                return m.group(1)
    return None


def is_satellite(project_dir=None):
    """True iff this checkout's origin is a local filesystem path."""
    root = project_dir or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    url = origin_url(root)
    if not url:
        return False
    lowered = url.lower()
    if lowered.startswith(_NONLOCAL_PREFIXES):
        return False
    if lowered.startswith("file://"):
        return True
    return bool(_LOCAL_PATH.match(url))
