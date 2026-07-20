# src/playlist_web/setup_state.py
"""First-class setup state for config-less boot (MixArc SP-1).

A wheel install (or a repo checkout with a fresh clone / no config.yaml yet)
must be able to boot the FastAPI app and serve a status page instead of
crashing before the browser can even open. ``derive_setup_state`` inspects
the resolved ``MixarcHome`` and answers with exactly one of three states so
the frontend (Task 5) and the startup worker guard (this task) can act on it
without re-deriving the logic.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import yaml

from src.config_loader import resolve_database_path
from src.mixarc.paths import MixarcHome


class SetupState(str, Enum):
    NEEDS_SETUP = "needs_setup"
    NEEDS_ANALYZE = "needs_analyze"
    READY = "ready"


@dataclass
class SetupStatus:
    state: SetupState
    config_path: str
    config_exists: bool
    music_directory: str | None
    db_path: str | None
    track_count: int | None
    detail: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d


def derive_setup_state(home: MixarcHome) -> SetupStatus:
    cfg_path = home.config_path
    if not cfg_path.exists():
        return SetupStatus(SetupState.NEEDS_SETUP, str(cfg_path), False, None, None, None,
                            "No config.yaml — run setup.")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # unreadable config counts as needs-setup, loudly
        return SetupStatus(SetupState.NEEDS_SETUP, str(cfg_path), True, None, None, None,
                            f"config.yaml unreadable: {exc}")
    music = ((raw.get("library") or {}).get("music_directory") or "").strip() or None
    if not music or not Path(music).is_dir():
        return SetupStatus(SetupState.NEEDS_SETUP, str(cfg_path), True, music, None, None,
                            "library.music_directory unset or not a directory.")
    db_path = resolve_database_path(raw, anchor=home.anchor_dir)
    count = _track_count(db_path)
    if count is None or count == 0:
        return SetupStatus(SetupState.NEEDS_ANALYZE, str(cfg_path), True, music, db_path, count,
                            "Library not analyzed yet.")
    return SetupStatus(SetupState.READY, str(cfg_path), True, music, db_path, count, "Ready.")


def _track_count(db_path: str) -> int | None:
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='tracks'").fetchone()
            if not row or row[0] == 0:
                return None
            return int(conn.execute("SELECT count(*) FROM tracks").fetchone()[0])
    except sqlite3.Error:
        return None
