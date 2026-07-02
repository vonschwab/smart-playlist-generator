"""Track-id -> file-path map from metadata.db (read-only helper for analyze stages)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def load_paths(db_path: Path | str) -> dict[str, str]:
    """track_id -> file_path from metadata.db (read-only, URI mode=ro)."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT track_id, file_path FROM tracks").fetchall()
    finally:
        con.close()
    return {str(t): p for t, p in rows if p}
