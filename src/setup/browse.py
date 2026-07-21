"""Server-side directory listing for the setup wizard's folder picker."""
from __future__ import annotations

from pathlib import Path

AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".wma", ".aiff", ".aif"}
_COUNT_CAP = 5000  # bound the scan so a giant folder returns fast


def _audio_count(d: Path) -> int:
    n = 0
    try:
        for entry in d.iterdir():
            if entry.is_file() and entry.suffix.lower() in AUDIO_EXTS:
                n += 1
                if n >= _COUNT_CAP:
                    break
    except (PermissionError, OSError):
        return 0
    return n


def list_directory(path: str | None) -> dict:
    base = Path(path).expanduser() if path else Path.home()
    base = base.resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(str(base))
    entries = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, OSError):
        children = []
    for child in children:
        try:
            if not child.is_dir():
                continue
        except OSError:
            continue
        entries.append({"name": child.name, "path": str(child), "audio_count": _audio_count(child)})
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": entries, "is_music_dir": _audio_count(base) > 0}
