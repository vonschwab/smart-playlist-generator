"""Range-aware local audio file streaming for the browser player."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

_MIME = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
}

_CHUNK = 256 * 1024


def _lookup_path(track_id: str, db_path: Path) -> Optional[str]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT file_path FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _content_type(path: str) -> str:
    return _MIME.get(Path(path).suffix.lower(), "application/octet-stream")


def stream_audio(track_id: str, db_path: Path, request: Request) -> Response:
    """Stream the audio file for track_id, honouring an optional Range header."""
    file_path = _lookup_path(track_id, db_path)
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Track audio not found")

    file_size = os.path.getsize(file_path)
    content_type = _content_type(file_path)
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            file_path,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"},
        )

    # Parse "bytes=START-END"
    try:
        units, _, rng = range_header.partition("=")
        if units.strip() != "bytes":
            raise ValueError
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if start >= file_size or start < 0:
        raise HTTPException(
            status_code=416,
            detail="Range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)
    length = end - start + 1

    def _iter():
        with open(file_path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": content_type,
    }
    return StreamingResponse(_iter(), status_code=206, headers=headers)
