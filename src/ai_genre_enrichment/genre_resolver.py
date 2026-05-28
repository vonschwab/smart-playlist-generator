"""Read-only access to enriched genre signatures from the sidecar DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .tag_classification import normalize_source_tag


class EnrichedGenreResolver:
    """Resolves enriched genres for a (artist, album) tuple.

    Opens the sidecar DB read-only. Returns None when no enriched signature
    exists for the release — callers fall back to raw metadata.
    """

    def __init__(self, sidecar_db_path: str | Path):
        self._db_path = Path(sidecar_db_path).resolve()
        self._reverse_index_cache: dict[str, set[str]] | None = None
        self._all_enriched_cache: set[str] | None = None

    def get_enriched_genres(self, *, artist: str, album: str | None) -> list[str] | None:
        if not album:
            return None
        release_key = self._release_key(artist, album)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
                (release_key,),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["signature_json"])
        genres = payload.get("genres") or []
        return list(genres) if genres else None

    def is_enriched(self, *, artist: str, album: str | None) -> bool:
        return self.get_enriched_genres(artist=artist, album=album) is not None

    def get_artist_enrichment_status(self, artist: str) -> dict:
        """Return enrichment status for an artist.

        Result keys: enriched_count (int), enriched_albums (list[str] — normalized album names).
        """
        normalized_artist = normalize_source_tag(artist)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT normalized_album FROM enriched_genre_signatures "
                "WHERE normalized_artist = ? ORDER BY normalized_album",
                (normalized_artist,),
            ).fetchall()
        albums = [row["normalized_album"] for row in rows]
        return {"enriched_count": len(albums), "enriched_albums": albums}

    def _release_key(self, artist: str, album: str) -> str:
        return f"{normalize_source_tag(artist)}::{normalize_source_tag(album)}"

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path.exists():
            # Return an in-memory empty DB so callers always get None
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE enriched_genre_signatures(release_key TEXT, "
                "normalized_artist TEXT, normalized_album TEXT, signature_json TEXT)"
            )
            return conn
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
