"""Read-only access to enriched genre signatures from the sidecar DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

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
        self._all_enriched_genres_cache: dict[str, list[str]] | None = None

    def get_enriched_genres(self, *, artist: str, album: str | None) -> list[str] | None:
        if not album:
            return None
        release_key = self._release_key(artist, album)
        with self._connect() as conn:
            sig_row = conn.execute(
                "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
                (release_key,),
            ).fetchone()
            override_row = (
                conn.execute(
                    "SELECT genres_add_json, genres_remove_json "
                    "FROM ai_genre_user_overrides WHERE release_key = ?",
                    (release_key,),
                ).fetchone()
                if self._user_overrides_exist(conn)
                else None
            )

        sig_genres: list[str] = []
        if sig_row:
            payload = json.loads(sig_row["signature_json"])
            sig_genres = list(payload.get("genres") or [])

        if not override_row:
            return sig_genres if sig_genres else None

        # Both sets stored casefolded; compare sig genres by casefold to match.
        add = list(json.loads(override_row["genres_add_json"]))
        remove_lower = set(json.loads(override_row["genres_remove_json"]))
        combined: list[str] = []
        combined_lower: set[str] = set()
        for g in sig_genres:
            gk = g.casefold()
            if gk not in remove_lower and gk not in combined_lower:
                combined.append(g)
                combined_lower.add(gk)
        for g in add:
            gk = g.casefold()
            if gk not in combined_lower:
                combined.append(g)
                combined_lower.add(gk)
        return combined if combined else None

    @staticmethod
    def _user_overrides_exist(conn) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='ai_genre_user_overrides'"
        ).fetchone()
        return row is not None

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

    def get_release_keys_with_genre(self, genre: str) -> set[str]:
        """Return release_keys whose enriched signature contains the given genre (casefold-matched)."""
        index = self._build_reverse_index()
        return index.get(genre.casefold(), set())

    def get_all_enriched_release_keys(self) -> set[str]:
        """Return the set of all release_keys with an enriched signature."""
        if self._all_enriched_cache is None:
            self._all_enriched_cache = set(self.get_all_enriched_genres().keys())
        return self._all_enriched_cache

    def get_all_enriched_genres(self) -> dict[str, list[str]]:
        """Return {release_key: [genres]} for every enriched signature in one query.

        Result is cached for the lifetime of this resolver instance. Use this
        in bulk paths (artifact build, library preload) to avoid per-track
        connection opens.
        """
        if self._all_enriched_genres_cache is not None:
            return self._all_enriched_genres_cache
        mapping: dict[str, list[str]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT release_key, signature_json FROM enriched_genre_signatures"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["signature_json"])
            genres = payload.get("genres") or []
            if genres:
                mapping[row["release_key"]] = list(genres)
        self._all_enriched_genres_cache = mapping
        return mapping

    def make_release_key(self, artist: str, album: str) -> str:
        """Return the normalized release key for (artist, album)."""
        return self._release_key(artist, album)

    def _build_reverse_index(self) -> dict[str, set[str]]:
        if self._reverse_index_cache is not None:
            return self._reverse_index_cache
        index: dict[str, set[str]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT release_key, signature_json FROM enriched_genre_signatures"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["signature_json"])
            for genre in (payload.get("genres") or []):
                index.setdefault(genre.casefold(), set()).add(row["release_key"])
        self._reverse_index_cache = index
        return index

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
            conn.execute(
                "CREATE TABLE ai_genre_user_overrides("
                "release_key TEXT, normalized_artist TEXT, normalized_album TEXT, "
                "genres_add_json TEXT, genres_remove_json TEXT, updated_at TEXT)"
            )
            return conn
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
