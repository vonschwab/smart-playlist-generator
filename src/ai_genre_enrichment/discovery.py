from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.genre.normalize_unified import normalize_genre_token

from .normalization import make_release_key, normalize_release_artist, normalize_release_name

GENERIC_OR_DESCRIPTOR_TAGS = {
    "alternative",
    "classic",
    "contemporary",
    "electronic",
    "experimental",
    "folk",
    "hip hop",
    "indie",
    "indie rock",
    "instrumental",
    "jazz",
    "live",
    "modern",
    "pop",
    "remastered",
    "rock",
    "singer-songwriter",
    "soundtrack",
    "underground",
    "world",
}


@dataclass(frozen=True)
class ReleasePayload:
    artist: str
    album: str
    normalized_artist: str
    normalized_album: str
    release_key: str
    album_id: str | None
    identifiers: dict[str, str]
    year: int | None
    track_titles: list[str]
    existing_genres_by_source: dict[str, list[str]]
    genre_counts: dict[str, int]

    def to_request_payload(self) -> dict[str, Any]:
        return asdict(self)


def discover_releases(
    metadata_db_path: str | Path = "data/metadata.db",
    *,
    limit: int | None = None,
    artist: str | None = None,
    album: str | None = None,
    generic_only: bool = False,
    min_existing_specific_genres: int | None = None,
    track_title_cap: int = 25,
) -> list[ReleasePayload]:
    conn = _connect_read_only(metadata_db_path)
    try:
        rows = _load_track_rows(conn, artist=artist, album=album)
        releases = _rows_to_releases(conn, rows, track_title_cap=track_title_cap)
    finally:
        conn.close()

    filtered: list[ReleasePayload] = []
    for release in releases:
        if generic_only and not is_generic_only_release(release):
            continue
        if min_existing_specific_genres is not None:
            if count_specific_existing_genres(release) >= min_existing_specific_genres:
                continue
        filtered.append(release)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def compute_input_hash(
    payload: ReleasePayload,
    prompt_version: str,
    taxonomy_version: str,
    *,
    web_mode: str = "off",
    source_evidence_hash: str = "none",
    response_schema_version: str = "ai-genre-response-v1",
) -> str:
    normalized = {
        "normalized_artist": payload.normalized_artist,
        "normalized_album": payload.normalized_album,
        "album_id": payload.album_id,
        "identifiers": dict(sorted(payload.identifiers.items())),
        "track_titles": sorted({title.strip() for title in payload.track_titles if title.strip()}),
        "existing_genres_by_source": {
            source: sorted({genre.strip() for genre in genres if genre.strip()})
            for source, genres in sorted(payload.existing_genres_by_source.items())
        },
        "prompt_version": prompt_version,
        "taxonomy_version": taxonomy_version,
        "web_mode": web_mode,
        "source_evidence_hash": source_evidence_hash,
        "response_schema_version": response_schema_version,
    }
    blob = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_generic_only_release(payload: ReleasePayload) -> bool:
    genres = list(_iter_existing_genres(payload))
    if not genres:
        return True
    return all(_genre_key(genre) in GENERIC_OR_DESCRIPTOR_TAGS for genre in genres)


def count_specific_existing_genres(payload: ReleasePayload) -> int:
    return len({_genre_key(genre) for genre in _iter_existing_genres(payload) if _genre_key(genre) not in GENERIC_OR_DESCRIPTOR_TAGS})


def _connect_read_only(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).resolve()
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)).fetchone()
        is not None
    )


def _first_existing(columns: set[str], names: Iterable[str]) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def _load_track_rows(conn: sqlite3.Connection, *, artist: str | None, album: str | None) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "tracks")
    if not columns:
        return []
    select_cols = [
        col
        for col in [
            "track_id",
            "artist",
            "album_artist",
            "title",
            "album",
            "album_id",
            "year",
            "date",
            "musicbrainz_release_mbid",
            "release_mbid",
            "mbid",
            "discogs_release_id",
            "discogs_id",
        ]
        if col in columns
    ]
    if not {"artist", "album", "title"} <= set(select_cols):
        return []

    clauses = ["album IS NOT NULL", "TRIM(album) != ''", "artist IS NOT NULL", "TRIM(artist) != ''"]
    params: list[Any] = []
    if artist:
        clauses.append("LOWER(artist) LIKE LOWER(?)")
        params.append(f"%{artist}%")
    if album:
        clauses.append("LOWER(album) LIKE LOWER(?)")
        params.append(f"%{album}%")

    sql = f"SELECT rowid AS _rowid, {', '.join(select_cols)} FROM tracks WHERE {' AND '.join(clauses)} ORDER BY artist, album, rowid"
    return [dict(row) for row in conn.execute(sql, params)]


def _rows_to_releases(conn: sqlite3.Connection, rows: list[dict[str, Any]], *, track_title_cap: int) -> list[ReleasePayload]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = make_release_key(row.get("album_artist") or row.get("artist"), row.get("album"))
        if key != "::":
            grouped[key].append(row)

    artist_genres = _load_artist_genres(conn)
    album_genres = _load_album_genres(conn)
    track_genres = _load_track_genres(conn)
    releases: list[ReleasePayload] = []
    for release_key in sorted(grouped):
        group = grouped[release_key]
        first = group[0]
        raw_artist = str(first.get("album_artist") or first.get("artist") or "")
        raw_album = str(first.get("album") or "")
        album_id = _most_common(row.get("album_id") for row in group)
        track_ids = [str(row.get("track_id")) for row in group if row.get("track_id")]
        track_titles = [str(row.get("title")) for row in group if row.get("title")]
        year = _first_year(group)
        identifiers = _collect_identifiers(group)
        existing_by_source: dict[str, list[str]] = defaultdict(list)

        normalized_artist = normalize_release_artist(raw_artist)
        normalized_album = normalize_release_name(raw_album)
        for source, genres in artist_genres.get(normalized_artist, {}).items():
            existing_by_source[f"artist:{source}"].extend(genres)
        if album_id:
            for source, genres in album_genres.get(album_id, {}).items():
                existing_by_source[f"album:{source}"].extend(genres)
        for track_id in track_ids:
            for source, genres in track_genres.get(track_id, {}).items():
                existing_by_source[f"track:{source}"].extend(genres)

        compact_existing = {
            source: sorted({genre for genre in genres if genre and genre != "__EMPTY__"})
            for source, genres in sorted(existing_by_source.items())
            if any(genre and genre != "__EMPTY__" for genre in genres)
        }
        genre_counts = Counter(_genre_key(genre) for genres in compact_existing.values() for genre in genres)
        releases.append(
            ReleasePayload(
                artist=raw_artist,
                album=raw_album,
                normalized_artist=normalized_artist,
                normalized_album=normalized_album,
                release_key=release_key,
                album_id=album_id,
                identifiers=identifiers,
                year=year,
                track_titles=track_titles[:track_title_cap],
                existing_genres_by_source=compact_existing,
                genre_counts=dict(sorted(genre_counts.items())),
            )
        )
    return releases


def _load_artist_genres(conn: sqlite3.Connection) -> dict[str, dict[str, list[str]]]:
    if not _table_exists(conn, "artist_genres"):
        return {}
    columns = _table_columns(conn, "artist_genres")
    artist_col = _first_existing(columns, ["artist", "artist_name"])
    if not artist_col or "genre" not in columns:
        return {}
    source_col = "source" if "source" in columns else None
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    source_expr = source_col or "'unknown'"
    for row in conn.execute(f"SELECT {artist_col} AS artist, genre, {source_expr} AS source FROM artist_genres"):
        result[normalize_release_artist(row["artist"])][row["source"]].append(row["genre"])
    return result


def _load_album_genres(conn: sqlite3.Connection) -> dict[str, dict[str, list[str]]]:
    if not _table_exists(conn, "album_genres"):
        return {}
    columns = _table_columns(conn, "album_genres")
    album_id_col = _first_existing(columns, ["album_id", "album_name"])
    if not album_id_col or "genre" not in columns:
        return {}
    source_col = "source" if "source" in columns else None
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    source_expr = source_col or "'unknown'"
    for row in conn.execute(f"SELECT {album_id_col} AS album_id, genre, {source_expr} AS source FROM album_genres"):
        result[str(row["album_id"])][row["source"]].append(row["genre"])
    return result


def _load_track_genres(conn: sqlite3.Connection) -> dict[str, dict[str, list[str]]]:
    if not _table_exists(conn, "track_genres"):
        return {}
    columns = _table_columns(conn, "track_genres")
    if "track_id" not in columns or "genre" not in columns:
        return {}
    source_col = "source" if "source" in columns else None
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    source_expr = source_col or "'unknown'"
    for row in conn.execute(f"SELECT track_id, genre, {source_expr} AS source FROM track_genres"):
        result[str(row["track_id"])][row["source"]].append(row["genre"])
    return result


def _most_common(values: Iterable[Any]) -> str | None:
    counter = Counter(str(value) for value in values if value)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _first_year(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        raw = row.get("year") or row.get("date")
        if raw is None:
            continue
        text = str(raw)
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
    return None


def _collect_identifiers(rows: list[dict[str, Any]]) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for row in rows:
        for key in [
            "musicbrainz_release_mbid",
            "release_mbid",
            "mbid",
            "discogs_release_id",
            "discogs_id",
        ]:
            value = row.get(key)
            if value and key not in identifiers:
                identifiers[key] = str(value)
    return dict(sorted(identifiers.items()))


def _iter_existing_genres(payload: ReleasePayload) -> Iterable[str]:
    for genres in payload.existing_genres_by_source.values():
        yield from genres


def _genre_key(genre: str) -> str:
    return normalize_genre_token(genre) or normalize_release_name(genre)
