"""Read API for the published unified genre store in metadata.db.

Single import point for playlist features (SP2+). All reads come from the
materialized release_effective_genres table; taxonomy-structure helpers delegate
to the loaded LayeredTaxonomy.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GenreRow:
    genre_id: str
    assignment_layer: str
    confidence: float
    source: str


def resolved_genres_for_album(conn: sqlite3.Connection, album_id: str) -> list[GenreRow]:
    rows = conn.execute(
        "SELECT genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres WHERE album_id = ? "
        "ORDER BY assignment_layer, genre_id",
        (album_id,),
    ).fetchall()
    return [GenreRow(r[0], r[1], r[2], r[3]) for r in rows]


def resolved_genres_for_track(conn: sqlite3.Connection, track_id: str) -> list[GenreRow]:
    row = conn.execute(
        "SELECT album_id FROM tracks WHERE track_id = ?", (track_id,)
    ).fetchone()
    if not row or not row[0]:
        return []
    return resolved_genres_for_album(conn, row[0])


def resolved_genres_by_album(conn: sqlite3.Connection) -> dict[str, list[GenreRow]]:
    """All published genres, batched per album (one query, no N+1)."""
    by_album: dict[str, list[GenreRow]] = {}
    rows = conn.execute(
        "SELECT album_id, genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres "
        "ORDER BY album_id, assignment_layer, genre_id"
    ).fetchall()
    for album_id, genre_id, layer, confidence, source in rows:
        by_album.setdefault(album_id, []).append(
            GenreRow(genre_id, layer, confidence, source)
        )
    return by_album


def canonical_genre_names(conn: sqlite3.Connection) -> dict[str, str]:
    """genre_id -> display name from the published taxonomy copy."""
    return dict(
        conn.execute(
            "SELECT genre_id, name FROM genre_graph_canonical_genres"
        ).fetchall()
    )


def display_genre_names_for_track(conn: sqlite3.Connection, track_id: str) -> list[str]:
    """Published genres for a track's album as display names, deduped.

    The read for GUI display paths (chips, search results, staged seeds):
    observed + inferred layers, genre_id mapped to the canonical display name
    (unmapped ids pass through as-is). Returns [] when the track's album is
    unpublished or the authority tables are absent — display callers fall back
    to other sources, they don't crash.
    """
    try:
        rows = conn.execute(
            "SELECT reg.genre_id, COALESCE(g.name, reg.genre_id) "
            "FROM release_effective_genres reg "
            "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
            "WHERE reg.album_id = (SELECT album_id FROM tracks WHERE track_id = ?) "
            "ORDER BY reg.assignment_layer, reg.genre_id",
            (str(track_id),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for genre_id, name in rows:
        if genre_id not in seen:
            seen.add(genre_id)
            out.append(name)
    return out


def genre_source_for_album(conn: sqlite3.Connection, album_id: str) -> str:
    base = conn.execute(
        "SELECT source FROM release_effective_genres "
        "WHERE album_id = ? AND source != 'user' LIMIT 1",
        (album_id,),
    ).fetchone()
    return base[0] if base else "none"


@lru_cache(maxsize=1)
def _taxonomy():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    return load_default_layered_taxonomy()


def parents_for(conn: sqlite3.Connection, genre_id: str) -> list[str]:
    return [g.genre_id for g in _taxonomy().parents_for_genre(genre_id)]


def families_for(conn: sqlite3.Connection, genre_id: str) -> list[str]:
    return [g.genre_id for g in _taxonomy().families_for_genre(genre_id)]


def is_facet(conn: sqlite3.Connection, genre_id: str) -> bool:
    return _taxonomy().facet_by_id(genre_id) is not None
