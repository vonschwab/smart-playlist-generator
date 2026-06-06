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
