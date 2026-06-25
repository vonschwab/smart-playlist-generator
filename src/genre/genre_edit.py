"""User genre edit orchestration: resolve terms, locate album, apply override.

Writes the durable add/remove override (ai_genre_user_overrides) AND the
surgical release_effective_genres rows via the shared publish materializer, so
the edit is authoritative immediately and reproduced byte-for-byte by a later
full publish.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ai_genre_enrichment.normalization import (
    make_release_key,
    normalize_release_artist,
    normalize_release_name,
)
from src.genre import genre_publish
from src.genre.authority import canonical_genre_names, resolved_genres_for_album


def resolve_terms_to_genre_ids(taxonomy, names: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map free-typed names to canonical genre_ids. Unresolved names returned."""
    resolved: dict[str, str] = {}
    unknown: list[str] = []
    for name in names:
        term = (name or "").strip()
        if not term:
            continue
        gid = genre_publish._term_to_genre_id(taxonomy, term)
        if gid:
            resolved[term] = gid
        else:
            unknown.append(term)
    return resolved, unknown


def album_id_for_release(conn, artist: str, album: str) -> str | None:
    """Resolve album_id from the tracks table (orphan-safe).

    Exact (artist, album) first; else normalized release_key grouped over
    tracks, picking the album_id with the most tracks (ties: lexicographically
    smallest) for determinism.
    """
    row = conn.execute(
        "SELECT album_id, COUNT(*) c FROM tracks "
        "WHERE artist = ? AND album = ? AND album_id IS NOT NULL AND album_id != '' "
        "GROUP BY album_id ORDER BY c DESC, album_id ASC LIMIT 1",
        (artist, album),
    ).fetchone()
    if row and row[0]:
        return row[0]

    target_key = make_release_key(artist, album)
    counts: dict[str, int] = {}
    for aid, a, alb in conn.execute(
        "SELECT album_id, artist, album FROM tracks "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        if make_release_key(a, alb) == target_key:
            counts[aid] = counts.get(aid, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _album_in_graph(conn, album_id: str) -> set[str]:
    row = conn.execute(
        "SELECT 1 FROM genre_graph_release_genre_assignments WHERE album_id = ? LIMIT 1",
        (album_id,),
    ).fetchone()
    return {album_id} if row else set()
