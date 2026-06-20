"""Per-album evidence assembly for the album adjudicator.

Promoted from scripts/research/run_album_adjudicator.py so the analyze pipeline stages
can import it. Reads only metadata.db tables (albums, tracks, album_genres, track_genres,
release_effective_genres).
"""
from __future__ import annotations

import sqlite3


def build_evidence(conn: sqlite3.Connection, album_id: str, id2name: dict[str, str]) -> dict:
    row = conn.execute(
        "SELECT artist, title, release_year, musicbrainz_release_id FROM albums WHERE album_id=?",
        (album_id,),
    ).fetchone()
    artist, title, year, mbid = row if row else (None, None, None, None)
    tracks = [r[0] for r in conn.execute(
        "SELECT title FROM tracks WHERE album_id=? AND title IS NOT NULL LIMIT 8", (album_id,)
    )]
    by_source: dict[str, list[str]] = {}
    for genre, source in conn.execute(
        "SELECT DISTINCT genre, source FROM album_genres WHERE album_id=? AND genre NOT IN ('__EMPTY__','__empty__')",
        (album_id,),
    ):
        by_source.setdefault((source or "album").lower(), []).append(genre)
    trk = [g for (g,) in conn.execute(
        "SELECT DISTINCT tg.genre FROM track_genres tg JOIN tracks t ON t.track_id=tg.track_id "
        "WHERE t.album_id=? AND tg.genre NOT IN ('__EMPTY__','__empty__')",
        (album_id,),
    )]
    if trk:
        by_source["file_track"] = sorted(set(trk))
    file_tags = sorted(set(trk))
    try:
        observed = [
            id2name[g] for (g,) in conn.execute(
                "SELECT genre_id FROM release_effective_genres WHERE album_id=? AND assignment_layer='observed_leaf'",
                (album_id,),
            ) if g in id2name
        ]
    except sqlite3.OperationalError:
        observed = []
    identifiers = {"mbid": mbid} if mbid else {}
    return {
        "artist": artist, "album": title, "album_id": album_id, "year": year,
        "identifiers": identifiers, "track_titles": tracks, "file_tags": file_tags,
        "existing_genres_by_source": by_source, "current_observed_leaf": observed,
    }
