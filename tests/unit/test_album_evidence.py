from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.album_evidence import build_evidence


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "m.db")
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO tracks VALUES ('t1','a1','Alison');
        INSERT INTO album_genres VALUES ('a1','shoegaze','discogs_release');
        INSERT INTO track_genres VALUES ('t1','dream pop');
        """
    )
    conn.commit()
    return conn


def test_build_evidence_collects_sources_titles_and_observed(tmp_path):
    conn = _db(tmp_path)
    ev = build_evidence(conn, "a1", id2name={})
    assert ev["artist"] == "Slowdive"
    assert ev["album"] == "Souvlaki"
    assert ev["year"] == 1993
    assert ev["track_titles"] == ["Alison"]
    assert ev["file_tags"] == ["dream pop"]
    assert ev["existing_genres_by_source"]["discogs_release"] == ["shoegaze"]
    assert ev["existing_genres_by_source"]["file_track"] == ["dream pop"]
    assert ev["current_observed_leaf"] == []
