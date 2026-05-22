import logging
import sqlite3
from pathlib import Path

import scripts.fetch_mbids_musicbrainz as mbid
import scripts.update_discogs_genres as discogs
from scripts.update_genres_v3_normalized import (
    STATUS_NO_MATCH,
    STATUS_OK,
    STATUS_UNKNOWN,
    ensure_enrichment_status_schema,
    get_enrichment_status,
    set_enrichment_status,
)


def _make_temp_db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    return conn


def test_mbid_markers_update_and_skip(tmp_path):
    conn = _make_temp_db(tmp_path)
    conn.execute(
        "CREATE TABLE tracks (track_id TEXT PRIMARY KEY, title TEXT, artist TEXT, duration_ms INTEGER, musicbrainz_id TEXT)"
    )
    conn.commit()
    mbid.ensure_mbid_status_columns(conn)

    conn.execute(
        "INSERT INTO tracks (track_id, title, artist, duration_ms, musicbrainz_id) VALUES (?,?,?,?,?)",
        ("t1", "Song A", "Artist A", 1000, None),
    )
    conn.execute(
        "INSERT INTO tracks (track_id, title, artist, duration_ms, musicbrainz_id) VALUES (?,?,?,?,?)",
        ("t2", "Song B", "Artist B", 1000, mbid.NO_MATCH_MARKER),
    )
    conn.commit()

    candidates = mbid.load_candidates(
        conn,
        limit=10,
        force=False,
        force_no_match=False,
        force_error=False,
        force_reject=False,
        force_all=False,
        artist_like=None,
    )
    # Only unknown/failed should be processed (t1)
    assert [c[0] for c in candidates] == ["t1"]

    mbid._update_mbid_status(conn, "t1", "mbid-123", mbid.MBID_STATUS_OK, last_error=None)
    row = conn.execute(
        "SELECT musicbrainz_id, mbid_status, mbid_attempt_count, mbid_last_error FROM tracks WHERE track_id='t1'"
    ).fetchone()
    assert row[0] == "mbid-123"
    assert row[1] == mbid.MBID_STATUS_OK
    assert row[2] == 1
    assert row[3] is None

    # Subsequent scan should skip once status is ok
    assert (
        mbid.load_candidates(
            conn,
            limit=10,
            force=False,
            force_no_match=False,
            force_error=False,
            force_reject=False,
            force_all=False,
            artist_like=None,
        )
        == []
    )


def test_discogs_markers_increment(tmp_path):
    conn = _make_temp_db(tmp_path)
    conn.execute("CREATE TABLE albums (album_id TEXT PRIMARY KEY, artist TEXT, title TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.commit()
    discogs.ensure_discogs_marker_columns(conn)

    conn.execute("INSERT INTO albums (album_id, artist, title) VALUES (?,?,?)", ("a1", "Artist", "Album"))
    conn.commit()

    assert discogs.get_discogs_marker(conn, "a1") == discogs.DISCOGS_STATUS_UNKNOWN
    discogs.set_discogs_marker(conn, "a1", discogs.DISCOGS_STATUS_NO_MATCH, last_error=None)
    row = conn.execute(
        "SELECT discogs_status, discogs_attempt_count, discogs_last_error FROM albums WHERE album_id='a1'"
    ).fetchone()
    assert row[0] == discogs.DISCOGS_STATUS_NO_MATCH
    assert row[1] == 1
    assert row[2] is None

    discogs.set_discogs_marker(conn, "a1", discogs.DISCOGS_STATUS_FAILED, last_error="timeout")
    row = conn.execute(
        "SELECT discogs_status, discogs_attempt_count, discogs_last_error FROM albums WHERE album_id='a1'"
    ).fetchone()
    assert row[0] == discogs.DISCOGS_STATUS_FAILED
    assert row[1] == 2
    assert row[2] == "timeout"


def test_enrichment_status_table(tmp_path):
    conn = _make_temp_db(tmp_path)
    ensure_enrichment_status_schema(conn, logger=logging.getLogger("test"))
    assert get_enrichment_status(conn, "artist", "key1") == STATUS_UNKNOWN
    set_enrichment_status(conn, "artist", "key1", STATUS_NO_MATCH, last_error=None)
    assert get_enrichment_status(conn, "artist", "key1") == STATUS_NO_MATCH
    set_enrichment_status(conn, "artist", "key1", STATUS_OK, last_error=None)
    row = conn.execute(
        "SELECT status, attempt_count, last_error FROM enrichment_status WHERE entity_type='artist' AND entity_id='key1'"
    ).fetchone()
    assert row[0] == STATUS_OK
    assert row[1] == 2
    assert row[2] is None
