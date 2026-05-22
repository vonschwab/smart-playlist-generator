import sqlite3

from src.metadata_client import MetadataClient


def _fetch_manual_track_blacklist_ids(metadata):
    cursor = metadata.conn.cursor()
    cursor.execute("SELECT track_id FROM track_blacklist")
    return {str(row["track_id"]) for row in cursor.fetchall()}


def _create_legacy_blacklist_tables(conn):
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            musicbrainz_id TEXT,
            title TEXT,
            artist TEXT,
            artist_key TEXT,
            album TEXT,
            file_path TEXT,
            duration_ms INTEGER,
            is_blacklisted INTEGER NOT NULL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE artist_blacklist (
            artist_key TEXT PRIMARY KEY,
            artist_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE album_blacklist (
            artist_key TEXT,
            album_key TEXT,
            artist_name TEXT,
            album_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (artist_key, album_key)
        )
        """
    )


def test_artist_blacklist_marks_existing_and_new_tracks(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Brian Eno", "Ambient 1", 100)
    metadata.add_track("2", "B", "Brian Eno", "Another Green World", 100)
    metadata.add_track("3", "C", "Roxy Music", "For Your Pleasure", 100)

    updated = metadata.set_artist_blacklisted("Brian Eno", True)

    assert updated == 2
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2"}

    metadata.add_track("4", "D", "brian eno", "Before and After Science", 100)
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2", "4"}


def test_album_blacklist_marks_matching_artist_album_only(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "Chairs Missing", 100)
    metadata.add_track("3", "C", "Other Wire", "Chairs Missing", 100)

    updated = metadata.set_album_blacklisted("Wire", "Chairs Missing", True)

    assert updated == 2
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2"}

    metadata.set_album_blacklisted("Wire", "Chairs Missing", False)
    assert metadata.fetch_blacklisted_track_ids() == set()


def test_artist_scope_removal_preserves_manual_track_blacklist(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "154", 100)

    metadata.set_blacklisted(["1"], True)
    metadata.set_artist_blacklisted("Wire", True)
    metadata.set_artist_blacklisted("Wire", False)

    assert metadata.fetch_blacklisted_track_ids() == {"1"}


def test_album_scope_removal_preserves_manual_track_blacklist(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "Chairs Missing", 100)

    metadata.set_blacklisted(["1"], True)
    metadata.set_album_blacklisted("Wire", "Chairs Missing", True)
    metadata.set_album_blacklisted("Wire", "Chairs Missing", False)

    assert metadata.fetch_blacklisted_track_ids() == {"1"}


def test_manual_unblacklist_under_active_artist_scope_waits_for_scope_removal(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)

    metadata.set_blacklisted(["1"], True)
    metadata.set_artist_blacklisted("Wire", True)
    metadata.set_blacklisted(["1"], False)

    assert metadata.fetch_blacklisted_track_ids() == {"1"}

    metadata.set_artist_blacklisted("Wire", False)
    assert metadata.fetch_blacklisted_track_ids() == set()


def test_legacy_blacklisted_track_overlapping_artist_scope_backfills_as_manual(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as conn:
        _create_legacy_blacklist_tables(conn)
        conn.execute(
            """
            INSERT INTO tracks
            (track_id, title, artist, artist_key, album, duration_ms, is_blacklisted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("1", "A", "Wire", "wire", "Chairs Missing", 100, 1),
        )
        conn.execute(
            """
            INSERT INTO artist_blacklist (artist_key, artist_name)
            VALUES (?, ?)
            """,
            ("wire", "Wire"),
        )

    metadata = MetadataClient(str(db_path))
    assert _fetch_manual_track_blacklist_ids(metadata) == {"1"}

    metadata.set_artist_blacklisted("Wire", False)
    assert metadata.fetch_blacklisted_track_ids() == {"1"}


def test_partial_track_blacklist_backfills_missing_legacy_blacklisted_tracks(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as conn:
        _create_legacy_blacklist_tables(conn)
        conn.executemany(
            """
            INSERT INTO tracks
            (track_id, title, artist, artist_key, album, duration_ms, is_blacklisted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("1", "A", "Wire", "wire", "Chairs Missing", 100, 1),
                ("2", "B", "Wire", "wire", "154", 100, 1),
            ],
        )
        conn.execute(
            """
            CREATE TABLE track_blacklist (
                track_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO track_blacklist (track_id, title, artist, album)
            VALUES (?, ?, ?, ?)
            """,
            ("1", "A", "Wire", "Chairs Missing"),
        )

    metadata = MetadataClient(str(db_path))

    assert _fetch_manual_track_blacklist_ids(metadata) == {"1", "2"}


def test_scope_only_artist_blacklist_does_not_become_manual_after_restart(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "154", 100)
    metadata.set_artist_blacklisted("Wire", True)
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2"}
    metadata.close()

    restarted = MetadataClient(str(db_path))
    restarted.set_artist_blacklisted("Wire", False)

    assert restarted.fetch_blacklisted_track_ids() == set()
