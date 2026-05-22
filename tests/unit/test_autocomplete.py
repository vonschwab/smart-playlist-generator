import sqlite3

from src.playlist_gui.autocomplete import DatabaseCompleter


def _create_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT,
            title TEXT,
            artist TEXT,
            album TEXT,
            artist_key TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE track_effective_genres (
            track_id TEXT,
            genre TEXT,
            source TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?)",
        [
            ("deerhunter-1", "Nothing Ever Happened", "Deerhunter", "Microcastle", "deerhunter"),
            ("deerhoof-1", "The Perfect Me", "Deerhoof", "Friend Opportunity", "deerhoof"),
            ("atlas-1", "Walkabout", "Atlas Sound", "Logos", "atlas sound"),
        ],
    )
    conn.commit()
    conn.close()


def test_filter_tracks_matches_artist_first_seed_query(tmp_path):
    db_path = tmp_path / "metadata.db"
    _create_db(db_path)
    completer = DatabaseCompleter(str(db_path))
    assert completer.load_data()

    matches = completer.filter_tracks("Deerhunter Nothing Ever Happened")

    assert matches[0] == "Nothing Ever Happened - Deerhunter (Microcastle)"


def test_filter_tracks_requires_all_query_tokens(tmp_path):
    db_path = tmp_path / "metadata.db"
    _create_db(db_path)
    completer = DatabaseCompleter(str(db_path))
    assert completer.load_data()

    matches = completer.filter_tracks("Deerhunter Walkabout")

    assert matches == []


def test_filter_tracks_limits_broad_completion_results(tmp_path):
    db_path = tmp_path / "metadata.db"
    _create_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?)",
        [
            (
                f"bulk-{i}",
                f"Nothing Song {i:03d}",
                "Bulk Artist",
                "Bulk Album",
                "bulk artist",
            )
            for i in range(300)
        ],
    )
    conn.commit()
    conn.close()

    completer = DatabaseCompleter(str(db_path))
    assert completer.load_data()

    matches = completer.filter_tracks("Nothing", limit=25)

    assert len(matches) == 25
