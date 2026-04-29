import sqlite3

from src.playlist_gui.seed_resolver import resolve_track_from_display


def test_resolve_track_from_display_uses_album_to_disambiguate(tmp_path):
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
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
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?)",
        ("wrong", "Song", "Artist", "Original Album", "artist"),
    )
    conn.execute(
        "INSERT INTO tracks VALUES (?, ?, ?, ?, ?)",
        ("right", "Song", "Artist", "Selected Album", "artist"),
    )
    conn.commit()
    conn.close()

    chip = resolve_track_from_display("Song - Artist (Selected Album)", str(db_path))

    assert chip is not None
    assert chip.track_id == "right"
