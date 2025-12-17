#!/usr/bin/env python3
"""
Purge Last.FM genre/tag data from the metadata database.

Removes:
- artist_genres rows where source starts with 'lastfm'
- album_genres rows where source starts with 'lastfm'
- track_genres rows where source starts with 'lastfm'
- artists.lastfm_tags and artists.similar_artists columns (table is rebuilt without them)
"""
import argparse
import sqlite3
from pathlib import Path


def purge_lastfm(conn: sqlite3.Connection) -> dict:
    """Remove Last.FM-sourced data and rebuild artists table without Last.FM columns."""
    cursor = conn.cursor()
    results = {}

    for table in ("artist_genres", "album_genres"):
        cursor.execute(f"DELETE FROM {table} WHERE source LIKE 'lastfm%'")
        results[f"{table}_deleted"] = cursor.rowcount

    cursor.execute("DELETE FROM track_genres WHERE source LIKE 'lastfm%'")
    results["track_genres_deleted"] = cursor.rowcount

    # Rebuild artists table to drop lastfm_tags/similar_artists columns if present.
    cursor.execute("PRAGMA table_info(artists)")
    cols = [row[1] for row in cursor.fetchall()]
    if "lastfm_tags" in cols or "similar_artists" in cols:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artists_new (
                artist_name TEXT PRIMARY KEY,
                musicbrainz_id TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO artists_new (artist_name, musicbrainz_id, last_updated)
            SELECT artist_name, musicbrainz_id, last_updated FROM artists
        """)
        cursor.execute("DROP TABLE artists")
        cursor.execute("ALTER TABLE artists_new RENAME TO artists")
        results["artists_rebuilt"] = True
    else:
        results["artists_rebuilt"] = False

    conn.commit()
    return results


def main():
    parser = argparse.ArgumentParser(description="Purge Last.FM genre/tag data from metadata DB.")
    parser.add_argument("--db-path", type=Path, default=Path("data") / "metadata.db",
                        help="Path to metadata database (default: data/metadata.db)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        totals = purge_lastfm(conn)
    finally:
        conn.close()

    print("Last.FM purge complete:")
    for k, v in totals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
