"""
Purge all Last.FM data from the experimental DB (genres + artist tags).

1) Make a backup first:
   Copy-Item experiments/genre_similarity_lab/metadata_lab.db experiments/genre_similarity_lab/metadata_lab_before_lastfm_purge.db

2) Run:
   python -m experiments.genre_similarity_lab.purge_lastfm --db-path experiments/genre_similarity_lab/metadata_lab.db
"""

import argparse
import sqlite3


def purge_lastfm(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    ops = [
        ("artist_genres", "DELETE FROM artist_genres WHERE source = 'lastfm_artist'"),
        ("album_genres", "DELETE FROM album_genres WHERE source = 'lastfm_album'"),
        ("track_genres", "DELETE FROM track_genres WHERE source IN ('lastfm_track','lastfm')"),
        ("artists.lastfm_tags", "UPDATE artists SET lastfm_tags = NULL WHERE lastfm_tags IS NOT NULL"),
    ]
    totals = {}
    for name, sql in ops:
        cur.execute(sql)
        totals[name] = cur.rowcount
    conn.commit()
    return totals


def main():
    parser = argparse.ArgumentParser(description="Purge all Last.FM data from an experimental DB.")
    parser.add_argument(
        "--db-path",
        default="experiments/genre_similarity_lab/metadata_lab.db",
        help="Path to SQLite DB (default: experiments/genre_similarity_lab/metadata_lab.db)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    totals = purge_lastfm(conn)
    conn.execute("VACUUM")
    conn.close()

    print("Deleted rows/updates:")
    for k, v in totals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
