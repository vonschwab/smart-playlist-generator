"""
Export albums/releases that have Last.FM genres but no Discogs or MusicBrainz genres.

Usage:
    python -m experiments.genre_similarity_lab.export_lastfm_only_albums \
        --db-path experiments/genre_similarity_lab/metadata_lab.db \
        --out-path experiments/genre_similarity_lab/artifacts/lastfm_only_albums.csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Albums with Last.FM tags but no Discogs/MusicBrainz tags (with tags shown)."
    )
    parser.add_argument(
        "--db-path",
        default="experiments/genre_similarity_lab/metadata_lab.db",
        help="Path to SQLite DB (default: experiments/genre_similarity_lab/metadata_lab.db)",
    )
    parser.add_argument(
        "--out-path",
        default="experiments/genre_similarity_lab/artifacts/lastfm_only_albums.csv",
        help="Output CSV path (default: experiments/genre_similarity_lab/artifacts/lastfm_only_albums.csv)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Albums that have lastfm_album genres and zero discogs/musicbrainz genres
    query = """
    SELECT
        ag.album_id,
        MIN(t.artist) AS artist,
        MIN(t.album) AS album,
        GROUP_CONCAT(DISTINCT CASE
            WHEN ag2.source = 'lastfm_album' AND ag2.genre != '__EMPTY__' THEN ag2.genre
        END) AS lastfm_genres
    FROM album_genres ag
    JOIN album_genres ag2 ON ag.album_id = ag2.album_id
    LEFT JOIN tracks t ON t.album_id = ag.album_id
    WHERE ag.source = 'lastfm_album'
    GROUP BY ag.album_id
    HAVING
        SUM(CASE WHEN ag2.source = 'lastfm_album' AND ag2.genre != '__EMPTY__' THEN 1 ELSE 0 END) > 0
        AND SUM(CASE WHEN ag2.source IN ('musicbrainz_release','discogs_release','discogs_master') THEN 1 ELSE 0 END) = 0
    """

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["album_id", "artist", "album", "lastfm_genres"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "album_id": row["album_id"],
                    "artist": row["artist"] or "",
                    "album": row["album"] or "",
                    "lastfm_genres": row["lastfm_genres"] or "",
                }
            )

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
