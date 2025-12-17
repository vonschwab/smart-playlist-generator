"""
Export artists that have Last.FM genres but no MusicBrainz genres.

Usage:
    python -m experiments.genre_similarity_lab.export_lastfm_only_artists \
        --db-path experiments/genre_similarity_lab/metadata_lab.db \
        --out-path experiments/genre_similarity_lab/artifacts/lastfm_only_artists.csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Artists with Last.FM tags but no MusicBrainz tags.")
    parser.add_argument(
        "--db-path",
        default="experiments/genre_similarity_lab/metadata_lab.db",
        help="Path to SQLite DB (default: experiments/genre_similarity_lab/metadata_lab.db)",
    )
    parser.add_argument(
        "--out-path",
        default="experiments/genre_similarity_lab/artifacts/lastfm_only_artists.csv",
        help="Output CSV path (default: experiments/genre_similarity_lab/artifacts/lastfm_only_artists.csv)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
    SELECT
        ag.artist,
        GROUP_CONCAT(DISTINCT CASE
            WHEN ag2.source = 'lastfm_artist' AND ag2.genre != '__EMPTY__' THEN ag2.genre
        END) AS lastfm_genres
    FROM artist_genres ag
    JOIN artist_genres ag2 ON ag.artist = ag2.artist
    WHERE ag.source = 'lastfm_artist'
    GROUP BY ag.artist
    HAVING
        SUM(CASE WHEN ag2.source = 'lastfm_artist' AND ag2.genre != '__EMPTY__' THEN 1 ELSE 0 END) > 0
        AND SUM(CASE WHEN ag2.source = 'musicbrainz_artist' THEN 1 ELSE 0 END) = 0
    """

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "lastfm_genres"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "artist": row["artist"] or "",
                    "lastfm_genres": row["lastfm_genres"] or "",
                }
            )

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
