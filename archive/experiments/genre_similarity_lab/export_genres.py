"""
Export distinct genres by API source (MusicBrainz, Last.FM, Discogs) to CSV.

Usage:
    python -m experiments.genre_similarity_lab.export_genres \
        --db-path experiments/genre_similarity_lab/metadata_lab.db \
        --out-path experiments/genre_similarity_lab/artifacts/genre_catalog_by_api.csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Dict, List


def collect_genres(conn: sqlite3.Connection) -> List[Dict[str, str]]:
    """Collect distinct genres grouped by API family."""
    source_map = {
        "musicbrainz": {"artist_genres": ["musicbrainz_artist"], "album_genres": ["musicbrainz_release"]},
        "lastfm": {
            "artist_genres": ["lastfm_artist"],
            "album_genres": ["lastfm_album"],
            "track_genres": ["lastfm_track", "lastfm"],
        },
        "discogs": {"album_genres": ["discogs_release", "discogs_master"]},
    }

    api_genres: Dict[str, set] = {api: set() for api in source_map}
    cur = conn.cursor()

    for api, tables in source_map.items():
        for table, sources in tables.items():
            for src in sources:
                cur.execute(
                    f"SELECT DISTINCT genre FROM {table} WHERE source = ? AND genre != '__EMPTY__'",
                    (src,),
                )
                api_genres[api].update(row[0] for row in cur.fetchall() if row[0])

    rows: List[Dict[str, str]] = []
    for api, genres in api_genres.items():
        for g in sorted(genres):
            rows.append({"api": api, "genre": g})

    rows.sort(key=lambda r: (r["api"], r["genre"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export distinct genres by API to CSV.")
    parser.add_argument(
        "--db-path",
        default="experiments/genre_similarity_lab/metadata_lab.db",
        help="Path to SQLite DB (default: experiments/genre_similarity_lab/metadata_lab.db)",
    )
    parser.add_argument(
        "--out-path",
        default="experiments/genre_similarity_lab/artifacts/genre_catalog_by_api.csv",
        help="Output CSV path (default: experiments/genre_similarity_lab/artifacts/genre_catalog_by_api.csv)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    rows = collect_genres(conn)
    conn.close()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["api", "genre"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
