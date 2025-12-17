"""
Export Discogs genres with their occurrence counts (release/master) to CSV.

Includes only genres that appear more than once across Discogs sources.

Usage:
    python -m experiments.genre_similarity_lab.export_discogs_genre_counts \
        --db-path experiments/genre_similarity_lab/metadata_lab.db \
        --out-path experiments/genre_similarity_lab/artifacts/discogs_genre_counts.csv
"""

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable


def collect_discogs_genres(conn: sqlite3.Connection) -> Counter:
    """Aggregate Discogs genres from album_genres table."""
    source_map: Dict[str, Iterable[str]] = {
        "album_genres": ["discogs_release", "discogs_master"],
    }

    cur = conn.cursor()
    counts = Counter()

    for table, sources in source_map.items():
        for src in sources:
            cur.execute(
                f"SELECT genre FROM {table} WHERE source = ? AND genre != '__EMPTY__'",
                (src,),
            )
            for (genre,) in cur.fetchall():
                if genre:
                    counts[genre] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Export Discogs genre counts (>1) to CSV.")
    parser.add_argument(
        "--db-path",
        default="experiments/genre_similarity_lab/metadata_lab.db",
        help="Path to SQLite DB (default: experiments/genre_similarity_lab/metadata_lab.db)",
    )
    parser.add_argument(
        "--out-path",
        default="experiments/genre_similarity_lab/artifacts/discogs_genre_counts.csv",
        help="Output CSV path (default: experiments/genre_similarity_lab/artifacts/discogs_genre_counts.csv)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    counts = collect_discogs_genres(conn)
    conn.close()

    rows = [{"genre": g, "count": c} for g, c in counts.items() if c > 1]
    rows.sort(key=lambda r: (-r["count"], r["genre"]))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
