"""
Quick diagnostic: list every distinct track-artist string in the DB that
contains a given base artist substring.

Tells us what `is_collaboration_of` will see when matching collaborations.

Usage:
    python tools/inspect_artist_collabs.py "Greg Foat"
"""
from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.playlist.history_analyzer import is_collaboration_of  # noqa: E402
from src.string_utils import normalize_artist_key  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "metadata.db"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_artist_collabs.py \"Artist Name\"")
        sys.exit(2)

    base_artist = sys.argv[1].strip()
    if not base_artist:
        print("Empty artist name.")
        sys.exit(2)

    base_key = normalize_artist_key(base_artist)
    base_lower = base_artist.lower()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT artist, COUNT(*) AS n
        FROM tracks
        WHERE LOWER(artist) LIKE ?
          AND file_path IS NOT NULL
          AND is_blacklisted = 0
        GROUP BY artist
        ORDER BY n DESC, artist ASC
        """,
        (f"%{base_lower}%",),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"No tracks found whose artist contains '{base_artist}'.")
        return

    exact: list[tuple[str, int]] = []
    collabs: list[tuple[str, int]] = []
    no_match: list[tuple[str, int]] = []

    for row in rows:
        artist = row["artist"] or ""
        count = int(row["n"])
        if normalize_artist_key(artist) == base_key:
            exact.append((artist, count))
        elif is_collaboration_of(collaboration_name=artist, base_artist=base_artist):
            collabs.append((artist, count))
        else:
            no_match.append((artist, count))

    def _print(label: str, items: list[tuple[str, int]]) -> None:
        total = sum(c for _, c in items)
        print(f"\n{label}: {len(items)} distinct artist strings, {total} tracks")
        for artist, count in items:
            print(f"  [{count:>4}]  {artist}")

    print(f"Base artist: {base_artist!r}  (normalized key: {base_key!r})")
    _print("EXACT matches (already in seed pool today)", exact)
    _print("COLLAB matches (would join pool when 'Include collaborations' is ON)", collabs)
    _print("Substring matches but is_collaboration_of() returns False", no_match)


if __name__ == "__main__":
    main()
