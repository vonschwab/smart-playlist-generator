"""
Flag low-support genres ("tail") with heuristic suspect reasons.

Source: experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv
Output: experiments/genre_similarity_lab/artifacts/genre_tail_suspects.csv
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Identify suspect tail genres from frequency CSV.")
    parser.add_argument(
        "--freq-path",
        default="experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv",
        help="Input frequency CSV (default: experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv)",
    )
    parser.add_argument(
        "--output",
        default="experiments/genre_similarity_lab/artifacts/genre_tail_suspects.csv",
        help="Output CSV path (default: experiments/genre_similarity_lab/artifacts/genre_tail_suspects.csv)",
    )
    parser.add_argument(
        "--max-albums-for-suspect",
        type=int,
        default=3,
        help="Max album_count to consider as tail (default: 3)",
    )
    return parser.parse_args()


# Heuristic regexes
YEAR_RE = re.compile(r"^\d{4}$")
DECADE_RE = re.compile(r"^'?([0-9]{2})s$", re.IGNORECASE)


def suspect_reason_for(genre: str) -> str:
    """
    Apply heuristics to flag a genre as suspect. First match wins.
    Returns a reason string or 'manual_review' if no heuristic matched.
    """
    g = genre.strip()
    g_lower = g.lower()

    # (a) Year/decade tokens
    if YEAR_RE.match(g) or DECADE_RE.match(g):
        return "year_or_decade_token"

    # (b) Cover/artwork/meta
    if any(substr in g_lower for substr in ["cover", "on cover", "album art", "artwork", "cover art"]):
        return "cover_or_artwork_meta"

    # (c) Too short (length <= 2)
    if len(g) <= 2:
        return "too_short_token"

    # (d) Non-genre phrases
    if any(substr in g_lower for substr in ["soundtrack", "original score", "ost", "favorite", "favourite", "best of"]):
        return "non_genre_phrase"

    return "manual_review"


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    freq_path = Path(args.freq_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with freq_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    tail = [r for r in rows if int(r.get("album_count", 0)) <= args.max_albums_for_suspect]

    suspects: List[Dict[str, str]] = []
    reason_counts: Dict[str, int] = {}
    for r in tail:
        genre = r["source_genre"]
        album_count = int(r.get("album_count", 0))
        track_count = int(r.get("track_count", 0))
        sources = r.get("sources", "")
        reason = suspect_reason_for(genre)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        suspects.append(
            {
                "genre": genre,
                "album_count": album_count,
                "track_count": track_count,
                "sources": sources,
                "suspect_reason": reason,
            }
        )

    # Sort by album_count ascending, then genre alpha
    suspects.sort(key=lambda r: (r["album_count"], r["genre"]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "album_count", "track_count", "sources", "suspect_reason"])
        writer.writeheader()
        writer.writerows(suspects)

    total_tail = len(tail)
    flagged = total_tail - reason_counts.get("manual_review", 0)
    print(f"Total tail genres (album_count <= {args.max_albums_for_suspect}): {total_tail}")
    print(f"Flagged by heuristics: {flagged}")
    print("Counts by suspect_reason:")
    for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {reason}: {count}")

    # Sample per category (up to 5 each)
    print("\nSamples per suspect_reason:")
    for reason in sorted(reason_counts.keys()):
        samples = [r["genre"] for r in suspects if r["suspect_reason"] == reason][:5]
        print(f"  {reason}: {samples}")


if __name__ == "__main__":
    main()
