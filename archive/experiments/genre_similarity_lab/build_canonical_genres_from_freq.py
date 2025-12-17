"""
Build a canonical genre list with tiers from album-level frequency CSV.

Reads experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv
and emits canonical_genres.csv with a tier per genre (core/micro/singleton).
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Build canonical genres with tiers from frequency CSV.")
    parser.add_argument(
        "--freq-path",
        default="experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv",
        help="Input frequency CSV (default: experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv)",
    )
    parser.add_argument(
        "--output",
        default="experiments/genre_similarity_lab/artifacts/canonical_genres.csv",
        help="Output CSV path (default: experiments/genre_similarity_lab/artifacts/canonical_genres.csv)",
    )
    parser.add_argument("--core-min-albums", type=int, default=5, help="Album threshold for core tier (default: 5)")
    parser.add_argument("--micro-min-albums", type=int, default=2, help="Album threshold for micro tier (default: 2)")
    return parser.parse_args()


def assign_tier(album_count: int, core_min: int, micro_min: int) -> str:
    if album_count >= core_min:
        return "core"
    if album_count >= micro_min:
        return "micro"
    if album_count == 1:
        return "singleton"
    return "micro"


def main():
    args = parse_args()
    freq_path = Path(args.freq_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with freq_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    canonical = []
    tier_counts = {"core": 0, "micro": 0, "singleton": 0}
    for row in rows:
        genre = row["source_genre"]
        album_count = int(row.get("album_count", 0))
        track_count = int(row.get("track_count", 0))
        sources = row.get("sources", "")
        tier = assign_tier(album_count, args.core_min_albums, args.micro_min_albums)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        canonical.append(
            {
                "genre": genre,
                "album_count": album_count,
                "track_count": track_count,
                "sources": sources,
                "tier": tier,
            }
        )

    # Sort by tier (core, micro, singleton) then album_count desc
    tier_order = {"core": 0, "micro": 1, "singleton": 2}
    canonical.sort(key=lambda r: (tier_order.get(r["tier"], 1), -r["album_count"]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "album_count", "track_count", "sources", "tier"])
        writer.writeheader()
        writer.writerows(canonical)

    total_genres = len(canonical)
    print(f"Total genres: {total_genres}")
    print(f"Core: {tier_counts.get('core', 0)}")
    print(f"Micro: {tier_counts.get('micro', 0)}")
    print(f"Singleton: {tier_counts.get('singleton', 0)}")

    # Top 10 core
    core_top = [r for r in canonical if r["tier"] == "core"][:10]
    print("\nTop 10 core genres:")
    for r in core_top:
        print(f"  {r['genre']}: {r['album_count']}")

    # Top 10 micro
    micro_top = [r for r in canonical if r["tier"] == "micro"][:10]
    print("\nTop 10 micro genres:")
    for r in micro_top:
        print(f"  {r['genre']}: {r['album_count']}")

    print(f"\nSingleton count: {tier_counts.get('singleton', 0)}")


if __name__ == "__main__":
    main()
