#!/usr/bin/env python3
"""
Title Dedupe Validation Script

Tests fuzzy title deduplication on your local library to help tune settings.
Shows potential duplicates that would be filtered and helps validate configuration.

Usage:
    python scripts/validate_title_dedupe.py
    python scripts/validate_title_dedupe.py --threshold 90 --mode strict
    python scripts/validate_title_dedupe.py --artist "Radiohead" --limit 100
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import sqlite3
from collections import defaultdict
from typing import List, Dict, Tuple

from src.title_dedupe import (
    TitleDedupeTracker,
    normalize_title_for_dedupe,
    normalize_artist_key,
    title_similarity,
    calculate_version_preference_score,
)
from src.config_loader import Config


def load_tracks_from_db(db_path: str, artist_filter: str = None, limit: int = None) -> List[Dict]:
    """Load tracks from the metadata database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT track_id, title, artist, album FROM tracks WHERE title IS NOT NULL"
    params = []

    if artist_filter:
        query += " AND LOWER(artist) LIKE ?"
        params.append(f"%{artist_filter.lower()}%")

    query += " ORDER BY artist, title"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)
    tracks = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return tracks


def find_potential_duplicates(
    tracks: List[Dict],
    threshold: int = 92,
    mode: str = "loose",
    short_title_min_len: int = 6,
) -> List[Tuple[Dict, Dict, float]]:
    """
    Find all potential duplicate pairs in the track list.

    Returns list of (track1, track2, similarity_score) tuples.
    """
    duplicates = []

    # Group tracks by normalized artist key
    by_artist: Dict[str, List[Dict]] = defaultdict(list)
    for track in tracks:
        artist_key = normalize_artist_key(track.get('artist', ''))
        if artist_key:
            by_artist[artist_key].append(track)

    threshold_ratio = threshold / 100.0

    # Within each artist, find duplicate pairs
    for artist_key, artist_tracks in by_artist.items():
        if len(artist_tracks) < 2:
            continue

        # Pre-compute normalized titles
        normalized = []
        for track in artist_tracks:
            title = track.get('title', '')
            norm = normalize_title_for_dedupe(title, mode=mode)
            normalized.append((track, norm))

        # Compare all pairs
        for i, (track1, norm1) in enumerate(normalized):
            for track2, norm2 in normalized[i + 1:]:
                if not norm1 or not norm2:
                    continue

                # Skip exact ID matches
                if track1.get('track_id') == track2.get('track_id'):
                    continue

                # Short title check
                require_exact = len(norm1) < short_title_min_len or len(norm2) < short_title_min_len
                if require_exact:
                    if norm1 == norm2:
                        duplicates.append((track1, track2, 1.0))
                else:
                    score = title_similarity(norm1, norm2)
                    if score >= threshold_ratio:
                        duplicates.append((track1, track2, score))

    return duplicates


def analyze_duplicates(duplicates: List[Tuple[Dict, Dict, float]]) -> None:
    """Print analysis of found duplicates."""
    if not duplicates:
        print("\nNo potential duplicates found with current settings.")
        return

    print(f"\n{'=' * 80}")
    print(f"Found {len(duplicates)} potential duplicate pairs")
    print(f"{'=' * 80}\n")

    # Group by artist for readability
    by_artist = defaultdict(list)
    for t1, t2, score in duplicates:
        artist = t1.get('artist', 'Unknown')
        by_artist[artist].append((t1, t2, score))

    for artist, pairs in sorted(by_artist.items()):
        print(f"\n{artist} ({len(pairs)} pairs)")
        print("-" * 60)

        for t1, t2, score in pairs:
            title1 = t1.get('title', '')
            title2 = t2.get('title', '')
            album1 = t1.get('album', 'Unknown Album')
            album2 = t2.get('album', 'Unknown Album')

            # Calculate version preference
            pref1 = calculate_version_preference_score(title1)
            pref2 = calculate_version_preference_score(title2)

            preferred = title1 if pref1 >= pref2 else title2
            would_filter = title2 if pref1 >= pref2 else title1

            print(f"  Similarity: {score:.1%}")
            print(f"    KEEP:   \"{title1}\" (from: {album1}) [pref={pref1}]")
            print(f"    FILTER: \"{title2}\" (from: {album2}) [pref={pref2}]")
            print()


def show_normalization_examples(tracks: List[Dict], mode: str, limit: int = 20) -> None:
    """Show examples of how titles are normalized."""
    print(f"\n{'=' * 80}")
    print(f"Title Normalization Examples (mode={mode})")
    print(f"{'=' * 80}\n")

    # Find interesting titles (with parentheses, brackets, dashes)
    interesting = []
    for track in tracks:
        title = track.get('title', '')
        if any(c in title for c in '()[]- '):
            if 'remaster' in title.lower() or 'live' in title.lower() or 'demo' in title.lower():
                interesting.append(track)

    for track in interesting[:limit]:
        title = track.get('title', '')
        artist = track.get('artist', '')
        normalized = normalize_title_for_dedupe(title, mode=mode)
        print(f"  Original:   {artist} - {title}")
        print(f"  Normalized: {normalized}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate title deduplication on your library"
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Fuzzy match threshold (0-100). Higher = stricter. Default: from config"
    )
    parser.add_argument(
        "--mode", choices=["strict", "loose"], default=None,
        help="Normalization mode. Default: from config"
    )
    parser.add_argument(
        "--artist", type=str, default=None,
        help="Filter to specific artist (case-insensitive partial match)"
    )
    parser.add_argument(
        "--limit", type=int, default=5000,
        help="Maximum tracks to analyze (default: 5000)"
    )
    parser.add_argument(
        "--show-normalization", action="store_true",
        help="Show examples of title normalization"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to metadata database. Default: from config"
    )

    args = parser.parse_args()

    # Load config
    try:
        config = Config()
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        config = None

    # Get settings from config or args
    db_path = args.db
    if not db_path and config:
        db_path = config.library_database_path

    if not db_path:
        print("Error: No database path specified. Use --db or set in config.yaml")
        sys.exit(1)

    threshold = args.threshold
    if threshold is None and config:
        threshold = config.title_dedupe_threshold
    threshold = threshold or 92

    mode = args.mode
    if mode is None and config:
        mode = config.title_dedupe_mode
    mode = mode or "loose"

    short_title_min_len = 6
    if config:
        short_title_min_len = config.title_dedupe_short_title_min_len

    print(f"Title Dedupe Validation")
    print(f"=" * 40)
    print(f"Database: {db_path}")
    print(f"Threshold: {threshold}")
    print(f"Mode: {mode}")
    print(f"Short title min length: {short_title_min_len}")
    if args.artist:
        print(f"Artist filter: {args.artist}")
    print()

    # Load tracks
    print("Loading tracks from database...")
    try:
        tracks = load_tracks_from_db(db_path, args.artist, args.limit)
    except Exception as e:
        print(f"Error loading tracks: {e}")
        sys.exit(1)

    print(f"Loaded {len(tracks)} tracks")

    # Show normalization examples if requested
    if args.show_normalization:
        show_normalization_examples(tracks, mode)

    # Find duplicates
    print("\nSearching for potential duplicates...")
    duplicates = find_potential_duplicates(
        tracks,
        threshold=threshold,
        mode=mode,
        short_title_min_len=short_title_min_len,
    )

    # Analyze and report
    analyze_duplicates(duplicates)

    # Summary stats
    if duplicates:
        unique_artists = len(set(d[0].get('artist') for d in duplicates))
        print(f"\nSummary:")
        print(f"  Total duplicate pairs: {len(duplicates)}")
        print(f"  Affected artists: {unique_artists}")
        print(f"\nTo adjust sensitivity:")
        print(f"  - Increase threshold (current: {threshold}) for stricter matching")
        print(f"  - Use 'strict' mode to preserve version tags")
        print(f"  - Adjust short_title_min_len (current: {short_title_min_len})")


if __name__ == "__main__":
    main()
