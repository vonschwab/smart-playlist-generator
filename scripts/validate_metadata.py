#!/usr/bin/env python3
"""
Metadata Validation Script
===========================
Validates that tracks with file paths have accurate metadata and genres.
"""
import sqlite3
import random
from pathlib import Path
import os

def validate_metadata(db_path: str = "data/metadata.db", sample_size: int = 20):
    """
    Validate a random sample of tracks

    Args:
        db_path: Path to metadata database
        sample_size: Number of tracks to validate
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print(f"\nMetadata Validation Report")
    print("=" * 80)

    # Get sample of tracks with file paths
    cursor.execute("""
        SELECT track_id, artist, title, album, file_path,
               musicbrainz_id, sonic_features, sonic_source
        FROM tracks
        WHERE file_path IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (sample_size,))

    tracks = cursor.fetchall()

    # Validation results
    stats = {
        'total': len(tracks),
        'file_exists': 0,
        'file_missing': 0,
        'has_metadata': 0,
        'has_mbid': 0,
        'has_sonic': 0,
        'has_genres': 0,
        'genre_count': []
    }

    print(f"\nValidating {stats['total']} random tracks...\n")

    for i, track in enumerate(tracks, 1):
        file_exists = os.path.exists(track['file_path'])

        # Count genres for this track
        cursor.execute("""
            SELECT COUNT(*) FROM track_genres WHERE track_id = ?
        """, (track['track_id'],))
        genre_count = cursor.fetchone()[0]

        # Update stats
        if file_exists:
            stats['file_exists'] += 1
        else:
            stats['file_missing'] += 1

        if track['artist'] and track['title']:
            stats['has_metadata'] += 1

        if track['musicbrainz_id']:
            stats['has_mbid'] += 1

        if track['sonic_features']:
            stats['has_sonic'] += 1

        if genre_count > 0:
            stats['has_genres'] += 1
            stats['genre_count'].append(genre_count)

        # Print track details
        status = "[OK]" if file_exists else "[!!]"
        print(f"{i:2}. {status} {track['artist']} - {track['title']}")
        print(f"    File: {track['file_path']}")
        print(f"    MusicBrainz: {'Yes' if track['musicbrainz_id'] else 'No'} | "
              f"Sonic: {track['sonic_source'] or 'No'} | "
              f"Genres: {genre_count}")

        if not file_exists:
            print(f"    [!] FILE NOT FOUND")
        print()

    # Summary statistics
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nFile Status:")
    print(f"  Files exist: {stats['file_exists']}/{stats['total']} ({stats['file_exists']/stats['total']*100:.1f}%)")
    print(f"  Files missing: {stats['file_missing']}/{stats['total']} ({stats['file_missing']/stats['total']*100:.1f}%)")

    print(f"\nMetadata Completeness:")
    print(f"  Has artist/title: {stats['has_metadata']}/{stats['total']} ({stats['has_metadata']/stats['total']*100:.1f}%)")
    print(f"  Has MusicBrainz ID: {stats['has_mbid']}/{stats['total']} ({stats['has_mbid']/stats['total']*100:.1f}%)")
    print(f"  Has sonic features: {stats['has_sonic']}/{stats['total']} ({stats['has_sonic']/stats['total']*100:.1f}%)")
    print(f"  Has genres: {stats['has_genres']}/{stats['total']} ({stats['has_genres']/stats['total']*100:.1f}%)")

    if stats['genre_count']:
        avg_genres = sum(stats['genre_count']) / len(stats['genre_count'])
        print(f"\nGenre Statistics:")
        print(f"  Average genres per track: {avg_genres:.1f}")
        print(f"  Min genres: {min(stats['genre_count'])}")
        print(f"  Max genres: {max(stats['genre_count'])}")

    # Check overall database health
    print("\n" + "=" * 80)
    print("OVERALL DATABASE HEALTH")
    print("=" * 80)

    cursor.execute("""
        SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL
    """)
    total_with_paths = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM tracks
        WHERE file_path IS NOT NULL AND sonic_features IS NOT NULL
    """)
    total_with_sonic = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT track_id) FROM track_genres
        WHERE track_id IN (SELECT track_id FROM tracks WHERE file_path IS NOT NULL)
    """)
    tracks_with_genres = cursor.fetchone()[0]

    print(f"\nTracks with file paths: {total_with_paths:,}")
    print(f"  - With sonic features: {total_with_sonic:,} ({total_with_sonic/total_with_paths*100:.1f}%)")
    print(f"  - With genres: {tracks_with_genres:,} ({tracks_with_genres/total_with_paths*100:.1f}%)")
    print(f"  - Need sonic analysis: {total_with_paths - total_with_sonic:,}")

    # Check for multi-segment features
    cursor.execute("""
        SELECT sonic_features FROM tracks
        WHERE file_path IS NOT NULL AND sonic_features IS NOT NULL
        LIMIT 10
    """)

    import json
    multi_segment_count = 0
    single_segment_count = 0

    for row in cursor.fetchall():
        try:
            features = json.loads(row[0])
            if 'average' in features:
                multi_segment_count += 1
            else:
                single_segment_count += 1
        except:
            pass

    if multi_segment_count + single_segment_count > 0:
        print(f"\nSonic Feature Format (sample of 10):")
        print(f"  Multi-segment (new): {multi_segment_count}")
        print(f"  Single-segment (old): {single_segment_count}")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate metadata quality')
    parser.add_argument('--sample', type=int, default=20,
                       help='Number of tracks to validate (default: 20)')
    args = parser.parse_args()

    validate_metadata(sample_size=args.sample)
