#!/usr/bin/env python3
"""
Test Beat3Tower Feature Extraction
===================================

Test script to extract beat3tower features for a sample of tracks.

This validates the beat3tower extraction pipeline before full integration
into the scan process.

Usage:
    python scripts/test_beat3tower_extraction.py                # Extract 10 tracks
    python scripts/test_beat3tower_extraction.py --limit 50     # Extract 50 tracks
    python scripts/test_beat3tower_extraction.py --random       # Random sample
    python scripts/test_beat3tower_extraction.py --no-update    # Don't write to DB
    python scripts/test_beat3tower_extraction.py --artists "Fela Kuti,Miles Davis"

Output:
    - Extracts beat3tower features for sample tracks
    - Optionally updates database with new features
    - Generates validation report showing:
      * Feature dimensions per tower
      * BPM detection results
      * Beat count statistics
      * Extraction success rate
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.beat3tower_extractor import (
    Beat3TowerExtractor,
    Beat3TowerConfig,
    extract_beat3tower_features,
)
from src.features.beat3tower_types import Beat3TowerFeatures

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test beat3tower feature extraction on sample tracks"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of tracks to extract (default: 10)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Random sample instead of sequential",
    )
    parser.add_argument(
        "--artists",
        help="Comma-separated list of artists to filter (e.g., 'Fela Kuti,Miles Davis')",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update database (dry run)",
    )
    parser.add_argument(
        "--db-path",
        default="data/metadata.db",
        help="Path to metadata database",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def get_sample_tracks(
    db_path: str,
    limit: int,
    random: bool = False,
    artists: Optional[List[str]] = None,
) -> List[Dict]:
    """Get sample tracks from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build query
    where_clauses = ["file_path IS NOT NULL"]
    params = []

    if artists:
        artist_placeholders = ",".join(["?"] * len(artists))
        where_clauses.append(f"artist IN ({artist_placeholders})")
        params.extend(artists)

    where_clause = " AND ".join(where_clauses)
    order_clause = "ORDER BY RANDOM()" if random else "ORDER BY track_id"

    query = f"""
        SELECT track_id, file_path, artist, title, album, sonic_features
        FROM tracks
        WHERE {where_clause}
        {order_clause}
        LIMIT ?
    """
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def extract_track_features(
    file_path: str,
    config: Beat3TowerConfig,
) -> Optional[Dict]:
    """Extract beat3tower features for a single track."""
    try:
        extractor = Beat3TowerExtractor(config)
        result = extractor.extract_from_file(file_path)
        return result
    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {e}")
        return None


def update_track_features(
    db_path: str,
    track_id: str,
    features_dict: Dict,
) -> None:
    """Update track's sonic_features in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    features_json = json.dumps(features_dict)
    cursor.execute(
        "UPDATE tracks SET sonic_features = ? WHERE track_id = ?",
        (features_json, track_id),
    )

    conn.commit()
    conn.close()


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze extraction results and compute statistics."""
    stats = {
        'total': len(results),
        'successful': sum(1 for r in results if r['features'] is not None),
        'failed': sum(1 for r in results if r['features'] is None),
        'bpm_detected': [],
        'beat_counts': [],
        'rhythm_dims': [],
        'timbre_dims': [],
        'harmony_dims': [],
    }

    for result in results:
        if result['features'] is None:
            continue

        features = result['features']
        full = Beat3TowerFeatures.from_dict(features['full'])

        stats['bpm_detected'].append(full.bpm_info.primary_bpm)
        stats['beat_counts'].append(full.n_beats)
        stats['rhythm_dims'].append(full.rhythm.to_vector().shape[0])
        stats['timbre_dims'].append(full.timbre.to_vector().shape[0])
        stats['harmony_dims'].append(full.harmony.to_vector().shape[0])

    # Compute statistics
    if stats['bpm_detected']:
        stats['bpm_mean'] = float(np.mean(stats['bpm_detected']))
        stats['bpm_std'] = float(np.std(stats['bpm_detected']))
        stats['bpm_min'] = float(np.min(stats['bpm_detected']))
        stats['bpm_max'] = float(np.max(stats['bpm_detected']))

        stats['beats_mean'] = float(np.mean(stats['beat_counts']))
        stats['beats_std'] = float(np.std(stats['beat_counts']))
        stats['beats_min'] = int(np.min(stats['beat_counts']))
        stats['beats_max'] = int(np.max(stats['beat_counts']))

    return stats


def print_report(results: List[Dict], stats: Dict, args: argparse.Namespace) -> None:
    """Print validation report."""
    print("\n" + "=" * 70)
    print("BEAT3TOWER EXTRACTION TEST REPORT")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Database: {args.db_path}")
    print(f"  Sample size: {args.limit}")
    print(f"  Random sampling: {args.random}")
    if args.artists:
        print(f"  Artist filter: {args.artists}")
    print(f"  Update database: {not args.no_update}")

    print(f"\nExtraction Results:")
    print(f"  Total tracks: {stats['total']}")
    print(f"  Successful: {stats['successful']} ({100*stats['successful']/stats['total']:.1f}%)")
    print(f"  Failed: {stats['failed']} ({100*stats['failed']/stats['total']:.1f}%)")

    if stats['successful'] > 0:
        print(f"\nBPM Detection:")
        print(f"  Mean: {stats['bpm_mean']:.1f} BPM")
        print(f"  Std: {stats['bpm_std']:.1f}")
        print(f"  Range: {stats['bpm_min']:.1f} - {stats['bpm_max']:.1f}")

        print(f"\nBeat Counts:")
        print(f"  Mean: {stats['beats_mean']:.1f} beats")
        print(f"  Std: {stats['beats_std']:.1f}")
        print(f"  Range: {stats['beats_min']} - {stats['beats_max']}")

        print(f"\nFeature Dimensions:")
        print(f"  Rhythm: {stats['rhythm_dims'][0]} dims")
        print(f"  Timbre: {stats['timbre_dims'][0]} dims")
        print(f"  Harmony: {stats['harmony_dims'][0]} dims")
        print(f"  Total: {sum([stats['rhythm_dims'][0], stats['timbre_dims'][0], stats['harmony_dims'][0]])} dims")

    print(f"\nSample Tracks:")
    for i, result in enumerate(results[:5], 1):
        track = result['track']
        artist = track['artist'] or 'Unknown'
        title = track['title'] or 'Unknown'
        status = "OK" if result['features'] else "FAIL"

        print(f"  {i}. [{status}] {artist} - {title}")

        if result['features']:
            full = Beat3TowerFeatures.from_dict(result['features']['full'])
            print(f"     BPM: {full.bpm_info.primary_bpm:.1f}, Beats: {full.n_beats}, Stability: {full.bpm_info.tempo_stability:.2f}")

    if stats['failed'] > 0:
        print(f"\nFailed Tracks:")
        for result in results:
            if result['features'] is None:
                track = result['track']
                artist = track['artist'] or 'Unknown'
                title = track['title'] or 'Unknown'
                print(f"  - {artist} - {title}")
                print(f"    Reason: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 70)


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse artists if provided
    artists = None
    if args.artists:
        artists = [a.strip() for a in args.artists.split(",")]
        logger.info(f"Filtering for artists: {artists}")

    # Get sample tracks
    logger.info(f"Loading {args.limit} sample tracks from {args.db_path}...")
    tracks = get_sample_tracks(
        args.db_path,
        args.limit,
        random=args.random,
        artists=artists,
    )

    if not tracks:
        logger.error("No tracks found matching criteria")
        return

    logger.info(f"Found {len(tracks)} tracks")

    # Extract features
    config = Beat3TowerConfig()
    results = []

    for i, track in enumerate(tracks, 1):
        logger.info(f"[{i}/{len(tracks)}] Extracting: {track['artist']} - {track['title']}")

        file_path = track['file_path']
        if not Path(file_path).exists():
            logger.warning(f"  File not found: {file_path}")
            results.append({
                'track': track,
                'features': None,
                'error': 'File not found',
            })
            continue

        features = extract_track_features(file_path, config)

        if features:
            logger.info(f"  [OK] Extracted: {features['metadata']['n_beats_full']} beats, {features['metadata']['duration']:.1f}s")

            # Update database if requested
            if not args.no_update:
                update_track_features(args.db_path, track['track_id'], features)
                logger.info(f"  [OK] Updated database")

            results.append({
                'track': track,
                'features': features,
            })
        else:
            logger.warning(f"  [FAIL] Extraction failed")
            results.append({
                'track': track,
                'features': None,
                'error': 'Extraction failed',
            })

    # Analyze and print report
    stats = analyze_results(results)
    print_report(results, stats, args)

    # Summary
    if stats['successful'] > 0:
        print(f"\n[SUCCESS] Extracted beat3tower features for {stats['successful']}/{stats['total']} tracks")
        if not args.no_update:
            print(f"[SUCCESS] Database updated with new features")
        else:
            print(f"[INFO] Database NOT updated (--no-update flag)")
    else:
        print(f"\n[FAIL] No tracks successfully extracted")


if __name__ == "__main__":
    main()
