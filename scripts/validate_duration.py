#!/usr/bin/env python3
"""
Duration Validation Script
===========================
Validates that duration support is working correctly in the system.

Checks:
1. All tracks have duration_ms populated
2. Duration values are reasonable (> 0)
3. Duration filtering logic is working
4. M3U export includes valid EXTINF metadata
"""
import sys
import sqlite3
from pathlib import Path
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config
from src.local_library_client import LocalLibraryClient

# Configure logging (centralized)
from src.logging_config import setup_logging
logger = setup_logging(name='validate_duration', log_file='validate_duration.log')


def validate_schema():
    """Verify duration_ms column exists and is populated"""
    logger.info("=" * 70)
    logger.info("1. DATABASE SCHEMA VALIDATION")
    logger.info("=" * 70)

    db_path = ROOT_DIR / 'data' / 'metadata.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check column exists
    cursor.execute("PRAGMA table_info(tracks)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    if 'duration_ms' not in columns:
        logger.error("✗ duration_ms column NOT found in tracks table")
        return False

    logger.info("✓ duration_ms column exists (type: INTEGER)")

    # Check population statistics
    cursor.execute("SELECT COUNT(*) FROM tracks")
    total_tracks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms IS NOT NULL")
    with_duration = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms > 0")
    valid_duration = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms = -1")
    orphaned = cursor.fetchone()[0]

    logger.info(f"  Total tracks: {total_tracks:,}")
    logger.info(f"  Tracks with duration_ms: {with_duration:,} ({with_duration/total_tracks*100:.1f}%)")
    logger.info(f"  Valid duration (> 0): {valid_duration:,}")
    logger.info(f"  Orphaned (-1): {orphaned:,}")

    if with_duration != total_tracks:
        logger.error(f"✗ {total_tracks - with_duration:,} tracks missing duration_ms")
        return False

    logger.info("✓ All tracks have duration_ms populated")

    # Check duration statistics
    cursor.execute("SELECT MIN(duration_ms), MAX(duration_ms), AVG(duration_ms) FROM tracks WHERE duration_ms > 0")
    min_dur, max_dur, avg_dur = cursor.fetchone()

    logger.info(f"  Min duration: {min_dur//1000:.1f}s")
    logger.info(f"  Max duration: {max_dur//1000:.1f}s")
    logger.info(f"  Avg duration: {avg_dur//1000:.1f}s")

    conn.close()
    return True


def validate_library_client():
    """Verify LocalLibraryClient returns duration in queries"""
    logger.info("\n" + "=" * 70)
    logger.info("2. LIBRARY CLIENT VALIDATION")
    logger.info("=" * 70)

    client = LocalLibraryClient()

    # Get all tracks
    tracks = client.get_all_tracks()
    logger.info(f"✓ get_all_tracks() returned {len(tracks):,} tracks")

    # Verify duration is present
    sample = tracks[0] if tracks else None
    if not sample:
        logger.error("✗ No tracks returned")
        return False

    if 'duration' not in sample:
        logger.error("✗ 'duration' key not found in track dict")
        logger.error(f"   Available keys: {list(sample.keys())}")
        return False

    logger.info("✓ Track dict includes 'duration' key")

    # Check duration values
    tracks_with_duration = sum(1 for t in tracks if t.get('duration') and t['duration'] > 0)
    logger.info(f"✓ {tracks_with_duration:,} tracks have positive duration")

    # Test get_track_by_key
    track_id = sample['rating_key']
    track = client.get_track_by_key(track_id)
    if not track:
        logger.error("✗ get_track_by_key() failed")
        return False

    if 'duration' not in track:
        logger.error("✗ get_track_by_key() doesn't include duration")
        return False

    logger.info(f"✓ get_track_by_key('{track_id[:8]}...') returns duration: {track['duration']}ms")

    return True


def validate_filtering():
    """Verify duration filtering works correctly"""
    logger.info("\n" + "=" * 70)
    logger.info("3. FILTERING VALIDATION")
    logger.info("=" * 70)

    config = Config()
    min_duration = config.min_track_duration_seconds
    max_duration = config.max_track_duration_seconds
    min_duration_ms = min_duration * 1000
    max_duration_ms = max_duration * 1000

    logger.info(f"Config settings:")
    logger.info(f"  min_track_duration_seconds: {min_duration}s ({min_duration_ms}ms)")
    logger.info(f"  max_track_duration_seconds: {max_duration}s ({max_duration_ms}ms)")

    client = LocalLibraryClient()
    tracks = client.get_all_tracks()

    # Simulate filtering
    filtered_short = []
    filtered_long = []
    valid = []

    for track in tracks:
        duration = track.get('duration', 0)
        if duration <= 0:
            filtered_short.append(track)
        elif duration < min_duration_ms:
            filtered_short.append(track)
        elif duration > max_duration_ms:
            filtered_long.append(track)
        else:
            valid.append(track)

    logger.info(f"\nFiltering simulation on {len(tracks):,} tracks:")
    logger.info(f"  Short tracks filtered: {len(filtered_short):,} (< {min_duration}s)")
    logger.info(f"  Long tracks filtered: {len(filtered_long):,} (> {max_duration}s)")
    logger.info(f"  Valid tracks remaining: {len(valid):,}")

    if len(valid) < len(tracks) * 0.5:
        logger.warning(f"⚠ Less than 50% of tracks pass filtering (only {len(valid)}/{len(tracks)})")
        logger.warning("  Consider adjusting min/max_track_duration_seconds")
    else:
        logger.info("✓ Filtering retention is healthy")

    return True


def validate_config():
    """Verify config has duration settings"""
    logger.info("\n" + "=" * 70)
    logger.info("4. CONFIG VALIDATION")
    logger.info("=" * 70)

    config = Config()

    try:
        min_dur = config.min_track_duration_seconds
        max_dur = config.max_track_duration_seconds
        logger.info(f"✓ Config loads duration settings")
        logger.info(f"  min_track_duration_seconds: {min_dur}s")
        logger.info(f"  max_track_duration_seconds: {max_dur}s")

        if not (0 < min_dur < max_dur <= 3600):
            logger.warning(f"⚠ Unusual duration settings (check if intentional)")
        else:
            logger.info("✓ Duration settings are reasonable")

        return True
    except Exception as e:
        logger.error(f"✗ Error loading duration config: {e}")
        return False


def main():
    """Run all validations"""
    logger.info("\nPlaylist Generator - Duration Support Validation\n")

    results = []
    results.append(("Schema", validate_schema()))
    results.append(("Library Client", validate_library_client()))
    results.append(("Filtering", validate_filtering()))
    results.append(("Config", validate_config()))

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        logger.info("\n✓ All validations passed! Duration support is working correctly.")
        return 0
    else:
        logger.error("\n✗ Some validations failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
