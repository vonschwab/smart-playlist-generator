#!/usr/bin/env python3
"""
Quick Duration Health Check
===========================
Fast script to check duration data integrity.

Usage:
    python check_duration_health.py              # Show health stats
    python check_duration_health.py --find-zero  # Find tracks with 0 duration
    python check_duration_health.py --find-orphaned  # Find orphaned tracks (-1)
"""
import sys
import sqlite3
from pathlib import Path
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_db():
    """Get database connection"""
    db_path = ROOT_DIR / 'data' / 'metadata.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def health_check():
    """Show duration health statistics"""
    conn = get_db()
    cursor = conn.cursor()

    logger.info("Duration Health Check")
    logger.info("=" * 50)

    # Basic stats
    cursor.execute("SELECT COUNT(*) FROM tracks")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms > 0")
    valid = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms = 0")
    zero = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms = -1")
    orphaned = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms IS NULL")
    null_dur = cursor.fetchone()[0]

    logger.info(f"Total tracks: {total:,}")
    logger.info(f"  Valid (> 0): {valid:,} ✓")
    logger.info(f"  Zero: {zero:,}")
    logger.info(f"  Orphaned (-1): {orphaned:,}")
    logger.info(f"  NULL: {null_dur:,}")

    if valid == total:
        logger.info("\n✓ All tracks have valid duration!")
        health = True
    else:
        if orphaned > 0 and (valid + orphaned) == total:
            logger.info(f"\n✓ All tracks have duration_ms populated")
            logger.info(f"  ({orphaned} orphaned tracks are expected - files deleted/corrupted)")
            health = True
        else:
            logger.warning(f"\n⚠ {total - valid:,} tracks need attention")
            health = False

    # Duration stats
    cursor.execute("""
        SELECT MIN(duration_ms), MAX(duration_ms), AVG(duration_ms) FROM tracks WHERE duration_ms > 0
    """)

    row = cursor.fetchone()
    if row:
        logger.info(f"\nDuration statistics (in seconds):")
        logger.info(f"  Min: {int(row[0]//1000)}s")
        logger.info(f"  Max: {int(row[1]//1000)}s")
        logger.info(f"  Avg: {int(row[2]//1000)}s")

    conn.close()
    return health


def find_zero_duration():
    """Find tracks with 0 duration"""
    conn = get_db()
    cursor = conn.cursor()

    logger.info("\nTracks with zero duration:")
    logger.info("=" * 50)

    cursor.execute("SELECT track_id, artist, title FROM tracks WHERE duration_ms = 0 ORDER BY artist, title LIMIT 20")
    rows = cursor.fetchall()

    if not rows:
        logger.info("✓ No tracks with zero duration found")
    else:
        for row in rows:
            logger.info(f"  {row['artist']} - {row['title']} ({row['track_id'][:8]}...)")

    total = cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms = 0").fetchone()[0]
    if len(rows) < total:
        logger.info(f"  ... and {total - len(rows)} more")

    conn.close()


def find_orphaned():
    """Find orphaned tracks (-1 duration)"""
    conn = get_db()
    cursor = conn.cursor()

    logger.info("\nOrphaned tracks (marked with -1):")
    logger.info("=" * 50)

    cursor.execute("""
        SELECT track_id, artist, title, file_path
        FROM tracks
        WHERE duration_ms = -1
        ORDER BY artist, title LIMIT 20
    """)
    rows = cursor.fetchall()

    if not rows:
        logger.info("✓ No orphaned tracks found")
    else:
        for row in rows:
            logger.info(f"  {row['artist']} - {row['title']}")
            logger.info(f"    File: {row['file_path']}")

    total = cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration_ms = -1").fetchone()[0]
    if len(rows) < total:
        logger.info(f"  ... and {total - len(rows)} more orphaned tracks")

    conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check track duration health')
    parser.add_argument('--find-zero', action='store_true', help='Find tracks with 0 duration')
    parser.add_argument('--find-orphaned', action='store_true', help='Find orphaned tracks')
    args = parser.parse_args()

    try:
        if args.find_zero:
            find_zero_duration()
        elif args.find_orphaned:
            find_orphaned()
        else:
            healthy = health_check()
            return 0 if healthy else 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
