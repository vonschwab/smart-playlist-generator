#!/usr/bin/env python3
"""
Backfill Duration Script
========================
Fills in missing duration_ms values for tracks in the metadata database
by reading audio file metadata.

Usage:
    python backfill_duration.py                 # Backfill all missing durations
    python backfill_duration.py --dry-run       # Show what would be updated without committing
    python backfill_duration.py --track-id XXX  # Backfill specific track
"""
import sys
import sqlite3
from pathlib import Path
from typing import Optional, Tuple
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DurationBackfiller:
    """Backfills missing duration values from audio files"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize backfiller"""
        if config_path is None:
            config_path = ROOT_DIR / 'config.yaml'
        self.config = Config(config_path)
        self.db_path = ROOT_DIR / 'data' / 'metadata.db'
        self.conn = None

        # Try to import mutagen
        try:
            import mutagen
            self.mutagen = mutagen
            self.has_mutagen = True
        except ImportError:
            logger.error("mutagen not installed - cannot extract duration")
            logger.error("Install with: pip install mutagen")
            self.has_mutagen = False

        self._init_db()

    def _init_db(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row

    def extract_duration(self, file_path: Path) -> Optional[int]:
        """
        Extract duration from audio file in milliseconds

        Args:
            file_path: Path to audio file

        Returns:
            Duration in milliseconds or None if extraction failed
        """
        if not self.has_mutagen:
            return None

        try:
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return None

            audio = self.mutagen.File(file_path, easy=True)
            if audio is None:
                logger.debug(f"Could not read file: {file_path}")
                return None

            duration_seconds = getattr(audio.info, 'length', None)
            if duration_seconds is None:
                logger.debug(f"Could not extract duration from: {file_path}")
                return None

            # Convert to milliseconds
            duration_ms = int(duration_seconds * 1000)
            return duration_ms

        except Exception as e:
            logger.debug(f"Error extracting duration from {file_path}: {e}")
            return None

    def get_tracks_missing_duration(self, limit: Optional[int] = None) -> list:
        """
        Get tracks with missing or zero duration

        Args:
            limit: Maximum number of tracks to return

        Returns:
            List of (track_id, file_path) tuples
        """
        cursor = self.conn.cursor()

        query = """
            SELECT track_id, file_path
            FROM tracks
            WHERE (duration_ms IS NULL OR duration_ms = 0)
            AND file_path IS NOT NULL
            ORDER BY file_path
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return cursor.fetchall()

    def backfill_track(self, track_id: str, file_path: str, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Backfill duration for a single track

        Args:
            track_id: Track ID
            file_path: Path to audio file
            dry_run: If True, don't commit changes to database

        Returns:
            Tuple of (success, message)
        """
        file_path_obj = Path(file_path)

        # Extract duration from file
        duration_ms = self.extract_duration(file_path_obj)

        if duration_ms is None:
            # File doesn't exist or is unreadable - mark with -1
            # This indicates orphaned/unreadable track
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE tracks SET duration_ms = ? WHERE track_id = ?",
                (-1, track_id)
            )
            if not dry_run:
                self.conn.commit()
            return True, f"[ORPHANED] Marked {track_id} as unreadable (file missing/corrupt)"

        # Update database with extracted duration
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE tracks SET duration_ms = ? WHERE track_id = ?",
            (duration_ms, track_id)
        )

        if dry_run:
            return True, f"[DRY RUN] Would update {track_id} to {duration_ms}ms"
        else:
            self.conn.commit()
            return True, f"Updated {track_id} to {duration_ms}ms ({duration_ms/1000:.1f}s)"

    def run(self, dry_run: bool = False, limit: Optional[int] = None, track_id: Optional[str] = None):
        """
        Run the backfill process

        Args:
            dry_run: If True, don't commit changes
            limit: Maximum number of tracks to process
            track_id: If specified, only backfill this track
        """
        logger.info("=" * 70)
        logger.info("Duration Backfill Tool")
        logger.info("=" * 70)

        if not self.has_mutagen:
            logger.error("Cannot proceed without mutagen library")
            return

        if track_id:
            # Backfill specific track
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT track_id, file_path FROM tracks WHERE track_id = ?",
                (track_id,)
            )
            row = cursor.fetchone()
            if not row:
                logger.error(f"Track {track_id} not found")
                return

            tracks = [row]
        else:
            # Get all tracks with missing duration
            tracks = self.get_tracks_missing_duration(limit=limit)

        if not tracks:
            logger.info("No tracks with missing duration found")
            return

        logger.info(f"\nFound {len(tracks)} track(s) with missing duration")

        stats = {
            'success': 0,
            'failed': 0
        }

        logger.info(f"\nProcessing {len(tracks)} track(s)...")
        for i, row in enumerate(tracks, 1):
            track_id_val = row['track_id']
            file_path_val = row['file_path']

            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(tracks)}")

            success, message = self.backfill_track(track_id_val, file_path_val, dry_run=dry_run)

            if success:
                stats['success'] += 1
                logger.info(f"  ✓ {message}")
            else:
                stats['failed'] += 1
                logger.warning(f"  ✗ {message}")

        # Show results
        logger.info("\n" + "=" * 70)
        logger.info("Backfill Complete")
        logger.info("=" * 70)
        logger.info(f"  Successful: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Total processed: {len(tracks)}")

        if dry_run:
            logger.info("\nNote: This was a DRY RUN - no changes were committed to database")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Backfill missing track durations')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without committing')
    parser.add_argument('--limit', type=int,
                       help='Maximum number of tracks to process')
    parser.add_argument('--track-id', type=str,
                       help='Backfill specific track by ID')
    args = parser.parse_args()

    backfiller = DurationBackfiller()

    try:
        backfiller.run(dry_run=args.dry_run, limit=args.limit, track_id=args.track_id)
    finally:
        backfiller.close()
