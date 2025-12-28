#!/usr/bin/env python3
"""
Sonic Feature Updater
=====================
Analyzes and updates sonic features for tracks with built-in safety protections.

Usage:
    python update_sonic.py                   # SAFE MODE: Only analyze tracks without features
    python update_sonic.py --limit 100       # Analyze up to 100 tracks
    python update_sonic.py --workers 4       # Use 4 parallel workers
    python update_sonic.py --stats           # Show statistics (including beat3tower count)
    python update_sonic.py --beat3tower      # Use beat3tower extraction (recommended)

    ⚠️  DANGEROUS - USE WITH CAUTION:
    python update_sonic.py --force           # Re-analyze ALL tracks (requires confirmation)
                                             # This will overwrite existing beat3tower features!

Safety Features:
    - Safe mode (default): Only analyzes tracks without existing sonic features
    - Force mode protection: Requires explicit 'YES' confirmation to overwrite features
    - Beat3tower detection: Shows count of protected beat3tower features
    - Backup reminder: Warns to create backup before using --force mode

Features:
    - Multi-segment analysis: Extracts features from beginning, middle, and end (30s each)
    - Beat3tower extraction: Advanced beat-synchronized 3-tower analysis (recommended)
    - Parallel processing: Multi-core CPU utilization for faster analysis
    - Local analysis: Uses Librosa for feature extraction
    - Extracts MFCC, chroma, spectral features, tempo, rhythm descriptors, etc.
    - Backward compatible: Handles both old and new feature formats

Recommended workflow:
    1. Create a backup: python scripts/backup_sonic_features.py --backup
    2. Check stats: python update_sonic.py --stats
    3. Run analysis: python update_sonic.py --beat3tower
"""
import sys
import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Ensure project root is on sys.path so `src` imports work even when run from scripts/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Configure logging (centralized)
from src.logging_config import setup_logging
logger = setup_logging(name='update_sonic', log_file='sonic_analysis.log')
logger.info("Sonic feature analysis started")

DB_BUSY_TIMEOUT_MS = 30000
DB_RETRY_ATTEMPTS = 5
DB_RETRY_SLEEP_BASE = 0.5


def analyze_track_worker(track_data: Tuple[str, str, str, str, bool, bool]) -> Optional[Tuple[str, Dict[str, Any], str, str]]:
    """
    Worker function for parallel track analysis

    Args:
        track_data: Tuple of (track_id, file_path, artist, title, use_beat_sync, use_beat3tower)

    Returns:
        Tuple of (track_id, features, artist, title) or None on failure
    """
    import sys
    import logging
    from pathlib import Path
    import os

    # Suppress worker subprocess logging spam
    logging.getLogger().setLevel(logging.WARNING)

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.hybrid_sonic_analyzer import HybridSonicAnalyzer

    track_id, file_path, artist, title = track_data[:4]
    use_beat_sync = track_data[4] if len(track_data) > 4 else False
    use_beat3tower = track_data[5] if len(track_data) > 5 else False

    try:
        # Check file exists
        if not Path(file_path).exists():
            return None

        # Create analyzer instance (each worker needs its own)
        analyzer = HybridSonicAnalyzer(use_beat_sync=use_beat_sync, use_beat3tower=use_beat3tower)

        # Analyze track
        features = analyzer.analyze_track(file_path, artist, title)

        if features:
            return (track_id, features, artist, title)
        else:
            return None

    except Exception as e:
        # Avoid logger in subprocess; print minimal context for debugging
        print(f"[worker] Failed to analyze track {track_id}: {e}", file=sys.stderr)
        return None


class SonicFeaturePipeline:
    """Pipeline for analyzing tracks and storing sonic features"""

    def __init__(self, db_path: Optional[str] = None, use_beat_sync: bool = False, use_beat3tower: bool = False):
        """
        Initialize pipeline

        Args:
            db_path: Path to metadata database
            use_beat_sync: If True, use old beat-synchronized feature extraction
            use_beat3tower: If True, use 3-tower beat-synchronized extraction (recommended)
                           Takes precedence over use_beat_sync
        """
        from src.hybrid_sonic_analyzer import HybridSonicAnalyzer

        default_db = ROOT_DIR / "data" / "metadata.db"
        self.db_path = str(Path(db_path)) if db_path else str(default_db)
        self.conn = None
        self.use_beat_sync = use_beat_sync
        self.use_beat3tower = use_beat3tower
        self.analyzer = HybridSonicAnalyzer(use_beat_sync=use_beat_sync, use_beat3tower=use_beat3tower)
        self._init_db_connection()

        # Determine mode for logging
        if use_beat3tower:
            mode = "beat3tower"
        elif use_beat_sync:
            mode = "beat-sync"
        else:
            mode = "windowed"

        logger.info(f"Initialized SonicFeaturePipeline (mode={mode})")

    def _init_db_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=DB_BUSY_TIMEOUT_MS / 1000)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError as exc:
            logger.warning("Failed to apply SQLite pragmas (%s)", exc)
        logger.debug(f"Connected to database: {self.db_path}")

    def get_pending_tracks(self, limit: Optional[int] = None, force: bool = False) -> list:
        """
        Get tracks that need sonic analysis

        Args:
            limit: Maximum number of tracks to return (None = all)
            force: If True, re-analyze all tracks (even those already analyzed)
                  WARNING: This will overwrite existing beat3tower features!

        Returns:
            List of track rows
        """
        cursor = self.conn.cursor()

        if force:
            # Re-analyze everything (convert old format to multi-segment)
            # WARNING: This will overwrite beat3tower features!
            logger.warning("FORCE MODE: Will re-analyze ALL tracks, including those with existing beat3tower features!")

            query = """
                SELECT track_id, file_path, musicbrainz_id, artist, title, sonic_features
                FROM tracks
                WHERE file_path IS NOT NULL
                ORDER BY
                    CASE WHEN musicbrainz_id IS NOT NULL THEN 0 ELSE 1 END,
                    track_id
            """
        else:
            # Only analyze tracks without features (SAFE MODE - default)
            logger.info("Safe mode: Only analyzing tracks without existing sonic features")

            query = """
                SELECT track_id, file_path, musicbrainz_id, artist, title, sonic_features
                FROM tracks
                WHERE file_path IS NOT NULL
                  AND sonic_features IS NULL
                  AND sonic_failed_at IS NULL
                ORDER BY
                    CASE WHEN musicbrainz_id IS NOT NULL THEN 0 ELSE 1 END,
                    track_id
            """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        tracks = cursor.fetchall()

        # Count blacklisted tracks
        if not force:
            blacklisted = cursor.execute("""
                SELECT COUNT(*)
                FROM tracks
                WHERE file_path IS NOT NULL
                  AND sonic_features IS NULL
                  AND sonic_failed_at IS NOT NULL
            """).fetchone()[0]
            if blacklisted > 0:
                logger.info(f"Skipping {blacklisted} blacklisted tracks (previously failed analysis; use --force to retry)")

        # Count beat3tower tracks that would be overwritten
        if force:
            beat3tower_count = 0
            for track in tracks:
                if track['sonic_features']:
                    try:
                        features = json.loads(track['sonic_features'])
                        if isinstance(features, dict) and 'full' in features:
                            if isinstance(features['full'], dict) and features['full'].get('extraction_method') == 'beat3tower':
                                beat3tower_count += 1
                    except:
                        pass

            if beat3tower_count > 0:
                logger.warning(f"⚠️  FORCE MODE will overwrite {beat3tower_count} tracks with existing beat3tower features!")
                logger.warning(f"⚠️  Consider creating a backup first: python scripts/backup_sonic_features.py --backup")

        return tracks

    def analyze_track(self, track_id: str, file_path: str, mbid: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze a single track and return features

        Args:
            track_id: Track ID (rating key)
            file_path: Path to audio file
            mbid: MusicBrainz ID (optional)

        Returns:
            Features dictionary or None on failure
        """
        try:
            # Check file exists
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                return None

            # Analyze with hybrid analyzer
            features = self.analyzer.analyze_track(file_path, mbid)

            if features:
                logger.debug(f"Analyzed track {track_id} (source: {features.get('source', 'unknown')})")
                return features
            else:
                logger.warning(f"Failed to analyze track {track_id}")
                return None

        except Exception as e:
            logger.error(f"Error analyzing track {track_id}: {e}")
            return None

    def store_features(self, track_id: str, features: Dict[str, Any], commit: bool = True):
        """
        Store sonic features in database

        Args:
            track_id: Track ID
            features: Features dictionary
            commit: Whether to commit immediately (default True)
        """
        cursor = self.conn.cursor()

        # Extract source
        source = features.get('source', 'unknown')

        # Detect beat3tower format (no 'source' at root level)
        if source == 'unknown' and 'full' in features:
            if isinstance(features['full'], dict) and features['full'].get('extraction_method') == 'beat3tower':
                source = 'librosa'

        # Convert to JSON (remove 'source' as it's in separate column)
        features_copy = features.copy()
        features_copy.pop('source', None)
        features_json = json.dumps(features_copy)

        for attempt in range(1, DB_RETRY_ATTEMPTS + 1):
            try:
                # Update database (clear failed mark on success)
                cursor.execute("""
                    UPDATE tracks
                    SET sonic_features = ?,
                        sonic_source = ?,
                        sonic_analyzed_at = ?,
                        sonic_failed_at = NULL
                    WHERE track_id = ?
                """, (features_json, source, int(time.time()), track_id))

                if commit:
                    self.conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == DB_RETRY_ATTEMPTS:
                    raise
                sleep_for = DB_RETRY_SLEEP_BASE * attempt
                logger.warning("DB locked; retrying write for %s in %.1fs (attempt %d/%d)", track_id, sleep_for, attempt, DB_RETRY_ATTEMPTS)
                time.sleep(sleep_for)

    def mark_track_failed(self, track_id: str, commit: bool = True):
        """
        Mark a track as failed in the database to prevent repeated analysis attempts.

        Args:
            track_id: Track ID
            commit: Whether to commit immediately (default True)
        """
        cursor = self.conn.cursor()

        for attempt in range(1, DB_RETRY_ATTEMPTS + 1):
            try:
                cursor.execute("""
                    UPDATE tracks
                    SET sonic_failed_at = ?
                    WHERE track_id = ?
                """, (int(time.time()), track_id))

                if commit:
                    self.conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == DB_RETRY_ATTEMPTS:
                    raise
                sleep_for = DB_RETRY_SLEEP_BASE * attempt
                logger.warning("DB locked; retrying failed mark for %s in %.1fs (attempt %d/%d)", track_id, sleep_for, attempt, DB_RETRY_ATTEMPTS)
                time.sleep(sleep_for)

    def run(self, limit: Optional[int] = None, workers: Optional[int] = None, force: bool = False):
        """
        Run the analysis pipeline with parallel processing

        Args:
            limit: Maximum number of tracks to analyze (None = all)
            workers: Number of parallel workers (None = CPU count - 1)
            force: If True, re-analyze all tracks (even those already analyzed)
        """
        logger.info("=" * 70)
        if force:
            logger.info("Starting Sonic Feature Analysis Pipeline (FORCE MODE - Re-analyzing ALL tracks)")
        else:
            logger.info("Starting Sonic Feature Analysis Pipeline")
        logger.info("=" * 70)

        # Get pending tracks
        pending = self.get_pending_tracks(limit, force=force)
        total = len(pending)

        if total == 0:
            logger.info("No pending tracks to analyze!")
            return

        # Determine number of workers
        if workers is None:
            # Use more cores for CPU-intensive work, but leave some headroom
            workers = max(4, min(16, multiprocessing.cpu_count() - 2))

        logger.info(f"Found {total} tracks to analyze")
        logger.info(f"Using {workers} parallel workers")

        # Statistics
        stats = {
            'total': total,
            'analyzed': 0,
            'librosa': 0,
            'failed': 0
        }

        start_time = time.time()
        completed = 0
        batch_size = 50  # Commit every N tracks

        # Prepare track data for workers
        track_data_list = [
            (track['track_id'], track['file_path'], track['artist'], track['title'], self.use_beat_sync, self.use_beat3tower)
            for track in pending
        ]

        # Process tracks in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tracks
            future_to_index = {
                executor.submit(analyze_track_worker, track_data): i
                for i, track_data in enumerate(track_data_list)
            }

            # Process results as they complete
            for future in as_completed(future_to_index):
                completed += 1
                idx = future_to_index[future]
                track_data = track_data_list[idx]
                track_id, file_path, artist, title = track_data[:4]

                try:
                    result = future.result()

                    if result:
                        track_id, features, artist, title = result

                        # Store in database (batch commits)
                        commit_now = (completed % batch_size == 0) or (completed == total)
                        self.store_features(track_id, features, commit=commit_now)
                        stats['analyzed'] += 1

                        # Track source
                        source = features.get('source', 'unknown')
                        if source == 'librosa':
                            stats['librosa'] += 1

                        # Only log individual tracks on failure or debug mode
                        logger.debug(f"[{completed}/{total}] OK {artist} - {title} ({source})")
                    else:
                        stats['failed'] += 1
                        logger.warning(f"[{completed}/{total}] FAIL {artist} - {title}")
                        # Mark track as failed to prevent repeated analysis attempts
                        commit_now = (completed % batch_size == 0) or (completed == total)
                        self.mark_track_failed(track_id, commit=commit_now)

                    # Progress report every 10 tracks
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total - completed) / rate if rate > 0 else 0

                        logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) - "
                                  f"Rate: {rate:.1f} tracks/sec - "
                                  f"ETA: {remaining/60:.1f} min")
                        logger.info(f"  Stats: {stats['librosa']} Librosa, "
                                  f"{stats['failed']} failed")

                except Exception as e:
                    logger.error(f"Error processing result for {artist} - {title}: {e}")
                    stats['failed'] += 1

            # Final commit to catch any remaining
            self.conn.commit()

        # Final report
        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("Analysis Complete!")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Total analyzed: {stats['analyzed']}/{stats['total']}")
        if stats['total'] > 0:
            logger.info(f"  Librosa: {stats['librosa']} ({stats['librosa']/stats['total']*100:.1f}%)")
            logger.info(f"  Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
            logger.info(f"Average rate: {stats['total']/elapsed:.1f} tracks/sec")

    def get_stats(self) -> Dict[str, int]:
        """Get analysis statistics including beat3tower detection"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM tracks WHERE file_path IS NOT NULL")
        total = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) as analyzed FROM tracks WHERE sonic_features IS NOT NULL")
        analyzed = cursor.fetchone()['analyzed']

        cursor.execute("""
            SELECT sonic_source, COUNT(*) as count
            FROM tracks
            WHERE sonic_source IS NOT NULL
            GROUP BY sonic_source
        """)

        sources = {row['sonic_source']: row['count'] for row in cursor.fetchall()}

        # Count beat3tower features specifically
        cursor.execute("""
            SELECT sonic_features
            FROM tracks
            WHERE sonic_features IS NOT NULL
        """)

        beat3tower_count = 0
        legacy_count = 0

        for row in cursor.fetchall():
            try:
                features = json.loads(row['sonic_features'])
                if isinstance(features, dict) and 'full' in features:
                    if isinstance(features['full'], dict) and features['full'].get('extraction_method') == 'beat3tower':
                        beat3tower_count += 1
                    else:
                        legacy_count += 1
                else:
                    legacy_count += 1
            except:
                legacy_count += 1

        return {
            'total_tracks': total,
            'analyzed': analyzed,
            'pending': total - analyzed,
            'librosa': sources.get('librosa', 0),
            'beat3tower': beat3tower_count,
            'legacy': legacy_count
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze sonic features for tracks')
    parser.add_argument('--limit', type=int, help='Maximum number of tracks to analyze')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: CPU count - 2)')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--force', action='store_true', help='Re-analyze ALL tracks (converts old format to multi-segment)')
    parser.add_argument('--beat-sync', action='store_true', help='Use beat-synchronized feature extraction (old method)')
    parser.add_argument('--beat3tower', action='store_true', help='Use 3-tower beat-synchronized extraction (recommended)')
    args = parser.parse_args()

    pipeline = SonicFeaturePipeline(use_beat_sync=args.beat_sync, use_beat3tower=args.beat3tower)

    if args.stats:
        # Show statistics
        stats = pipeline.get_stats()
        print("\nSonic Analysis Statistics:")
        print("=" * 60)
        print(f"  Total tracks: {stats['total_tracks']:,}")
        print(f"  Analyzed: {stats['analyzed']:,}")
        print(f"  Pending: {stats['pending']:,}")
        print(f"\n  Feature formats:")
        print(f"    Beat3tower (recommended): {stats['beat3tower']:,}")
        print(f"    Legacy format: {stats['legacy']:,}")
        print(f"  Librosa source: {stats['librosa']:,}")
        print("=" * 60)
        if stats['beat3tower'] > 0:
            print(f"\n✓ {stats['beat3tower']:,} tracks protected with beat3tower features")
            print("  Run without --force to preserve these features (safe mode)")
    else:
        # Run analysis
        if args.force:
            # Get current stats to show what will be overwritten
            stats = pipeline.get_stats()

            print("\n" + "=" * 70)
            print("⚠️  WARNING: FORCE MODE ENABLED ⚠️")
            print("=" * 70)
            print(f"This will RE-ANALYZE ALL {stats['total_tracks']:,} tracks, including:")
            print(f"  • {stats['beat3tower']:,} tracks with beat3tower features")
            print(f"  • {stats['legacy']:,} tracks with legacy features")

            # Auto-backup if beat3tower features exist
            if stats['beat3tower'] > 0:
                print("\n🔒 Auto-backup: Creating safety backup of beat3tower features...")
                try:
                    # Import and run backup
                    sys.path.insert(0, str(ROOT_DIR / "scripts"))
                    from backup_sonic_features import SonicBackupManager

                    backup_manager = SonicBackupManager()
                    backup_path = backup_manager.create_backup(name="auto_before_force")
                    backup_manager._close()

                    print(f"✓ Backup created: {backup_path.name}")
                    print(f"  Restore with: python scripts/backup_sonic_features.py --restore auto_before_force\n")
                except Exception as e:
                    print(f"❌ Backup failed: {e}")
                    print("   Aborting force mode for safety.")
                    pipeline.close()
                    sys.exit(1)
            else:
                print("\n✓ No beat3tower features to backup.\n")

            # Require explicit confirmation
            response = input("Type 'YES' to confirm you want to overwrite existing features: ")
            if response.strip().upper() != 'YES':
                print("\n❌ Aborted. No tracks were modified.")
                pipeline.close()
                sys.exit(0)

            print("\n✓ Confirmed. Starting re-analysis...\n")

        pipeline.run(limit=args.limit, workers=args.workers, force=args.force)

    pipeline.close()
