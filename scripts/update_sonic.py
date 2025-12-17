#!/usr/bin/env python3
"""
Sonic Feature Updater
=====================
Analyzes and updates sonic features for tracks.

Usage:
    python update_sonic.py                   # Incremental update (only missing features)
    python update_sonic.py --limit 100       # Analyze up to 100 tracks
    python update_sonic.py --workers 4       # Use 4 parallel workers
    python update_sonic.py --stats           # Show statistics only
    python update_sonic.py --force           # Re-analyze ALL tracks (convert to multi-segment)
    python update_sonic.py --force --limit 1000  # Convert first 1000 tracks

Features:
    - Multi-segment analysis: Extracts features from beginning, middle, and end (30s each)
    - Parallel processing: Multi-core CPU utilization for faster analysis
    - Local analysis: Uses Librosa for feature extraction
    - Extracts MFCC, chroma, spectral features, tempo, etc.
    - Backward compatible: Handles both old and new feature formats
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


def analyze_track_worker(track_data: Tuple[str, str, str, str, bool]) -> Optional[Tuple[str, Dict[str, Any], str, str]]:
    """
    Worker function for parallel track analysis

    Args:
        track_data: Tuple of (track_id, file_path, artist, title, use_beat_sync)

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

    try:
        # Check file exists
        if not Path(file_path).exists():
            return None

        # Create analyzer instance (each worker needs its own)
        analyzer = HybridSonicAnalyzer(use_beat_sync=use_beat_sync)

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

    def __init__(self, db_path: Optional[str] = None, use_beat_sync: bool = False):
        """
        Initialize pipeline

        Args:
            db_path: Path to metadata database
            use_beat_sync: If True, use beat-synchronized feature extraction (Phase 2)
                          If False, use fixed-window extraction (legacy)
        """
        from src.hybrid_sonic_analyzer import HybridSonicAnalyzer

        default_db = ROOT_DIR / "data" / "metadata.db"
        self.db_path = str(Path(db_path)) if db_path else str(default_db)
        self.conn = None
        self.use_beat_sync = use_beat_sync
        self.analyzer = HybridSonicAnalyzer(use_beat_sync=use_beat_sync)
        self._init_db_connection()
        mode = "beat-sync" if use_beat_sync else "windowed"
        logger.info(f"Initialized SonicFeaturePipeline (mode={mode})")

    def _init_db_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.debug(f"Connected to database: {self.db_path}")

    def get_pending_tracks(self, limit: Optional[int] = None, force: bool = False) -> list:
        """
        Get tracks that need sonic analysis

        Args:
            limit: Maximum number of tracks to return (None = all)
            force: If True, re-analyze all tracks (even those already analyzed)

        Returns:
            List of track rows
        """
        cursor = self.conn.cursor()

        if force:
            # Re-analyze everything (convert old format to multi-segment)
            query = """
                SELECT track_id, file_path, musicbrainz_id, artist, title
                FROM tracks
                WHERE file_path IS NOT NULL
                ORDER BY
                    CASE WHEN musicbrainz_id IS NOT NULL THEN 0 ELSE 1 END,
                    track_id
            """
        else:
            # Only analyze tracks without features
            query = """
                SELECT track_id, file_path, musicbrainz_id, artist, title
                FROM tracks
                WHERE file_path IS NOT NULL
                  AND sonic_features IS NULL
                ORDER BY
                    CASE WHEN musicbrainz_id IS NOT NULL THEN 0 ELSE 1 END,
                    track_id
            """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return cursor.fetchall()

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

        # Convert to JSON (remove 'source' as it's in separate column)
        features_copy = features.copy()
        features_copy.pop('source', None)
        features_json = json.dumps(features_copy)

        # Update database
        cursor.execute("""
            UPDATE tracks
            SET sonic_features = ?,
                sonic_source = ?,
                sonic_analyzed_at = ?
            WHERE track_id = ?
        """, (features_json, source, int(time.time()), track_id))

        if commit:
            self.conn.commit()

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
            (track['track_id'], track['file_path'], track['artist'], track['title'], self.use_beat_sync)
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
        """Get analysis statistics"""
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

        return {
            'total_tracks': total,
            'analyzed': analyzed,
            'pending': total - analyzed,
            'librosa': sources.get('librosa', 0)
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
    parser.add_argument('--beat-sync', action='store_true', help='Use beat-synchronized feature extraction (Phase 2)')
    args = parser.parse_args()

    pipeline = SonicFeaturePipeline(use_beat_sync=args.beat_sync)

    if args.stats:
        # Show statistics
        stats = pipeline.get_stats()
        print("\nSonic Analysis Statistics:")
        print(f"  Total tracks: {stats['total_tracks']}")
        print(f"  Analyzed: {stats['analyzed']}")
        print(f"  Pending: {stats['pending']}")
        print(f"  Librosa: {stats['librosa']}")
    else:
        # Run analysis
        if args.force:
            print("\n[!] FORCE MODE: Re-analyzing ALL tracks with multi-segment extraction")
            print("    This will update old single-segment features to the new format.\n")
        pipeline.run(limit=args.limit, workers=args.workers, force=args.force)

    pipeline.close()
