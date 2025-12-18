#!/usr/bin/env python3
"""
Re-Normalize Existing Genres in Database
==========================================
This script re-normalizes all existing genre tags in the database using the new
normalization system. This includes:
- Translating foreign language tags (French, German, Dutch) to English
- Mapping synonyms to canonical forms
- Splitting composite tags
- Filtering overly-broad tags

This is a one-time migration script to clean up existing data.

Usage:
    python scripts/normalize_existing_genres.py                # Dry run (show what would change)
    python scripts/normalize_existing_genres.py --apply        # Actually update database
    python scripts/normalize_existing_genres.py --stats        # Show statistics
"""

import sys
import sqlite3
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import defaultdict

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.genre_normalization import normalize_genre_list
from src.config_loader import Config

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenreNormalizer:
    """Re-normalize existing genres in database"""

    def __init__(self, db_path: str = 'data/metadata.db'):
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()

    def _connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path, timeout=60.0)
        self.conn.row_factory = sqlite3.Row

    def analyze_genres(self) -> Dict[str, any]:
        """Analyze current genre state"""
        cursor = self.conn.cursor()

        stats = {}

        # Track genres
        cursor.execute("SELECT COUNT(DISTINCT genre) FROM track_genres")
        stats['unique_track_genres'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM track_genres")
        stats['total_track_genre_entries'] = cursor.fetchone()[0]

        # Artist genres
        cursor.execute("SELECT COUNT(DISTINCT genre) FROM artist_genres")
        stats['unique_artist_genres'] = cursor.fetchone()[0]

        # Album genres
        cursor.execute("SELECT COUNT(DISTINCT genre) FROM album_genres")
        stats['unique_album_genres'] = cursor.fetchone()[0]

        # Find foreign language tags
        cursor.execute("""
            SELECT genre, COUNT(*) as count
            FROM track_genres
            WHERE genre LIKE '%alternatif%'
               OR genre LIKE '%indé%'
               OR genre LIKE '%inde%'
               OR genre LIKE '%alternativ%'
               OR genre LIKE '%en indie%'
            GROUP BY genre
            ORDER BY count DESC
        """)
        stats['foreign_language_tags'] = cursor.fetchall()

        # Find composite tags (with commas, semicolons, slashes)
        cursor.execute("""
            SELECT genre, COUNT(*) as count
            FROM track_genres
            WHERE genre LIKE '%,%'
               OR genre LIKE '%;%'
               OR genre LIKE '%/%'
            GROUP BY genre
            ORDER BY count DESC
            LIMIT 20
        """)
        stats['composite_tags'] = cursor.fetchall()

        return stats

    def preview_normalization(self, limit: int = 50) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Preview what would change with normalization.

        Returns:
            (changes_dict, stats_dict)
            changes_dict: {old_genre: {normalized_genres}}
            stats_dict: counts of changes
        """
        cursor = self.conn.cursor()

        # Get all unique genres from all tables
        all_genres = set()

        cursor.execute("SELECT DISTINCT genre FROM track_genres")
        all_genres.update(row[0] for row in cursor.fetchall())

        cursor.execute("SELECT DISTINCT genre FROM artist_genres")
        all_genres.update(row[0] for row in cursor.fetchall())

        cursor.execute("SELECT DISTINCT genre FROM album_genres")
        all_genres.update(row[0] for row in cursor.fetchall())

        logger.info(f"Analyzing {len(all_genres)} unique genres...")

        # Normalize each genre
        changes = {}
        stats = {
            'genres_unchanged': 0,
            'genres_normalized': 0,
            'genres_split': 0,
            'genres_filtered_out': 0,
            'translations_applied': 0,
        }

        for old_genre in sorted(all_genres):
            normalized = normalize_genre_list([old_genre], filter_broad=True)

            if not normalized:
                # Genre was filtered out completely
                stats['genres_filtered_out'] += 1
                changes[old_genre] = set()
            elif normalized == {old_genre}:
                # No change
                stats['genres_unchanged'] += 1
            else:
                # Changed
                stats['genres_normalized'] += 1
                if len(normalized) > 1:
                    stats['genres_split'] += 1

                # Check if this was a translation
                if any(term in old_genre for term in ['alternatif', 'indé', 'alternativ', 'en indie']):
                    stats['translations_applied'] += 1

                changes[old_genre] = normalized

        return changes, stats

    def apply_normalization(self, dry_run: bool = True):
        """
        Apply normalization to all genres in database.

        Args:
            dry_run: If True, only show what would change without modifying database
        """
        cursor = self.conn.cursor()

        # Get changes preview
        changes, stats = self.preview_normalization()

        logger.info("=" * 80)
        logger.info("NORMALIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total unique genres: {len(changes)}")
        logger.info(f"Unchanged: {stats['genres_unchanged']}")
        logger.info(f"Normalized: {stats['genres_normalized']}")
        logger.info(f"Split into multiple: {stats['genres_split']}")
        logger.info(f"Filtered out: {stats['genres_filtered_out']}")
        logger.info(f"Translations applied: {stats['translations_applied']}")
        logger.info("=" * 80)

        if dry_run:
            logger.info("\n*** DRY RUN - No changes will be made ***")
            logger.info("\nSample changes (first 20):")
            count = 0
            for old_genre, new_genres in changes.items():
                if old_genre in new_genres and len(new_genres) == 1:
                    continue  # Skip unchanged
                count += 1
                if count > 20:
                    break
                if new_genres:
                    logger.info(f"  '{old_genre}' -> {sorted(new_genres)}")
                else:
                    logger.info(f"  '{old_genre}' -> [FILTERED OUT]")
            logger.info(f"\n... and {max(0, stats['genres_normalized'] - 20)} more changes")
            logger.info("\nRun with --apply to actually update the database")
            return

        logger.info("\nApplying changes to database...")

        # Process each table
        tables = [
            ('track_genres', 'track_id'),
            ('artist_genres', 'artist'),
            ('album_genres', 'album_id'),
        ]

        for table_name, id_column in tables:
            logger.info(f"\nProcessing {table_name}...")

            # Get all rows
            cursor.execute(f"SELECT ROWID, {id_column}, genre, source FROM {table_name}")
            rows = cursor.fetchall()

            updates = 0
            deletes = 0
            inserts = 0

            for row in rows:
                rowid, entity_id, old_genre, source = row

                if old_genre not in changes:
                    continue

                new_genres = changes[old_genre]

                # Delete old entry
                cursor.execute(f"DELETE FROM {table_name} WHERE ROWID = ?", (rowid,))
                deletes += 1

                # Insert normalized genres
                for new_genre in new_genres:
                    try:
                        cursor.execute(f"""
                            INSERT OR IGNORE INTO {table_name} ({id_column}, genre, source)
                            VALUES (?, ?, ?)
                        """, (entity_id, new_genre, source))
                        inserts += 1
                    except sqlite3.IntegrityError:
                        pass  # Genre already exists for this entity

            logger.info(f"  Deleted {deletes} old entries, inserted {inserts} new entries")

        self.conn.commit()
        logger.info("\nNormalization complete!")

    def show_stats(self):
        """Show genre statistics"""
        stats = self.analyze_genres()

        print("\n" + "=" * 80)
        print("GENRE DATABASE STATISTICS")
        print("=" * 80)
        print(f"\nUnique track genres: {stats['unique_track_genres']}")
        print(f"Total track genre entries: {stats['total_track_genre_entries']}")
        print(f"Unique artist genres: {stats['unique_artist_genres']}")
        print(f"Unique album genres: {stats['unique_album_genres']}")

        if stats['foreign_language_tags']:
            print(f"\nForeign language tags found: {len(stats['foreign_language_tags'])}")
            print("\nTop foreign language tags:")
            for row in stats['foreign_language_tags'][:10]:
                print(f"  {row['count']:5d}  {row['genre']}")

        if stats['composite_tags']:
            print(f"\nComposite tags found: {len(stats['composite_tags'])}")
            print("\nTop composite tags:")
            for row in stats['composite_tags'][:10]:
                print(f"  {row['count']:5d}  {row['genre']}")

        print("=" * 80)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Re-normalize existing genres in database')
    parser.add_argument('--apply', action='store_true', help='Actually apply changes (default is dry run)')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--db', type=str, default='data/metadata.db', help='Database path')

    args = parser.parse_args()

    normalizer = GenreNormalizer(db_path=args.db)

    try:
        if args.stats:
            normalizer.show_stats()
        else:
            normalizer.apply_normalization(dry_run=not args.apply)
    finally:
        normalizer.close()


if __name__ == '__main__':
    main()
