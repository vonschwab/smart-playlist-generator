#!/usr/bin/env python3
"""
Music Library Scanner
=====================
Scans the local music library and updates the metadata database.

Features:
- Scans for audio files (MP3, FLAC, M4A, OGG, OPUS, WMA, WAV)
- Extracts metadata from file tags (artist, title, album, genres, etc.)
- Adds new tracks to database
- Updates existing tracks if metadata has changed
- Tracks file modifications via mtime
- Multi-threaded for performance

Usage:
    python scan_library.py                  # Full library scan
    python scan_library.py --quick          # Only scan new/modified files
    python scan_library.py --stats          # Show statistics only
    python scan_library.py --limit 100      # Scan up to 100 files
"""
import sys
import sqlite3
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config

# Audio file extensions to scan
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.ogg', '.opus', '.wma', '.wav', '.aac'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LibraryScanner:
    """Scans music library and updates metadata database"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize scanner"""
        if config_path is None:
            config_path = ROOT_DIR / 'config.yaml'
        self.config = Config(config_path)
        self.music_dir = Path(self.config.get('library', 'music_directory'))
        # Use data/metadata.db as the actual path
        self.db_path = ROOT_DIR / 'data' / 'metadata.db'
        self.conn = None

        # Try to import mutagen
        try:
            import mutagen
            from mutagen.easyid3 import EasyID3
            from mutagen.flac import FLAC
            from mutagen.mp4 import MP4
            from mutagen.oggvorbis import OggVorbis
            from mutagen.oggopus import OggOpus
            self.mutagen = mutagen
            self.has_mutagen = True
        except ImportError:
            logger.warning("mutagen not installed - will use basic metadata extraction")
            logger.warning("Install with: pip install mutagen")
            self.has_mutagen = False

        self._init_db()

    def _init_db(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row

    def generate_track_id(self, file_path: str, artist: str, title: str) -> str:
        """
        Generate a unique track ID from file path, artist, and title

        Args:
            file_path: Path to audio file
            artist: Artist name
            title: Track title

        Returns:
            MD5 hash as track ID
        """
        # Use file path + artist + title for uniqueness
        unique_string = f"{file_path}|{artist}|{title}".lower()
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def extract_metadata(self, file_path: Path) -> Optional[Dict]:
        """
        Extract metadata from audio file

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary of metadata or None if extraction failed
        """
        if not self.has_mutagen:
            # Basic fallback - just use filename
            return self._extract_metadata_fallback(file_path)

        try:
            audio = self.mutagen.File(file_path, easy=True)
            if audio is None:
                logger.debug(f"Could not read file: {file_path}")
                return None

            # Extract common tags
            metadata = {
                'artist': self._get_tag(audio, ['artist', 'albumartist', 'TPE1']),
                'title': self._get_tag(audio, ['title', 'TIT2']),
                'album': self._get_tag(audio, ['album', 'TALB']),
                'date': self._get_tag(audio, ['date', 'year', 'TDRC']),
                'genre': self._get_tags_list(audio, ['genre', 'TCON']),
                'duration': getattr(audio.info, 'length', None),
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_modified': int(file_path.stat().st_mtime)
            }

            # If no artist or title, try to parse from filename
            if not metadata['artist'] or not metadata['title']:
                fallback = self._parse_filename(file_path)
                metadata['artist'] = metadata['artist'] or fallback['artist']
                metadata['title'] = metadata['title'] or fallback['title']

            return metadata

        except Exception as e:
            logger.debug(f"Error extracting metadata from {file_path}: {e}")
            return self._extract_metadata_fallback(file_path)

    def _get_tag(self, audio, tag_names: List[str]) -> Optional[str]:
        """Get first available tag value"""
        for tag in tag_names:
            if tag in audio:
                value = audio[tag]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif value:
                    return str(value)
        return None

    def _get_tags_list(self, audio, tag_names: List[str]) -> List[str]:
        """Get tag values as list"""
        for tag in tag_names:
            if tag in audio:
                value = audio[tag]
                if isinstance(value, list):
                    return [str(v) for v in value]
                elif value:
                    return [str(value)]
        return []

    def _parse_filename(self, file_path: Path) -> Dict[str, str]:
        """Parse artist and title from filename"""
        filename = file_path.stem

        # Try common patterns: "Artist - Title" or "Artist-Title"
        if ' - ' in filename:
            parts = filename.split(' - ', 1)
            return {'artist': parts[0].strip(), 'title': parts[1].strip()}
        elif '-' in filename:
            parts = filename.split('-', 1)
            return {'artist': parts[0].strip(), 'title': parts[1].strip()}

        # Fallback: use parent directory as artist
        artist = file_path.parent.name
        return {'artist': artist, 'title': filename}

    def _extract_metadata_fallback(self, file_path: Path) -> Dict:
        """Fallback metadata extraction without mutagen"""
        parsed = self._parse_filename(file_path)
        return {
            'artist': parsed['artist'],
            'title': parsed['title'],
            'album': None,
            'date': None,
            'genre': [],
            'duration': None,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_modified': int(file_path.stat().st_mtime)
        }

    def scan_files(self, quick: bool = False) -> List[Path]:
        """
        Scan music directory for audio files

        Args:
            quick: If True, only return new/modified files

        Returns:
            List of file paths to process
        """
        logger.info(f"Scanning directory: {self.music_dir}")

        files = []
        for ext in AUDIO_EXTENSIONS:
            pattern = f"**/*{ext}"
            files.extend(self.music_dir.glob(pattern))

        logger.info(f"Found {len(files)} audio files")

        if quick:
            # Filter to only new/modified files
            files = self._filter_modified_files(files)
            logger.info(f"  {len(files)} are new or modified")

        return files

    def _filter_modified_files(self, files: List[Path]) -> List[Path]:
        """Filter to only files that are new or have been modified"""
        cursor = self.conn.cursor()
        modified = []

        for file_path in files:
            file_str = str(file_path)
            file_modified = int(file_path.stat().st_mtime)

            # Check if file exists in database
            cursor.execute("""
                SELECT file_modified FROM tracks WHERE file_path = ?
            """, (file_str,))

            row = cursor.fetchone()
            if row is None:
                # New file
                modified.append(file_path)
            elif row['file_modified'] is None or row['file_modified'] < file_modified:
                # Modified file
                modified.append(file_path)

        return modified

    def upsert_track(self, metadata: Dict) -> Tuple[str, bool]:
        """
        Insert or update track in database

        Args:
            metadata: Track metadata dictionary

        Returns:
            Tuple of (track_id, is_new)
        """
        cursor = self.conn.cursor()

        artist = metadata['artist'] or 'Unknown Artist'
        title = metadata['title'] or 'Unknown Title'

        # Generate track ID
        track_id = self.generate_track_id(metadata['file_path'], artist, title)

        # Check if track exists by file_path (not track_id to prevent duplicates)
        cursor.execute("SELECT track_id FROM tracks WHERE file_path = ?", (metadata['file_path'],))
        existing = cursor.fetchone()

        if existing:
            # Update existing track (update track_id too in case metadata changed)
            old_track_id = existing[0]

            # Convert duration from seconds to milliseconds
            duration_ms = int(metadata['duration'] * 1000) if metadata.get('duration') else None

            cursor.execute("""
                UPDATE tracks
                SET track_id = ?,
                    artist = ?,
                    title = ?,
                    album = ?,
                    duration_ms = ?,
                    file_path = ?,
                    file_modified = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE file_path = ?
            """, (
                track_id,
                artist,
                title,
                metadata.get('album'),
                duration_ms,
                metadata['file_path'],
                metadata.get('file_modified'),
                metadata['file_path']
            ))

            # If track_id changed, update foreign key references
            if old_track_id != track_id:
                cursor.execute("""
                    UPDATE track_genres SET track_id = ? WHERE track_id = ?
                """, (track_id, old_track_id))

            is_new = False
        else:
            # Insert new track
            # Convert duration from seconds to milliseconds
            duration_ms = int(metadata['duration'] * 1000) if metadata.get('duration') else None

            cursor.execute("""
                INSERT INTO tracks (
                    track_id, artist, title, album, duration_ms, file_path, file_modified
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                artist,
                title,
                metadata.get('album'),
                duration_ms,
                metadata['file_path'],
                metadata.get('file_modified')
            ))
            is_new = True

        # Handle genres from file tags
        if metadata.get('genre'):
            self._upsert_genres(track_id, metadata['genre'])

        return track_id, is_new

    def _upsert_genres(self, track_id: str, genres: List[str]):
        """Insert or update genres for a track from file tags"""
        cursor = self.conn.cursor()

        # Remove existing file-sourced genres
        cursor.execute("""
            DELETE FROM track_genres
            WHERE track_id = ? AND source = 'file'
        """, (track_id,))

        # Insert new genres
        for genre in genres:
            genre = genre.strip().lower()
            if genre:
                cursor.execute("""
                    INSERT OR IGNORE INTO track_genres (track_id, genre, source)
                    VALUES (?, ?, 'file')
                """, (track_id, genre))

    def run(self, quick: bool = False, limit: Optional[int] = None):
        """
        Run the library scan

        Args:
            quick: If True, only scan new/modified files
            limit: Maximum number of files to process
        """
        logger.info("=" * 70)
        logger.info("Music Library Scanner")
        logger.info("=" * 70)

        # Scan for files
        files = self.scan_files(quick=quick)

        if limit:
            files = files[:limit]
            logger.info(f"Limited to {limit} files")

        if not files:
            logger.info("No files to process")
            return

        # Process files
        stats = {
            'total': len(files),
            'new': 0,
            'updated': 0,
            'failed': 0
        }

        logger.info(f"\nProcessing {len(files)} files...")

        for i, file_path in enumerate(files, 1):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(files)} ({i/len(files)*100:.1f}%)")

            try:
                # Extract metadata
                metadata = self.extract_metadata(file_path)
                if metadata is None:
                    stats['failed'] += 1
                    continue

                # Upsert track
                track_id, is_new = self.upsert_track(metadata)

                if is_new:
                    stats['new'] += 1
                else:
                    stats['updated'] += 1

                # Commit every 100 tracks
                if i % 100 == 0:
                    self.conn.commit()

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['failed'] += 1

        # Final commit
        self.conn.commit()

        # Show results
        logger.info("\n" + "=" * 70)
        logger.info("Scan Complete")
        logger.info("=" * 70)
        logger.info(f"  Total processed: {stats['total']}")
        logger.info(f"  New tracks: {stats['new']}")
        logger.info(f"  Updated tracks: {stats['updated']}")
        logger.info(f"  Failed: {stats['failed']}")

    def get_stats(self):
        """Show database statistics"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM tracks")
        total_tracks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL")
        tracks_with_files = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT track_id) FROM track_genres WHERE source = 'file'")
        tracks_with_file_genres = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT track_id) FROM track_genres")
        tracks_with_any_genres = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL")
        tracks_with_sonic = cursor.fetchone()[0]

        print("\nLibrary Statistics:")
        print("=" * 70)
        print(f"  Total tracks: {total_tracks:,}")
        print(f"  Tracks with file paths: {tracks_with_files:,} ({tracks_with_files/total_tracks*100:.1f}%)")
        print(f"  Tracks with file genres: {tracks_with_file_genres:,} ({tracks_with_file_genres/total_tracks*100:.1f}%)")
        print(f"  Tracks with any genres: {tracks_with_any_genres:,} ({tracks_with_any_genres/total_tracks*100:.1f}%)")
        print(f"  Tracks with sonic features: {tracks_with_sonic:,} ({tracks_with_sonic/total_tracks*100:.1f}%)")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scan music library and update metadata')
    parser.add_argument('--quick', action='store_true',
                       help='Only scan new/modified files')
    parser.add_argument('--limit', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics only')
    args = parser.parse_args()

    scanner = LibraryScanner()

    if args.stats:
        scanner.get_stats()
    else:
        scanner.run(quick=args.quick, limit=args.limit)

    scanner.close()
