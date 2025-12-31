#!/usr/bin/env python3
"""
Music Library Scanner
=====================
Scans the local music library and updates the metadata database.

Features:
- Scans for audio files (MP3, FLAC, M4A, OGG, OPUS, WMA, WAV)
- Extracts metadata from file tags (artist, title, album, genres, duration, etc.)
- Adds new tracks to database
- Updates existing tracks if metadata has changed
- Tracks file modifications via mtime
- Automatically extracts track duration via Mutagen
- Optional: removes tracks with missing files from database
- Comprehensive logging for duration and fallback extraction

Usage:
    python scan_library.py                          # Full library scan
    python scan_library.py --quick                  # Only scan new/modified files
    python scan_library.py --cleanup                # Remove missing files first, then scan
    python scan_library.py --cleanup --quick        # Clean up, then quick scan
    python scan_library.py --stats                  # Show statistics only
    python scan_library.py --limit 100              # Scan up to 100 files

Duration Extraction:
    - Automatically extracted for all supported audio formats
    - Stored in milliseconds (duration_ms) in database
    - If extraction fails, logs warning and stores NULL
    - Use scripts/backfill_duration.py to fix missing durations

File Cleanup:
    - Use --cleanup to remove tracks from database if files are deleted
    - Useful after external file deletion to keep DB in sync
    - Always runs before new scan to prevent re-adding deleted tracks
"""
import sys
import sqlite3
import hashlib
import os
import unicodedata
import random
import time
import os.path
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.artist_key_db import ensure_artist_key_schema
from src.config_loader import Config
from src.genre_normalization import normalize_genre_list
from src.string_utils import normalize_artist_key
from src.logging_utils import ProgressLogger

# Audio file extensions to scan
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.ogg', '.opus', '.wma', '.wav', '.aac'}

# Logging will be configured in main() - just get the logger here
logger = logging.getLogger('scan_library')


class LibraryScanner:
    """Scans music library and updates metadata database"""

    def __init__(self, config_path: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize scanner"""
        if config_path is None:
            config_path = ROOT_DIR / 'config.yaml'
        self.config = Config(config_path)
        self.music_dir = Path(self.config.get('library', 'music_directory'))
        # Use data/metadata.db as the actual path
        self.db_path = Path(db_path) if db_path else ROOT_DIR / 'data' / 'metadata.db'
        self.conn = None
        self._has_album_id = False
        self._has_album_artist = False
        self._has_fingerprints = False
        self._last_modified_reasons: Dict[str, int] = {}
        self._last_modified_examples: Dict[str, List[str]] = {}

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

    @staticmethod
    def _normalize_path_str(path: Path) -> str:
        """Normalize path for comparisons (case-fold + normalized separators)."""
        return os.path.normcase(os.path.normpath(str(path)))

    def _init_db(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._has_album_id = self._column_exists("tracks", "album_id")
        self._has_album_artist = self._column_exists("tracks", "album_artist")
        self._has_fingerprints = self._column_exists("tracks", "file_mtime_ns")
        if not self._has_fingerprints:
            try:
                self.conn.execute("ALTER TABLE tracks ADD COLUMN file_mtime_ns INTEGER")
                self.conn.execute("ALTER TABLE tracks ADD COLUMN file_size_bytes INTEGER")
                self.conn.execute("ALTER TABLE tracks ADD COLUMN tags_fingerprint TEXT")
                self.conn.commit()
                self._has_fingerprints = True
            except sqlite3.OperationalError:
                self.conn.rollback()
                self._has_fingerprints = self._column_exists("tracks", "file_mtime_ns")
        ensure_artist_key_schema(self.conn, logger=logger)

    def _table_exists(self, table_name: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",  
            (table_name,),
        )
        return cursor.fetchone() is not None

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
        except sqlite3.OperationalError:
            return False
        for row in cursor.fetchall():
            try:
                name = row["name"]
            except Exception:
                name = row[1]
            if name == column_name:
                return True
        return False

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

    def _compute_album_id(self, album_artist: Optional[str], album_title: Optional[str]) -> Optional[str]:
        """
        Compute a stable album_id from album artist/title.

        Normalizes Unicode (NFKD), casefolds, collapses whitespace, then hashes.
        Returns None when album title is missing/empty.
        """
        if not album_title or not str(album_title).strip():
            return None

        def _norm(value: Optional[str]) -> str:
            if not value:
                return ""
            text = unicodedata.normalize("NFKD", str(value))
            text = "".join(ch for ch in text if not unicodedata.combining(ch))
            text = text.casefold()
            return " ".join(text.split())

        title_norm = _norm(album_title)
        if not title_norm:
            return None

        artist_norm = _norm(album_artist)
        key = f"{artist_norm}|{title_norm}" if artist_norm else title_norm
        return hashlib.md5(key.encode('utf-8')).hexdigest()[:16]

    def _compute_tags_fingerprint(self, metadata: Dict) -> str:
        """
        Hash normalized key tags to detect tag-only changes.
        """
        parts = [
            metadata.get("artist") or "",
            metadata.get("album_artist") or "",
            metadata.get("title") or "",
            metadata.get("album") or "",
            str(metadata.get("duration") or ""),
        ]
        joined = "|".join(" ".join(str(p).casefold().split()) for p in parts)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    def _upsert_album(self, album_id: str, album_artist: Optional[str], album_title: Optional[str]) -> None:
        """Insert album into albums table if present."""
        if not self._table_exists("albums"):
            return
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO albums (album_id, artist, title)
                VALUES (?, ?, ?)
                """,
                (album_id, album_artist or "", album_title or ""),
            )
        except sqlite3.OperationalError:
            # albums table exists but schema differs; skip
            return

    def _find_move_candidate(self, artist: str, title: str, album_title: Optional[str], album_id: Optional[str]):
        """
        Detect a likely moved track by matching artist/title/album (or album_id) where existing file no longer exists.
        Returns (rowid, track_id, file_path) or None.
        """
        cursor = self.conn.cursor()

        # Prefer album_id match when available
        if album_id:
            row = cursor.execute(
                """
                SELECT rowid, track_id, file_path
                FROM tracks
                WHERE album_id = ?
                  AND artist IS NOT NULL AND title IS NOT NULL
                LIMIT 1
                """,
                (album_id,),
            ).fetchone()
            if row and row["file_path"] and not Path(row["file_path"]).exists():
                return row

        # Fallback: case-insensitive artist+title+album match
        row = cursor.execute(
            """
            SELECT rowid, track_id, file_path
            FROM tracks
            WHERE LOWER(artist) = LOWER(?)
              AND LOWER(title) = LOWER(?)
              AND (album IS NULL OR album = ? OR LOWER(album) = LOWER(?))
            LIMIT 1
            """,
            (artist, title, album_title, album_title),
        ).fetchone()
        if row and row["file_path"] and not Path(row["file_path"]).exists():
            return row
        return None

    def _rewrite_track_id(self, old_track_id: str, new_track_id: str) -> None:
        """
        Rewrite track_id across dependent tables (best-effort; only if tables exist).
        """
        if old_track_id == new_track_id:
            return
        cursor = self.conn.cursor()
        try:
            cursor.execute("UPDATE track_genres SET track_id = ? WHERE track_id = ?", (new_track_id, old_track_id))
        except sqlite3.OperationalError:
            pass
        if self._table_exists("track_effective_genres"):
            try:
                cursor.execute(
                    "UPDATE track_effective_genres SET track_id = ? WHERE track_id = ?",
                    (new_track_id, old_track_id),
                )
            except sqlite3.OperationalError:
                pass

    def verify_fingerprints(self, sample_size: int = 20) -> Dict[str, int]:
        """
        Sample tracks and verify tags_fingerprint matches current stored tags.
        No writes are performed.
        """
        stats = {"checked": 0, "mismatched": 0}
        if not self._has_fingerprints:
            logger.warning("tags_fingerprint column missing; nothing to verify.")
            return stats
        cursor = self.conn.cursor()
        rows = cursor.execute(
            """
            SELECT track_id, artist, title, album, album_artist, duration_ms, tags_fingerprint
            FROM tracks
            WHERE tags_fingerprint IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (sample_size,),
        ).fetchall()
        for row in rows:
            meta = {
                "artist": row["artist"],
                "album_artist": row["album_artist"] if "album_artist" in row.keys() else None,
                "title": row["title"],
                "album": row["album"],
                "duration": (row["duration_ms"] / 1000.0) if row["duration_ms"] is not None else None,
            }
            recomputed = self._compute_tags_fingerprint(meta)
            stats["checked"] += 1
            if recomputed != row["tags_fingerprint"]:
                stats["mismatched"] += 1
                logger.warning("Fingerprint mismatch for track_id=%s", row["track_id"])
        logger.info("Fingerprint verify: checked=%d mismatched=%d", stats["checked"], stats["mismatched"])
        return stats

    def extract_metadata(self, file_path: Path) -> Optional[Dict]:
        """
        Extract metadata from audio file

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary of metadata or None if extraction failed
        """
        try:
            stat = file_path.stat()
            mtime_ns = stat.st_mtime_ns
            size_bytes = stat.st_size
        except FileNotFoundError:
            return None

        if not self.has_mutagen:
            # Basic fallback - just use filename
            return self._extract_metadata_fallback(file_path)

        try:
            audio = self.mutagen.File(file_path, easy=True)
            if audio is None:
                logger.warning(f"Could not read file format: {file_path}")
                return None

            # Extract common tags
            metadata = {
                'artist': self._get_tag(audio, ['artist', 'albumartist', 'TPE1']),
                'album_artist': self._get_tag(audio, ['albumartist', 'TPE2']),
                'title': self._get_tag(audio, ['title', 'TIT2']),
                'album': self._get_tag(audio, ['album', 'TALB']),
                'date': self._get_tag(audio, ['date', 'year', 'TDRC']),
                'genre': self._get_tags_list(audio, ['genre', 'TCON']),
                'duration': getattr(audio.info, 'length', None),
                'file_path': str(file_path),
                'file_size': size_bytes,
                'file_modified': int(stat.st_mtime),
                'file_mtime_ns': mtime_ns,
            }

            # Warn if duration is missing
            if metadata['duration'] is None:
                logger.warning(f"Could not extract duration from {file_path} - audio info missing")

            # If no artist or title, try to parse from filename
            if not metadata['artist'] or not metadata['title']:
                fallback = self._parse_filename(file_path)
                metadata['artist'] = metadata['artist'] or fallback['artist']
                metadata['title'] = metadata['title'] or fallback['title']

            return metadata

        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")
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
        try:
            stat = file_path.stat()
            mtime_ns = stat.st_mtime_ns
            size_bytes = stat.st_size
            file_modified = int(stat.st_mtime)
        except FileNotFoundError:
            mtime_ns = None
            size_bytes = None
            file_modified = None
        parsed = self._parse_filename(file_path)
        logger.warning(f"Using fallback extraction for {file_path} - duration not available")
        return {
            'artist': parsed['artist'],
            'title': parsed['title'],
            'album': None,
            'date': None,
            'genre': [],
            'duration': None,
            'file_path': str(file_path),
            'file_size': size_bytes,
            'file_mtime_ns': mtime_ns,
            'file_modified': file_modified,
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

        self._last_modified_reasons = {}
        self._last_modified_examples = {}
        if quick:
            # Filter to only new/modified files
            files, reasons, examples = self._filter_modified_files(files)
            self._last_modified_reasons = reasons
            self._last_modified_examples = examples
            if reasons:
                reason_bits = [f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: kv[0])]
                logger.info("  Modified breakdown: %s", ", ".join(reason_bits))
                top_reasons = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:2]
                for reason, _ in top_reasons:
                    ex = examples.get(reason) or []
                    if ex:
                        logger.debug("    %s examples: %s", reason, "; ".join(ex))
            logger.info(f"  {len(files)} are new or modified")

        return files

    def _filter_modified_files(self, files: List[Path]) -> Tuple[List[Path], Dict[str, int], Dict[str, List[str]]]:
        """Filter to only files that are new or have been modified"""
        cursor = self.conn.cursor()
        modified = []
        reason_counts: Dict[str, int] = {}
        reason_examples: Dict[str, List[str]] = {}

        # Preload file fingerprints to avoid per-file queries
        if self._has_fingerprints:
            cursor.execute(
                "SELECT file_path, file_mtime_ns, file_size_bytes, file_modified FROM tracks WHERE file_path IS NOT NULL"
            )
            db_info = {
                self._normalize_path_str(row["file_path"]): (
                    row["file_path"],
                    row["file_mtime_ns"],
                    row["file_size_bytes"],
                    row["file_modified"],
                )
                for row in cursor.fetchall()
            }
        else:
            cursor.execute("SELECT file_path, file_modified FROM tracks WHERE file_path IS NOT NULL")
            db_info = {
                self._normalize_path_str(row["file_path"]): (row["file_path"], row["file_modified"], None, row["file_modified"])
                for row in cursor.fetchall()
            }

        def _add_reason(reason: str, file_path: Path) -> None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            examples = reason_examples.setdefault(reason, [])
            if len(examples) < 3:
                try:
                    examples.append(str(file_path.relative_to(self.music_dir)))
                except Exception:
                    examples.append(str(file_path))

        for file_path in files:
            file_str = self._normalize_path_str(file_path)
            try:
                stat = file_path.stat()
                file_mtime_ns = stat.st_mtime_ns
                file_size = stat.st_size
            except FileNotFoundError:
                continue

            db_entry = db_info.get(file_str)
            if db_entry is None:
                modified.append(file_path)  # New file
                _add_reason("new_not_in_db", file_path)
                continue
            stored_path, db_mtime_ns, db_size, db_file_modified = db_entry
            if stored_path and stored_path != str(file_path) and self._normalize_path_str(stored_path) == file_str:
                _add_reason("path_normalized_match", file_path)

            if db_mtime_ns is not None or db_size is not None:
                if (db_mtime_ns is not None and db_mtime_ns != file_mtime_ns) or (db_size is not None and db_size != file_size):
                    modified.append(file_path)  # Modified file
                    _add_reason("stat_changed", file_path)
                # else unchanged
                continue

            # Fallback when fingerprint columns exist but values are null; use legacy file_modified seconds if present
            if db_file_modified is not None and int(db_file_modified) == int(stat.st_mtime):
                continue

            modified.append(file_path)
            _add_reason("missing_fingerprint", file_path)

        return modified, reason_counts, reason_examples

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
        album_artist = metadata.get('album_artist') or artist
        title = metadata['title'] or 'Unknown Title'
        artist_key = normalize_artist_key(artist)
        album_title = metadata.get('album')
        album_id = self._compute_album_id(album_artist, album_title) if self._has_album_id else None

        # Generate track ID
        track_id = self.generate_track_id(metadata['file_path'], artist, title)

        # Check if track exists by file_path (not track_id to prevent duplicates)
        cursor.execute(
            """
            SELECT rowid, track_id, file_path, file_mtime_ns, file_size_bytes, tags_fingerprint
            FROM tracks WHERE file_path = ?
            """,
            (metadata['file_path'],),
        )
        existing_row = cursor.fetchone()
        moved_candidate = None
        if not existing_row:
            moved_candidate = self._find_move_candidate(artist, title, album_title, album_id)

        existing = existing_row or moved_candidate

        if existing:
            # Update existing track (update track_id too in case metadata changed)
            existing_rowid = existing[0]
            old_track_id = existing[1]
            existing_mtime_ns = existing[3] if len(existing) > 3 else None
            existing_size = existing[4] if len(existing) > 4 else None
            existing_tags_fp = existing[5] if len(existing) > 5 else None

            # Convert duration from seconds to milliseconds
            duration_ms = int(metadata['duration'] * 1000) if metadata.get('duration') else None
            tags_fp = self._compute_tags_fingerprint(metadata) if self._has_fingerprints else None
            # If fingerprints match, skip write churn
            if (
                self._has_fingerprints
                and existing_mtime_ns == metadata.get("file_mtime_ns")
                and existing_size == metadata.get("file_size")
                and existing_tags_fp == tags_fp
            ):
                return old_track_id, False

            update_fields = [
                "track_id = ?",
                "artist = ?",
                "artist_key = ?",
                "title = ?",
                "album = ?",
            ]
            params = [
                track_id,
                artist,
                artist_key,
                title,
                album_title,
            ]
            if self._has_album_id:
                update_fields.append("album_id = ?")
                params.append(album_id)
            if self._has_fingerprints:
                update_fields.extend(["file_mtime_ns = ?", "file_size_bytes = ?", "tags_fingerprint = ?"])
                params.extend([
                    metadata.get("file_mtime_ns"),
                    metadata.get("file_size"),
                    tags_fp,
                ])
            update_fields.extend([
                "duration_ms = ?",
                "file_path = ?",
                "file_modified = ?",
                "last_updated = CURRENT_TIMESTAMP",
            ])
            params.extend([
                duration_ms,
                metadata['file_path'],
                metadata.get('file_modified'),
            ])

            cursor.execute(f"""
                UPDATE tracks
                SET {", ".join(update_fields)}
                WHERE rowid = ?
            """, (*params, existing_rowid))

            is_new = False
        else:
            # Insert new track
            # Convert duration from seconds to milliseconds
            duration_ms = int(metadata['duration'] * 1000) if metadata.get('duration') else None

            insert_columns = [
                "track_id", "artist", "artist_key", "title", "album"
            ]
            insert_values = [
                track_id,
                artist,
                artist_key,
                title,
                album_title,
            ]
            if self._has_album_id:
                insert_columns.append("album_id")
                insert_values.append(album_id)
            if self._has_fingerprints:
                insert_columns.extend(["file_mtime_ns", "file_size_bytes", "tags_fingerprint"])
                insert_values.extend([
                    metadata.get("file_mtime_ns"),
                    metadata.get("file_size"),
                    self._compute_tags_fingerprint(metadata),
                ])
            insert_columns.extend(["duration_ms", "file_path", "file_modified"])
            insert_values.extend([
                duration_ms,
                metadata['file_path'],
                metadata.get('file_modified')
            ])

            placeholders = ", ".join(["?"] * len(insert_columns))
            cursor.execute(f"""
                INSERT INTO tracks (
                    {", ".join(insert_columns)}
                )
                VALUES ({placeholders})
            """, tuple(insert_values))
            is_new = True

        # Upsert album row if album_id available and albums table present
        if album_id:
            self._upsert_album(album_id, album_artist, album_title)

        # If we updated a moved track and the file_path changed, clean old path duplicates if any
        if moved_candidate:
            old_path = moved_candidate[2]
            if old_path and old_path != metadata['file_path']:
                cursor.execute("DELETE FROM tracks WHERE file_path = ? AND rowid != ?", (old_path, existing_rowid))
        if existing and old_track_id != track_id:
            self._rewrite_track_id(old_track_id, track_id)

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

        # Insert new genres (with normalization)
        normalized_genres = normalize_genre_list(genres, filter_broad=True)
        for genre in normalized_genres:
            cursor.execute("""
                INSERT OR IGNORE INTO track_genres (track_id, genre, source)
                VALUES (?, ?, 'file')
            """, (track_id, genre))

    def cleanup_missing_files(self) -> int:
        """
        Remove tracks from database if their files no longer exist in the filesystem.

        Returns:
            Number of tracks removed
        """
        logger.info("\nCleaning up missing files...")
        cursor = self.conn.cursor()

        # Get all tracks with file paths
        cursor.execute("SELECT track_id, file_path FROM tracks WHERE file_path IS NOT NULL AND file_path != ''")
        rows = cursor.fetchall()

        removed_count = 0
        for row in rows:
            track_id = row['track_id']
            file_path = row['file_path']

            # Check if file exists
            if not Path(file_path).exists():
                # Remove track and associated genres
                cursor.execute("DELETE FROM track_genres WHERE track_id = ?", (track_id,))
                if self._table_exists("track_effective_genres"):
                    cursor.execute("DELETE FROM track_effective_genres WHERE track_id = ?", (track_id,))
                cursor.execute("DELETE FROM tracks WHERE track_id = ?", (track_id,))
                removed_count += 1
                logger.info(f"  Removed: {file_path}")

        # Remove rows with missing file_path values
        cursor.execute("SELECT track_id FROM tracks WHERE file_path IS NULL OR file_path = ''")
        rows = cursor.fetchall()
        for row in rows:
            track_id = row['track_id']
            cursor.execute("DELETE FROM track_genres WHERE track_id = ?", (track_id,))
            if self._table_exists("track_effective_genres"):
                cursor.execute("DELETE FROM track_effective_genres WHERE track_id = ?", (track_id,))
            cursor.execute("DELETE FROM tracks WHERE track_id = ?", (track_id,))
            removed_count += 1

        if removed_count > 0:
            self.conn.commit()
            logger.info(f"Removed {removed_count} orphaned track(s) from database")
        else:
            logger.info("No missing files found")

        return removed_count

    def cleanup_orphaned_metadata(self) -> Dict[str, int]:
        """
        Remove metadata not tied to tracks with valid file paths.

        Returns:
            Dict of deleted row counts by table
        """
        cursor = self.conn.cursor()
        stats: Dict[str, int] = {}

        cursor.execute("""
            DELETE FROM track_genres
            WHERE track_id NOT IN (
                SELECT track_id FROM tracks WHERE file_path IS NOT NULL AND file_path != ''
            )
        """)
        stats["track_genres"] = cursor.rowcount if cursor.rowcount is not None else 0

        if self._table_exists("track_effective_genres"):
            cursor.execute("""
                DELETE FROM track_effective_genres
                WHERE track_id NOT IN (
                    SELECT track_id FROM tracks WHERE file_path IS NOT NULL AND file_path != ''
                )
            """)
            stats["track_effective_genres"] = cursor.rowcount if cursor.rowcount is not None else 0

        if self._table_exists("album_genres"):
            cursor.execute("""
                DELETE FROM album_genres
                WHERE album_id NOT IN (
                    SELECT DISTINCT album_id
                    FROM tracks
                    WHERE file_path IS NOT NULL
                      AND file_path != ''
                      AND album_id IS NOT NULL
                      AND album != ''
                )
            """)
            stats["album_genres"] = cursor.rowcount if cursor.rowcount is not None else 0

        if self._table_exists("artist_genres"):
            cursor.execute("""
                DELETE FROM artist_genres
                WHERE artist NOT IN (
                    SELECT DISTINCT artist
                    FROM tracks
                    WHERE file_path IS NOT NULL
                      AND file_path != ''
                      AND artist IS NOT NULL
                      AND TRIM(artist) != ''
                )
            """)
            stats["artist_genres"] = cursor.rowcount if cursor.rowcount is not None else 0

        if self._table_exists("albums"):
            cursor.execute("""
                DELETE FROM albums
                WHERE album_id NOT IN (
                    SELECT DISTINCT album_id
                    FROM tracks
                    WHERE file_path IS NOT NULL
                      AND file_path != ''
                      AND album_id IS NOT NULL
                      AND album != ''
                )
            """)
            stats["albums"] = cursor.rowcount if cursor.rowcount is not None else 0

        if any(value > 0 for value in stats.values()):
            self.conn.commit()
            logger.info("Orphaned metadata cleanup: %s", stats)
        return stats

    def backfill_album_ids(self, limit: Optional[int] = None, batch_size: int = 500) -> Dict[str, int]:
        """
        Populate missing album_id values using stored artist/album metadata only.

        Args:
            limit: optional cap on rows processed
            batch_size: commit frequency
        """
        if not self._has_album_id:
            logger.warning("tracks.album_id column missing; skipping backfill.")
            return {"total": 0, "updated": 0, "skipped_missing_album": 0}

        album_artist_available = self._column_exists("tracks", "album_artist")
        select_extra = ", album_artist" if album_artist_available else ""
        sql = f"""
            SELECT rowid, artist, album, album_id{select_extra}
            FROM tracks
            WHERE (album_id IS NULL OR album_id = '')
              AND album IS NOT NULL AND TRIM(album) != ''
        """
        params: List = []
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.cursor()
        rows = cursor.execute(sql, params).fetchall()
        stats = {"total": len(rows), "updated": 0, "skipped_missing_album": 0}

        for row in rows:
            album_title = row["album"]
            album_artist = row["album_artist"] if album_artist_available else row["artist"]
            album_id = self._compute_album_id(album_artist, album_title)
            if not album_id:
                stats["skipped_missing_album"] += 1
                continue
            cursor.execute("UPDATE tracks SET album_id = ? WHERE rowid = ?", (album_id, row["rowid"]))
            self._upsert_album(album_id, album_artist, album_title)
            stats["updated"] += 1
            if batch_size and stats["updated"] % batch_size == 0:
                self.conn.commit()

        self.conn.commit()
        logger.info(
            "Album backfill: total=%d updated=%d skipped=%d",
            stats["total"],
            stats["updated"],
            stats["skipped_missing_album"],
        )
        return stats

    def run(
        self,
        quick: bool = False,
        limit: Optional[int] = None,
        cleanup: bool = False,
        force_cleanup: bool = False,
        progress: bool = True,
        progress_interval: float = 15.0,
        progress_every: int = 500,
        verbose_each: bool = False,
    ):
        """
        Run the library scan

        Args:
            quick: If True, only scan new/modified files
            limit: Maximum number of files to process
            cleanup: If True, remove tracks with missing files
            force_cleanup: If True, allow deletion even when discovery looks invalid
        """
        logger.info("=" * 70)
        logger.info("Music Library Scanner")
        logger.info("=" * 70)

        # Scan for files
        files = self.scan_files(quick=quick)
        discovered_total = len(files)
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL AND file_path != ''")
        existing_tracks = cursor.fetchone()[0]

        # Safety guard for deletions
        safe_to_delete = True
        if not self.music_dir.exists():
            safe_to_delete = False
            logger.warning("Library root %s is missing; skipping cleanup/orphan pruning (use --force-cleanup to override)", self.music_dir)
        elif discovered_total == 0 and existing_tracks > 0:
            safe_to_delete = False
            logger.warning("Discovered 0 files but DB has %d tracks; skipping cleanup/orphan pruning (use --force-cleanup to override)", existing_tracks)
        elif existing_tracks >= 50 and discovered_total < max(10, int(existing_tracks * 0.1)):
            safe_to_delete = False
            logger.warning(
                "Discovered %d files vs %d tracked; skipping cleanup/orphan pruning (use --force-cleanup to override)",
                discovered_total,
                existing_tracks,
            )
        if force_cleanup:
            safe_to_delete = True

        # Cleanup missing files if requested and safe
        if cleanup and safe_to_delete:
            self.cleanup_missing_files()
        elif cleanup and not safe_to_delete:
            logger.info("Cleanup requested but skipped due to safety guard.")

        if limit:
            files = files[:limit]
            logger.info(f"Limited to {limit} files")

        if not files:
            logger.info("No files to process")
            orphaned = self.cleanup_orphaned_metadata() if safe_to_delete else {}
            return {"total": 0, "new": 0, "updated": 0, "failed": 0, "orphaned": orphaned}

        # Process files
        stats = {
            'total': len(files),
            'new': 0,
            'updated': 0,
            'failed': 0
        }
        if quick:
            stats["modified_reasons"] = dict(self._last_modified_reasons)
            stats["modified_examples"] = dict(self._last_modified_examples)

        logger.info(f"\nProcessing {len(files)} files...")

        prog = ProgressLogger(
            logger,
            total=len(files),
            label="scan_library",
            unit="files",
            interval_s=progress_interval,
            every_n=progress_every,
            verbose_each=verbose_each,
        ) if progress else None

        for i, file_path in enumerate(files, 1):
            if prog:
                prog.update(detail=str(file_path))

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

        if prog:
            prog.finish()

        # Show results
        logger.info("\n" + "=" * 70)
        logger.info("Scan Complete")
        logger.info("=" * 70)
        logger.info(f"  Total processed: {stats['total']}")
        logger.info(f"  New tracks: {stats['new']}")
        logger.info(f"  Updated tracks: {stats['updated']}")
        logger.info(f"  Failed: {stats['failed']}")
        stats["orphaned"] = self.cleanup_orphaned_metadata() if safe_to_delete else {}
        return stats

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

        logger.info("Library Statistics:")
        logger.info("=" * 70)
        logger.info("  Total tracks: %s", f"{total_tracks:,}")
        logger.info("  Tracks with file paths: %s (%.1f%%)", f"{tracks_with_files:,}", tracks_with_files/total_tracks*100 if total_tracks else 0)
        logger.info("  Tracks with file genres: %s (%.1f%%)", f"{tracks_with_file_genres:,}", tracks_with_file_genres/total_tracks*100 if total_tracks else 0)
        logger.info("  Tracks with any genres: %s (%.1f%%)", f"{tracks_with_any_genres:,}", tracks_with_any_genres/total_tracks*100 if total_tracks else 0)
        logger.info("  Tracks with sonic features: %s (%.1f%%)", f"{tracks_with_sonic:,}", tracks_with_sonic/total_tracks*100 if total_tracks else 0)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import argparse
    from src.logging_utils import configure_logging, add_logging_args, resolve_log_level

    parser = argparse.ArgumentParser(description='Scan music library and update metadata')
    parser.add_argument('--quick', action='store_true',
                       help='Only scan new/modified files')
    parser.add_argument('--limit', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove tracks with missing files before scanning')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics only')
    parser.add_argument('--backfill-album-ids', action='store_true',
                       help='DB-only backfill of missing album_id values (no file scan)')
    parser.add_argument('--force-cleanup', action='store_true',
                       help='Allow cleanup/orphan pruning even if discovery looks suspiciously low')
    parser.add_argument('--verify-fingerprints', action='store_true',
                       help='Sample and validate stored tags/file fingerprints (no writes)')
    parser.add_argument('--verify-sample', type=int, default=20,
                       help='Sample size for --verify-fingerprints (default: 20)')
    parser.add_argument('--progress', dest='progress', action='store_true', default=True,
                        help='Enable progress logging (default)')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Disable progress logging')
    parser.add_argument('--progress-interval', type=float, default=15.0,
                        help='Seconds between progress updates (default: 15)')
    parser.add_argument('--progress-every', type=int, default=500,
                        help='Items between progress updates (default: 500)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose per-track progress (DEBUG)')
    add_logging_args(parser)
    args = parser.parse_args()

    # Configure logging
    log_level = resolve_log_level(args)
    if args.verbose and not args.debug and not args.quiet and args.log_level.upper() == "INFO":
        log_level = "DEBUG"
    log_file = getattr(args, 'log_file', None) or 'scan_library.log'
    configure_logging(level=log_level, log_file=log_file)

    scanner = LibraryScanner()

    if args.backfill_album_ids:
        scanner.backfill_album_ids(limit=args.limit)
        if args.stats:
            scanner.get_stats()
    elif args.stats:
        scanner.get_stats()
    elif args.verify_fingerprints:
        scanner.verify_fingerprints(sample_size=args.verify_sample)
    else:
        scanner.run(
            quick=args.quick,
            limit=args.limit,
            cleanup=args.cleanup,
            force_cleanup=args.force_cleanup,
            progress=args.progress,
            progress_interval=args.progress_interval,
            progress_every=args.progress_every,
            verbose_each=args.verbose,
        )

    scanner.close()
