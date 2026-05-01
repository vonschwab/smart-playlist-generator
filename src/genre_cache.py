"""
Genre Cache - Persistent cache for MusicBrainz/Discogs genre lookups

Eliminates redundant API calls by caching genre results with timestamps.
Supports TTL (time-to-live) for cache invalidation and bulk operations.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached genre data for an artist or album."""
    mbid: str  # MusicBrainz ID
    genres: List[str]
    source: str  # 'musicbrainz' or 'discogs'
    timestamp: str  # ISO format timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)

    def is_stale(self, ttl_days: int) -> bool:
        """Check if entry is older than TTL."""
        try:
            cached_time = datetime.fromisoformat(self.timestamp)
            age = datetime.now() - cached_time
            return age > timedelta(days=ttl_days)
        except (ValueError, TypeError):
            # Invalid timestamp, consider stale
            return True


class GenreCache:
    """
    Persistent cache for MusicBrainz/Discogs genre lookups.

    Features:
    - TTL-based cache invalidation (default 90 days)
    - Bulk load/save operations
    - Thread-safe (single process assumption)
    - JSON-based storage

    Usage:
        cache = GenreCache("data/genre_cache.json", ttl_days=90)

        # Check cache before API call
        genres = cache.get("mbid-12345")
        if genres is None:
            # Not in cache or stale, fetch from API
            genres = fetch_from_musicbrainz("mbid-12345")
            cache.set("mbid-12345", genres, source="musicbrainz")

        # Bulk operations
        cached = cache.bulk_load(["mbid-1", "mbid-2", "mbid-3"])
        missing = [mbid for mbid in mbids if mbid not in cached]

        # Periodic cleanup
        cache.cleanup_stale()
        cache.save()
    """

    def __init__(self, cache_file: Path, ttl_days: int = 90):
        """
        Initialize the genre cache.

        Args:
            cache_file: Path to cache file (JSON)
            ttl_days: Days before cache entry is considered stale (default 90)
        """
        self.cache_file = Path(cache_file)
        self.ttl_days = ttl_days
        self._cache: Dict[str, CacheEntry] = {}
        self._dirty = False  # Track if cache needs saving

        # Load existing cache
        if self.cache_file.exists():
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert dict to CacheEntry objects
            self._cache = {
                mbid: CacheEntry.from_dict(entry_data)
                for mbid, entry_data in data.items()
            }

            logger.info(f"Loaded genre cache with {len(self._cache)} entries from {self.cache_file}")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.warning(f"Failed to load genre cache: {e}")
            self._cache = {}

    def save(self, force: bool = False) -> None:
        """
        Save cache to disk.

        Args:
            force: Save even if cache hasn't been modified
        """
        if not self._dirty and not force:
            return

        try:
            # Ensure parent directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert CacheEntry objects to dicts
            data = {
                mbid: entry.to_dict()
                for mbid, entry in self._cache.items()
            }

            # Write atomically (write to temp file, then rename)
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            temp_file.replace(self.cache_file)
            self._dirty = False
            logger.debug(f"Saved genre cache with {len(self._cache)} entries to {self.cache_file}")

        except (OSError, IOError) as e:
            logger.error(f"Failed to save genre cache: {e}")

    def get(self, mbid: str) -> Optional[List[str]]:
        """
        Get cached genres for an MBID.

        Args:
            mbid: MusicBrainz ID

        Returns:
            List of genres if cached and not stale, None otherwise
        """
        entry = self._cache.get(mbid)
        if entry is None:
            return None

        if entry.is_stale(self.ttl_days):
            # Stale entry, treat as cache miss
            logger.debug(f"Cache entry for {mbid} is stale (age > {self.ttl_days} days)")
            return None

        return entry.genres

    def set(self, mbid: str, genres: List[str], source: str = "musicbrainz") -> None:
        """
        Store genres in cache.

        Args:
            mbid: MusicBrainz ID
            genres: List of genre strings
            source: Source of genres ('musicbrainz', 'discogs', etc.)
        """
        entry = CacheEntry(
            mbid=mbid,
            genres=list(genres),
            source=source,
            timestamp=datetime.now().isoformat()
        )
        self._cache[mbid] = entry
        self._dirty = True

    def bulk_load(self, mbids: List[str]) -> Dict[str, List[str]]:
        """
        Load multiple entries from cache.

        Args:
            mbids: List of MusicBrainz IDs to load

        Returns:
            Dictionary mapping MBIDs to genre lists (only non-stale entries)
        """
        result = {}
        for mbid in mbids:
            genres = self.get(mbid)
            if genres is not None:
                result[mbid] = genres

        return result

    def bulk_set(self, entries: Dict[str, List[str]], source: str = "musicbrainz") -> None:
        """
        Store multiple entries in cache.

        Args:
            entries: Dictionary mapping MBIDs to genre lists
            source: Source of genres
        """
        for mbid, genres in entries.items():
            self.set(mbid, genres, source=source)

    def has(self, mbid: str) -> bool:
        """
        Check if MBID is in cache and not stale.

        Args:
            mbid: MusicBrainz ID

        Returns:
            True if cached and not stale, False otherwise
        """
        return self.get(mbid) is not None

    def cleanup_stale(self) -> int:
        """
        Remove stale entries from cache.

        Returns:
            Number of entries removed
        """
        stale_mbids = [
            mbid for mbid, entry in self._cache.items()
            if entry.is_stale(self.ttl_days)
        ]

        for mbid in stale_mbids:
            del self._cache[mbid]

        if stale_mbids:
            self._dirty = True
            logger.info(f"Removed {len(stale_mbids)} stale entries from genre cache")

        return len(stale_mbids)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._dirty = True
        logger.info("Cleared genre cache")

    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (total, stale, by_source)
        """
        total = len(self._cache)
        stale = sum(1 for entry in self._cache.values() if entry.is_stale(self.ttl_days))

        by_source = {}
        for entry in self._cache.values():
            by_source[entry.source] = by_source.get(entry.source, 0) + 1

        return {
            "total": total,
            "fresh": total - stale,
            "stale": stale,
            "by_source": by_source
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache."""
        if self._dirty:
            self.save()
