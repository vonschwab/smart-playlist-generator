"""
Artist Similarity Cache - Persistent storage for Last.FM artist similarity data
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ArtistSimilarityCache:
    """Handles caching of Last.FM artist similarity data to disk"""

    def __init__(self, cache_file: str = "artist_similarity_cache.json", expiry_days: int = 30):
        """
        Initialize the cache

        Args:
            cache_file: Path to cache file
            expiry_days: Number of days before cache entries expire
        """
        self.cache_file = Path(cache_file)
        self.expiry_days = expiry_days
        self.cache_data = self._load_cache()
        logger.info(f"Initialized artist similarity cache: {self.cache_file}")
        logger.info(f"Cache expiry: {self.expiry_days} days")
        logger.info(f"Cached artists: {len(self.cache_data.get('artists', {}))}")

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if not self.cache_file.exists():
            logger.info("No existing cache file found, starting fresh")
            return {
                "cache_version": "1.0",
                "artists": {}
            }

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded cache with {len(data.get('artists', {}))} artists")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, starting fresh")
            return {
                "cache_version": "1.0",
                "artists": {}
            }

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache with {len(self.cache_data.get('artists', {}))} artists")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _is_expired(self, fetched_at: str) -> bool:
        """Check if a cache entry has expired"""
        try:
            fetched_time = datetime.fromisoformat(fetched_at)
            age = datetime.now() - fetched_time
            return age > timedelta(days=self.expiry_days)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {fetched_at}: {e}")
            return True  # Treat invalid timestamps as expired

    def get_similar_artists(self, artist_name: str) -> Optional[List[str]]:
        """
        Get similar artists from cache

        Args:
            artist_name: Name of the artist

        Returns:
            List of similar artist names, or None if not in cache or expired
        """
        artists = self.cache_data.get("artists", {})
        artist_data = artists.get(artist_name)

        if not artist_data:
            logger.debug(f"Cache MISS: {artist_name}")
            return None

        # Check if expired
        if self._is_expired(artist_data.get("fetched_at", "")):
            logger.debug(f"Cache EXPIRED: {artist_name}")
            return None

        logger.debug(f"Cache HIT: {artist_name}")
        return artist_data.get("similar_artists", [])

    def set_similar_artists(self, artist_name: str, similar_artists: List[str]):
        """
        Store similar artists in cache

        Args:
            artist_name: Name of the artist
            similar_artists: List of similar artist names
        """
        if "artists" not in self.cache_data:
            self.cache_data["artists"] = {}

        self.cache_data["artists"][artist_name] = {
            "similar_artists": similar_artists,
            "fetched_at": datetime.now().isoformat()
        }

        logger.debug(f"Cached {len(similar_artists)} similar artists for {artist_name}")
        self._save_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        artists = self.cache_data.get("artists", {})
        total = len(artists)
        expired = sum(1 for data in artists.values()
                     if self._is_expired(data.get("fetched_at", "")))
        fresh = total - expired

        return {
            "total_artists": total,
            "fresh_entries": fresh,
            "expired_entries": expired,
            "cache_file": str(self.cache_file),
            "expiry_days": self.expiry_days
        }

    def clear_expired(self):
        """Remove expired entries from cache"""
        artists = self.cache_data.get("artists", {})
        before = len(artists)

        # Remove expired entries
        self.cache_data["artists"] = {
            name: data for name, data in artists.items()
            if not self._is_expired(data.get("fetched_at", ""))
        }

        after = len(self.cache_data["artists"])
        removed = before - after

        if removed > 0:
            logger.info(f"Cleared {removed} expired cache entries")
            self._save_cache()

        return removed


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cache = ArtistSimilarityCache()

    # Test setting
    cache.set_similar_artists("Ariel Pink", ["John Maus", "Part Time", "R. Stevie Moore"])

    # Test getting
    similar = cache.get_similar_artists("Ariel Pink")
    logger.info(f"Similar to Ariel Pink: {similar}")

    # Stats
    stats = cache.get_cache_stats()
    logger.info(f"Cache stats: {stats}")
