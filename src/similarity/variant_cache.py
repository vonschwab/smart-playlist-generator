"""
Variant Cache
=============

Extracted from sonic_variant.py (Phase 5.3).

This module provides enhanced caching for computed sonic variant matrices,
with LRU eviction, size limits, and manual cache management.

Usage:
    cache = VariantCache(max_size=10)

    # Get or compute variant
    matrix, stats = cache.get_or_compute(
        X_sonic=artifact.X_sonic,
        variant="tower_pca",
        l2=True,
        compute_fn=lambda: compute_sonic_variant_matrix(X_sonic, "tower_pca", l2=True)
    )

    # Clear cache when needed
    cache.clear()
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VariantCache:
    """Cache for computed sonic variant matrices with LRU eviction.

    Uses OrderedDict for LRU ordering and limits cache size to prevent
    unbounded memory growth.

    Cache Key:
        Combines array shape, variant name, l2 flag, and optional
        config weights to create a stable cache key.

    Eviction:
        When cache exceeds max_size, oldest (least recently used) entry is removed.
    """

    def __init__(self, max_size: int = 10):
        """Initialize variant cache.

        Args:
            max_size: Maximum number of cached matrices (default 10)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Tuple[np.ndarray, dict]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self.logger = logging.getLogger(__name__)

    def _make_key(
        self,
        shape: Tuple[int, ...],
        variant: str,
        l2: bool,
        config_weights: Optional[Tuple[float, ...]] = None,
        config_dims: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """Create stable cache key from parameters.

        Args:
            shape: Array shape tuple
            variant: Variant name (tower_pca, raw, etc.)
            l2: Whether L2 normalization applied
            config_weights: Optional tower weights
            config_dims: Optional PCA dimensions

        Returns:
            Cache key string
        """
        # Build key components
        key_parts = [
            f"shape={shape}",
            f"variant={variant}",
            f"l2={l2}",
        ]

        if config_weights is not None:
            # Round to 3 decimals for stability
            weights_str = ",".join(f"{w:.3f}" for w in config_weights)
            key_parts.append(f"weights={weights_str}")

        if config_dims is not None:
            dims_str = ",".join(str(d) for d in config_dims)
            key_parts.append(f"dims={dims_str}")

        # Create stable key
        key_str = "|".join(key_parts)

        # Hash for shorter keys (optional, but cleaner)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]

        return f"{variant}_{key_hash}"

    def get(
        self,
        shape: Tuple[int, ...],
        variant: str,
        l2: bool = False,
        config_weights: Optional[Tuple[float, ...]] = None,
        config_dims: Optional[Tuple[int, ...]] = None,
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """Get cached variant matrix if available.

        Args:
            shape: Array shape tuple
            variant: Variant name
            l2: Whether L2 normalization applied
            config_weights: Optional tower weights
            config_dims: Optional PCA dimensions

        Returns:
            Cached (matrix, stats) tuple or None if not found
        """
        key = self._make_key(shape, variant, l2, config_weights, config_dims)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            self.logger.debug(f"Cache hit: {key} (hits={self._hits}, misses={self._misses})")
            return self._cache[key]

        self._misses += 1
        return None

    def put(
        self,
        shape: Tuple[int, ...],
        variant: str,
        l2: bool,
        matrix: np.ndarray,
        stats: dict,
        config_weights: Optional[Tuple[float, ...]] = None,
        config_dims: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Store variant matrix in cache with LRU eviction.

        Args:
            shape: Array shape tuple
            variant: Variant name
            l2: Whether L2 normalization applied
            matrix: Computed variant matrix
            stats: Variant statistics
            config_weights: Optional tower weights
            config_dims: Optional PCA dimensions
        """
        key = self._make_key(shape, variant, l2, config_weights, config_dims)

        # Store in cache
        self._cache[key] = (matrix, stats)
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self.logger.debug(f"Cache evicted: {oldest_key} (size={len(self._cache)})")

        self.logger.debug(f"Cache stored: {key} (size={len(self._cache)})")

    def get_or_compute(
        self,
        X_sonic: np.ndarray,
        variant: str,
        l2: bool,
        compute_fn: Callable[[], Tuple[np.ndarray, dict]],
        config_weights: Optional[Tuple[float, ...]] = None,
        config_dims: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Get cached variant or compute if not cached.

        Args:
            X_sonic: Sonic feature matrix
            variant: Variant name
            l2: Whether to apply L2 normalization
            compute_fn: Function to compute variant if not cached
            config_weights: Optional tower weights
            config_dims: Optional PCA dimensions

        Returns:
            Tuple of (variant_matrix, stats)
        """
        shape = X_sonic.shape

        # Try cache first
        cached = self.get(shape, variant, l2, config_weights, config_dims)
        if cached is not None:
            return cached

        # Compute and cache
        matrix, stats = compute_fn()
        self.put(shape, variant, l2, matrix, stats, config_weights, config_dims)

        return matrix, stats

    def clear(self) -> None:
        """Clear all cached entries."""
        num_cleared = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self.logger.info(f"Cache cleared: {num_cleared} entries removed")

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of cached entries
        """
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, hit_rate)
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": hit_rate,
        }


# Global cache instance (optional, for backward compatibility)
_global_cache: Optional[VariantCache] = None


def get_global_cache(max_size: int = 10) -> VariantCache:
    """Get or create global variant cache.

    Args:
        max_size: Maximum cache size

    Returns:
        Global VariantCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = VariantCache(max_size=max_size)
    return _global_cache


def clear_global_cache() -> None:
    """Clear global variant cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
