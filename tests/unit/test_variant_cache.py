"""Unit tests for variant cache.

Tests variant caching infrastructure extracted from sonic_variant.py (Phase 5.3).

Coverage:
- VariantCache LRU eviction
- Cache key generation
- Cache statistics
- Get/Put operations
- Global cache management
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.similarity.variant_cache import (
    VariantCache,
    clear_global_cache,
    get_global_cache,
)


# =============================================================================
# VariantCache Tests
# =============================================================================

class TestVariantCache:
    """Test VariantCache."""

    def test_create_cache(self):
        """Test creating cache."""
        cache = VariantCache(max_size=10)

        assert cache.max_size == 10
        assert cache.size() == 0

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = VariantCache(max_size=10)

        result = cache.get(
            shape=(100, 137),
            variant="tower_pca",
            l2=True,
        )

        assert result is None

    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        cache = VariantCache(max_size=10)

        # Create mock matrix and stats
        matrix = np.random.rand(100, 137)
        stats = {"mean_norm": 1.0}

        # Put in cache
        cache.put(
            shape=(100, 137),
            variant="tower_pca",
            l2=True,
            matrix=matrix,
            stats=stats,
        )

        # Get from cache
        cached_matrix, cached_stats = cache.get(
            shape=(100, 137),
            variant="tower_pca",
            l2=True,
        )

        assert np.array_equal(cached_matrix, matrix)
        assert cached_stats == stats
        assert cache.size() == 1

    def test_cache_different_keys(self):
        """Test that different parameters create different cache entries."""
        cache = VariantCache(max_size=10)

        matrix1 = np.random.rand(100, 137)
        matrix2 = np.random.rand(100, 137)

        # Store with l2=True
        cache.put((100, 137), "tower_pca", True, matrix1, {})

        # Store with l2=False (different key)
        cache.put((100, 137), "tower_pca", False, matrix2, {})

        assert cache.size() == 2

        # Retrieve both
        result1 = cache.get((100, 137), "tower_pca", True)
        result2 = cache.get((100, 137), "tower_pca", False)

        assert np.array_equal(result1[0], matrix1)
        assert np.array_equal(result2[0], matrix2)

    def test_cache_eviction_lru(self):
        """Test LRU eviction when cache is full."""
        cache = VariantCache(max_size=3)

        # Fill cache
        for i in range(3):
            matrix = np.random.rand(100, 137)
            cache.put((100, 137), f"variant_{i}", False, matrix, {})

        assert cache.size() == 3

        # Add fourth entry (should evict variant_0)
        matrix4 = np.random.rand(100, 137)
        cache.put((100, 137), "variant_3", False, matrix4, {})

        assert cache.size() == 3

        # variant_0 should be evicted
        result0 = cache.get((100, 137), "variant_0", False)
        assert result0 is None

        # Others should still be cached
        result1 = cache.get((100, 137), "variant_1", False)
        result2 = cache.get((100, 137), "variant_2", False)
        result3 = cache.get((100, 137), "variant_3", False)

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

    def test_cache_lru_access_updates_order(self):
        """Test that accessing entry moves it to end (most recently used)."""
        cache = VariantCache(max_size=3)

        # Fill cache
        matrices = []
        for i in range(3):
            matrix = np.random.rand(100, 137)
            matrices.append(matrix)
            cache.put((100, 137), f"variant_{i}", False, matrix, {})

        # Access variant_0 (moves to end)
        cache.get((100, 137), "variant_0", False)

        # Add fourth entry (should evict variant_1, not variant_0)
        matrix4 = np.random.rand(100, 137)
        cache.put((100, 137), "variant_3", False, matrix4, {})

        # variant_1 should be evicted (was oldest)
        result1 = cache.get((100, 137), "variant_1", False)
        assert result1 is None

        # variant_0 should still be cached (was accessed recently)
        result0 = cache.get((100, 137), "variant_0", False)
        assert result0 is not None

    def test_cache_with_config_weights(self):
        """Test cache with config weights."""
        cache = VariantCache(max_size=10)

        matrix = np.random.rand(100, 137)

        # Store with weights
        cache.put(
            shape=(100, 137),
            variant="tower_weighted",
            l2=False,
            matrix=matrix,
            stats={},
            config_weights=(0.2, 0.5, 0.3),
        )

        # Get with same weights
        result = cache.get(
            shape=(100, 137),
            variant="tower_weighted",
            l2=False,
            config_weights=(0.2, 0.5, 0.3),
        )

        assert result is not None

        # Get with different weights (should miss)
        result2 = cache.get(
            shape=(100, 137),
            variant="tower_weighted",
            l2=False,
            config_weights=(0.3, 0.4, 0.3),
        )

        assert result2 is None

    def test_cache_with_config_dims(self):
        """Test cache with PCA dimensions."""
        cache = VariantCache(max_size=10)

        matrix = np.random.rand(100, 137)

        # Store with dims
        cache.put(
            shape=(100, 137),
            variant="tower_pca",
            l2=False,
            matrix=matrix,
            stats={},
            config_dims=(8, 16, 8),
        )

        # Get with same dims
        result = cache.get(
            shape=(100, 137),
            variant="tower_pca",
            l2=False,
            config_dims=(8, 16, 8),
        )

        assert result is not None

        # Get with different dims (should miss)
        result2 = cache.get(
            shape=(100, 137),
            variant="tower_pca",
            l2=False,
            config_dims=(10, 20, 10),
        )

        assert result2 is None

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = VariantCache(max_size=10)

        # Add entries
        for i in range(5):
            matrix = np.random.rand(100, 137)
            cache.put((100, 137), f"variant_{i}", False, matrix, {})

        assert cache.size() == 5

        # Clear
        cache.clear()

        assert cache.size() == 0

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = VariantCache(max_size=10)

        matrix = np.random.rand(100, 137)
        cache.put((100, 137), "tower_pca", True, matrix, {})

        # Hit
        cache.get((100, 137), "tower_pca", True)

        # Miss
        cache.get((100, 137), "tower_pca", False)

        stats = cache.stats()

        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5

    def test_get_or_compute_cache_hit(self):
        """Test get_or_compute with cache hit."""
        cache = VariantCache(max_size=10)

        matrix = np.random.rand(100, 137)
        stats = {"mean_norm": 1.0}

        # Pre-populate cache
        cache.put((100, 137), "tower_pca", True, matrix, stats)

        # Define compute function that should NOT be called
        compute_called = False

        def compute_fn():
            nonlocal compute_called
            compute_called = True
            return np.random.rand(100, 137), {}

        # Get or compute (should use cache)
        X_sonic = np.random.rand(100, 137)
        result_matrix, result_stats = cache.get_or_compute(
            X_sonic=X_sonic,
            variant="tower_pca",
            l2=True,
            compute_fn=compute_fn,
        )

        assert not compute_called  # Should not have called compute_fn
        assert np.array_equal(result_matrix, matrix)
        assert result_stats == stats

    def test_get_or_compute_cache_miss(self):
        """Test get_or_compute with cache miss."""
        cache = VariantCache(max_size=10)

        matrix = np.random.rand(100, 137)
        stats = {"mean_norm": 1.5}

        compute_called = False

        def compute_fn():
            nonlocal compute_called
            compute_called = True
            return matrix, stats

        # Get or compute (should compute and cache)
        X_sonic = np.random.rand(100, 137)
        result_matrix, result_stats = cache.get_or_compute(
            X_sonic=X_sonic,
            variant="tower_pca",
            l2=True,
            compute_fn=compute_fn,
        )

        assert compute_called  # Should have called compute_fn
        assert np.array_equal(result_matrix, matrix)
        assert result_stats == stats

        # Now should be cached
        cache_stats = cache.stats()
        assert cache_stats["size"] == 1


# =============================================================================
# Global Cache Tests
# =============================================================================

class TestGlobalCache:
    """Test global cache management."""

    def test_get_global_cache(self):
        """Test getting global cache."""
        cache = get_global_cache(max_size=10)

        assert isinstance(cache, VariantCache)
        assert cache.max_size == 10

    def test_global_cache_singleton(self):
        """Test global cache is singleton."""
        cache1 = get_global_cache(max_size=10)
        cache2 = get_global_cache(max_size=20)  # Should ignore max_size

        assert cache1 is cache2

    def test_clear_global_cache(self):
        """Test clearing global cache."""
        cache = get_global_cache(max_size=10)

        # Add entry
        matrix = np.random.rand(100, 137)
        cache.put((100, 137), "tower_pca", True, matrix, {})

        assert cache.size() == 1

        # Clear
        clear_global_cache()

        assert cache.size() == 0


# =============================================================================
# Cache Key Tests
# =============================================================================

class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_different_shapes(self):
        """Test different shapes create different keys."""
        cache = VariantCache(max_size=10)

        key1 = cache._make_key((100, 137), "tower_pca", True)
        key2 = cache._make_key((200, 137), "tower_pca", True)

        assert key1 != key2

    def test_cache_key_different_variants(self):
        """Test different variants create different keys."""
        cache = VariantCache(max_size=10)

        key1 = cache._make_key((100, 137), "tower_pca", True)
        key2 = cache._make_key((100, 137), "raw", True)

        assert key1 != key2

    def test_cache_key_different_l2(self):
        """Test different l2 values create different keys."""
        cache = VariantCache(max_size=10)

        key1 = cache._make_key((100, 137), "tower_pca", True)
        key2 = cache._make_key((100, 137), "tower_pca", False)

        assert key1 != key2

    def test_cache_key_with_weights(self):
        """Test cache keys with config weights."""
        cache = VariantCache(max_size=10)

        key1 = cache._make_key(
            (100, 137),
            "tower_weighted",
            False,
            config_weights=(0.2, 0.5, 0.3),
        )
        key2 = cache._make_key(
            (100, 137),
            "tower_weighted",
            False,
            config_weights=(0.3, 0.4, 0.3),
        )

        assert key1 != key2

    def test_cache_key_weight_rounding(self):
        """Test that similar weights round to same key."""
        cache = VariantCache(max_size=10)

        # Weights that round to same 3 decimals should match
        key1 = cache._make_key(
            (100, 137),
            "tower_weighted",
            False,
            config_weights=(0.2001, 0.5001, 0.3001),
        )
        key2 = cache._make_key(
            (100, 137),
            "tower_weighted",
            False,
            config_weights=(0.2002, 0.5002, 0.3002),
        )

        assert key1 == key2
