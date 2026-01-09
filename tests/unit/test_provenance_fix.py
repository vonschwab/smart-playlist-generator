"""
Unit test to verify provenance counting is mutually exclusive.

Tests that tracks appearing in multiple source sets (local, toward, genre)
are counted exactly once with the correct priority.
"""

from src.playlist.pier_bridge_builder import _compute_chosen_source_counts


def test_provenance_mutually_exclusive():
    """
    Test that each track is counted in exactly one category.

    Priority order:
    1. genre (highest)
    2. toward
    3. local
    4. baseline_only (lowest)
    """
    # Create overlapping source sets
    sources = {
        "local": {1, 2, 3, 4},      # Tracks 1-4 in local
        "toward": {3, 4, 5, 6},     # Tracks 3-6 in toward (overlap with local)
        "genre": {5, 6, 7, 8},      # Tracks 5-8 in genre (overlap with toward)
    }
    baseline_pool = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  # All tracks in baseline

    # Path with tracks from various sources
    path = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    counts = _compute_chosen_source_counts(
        path,
        sources=sources,
        baseline_pool=baseline_pool,
    )

    # Verify mutually exclusive counts
    # Track 1: local only → local
    # Track 2: local only → local
    # Track 3: local + toward → toward (priority)
    # Track 4: local + toward → toward (priority)
    # Track 5: toward + genre → genre (priority)
    # Track 6: toward + genre → genre (priority)
    # Track 7: genre only → genre
    # Track 8: genre only → genre
    # Track 9: baseline only → baseline_only

    assert counts["chosen_from_local_count"] == 2, "Tracks 1,2 should be local"
    assert counts["chosen_from_toward_count"] == 2, "Tracks 3,4 should be toward"
    assert counts["chosen_from_genre_count"] == 4, "Tracks 5,6,7,8 should be genre"
    assert counts["chosen_from_baseline_only_count"] == 1, "Track 9 should be baseline_only"

    # Verify sum equals path length (no double-counting)
    total = (
        counts["chosen_from_local_count"] +
        counts["chosen_from_toward_count"] +
        counts["chosen_from_genre_count"] +
        counts["chosen_from_baseline_only_count"]
    )
    assert total == len(path), f"Sum {total} should equal path length {len(path)}"


def test_provenance_no_sources():
    """Test provenance when no DJ sources are provided (all baseline_only)."""
    path = [1, 2, 3, 4, 5]
    baseline_pool = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    counts = _compute_chosen_source_counts(
        path,
        sources={},  # No DJ sources
        baseline_pool=baseline_pool,
    )

    assert counts["chosen_from_local_count"] == 0
    assert counts["chosen_from_toward_count"] == 0
    assert counts["chosen_from_genre_count"] == 0
    assert counts["chosen_from_baseline_only_count"] == 5, "All tracks should be baseline_only"


def test_provenance_genre_priority():
    """Test that genre has highest priority when track is in all sources."""
    sources = {
        "local": {1},
        "toward": {1},
        "genre": {1},
    }
    baseline_pool = {1, 2, 3}
    path = [1]

    counts = _compute_chosen_source_counts(
        path,
        sources=sources,
        baseline_pool=baseline_pool,
    )

    # Track 1 is in all sources, but should be counted as genre (highest priority)
    assert counts["chosen_from_genre_count"] == 1
    assert counts["chosen_from_toward_count"] == 0
    assert counts["chosen_from_local_count"] == 0
    assert counts["chosen_from_baseline_only_count"] == 0
