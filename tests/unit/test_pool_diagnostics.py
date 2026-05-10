import pytest

from src.playlist.pier_bridge_builder import (
    _compute_chosen_source_counts,
    _compute_pool_overlap_metrics,
)


def test_compute_pool_overlap_metrics():
    baseline = [1, 2, 3]
    union = [3, 4]
    metrics = _compute_pool_overlap_metrics(baseline, union)

    assert metrics["pool_overlap_baseline_size"] == 3
    assert metrics["pool_overlap_union_size"] == 2
    assert metrics["pool_overlap_intersection"] == 1
    assert metrics["pool_overlap_baseline_only"] == 2
    assert metrics["pool_overlap_union_only"] == 1
    assert metrics["pool_overlap_jaccard"] == 0.25


def test_compute_chosen_source_counts():
    """Verify priority-based exclusive counts and membership counts.

    Phase 3 semantics (commits 0bc482f / e023d10 / 4fec043):
      * Exclusive counts use priority order: genre > toward > local > baseline_only.
      * Membership counts track all overlaps (local_only, local+toward, etc.).
    """
    path = [1, 2, 3, 4]
    sources = {
        "local": {1, 2},
        "toward": {2, 3},
        "genre": {3, 4},
    }
    baseline_pool = {1, 4}
    counts = _compute_chosen_source_counts(
        path, sources=sources, baseline_pool=baseline_pool
    )

    # Priority assignment:
    #   1 → local (only in local)            -> local
    #   2 → local+toward, priority toward    -> toward
    #   3 → toward+genre, priority genre     -> genre
    #   4 → genre+baseline, priority genre   -> genre
    assert counts["chosen_from_local_count"] == 1
    assert counts["chosen_from_toward_count"] == 1
    assert counts["chosen_from_genre_count"] == 2
    assert counts["chosen_from_baseline_only_count"] == 0

    # Membership counts (all overlaps tracked):
    #   1 → local_only
    #   2 → local+toward
    #   3 → toward+genre
    #   4 → genre_only (baseline membership not counted here)
    assert counts["local_only"] == 1
    assert counts["local+toward"] == 1
    assert counts["toward+genre"] == 1
    assert counts["genre_only"] == 1
    assert counts["local+toward+genre"] == 0
    assert counts["baseline_only"] == 0
