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

    assert counts["chosen_from_local_count"] == 2
    assert counts["chosen_from_toward_count"] == 2
    assert counts["chosen_from_genre_count"] == 2
    assert counts["chosen_from_baseline_count"] == 2
    assert counts["chosen_from_multiple_sources_count"] == 4
