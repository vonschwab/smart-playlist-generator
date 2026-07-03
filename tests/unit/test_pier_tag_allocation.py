"""Unit tests for tag-weighted pier allocation across clusters."""
from src.playlist.artist_style import allocate_piers_by_tag_affinity


def _counts(selected, medoids_by_cluster):
    """How many selected piers came from each cluster (clusters are disjoint index sets)."""
    sets = [set(m) for m in medoids_by_cluster]
    return [sum(1 for i in selected if i in s) for s in sets]


def test_skew_zero_is_balanced_ignoring_affinity():
    # 3 clusters, distinct indices, high/mid/low affinity. skew=0 => even split, affinity ignored.
    mbc = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.5, 0.1], target_pier_count=6, skew=0.0)
    assert len(sel) == 6
    assert _counts(sel, mbc) == [2, 2, 2]


def test_skew_one_favors_high_affinity_cluster():
    mbc = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.5, 0.1], target_pier_count=6, skew=1.0)
    # Floor 1 each, remaining 3 flow to the highest-affinity cluster.
    assert _counts(sel, mbc) == [4, 1, 1]
    # High-affinity cluster contributes its TOP tag-ranked medoids (list order preserved).
    assert sel[:4] == [0, 1, 2, 3]


def test_soft_skew_between():
    mbc = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.5, 0.1], target_pier_count=6, skew=0.6)
    c = _counts(sel, mbc)
    assert sum(c) == 6
    assert c[0] > c[2]           # skews toward high affinity
    assert min(c) >= 1           # floor preserved (arc holds)


def test_cap_at_cluster_size():
    # High-affinity cluster is tiny (size 1); it cannot exceed its size even at skew=1.
    mbc = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.4, 0.1], target_pier_count=6, skew=1.0)
    assert len(sel) == 6
    assert _counts(sel, mbc)[0] == 1      # capped at size


def test_floor_per_nonempty_cluster_when_p_ge_k():
    mbc = [[0, 1], [2, 3], [4, 5], [6, 7]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.1, 0.1, 0.1], target_pier_count=5, skew=1.0)
    assert min(_counts(sel, mbc)) >= 1    # every cluster represented


def test_p_less_than_k_gives_floor_to_top_weight_clusters():
    mbc = [[0, 1], [2, 3], [4, 5], [6, 7]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.8, 0.1, 0.05], target_pier_count=2, skew=1.0)
    assert len(sel) == 2
    c = _counts(sel, mbc)
    assert c[0] == 1 and c[1] == 1 and c[2] == 0 and c[3] == 0


def test_few_tracks_takes_all():
    mbc = [[0, 1], [2]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.1], target_pier_count=10, skew=0.6)
    assert sorted(sel) == [0, 1, 2]       # total_available <= P => everything


def test_empty_input():
    assert allocate_piers_by_tag_affinity([], [], target_pier_count=10, skew=0.6) == []
    assert allocate_piers_by_tag_affinity([[], []], [0.0, 0.0], target_pier_count=10, skew=0.6) == []


def test_all_equal_affinity_is_uniform():
    mbc = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.5, 0.5, 0.5], target_pier_count=6, skew=1.0)
    assert _counts(sel, mbc) == [2, 2, 2]
