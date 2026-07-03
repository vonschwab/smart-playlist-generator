"""Pier lever: tag affinity shifts medoid choice within a cluster."""
import numpy as np

from src.playlist.artist_style import _medoids_for_cluster


def _select(tag_weight, tag_affinity):
    # Three cluster members; member 0 is closest to the centroid.
    x = np.array([[1.0, 0.0], [0.98, 0.02], [0.96, 0.04]])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return _medoids_for_cluster(
        x,
        [0, 1, 2],
        x[0],
        ["t0", "t1", "t2"],
        1,                       # per_cluster
        np.random.default_rng(0),
        1,                       # top_k
        None,                    # artist_duration_stats
        None,                    # track_durations_ms
        1.0,                     # similarity_weight
        0.0,                     # duration_weight
        0.0,                     # energy_weight
        None,                    # energy_proximity
        0.0,                     # popularity_weight
        None,                    # popularity_values
        tag_weight,
        tag_affinity,
    )


def test_zero_weight_keeps_sonic_medoid():
    assert _select(0.0, np.array([0.0, 0.0, 1.0])) == [0]


def test_tag_affinity_promotes_on_tag_member():
    # Sonic gap 0->2 is ~0.04; affinity gap is 1.0 at weight 0.5 -> member 2 wins.
    assert _select(0.5, np.array([0.0, 0.0, 1.0])) == [2]


def test_none_affinity_is_inert_even_with_weight():
    assert _select(0.5, None) == [0]
