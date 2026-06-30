"""SP2 seed-character anti-collapse scoring — pure-function unit tests.

A (hubness): deflate a candidate's pier-similarities by its k-NN in-degree in the
segment pool (hubs = the generic blur). B (anti_center): penalize a candidate by how
much closer it sits to the local pool center than to its own piers (anti-sag).
"""
import numpy as np

from src.playlist.pier_bridge.seed_character import (
    anti_center_penalty,
    hubness_deflated_bridge,
    pool_hubness,
)


def _unit(M):
    M = np.asarray(M, dtype=float)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def test_pool_hubness_flags_the_central_track():
    # index 0 sits near all four axis points; the axis points are mutually distant.
    X = _unit([[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    hub = pool_hubness(X, k=2)
    assert hub.shape == (5,)
    assert hub[0] == hub.max()          # the central track is the biggest hub
    assert 0.0 <= hub.min() and hub.max() <= 1.0


def test_pool_hubness_handles_tiny_pools():
    assert pool_hubness(_unit(np.eye(3)), k=1).shape == (3,)
    assert pool_hubness(np.zeros((1, 4)), k=5).tolist() == [0.0]


def test_hubness_deflation_off_is_plain_harmonic_mean():
    got = hubness_deflated_bridge(0.6, 0.4, hub=0.9, strength=0.0)
    assert abs(got - (2 * 0.6 * 0.4) / (0.6 + 0.4)) < 1e-9   # strength 0 => untouched


def test_hubness_deflation_drops_a_hub_below_a_non_hub():
    non_hub = hubness_deflated_bridge(0.6, 0.6, hub=0.0, strength=0.3)
    hub = hubness_deflated_bridge(0.6, 0.6, hub=1.0, strength=0.3)
    assert hub < non_hub                # equal raw sims, but the hub is deflated


def test_anti_center_penalty_fires_only_when_more_central_than_pier_like():
    assert anti_center_penalty(cand_center_sim=0.7, bridge_score=0.4, strength=0.5) > 0
    assert anti_center_penalty(cand_center_sim=0.3, bridge_score=0.6, strength=0.5) == 0.0
    assert anti_center_penalty(0.9, 0.1, strength=0.0) == 0.0   # strength 0 => inert
