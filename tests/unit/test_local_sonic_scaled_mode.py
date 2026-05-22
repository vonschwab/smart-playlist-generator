"""Tests for local-sonic-edge penalty scaled mode (Task 5).

Verifies that:
- Legacy mode preserves existing strength*(threshold-edge_cos) behavior
- Scaled mode applies scale*(threshold-edge_cos) for larger penalties
- Both modes return 0.0 when edge_cos >= threshold
- Unknown modes fall back to legacy
"""
from src.playlist.pier_bridge.beam import _local_sonic_penalty_value


def test_legacy_mode_matches_existing_math():
    # strength=0.30, threshold=0.10, edge_cos=0.03
    # penalty = 0.30 * (0.10 - 0.03) = 0.021
    p = _local_sonic_penalty_value(
        edge_cos=0.03, threshold=0.10, strength=0.30, scale=1.0, mode="legacy",
    )
    assert abs(p - 0.021) < 1e-9


def test_scaled_mode_applies_scale():
    # mode=scaled, scale=2.0 → penalty = 2.0 * (0.10 - 0.03) = 0.14
    p = _local_sonic_penalty_value(
        edge_cos=0.03, threshold=0.10, strength=0.30, scale=2.0, mode="scaled",
    )
    assert abs(p - 0.14) < 1e-9


def test_above_threshold_no_penalty_both_modes():
    for mode in ("legacy", "scaled"):
        p = _local_sonic_penalty_value(
            edge_cos=0.5, threshold=0.10, strength=0.30, scale=2.0, mode=mode,
        )
        assert p == 0.0


def test_unknown_mode_falls_back_to_legacy():
    p = _local_sonic_penalty_value(
        edge_cos=0.03, threshold=0.10, strength=0.30, scale=5.0, mode="bogus",
    )
    assert abs(p - 0.021) < 1e-9
