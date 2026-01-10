import numpy as np

from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    _build_dj_relaxation_attempts,
    _select_micro_pier_candidates,
    _should_attempt_micro_pier,
)


def test_relaxation_sequence_order_and_diff():
    cfg = PierBridgeConfig(
        dj_relaxation_enabled=True,
        dj_relaxation_max_attempts=5,
        dj_relaxation_allow_floor_relaxation=True,
        dj_waypoint_weight=0.2,
        dj_waypoint_floor=0.1,
        dj_waypoint_penalty=0.1,
        segment_pool_max=100,
        max_segment_pool_max=400,
        dj_pooling_k_local=100,
        dj_pooling_k_toward=80,
        dj_pooling_k_genre=60,
        dj_pooling_k_union_max=300,
        initial_beam_width=10,
        max_beam_width=40,
    )
    attempts = _build_dj_relaxation_attempts(cfg)
    labels = [a["label"] for a in attempts]
    assert labels == [
        "baseline",
        "relax_waypoint",
        "relax_effort",
        "relax_connectors",
        "relax_transition_floor",
    ]
    assert attempts[1]["cfg"].dj_waypoint_weight == 0.1
    assert attempts[1]["cfg"].dj_waypoint_floor == 0.0
    assert attempts[1]["cfg"].dj_waypoint_penalty == 0.0
    assert attempts[2]["cfg"].segment_pool_max >= cfg.segment_pool_max
    assert attempts[2]["cfg"].initial_beam_width >= cfg.initial_beam_width
    assert attempts[3]["force_allow_detours"] is True
    assert attempts[4]["cfg"].transition_floor < cfg.transition_floor


def test_micro_pier_selection_metric():
    X_full_norm = np.array(
        [
            [1.0, 0.0],  # pier_a
            [0.0, 1.0],  # pier_b
            [0.9, 0.2],  # candidate 2 (min=0.2)
            [0.6, 0.6],  # candidate 3 (min=0.6)
        ],
        dtype=float,
    )
    scored = _select_micro_pier_candidates([2, 3], X_full_norm, 0, 1, top_k=2)
    assert scored[0][0] == 3
    assert scored[0][1] > scored[1][1]


def test_micro_pier_only_after_failures():
    assert _should_attempt_micro_pier(relaxation_enabled=True, segment_path=None) is True
    assert _should_attempt_micro_pier(relaxation_enabled=True, segment_path=[1]) is False
    assert _should_attempt_micro_pier(relaxation_enabled=False, segment_path=None) is False
