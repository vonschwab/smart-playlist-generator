"""
Pier-bridge smoke golden tests — safety net for Tier-3.1 PR-7+.

Strategy: directly call build_pier_bridge_playlist with a synthetic
ArtifactBundle (no NPZ, no pipeline), capture the ordered track_ids
and key stats, compare against JSON goldens in
tests/unit/goldens/pier_bridge/.

This test exercises the algorithmic core — beam search, pool building,
min-gap enforcement — so any extraction in PR-7 (pool.py), PR-8
(beam.py + genre_targets.py), and PR-9 (assemble.py) that silently
changes behaviour will be caught here.

Also contains direct unit tests for three small functions that are
currently untested and are targets for extraction:
  _compute_bridge_score, _compute_edge_scores, _enforce_min_gap_global.

To re-baseline after an INTENTIONAL behaviour change, delete the
relevant golden file and re-run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    _compute_bridge_score,
    _compute_edge_scores,
    _enforce_min_gap_global,
    build_pier_bridge_playlist,
)

GOLDEN_DIR = Path(__file__).parent / "goldens" / "pier_bridge"


# ---------------------------------------------------------------------------
# Direct unit tests — _compute_bridge_score
# ---------------------------------------------------------------------------


class TestComputeBridgeScore:
    def test_no_experiment_is_harmonic_mean(self):
        # H-mean(0.4, 0.6) = 2*0.4*0.6 / (0.4+0.6) = 0.48
        score = _compute_bridge_score(
            0.4, 0.6,
            experiment_enabled=False,
            experiment_min_weight=0.25,
            experiment_balance_weight=0.15,
        )
        assert abs(score - 0.48) < 1e-9

    def test_equal_sims_no_experiment(self):
        score = _compute_bridge_score(
            0.7, 0.7,
            experiment_enabled=False,
            experiment_min_weight=0.0,
            experiment_balance_weight=0.0,
        )
        assert abs(score - 0.7) < 1e-9

    def test_zero_denominator_returns_zero(self):
        score = _compute_bridge_score(
            0.0, 0.0,
            experiment_enabled=False,
            experiment_min_weight=0.0,
            experiment_balance_weight=0.0,
        )
        assert score == 0.0

    def test_experiment_enabled_returns_float_in_range(self):
        score = _compute_bridge_score(
            0.6, 0.8,
            experiment_enabled=True,
            experiment_min_weight=0.25,
            experiment_balance_weight=0.30,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Direct unit tests — _compute_edge_scores
# ---------------------------------------------------------------------------


class TestComputeEdgeScores:
    @pytest.fixture
    def cfg(self):
        return PierBridgeConfig(center_transitions=False, transition_floor=0.0)

    def test_empty_path_returns_one_one(self, cfg):
        X = np.eye(4)
        assert _compute_edge_scores([], X, None, None, None, cfg) == (1.0, 1.0)

    def test_single_track_returns_one_one(self, cfg):
        X = np.eye(4)
        assert _compute_edge_scores([2], X, None, None, None, cfg) == (1.0, 1.0)

    def test_two_track_path_min_equals_mean(self, cfg):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 8))
        worst, mean = _compute_edge_scores([0, 1], X, None, None, None, cfg)
        assert abs(worst - mean) < 1e-9

    def test_three_track_path_min_leq_mean(self, cfg):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 8))
        worst, mean = _compute_edge_scores([0, 1, 2], X, None, None, None, cfg)
        assert worst <= mean + 1e-9


# ---------------------------------------------------------------------------
# Direct unit tests — _enforce_min_gap_global
# ---------------------------------------------------------------------------


class TestEnforceMinGapGlobal:
    def test_empty_input(self):
        result, dropped = _enforce_min_gap_global([], min_gap=1)
        assert result == []
        assert dropped == 0

    def test_min_gap_zero_is_noop(self):
        artist_keys = np.array(["a", "a", "a"])
        result, dropped = _enforce_min_gap_global([0, 1, 2], artist_keys, min_gap=0)
        assert result == [0, 1, 2]
        assert dropped == 0

    def test_drops_adjacent_same_artist(self):
        # min_gap=1: "a" at idx=0 blocks "a" at idx=1; "b" at idx=2 passes
        artist_keys = np.array(["a", "a", "b"])
        result, dropped = _enforce_min_gap_global([0, 1, 2], artist_keys, min_gap=1)
        assert 0 in result
        assert 1 not in result
        assert 2 in result
        assert dropped == 1

    def test_gap_2_drops_within_window(self):
        # min_gap=2: "a" appears at position 0 and 2 — position 2 is within the window
        artist_keys = np.array(["a", "b", "a", "c"])
        result, dropped = _enforce_min_gap_global([0, 1, 2, 3], artist_keys, min_gap=2)
        assert 0 in result
        assert 1 in result
        assert 2 not in result
        assert 3 in result
        assert dropped == 1

    def test_distinct_artists_nothing_dropped(self):
        artist_keys = np.array(["a", "b", "c", "d"])
        result, dropped = _enforce_min_gap_global([0, 1, 2, 3], artist_keys, min_gap=2)
        assert result == [0, 1, 2, 3]
        assert dropped == 0


# ---------------------------------------------------------------------------
# Smoke golden tests — build_pier_bridge_playlist end-to-end
# ---------------------------------------------------------------------------


def _make_bundle(n: int, sonic_dim: int, genre_dim: int, num_artists: int) -> ArtifactBundle:
    """Deterministic ArtifactBundle constructed directly (no NPZ file)."""
    rng = np.random.default_rng(7)
    track_ids = np.array([f"t{i}" for i in range(n)])
    artist_keys = np.array([f"a{i % num_artists}" for i in range(n)])
    track_artists = np.array([f"Artist {i % num_artists}" for i in range(n)])
    track_titles = np.array([f"Song {i}" for i in range(n)])
    X_sonic = rng.standard_normal((n, sonic_dim))
    X_genre_raw = (rng.random((n, genre_dim)) > 0.7).astype(float)
    X_genre_smoothed = np.clip(X_genre_raw + 0.05 * rng.standard_normal((n, genre_dim)), 0.0, 1.0)
    genre_vocab = np.array([f"g{i}" for i in range(genre_dim)])
    durations_ms = np.full(n, 200_000, dtype=np.int64)
    track_id_to_index = {str(tid): i for i, tid in enumerate(track_ids)}

    return ArtifactBundle(
        artifact_path=Path("smoke_test"),
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        track_id_to_index=track_id_to_index,
        durations_ms=durations_ms,
    )


@pytest.fixture(scope="module")
def smoke_bundle():
    return _make_bundle(n=50, sonic_dim=16, genre_dim=8, num_artists=10)


SMOKE_SCENARIOS: Dict[str, Any] = {
    # Two piers from different artists, default config, no genre gate.
    "two_seeds_default": {
        "seed_track_ids": ["t0", "t10"],
        "total_tracks": 10,
        "cfg_kwargs": {
            "bridge_floor": 0.0,
            "transition_floor": 0.0,
            "center_transitions": False,
        },
    },
    # Single-seed arc: one pier doubles as start and end anchor.
    "single_seed_arc": {
        "seed_track_ids": ["t0"],
        "total_tracks": 8,
        "cfg_kwargs": {
            "bridge_floor": 0.0,
            "transition_floor": 0.0,
            "center_transitions": False,
        },
    },
    # Three piers (two bridge segments), centered transitions — exercises the
    # production center_transitions=True code path and multi-segment assembly.
    "three_seeds_centered": {
        "seed_track_ids": ["t0", "t10", "t20"],
        "total_tracks": 15,
        "cfg_kwargs": {
            "bridge_floor": 0.0,
            "transition_floor": 0.0,
            "center_transitions": True,
        },
    },
}


def _serialize_result(result: Any) -> Dict[str, Any]:
    """Reduce PierBridgeResult to a JSON-comparable snapshot.

    Track IDs catch any reordering; the four scalar stats catch structural
    and quality regressions. Transition metrics are rounded to 4 dp so
    negligible float drift doesn't break the golden.
    """
    stats = result.stats or {}
    min_t = stats.get("min_transition")
    mean_t = stats.get("mean_transition")
    return {
        "track_ids": list(result.track_ids),
        "actual_tracks": int(stats.get("actual_tracks", len(result.track_ids))),
        "num_seeds": int(stats.get("num_seeds", 0)),
        "single_seed_arc": bool(stats.get("single_seed_arc", False)),
        "segments_successful": int(stats.get("segments_successful", 0)),
        "success": bool(result.success),
        "min_transition": round(float(min_t), 4) if min_t is not None else None,
        "mean_transition": round(float(mean_t), 4) if mean_t is not None else None,
    }


@pytest.mark.parametrize("scenario_name", sorted(SMOKE_SCENARIOS.keys()))
def test_pier_bridge_smoke_golden(scenario_name, smoke_bundle):
    scenario = SMOKE_SCENARIOS[scenario_name]
    # Smoke goldens are stable CORE-beam regression baselines: pin variable bridge
    # length OFF (it became the live default 2026-06-28) so these track the rigid
    # even-split beam, not the flex — which has its own coverage in
    # tests/unit/test_var_bridge_integration.py.
    cfg = PierBridgeConfig(variable_bridge_length=False, **scenario["cfg_kwargs"])

    seed_ids = scenario["seed_track_ids"]
    seed_idx_set = {smoke_bundle.track_id_to_index[s] for s in seed_ids}
    candidate_pool = [i for i in range(len(smoke_bundle.track_ids)) if i not in seed_idx_set]

    result = build_pier_bridge_playlist(
        seed_track_ids=seed_ids,
        total_tracks=scenario["total_tracks"],
        bundle=smoke_bundle,
        candidate_pool_indices=candidate_pool,
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=None,
    )

    snapshot = _serialize_result(result)

    golden_path = GOLDEN_DIR / f"{scenario_name}.json"
    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        pytest.skip(f"Created golden baseline at {golden_path}; rerun to verify.")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))

    assert snapshot["track_ids"] == expected["track_ids"], (
        f"Track ordering changed for {scenario_name}.\n"
        f"  Got:      {snapshot['track_ids']}\n"
        f"  Expected: {expected['track_ids']}"
    )
    for key in ("actual_tracks", "num_seeds", "single_seed_arc", "segments_successful", "success"):
        assert snapshot[key] == expected[key], (
            f"{key} changed for {scenario_name}: {snapshot[key]!r} vs {expected[key]!r}"
        )
    for key in ("min_transition", "mean_transition"):
        sv, ev = snapshot[key], expected[key]
        if sv is not None and ev is not None:
            assert abs(sv - ev) < 0.01, (
                f"{key} drifted for {scenario_name}: {sv} vs {ev}"
            )
