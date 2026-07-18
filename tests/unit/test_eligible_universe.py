"""Unit tests for the eligible-universe assembly module (Phase 1 Task 2).

Spec: docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.
Computes ONCE per generation the index set + aligned normalized sonic rows
surviving all standing hard exclusions, plus a `duration_rank_penalty`
placeholder vector.

`duration_rank_penalty` used to carry a REAL per-row C1 (duration) / C10
(instrumental-lean) penalty factor here, intended as a multiplicative factor
on Task 3 corridor's `rank_scores` -- but `build_corridor` never actually
read it (the beam ended up owning both effects' real selection impact
instead, to avoid double-applying at the segment_pool_max cap margin), so
computing the real math for every row of the whole eligible universe was a
dead full-universe pass (final-review finding, corridor-phase1-pooling,
2026-07-18 -- see eligible_universe.py's module docstring). The vector is
now always `1.0` (neutral) for every row, with no per-row work; the "penalty-
vector fidelity" tests below were updated to pin that neutral contract. The
real math (same formulas, same reference/weight semantics) now lives in
pier_bridge_builder.py, computed directly over each segment's small accepted
corridor -- see
tests/integration/test_corridor_pooling.py::test_mean_duration_penalty_diagnostic_reflects_real_penalty_math
for its regression coverage.

Reference math reused (not copied) from src/playlist/candidate_pool.py:
  - duration hard cutoff: :678
  - title hygiene hard-exclude bitmask check: :1084-1089 (`detect_title_artifacts`)
Seed exemption semantics mirror src/playlist/pipeline/bundle_restrict.py:66-128.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from src.playlist.pier_bridge.eligible_universe import build_eligible_universe


@dataclass
class _StubBundle:
    """Duck-typed ArtifactBundle stub -- only the fields build_eligible_universe reads."""

    track_ids: np.ndarray
    X_sonic: np.ndarray
    track_titles: Optional[np.ndarray] = None
    durations_ms: Optional[np.ndarray] = None
    track_artists: Optional[np.ndarray] = None


def _mk_sonic(n=8, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    return X


def _l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _mk_bundle(n=8, **overrides):
    track_ids = np.array([f"t{i}" for i in range(n)])
    X_sonic = _mk_sonic(n=n)
    titles = np.array(["Normal Title" for _ in range(n)])
    durations = np.full(n, 200_000.0)
    kwargs = dict(track_ids=track_ids, X_sonic=X_sonic, track_titles=titles, durations_ms=durations)
    kwargs.update(overrides)
    return _StubBundle(**kwargs)


BASE_KWARGS = dict(
    seed_indices=[],
    excluded_track_ids=set(),
    relevance_mask=None,
    duration_reference_ms=None,
    duration_cutoff_multiplier=2.0,
    duration_penalty_weight=1.0,
    title_hard_exclude_flags=frozenset(),
    instrumental_enabled=False,
    instrumental_penalty_weight=0.0,
    voice_prob=None,
)


def _call(bundle, **overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return build_eligible_universe(bundle=bundle, **kwargs)


# --- empty-exclusions identity ------------------------------------------------


def test_empty_exclusions_identity():
    bundle = _mk_bundle(n=8)
    result = _call(bundle)
    assert list(result.indices) == list(range(8))
    expected_norm = _l2norm(bundle.X_sonic)
    assert np.allclose(result.X_norm, expected_norm)
    assert np.allclose(result.duration_rank_penalty, np.ones(8))
    assert result.stats["excluded_total"] == 0
    assert result.stats["eligible_count"] == 8


# --- per-exclusion-class cases ------------------------------------------------


def test_recency_blacklist_exclusion():
    bundle = _mk_bundle(n=8)
    result = _call(bundle, excluded_track_ids={"t3"})
    assert 3 not in result.indices
    assert list(result.indices) == [0, 1, 2, 4, 5, 6, 7]
    assert result.stats["excluded_recency_blacklist"] == 1
    assert result.stats["excluded_total"] == 1


def test_duration_hard_cutoff_exclusion():
    n = 8
    durations = np.full(n, 200_000.0)
    durations[4] = 900_000.0  # 4.5x reference -- well past a 2.0x cutoff
    bundle = _mk_bundle(n=n, durations_ms=durations)
    result = _call(
        bundle,
        duration_reference_ms=200_000.0,
        duration_cutoff_multiplier=2.0,
    )
    assert 4 not in result.indices
    assert result.stats["excluded_duration_cutoff"] == 1
    assert result.stats["excluded_total"] == 1


def test_title_hygiene_exclusion():
    n = 8
    titles = np.array(["Normal Title" for _ in range(n)])
    titles[2] = "Interlude"
    bundle = _mk_bundle(n=n, track_titles=titles)
    result = _call(bundle, title_hard_exclude_flags=frozenset({"interlude", "skit", "acapella"}))
    assert 2 not in result.indices
    assert result.stats["excluded_title_hygiene"] == 1
    assert result.stats["excluded_total"] == 1


def test_relevance_mask_exclusion():
    n = 8
    mask = np.ones(n, dtype=bool)
    mask[5] = False
    bundle = _mk_bundle(n=n)
    result = _call(bundle, relevance_mask=mask)
    assert 5 not in result.indices
    assert result.stats["excluded_relevance_mask"] == 1
    assert result.stats["excluded_total"] == 1


def test_relevance_mask_none_is_passthrough():
    bundle = _mk_bundle(n=8)
    result = _call(bundle, relevance_mask=None)
    assert list(result.indices) == list(range(8))
    assert result.stats["excluded_relevance_mask"] == 0


# --- seed exemption ------------------------------------------------------------


def test_seed_exempt_from_every_exclusion():
    n = 8
    durations = np.full(n, 200_000.0)
    durations[1] = 900_000.0
    titles = np.array(["Normal Title" for _ in range(n)])
    titles[1] = "Interlude"
    mask = np.ones(n, dtype=bool)
    mask[1] = False
    bundle = _mk_bundle(n=n, durations_ms=durations, track_titles=titles)

    result = _call(
        bundle,
        seed_indices=[1],
        excluded_track_ids={"t1"},
        relevance_mask=mask,
        duration_reference_ms=200_000.0,
        duration_cutoff_multiplier=2.0,
        title_hard_exclude_flags=frozenset({"interlude", "skit", "acapella"}),
    )

    assert 1 in result.indices
    assert result.stats["excluded_recency_blacklist"] == 0
    assert result.stats["excluded_duration_cutoff"] == 0
    assert result.stats["excluded_title_hygiene"] == 0
    assert result.stats["excluded_relevance_mask"] == 0
    # Seed is also exempt from the (now-placeholder) penalty vector -- trivially
    # true post-cleanup since every row is 1.0, kept as documentation of intent.
    pos = list(result.indices).index(1)
    assert result.duration_rank_penalty[pos] == pytest.approx(1.0)


# --- penalty-vector neutrality (post-cleanup contract) --------------------------
#
# `duration_rank_penalty` used to carry REAL per-row duration/instrumental
# penalty factors here (see the module docstring's history). Final-review
# cleanup (corridor-phase1-pooling, 2026-07-18) removed that dead
# full-universe computation -- nothing ever consumed it except a diagnostic
# mean now computed directly over each segment's tiny accepted corridor in
# pier_bridge_builder.py (see
# tests/integration/test_corridor_pooling.py::test_mean_duration_penalty_diagnostic_reflects_real_penalty_math).
# These tests pin the NEW contract: `build_eligible_universe` returns an
# all-ones vector regardless of duration/instrumental configuration -- a
# regression here (a non-1.0 value reappearing) would mean the expensive
# full-universe loop was silently reintroduced.


def test_duration_penalty_is_neutral_placeholder_even_when_configured():
    n = 8
    durations = np.full(n, 200_000.0)
    durations[3] = 260_000.0  # 30% excess -- would have been a real penalty pre-cleanup
    bundle = _mk_bundle(n=n, durations_ms=durations)
    result = _call(
        bundle,
        duration_reference_ms=200_000.0,
        duration_cutoff_multiplier=2.0,
        duration_penalty_weight=0.5,
    )
    assert np.allclose(result.duration_rank_penalty, np.ones(n)), (
        "duration_rank_penalty must stay the neutral placeholder -- the real "
        "math lives in pier_bridge_builder.py now, not here"
    )


def test_instrumental_penalty_is_neutral_placeholder_even_when_configured():
    n = 8
    voice_prob = np.full(n, 0.1)
    voice_prob[5] = 0.9  # would have been a real penalty pre-cleanup
    bundle = _mk_bundle(n=n)
    result = _call(
        bundle,
        instrumental_enabled=True,
        instrumental_penalty_weight=0.4,
        voice_prob=voice_prob,
    )
    assert np.allclose(result.duration_rank_penalty, np.ones(n)), (
        "duration_rank_penalty must stay the neutral placeholder -- the real "
        "math lives in pier_bridge_builder.py now, not here"
    )


def test_instrumental_disabled_is_inert_even_with_voice_prob():
    n = 8
    voice_prob = np.full(n, 0.9)
    bundle = _mk_bundle(n=n)
    result = _call(bundle, instrumental_enabled=False, voice_prob=voice_prob, instrumental_penalty_weight=0.4)
    assert np.allclose(result.duration_rank_penalty, np.ones(n))
