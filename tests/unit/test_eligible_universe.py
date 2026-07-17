"""Unit tests for the eligible-universe assembly module (Phase 1 Task 2).

Spec: docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.
Computes ONCE per generation the index set + aligned normalized sonic rows
surviving all standing hard exclusions, plus the rehomed C1 (duration) / C10
(instrumental-lean) rank-penalty vector that Task 3's corridor consumes as a
multiplicative factor on `rank_scores`.

Reference math reused (not copied) from src/playlist/candidate_pool.py:
  - duration hard cutoff + soft penalty: :678, `_compute_duration_penalty` (:112-147)
  - title hygiene hard-exclude bitmask check: :1084-1089 (`detect_title_artifacts`)
  - instrumental pool-demote math: :709-712 (`compute_instrumental_penalty`)
Seed exemption semantics mirror src/playlist/pipeline/bundle_restrict.py:66-128.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from src.playlist.candidate_pool import compute_duration_penalty
from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty
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
    # Seed also exempt from the soft rank penalty (mirrors candidate_pool's seed_mask skip).
    pos = list(result.indices).index(1)
    assert result.duration_rank_penalty[pos] == pytest.approx(1.0)


# --- penalty-vector fidelity ----------------------------------------------------


def test_duration_penalty_matches_reference_math():
    n = 8
    durations = np.full(n, 200_000.0)
    durations[3] = 260_000.0  # 30% excess -- moderate phase, no cutoff at 2.0x
    bundle = _mk_bundle(n=n, durations_ms=durations)
    result = _call(
        bundle,
        duration_reference_ms=200_000.0,
        duration_cutoff_multiplier=2.0,
        duration_penalty_weight=0.5,
    )
    expected_penalty = compute_duration_penalty(260_000.0, 200_000.0, 0.5)
    expected_factor = max(0.0, 1.0 - expected_penalty)
    pos = list(result.indices).index(3)
    assert result.duration_rank_penalty[pos] == pytest.approx(expected_factor)
    # Untouched rows keep the neutral factor.
    pos0 = list(result.indices).index(0)
    assert result.duration_rank_penalty[pos0] == pytest.approx(1.0)


def test_instrumental_penalty_matches_reference_math():
    n = 8
    voice_prob = np.full(n, 0.1)
    voice_prob[5] = 0.9
    bundle = _mk_bundle(n=n)
    result = _call(
        bundle,
        instrumental_enabled=True,
        instrumental_penalty_weight=0.4,
        voice_prob=voice_prob,
    )
    expected_penalty = compute_instrumental_penalty(voice_prob, cand=5, weight=0.4)
    expected_factor = max(0.0, 1.0 - expected_penalty)
    pos = list(result.indices).index(5)
    assert result.duration_rank_penalty[pos] == pytest.approx(expected_factor)


def test_duration_and_instrumental_penalties_combine_multiplicatively():
    n = 8
    durations = np.full(n, 200_000.0)
    durations[2] = 260_000.0
    voice_prob = np.full(n, 0.0)
    voice_prob[2] = 0.5
    bundle = _mk_bundle(n=n, durations_ms=durations)
    result = _call(
        bundle,
        duration_reference_ms=200_000.0,
        duration_cutoff_multiplier=2.0,
        duration_penalty_weight=0.5,
        instrumental_enabled=True,
        instrumental_penalty_weight=0.4,
        voice_prob=voice_prob,
    )
    dp = compute_duration_penalty(260_000.0, 200_000.0, 0.5)
    ip = compute_instrumental_penalty(voice_prob, cand=2, weight=0.4)
    expected = max(0.0, 1.0 - dp) * max(0.0, 1.0 - ip)
    pos = list(result.indices).index(2)
    assert result.duration_rank_penalty[pos] == pytest.approx(expected)


def test_instrumental_disabled_is_inert_even_with_voice_prob():
    n = 8
    voice_prob = np.full(n, 0.9)
    bundle = _mk_bundle(n=n)
    result = _call(bundle, instrumental_enabled=False, voice_prob=voice_prob, instrumental_penalty_weight=0.4)
    assert np.allclose(result.duration_rank_penalty, np.ones(n))
