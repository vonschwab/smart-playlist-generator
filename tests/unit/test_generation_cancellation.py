"""Cooperative cancellation for playlist generation.

Regression coverage for the bug where a `cancel` command was received by the
worker but generation kept grinding for minutes because the generation core
(`build_pier_bridge_playlist` -> beam search) never polled the cancellation
flag. See the worker's cooperative-cancellation contract: the running command
must poll at its own checkpoints and unwind.

These tests exercise:
  - the process-global cancel-hook primitive in ``src.cancellation``;
  - that ``build_pier_bridge_playlist`` aborts promptly when the hook signals
    cancellation, and that it polls the hook repeatedly during a normal run
    (proving the checkpoints live in the iterative core, not a single guard).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cancellation import (
    OperationCancelled,
    raise_if_cancelled,
    set_cancellation_hook,
)
from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    build_pier_bridge_playlist,
)


@pytest.fixture(autouse=True)
def _reset_cancel_hook():
    """The hook is process-global; never let a test leak it into another."""
    set_cancellation_hook(None)
    yield
    set_cancellation_hook(None)


# ---------------------------------------------------------------------------
# Primitive: set_cancellation_hook / raise_if_cancelled / OperationCancelled
# ---------------------------------------------------------------------------


def test_operation_cancelled_is_baseexception_not_exception():
    # Must unwind through the generation core's broad `except Exception`
    # handlers, like asyncio.CancelledError. If it were Exception-derived a
    # stray handler would convert cancellation into a "segment failed" and the
    # cascade would grind on.
    assert issubclass(OperationCancelled, BaseException)
    assert not issubclass(OperationCancelled, Exception)


def test_raise_if_cancelled_noop_when_no_hook():
    set_cancellation_hook(None)
    raise_if_cancelled()  # no hook registered -> no-op (CLI / tests)


def test_raise_if_cancelled_noop_when_predicate_false():
    set_cancellation_hook(lambda: False)
    raise_if_cancelled()  # predicate says "not cancelled" -> no-op


def test_raise_if_cancelled_raises_when_predicate_true():
    set_cancellation_hook(lambda: True)
    with pytest.raises(OperationCancelled):
        raise_if_cancelled()


# ---------------------------------------------------------------------------
# Builder: build_pier_bridge_playlist polls the hook and aborts
# ---------------------------------------------------------------------------


def _make_bundle(n: int = 50, sonic_dim: int = 16, genre_dim: int = 8, num_artists: int = 10) -> ArtifactBundle:
    """Deterministic synthetic ArtifactBundle (mirrors the smoke-golden harness)."""
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
        artifact_path=Path("cancel_test"),
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


def _run_builder(bundle: ArtifactBundle):
    # Three piers -> two bridge segments -> the multi-segment beam cascade runs.
    cfg = PierBridgeConfig(bridge_floor=0.0, transition_floor=0.0, center_transitions=True)
    seed_ids = ["t0", "t10", "t20"]
    seed_idx_set = {bundle.track_id_to_index[s] for s in seed_ids}
    candidate_pool = [i for i in range(len(bundle.track_ids)) if i not in seed_idx_set]
    return build_pier_bridge_playlist(
        seed_track_ids=seed_ids,
        total_tracks=15,
        bundle=bundle,
        candidate_pool_indices=candidate_pool,
        cfg=cfg,
        X_genre_smoothed=None,
    )


def test_builder_aborts_when_cancelled():
    bundle = _make_bundle()
    set_cancellation_hook(lambda: True)
    with pytest.raises(OperationCancelled):
        _run_builder(bundle)


def test_builder_polls_cancellation_repeatedly_during_normal_run():
    bundle = _make_bundle()
    calls = {"n": 0}

    def _pred() -> bool:
        calls["n"] += 1
        return False

    set_cancellation_hook(_pred)
    result = _run_builder(bundle)

    # Polled more than once per segment (2 segments here): the checkpoints live
    # inside the per-segment beam cascade, so a long single segment cannot run
    # uninterruptibly.
    assert calls["n"] > 2, f"expected repeated polling, got {calls['n']}"
    assert result is not None
