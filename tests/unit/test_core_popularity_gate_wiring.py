"""Unit tests for Task 4 / Task SM: _banger_gate_inputs wiring in core.generate_playlist_ds.

Tests the pure helper _banger_gate_inputs directly (monkeypatching the loader) so
we can verify gate forwarding without artifacts or a full generate_playlist_ds call.

Task SM refactored _banger_gate_inputs to take a rank_cutoff int directly (instead of
a pb_cfg object), so the effective cutoff can be computed at the call site from either
pier_bridge_config (artist mode) or pb_overrides (seed mode).
"""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

import src.playlist.pipeline.core as core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(n: int = 3) -> SimpleNamespace:
    """Minimal fake bundle with the attributes _banger_gate_inputs needs."""
    return SimpleNamespace(
        track_ids=np.array([f"t{i}" for i in range(n)], dtype=object),
        artist_keys=np.array(["nirvana"] * n, dtype=object),
        track_titles=np.array([f"Song {i}" for i in range(n)], dtype=object),
    )


# ---------------------------------------------------------------------------
# Tests for _banger_gate_inputs (Task SM: takes cutoff int, not pb_cfg)
# ---------------------------------------------------------------------------

def test_gate_inputs_none_when_cutoff_none(monkeypatch):
    """Gate is inactive when rank_cutoff is None — loader must NOT be called."""
    called = {"n": 0}
    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or np.array([0]),
    )
    ranks, cutoff = core._banger_gate_inputs(object(), None, db_path="")
    assert ranks is None and cutoff is None and called["n"] == 0   # loader NOT called


def test_gate_inputs_loads_when_cutoff_set(monkeypatch):
    """When cutoff is set, the loader is called and (rank_array, cutoff) is returned."""
    bundle = _make_bundle(2)
    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        lambda b, idx, *, db_path: np.array([0, 5]),
    )
    ranks, cutoff = core._banger_gate_inputs(bundle, 10, db_path="")
    assert cutoff == 10 and ranks is not None and list(ranks) == [0, 5]


def test_gate_inputs_cutoff_is_int(monkeypatch):
    """The returned cutoff must be a Python int."""
    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        lambda b, idx, *, db_path: np.full(len(b.track_ids), -1, dtype=int),
    )
    _, cutoff = core._banger_gate_inputs(_make_bundle(2), 50, db_path="")
    assert isinstance(cutoff, int)
    assert cutoff == 50
