"""Unit tests for Task 4: _banger_gate_inputs wiring in core.generate_playlist_ds.

Tests the pure helper _banger_gate_inputs directly (monkeypatching the loader) so
we can verify gate forwarding without artifacts or a full generate_playlist_ds call.
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


def _make_pb_cfg(cutoff: int | None) -> SimpleNamespace:
    """Minimal fake PierBridgeConfig with just popularity_rank_cutoff."""
    return SimpleNamespace(popularity_rank_cutoff=cutoff)


# ---------------------------------------------------------------------------
# Tests for _banger_gate_inputs
# ---------------------------------------------------------------------------

def test_banger_gate_inputs_returns_none_when_cutoff_is_none(monkeypatch):
    """Gate is inactive when popularity_rank_cutoff is None."""
    called = []

    def fake_loader(bundle, indices, *, db_path):
        called.append(True)
        return np.array([-1, -1, -1], dtype=int)

    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        fake_loader,
    )

    ranks, cutoff = core._banger_gate_inputs(
        _make_bundle(), _make_pb_cfg(None), db_path=":memory:"
    )

    assert ranks is None
    assert cutoff is None
    assert called == [], "loader must NOT be called when cutoff is None"


def test_banger_gate_inputs_returns_none_when_pb_cfg_is_none(monkeypatch):
    """Gate is inactive when no pier_bridge_config is provided."""
    called = []

    def fake_loader(bundle, indices, *, db_path):
        called.append(True)
        return np.array([-1, -1, -1], dtype=int)

    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        fake_loader,
    )

    ranks, cutoff = core._banger_gate_inputs(
        _make_bundle(), None, db_path=":memory:"
    )

    assert ranks is None
    assert cutoff is None
    assert called == []


def test_banger_gate_inputs_loads_ranks_and_returns_cutoff(monkeypatch):
    """When cutoff is set, the loader is called and (rank_array, cutoff) is returned."""
    fake_ranks = np.array([0, 5, -1], dtype=int)

    def fake_loader(bundle, indices, *, db_path):
        assert list(indices) == [0, 1, 2], "should load the full bundle range"
        return fake_ranks

    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        fake_loader,
    )

    ranks, cutoff = core._banger_gate_inputs(
        _make_bundle(3), _make_pb_cfg(10), db_path=":memory:"
    )

    assert cutoff == 10
    assert ranks is not None
    assert ranks.tolist() == [0, 5, -1]


def test_banger_gate_inputs_cutoff_is_int(monkeypatch):
    """The returned cutoff must be a Python int regardless of how the field is stored."""
    monkeypatch.setattr(
        "src.analyze.popularity_runner.load_pool_popularity_ranks_cached",
        lambda b, idx, *, db_path: np.full(len(b.track_ids), -1, dtype=int),
    )

    _, cutoff = core._banger_gate_inputs(
        _make_bundle(2), _make_pb_cfg(50), db_path=":memory:"
    )
    assert isinstance(cutoff, int)
    assert cutoff == 50
