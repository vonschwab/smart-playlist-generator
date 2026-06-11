"""Variant-aware sonic matrix resolution + ``artifacts.sonic_variant_override``.

MERT plan Phase 5, items 1-2 (docs/superpowers/plans/2026-06-11-mert-sonic-embedding.md):

1. When a sonic variant is active, segment resolution must prefer
   ``X_sonic_{variant}_{start|mid|end}`` and fall back to the legacy
   ``X_sonic_start|mid|end`` keys with an INFO log.
2. ``artifacts.sonic_variant_override`` (config) wins over the artifact's
   declared ``X_sonic_variant``; an override naming a variant whose
   ``X_sonic_{variant}`` key is missing is a startup error (configured knob
   that cannot act), never a silent fallback.

All artifacts here are small synthetic NPZ files in tmp_path — the production
artifact is never touched.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from src.features.artifacts import (
    get_sonic_variant_override,
    load_artifact_bundle,
    set_sonic_variant_override,
)

N = 4
DIM = 6


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _base_payload() -> dict:
    """Legacy artifact payload: raw X_sonic + legacy segment keys, no variant."""
    return {
        "track_ids": np.array([f"t{i}" for i in range(N)], dtype=object),
        "artist_keys": np.array([f"artist-{i}" for i in range(N)], dtype=object),
        "track_artists": np.array([f"Artist {i}" for i in range(N)], dtype=object),
        "X_sonic": _rng(0).normal(size=(N, DIM)).astype(np.float32),
        "X_sonic_start": _rng(1).normal(size=(N, DIM)).astype(np.float32),
        "X_sonic_mid": _rng(2).normal(size=(N, DIM)).astype(np.float32),
        "X_sonic_end": _rng(3).normal(size=(N, DIM)).astype(np.float32),
        "X_genre_raw": np.ones((N, 2), np.float32),
        "X_genre_smoothed": np.ones((N, 2), np.float32),
        "genre_vocab": np.array(["x", "y"], dtype=object),
    }


def _mert_keys(*, segments: bool) -> dict:
    keys = {"X_sonic_mert": _rng(10).normal(size=(N, 8)).astype(np.float32)}
    if segments:
        keys["X_sonic_mert_start"] = _rng(11).normal(size=(N, 8)).astype(np.float32)
        keys["X_sonic_mert_mid"] = _rng(12).normal(size=(N, 8)).astype(np.float32)
        keys["X_sonic_mert_end"] = _rng(13).normal(size=(N, 8)).astype(np.float32)
    return keys


def _write(tmp_path, name: str, payload: dict):
    p = tmp_path / name
    np.savez(p, **payload)
    return p


@pytest.fixture(autouse=True)
def _clean_override_state():
    """Tests must never leak a process-wide override into each other."""
    set_sonic_variant_override(None)
    load_artifact_bundle.cache_clear()
    yield
    set_sonic_variant_override(None)
    load_artifact_bundle.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 — variant-aware start/mid/end resolution
# ─────────────────────────────────────────────────────────────────────────────

def test_variant_segment_keys_preferred_when_present(tmp_path):
    payload = _base_payload()
    payload["X_sonic_variant"] = np.array("mert")
    payload.update(_mert_keys(segments=True))
    p = _write(tmp_path, "variant_segs.npz", payload)

    b = load_artifact_bundle(p)

    assert b.sonic_variant == "mert"
    assert b.sonic_pre_scaled is True
    assert np.array_equal(b.X_sonic, payload["X_sonic_mert"])
    assert np.array_equal(b.X_sonic_start, payload["X_sonic_mert_start"])
    assert np.array_equal(b.X_sonic_mid, payload["X_sonic_mert_mid"])
    assert np.array_equal(b.X_sonic_end, payload["X_sonic_mert_end"])
    # Legacy segment matrices must NOT leak into a variant-active bundle.
    assert not np.array_equal(b.X_sonic_start, payload["X_sonic_start"])


def test_variant_segments_fall_back_to_legacy_with_info_log(tmp_path, caplog):
    payload = _base_payload()
    payload["X_sonic_variant"] = np.array("mert")
    payload.update(_mert_keys(segments=False))
    p = _write(tmp_path, "variant_no_segs.npz", payload)

    with caplog.at_level(logging.INFO, logger="src.features.artifacts"):
        b = load_artifact_bundle(p)

    assert np.array_equal(b.X_sonic, payload["X_sonic_mert"])
    assert np.array_equal(b.X_sonic_start, payload["X_sonic_start"])
    assert np.array_equal(b.X_sonic_mid, payload["X_sonic_mid"])
    assert np.array_equal(b.X_sonic_end, payload["X_sonic_end"])
    info_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "X_sonic_mert_start" in r.getMessage()
    ]
    assert info_msgs, "expected an INFO log naming the missing variant segment keys"


def test_no_variant_declared_is_legacy_behavior(tmp_path):
    payload = _base_payload()
    p = _write(tmp_path, "legacy.npz", payload)

    b = load_artifact_bundle(p)

    assert b.sonic_variant is None
    assert b.sonic_pre_scaled is False
    assert np.array_equal(b.X_sonic, payload["X_sonic"])
    assert np.array_equal(b.X_sonic_start, payload["X_sonic_start"])
    assert np.array_equal(b.X_sonic_mid, payload["X_sonic_mid"])
    assert np.array_equal(b.X_sonic_end, payload["X_sonic_end"])


# ─────────────────────────────────────────────────────────────────────────────
# Item 2 — artifacts.sonic_variant_override
# ─────────────────────────────────────────────────────────────────────────────

def _tower_weighted_payload() -> dict:
    payload = _base_payload()
    payload["X_sonic_variant"] = np.array("tower_weighted")
    payload["X_sonic_tower_weighted"] = _rng(20).normal(size=(N, DIM)).astype(np.float32)
    return payload


def test_override_wins_over_declared_variant(tmp_path):
    payload = _tower_weighted_payload()
    payload.update(_mert_keys(segments=True))
    p = _write(tmp_path, "override_wins.npz", payload)

    b = load_artifact_bundle(p, sonic_variant_override="mert")

    assert b.sonic_variant == "mert"
    assert b.sonic_pre_scaled is True
    assert np.array_equal(b.X_sonic, payload["X_sonic_mert"])
    assert np.array_equal(b.X_sonic_start, payload["X_sonic_mert_start"])


def test_override_with_missing_variant_key_raises(tmp_path):
    payload = _tower_weighted_payload()  # no X_sonic_mert key
    p = _write(tmp_path, "override_missing.npz", payload)

    with pytest.raises(ValueError, match=r"X_sonic_mert") as excinfo:
        load_artifact_bundle(p, sonic_variant_override="mert")
    assert "sonic_variant_override" in str(excinfo.value)


def test_override_absent_uses_declared_variant(tmp_path):
    payload = _tower_weighted_payload()
    p = _write(tmp_path, "no_override.npz", payload)

    b = load_artifact_bundle(p)

    assert b.sonic_variant == "tower_weighted"
    assert np.array_equal(b.X_sonic, payload["X_sonic_tower_weighted"])


def test_process_wide_override_setter_applies_to_plain_loads(tmp_path):
    """The config plumbing path: set_sonic_variant_override() at config load
    must make every subsequent load_artifact_bundle(path) call resolve the
    override — call sites do not pass it explicitly."""
    payload = _tower_weighted_payload()
    payload.update(_mert_keys(segments=True))
    p = _write(tmp_path, "global_override.npz", payload)

    set_sonic_variant_override("mert")
    assert get_sonic_variant_override() == "mert"
    b = load_artifact_bundle(p)

    assert b.sonic_variant == "mert"
    assert np.array_equal(b.X_sonic, payload["X_sonic_mert"])

    # Clearing the override invalidates the cache and restores declared behavior.
    set_sonic_variant_override(None)
    b2 = load_artifact_bundle(p)
    assert b2.sonic_variant == "tower_weighted"
