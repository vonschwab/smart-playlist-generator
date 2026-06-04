"""Unit tests for fold_2dftm_into_artifact.fold_harmony.

Focus: the fold must derive the rhythm/timbre tower widths from the ACTUAL
per-tower arrays in the artifact, never hardcode them. The builder's per-tower
dims come from PCA variance retention, so they are data-dependent and can shift
on a fresh build. A hardcoded [9, 57, 96] would silently mislabel the blend
layout and break downstream axis-slicing (the exact class of bug the
sonic_pre_scaled/tower_dims restoration fixed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.fold_2dftm_into_artifact import fold_harmony

TWODFTM_DIM = 96


def _write_artifact(path: Path, *, n: int, r_dim: int, t_dim: int, h_dim: int) -> list[str]:
    """Write a minimal synthetic artifact npz with non-standard tower dims.

    Returns the track_ids so the sidecar can be aligned to them.
    """
    rng = np.random.default_rng(0)
    tids = [f"t{i}" for i in range(n)]

    def _tower(dim):
        return rng.standard_normal((n, dim)).astype(np.float32)

    r, t, h = _tower(r_dim), _tower(t_dim), _tower(h_dim)
    blend = np.hstack([r, t, h]).astype(np.float32)  # legacy raw concat
    names = (
        [f"rhythm_{i:02d}" for i in range(r_dim)]
        + [f"timbre_{i:02d}" for i in range(t_dim)]
        + [f"harmony_{i:02d}" for i in range(h_dim)]
    )
    np.savez(
        path,
        track_ids=np.array(tids),
        X_sonic=blend,
        X_sonic_rhythm=r, X_sonic_timbre=t, X_sonic_harmony=h,
        X_sonic_rhythm_start=r, X_sonic_timbre_start=t, X_sonic_harmony_start=h,
        X_sonic_rhythm_mid=r, X_sonic_timbre_mid=t, X_sonic_harmony_mid=h,
        X_sonic_rhythm_end=r, X_sonic_timbre_end=t, X_sonic_harmony_end=h,
        X_sonic_start=blend, X_sonic_mid=blend, X_sonic_end=blend,
        sonic_feature_names=np.array(names, dtype=object),
        tower_dims=np.array([r_dim, t_dim, h_dim], dtype=np.int64),
        X_sonic_variant=np.array("robust_whiten"),  # fold overrides to tower_weighted
    )
    return tids


def _write_sidecar(path: Path, tids: list[str]) -> None:
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((len(tids), TWODFTM_DIM)).astype(np.float32)
    np.savez(path, track_ids=np.array(tids), features=feats)


def test_fold_derives_tower_dims_from_arrays_not_hardcoded(tmp_path):
    """Non-standard rhythm/timbre widths (10, 55) must be reflected in tower_dims.

    A hardcoded [9, 57, 96] would fail the internal assertion or write a wrong
    layout; this proves the derivation reads the real array widths.
    """
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "sidecar.npz"
    tids = _write_artifact(artifact, n=8, r_dim=10, t_dim=55, h_dim=20)
    _write_sidecar(sidecar, tids)

    fold_harmony(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        assert list(z["tower_dims"]) == [10, 55, TWODFTM_DIM]
        assert z["X_sonic"].shape[1] == 10 + 55 + TWODFTM_DIM
        assert z["X_sonic_harmony"].shape[1] == TWODFTM_DIM
        assert str(z["X_sonic_variant"]) == "tower_weighted"
        assert bool(z["X_sonic_pre_scaled"]) is True
        # Feature names must align with the new blend width.
        assert len(z["sonic_feature_names"]) == 10 + 55 + TWODFTM_DIM


def test_fold_missing_sidecar_tracks_get_zero_harmony(tmp_path):
    """Tracks absent from the sidecar must get zero harmony vectors, not crash."""
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "sidecar.npz"
    tids = _write_artifact(artifact, n=6, r_dim=9, t_dim=57, h_dim=20)
    # Sidecar covers only the first 4 of 6 tracks.
    _write_sidecar(sidecar, tids[:4])

    fold_harmony(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        harmony = z["X_sonic_harmony"]
        assert harmony.shape == (6, TWODFTM_DIM)
        # The two uncovered tracks must be exactly zero.
        assert np.allclose(harmony[4], 0.0)
        assert np.allclose(harmony[5], 0.0)
        # Covered tracks are z-scored, so at least one is non-zero.
        assert not np.allclose(harmony[0], 0.0)


def test_fold_dry_run_writes_nothing(tmp_path):
    """--dry-run must not modify the artifact on disk."""
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "sidecar.npz"
    tids = _write_artifact(artifact, n=5, r_dim=9, t_dim=57, h_dim=20)
    _write_sidecar(sidecar, tids)

    before = artifact.read_bytes()
    fold_harmony(artifact, sidecar, dry_run=True, no_backup=True, log_fn=lambda *a, **k: None)
    assert artifact.read_bytes() == before
