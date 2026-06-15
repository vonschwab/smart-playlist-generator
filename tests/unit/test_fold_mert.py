"""Unit tests for fold_mert_into_artifact.fold_mert.

The fold takes the MERT sidecar (raw 768-d start/mid/end clip embeddings),
fits whiten_l2 on the full valid pool, applies it, and writes the
X_sonic_mert{,_start,_mid,_end} artifact keys — WITHOUT touching the existing
tower blend (X_sonic / X_sonic_tower_weighted stay intact as the rollback path).
It flips X_sonic_variant to the chosen active variant (default 'mert').

Embedding dim is derived from the sidecar, never hardcoded (tests use 16-d).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.fold_mert_into_artifact import fold_mert

MERT_DIM = 16
TOWER_DIM = 12  # arbitrary stand-in for the 162-d tower blend


def _write_artifact(path: Path, *, n: int) -> list[str]:
    rng = np.random.default_rng(0)
    tids = [f"t{i}" for i in range(n)]
    blend = rng.standard_normal((n, TOWER_DIM)).astype(np.float32)
    np.savez(
        path,
        track_ids=np.array(tids),
        X_sonic=blend,
        X_sonic_tower_weighted=blend,
        X_sonic_start=blend, X_sonic_mid=blend, X_sonic_end=blend,
        tower_dims=np.array([3, 6, 3], dtype=np.int64),
        X_sonic_variant=np.array("tower_weighted"),
    )
    return tids


def _write_sidecar(path: Path, tids: list[str], *, dim: int = MERT_DIM) -> None:
    rng = np.random.default_rng(1)

    def clip():
        return rng.standard_normal((len(tids), dim)).astype(np.float32)

    np.savez(
        path,
        track_ids=np.array(tids),
        emb_start=clip(), emb_mid=clip(), emb_end=clip(),
        model_name=np.array("m-a-p/MERT-v1-95M"),
        model_revision=np.array("deadbeef"),
    )


def test_fold_writes_mert_keys_default_active(tmp_path):
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    tids = _write_artifact(artifact, n=10)
    _write_sidecar(sidecar, tids)

    fold_mert(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        for key in ("X_sonic_mert", "X_sonic_mert_start", "X_sonic_mert_mid", "X_sonic_mert_end"):
            assert key in z, f"missing {key}"
            assert z[key].shape == (10, MERT_DIM)
        # variant flipped to mert (the chosen default)
        assert str(z["X_sonic_variant"]) == "mert"
        # tower blend preserved untouched for rollback
        assert z["X_sonic"].shape[1] == TOWER_DIM
        assert "X_sonic_tower_weighted" in z


def test_fold_rows_are_l2_normalized(tmp_path):
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    tids = _write_artifact(artifact, n=20)
    _write_sidecar(sidecar, tids)

    fold_mert(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        for key in ("X_sonic_mert", "X_sonic_mert_start", "X_sonic_mert_end"):
            norms = np.linalg.norm(z[key], axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), f"{key} rows not unit-norm"


def test_fold_missing_tracks_get_zero_vectors(tmp_path):
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    tids = _write_artifact(artifact, n=6)
    _write_sidecar(sidecar, tids[:4])  # only first 4 covered

    fold_mert(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        m = z["X_sonic_mert"]
        assert m.shape == (6, MERT_DIM)
        assert np.allclose(m[4], 0.0)
        assert np.allclose(m[5], 0.0)
        assert not np.allclose(m[0], 0.0)


def test_fold_stores_transform_params_and_revision(tmp_path):
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    tids = _write_artifact(artifact, n=8)
    _write_sidecar(sidecar, tids)

    fold_mert(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        assert z["mert_transform_mean"].shape == (MERT_DIM,)
        assert z["mert_transform_std"].shape == (MERT_DIM,)
        assert str(z["mert_model_revision"]) == "deadbeef"


def test_fold_set_active_tower_weighted_keeps_variant(tmp_path):
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    tids = _write_artifact(artifact, n=8)
    _write_sidecar(sidecar, tids)

    fold_mert(artifact, sidecar, set_active="tower_weighted",
              no_backup=True, log_fn=lambda *a, **k: None)

    with np.load(artifact, allow_pickle=True) as z:
        # keys written, but variant pointer NOT flipped
        assert "X_sonic_mert" in z
        assert str(z["X_sonic_variant"]) == "tower_weighted"


def test_fold_dry_run_writes_nothing(tmp_path):
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    tids = _write_artifact(artifact, n=5)
    _write_sidecar(sidecar, tids)

    before = artifact.read_bytes()
    fold_mert(artifact, sidecar, dry_run=True, no_backup=True, log_fn=lambda *a, **k: None)
    assert artifact.read_bytes() == before


def test_fold_no_overlap_errors(tmp_path):
    """A sidecar whose tracks don't intersect the artifact is a hard error,
    not a silently all-zero MERT matrix."""
    artifact = tmp_path / "art.npz"
    sidecar = tmp_path / "mert.npz"
    _write_artifact(artifact, n=6)
    _write_sidecar(sidecar, ["x0", "x1", "x2"])  # foreign tracks only

    with pytest.raises(ValueError, match="overlap|coverage|match"):
        fold_mert(artifact, sidecar, no_backup=True, log_fn=lambda *a, **k: None)
