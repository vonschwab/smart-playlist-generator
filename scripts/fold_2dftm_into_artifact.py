"""Fold the 2DFTM harmony sidecar into the sonic artifact, replacing the legacy
chroma-median harmony tower with the key-invariant 2D Fourier Transform Magnitude.

What changes
------------
- X_sonic_harmony / _start / _mid / _end: 20-dim chroma_median → 96-dim 2DFTM
  (segment positions use the full-track 2DFTM — harmonic character is global)
- X_sonic / X_sonic_start / _mid / _end: rebuilt via sqrt(w)*L2 with new harmony
- X_sonic_tower_weighted: same as X_sonic
- tower_dims: [9, 57, 20] → [9, 57, 96]
- sonic_feature_names: updated accordingly

What does NOT change
--------------------
- rhythm / timbre per-tower and segment arrays (untouched)
- genre matrices, track metadata, bpm, durations (byte-identical)
- X_sonic_raw, X_sonic_robust_whiten, normalizer_params (kept as-is)

Missing tracks (70 without audio / not in sidecar) get zero harmony vectors,
consistent with existing behavior for unanalyzed tracks.

Usage
-----
    python scripts/fold_2dftm_into_artifact.py [--dry-run] [--no-backup]

Safety
------
- Backs up the original artifact (with timestamp) before writing, unless --no-backup
- Uses atomic write (tmp rename) to avoid partial artifacts on failure
- metadata.db and audio files are never touched
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
SIDECAR  = ROOT / "data/artifacts/beat3tower_32k/harmony_2dftm_sidecar.npz"
WEIGHTS  = (0.20, 0.50, 0.30)   # rhythm, timbre, harmony — must match config.yaml
TWODFTM_DIM = 96


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float64)
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return (mat / norms).astype(np.float32)


def load_sidecar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load sidecar via zipfile+BytesIO (avoids Windows NpzFile mmap bug).

    Returns (track_ids: (N,) str array, features: (N, 96) float32).
    """
    with zipfile.ZipFile(path) as zf:
        tids   = np.load(io.BytesIO(zf.read("track_ids.npy")))
        feats  = np.load(io.BytesIO(zf.read("features.npy")))
    return tids, feats


def align_sidecar(
    sidecar_tids: np.ndarray,
    sidecar_feats: np.ndarray,
    artifact_tids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_2dftm, valid_mask) aligned to artifact_tids row order.

    Rows for tracks not in the sidecar are zero (consistent with unanalyzed tracks).
    """
    feat_by_tid = {str(t): sidecar_feats[i] for i, t in enumerate(sidecar_tids)}
    N = len(artifact_tids)
    X = np.zeros((N, TWODFTM_DIM), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i, tid in enumerate(artifact_tids):
        v = feat_by_tid.get(str(tid))
        if v is not None:
            X[i] = v
            valid[i] = True
    return X, valid


def zscore_valid(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Z-score each column using the mean/std of valid rows only.

    Missing rows are re-zeroed after z-scoring so they stay as zero vectors
    and don't pick up a false direction from the column means.
    """
    X = X.astype(np.float64)
    sub = X[valid]
    mu  = sub.mean(axis=0)
    sd  = sub.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X = (X - mu) / sd
    X[~valid] = 0.0          # re-zero missing rows after z-scoring
    return X.astype(np.float32)


def tower_weighted(
    rhythm: np.ndarray,
    timbre: np.ndarray,
    harmony: np.ndarray,
    weights: tuple[float, float, float] = WEIGHTS,
) -> np.ndarray:
    w_r, w_t, w_h = (float(w) for w in weights)
    scales = np.sqrt([w_r, w_t, w_h])
    return np.concatenate(
        [
            scales[0] * _l2_rows(rhythm),
            scales[1] * _l2_rows(timbre),
            scales[2] * _l2_rows(harmony),
        ],
        axis=1,
    ).astype(np.float32)


def fold_harmony(
    artifact_path: Path,
    sidecar_path: Path,
    *,
    dry_run: bool = False,
    no_backup: bool = False,
    log_fn=print,
) -> None:
    """Fold the 2DFTM harmony sidecar into the artifact (programmatic entry point).

    Callable from other scripts (e.g. analyze_library stage_artifacts). When
    ``no_backup=True`` the caller is responsible for preserving the original if
    needed. ``log_fn`` is called for progress messages (defaults to print).
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    log_fn("Loading sidecar...", flush=True)
    sc_tids, sc_feats = load_sidecar(sidecar_path)
    log_fn(f"  Sidecar: {len(sc_tids)} tracks, {sc_feats.shape[1]}-dim", flush=True)

    log_fn("Loading artifact...", flush=True)
    with zipfile.ZipFile(artifact_path) as zf:
        arrays: dict[str, np.ndarray] = {}
        for name in zf.namelist():
            arrays[name.replace(".npy", "")] = np.load(
                io.BytesIO(zf.read(name)), allow_pickle=True
            )
    N = len(arrays["track_ids"])
    log_fn(f"  Artifact: {N} tracks, {arrays['X_sonic'].shape[1]}-dim blend", flush=True)

    log_fn("Aligning sidecar to artifact track order...", flush=True)
    X_2dftm, valid = align_sidecar(sc_tids, sc_feats, arrays["track_ids"])
    n_valid = int(valid.sum())
    n_missing = N - n_valid
    log_fn(f"  Valid: {n_valid}  Missing (zero): {n_missing}", flush=True)

    log_fn("Z-scoring over valid pool...", flush=True)
    X_2dftm_z = zscore_valid(X_2dftm, valid)

    log_fn("Building new harmony tower (full-track used for all segment positions)...", flush=True)
    harmony_full = X_2dftm_z

    log_fn("Recomputing blend arrays...", flush=True)
    new_X_sonic       = tower_weighted(arrays["X_sonic_rhythm"],      arrays["X_sonic_timbre"],      harmony_full)
    new_X_sonic_start = tower_weighted(arrays["X_sonic_rhythm_start"], arrays["X_sonic_timbre_start"], harmony_full)
    new_X_sonic_mid   = tower_weighted(arrays["X_sonic_rhythm_mid"],   arrays["X_sonic_timbre_mid"],   harmony_full)
    new_X_sonic_end   = tower_weighted(arrays["X_sonic_rhythm_end"],   arrays["X_sonic_timbre_end"],   harmony_full)

    new_dim = new_X_sonic.shape[1]
    log_fn(f"  New blend dim: {new_dim}  (was {arrays['X_sonic'].shape[1]})", flush=True)

    old_names = list(arrays.get("sonic_feature_names", []))
    rhythm_names  = [n for n in old_names if "rhythm" in str(n)]
    timbre_names  = [n for n in old_names if "timbre" in str(n)]
    harmony_names = [f"harmony_2dftm_{i}" for i in range(TWODFTM_DIM)]
    new_names = rhythm_names + timbre_names + harmony_names
    assert len(new_names) == new_dim, f"name count {len(new_names)} != dim {new_dim}"

    new_tower_dims = np.array([9, 57, TWODFTM_DIM], dtype=np.int64)

    if dry_run:
        log_fn("\nDry run — no files written.", flush=True)
        log_fn(f"Would replace: X_sonic_harmony (→{TWODFTM_DIM}-dim), all segment harmony, all blends", flush=True)
        log_fn(f"Would update:  tower_dims, sonic_feature_names", flush=True)
        log_fn(f"New artifact size estimate: {new_dim}-dim blend, {N} tracks", flush=True)
        return

    if not no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = artifact_path.with_name(artifact_path.name + f".bak_{ts}")
        log_fn(f"\nBacking up → {backup.name} ...", flush=True)
        shutil.copy2(artifact_path, backup)
        log_fn(f"  Backup written ({backup.stat().st_size / 1e6:.0f} MB)", flush=True)

    out = dict(arrays)
    out["X_sonic_harmony"]        = harmony_full.astype(np.float32)
    out["X_sonic_harmony_start"]  = harmony_full.astype(np.float32)
    out["X_sonic_harmony_mid"]    = harmony_full.astype(np.float32)
    out["X_sonic_harmony_end"]    = harmony_full.astype(np.float32)
    out["X_sonic"]                = new_X_sonic
    out["X_sonic_tower_weighted"] = new_X_sonic
    out["X_sonic_start"]          = new_X_sonic_start
    out["X_sonic_mid"]            = new_X_sonic_mid
    out["X_sonic_end"]            = new_X_sonic_end
    out["tower_dims"]             = new_tower_dims
    out["sonic_feature_names"]    = np.array(new_names, dtype=object)
    out["X_sonic_variant"]        = np.array("tower_weighted")
    out["X_sonic_pre_scaled"]     = np.array(True)

    tmp = artifact_path.with_name(artifact_path.stem + ".rebuild2dftm.npz")
    log_fn("Writing rebuilt artifact...", flush=True)
    try:
        np.savez(tmp, **out)
        tmp.replace(artifact_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log_fn(f"Done. {artifact_path.name} rebuilt with 2DFTM harmony ({new_dim}-dim blend).", flush=True)
    if not no_backup:
        log_fn(f"Original preserved as {backup.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact",   default=str(ARTIFACT))
    ap.add_argument("--sidecar",    default=str(SIDECAR))
    ap.add_argument("--dry-run",    action="store_true", help="print plan, don't write")
    ap.add_argument("--no-backup",  action="store_true")
    args = ap.parse_args()

    artifact_path = Path(args.artifact)
    sidecar_path  = Path(args.sidecar)

    if not artifact_path.exists():
        print(f"Artifact not found: {artifact_path}")
        sys.exit(1)
    if not sidecar_path.exists():
        print(f"Sidecar not found: {sidecar_path}")
        sys.exit(1)

    fold_harmony(
        artifact_path,
        sidecar_path,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":
    main()
