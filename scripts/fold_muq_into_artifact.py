"""Fold the MuQ-MuLan sidecar into the sonic artifact as the `muq` sonic variant.

MuQ-MuLan is a CONTRASTIVE embedding (trained so cosine == similarity), validated to
beat MERT on Dylan's trusted soundalike gold set (trusted-triplet 84% vs 73%,
cross-artist retrieval median ~halved). See docs/SONIC_SPACE_EVALUATION_2026-06-29.md
and [[project_foundation_similarity_research]].

What this writes (ADDED keys — the MERT/tower variants are left intact for rollback):
- X_sonic_muq                 whole-track MuQ (center_l2 of the sidecar embedding)
- X_sonic_muq_start/_mid/_end  per-clip MuQ — replicated from the whole-track vector
  (the 1-window scan stored ONE middle-10s embedding per track; no real segments yet,
  so the beam's cos(end_A, start_B) reduces to cos(whole_A, whole_B). A future 3-window
  scan can populate real segments.)
- muq_transform_mean          the fitted centering mean (provenance)
- muq_model                   the HF model id the embeddings came from
- X_sonic_variant -> 'muq', X_sonic_pre_scaled -> True

Transform: **center_l2** (mean-center then L2) — NOT whiten. Whitening was empirically
shown to HURT MuQ (it's already well-conditioned); centering gives a small retrieval
win. One mean is fit on the valid whole-track pool and applied to all. Tracks absent
from the sidecar (the ~39 unreadable mp3/m4a) get zero vectors (cosine 0).

Usage:
    python scripts/fold_muq_into_artifact.py [--set-active muq|mert|''] [--no-center]
                                             [--dry-run] [--no-backup]

Safety: timestamped artifact backup before writing (unless --no-backup); atomic
rename. metadata.db, audio files, and the MERT shards/sidecar are never written.
"""
from __future__ import annotations

import argparse
import io
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

# Absolute real-checkout path so the fold targets the LIVE artifact regardless of cwd
# (the worktree's data/ is not symlinked here). Matches embed_muq_full.py.
ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")

ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
SIDECAR = ROOT / "data/artifacts/beat3tower_32k/muq_sidecar.npz"
_MIN_COVERAGE = 0.90  # MuQ should cover ~99.9%; this also guards against a partial scan


def load_muq_sidecar(path: Path) -> dict:
    """Load the MuQ sidecar (track_ids, embeddings, model)."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        def _read(stem: str):
            return np.load(io.BytesIO(zf.read(f"{stem}.npy")), allow_pickle=True)

        out = {
            "track_ids": _read("track_ids"),
            "embeddings": _read("embeddings"),
            "model": str(_read("model")) if "model.npy" in names else "",
        }
    return out


def align(sidecar_tids: np.ndarray, emb: np.ndarray, artifact_tids: np.ndarray):
    """Align the embedding matrix to artifact track order; zeros + invalid for misses."""
    dim = int(emb.shape[1])
    pos = {str(t): i for i, t in enumerate(sidecar_tids)}
    n = len(artifact_tids)
    whole = np.zeros((n, dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, tid in enumerate(artifact_tids):
        j = pos.get(str(tid))
        if j is not None:
            whole[i] = emb[j]
            valid[i] = True
    return whole, valid


def apply_center_l2(X: np.ndarray, mu: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean-center then L2-normalize rows; re-zero invalid rows."""
    Xc = X.astype(np.float64) - mu
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    out = (Xc / norms).astype(np.float32)
    out[~valid] = 0.0
    return out


def fold_muq(
    artifact_path: Path,
    sidecar_path: Path,
    *,
    set_active: str = "muq",
    center: bool = True,
    dry_run: bool = False,
    no_backup: bool = False,
    log_fn=print,
) -> None:
    if not Path(artifact_path).exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not Path(sidecar_path).exists():
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    log_fn("Loading MuQ sidecar...", flush=True)
    sc = load_muq_sidecar(Path(sidecar_path))
    dim = int(sc["embeddings"].shape[1])
    log_fn(f"  Sidecar: {len(sc['track_ids'])} tracks, {dim}-dim, model={sc['model']}", flush=True)

    log_fn("Loading artifact...", flush=True)
    with zipfile.ZipFile(artifact_path) as zf:
        arrays: dict[str, np.ndarray] = {}
        for name in zf.namelist():
            arrays[name.replace(".npy", "")] = np.load(io.BytesIO(zf.read(name)), allow_pickle=True)
    artifact_tids = arrays["track_ids"]
    n = len(artifact_tids)
    log_fn(f"  Artifact: {n} tracks", flush=True)

    log_fn("Aligning embeddings to artifact track order...", flush=True)
    whole, valid = align(sc["track_ids"], sc["embeddings"], artifact_tids)
    n_valid = int(valid.sum())
    coverage = n_valid / n if n else 0.0
    log_fn(f"  Covered: {n_valid}/{n} ({coverage:.1%})  Missing (zero): {n - n_valid}", flush=True)
    if coverage < _MIN_COVERAGE:
        raise ValueError(
            f"MuQ sidecar covers only {n_valid}/{n} artifact tracks ({coverage:.1%}); "
            f"below the {_MIN_COVERAGE:.0%} floor — the scan is incomplete or the wrong sidecar."
        )

    mu = whole[valid].astype(np.float64).mean(axis=0) if center else np.zeros(dim)
    log_fn(f"Applying {'center_l2' if center else 'l2'} transform...", flush=True)
    X_muq = apply_center_l2(whole, mu, valid)

    if dry_run:
        log_fn("\nDry run — no files written.", flush=True)
        log_fn(f"Would write X_sonic_muq{{,_start,_mid,_end}} ({dim}-dim) for {n} tracks", flush=True)
        log_fn(f"Would set X_sonic_variant -> {set_active!r}", flush=True)
        return

    if not no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = Path(artifact_path).with_name(Path(artifact_path).name + f".bak_{ts}")
        log_fn(f"\nBacking up -> {backup.name} ...", flush=True)
        shutil.copy2(artifact_path, backup)
        log_fn(f"  Backup written ({backup.stat().st_size / 1e6:.0f} MB)", flush=True)

    out = dict(arrays)
    out["X_sonic_muq"] = X_muq
    out["X_sonic_muq_start"] = X_muq  # no real segments yet — replicate the whole-track vector
    out["X_sonic_muq_mid"] = X_muq
    out["X_sonic_muq_end"] = X_muq
    out["muq_transform_mean"] = mu.astype(np.float32)
    out["muq_model"] = np.array(sc["model"])
    if set_active:
        out["X_sonic_variant"] = np.array(str(set_active))
        out["X_sonic_pre_scaled"] = np.array(True)

    tmp = Path(artifact_path).with_name(Path(artifact_path).stem + ".rebuildmuq.npz")
    log_fn("Writing rebuilt artifact...", flush=True)
    try:
        np.savez(tmp, **out)  # type: ignore[arg-type]
        tmp.replace(artifact_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log_fn(f"Done. Wrote X_sonic_muq{{,_start,_mid,_end}} ({dim}-dim); X_sonic_variant={set_active!r}.", flush=True)
    if not no_backup:
        log_fn(f"Original preserved as {backup.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=str(ARTIFACT))
    ap.add_argument("--sidecar", default=str(SIDECAR))
    ap.add_argument("--set-active", default="muq",
                    help="Variant to flip X_sonic_variant to ('muq' default, 'mert' to keep MERT active, '' to leave unchanged).")
    ap.add_argument("--no-center", action="store_true", help="use plain L2 instead of center_l2")
    ap.add_argument("--dry-run", action="store_true", help="print plan, don't write")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    fold_muq(
        Path(args.artifact),
        Path(args.sidecar),
        set_active=args.set_active,
        center=not args.no_center,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":
    main()
