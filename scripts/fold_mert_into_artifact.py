"""Fold the MERT sidecar into the sonic artifact as the `mert` sonic variant.

What this writes (ADDED keys — the tower blend is left intact for rollback):
- X_sonic_mert            whole-track MERT (whiten_l2 of the mean of the 3 clips)
- X_sonic_mert_start/_mid/_end   per-clip MERT (same whiten_l2 transform)
- mert_transform_mean / mert_transform_std   the fitted whiten params (provenance)
- mert_model_revision     pinned HF revision the embeddings came from
- X_sonic_variant         flipped to the chosen active variant (default 'mert')
- X_sonic_pre_scaled      True (mert matrices are centred/whitened/L2 already)

What does NOT change:
- X_sonic / X_sonic_tower_weighted / X_sonic_start|mid|end  (the tower blend —
  the one-line rollback path: set artifacts.sonic_variant_override: tower_weighted)
- genre matrices, track metadata, bpm, durations, all other keys (byte-identical)

Transform (locked by calibration, see docs/run_audits/mert_full/VERDICT.md):
**whiten_l2** — mean-center → per-dim std → L2. ONE transform (mean, std) is fit
on the whole-track vectors over the valid pool and applied to every clip, so
start/mid/end live in the same space (the beam scores cos(end_A, start_B)).
Tracks absent from the sidecar get zero vectors (cosine 0 — same as unanalyzed).

Usage
-----
    python scripts/fold_mert_into_artifact.py [--set-active mert|tower_weighted]
                                              [--dry-run] [--no-backup]

Safety
------
- Timestamped artifact backup before writing (unless --no-backup); atomic rename.
- metadata.db, audio files, and the MERT shards/sidecar are never written.
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
SIDECAR = ROOT / "data/artifacts/beat3tower_32k/mert_sidecar.npz"

# Minimum fraction of artifact tracks that must be covered by the sidecar before
# we trust the fold. Below this it's almost certainly the wrong sidecar.
_MIN_COVERAGE = 0.50


def load_mert_sidecar(path: Path) -> dict:
    """Load the MERT sidecar (track_ids, emb_start|mid|end, model_revision)."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        def _read(stem: str):
            return np.load(io.BytesIO(zf.read(f"{stem}.npy")), allow_pickle=True)

        out = {
            "track_ids": _read("track_ids"),
            "emb_start": _read("emb_start"),
            "emb_mid": _read("emb_mid"),
            "emb_end": _read("emb_end"),
        }
        out["model_revision"] = (
            str(_read("model_revision")) if "model_revision.npy" in names else ""
        )
    return out


def align_clips(
    sidecar_tids: np.ndarray,
    emb_start: np.ndarray,
    emb_mid: np.ndarray,
    emb_end: np.ndarray,
    artifact_tids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align the three clip matrices to artifact track order.

    Returns (start, mid, end, valid_mask). Rows for tracks missing from the
    sidecar are left as zeros and marked invalid.
    """
    dim = int(emb_mid.shape[1])
    pos = {str(t): i for i, t in enumerate(sidecar_tids)}
    n = len(artifact_tids)
    start = np.zeros((n, dim), dtype=np.float32)
    mid = np.zeros((n, dim), dtype=np.float32)
    end = np.zeros((n, dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, tid in enumerate(artifact_tids):
        j = pos.get(str(tid))
        if j is not None:
            start[i] = emb_start[j]
            mid[i] = emb_mid[j]
            end[i] = emb_end[j]
            valid[i] = True
    return start, mid, end, valid


def fit_whiten(whole_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit whiten_l2 params (per-dim mean, std) on the valid whole-track pool."""
    X = whole_valid.astype(np.float64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return mu, sigma


def apply_whiten_l2(
    X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Apply (x-mu)/sigma then L2-normalize rows; re-zero invalid rows."""
    Xw = (X.astype(np.float64) - mu) / sigma
    norms = np.linalg.norm(Xw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    out = (Xw / norms).astype(np.float32)
    out[~valid] = 0.0
    return out


def fold_mert(
    artifact_path: Path,
    sidecar_path: Path,
    *,
    set_active: str = "mert",
    dry_run: bool = False,
    no_backup: bool = False,
    log_fn=print,
) -> None:
    """Fold the MERT sidecar into the artifact as the `mert` sonic variant."""
    if not Path(artifact_path).exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not Path(sidecar_path).exists():
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    log_fn("Loading MERT sidecar...", flush=True)
    sc = load_mert_sidecar(Path(sidecar_path))
    dim = int(sc["emb_mid"].shape[1])
    log_fn(f"  Sidecar: {len(sc['track_ids'])} tracks, {dim}-dim, rev={sc['model_revision'][:12]}", flush=True)

    log_fn("Loading artifact...", flush=True)
    with zipfile.ZipFile(artifact_path) as zf:
        arrays: dict[str, np.ndarray] = {}
        for name in zf.namelist():
            arrays[name.replace(".npy", "")] = np.load(
                io.BytesIO(zf.read(name)), allow_pickle=True
            )
    artifact_tids = arrays["track_ids"]
    n = len(artifact_tids)
    log_fn(f"  Artifact: {n} tracks", flush=True)

    log_fn("Aligning clips to artifact track order...", flush=True)
    start, mid, end, valid = align_clips(
        sc["track_ids"], sc["emb_start"], sc["emb_mid"], sc["emb_end"], artifact_tids
    )
    n_valid = int(valid.sum())
    coverage = n_valid / n if n else 0.0
    log_fn(f"  Covered: {n_valid}/{n} ({coverage:.1%})  Missing (zero): {n - n_valid}", flush=True)
    if coverage < _MIN_COVERAGE:
        raise ValueError(
            f"MERT sidecar covers only {n_valid}/{n} artifact tracks ({coverage:.1%}); "
            f"below the {_MIN_COVERAGE:.0%} overlap floor — wrong sidecar or track-id mismatch."
        )

    log_fn("Fitting whiten_l2 on the valid whole-track pool...", flush=True)
    whole_raw = (start + mid + end) / 3.0
    mu, sigma = fit_whiten(whole_raw[valid])

    log_fn("Applying whiten_l2 to whole + clips...", flush=True)
    X_mert = apply_whiten_l2(whole_raw, mu, sigma, valid)
    X_mert_start = apply_whiten_l2(start, mu, sigma, valid)
    X_mert_mid = apply_whiten_l2(mid, mu, sigma, valid)
    X_mert_end = apply_whiten_l2(end, mu, sigma, valid)

    if dry_run:
        log_fn("\nDry run — no files written.", flush=True)
        log_fn(f"Would write X_sonic_mert{{,_start,_mid,_end}} ({dim}-dim) for {n} tracks", flush=True)
        log_fn(f"Would set X_sonic_variant -> {set_active!r}", flush=True)
        return

    if not no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = Path(artifact_path).with_name(Path(artifact_path).name + f".bak_{ts}")
        log_fn(f"\nBacking up -> {backup.name} ...", flush=True)
        shutil.copy2(artifact_path, backup)
        log_fn(f"  Backup written ({backup.stat().st_size / 1e6:.0f} MB)", flush=True)

    out = dict(arrays)
    out["X_sonic_mert"] = X_mert
    out["X_sonic_mert_start"] = X_mert_start
    out["X_sonic_mert_mid"] = X_mert_mid
    out["X_sonic_mert_end"] = X_mert_end
    out["mert_transform_mean"] = mu.astype(np.float32)
    out["mert_transform_std"] = sigma.astype(np.float32)
    out["mert_model_revision"] = np.array(sc["model_revision"])
    if set_active:
        out["X_sonic_variant"] = np.array(str(set_active))
        out["X_sonic_pre_scaled"] = np.array(True)

    tmp = Path(artifact_path).with_name(Path(artifact_path).stem + ".rebuildmert.npz")
    log_fn("Writing rebuilt artifact...", flush=True)
    try:
        np.savez(tmp, **out)  # type: ignore[arg-type]
        tmp.replace(artifact_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log_fn(
        f"Done. Wrote X_sonic_mert{{,_start,_mid,_end}} ({dim}-dim); "
        f"X_sonic_variant={set_active!r}.",
        flush=True,
    )
    if not no_backup:
        log_fn(f"Original preserved as {backup.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=str(ARTIFACT))
    ap.add_argument("--sidecar", default=str(SIDECAR))
    ap.add_argument(
        "--set-active",
        default="mert",
        help="Variant to flip X_sonic_variant to ('mert' default, 'tower_weighted' to keep towers active, '' to leave unchanged).",
    )
    ap.add_argument("--dry-run", action="store_true", help="print plan, don't write")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    artifact_path = Path(args.artifact)
    sidecar_path = Path(args.sidecar)
    if not artifact_path.exists():
        print(f"Artifact not found: {artifact_path}")
        sys.exit(1)
    if not sidecar_path.exists():
        print(f"Sidecar not found: {sidecar_path}")
        sys.exit(1)

    fold_mert(
        artifact_path,
        sidecar_path,
        set_active=args.set_active,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":
    main()
