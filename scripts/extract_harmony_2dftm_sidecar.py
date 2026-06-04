"""Extract frame-level 2DFTM (key-invariant) harmony features for the whole library
into a sidecar npz, aligned to the artifact's track_ids.

This is the expensive step that powers (a) the blind legacy-vs-2DFTM audition
head-to-head and (b) — if the audition confirms the win — the harmony-tower
rebuild (fold the sidecar in via a cheap re-concat).

Representation (validated by the probe, +0.210 isolated / +0.361 blend):
  load -> effects.harmonic(margin=8) -> chroma_cqt -> resample time to 256
       -> |FFT2| -> keep low TWODFTM_K time-freq bins -> flatten (12*K dims).

SAFETY: audio is read-only (librosa.load only; never written/moved). metadata.db
is read-only (URI mode=ro). Output is a NEW sidecar npz; the artifact and the DB
are never modified. Resumable: re-run to continue; already-done track_ids skip.

Usage:
    python scripts/extract_harmony_2dftm_sidecar.py [--workers N] [--limit N]
"""
from __future__ import annotations

# Pin per-process math threads BEFORE numpy/librosa import, so N worker processes
# don't oversubscribe cores. (Workers re-import this module on spawn.)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
DB = ROOT / "data/metadata.db"
SIDECAR = ROOT / "data/artifacts/beat3tower_32k/harmony_2dftm_sidecar.npz"
SR = 22050
HOP = 512
TWODFTM_K = 8
T_FIXED = 256
FEAT_DIM = 12 * TWODFTM_K


def extract_one(args):
    """Worker: (track_id, path) -> (track_id, float32[FEAT_DIM] or None). Read-only."""
    tid, path = args
    import librosa

    try:
        if not path or not os.path.exists(path):
            return tid, None
        y, _ = librosa.load(path, sr=SR, mono=True)
        if y.size < SR:
            return tid, None
        y_h = librosa.effects.harmonic(y, margin=8)
        chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP)  # (12, T)
        if chroma.shape[1] >= 2:
            idx = np.linspace(0, chroma.shape[1] - 1, T_FIXED)
            rs = np.stack([np.interp(idx, np.arange(chroma.shape[1]), r) for r in chroma])
        else:
            rs = np.repeat(chroma, T_FIXED, axis=1)[:, :T_FIXED]
        v = np.abs(np.fft.fft2(rs))[:, :TWODFTM_K].flatten().astype(np.float32)
        return tid, v
    except Exception:
        return tid, None


def load_paths(track_ids: list[str]) -> dict[str, str]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT track_id, file_path FROM tracks").fetchall()
    finally:
        con.close()
    m = {str(t): p for t, p in rows}
    return {t: m.get(t) for t in track_ids}


def load_sidecar() -> dict[str, np.ndarray]:
    if not SIDECAR.exists():
        return {}
    z = np.load(SIDECAR, allow_pickle=True)
    return {str(t): z["features"][i] for i, t in enumerate(z["track_ids"])}


def save_sidecar(done: dict[str, np.ndarray]) -> None:
    tids = list(done.keys())
    feats = np.stack([done[t] for t in tids]) if tids else np.zeros((0, FEAT_DIM), np.float32)
    # Temp name must end in .npz, else np.savez appends another .npz and the
    # subsequent replace() can't find the file it actually wrote.
    tmp = SIDECAR.with_name(SIDECAR.stem + ".tmp.npz")
    np.savez(tmp, track_ids=np.array(tids), features=feats)
    tmp.replace(SIDECAR)  # atomic; survives interruption


def main() -> None:
    import multiprocessing as mp

    from src.features.artifacts import load_artifact_bundle

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="extract at most N (smoke test)")
    ap.add_argument("--checkpoint-every", type=int, default=400)
    args = ap.parse_args()

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    track_ids = [str(t) for t in bundle.track_ids]
    paths = load_paths(track_ids)

    done = load_sidecar()
    todo = [(t, paths[t]) for t in track_ids if t not in done and paths.get(t)]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Library: {len(track_ids)} tracks. Already done: {len(done)}. "
          f"To extract: {len(todo)}. Workers: {args.workers}.", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    t0 = time.time()
    n_ok = n_fail = 0
    with mp.Pool(args.workers) as pool:
        for k, (tid, vec) in enumerate(pool.imap_unordered(extract_one, todo, chunksize=4), 1):
            if vec is not None:
                done[tid] = vec
                n_ok += 1
            else:
                n_fail += 1
            if k % args.checkpoint_every == 0:
                save_sidecar(done)
                rate = k / (time.time() - t0)
                eta_h = (len(todo) - k) / rate / 3600
                print(f"  {k}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                      f"{rate:.1f} trk/s  ETA {eta_h:.1f}h", flush=True)
    save_sidecar(done)
    print(f"Done. {n_ok} extracted, {n_fail} failed/missing. "
          f"Sidecar: {len(done)} total -> {SIDECAR}", flush=True)


if __name__ == "__main__":
    main()
