"""Isolated Essentia voice/instrumental sidecar extractor.

Runs under WSL (invoked by src/analyze/energy_runner-style plumbing or directly).
Shares the msd-musicnn embedding with the energy pass but writes a SEPARATE
sidecar under <artifact>/instrumental/ — it never touches the energy/pace path.

voice_prob = mean over frames of the *voice* softmax column (column order read
from the model .json, never guessed).
"""
from __future__ import annotations

import argparse
import json
import os

# Must be set BEFORE numpy is imported: this venv's numpy links OpenBLAS
# (MAX_THREADS=64) which sizes its pthread pool at first use, and under
# multiprocessing spawn each worker re-imports this module fresh, so setting
# these later (e.g. in the pool initializer) is too late for the numpy import
# below. Uncapped, N worker processes x OpenBLAS's own threads oversubscribes
# the WSL VM's cores. Measured ~12% wall-clock win at --workers 14 (2026-07-10).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from scripts.ess_sidecar_common import (
    append_checkpoint,
    merge_sidecar_npz,
    read_checkpoint_ids,
    win_to_wsl_path,
)

# --- paths (mirror extract_energy_sidecar.py constants exactly; only OUTDIR/CKPT/
# SIDECAR differ, pointing at the instrumental/ subdir instead of energy/) ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "data", "artifacts", "beat3tower_32k")
NPZ = os.path.join(ART, "data_matrices_step1.npz")
DB = os.path.join(ROOT, "data", "metadata.db")
OUTDIR = os.path.join(ART, "instrumental")
CKPT = os.path.join(OUTDIR, "checkpoint.jsonl")
SIDECAR = os.path.join(OUTDIR, "instrumental_sidecar.npz")
MODELS = "/opt/ess/models"

EMB_PB = f"{MODELS}/msd-musicnn-1.pb"
# Confirmed at Checkpoint B (2026-07-10) against the installed model's companion
# .json: real Essentia model-zoo name is voice_instrumental-msd-musicnn-1 (task-
# then-embedding word order, version 1) — the design doc's guessed
# voice_instrumental-musicnn-msd-2 does not exist. Output node ("model/Softmax"),
# input shape (200, matching msd-musicnn-1's embedding), and classes
# (["instrumental", "voice"]) all match this file's assumptions unchanged.
VI_PB = f"{MODELS}/voice_instrumental-msd-musicnn-1.pb"
VI_JSON = f"{MODELS}/voice_instrumental-msd-musicnn-1.json"

_emb = None
_vi = None
_VOICE_COL = 1  # overwritten in _init() from the model .json


def voice_column_index(model_json_path: str) -> int:
    """Return the softmax column index for the *voice* class, from the model .json."""
    with open(model_json_path, encoding="utf-8") as f:
        meta = json.load(f)
    classes = [str(c).strip().lower() for c in meta.get("classes", [])]
    if len(classes) != 2:
        raise ValueError(f"expected 2 classes, got {classes!r}")
    matches = [i for i, c in enumerate(classes) if "voc" in c or c == "voice"]
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely identify a voice column in {classes!r}")
    return matches[0]


def _init() -> None:
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings

    warnings.filterwarnings("ignore")
    import essentia

    essentia.log.warningActive = False
    import essentia.standard as es

    global _emb, _vi, _VOICE_COL
    _emb = es.TensorflowPredictMusiCNN(graphFilename=EMB_PB, output="model/dense/BiasAdd")
    _vi = es.TensorflowPredict2D(graphFilename=VI_PB, output="model/Softmax")
    _VOICE_COL = voice_column_index(VI_JSON)


def _process(item: tuple[str, str | None]) -> dict:
    tid, path = item
    if not path or not os.path.exists(path):
        return {"track_id": tid, "missing": True}
    try:
        import essentia.standard as es

        audio = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        if len(audio) == 0:
            return {"track_id": tid, "error": "empty_audio"}
        emb = _emb(audio)
        vi = _vi(emb)  # (frames, 2)
        voice_prob = float(np.mean(vi[:, _VOICE_COL]))
        return {"track_id": tid, "voice_prob": round(voice_prob, 4), "frames": int(emb.shape[0])}
    except Exception as exc:  # never crash the pool on one track
        return {"track_id": tid, "error": repr(exc)[:200]}


def _artifact_track_ids() -> list[str]:
    z = np.load(NPZ, allow_pickle=True)
    return [str(t) for t in z["track_ids"]]


def _paths_for(track_ids: list[str]) -> dict[str, str]:
    # immutable=1: read metadata.db as a fixed snapshot of the MAIN file, ignoring the
    # WAL/-shm. Over the /mnt/c (DrvFs) boundary WAL's shared-memory index can't be
    # coordinated with the Windows side, which raised "disk I/O error" under mode=ro.
    # The analyze pipeline checkpoints the WAL into the main file before invoking this,
    # so the immutable snapshot is complete and current.
    import sqlite3

    con = sqlite3.connect(f"file:{DB}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    out: dict[str, str] = {}
    B = 900
    try:
        for i in range(0, len(track_ids), B):
            batch = track_ids[i : i + B]
            ph = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT track_id, file_path FROM tracks WHERE track_id IN ({ph})",
                tuple(batch),
            )
            for r in cur.fetchall():
                if r["file_path"]:
                    out[str(r["track_id"])] = win_to_wsl_path(r["file_path"])
    finally:
        con.close()
    return out


def _load_todo(force: bool, limit: int, track_ids: list[str] | None = None) -> list[tuple[str, str | None]]:
    """Track_id + file_path list, scoped to the artifact's track_ids, minus
    already-checkpointed ids. Mirrors extract_energy_sidecar.py's main().

    track_ids, when given, restricts to that explicit set (validate-first curated
    smoke, e.g. a vocoder + spoken-word + pure-instrumental label set) instead of
    the arbitrary front-of-artifact order --limit alone would give.
    """
    tids = _artifact_track_ids()
    if track_ids:
        wanted = set(track_ids)
        missing = wanted - set(tids)
        if missing:
            raise ValueError(f"track_ids not in artifact: {sorted(missing)}")
        tids = [t for t in tids if t in wanted]
    paths = _paths_for(tids)
    done = set() if force else read_checkpoint_ids(CKPT)
    todo = [(t, paths.get(t)) for t in tids if t not in done]
    if limit > 0:
        todo = todo[:limit]
    return todo


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="process at most N (smoke test)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-process all, ignoring checkpoint")
    ap.add_argument(
        "--track-ids",
        type=str,
        default="",
        help="comma-separated track_ids to restrict to (validate-first curated smoke)",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.merge_only:
        print(
            f"SIDECAR: {merge_sidecar_npz(SIDECAR, CKPT, columns={'voice_prob': 'voice_prob'}, meta={'model': 'voice_instrumental-msd-musicnn-1'})}"
        )
        return

    tids = _artifact_track_ids()
    done = read_checkpoint_ids(CKPT)
    explicit_ids = [t.strip() for t in args.track_ids.split(",") if t.strip()] or None
    todo = _load_todo(args.force, args.limit, track_ids=explicit_ids)
    total = len(todo)
    print(
        f"artifact={len(tids)} done={len(done)} todo={total} workers={args.workers}",
        flush=True,
    )
    if not todo:
        print("nothing to do.")
        return

    import multiprocessing as mp

    ok = missing = error = 0
    ctx = mp.get_context("spawn")
    with open(CKPT, "a", encoding="utf-8") as f:
        with ctx.Pool(args.workers, initializer=_init) as pool:
            for d in pool.imap_unordered(_process, todo, chunksize=4):
                append_checkpoint(f, d)
                if d.get("missing"):
                    missing += 1
                elif d.get("error"):
                    error += 1
                else:
                    ok += 1
    print(f"RESULT ok={ok} missing={missing} error={error} total={total}")
    print(
        f"SIDECAR: {merge_sidecar_npz(SIDECAR, CKPT, columns={'voice_prob': 'voice_prob'}, meta={'model': 'voice_instrumental-msd-musicnn-1'})}"
    )


if __name__ == "__main__":
    main()
