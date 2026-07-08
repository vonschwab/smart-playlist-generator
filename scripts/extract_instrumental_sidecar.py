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

import numpy as np

from scripts.ess_sidecar_common import (
    append_checkpoint,
    merge_sidecar_npz,
    read_checkpoint_ids,
)

# --- paths (mirror extract_energy_sidecar.py constants; ART resolved the same way) ---
ART = os.environ.get("PLAYLIST_ARTIFACT_DIR", "data/artifacts/beat3tower_32k")
MODELS = os.environ.get("ESS_MODELS", "/opt/ess/models")
OUTDIR = os.path.join(ART, "instrumental")
CKPT = os.path.join(OUTDIR, "checkpoint.jsonl")
SIDECAR = os.path.join(OUTDIR, "instrumental_sidecar.npz")

EMB_PB = f"{MODELS}/msd-musicnn-1.pb"
VI_PB = f"{MODELS}/voice_instrumental-musicnn-msd-2.pb"     # confirm exact name at impl time
VI_JSON = f"{MODELS}/voice_instrumental-musicnn-msd-2.json"  # confirm exact name at impl time

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
    for i, c in enumerate(classes):
        if "voc" in c or c == "voice" or "vocal" in c:
            return i
    raise ValueError(f"cannot identify a voice column in classes {classes!r}")


def _init() -> None:
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
        return {"track_id": tid, "error": str(exc)[:200]}


def _load_todo(force: bool, limit: int) -> list[tuple[str, str | None]]:
    """Track_id + file_path list from metadata.db, minus already-checkpointed ids."""
    import sqlite3

    from src.config_loader import Config, resolve_database_path

    db_path = resolve_database_path(Config("config.yaml"))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute("SELECT track_id, file_path FROM tracks").fetchall()
    con.close()
    done = set() if force else read_checkpoint_ids(CKPT)
    todo = [(str(t), p) for (t, p) in rows if str(t) not in done]
    if limit > 0:
        todo = todo[:limit]
    return todo


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="process at most N (smoke test)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-process all, ignoring checkpoint")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.merge_only:
        print(f"SIDECAR: {merge_sidecar_npz(SIDECAR, CKPT, columns={'voice_prob': 'voice_prob'})}")
        return

    import multiprocessing as mp

    todo = _load_todo(args.force, args.limit)
    total = len(todo)
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
    print(f"SIDECAR: {merge_sidecar_npz(SIDECAR, CKPT, columns={'voice_prob': 'voice_prob'})}")


if __name__ == "__main__":
    main()
