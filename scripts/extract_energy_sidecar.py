"""Extract Essentia energy descriptors (arousal distribution + danceability) for
the whole library into a sidecar npz aligned to the beat3tower artifact.

RUNS UNDER WSL ONLY (Essentia lives in the WSL venv at /opt/ess; the Windows
runtime never imports this). Invoke with /opt/ess/bin/python.

- Scope = the artifact's track_ids (the exact generation set); paths from
  metadata.db (READ-ONLY). Energy is a PACE-axis sidecar, never a metadata.db
  write and never folded into the sonic-similarity blend.
- Per track: emoMusic arousal (p10/p50/p90 -- distribution, the mean masks
  dynamics), valence (p50), danceability P. msd-musicnn embeddings @16kHz.
- Resumable: append-only JSONL checkpoint; re-running skips done track_ids.
- Parallel: spawn pool (TF-safe -- essentia imported only inside workers,
  AFTER thread env is pinned to 1, never at module top).
- Writes ONLY to <artifact>/energy/; backs up an existing sidecar timestamped
  before overwrite. Touches nothing else in the artifact dir.

Usage (from WSL, repo on /mnt/c):
    /opt/ess/bin/python scripts/extract_energy_sidecar.py --workers 14
    /opt/ess/bin/python scripts/extract_energy_sidecar.py --merge-only
    /opt/ess/bin/python scripts/extract_energy_sidecar.py --limit 50   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "data", "artifacts", "beat3tower_32k")
NPZ = os.path.join(ART, "data_matrices_step1.npz")
DB = os.path.join(ROOT, "data", "metadata.db")
OUTDIR = os.path.join(ART, "energy")
CKPT = os.path.join(OUTDIR, "checkpoint.jsonl")
SIDECAR = os.path.join(OUTDIR, "energy_sidecar.npz")
MODELS = "/opt/ess/models"

EMB_PB = f"{MODELS}/msd-musicnn-1.pb"
AV_PB = f"{MODELS}/emomusic-msd-musicnn-2.pb"
DANCE_PB = f"{MODELS}/danceability-msd-musicnn-1.pb"

_emb = _av = _dc = None


def _win_to_wsl(p: str) -> str:
    p = p.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def _artifact_track_ids() -> list[str]:
    z = np.load(NPZ, allow_pickle=True)
    return [str(t) for t in z["track_ids"]]


def _paths_for(track_ids: list[str]) -> dict[str, str]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
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
                    out[str(r["track_id"])] = _win_to_wsl(r["file_path"])
    finally:
        con.close()
    return out


def _done_ids() -> set[str]:
    done: set[str] = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["track_id"])
                except Exception:
                    continue
    return done


def _init() -> None:
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings

    warnings.filterwarnings("ignore")
    import essentia

    essentia.log.warningActive = False
    import essentia.standard as es

    global _emb, _av, _dc
    _emb = es.TensorflowPredictMusiCNN(graphFilename=EMB_PB, output="model/dense/BiasAdd")
    _av = es.TensorflowPredict2D(graphFilename=AV_PB, output="model/Identity")
    _dc = es.TensorflowPredict2D(graphFilename=DANCE_PB, output="model/Softmax")


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
        av = _av(emb)
        dc = _dc(emb)
        aro = av[:, 1]
        return {
            "track_id": tid,
            "valence": round(float(np.mean(av[:, 0])), 4),
            "arousal_p10": round(float(np.percentile(aro, 10)), 4),
            "arousal_p50": round(float(np.percentile(aro, 50)), 4),
            "arousal_p90": round(float(np.percentile(aro, 90)), 4),
            "danceability": round(float(np.mean(dc[:, 0])), 4),
            "frames": int(emb.shape[0]),
        }
    except Exception as ex:  # noqa: BLE001
        return {"track_id": tid, "error": repr(ex)[:200]}


def _merge() -> None:
    tids = _artifact_track_ids()
    rec: dict[str, dict] = {}
    with open(CKPT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec[d["track_id"]] = d
    n = len(tids)
    val = np.full(n, np.nan, np.float32)
    p10 = np.full(n, np.nan, np.float32)
    p50 = np.full(n, np.nan, np.float32)
    p90 = np.full(n, np.nan, np.float32)
    dance = np.full(n, np.nan, np.float32)
    frames = np.zeros(n, np.int32)
    ok = miss = err = 0
    for i, tid in enumerate(tids):
        d = rec.get(tid)
        if not d or "arousal_p50" not in d:
            if d and d.get("error"):
                err += 1
            else:
                miss += 1
            continue
        val[i] = d["valence"]
        p10[i] = d["arousal_p10"]
        p50[i] = d["arousal_p50"]
        p90[i] = d["arousal_p90"]
        dance[i] = d["danceability"]
        frames[i] = d["frames"]
        ok += 1
    if os.path.exists(SIDECAR):
        bak = SIDECAR + "." + time.strftime("%Y%m%d_%H%M%S") + ".bak"
        os.rename(SIDECAR, bak)
        print(f"backed up existing sidecar -> {bak}")
    np.savez_compressed(
        SIDECAR,
        track_ids=np.array(tids, dtype=object),
        valence=val,
        arousal_p10=p10,
        arousal_p50=p50,
        arousal_p90=p90,
        danceability=dance,
        frames=frames,
        model=np.array("emomusic-msd-musicnn-2 + danceability-msd-musicnn-1", dtype=object),
    )
    print(f"wrote {SIDECAR}: ok={ok} missing={miss} error={err} total={n}")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="process at most N (smoke test)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-process all tracks, ignoring checkpoint")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    if args.merge_only:
        _merge()
        return

    tids = _artifact_track_ids()
    paths = _paths_for(tids)
    done = _done_ids()
    todo = [(t, paths.get(t)) for t in tids if (args.force or t not in done)]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"artifact={len(tids)} done={len(done)} todo={len(todo)} workers={args.workers}",
        flush=True,
    )
    if not todo:
        print("nothing to do; merging.")
        _merge()
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t0 = time.time()
    n = 0
    with open(CKPT, "a", encoding="utf-8") as f:
        with ctx.Pool(args.workers, initializer=_init) as pool:
            for d in pool.imap_unordered(_process, todo, chunksize=4):
                f.write(json.dumps(d) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0:
                    rate = n / (time.time() - t0)
                    eta_h = (len(todo) - n) / rate / 3600 if rate else float("inf")
                    print(f"  {n}/{len(todo)}  {rate:.2f} trk/s  ETA {eta_h:.1f}h", flush=True)
    print(f"scan pass done: {n} tracks in {(time.time()-t0)/3600:.2f}h", flush=True)
    _merge()


if __name__ == "__main__":
    main()
