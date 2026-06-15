"""Extract MERT-v1-95M embeddings for the library into resumable shards + a merged sidecar npz.

Phase 1 of the MERT sonic embedding plan
(docs/superpowers/plans/2026-06-11-mert-sonic-embedding.md). Per track, three
24 s clips (start / mid / end) are embedded as the mean over all 13 hidden-state
layers, mean over time -> 768-d float32 (the prototype-validated recipe).

SAFETY: audio is read-only (librosa.load only; never written/moved). metadata.db
is read-only (URI mode=ro, file paths only). The production artifact is read
read-only for track ordering. Output is ONLY new files: shard npz files +
manifest.json under data/artifacts/beat3tower_32k/mert_shards/, and the merged
sidecar data/artifacts/beat3tower_32k/mert_sidecar.npz.

Resumable: re-run to continue; track_ids already in the manifest (done or
failed) are skipped under --resume (the default). Per-track errors are logged,
recorded in the manifest with a reason, and never crash the run.

Usage:
    python scripts/extract_mert_sidecar.py [--limit N] [--track-ids FILE]
        [--device cpu|cuda] [--shard-size 500] [--resume/--no-resume]
    python scripts/extract_mert_sidecar.py --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
DB = ROOT / "data/metadata.db"
SHARD_DIR = ROOT / "data/artifacts/beat3tower_32k/mert_shards"
SIDECAR = ROOT / "data/artifacts/beat3tower_32k/mert_sidecar.npz"

MODEL_NAME = "m-a-p/MERT-v1-95M"
# Pinned HF revision (trust_remote_code executes repo files; the pin is mandatory).
# Discovered from the local HF cache snapshot created by the validated prototype run.
DEFAULT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"

SR = 24000
CLIP_S = 24.0
EMB_DIM = 768
# Below this duration there isn't room for three meaningfully distinct clips;
# use a single (centered) window for all three slots.
MIN_THREE_WINDOW_S = 30.0

Embedder = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- clips


def clip_windows(duration_s: float, clip_s: float = CLIP_S) -> list[tuple[float, float]]:
    """Return exactly 3 (offset, duration) windows: start, mid, end.

    - normal track: start at 0, mid centered at duration/2, end = final clip_s.
    - shorter than MIN_THREE_WINDOW_S: one centered window replicated 3x.
    - shorter than clip_s: the whole track, replicated 3x.
    """
    if duration_s <= 0:
        raise ValueError(f"non-positive duration: {duration_s}")
    if duration_s < clip_s:
        return [(0.0, duration_s)] * 3
    if duration_s < MIN_THREE_WINDOW_S:
        off = (duration_s - clip_s) / 2.0
        return [(off, clip_s)] * 3
    start = (0.0, clip_s)
    mid = (duration_s / 2.0 - clip_s / 2.0, clip_s)
    end = (duration_s - clip_s, clip_s)
    return [start, mid, end]


# --------------------------------------------------------------------------- shards


class ShardStore:
    """Append-only shard writer with a JSON manifest for resume.

    Embeddings are buffered and written to shard_NNNN.npz files (track_ids,
    emb_start, emb_mid, emb_end). The manifest tracks done ids, failed ids with
    reasons, and the model identity. Ids only enter `done` once their shard hits
    disk, so an interrupted run re-extracts at most one buffered shard.
    """

    def __init__(
        self,
        shard_dir: Path,
        *,
        model_name: str = MODEL_NAME,
        model_revision: str = DEFAULT_REVISION,
        emb_dim: int = EMB_DIM,
        shard_size: int = 500,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.manifest_path = self.shard_dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest: dict = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for key, want in (("model_name", model_name), ("model_revision", model_revision),
                              ("emb_dim", emb_dim)):
                have = self.manifest.get(key)
                if have != want:
                    raise ValueError(
                        f"manifest {key}={have!r} does not match requested {want!r} "
                        f"({self.manifest_path}); refusing to mix embeddings from different models"
                    )
        else:
            self.manifest = {
                "model_name": model_name,
                "model_revision": model_revision,
                "emb_dim": emb_dim,
                "done": [],
                "failed": {},
            }
            self._save_manifest()
        self._buf: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    def skip_ids(self) -> set[str]:
        """Ids already in the manifest (done or failed) — skipped under --resume."""
        return set(self.manifest["done"]) | set(self.manifest["failed"])

    def add(self, track_id: str, emb_start: np.ndarray, emb_mid: np.ndarray, emb_end: np.ndarray) -> None:
        self._buf.append((
            track_id,
            np.asarray(emb_start, np.float32),
            np.asarray(emb_mid, np.float32),
            np.asarray(emb_end, np.float32),
        ))
        if len(self._buf) >= self.shard_size:
            self.flush()

    def record_failure(self, track_id: str, reason: str) -> None:
        self.manifest["failed"][track_id] = reason
        self._save_manifest()

    def flush(self) -> None:
        """Write buffered embeddings as the next shard and mark them done."""
        if not self._buf:
            self._save_manifest()
            return
        existing = sorted(self.shard_dir.glob("shard_*.npz"))
        next_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
        path = self.shard_dir / f"shard_{next_idx:04d}.npz"
        tids = np.array([t for t, *_ in self._buf], dtype=object)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            track_ids=tids,
            emb_start=np.stack([s for _, s, _, _ in self._buf]),
            emb_mid=np.stack([m for _, _, m, _ in self._buf]),
            emb_end=np.stack([e for _, _, _, e in self._buf]),
        )
        tmp.replace(path)  # atomic; survives interruption
        self.manifest["done"].extend(str(t) for t in tids)
        self._buf = []
        self._save_manifest()

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)


def merge_shards(shard_dir: Path, out_path: Path) -> dict:
    """Merge all shards into one sidecar npz; each track_id exactly once (newest shard wins)."""
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.json"
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not manifest_path.exists() or not shards:
        raise FileNotFoundError(f"no shards/manifest to merge in {shard_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    order: list[str] = []
    rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for shard in shards:
        z = np.load(shard, allow_pickle=True)
        for i, tid in enumerate(z["track_ids"]):
            tid = str(tid)
            if tid not in rows:
                order.append(tid)
            rows[tid] = (z["emb_start"][i], z["emb_mid"][i], z["emb_end"][i])

    # The sidecar is irreplaceable (regenerable from shards, but only while the
    # shards survive). Back up any existing sidecar with a timestamp before the
    # atomic replace below — same backup discipline the publish stage applies to
    # metadata.db. The atomic replace prevents corruption; this guards against a
    # bad merge silently clobbering a good sidecar.
    out_path = Path(out_path)
    if out_path.exists():
        bak = out_path.with_name(f"{out_path.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(out_path, bak)

    emb_dim = int(manifest["emb_dim"])
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(
        tmp,
        track_ids=np.array(order, dtype=object),
        emb_start=(np.stack([rows[t][0] for t in order]) if order
                   else np.zeros((0, emb_dim), np.float32)).astype(np.float32),
        emb_mid=(np.stack([rows[t][1] for t in order]) if order
                 else np.zeros((0, emb_dim), np.float32)).astype(np.float32),
        emb_end=(np.stack([rows[t][2] for t in order]) if order
                 else np.zeros((0, emb_dim), np.float32)).astype(np.float32),
        model_name=np.array(manifest["model_name"]),
        model_revision=np.array(manifest["model_revision"]),
    )
    tmp.replace(out_path)
    return {"tracks": len(order), "shards": len(shards), "failed": len(manifest["failed"])}


# --------------------------------------------------------------------------- audio


def probe_duration(fp: str) -> float:
    """Track duration in seconds via soundfile header read (audio is read-only)."""
    import soundfile as sf

    info = sf.info(fp)
    return float(info.frames) / float(info.samplerate)


def load_window(fp: str, offset: float, duration: float) -> np.ndarray:
    """Decode one mono 24 kHz window (audio is read-only)."""
    import librosa

    y, _ = librosa.load(fp, sr=SR, mono=True, offset=offset, duration=duration)
    return y


# --------------------------------------------------------------------------- embedder


def build_real_embedder(device: str = "cpu", revision: str = DEFAULT_REVISION) -> Embedder:
    """Load MERT-v1-95M pinned to `revision`; return waveform -> 768-d float32.

    Recipe (validated by the Phase-0 prototype): mean over all 13 hidden-state
    layers, mean over time.
    """
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=revision)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=revision)
    model.eval()
    model.to(device)

    def embed(y: np.ndarray) -> np.ndarray:
        inp = proc(y, sampling_rate=SR, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = torch.stack(out.hidden_states)          # [13, 1, T, 768]
        emb = h.mean(0).mean(1).squeeze()           # [768]
        return emb.cpu().numpy().astype(np.float32)

    return embed


def build_layer_embedder(device: str = "cpu", revision: str = DEFAULT_REVISION) -> Callable[[np.ndarray], np.ndarray]:
    """Load MERT-v1-95M; return waveform -> (13, 768) float32 (per-layer, mean over time).

    For layer-selection calibration only — not used in the main shard pipeline.
    """
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=revision)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=revision)
    model.eval()
    model.to(device)

    def embed(y: np.ndarray) -> np.ndarray:
        inp = proc(y, sampling_rate=SR, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = torch.stack(out.hidden_states)   # [13, 1, T, 768]
        per_layer = h.mean(2).squeeze(1)     # [13, 768] — mean over time, keep layers
        return per_layer.cpu().numpy().astype(np.float32)

    return embed


def run_layer_extraction(
    items: list[tuple[str, str | None]],
    embedder: Callable[[np.ndarray], np.ndarray],
    out_path: Path,
    *,
    prober: Callable[[str], float] = probe_duration,
    loader: Callable[[str, float, float], np.ndarray] = load_window,
    clip_s: float = CLIP_S,
) -> None:
    """Extract per-layer start/mid/end embeddings for each track; write one NPZ.

    No sharding, no manifest — intended for small targeted runs (calibration seeds).
    Output keys:
        track_ids:        (N,)    object
        emb_start_layers: (N, 13, 768) float32
        emb_mid_layers:   (N, 13, 768) float32
        emb_end_layers:   (N, 13, 768) float32
    """
    n_layers = 13
    records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    t0 = time.time()
    for k, (tid, fp) in enumerate(items, 1):
        try:
            if not fp or not os.path.exists(fp):
                raise FileNotFoundError(f"missing file: {fp!r}")
            duration_s = prober(fp)
            windows = clip_windows(duration_s, clip_s)
            embs: dict[tuple[float, float], np.ndarray] = {}
            for win in windows:
                if win not in embs:
                    y = loader(fp, *win)
                    if y.size == 0:
                        raise ValueError(f"empty window {win}")
                    embs[win] = embedder(y)   # (13, 768)
            layer_start, layer_mid, layer_end = (embs[w] for w in windows)
            records.append((tid, layer_start, layer_mid, layer_end))
            rate = (time.time() - t0) / k
            print(f"  {k}/{len(items)} ok  {rate:.1f}s/track  {tid[:16]}", flush=True)
        except Exception as e:
            print(f"  WARN {tid}: {type(e).__name__}: {e}", flush=True)

    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    n = len(records)
    np.savez(
        tmp,
        track_ids=np.array([r[0] for r in records], dtype=object),
        emb_start_layers=(np.stack([r[1] for r in records]) if records
                          else np.zeros((0, n_layers, EMB_DIM), np.float32)),
        emb_mid_layers=(np.stack([r[2] for r in records]) if records
                        else np.zeros((0, n_layers, EMB_DIM), np.float32)),
        emb_end_layers=(np.stack([r[3] for r in records]) if records
                        else np.zeros((0, n_layers, EMB_DIM), np.float32)),
    )
    tmp.replace(out_path)
    print(f"Layer sidecar: {n} tracks -> {out_path}", flush=True)


# --------------------------------------------------------------------------- extraction


def run_extraction(
    items: list[tuple[str, str | None]],
    embedder: Embedder,
    store: ShardStore,
    *,
    prober: Callable[[str], float] = probe_duration,
    loader: Callable[[str, float, float], np.ndarray] = load_window,
    clip_s: float = CLIP_S,
    log_every: int = 10,
    cancellation_check: Callable[[], None] | None = None,
) -> dict:
    """Embed start/mid/end clips for each (track_id, file_path); never crash on one track.

    ``cancellation_check`` (if given) is polled at the top of each track — a safe
    boundary: shards are append-only and atomically flushed, the manifest marks a
    track done only after flush, and nothing here writes metadata.db. If it
    raises, the ``finally`` flushes the already-completed (buffered) embeddings
    before the exception propagates, so a cancelled run leaves no partial data and
    resumes cleanly from the manifest.
    """
    n_ok = n_fail = 0
    t0 = time.time()
    try:
        for k, (tid, fp) in enumerate(items, 1):
            if cancellation_check is not None:
                # Between-track boundary only — never mid-write. Let whatever it
                # raises propagate; the finally below persists completed work.
                cancellation_check()
            try:
                if not fp or not os.path.exists(fp):
                    raise FileNotFoundError(f"missing file: {fp!r}")
                duration_s = prober(fp)
                windows = clip_windows(duration_s, clip_s)
                embs: dict[tuple[float, float], np.ndarray] = {}
                for win in windows:
                    if win not in embs:  # short tracks replicate one window — embed once
                        y = loader(fp, *win)
                        if y.size == 0:
                            raise ValueError(f"empty window {win}")
                        embs[win] = np.asarray(embedder(y), np.float32)
                store.add(tid, *(embs[w] for w in windows))
                n_ok += 1
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                print(f"  WARN {tid}: {reason}", flush=True)
                store.record_failure(tid, reason)
                n_fail += 1
            if k % log_every == 0:
                rate = (time.time() - t0) / k
                eta_h = (len(items) - k) * rate / 3600
                print(f"  {k}/{len(items)}  ok={n_ok} fail={n_fail}  {rate:.1f}s/track  ETA {eta_h:.1f}h",
                      flush=True)
    finally:
        store.flush()
    return {"ok": n_ok, "failed": n_fail}


# --------------------------------------------------------------------------- CLI


def load_artifact_track_ids(artifact_path: Path = ARTIFACT) -> list[str]:
    """Track ordering comes from the production artifact (read-only)."""
    z = np.load(artifact_path, allow_pickle=True)
    return [str(t) for t in z["track_ids"]]


def load_paths(db_path: Path = DB) -> dict[str, str]:
    """track_id -> file_path from metadata.db (read-only, URI mode=ro)."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT track_id, file_path FROM tracks").fetchall()
    finally:
        con.close()
    return {str(t): p for t, p in rows if p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="extract at most N pending tracks (smoke test)")
    ap.add_argument("--track-ids", type=Path, default=None,
                    help="newline-delimited track_id file; restrict extraction to these ids")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="pinned HF model revision hash")
    ap.add_argument("--shard-dir", type=Path, default=None,
                    help="override shard directory (default: data/artifacts/beat3tower_32k/mert_shards). "
                         "Use with --track-ids to run parallel instances writing to separate dirs, "
                         "then merge with --merge-only --shard-dir dir1 --extra-shard-dirs dir2 dir3")
    ap.add_argument("--extra-shard-dirs", type=Path, nargs="*", default=[],
                    help="additional shard directories to include when running --merge-only")
    ap.add_argument("--merge-only", action="store_true",
                    help=f"skip extraction; merge shards into {SIDECAR.name}")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip track_ids already in the manifest (done or failed)")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="override PyTorch CPU thread count (0 = use PyTorch default, currently 12)")
    ap.add_argument("--layers-out", type=Path, default=None,
                    help="extract per-layer embeddings (13×768) to this NPZ path "
                         "(for layer-selection calibration; runs instead of normal sharding)")
    args = ap.parse_args()

    shard_dir = args.shard_dir if args.shard_dir is not None else SHARD_DIR

    if args.torch_threads > 0:
        import torch
        torch.set_num_threads(args.torch_threads)
        print(f"PyTorch CPU threads: {args.torch_threads}", flush=True)

    if args.merge_only:
        all_dirs = [shard_dir] + list(args.extra_shard_dirs)
        # Merge all dirs into a single combined shard dir first if multiple given
        if len(all_dirs) > 1:
            print(f"Merging {len(all_dirs)} shard dirs: {[str(d) for d in all_dirs]}", flush=True)
            import shutil
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp(prefix="mert_merge_"))
            try:
                shard_idx = 0
                for src_dir in all_dirs:
                    for shard_file in sorted(src_dir.glob("shard_*.npz")):
                        shutil.copy2(shard_file, tmp_dir / f"shard_{shard_idx:04d}.npz")
                        shard_idx += 1
                summary = merge_shards(tmp_dir, SIDECAR)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            summary = merge_shards(shard_dir, SIDECAR)
        print(f"Merged {summary['tracks']} tracks from {summary['shards']} shard(s) -> {SIDECAR} "
              f"({summary['failed']} failed ids excluded)", flush=True)
        return

    track_ids = load_artifact_track_ids()
    paths = load_paths()
    store = ShardStore(shard_dir, model_name=MODEL_NAME, model_revision=args.revision,
                       emb_dim=EMB_DIM, shard_size=args.shard_size)

    if args.track_ids:
        wanted = {ln.strip() for ln in args.track_ids.read_text(encoding="utf-8").splitlines() if ln.strip()}
        unknown = wanted - set(track_ids)
        if unknown:
            print(f"WARN: {len(unknown)} requested id(s) not in the artifact; skipping them", flush=True)
        track_ids = [t for t in track_ids if t in wanted]

    skip = store.skip_ids() if args.resume else set()
    todo: list[tuple[str, str | None]] = [(t, paths.get(t)) for t in track_ids if t not in skip]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Artifact tracks: {len(track_ids)}. Already in manifest: {len(skip)}. "
          f"To extract: {len(todo)}. Device: {args.device}. Shards: {shard_dir}. "
          f"Model: {MODEL_NAME}@{args.revision}",
          flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    if args.layers_out:
        layer_embedder = build_layer_embedder(args.device, args.revision)
        run_layer_extraction(todo, layer_embedder, args.layers_out)
    else:
        embedder = build_real_embedder(args.device, args.revision)
        t0 = time.time()
        summary = run_extraction(todo, embedder, store, log_every=5)
        dt = time.time() - t0
        print(f"Done in {dt:.0f}s ({dt / max(1, len(todo)):.1f}s/track): "
              f"{summary['ok']} ok, {summary['failed']} failed. Shards: {SHARD_DIR}", flush=True)


if __name__ == "__main__":
    main()

