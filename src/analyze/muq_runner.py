"""MuQ-MuLan sonic embedding extraction for the analyze `muq` stage.

Incremental + resumable: embeds only tracks lacking a MuQ vector into muq_sidecar.npz
(single npz, atomic save). Backs up the existing sidecar before the first overwrite —
it is a ~16-29h CPU artifact, treated like the irreplaceable MERT data. Productionized
from scripts/research/embed_muq_full.py.

A bad file's failure is recorded in the returned `fails` list, never fatal — the
calling stage logs the summary.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_NAME = "OpenMuQ/MuQ-MuLan-large"
SAVE_EVERY = 500


def sidecar_ids(sidecar_path) -> set[str]:
    p = Path(sidecar_path)
    if not p.exists():
        return set()
    with np.load(str(p), allow_pickle=True) as z:
        return {str(t) for t in z["track_ids"]}


def pending_muq(sidecar_path, universe_ids: Sequence[str]) -> Tuple[List[str], int]:
    have = sidecar_ids(sidecar_path)
    return [t for t in universe_ids if t not in have], len(have)


def _load_existing(sidecar_path) -> Dict[str, np.ndarray]:
    p = Path(sidecar_path)
    if not p.exists():
        return {}
    out: Dict[str, np.ndarray] = {}
    with np.load(str(p), allow_pickle=True) as z:
        for tid, emb in zip(z["track_ids"], z["embeddings"]):
            out[str(tid)] = np.asarray(emb, dtype=np.float32)
    return out


def _atomic_save(sidecar_path, done: Dict[str, np.ndarray]) -> None:
    if not done:
        return
    p = Path(sidecar_path)
    tids = list(done.keys())
    embs = np.stack([done[t] for t in tids]).astype(np.float32)
    tmp = p.with_name(p.stem + ".tmp.npz")   # must end in .npz (savez appends otherwise)
    np.savez(str(tmp), track_ids=np.array(tids, dtype=object), embeddings=embs, model=MODEL_NAME)
    tmp.replace(p)


def _backup(sidecar_path, stamp: str) -> Optional[Path]:
    p = Path(sidecar_path)
    if not p.exists():
        return None
    bak = p.with_name(f"{p.stem}.bak_{stamp}.npz")
    shutil.copy2(str(p), str(bak))
    return bak


def build_muq_embedder(device: str = "cpu", torch_threads: int = 0) -> Callable[[str], np.ndarray]:
    """Load MuQ-MuLan once; return embed(path) -> unit-norm float32 vector (middle 10s)."""
    import librosa
    import torch
    from muq import MuQMuLan
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    model = MuQMuLan.from_pretrained(MODEL_NAME).to(device).eval()

    def embed(path: str) -> np.ndarray:
        d = librosa.get_duration(path=path)
        y, _ = librosa.load(path, sr=24000, mono=True, offset=max(0.0, d * 0.5 - 5), duration=10.0)
        with torch.no_grad():
            wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            v = model(wavs=wav)[0].detach().cpu().numpy()
        return (v / np.linalg.norm(v)).astype(np.float32)

    return embed


def run_muq_extraction(
    items: Sequence[Tuple[str, Optional[str]]],
    embed_fn: Callable[[str], np.ndarray],
    sidecar_path,
    *,
    backup_stamp: Optional[str] = None,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    """Embed each (track_id, path); append into the sidecar (atomic, resumable). Backs up
    the existing sidecar once (when backup_stamp given) before the first write. A bad file's
    failure is recorded in the returned `fails` list, never fatal. Returns {ok, failed, fails}."""
    done = _load_existing(sidecar_path)
    if backup_stamp is not None:
        _backup(sidecar_path, backup_stamp)
    ok = 0
    fails: List[Tuple[str, str]] = []
    for k, (tid, path) in enumerate(items, 1):
        if not path:
            fails.append((tid, "no_path"))
            continue
        try:
            done[tid] = np.asarray(embed_fn(path), dtype=np.float32)
            ok += 1
        except Exception as exc:  # one bad file must not kill the scan
            fails.append((tid, type(exc).__name__))
        if k % save_every == 0:
            _atomic_save(sidecar_path, done)
    _atomic_save(sidecar_path, done)
    return {"ok": ok, "failed": len(fails), "fails": fails}
