"""Pure, side-effect-free scaffold shared by Essentia sidecar extractors.

Intentionally does NOT import essentia — safe to import under plain pytest.
Mirrors (does not modify) the inline helpers in scripts/extract_energy_sidecar.py;
de-duplicating the energy script is deferred housekeeping (isolation of the live
pace axis takes priority).
"""
from __future__ import annotations

import json
import os
import time
from typing import IO, Iterable

import numpy as np


def win_to_wsl_path(path: str) -> str:
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p


def read_checkpoint_ids(ckpt_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["track_id"])
            except (ValueError, KeyError):
                continue
    return done


def append_checkpoint(fh: IO[str], record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def _iter_records(ckpt_path: str) -> Iterable[dict]:
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def merge_sidecar_npz(sidecar_path: str, ckpt_path: str, columns: dict[str, str]) -> str:
    """Assemble aligned float32 columns from the checkpoint JSONL and write the sidecar.

    columns: {output_array_name: checkpoint_record_key}. Records lacking a key
    (missing/errored tracks) get np.nan so the track_id row is still present.
    """
    tids: list[str] = []
    cols: dict[str, list[float]] = {name: [] for name in columns}
    seen: set[str] = set()
    for rec in _iter_records(ckpt_path):
        tid = rec.get("track_id")
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        tids.append(str(tid))
        for name, key in columns.items():
            val = rec.get(key)
            cols[name].append(float(val) if isinstance(val, (int, float)) else float("nan"))

    if os.path.exists(sidecar_path):
        bak = sidecar_path + "." + time.strftime("%Y%m%d_%H%M%S") + ".bak"
        os.rename(sidecar_path, bak)
        print(f"backed up existing sidecar -> {bak}")

    arrays = {name: np.asarray(vals, dtype=np.float32) for name, vals in cols.items()}
    np.savez_compressed(
        sidecar_path,
        track_ids=np.array(tids, dtype=object),
        **arrays,
    )
    return sidecar_path
