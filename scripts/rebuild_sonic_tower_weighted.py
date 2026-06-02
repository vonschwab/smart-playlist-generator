"""Surgically rebuild an artifact's sonic matrices as the tower_weighted variant.

Reads the existing npz, recomputes sonic matrices (full + start/mid/end) via
src.features.sonic_rebuild, backs up the original, and writes the rebuilt npz
in place (same stem -> dense-genre sidecar stays valid). No DB / audio access.

Usage:
    python scripts/rebuild_sonic_tower_weighted.py \
        --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.sonic_rebuild import build_tower_weighted_arrays  # noqa: E402


def _weights_from_config(config_path: str) -> Tuple[float, float, float]:
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    tw = cfg["playlists"]["ds_pipeline"]["tower_weights"]
    return (float(tw["rhythm"]), float(tw["timbre"]), float(tw["harmony"]))


def rebuild_artifact(
    artifact: str,
    weights: Tuple[float, float, float],
    *,
    backup: bool = True,
) -> Optional[Path]:
    """Rebuild ``artifact`` in place as tower_weighted. Returns backup path or None."""
    path = Path(artifact)
    with np.load(path, allow_pickle=True) as data:
        out = build_tower_weighted_arrays(data, weights)

    backup_path: Optional[Path] = None
    if backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(path.name + f".bak_{ts}")
        shutil.copy2(path, backup_path)

    tmp = path.with_name(path.stem + ".rebuild.npz")
    try:
        np.savez(tmp, **out)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        default="data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    weights = _weights_from_config(args.config)
    backup = rebuild_artifact(args.artifact, weights, backup=not args.no_backup)
    print(f"Rebuilt {args.artifact} as tower_weighted weights={weights}")
    if backup:
        print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
