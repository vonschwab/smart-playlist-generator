#!/usr/bin/env python3
"""
Diagnose sonic similarity floor behavior for DS pipeline.

Example:
  python scripts/diagnose_sonic_floor.py --seed "Artist - Title" --ds-mode dynamic --floor 0.0 --top 15 --show-rejected
  python scripts/diagnose_sonic_floor.py --seed-track-id <track_id> --ds-mode narrow --show-borderline
"""
import argparse
import sqlite3
import logging
from typing import List, Tuple

import numpy as np

from src.config_loader import Config
from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_variant import resolve_sonic_variant, compute_sonic_variant_matrix
from src.logging_utils import configure_logging
from src.playlist.config import get_min_sonic_similarity

logger = logging.getLogger(__name__)


def _resolve_seed_track_id(db_path: str, seed: str) -> str:
    if not seed:
        raise ValueError("Seed string is empty")
    if " - " not in seed:
        raise ValueError("Seed must be in the format 'Artist - Title'")
    artist, title = seed.split(" - ", 1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT track_id FROM tracks WHERE artist = ? AND title = ? LIMIT 1",
        (artist.strip(), title.strip()),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Seed track not found in DB: {seed}")
    return str(row["track_id"])


def summarize_sims(sims: np.ndarray) -> str:
    return "min={:.3f} p05={:.3f} median={:.3f} p95={:.3f} max={:.3f}".format(
        float(np.min(sims)),
        float(np.percentile(sims, 5)),
        float(np.percentile(sims, 50)),
        float(np.percentile(sims, 95)),
        float(np.max(sims)),
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnose sonic similarity floor for DS pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", help='Seed as "Artist - Title"')
    parser.add_argument("--seed-track-id", help="Seed track_id (overrides --seed)")
    parser.add_argument("--ds-mode", default="narrow", choices=["narrow", "dynamic", "discover", "sonic_only"])
    parser.add_argument("--floor", type=float, help="Override sonic floor (default from config)")
    parser.add_argument("--top", type=int, default=20, help="Top N to display by sonic similarity")
    parser.add_argument("--show-rejected", action="store_true", help="List candidates rejected by floor")
    parser.add_argument("--show-borderline", action="store_true", help="List candidates near the floor (+/-0.05)")
    args = parser.parse_args()

    configure_logging(level="INFO")

    cfg = Config(args.config)
    ds_cfg = cfg.config.get("playlists", {}).get("ds_pipeline", {}) or {}
    candidate_cfg = ds_cfg.get("candidate_pool", {}) or {}
    artifact_path = ds_cfg.get("artifact_path") or cfg.config.get("artifacts", {}).get("path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz")

    # Resolve floor from config or override
    floor_key = f"min_sonic_similarity_{args.ds_mode}"
    floor = args.floor
    if floor is None:
        floor = get_min_sonic_similarity(candidate_cfg, args.ds_mode)

    seed_track_id = args.seed_track_id or _resolve_seed_track_id(cfg.library_database_path, args.seed or "")

    bundle = load_artifact_bundle(artifact_path)
    seed_idx = bundle.track_id_to_index.get(str(seed_track_id))
    if seed_idx is None:
        raise ValueError(f"Seed track_id {seed_track_id} not found in artifact")

    X_sonic = bundle.X_sonic
    if getattr(bundle, "sonic_variant", None) != "raw":
        X_sonic, _ = compute_sonic_variant_matrix(bundle.X_sonic, resolve_sonic_variant(explicit_variant=None, config_variant=None), l2=False)
    sonic_norm = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
    seed_vec = sonic_norm[seed_idx]
    sims = np.dot(sonic_norm, seed_vec)
    sims[seed_idx] = -1.0

    logger.info(f"Mode={args.ds_mode} floor={floor if floor is not None else float('nan'):.2f} candidates={len(sims)-1}")
    logger.info("Distribution: %s", summarize_sims(sims[1:]))

    order = np.argsort(sims)[::-1]
    logger.info(f"\nTop {args.top} by sonic similarity:")
    logger.info("rank\ttrack_id\tsim\tartist - title")
    shown = 0
    for idx in order:
        if idx == seed_idx:
            continue
        artist = bundle.track_artists[idx] if getattr(bundle, "track_artists", None) is not None else ""
        title = bundle.track_titles[idx] if getattr(bundle, "track_titles", None) is not None else ""
        logger.info(f"{shown+1}\t{bundle.track_ids[idx]}\t{sims[idx]:.3f}\t{artist} - {title}")
        shown += 1
        if shown >= args.top:
            break

    effective_floor = float("-inf") if floor is None else float(floor)

    if args.show_rejected:
        rejected = [(bundle.track_ids[i], sims[i]) for i in range(len(sims)) if i != seed_idx and sims[i] < effective_floor]
        logger.info(f"\nRejected by sonic floor (<{effective_floor:.2f}): {len(rejected)}")
        for tid, val in sorted(rejected, key=lambda t: t[1])[:20]:
            logger.info(f"{tid}\t{val:.3f}")

    if args.show_borderline:
        delta = 0.05
        borderline = [(bundle.track_ids[i], sims[i]) for i in range(len(sims)) if i != seed_idx and abs(sims[i] - effective_floor) <= delta]
        logger.info(f"\nBorderline within +/-{delta} of floor {effective_floor:.2f}: {len(borderline)}")
        for tid, val in sorted(borderline, key=lambda t: abs(t[1] - effective_floor))[:20]:
            logger.info(f"{tid}\t{val:.3f}")


if __name__ == "__main__":
    main()
