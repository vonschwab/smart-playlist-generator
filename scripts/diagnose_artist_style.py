#!/usr/bin/env python3
"""
Diagnostics for style-aware artist playlists.

Usage:
  python scripts/diagnose_artist_style.py --artist "Artist Name" --config config.yaml --cohesion-mode narrow
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import Config
from src.features.artifacts import load_artifact_bundle
from src.playlist.artist_style import (
    ArtistStyleConfig,
    cluster_artist_tracks,
    get_internal_connectors,
    order_clusters,
)
from src.playlist.config import get_min_sonic_similarity
from src.string_utils import normalize_artist_key
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohesion-mode", default="narrow", choices=["narrow", "dynamic"])
    parser.add_argument("--floors", default="0.00,0.05,0.10,0.20", help="Comma-separated bridge floors to sweep (e.g. 0.00,0.05,0.10,0.20)")
    args = parser.parse_args()

    configure_logging(level="INFO", console=True, log_file=None)
    cfg = Config(args.config)
    ds_cfg = cfg.config.get("playlists", {}).get("ds_pipeline", {}) or {}
    art_path = ds_cfg.get("artifact_path") or Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
    bundle = load_artifact_bundle(art_path)

    style_raw = ds_cfg.get("artist_style", {}) or {}
    style_cfg = ArtistStyleConfig(
        enabled=True,
        cluster_k_min=style_raw.get("cluster_k_min", 3),
        cluster_k_max=style_raw.get("cluster_k_max", 6),
        cluster_k_heuristic_enabled=style_raw.get("cluster_k_heuristic_enabled", True),
        piers_per_cluster=style_raw.get("piers_per_cluster", 1),
        per_cluster_candidate_pool_size=style_raw.get("per_cluster_candidate_pool_size", 400),
        pool_balance_mode=style_raw.get("pool_balance_mode", "equal"),
        internal_connector_priority=style_raw.get("internal_connector_priority", True),
        internal_connector_max_per_segment=style_raw.get("internal_connector_max_per_segment", 2),
        bridge_floor_narrow=style_raw.get("bridge_floor", {}).get("narrow", 0.08),
        bridge_floor_dynamic=style_raw.get("bridge_floor", {}).get("dynamic", 0.03),
    )

    clusters, medoids, medoids_by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle,
        artist_name=args.artist,
        cfg=style_cfg,
        random_seed=ds_cfg.get("random_seed", 0),
    )
    ordered_medoids = order_clusters(medoids, X_norm)

    def _label(idx: int) -> str:
        tid = str(bundle.track_ids[idx])
        artist = (
            str(bundle.track_artists[idx])
            if bundle.track_artists is not None
            else str(bundle.artist_keys[idx])
        )
        title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""
        return f"{artist} - {title} [{tid}]"

    logger.info("Clusters: %d (k=%s-%s)", len(clusters), style_cfg.cluster_k_min, style_cfg.cluster_k_max)
    for idx, members in enumerate(clusters):
        logger.info(
            "  Cluster %d: size=%d medoids=%s",
            idx,
            len(members),
            [_label(m) for m in medoids_by_cluster[idx]],
        )
    logger.info("Pier order (medoids): %s", [_label(i) for i in ordered_medoids])
    global_floor = get_min_sonic_similarity(ds_cfg.get("candidate_pool", {}), args.cohesion_mode)
    bridge_floor = style_cfg.bridge_floor_narrow if args.cohesion_mode == "narrow" else style_cfg.bridge_floor_dynamic
    pier_ids = [str(bundle.track_ids[m]) for m in ordered_medoids]
    internal_ids = (
        get_internal_connectors(
            bundle=bundle,
            artist_key=normalize_artist_key(args.artist),
            exclude_indices=medoids,
            global_floor=global_floor,
            pier_indices=medoids,
            X_norm=X_norm,
        )
        if style_cfg.internal_connector_priority
        else []
    )
    # NOTE (Phase 1 Task 8/9): the per-cluster EXTERNAL candidate pool this
    # diagnostic used to report on (`build_balanced_candidate_pool`, keyed by
    # `per_cluster_candidate_pool_size`/`pool_balance_mode`) was deleted along
    # with the rest of the legacy pier-bridge pooling machinery -- Artist mode
    # now scans corridor's own eligible universe (percentile-membership over
    # the ~43k-track library, see build_eligible_universe /
    # resolve_corridor_width_percentile), which has no per-cluster-sized-pool
    # equivalent to inspect here. `allowed_ids` below is therefore piers +
    # internal connectors only; the per-segment pass-count sweep further down
    # is scoped to that reduced set, not the corridor's actual universe.
    allowed_ids = list(dict.fromkeys(pier_ids + list(internal_ids or [])))
    logger.info(
        "Pools: piers=%d internal=%d allowed_total=%d global_floor=%.2f bridge_floor_default=%.2f "
        "(NOTE: no external per-cluster pool under corridor pooling -- see comment above)",
        len(pier_ids),
        len(internal_ids or []),
        len(allowed_ids),
        float(global_floor) if global_floor is not None else float("nan"),
        float(bridge_floor),
    )

    floors = []
    for part in str(args.floors).split(","):
        part = part.strip()
        if not part:
            continue
        floors.append(float(part))
    if not floors:
        floors = [float(bridge_floor)]

    # Per-segment pass counts over allowed pool (external + internal + piers)
    sim_matrix = np.dot(X_norm, X_norm[ordered_medoids].T)
    id_to_idx = {str(tid): i for i, tid in enumerate(bundle.track_ids)}
    allowed_indices = [id_to_idx[tid] for tid in allowed_ids if tid in id_to_idx]
    for i in range(len(ordered_medoids) - 1):
        a_idx = ordered_medoids[i]
        b_idx = ordered_medoids[i + 1]
        sim_a = sim_matrix[:, i]
        sim_b = sim_matrix[:, i + 1]
        counts = {}
        if allowed_indices:
            a_vals = sim_a[allowed_indices]
            b_vals = sim_b[allowed_indices]
            for f in floors:
                counts[f] = int(np.sum(np.logical_and(a_vals >= f, b_vals >= f)))
        logger.info("  Segment %d: %s -> %s", i, _label(a_idx), _label(b_idx))
        for f in floors:
            logger.info("    pass[min(simA,simB)>=%0.2f]=%d", f, counts.get(f, 0))


if __name__ == "__main__":
    main()
