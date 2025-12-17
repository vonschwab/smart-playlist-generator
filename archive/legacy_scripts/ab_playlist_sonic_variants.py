#!/usr/bin/env python3
"""
Generate A/B DS playlists for multiple sonic variants and export M3Us + a diff report.

Smoke run examples:
  python scripts/ab_playlist_sonic_variants.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --seed-track-id <id> --tracks 30 --random-seed 1 --variants raw,z_clip --export-m3u-dir E:\\PLAYLISTS --db data/metadata.db
  python scripts/ab_playlist_sonic_variants.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --artist "Radiohead" --tracks 20 --random-seed 2 --variants raw,z --export-m3u-dir diagnostics/ab_m3u --db data/metadata.db
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ensure root on path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from src.similarity.sonic_variant import resolve_sonic_variant
from scripts.batch_eval_sonic_variant import recompute_edges, summarize_edges, _load_metadata, _write_m3u, _edges_with_labels

VARIANT_HELP = {
    "raw": "cosine on original X_sonic",
    "centered": "mean-center features then cosine",
    "z": "per-dimension z-score then cosine",
    "z_clip": "z-score, clip to [-3,3], then cosine",
    "whiten_pca": "z-score then PCA-whiten then cosine",
}


def _select_seed_from_artist(bundle, artist: str) -> Optional[str]:
    artists = getattr(bundle, "track_artists", None)
    if artists is None:
        return None
    artist_norm = artist.strip().lower()
    for tid, art in zip(bundle.track_ids, artists):
        if str(art).strip().lower() == artist_norm:
            return str(tid)
    return None


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(map(str, a))
    sb = set(map(str, b))
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


def _weak_edges(edges: List[Dict[str, Any]], key: str, k: int = 5) -> List[Dict[str, Any]]:
    vals = [e for e in edges if key in e and np.isfinite(e.get(key, float("nan")))]
    return sorted(vals, key=lambda e: e.get(key, float("inf")))[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DS playlists across sonic variants and export M3Us + diff.")
    parser.add_argument("--artifact", required=True, help="Path to DS artifact NPZ")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed-track-id", help="Seed track_id for DS pipeline")
    group.add_argument("--artist", help="Artist name to pick a seed from artifact metadata")
    parser.add_argument("--tracks", type=int, default=30, help="Playlist length")
    parser.add_argument("--random-seed", type=int, default=0, help="Deterministic RNG seed")
    parser.add_argument("--variants", default="raw,z_clip", help="Comma-separated variants to run")
    parser.add_argument("--export-m3u-dir", type=Path, required=True, help="Directory to write M3Us")
    parser.add_argument("--db", type=Path, default=Path("data/metadata.db"), help="metadata.db for labels/paths")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    bundle = load_artifact_bundle(artifact_path)
    seed_id = args.seed_track_id
    if not seed_id and args.artist:
        seed_id = _select_seed_from_artist(bundle, args.artist)
    if not seed_id:
        raise SystemExit("No seed_track_id could be resolved.")
    if seed_id not in bundle.track_id_to_index:
        raise SystemExit(f"Seed {seed_id} not found in artifact.")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    meta = _load_metadata(args.db)
    # enrich meta with artifact artist/title
    artists = []
    titles = []
    if getattr(bundle, "track_artists", None) is not None:
        artists = list(bundle.track_artists)
    if getattr(bundle, "track_titles", None) is not None:
        titles = list(bundle.track_titles)
    for tid, art, title in zip(bundle.track_ids, artists or [], titles or []):
        meta.setdefault(str(tid), {}).update({"artist": art, "title": title})

    results: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        resolved = resolve_sonic_variant(explicit_variant=variant)
        res = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            mode="dynamic",
            length=args.tracks,
            random_seed=args.random_seed,
            sonic_variant=resolved,
        )
        track_ids = res.track_ids
        edges = recompute_edges(artifact_path, track_ids, resolved)
        edges_labeled = _edges_with_labels(edges, bundle, meta)
        stats = summarize_edges(edges)
        results[resolved] = {
            "track_ids": track_ids,
            "edges": edges_labeled,
            "stats": stats,
        }
        # export M3U
        m3u_path = args.export_m3u_dir / f"{seed_id}_sonic-{resolved}.m3u8"
        _write_m3u(track_ids, meta, m3u_path)

    raw_tracks = results.get("raw", {}).get("track_ids", [])
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    diag_dir = Path("diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)
    report_path = diag_dir / f"AB_PLAYLIST_{seed_id}_{ts}.md"
    lines = [
        f"# A/B playlists for seed {seed_id}",
        f"- Artifact: {artifact_path}",
        f"- Random seed: {args.random_seed}",
        "",
        "## Variants",
    ]
    for v in variants:
        lines.append(f"- {v}: {VARIANT_HELP.get(v, '')}")
    lines.append("")
    for variant in variants:
        vres = results.get(variant, {})
        if not vres:
            continue
        tracks = vres["track_ids"]
        lines.append(f"## Variant {variant}")
        if raw_tracks:
            j = _jaccard(raw_tracks, tracks)
            only_v = [t for t in tracks if t not in raw_tracks]
            only_raw = [t for t in raw_tracks if t not in tracks]
            lines.append(f"- Jaccard vs raw: {j:.3f}")
            lines.append(f"- Only in {variant}: {len(only_v)}")
            lines.append(f"- Only in raw: {len(only_raw)}")
        stats = vres.get("stats", {})
        lines.append(
            f"- S_spread={stats.get('S_spread','na')} corr_S_T={stats.get('corr_S_T','na')} min_T={stats.get('T_min','na')} mean_T={stats.get('T_mean','na')}"
        )
        edges = vres.get("edges", [])
        lines.append("- Weakest by T:")
        for e in _weak_edges(edges, "T"):
            lines.append(
                f"  - T={e.get('T'):.3f} | {e.get('prev_artist','')} - {e.get('prev_title','')} -> {e.get('cur_artist','')} - {e.get('cur_title','')}"
            )
        lines.append("- Weakest by S:")
        for e in _weak_edges(edges, "S"):
            lines.append(
                f"  - S={e.get('S'):.3f} | {e.get('prev_artist','')} - {e.get('prev_title','')} -> {e.get('cur_artist','')} - {e.get('cur_title','')}"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote diff report to {report_path}")


if __name__ == "__main__":
    main()
