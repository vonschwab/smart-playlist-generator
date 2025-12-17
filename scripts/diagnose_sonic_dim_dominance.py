#!/usr/bin/env python3
"""
Diagnose per-dimension dominance in X_sonic.

Outputs JSON + log with per-dim variance/std, contribution to cosine, and top tracks by abs value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ensure root
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_schema import dim_label, dim_labels
from src.similarity.sonic_variant import compute_sonic_variant_norm


def top_tracks_for_dim(values: np.ndarray, track_ids: np.ndarray, track_artists: Any, track_titles: Any, k: int = 5) -> List[Dict[str, Any]]:
    idx = np.argsort(-np.abs(values))[:k]
    out = []
    for i in idx:
        out.append(
            {
                "track_id": str(track_ids[i]),
                "value": float(values[i]),
                "artist": str(track_artists[i]) if track_artists is not None else "",
                "title": str(track_titles[i]) if track_titles is not None else "",
            }
        )
    return out


def dim_contributions(normed: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    # contribution per dim = normed[ia]*normed[ib]; aggregate mean abs over pairs
    contrib = normed[ia] * normed[ib]
    return np.mean(np.abs(contrib), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-dim dominance diagnostics for X_sonic.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=str, help="Optional metadata.db to enrich top tracks with artist/title/album.")
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact)
    labels = dim_labels(bundle)
    id_to_meta = {}
    if args.db:
        import sqlite3

        con = sqlite3.connect(args.db)
        cur = con.cursor()
        rows = cur.execute("SELECT track_id, artist, album, title FROM tracks").fetchall()
        id_to_meta = {str(r[0]): (r[1], r[2], r[3]) for r in rows}
        con.close()
    X = bundle.X_sonic
    rng = np.random.default_rng(args.seed)
    N = X.shape[0]
    ia = rng.integers(0, N, size=args.n)
    ib = rng.integers(0, N, size=args.n)
    same = ia == ib
    if same.any():
        ib[same] = (ib[same] + 1) % N

    stats: Dict[str, Any] = {}
    stats["per_dim_mean"] = X.mean(axis=0).tolist()
    stats["per_dim_std"] = X.std(axis=0).tolist()
    norm_raw, _ = compute_sonic_variant_norm(X, "raw")
    contrib_raw = dim_contributions(norm_raw, ia, ib)
    stats["per_dim_contrib_raw"] = contrib_raw.tolist()

    # Identify dominant dim
    top_idx = int(np.argmax(contrib_raw))
    top_tracks = top_tracks_for_dim(
        X[:, top_idx],
        bundle.track_ids,
        getattr(bundle, "track_artists", None),
        getattr(bundle, "track_titles", None),
        k=args.top_k,
    )
    if id_to_meta:
        for t in top_tracks:
            meta = id_to_meta.get(str(t["track_id"]))
            if meta:
                t["artist_db"], t["album_db"], t["title_db"] = meta
    stats["dominant_dim"] = {
        "index": top_idx,
        "dim_label": dim_label(bundle, top_idx),
        "mean": float(X[:, top_idx].mean()),
        "std": float(X[:, top_idx].std()),
        "contrib_raw": float(contrib_raw[top_idx]),
        "top_tracks": top_tracks,
    }

    diagnostics = {
        "artifact": str(args.artifact),
        "n_pairs": args.n,
        "seed": args.seed,
        "stats": stats,
    }

    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    ts = np.datetime64("now").astype(str).replace("-", "").replace(":", "").split(".")[0]
    out_json = diagnostics_dir / f"sonic_dim_dominance_{ts}.json"
    out_log = diagnostics_dir / "sonic_dim_dominance_runs.log"
    out_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    with out_log.open("a", encoding="utf-8") as f:
        f.write(f"{ts} dominant_dim={stats['dominant_dim']['index']} contrib={stats['dominant_dim']['contrib_raw']}\n")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
