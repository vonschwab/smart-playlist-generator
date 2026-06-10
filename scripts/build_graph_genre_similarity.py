#!/usr/bin/env python3
"""
Build Graph-Derived Genre Similarity (Stage 2)
==============================================

Reads the SP3a layered taxonomy (data/layered_genre_taxonomy.yaml), converts
graph relationships into a genre-to-genre similarity matrix, and writes an NPZ
with the same key layout as the co-occurrence matrix from analyze_library.py
({genre_vocab, S, stats}). Optionally also writes a {genre: {neighbor: sim}}
YAML in the data/genre_similarity.yaml schema for the DJ-bridge label scorer.

Nothing consumes these outputs yet — Stage 3 wires them in behind config
flags. This script never touches data/metadata.db or the hand-maintained
data/genre_similarity.yaml.

Usage:
    python scripts/build_graph_genre_similarity.py
    python scripts/build_graph_genre_similarity.py --yaml-out data/genre_similarity_graph.yaml
    python scripts/build_graph_genre_similarity.py --include-review --top-neighbors shoegaze

Exit codes: 0 success, 1 error
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.genre.graph_adapter import load_graph_adapter
from src.genre.graph_similarity import (
    build_graph_similarity,
    export_neighbor_yaml,
    save_graph_similarity_npz,
)
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_OUT = ROOT / "data" / "genre_similarity_graph.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build graph-derived genre similarity matrix")
    parser.add_argument("--taxonomy", type=Path, default=None,
                        help="Taxonomy YAML (default: data/layered_genre_taxonomy.yaml)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--yaml-out", type=Path, default=None,
                        help="Also write a neighbor YAML in the data/genre_similarity.yaml schema")
    parser.add_argument("--include-review", action="store_true",
                        help="Include review-status genres as vocabulary dimensions")
    parser.add_argument("--min-sim", type=float, default=0.05, help="YAML export floor")
    parser.add_argument("--top-k", type=int, default=12, help="YAML export neighbors per genre")
    parser.add_argument("--top-neighbors", default=None, metavar="GENRE",
                        help="Print the top-10 neighbors for GENRE and exit without writing")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)

    try:
        adapter = load_graph_adapter(args.taxonomy)
        result = build_graph_similarity(adapter, include_review=args.include_review)

        if args.top_neighbors:
            verdict = adapter.canonicalize_tag(args.top_neighbors)
            if verdict.canonical is None:
                logger.error("Not a canonical genre: %s (resolution=%s)",
                             args.top_neighbors, verdict.resolution)
                return 1
            index = {g: i for i, g in enumerate(result.genre_vocab)}
            row = result.S[index[verdict.canonical]]
            order = np.argsort(-row)
            print(f"Top neighbors of {verdict.canonical!r} (taxonomy {result.stats['taxonomy_version']}):")
            shown = 0
            for j in order:
                name = result.genre_vocab[int(j)]
                if name == verdict.canonical:
                    continue
                sim = float(row[int(j)])
                if sim <= 0 or shown >= 10:
                    break
                print(f"  {sim:.3f}  {name}")
                shown += 1
            return 0

        save_graph_similarity_npz(result, args.out)
        if args.yaml_out is not None:
            export_neighbor_yaml(result, args.yaml_out, min_sim=args.min_sim, top_k=args.top_k)
        print(
            f"Built graph similarity: {result.stats['genres_kept']} genres, "
            f"{result.stats['edges_used']} edges used, "
            f"{100 * result.stats['nonzero_offdiag_fraction']:.1f}% off-diag nonzero "
            f"(taxonomy {result.stats['taxonomy_version']})"
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
