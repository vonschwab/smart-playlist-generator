"""
Build a genre co-occurrence and Jaccard similarity matrix from album-level tags.

Uses canonical_genres.csv (core/micro by default) and album genres from
MusicBrainz + Discogs, normalized via the production SimilarityCalculator
logic (including broad/garbage/meta filters provided).
"""

import argparse
import csv
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.config_loader import Config
from src.similarity_calculator import SimilarityCalculator
from experiments.genre_similarity_lab.genre_normalization import (
    load_filter_sets,
    normalize_and_filter_genres,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Genre co-occurrence + Jaccard matrix (album-level).")
    parser.add_argument(
        "--db-path",
        help="Path to metadata DB (default from config.yaml).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--canonical-path",
        default="experiments/genre_similarity_lab/artifacts/canonical_genres.csv",
        help="Path to canonical_genres.csv.",
    )
    parser.add_argument(
        "--garbage-path",
        default="data/genre_filters/garbage_hard_block.csv",
        help="CSV with column 'genre' for hard-blocked genres (optional).",
    )
    parser.add_argument(
        "--meta-path",
        default="data/genre_filters/meta_broad.csv",
        help="CSV with column 'genre' for meta/broad genres (optional).",
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Include singleton tier genres in the matrix (default: only core+micro).",
    )
    parser.add_argument(
        "--output-matrix",
        default="experiments/genre_similarity_lab/artifacts/genre_cooc_matrix.npz",
        help="Output npz path for matrices.",
    )
    parser.add_argument(
        "--output-summary",
        default="experiments/genre_similarity_lab/artifacts/genre_cooc_summary.csv",
        help="Output CSV summary path.",
    )
    return parser.parse_args()


def resolve_db_path(config_path: str, explicit_db: Optional[str]) -> str:
    if explicit_db:
        return explicit_db
    try:
        cfg = Config(config_path)
        return cfg.get("library", "database_path", default="data/metadata.db")
    except Exception as exc:
        logger.warning("Failed to load config %s (%s); falling back to data/metadata.db", config_path, exc)
        return "data/metadata.db"


def load_set_csv(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = (row.get("genre") or "").strip().lower()
            if g:
                out.add(g)
    return out


def load_canonical(canonical_path: Path, include_singletons: bool) -> Tuple[List[str], Dict[str, int]]:
    genres = []
    with canonical_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tier = row.get("tier")
            g = row.get("genre")
            if not g:
                continue
            if not include_singletons and tier == "singleton":
                continue
            genres.append(g)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    return genres, genre_to_idx


def normalize_genre(sim_calc: SimilarityCalculator, genre: str) -> str:
    """Use production normalization if available."""
    try:
        return sim_calc._normalize_genre(genre)  # type: ignore[attr-defined]
    except Exception:
        return genre.strip().lower()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = resolve_db_path(args.config, args.db_path)
    logger.info("Using DB: %s", db_path)

    canonical_path = Path(args.canonical_path)
    genres, genre_to_idx = load_canonical(canonical_path, args.include_singletons)
    N = len(genres)
    canonical_set = set(genres)
    logger.info("Loaded %d canonical genres (include_singletons=%s)", N, args.include_singletons)

    # Initialize SimilarityCalculator for normalization/broad filtering consistency.
    try:
        cfg = Config(args.config)
        sim_calc = SimilarityCalculator(db_path=db_path, config=cfg.config)
    except Exception:
        sim_calc = SimilarityCalculator(db_path=db_path, config={})

    broad_filters = getattr(sim_calc, "broad_filters", set())
    broad_set, garbage_set, meta_set = load_filter_sets(broad_filters, args.garbage_path, args.meta_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Preload album -> genres (MB/Discogs only)
    sources = ["musicbrainz_release", "discogs_release", "discogs_master"]
    cur.execute(
        f"""
        SELECT album_id, genre, source
        FROM album_genres
        WHERE source IN ({','.join('?' for _ in sources)})
          AND genre != '__EMPTY__'
        """,
        sources,
    )

    album_genres: Dict[str, Set[str]] = defaultdict(set)
    album_raw_tags: Dict[str, List[str]] = defaultdict(list)
    album_srcs: Dict[str, Set[str]] = defaultdict(set)
    rows = cur.fetchall()
    logger.info("Fetched %d album-genre rows", len(rows))
    for row in rows:
        album_id = row["album_id"]
        g_raw = row["genre"]
        source = row["source"]
        album_raw_tags[album_id].append(g_raw)
        album_srcs[album_id].add("musicbrainz" if source == "musicbrainz_release" else "discogs")

    for album_id, raw_tags in album_raw_tags.items():
        normalized = normalize_and_filter_genres(
            raw_tags,
            broad_set=broad_set,
            garbage_set=garbage_set,
            meta_set=meta_set,
            canonical_set=canonical_set,
        )
        if not normalized:
            continue
        album_genres[album_id] = normalized

    # Co-occurrence matrix
    C = np.zeros((N, N), dtype=np.int32)

    for album_id, gset in album_genres.items():
        if not gset:
            continue
        idxs = [genre_to_idx[g] for g in gset if g in genre_to_idx]
        if not idxs:
            continue
        # Diagonal increments
        for i in idxs:
            C[i, i] += 1
        # Off-diagonal co-occurrence (unordered pairs)
        for i_pos, i in enumerate(idxs):
            for j in idxs[i_pos + 1 :]:
                C[i, j] += 1
                C[j, i] += 1

    # Jaccard similarity
    C_float = C.astype(np.float32)
    diag = np.diag(C_float)
    S = np.zeros_like(C_float, dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                S[i, j] = 1.0
                continue
            n_i = diag[i]
            n_j = diag[j]
            n_ij = C_float[i, j]
            denom = n_i + n_j - n_ij
            S[i, j] = (n_ij / denom) if denom > 0 else 0.0

    # Save matrices
    out_npz = Path(args.output_matrix)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        genres=np.array(genres),
        cooc_counts=C,
        sim_cooc=S,
    )
    logger.info("Saved co-occurrence npz: %s", out_npz)

    # Summary CSV
    out_csv = Path(args.output_summary)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "album_count", "top_neighbors"])
        writer.writeheader()
        for i, g in enumerate(genres):
            album_count = int(C[i, i])
            sims = S[i].copy()
            sims[i] = -1  # exclude self
            top_idx = np.argsort(sims)[::-1][:5]
            neigh = []
            for j in top_idx:
                if sims[j] <= 0:
                    continue
                neigh.append(f"{genres[j]}:{sims[j]:.2f}")
            writer.writerow(
                {
                    "genre": g,
                    "album_count": album_count,
                    "top_neighbors": "; ".join(neigh),
                }
            )
    logger.info("Saved summary CSV: %s", out_csv)

    # Diagnostics
    off_diag = S[np.triu_indices(N, k=1)]
    nonzero = off_diag[off_diag > 0]
    if nonzero.size:
        logger.info(
            "Off-diagonal Jaccard stats: min=%.4f max=%.4f mean=%.4f (nonzero count=%d)",
            nonzero.min(),
            nonzero.max(),
            nonzero.mean(),
            nonzero.size,
        )
    else:
        logger.info("No nonzero off-diagonal similarities.")

    anchors = ["jazz", "hard bop", "cool jazz", "rock", "post-punk", "shoegaze", "dream pop", "hip hop"]
    for anchor in anchors:
        if anchor not in genre_to_idx:
            continue
        i = genre_to_idx[anchor]
        sims = S[i].copy()
        sims[i] = -1
        top_idx = np.argsort(sims)[::-1][:5]
        neigh = [f"{genres[j]}:{sims[j]:.2f}" for j in top_idx if sims[j] > 0]
        logger.info("Anchor %s -> %s", anchor, "; ".join(neigh))


if __name__ == "__main__":
    main()
