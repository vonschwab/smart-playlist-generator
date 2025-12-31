#!/usr/bin/env python3
"""
Diagnose DS candidate sonic/genre similarity.

Examples:
  python scripts/diagnose_candidate_scores.py --seed "Songs: Ohia - Farewell Transmission" --ds-mode narrow --top 15 --show-offenders --explain
  python scripts/diagnose_candidate_scores.py --seed-track-id <track_id> --ds-mode narrow
"""
import argparse
import sqlite3
import logging
from typing import List, Tuple, Dict

import numpy as np

from src.config_loader import Config
from src.features.artifacts import load_artifact_bundle
from src.playlist.candidate_pool import _compute_genre_similarity
from src.similarity.sonic_variant import resolve_sonic_variant, compute_sonic_variant_matrix
from src.genre_similarity_v2 import GenreSimilarityV2
from src.similarity_calculator import SimilarityCalculator
from src.logging_utils import configure_logging

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


def _mask_genres(bundle, broad_filters: List[str]):
    genre_vocab = list(getattr(bundle, "genre_vocab", []) or [])
    mask = None
    if broad_filters and genre_vocab:
        mask = np.array([g.lower() not in broad_filters for g in genre_vocab], dtype=bool)
        if mask.shape[0] != len(genre_vocab):
            mask = None
    X_genre_raw = bundle.X_genre_raw
    X_genre_smoothed = bundle.X_genre_smoothed
    if mask is not None:
        if X_genre_raw is not None and mask.shape[0] == X_genre_raw.shape[1]:
            X_genre_raw = X_genre_raw[:, mask]
            genre_vocab = [g for g, keep in zip(genre_vocab, mask) if keep]
        if X_genre_smoothed is not None and mask.shape[0] == X_genre_smoothed.shape[1]:
            X_genre_smoothed = X_genre_smoothed[:, mask]
    return X_genre_raw, X_genre_smoothed, genre_vocab, mask


def _compute_sims(
    bundle,
    seed_idx: int,
    broad_filters: List[str],
    min_genre_similarity: float,
    genre_method: str,
    w_sonic: float,
    w_genre: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_genre_raw, X_genre_smoothed, genre_vocab, mask = _mask_genres(bundle, broad_filters)
    # Sonic
    X_sonic = bundle.X_sonic
    if getattr(bundle, "sonic_variant", None) != "raw":
        X_sonic, _ = compute_sonic_variant_matrix(bundle.X_sonic, resolve_sonic_variant(explicit_variant=None, config_variant=None), l2=False)
    sonic_norm = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
    seed_vec_sonic = sonic_norm[seed_idx]
    sonic_sim = np.dot(sonic_norm, seed_vec_sonic)
    sonic_sim[seed_idx] = -1.0

    # Genre similarity (hard overlap guard using raw binary if available)
    genre_sim = np.zeros_like(sonic_sim)
    if X_genre_raw is not None or X_genre_smoothed is not None:
        genre_matrix = X_genre_smoothed if genre_method != "weighted_jaccard" and X_genre_smoothed is not None else X_genre_raw
        if genre_matrix is None and X_genre_raw is not None:
            genre_matrix = X_genre_raw
        if genre_matrix is not None:
            seed_genres = genre_matrix[seed_idx]
            genre_sim = _compute_genre_similarity(seed_genres, genre_matrix, method=genre_method)
            genre_sim[seed_idx] = 1.0
            if mask is not None and X_genre_raw is not None:
                seed_binary = (X_genre_raw[seed_idx] > 0).astype(float)
                overlaps = (X_genre_raw > 0) & (seed_binary > 0)
                zero_overlap = overlaps.sum(axis=1) == 0
                genre_sim = np.where(zero_overlap, 0.0, genre_sim)

    hybrid_sim = w_sonic * sonic_sim + w_genre * genre_sim
    return sonic_sim, genre_sim, hybrid_sim


def main():
    parser = argparse.ArgumentParser(description="Diagnose DS candidate scores")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", help='Seed as "Artist - Title"')
    parser.add_argument("--seed-track-id", help="Seed track_id (overrides --seed)")
    parser.add_argument("--ds-mode", default="narrow", choices=["narrow", "dynamic", "discover", "sonic_only"])
    parser.add_argument("--top", type=int, default=25, help="Top N candidates to display by hybrid sim")
    parser.add_argument("--show-offenders", action="store_true", help="Show low-sonic / high-genre offenders")
    parser.add_argument("--explain", action="store_true", help="Show genre similarity explanation for offenders")
    args = parser.parse_args()

    configure_logging(level="INFO")

    cfg = Config(args.config)
    ds_cfg = cfg.config.get("playlists", {}).get("ds_pipeline", {}) or {}
    candidate_cfg = ds_cfg.get("candidate_pool", {}) or {}
    embedding_cfg = ds_cfg.get("embedding", {}) or {}
    artifact_path = ds_cfg.get("artifact_path") or cfg.config.get("artifacts", {}).get("path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz")
    broad_filters = [str(b).lower() for b in candidate_cfg.get("broad_filters", ["rock", "indie", "alternative", "pop"])]
    min_genre_similarity = candidate_cfg.get("min_genre_similarity", 0.3)
    genre_method = ds_cfg.get("genre_method") or "ensemble"
    w_sonic = float(embedding_cfg.get("sonic_weight", 0.6))
    w_genre = float(embedding_cfg.get("genre_weight", 0.4))

    seed_track_id = args.seed_track_id or _resolve_seed_track_id(cfg.library_database_path, args.seed or "")

    bundle = load_artifact_bundle(artifact_path)
    seed_idx = bundle.track_id_to_index.get(str(seed_track_id))
    if seed_idx is None:
        raise ValueError(f"Seed track_id {seed_track_id} not found in artifact")

    sonic_sim, genre_sim, hybrid_sim = _compute_sims(
        bundle,
        seed_idx,
        broad_filters,
        min_genre_similarity,
        genre_method,
        w_sonic,
        w_genre,
    )

    def _label(idx: int) -> str:
        artist = bundle.track_artists[idx] if getattr(bundle, "track_artists", None) is not None else ""
        title = bundle.track_titles[idx] if getattr(bundle, "track_titles", None) is not None else ""
        return f"{artist} - {title}"

    order = np.argsort(hybrid_sim)[::-1]
    logger.info(f"Top {args.top} by hybrid similarity (mode={args.ds_mode}):")
    logger.info("rank\ttrack_id\thybrid\tsonic\tgenre\ttitle")
    shown = 0
    for idx in order:
        if idx == seed_idx:
            continue
        logger.info(f"{shown+1}\t{bundle.track_ids[idx]}\t{hybrid_sim[idx]:.3f}\t{sonic_sim[idx]:.3f}\t{genre_sim[idx]:.3f}\t{_label(idx)}")
        shown += 1
        if shown >= args.top:
            break

    if args.show_offenders:
        logger.info("\nOffenders (sonic<0 OR genre>0.6 with zero filtered overlap):")
        offenders = []
        sim_calc = None
        genre_calc = None
        if args.explain:
            sim_calc = SimilarityCalculator(db_path=cfg.library_database_path, config=cfg.config)
            genre_calc = GenreSimilarityV2()
        # For overlap, recompute with raw mask
        X_genre_raw, _, _, mask = _mask_genres(bundle, broad_filters)
        seed_binary = (X_genre_raw[seed_idx] > 0).astype(float) if X_genre_raw is not None else None
        for idx in range(len(bundle.track_ids)):
            if idx == seed_idx:
                continue
            zero_overlap = False
            if X_genre_raw is not None and seed_binary is not None:
                overlap = (X_genre_raw[idx] > 0) & (seed_binary > 0)
                zero_overlap = overlap.sum() == 0
            if sonic_sim[idx] < 0 or (genre_sim[idx] > 0.6 and zero_overlap):
                offenders.append(idx)
        if not offenders:
            logger.info("None")
        else:
            for idx in offenders:
                logger.info(f"{bundle.track_ids[idx]}\t{sonic_sim[idx]:.3f}\t{genre_sim[idx]:.3f}\t{_label(idx)}")
                if args.explain:
                    seed_genres = sim_calc.get_filtered_combined_genres_for_track(str(seed_track_id))
                    cand_genres = sim_calc.get_filtered_combined_genres_for_track(str(bundle.track_ids[idx]))
                    score, details = genre_calc.calculate_similarity_with_explain(
                        seed_genres,
                        cand_genres,
                        broad_filters=broad_filters,
                    )
                    logger.info(f"  explain score={score:.3f} seed_filtered={details.get('seed_genres_filtered')} cand_filtered={details.get('cand_genres_filtered')}")
                    if details.get("top_pairs"):
                        for pair in details["top_pairs"][:3]:
                            logger.info(f"    {pair['seed_genre']} vs {pair['cand_genre']}: {pair['pair_score']:.3f} ({pair['confidence']})")


if __name__ == "__main__":
    main()
