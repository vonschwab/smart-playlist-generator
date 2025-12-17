"""
Step 1: Build numeric sonic and genre matrices from the metadata DB.

This script:
- Pulls tracks + sonic_features from the library DB.
- Converts sonic_features JSON into aligned numeric vectors via SimilarityCalculator.
- Builds a binary genre presence matrix using combined track genres.
- Prints diagnostics and optionally saves matrices to artifacts/.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import re

from src.config_loader import Config
from src.metadata_client import MetadataClient
from src.similarity_calculator import SimilarityCalculator

logger = logging.getLogger(__name__)


def _resolve_db_path(config_path: str, explicit_db: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Resolve DB path from CLI override or config.yaml."""
    config_data: Dict[str, Any] = {}
    if explicit_db:
        return explicit_db, config_data
    try:
        cfg = Config(config_path)
        config_data = cfg.config
        return cfg.get('library', 'database_path', default='data/metadata.db'), config_data
    except Exception as exc:
        logger.warning("Falling back to default DB path after config load failure: %s", exc)
        return 'data/metadata.db', config_data


def _table_has_column(conn, table: str, column: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row['name'] == column for row in cursor.fetchall())


def _table_exists(conn, table: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cursor.fetchone() is not None


def _fetch_tracks(conn, limit: int) -> Sequence[Dict[str, any]]:
    """Fetch a batch of tracks with sonic features."""
    if not _table_has_column(conn, 'tracks', 'sonic_features'):
        raise RuntimeError("tracks table does not contain sonic_features column.")

    cursor = conn.cursor()
    if limit and limit > 0:
        cursor.execute(
            """
            SELECT track_id, title, artist, album, norm_artist, sonic_features
            FROM tracks
            WHERE sonic_features IS NOT NULL
            LIMIT ?
            """,
            (limit,),
        )
    else:
        cursor.execute(
            """
            SELECT track_id, title, artist, album, norm_artist, sonic_features
            FROM tracks
            WHERE sonic_features IS NOT NULL
            """
        )

    results = []
    for row in cursor.fetchall():
        row_dict = {key: row[key] for key in row.keys()}
        results.append(row_dict)
    return results


def _human_track_label(track_id: str, artist: Optional[str], title: Optional[str]) -> str:
    """Readable track label for diagnostics."""
    artist = artist or "Unknown Artist"
    title = title or "Unknown Title"
    return f"{track_id} | {artist} - {title}"


def normalize_artist_key(raw: Optional[str]) -> str:
    """
    Normalize an artist identifier for grouping.

    - Lowercase and strip whitespace.
    - Remove common featuring suffixes (feat/ft/featuring and anything after).
    - Remove punctuation except spaces.
    - Collapse multiple spaces.
    """
    if not raw:
        return ""
    text = str(raw).lower().strip()
    # Remove feat/featuring clauses
    text = re.split(r"\b(feat|ft|featuring)\b", text)[0]
    # Keep alphanumerics and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Build sonic/genre matrices from the DB (Step 1).")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--db-path", help="Optional explicit DB path (overrides config)")
    parser.add_argument("--max-tracks", type=int, default=5000, help="Maximum tracks to load (default: 5000)")
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save matrices to experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz",
    )
    parser.add_argument(
        "--use-genre-sim",
        action="store_true",
        help="Apply genre similarity smoothing (X_genre @ S_sub) before PCA.",
    )
    parser.add_argument(
        "--genre-sim-path",
        default="experiments/genre_similarity_lab/artifacts/genre_similarity_matrix.npz",
        help="Path to genre similarity matrix npz (default: experiments/genre_similarity_lab/artifacts/genre_similarity_matrix.npz)",
    )
    parser.add_argument(
        "--export-sonic-segments",
        action="store_true",
        help="Export segment-level sonic matrices (start/mid/end) alongside X_sonic.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path, config_data = _resolve_db_path(args.config, args.db_path)
    logger.info("Using database: %s", db_path)

    metadata = MetadataClient(db_path)
    sim_calc = SimilarityCalculator(db_path=db_path, config=config_data)

    try:
        tracks = _fetch_tracks(sim_calc.conn, args.max_tracks)
    except RuntimeError as exc:
        logger.error("Failed to fetch tracks: %s", exc)
        return

    requested = len(tracks)

    # First pass: collect filtered genres per track using production logic
    track_genres_map: Dict[str, List[str]] = {}
    vocab_set = set()
    garbage: set = set()
    meta: set = set()
    # Instrumentation counters
    total_tracks_seen = 0
    total_tracks_with_raw = 0
    total_tracks_with_filtered = 0
    raw_genre_set = set()
    filtered_genre_set = set()
    empty_filtered_examples = []

    logger.info(
        "Filter sets: broad=%d, garbage=%d, meta=%d",
        len(getattr(sim_calc, "broad_filters", [])),
        len(garbage),
        len(meta),
    )

    for row in tracks:
        track_id = row.get('track_id')
        if not track_id:
            continue
        total_tracks_seen += 1

        raw_genres = sim_calc._get_combined_genres(track_id) if hasattr(sim_calc, "_get_combined_genres") else []
        if raw_genres:
            total_tracks_with_raw += 1
            raw_genre_set.update(raw_genres)

        genres = sim_calc.get_filtered_combined_genres_for_track(track_id) or []
        if genres:
            total_tracks_with_filtered += 1
            filtered_genre_set.update(genres)
        else:
            if raw_genres and len(empty_filtered_examples) < 5:
                empty_filtered_examples.append(
                    {
                        "track_id": track_id,
                        "artist": row.get("artist", ""),
                        "title": row.get("title", ""),
                        "raw_genres": raw_genres,
                        "filtered_genres": genres,
                    }
                )

        track_genres_map[track_id] = genres
        vocab_set.update(genres)

    genre_vocab = sorted(vocab_set)
    logger.warning(
        "Genre filtering stats: tracks_seen=%d, tracks_with_raw=%d, tracks_with_filtered=%d, unique_raw=%d, unique_filtered=%d",
        total_tracks_seen,
        total_tracks_with_raw,
        total_tracks_with_filtered,
        len(raw_genre_set),
        len(filtered_genre_set),
    )
    if empty_filtered_examples:
        for ex in empty_filtered_examples:
            logger.debug(
                "Filtered to empty genres. track=%s artist=%s title=%s raw=%s filtered=%s",
                ex["track_id"],
                ex["artist"],
                ex["title"],
                ex["raw_genres"],
                ex["filtered_genres"],
            )

    if not genre_vocab:
        logger.warning("No genres found after filtering; aborting.")
        return

    genre_index = {g: idx for idx, g in enumerate(genre_vocab)}

    X_sonic: List[np.ndarray] = []
    X_sonic_start: List[np.ndarray] = []
    X_sonic_mid: List[np.ndarray] = []
    X_sonic_end: List[np.ndarray] = []
    X_genre: List[np.ndarray] = []
    track_ids: List[str] = []
    artist_names: List[str] = []
    artist_keys: List[str] = []
    artist_key_source: List[str] = []
    artist_key_missing: List[bool] = []
    track_titles: List[str] = []

    skipped_missing_sonic = 0
    skipped_missing_genres = 0
    for row in tracks:
        track_id = row.get('track_id')
        title = row.get('title')
        artist = row.get('artist')
        artist_key = row.get('norm_artist') or ""

        raw_features = row.get('sonic_features')
        if not raw_features:
            skipped_missing_sonic += 1
            continue

        try:
            sonic_features = json.loads(raw_features)
        except json.JSONDecodeError:
            skipped_missing_sonic += 1
            continue

        sonic_vec = sim_calc.build_sonic_feature_vector(sonic_features)
        if sonic_vec.size == 0:
            skipped_missing_sonic += 1
            continue
        if args.export_sonic_segments:
            start_vec = sim_calc.build_sonic_feature_vector_by_segment(sonic_features, "start")
            mid_vec = sim_calc.build_sonic_feature_vector_by_segment(sonic_features, "mid")
            end_vec = sim_calc.build_sonic_feature_vector_by_segment(sonic_features, "end")
            # Enforce same shape as base vector
            if start_vec.shape != sonic_vec.shape:
                start_vec = sonic_vec
            if mid_vec.shape != sonic_vec.shape:
                mid_vec = sonic_vec
            if end_vec.shape != sonic_vec.shape:
                end_vec = sonic_vec

        genres = track_genres_map.get(track_id, [])
        if not genres:
            skipped_missing_genres += 1
            continue

        genre_vec = np.zeros(len(genre_vocab), dtype=float)
        for g in genres:
            idx = genre_index.get(g)
            if idx is not None:
                genre_vec[idx] = 1.0

        if genre_vec.sum() == 0:
            skipped_missing_genres += 1
            continue

        # Artist key normalization with fallbacks
        norm_artist_raw = row.get('norm_artist')
        normalized_key = normalize_artist_key(norm_artist_raw)
        source = "norm_artist"
        if not normalized_key:
            fallback_display = normalize_artist_key(artist)
            if fallback_display:
                normalized_key = fallback_display
                source = "fallback_display"
            else:
                normalized_key = f"unknown:{track_id}"
                source = "unknown"

        X_sonic.append(sonic_vec)
        if args.export_sonic_segments:
            X_sonic_start.append(start_vec)
            X_sonic_mid.append(mid_vec)
            X_sonic_end.append(end_vec)
        X_genre.append(genre_vec)
        track_ids.append(track_id)
        artist_names.append(artist or "")
        track_titles.append(title or "")
        artist_keys.append(normalized_key)
        artist_key_source.append(source)
        artist_key_missing.append(source == "unknown")

    if not X_sonic or not X_genre:
        logger.warning("No usable tracks after filtering; nothing to report.")
        return

    X_sonic_matrix = np.vstack(X_sonic)
    X_sonic_start_matrix = np.vstack(X_sonic_start) if args.export_sonic_segments else None
    X_sonic_mid_matrix = np.vstack(X_sonic_mid) if args.export_sonic_segments else None
    X_sonic_end_matrix = np.vstack(X_sonic_end) if args.export_sonic_segments else None
    X_genre_matrix = np.vstack(X_genre)
    X_genre_raw = X_genre_matrix.astype(np.float32, copy=True)  # Preserve binary/raw matrix
    X_genre_smoothed = X_genre_matrix  # Default: no smoothing applied

    if args.use_genre_sim:
        sim_path = Path(args.genre_sim_path)
        logger.info("Applying genre similarity smoothing using %s", sim_path)
        sim_npz = np.load(sim_path, allow_pickle=True)
        sim_genres = sim_npz["genres"].tolist()
        S_full = sim_npz["S"]
        logger.info(
            "Genre similarity matrix loaded: G=%d, vocab size=%d",
            len(sim_genres),
            len(genre_vocab),
        )

        sim_index = {g: i for i, g in enumerate(sim_genres)}
        missing = [g for g in genre_vocab if g not in sim_index]
        present = [i for i, g in enumerate(genre_vocab) if g in sim_index]
        if missing:
            logger.warning(
                "Missing genres in similarity matrix (using identity for them): count=%d sample=%s",
                len(missing),
                missing[:10],
            )

        # Build submatrix aligned to genre_vocab; missing genres get identity rows/cols.
        S_sub = np.eye(len(genre_vocab), dtype=np.float32)
        if present:
            sim_idx = [sim_index[genre_vocab[j]] for j in present]
            S_sub[np.ix_(present, present)] = S_full[np.ix_(sim_idx, sim_idx)].astype(np.float32)
        logger.info(
            "Genre similarity submatrix shape: %s (from %s) with %d present, %d missing",
            S_sub.shape,
            S_full.shape,
            len(present),
            len(missing),
        )

        X_genre_smoothed = X_genre_matrix.astype(np.float32) @ S_sub
        logger.info(
            "Smoothed genre matrix: X_genre_raw (binary) %s @ S_sub (%s) -> X_genre_smoothed %s",
            X_genre_matrix.shape,
            S_sub.shape,
            X_genre_smoothed.shape,
        )
        X_genre_matrix = X_genre_smoothed

    kept = len(track_ids)
    # Artist key diagnostics
    missing_artist_keys = sum(artist_key_missing)
    missing_frac = missing_artist_keys / kept
    from collections import Counter

    key_counts = Counter(artist_keys)
    fallback_counts = Counter(k for k, src in zip(artist_keys, artist_key_source) if src == "fallback_display")
    top_keys = key_counts.most_common(10)
    top_fallback = fallback_counts.most_common(10)
    logger.info(
        "Artist key stats: kept=%d, unique_keys=%d, missing=%d (%.2f%%)",
        kept,
        len(key_counts),
        missing_artist_keys,
        missing_frac * 100,
    )
    if top_keys:
        logger.info("Top artist_keys: %s", top_keys)
    if top_fallback:
        logger.info("Top fallback_display keys: %s", top_fallback)

    print("=== Data Matrix Diagnostics ===")
    print(f"Tracks requested: {requested}")
    print(f"Tracks kept:      {kept}")
    print(f"Skipped (sonic missing/invalid): {skipped_missing_sonic}")
    print(f"Skipped (genres missing):        {skipped_missing_genres}")
    print(f"X_sonic shape: {X_sonic_matrix.shape}")
    print(f"X_genre shape: {X_genre_matrix.shape}")
    print(f"Genre vocabulary size: {len(genre_vocab)}")
    print(f"Unique artist_keys: {len(set(artist_keys))}")
    if args.export_sonic_segments and X_sonic_start_matrix is not None:
        print(
            f"Sonic segment shapes: start={X_sonic_start_matrix.shape}, mid={X_sonic_mid_matrix.shape}, end={X_sonic_end_matrix.shape}"
        )
        print(
            "Segment fallback counts: start=%d mid=%d end=%d"
            % (
                sim_calc.segment_fallback_counts.get("start", 0),
                sim_calc.segment_fallback_counts.get("mid", 0),
                sim_calc.segment_fallback_counts.get("end", 0),
            )
        )
    print()

    rows_to_show = min(5, kept)
    for idx in range(rows_to_show):
        genre_indices = np.nonzero(X_genre_matrix[idx])[0].tolist()
        genre_names = [genre_vocab[i] for i in genre_indices]
        # Encode-safe display for Windows console
        safe_label = _human_track_label(track_ids[idx], artist_names[idx], track_titles[idx]).encode('cp1252', errors='replace').decode('cp1252')
        safe_genres = [g.encode('cp1252', errors='replace').decode('cp1252') for g in genre_names]
        print(f"{idx + 1}. {safe_label}")
        print(f"   Sonic vector length: {X_sonic_matrix.shape[1]}")
        print(f"   Genres ({len(genre_indices)}): {safe_genres}")

    if args.save_artifacts:
        artifacts_dir = Path("experiments/genre_similarity_lab/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_dir / "data_matrices_step1.npz"
        np.savez(
            artifact_path,
            X_sonic=X_sonic_matrix,
            X_genre=X_genre_matrix,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            X_sonic_start=X_sonic_start_matrix if args.export_sonic_segments else None,
            X_sonic_mid=X_sonic_mid_matrix if args.export_sonic_segments else None,
            X_sonic_end=X_sonic_end_matrix if args.export_sonic_segments else None,
            track_ids=np.array(track_ids),
            artist_names=np.array(artist_names),
            track_titles=np.array(track_titles),
            artist_keys=np.array(artist_keys),
            artist_key_source=np.array(artist_key_source),
            artist_key_missing=np.array(artist_key_missing),
            genre_vocab=np.array(genre_vocab),
        )
        print(f"\nSaved matrices to {artifact_path}")


if __name__ == "__main__":
    main()
