#!/usr/bin/env python3
"""
Build Beat 3-Tower Artifacts
============================

Builds normalized 3-tower artifacts from tracks with beat3tower features.

Usage:
    python scripts/build_beat3tower_artifacts.py \
        --db-path data/metadata.db \
        --config config.yaml \
        --output experiments/genre_similarity_lab/artifacts/data_matrices_beat3tower.npz

Features:
- Loads beat3tower features from database
- Separates into rhythm/timbre/harmony towers
- Applies per-tower robust normalization with optional PCA whitening
- Computes calibration statistics for weighted combination
- Saves all required matrices for playlist generation

Output NPZ contents:
- X_sonic_rhythm, X_sonic_timbre, X_sonic_harmony: Per-tower embeddings
- X_sonic_rhythm_start/end, etc.: Segment embeddings for transitions
- X_sonic: Concatenated towers (backward compatibility)
- tower_calibration: Statistics for calibrated similarity
- normalizer_params: For reproducibility
- tower_dims: Dimension counts per tower
- track_ids, artist_keys, etc.: Metadata
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.beat3tower_normalizer import (
    Beat3TowerNormalizer,
    NormalizerConfig,
    compute_tower_calibration_stats,
    l2_normalize,
)
from src.features.beat3tower_types import Beat3TowerFeatures

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 3-tower artifacts from beat3tower features"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to metadata database",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for NPZ artifact",
    )
    parser.add_argument(
        "--genre-sim-path",
        help="Path to genre similarity matrix NPZ (optional)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="Maximum tracks to include (0 = all)",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA whitening (use robust standardization only)",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Fraction of variance to retain in PCA (default: 0.95)",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=3.0,
        help="Sigma for outlier clipping (default: 3.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def normalize_artist_key(raw: Optional[str], track_id: Optional[str]) -> str:
    """Normalize artist identifier with fallbacks."""
    if raw is None:
        raw = ""
    text = str(raw).lower().strip()
    text = re.split(r"\b(feat|ft|featuring)\b", text)[0]
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        return text
    if track_id:
        return f"unknown:{track_id}"
    return "unknown"


def load_tracks_with_beat3tower(
    db_path: str,
    max_tracks: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load tracks that have beat3tower features.

    Returns:
        Tuple of (tracks_metadata, features_list)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check for required columns
    cursor.execute("PRAGMA table_info(tracks)")
    columns = {row[1] for row in cursor.fetchall()}

    if "sonic_features" not in columns:
        raise RuntimeError("tracks table missing sonic_features column")

    has_norm_artist = "norm_artist" in columns
    norm_artist_expr = "norm_artist" if has_norm_artist else "artist"

    limit_clause = f" LIMIT {int(max_tracks)}" if max_tracks > 0 else ""

    cursor.execute(
        f"""
        SELECT track_id, artist, title, album, {norm_artist_expr} as norm_artist, sonic_features
        FROM tracks
        WHERE sonic_features IS NOT NULL
        {limit_clause}
        """
    )
    rows = cursor.fetchall()
    conn.close()

    tracks = []
    features_list = []
    skipped_non_beat3tower = 0
    skipped_parse_error = 0

    for row in rows:
        track_id = row["track_id"]
        raw_features = row["sonic_features"]

        try:
            features = json.loads(raw_features)
        except Exception:
            skipped_parse_error += 1
            continue

        # Check if this is a beat3tower feature set
        if not _is_beat3tower_features(features):
            skipped_non_beat3tower += 1
            continue

        tracks.append({
            "track_id": track_id,
            "artist": row["artist"] or "",
            "title": row["title"] or "",
            "album": row["album"] or "",
            "norm_artist": row["norm_artist"] or row["artist"] or "",
        })
        features_list.append(features)

    logger.info(
        f"Loaded {len(tracks)} tracks with beat3tower features "
        f"(skipped {skipped_non_beat3tower} non-beat3tower, {skipped_parse_error} parse errors)"
    )

    return tracks, features_list


def _is_beat3tower_features(features: Dict[str, Any]) -> bool:
    """Check if features are from beat3tower extraction."""
    # Check for beat3tower structure
    if "full" in features and isinstance(features["full"], dict):
        full = features["full"]
        # Check for tower structure
        if all(key in full for key in ["rhythm", "timbre", "harmony"]):
            return True
        # Check for extraction_method marker
        if full.get("extraction_method") == "beat3tower":
            return True
    return False


def extract_tower_vectors(
    features_list: List[Dict[str, Any]],
    segment: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Extract tower vectors from feature dictionaries.

    Args:
        features_list: List of beat3tower feature dictionaries
        segment: Which segment to extract ('full', 'start', 'mid', 'end')

    Returns:
        Tuple of (X_rhythm, X_timbre, X_harmony, bpm_list)
    """
    X_rhythm = []
    X_timbre = []
    X_harmony = []
    bpm_list = []

    for features in features_list:
        seg_features = features.get(segment, features.get("full", {}))
        feat_obj = Beat3TowerFeatures.from_dict(seg_features)

        X_rhythm.append(feat_obj.rhythm.to_vector())
        X_timbre.append(feat_obj.timbre.to_vector())
        X_harmony.append(feat_obj.harmony.to_vector())
        bpm_list.append(feat_obj.bpm_info.primary_bpm)

    return (
        np.vstack(X_rhythm) if X_rhythm else np.array([]),
        np.vstack(X_timbre) if X_timbre else np.array([]),
        np.vstack(X_harmony) if X_harmony else np.array([]),
        bpm_list,
    )


def smooth_genres(
    X: np.ndarray,
    S: Optional[np.ndarray],
    vocab: List[str],
    sim_vocab: Optional[List[str]],
) -> np.ndarray:
    """Apply genre similarity smoothing."""
    if S is None or sim_vocab is None:
        return X.astype(np.float32)
    index = {g: i for i, g in enumerate(sim_vocab)}
    present = [i for i, g in enumerate(vocab) if g in index]
    if not present:
        return X.astype(np.float32)
    S_sub = np.eye(len(vocab), dtype=np.float32)
    sim_idx = [index[vocab[i]] for i in present]
    S_sub[np.ix_(present, present)] = S[np.ix_(sim_idx, sim_idx)].astype(np.float32)
    return X.astype(np.float32) @ S_sub


def load_genres_for_tracks(
    db_path: str,
    track_ids: List[str],
) -> Tuple[List[List[str]], List[str]]:
    """
    Load genres for tracks from database.

    Returns:
        Tuple of (genre_lists per track, vocabulary)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check for track_genres table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='track_genres'")
    has_track_genres = cursor.fetchone() is not None

    genre_lists = []
    vocab_counts: Dict[str, int] = {}

    if has_track_genres:
        # Query genres from track_genres table
        placeholders = ",".join(["?"] * len(track_ids))
        cursor.execute(
            f"""
            SELECT track_id, genre, source
            FROM track_genres
            WHERE track_id IN ({placeholders})
            """,
            track_ids,
        )

        track_genres: Dict[str, List[str]] = {tid: [] for tid in track_ids}
        for row in cursor.fetchall():
            source = row["source"] or ""
            # Filter to MB/Discogs sources as per DS plan
            if source.lower() in ("musicbrainz", "mb", "discogs"):
                genre = row["genre"]
                track_genres[row["track_id"]].append(genre)
                vocab_counts[genre] = vocab_counts.get(genre, 0) + 1

        genre_lists = [track_genres.get(tid, []) for tid in track_ids]
    else:
        # No genres available
        genre_lists = [[] for _ in track_ids]

    conn.close()

    vocab = sorted(vocab_counts.keys())
    return genre_lists, vocab


def build_genre_matrices(
    genre_lists: List[List[str]],
    vocab: List[str],
    genre_sim_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build genre matrices.

    Returns:
        Tuple of (X_genre_raw, X_genre_smoothed)
    """
    vocab_index = {g: i for i, g in enumerate(vocab)}
    n_tracks = len(genre_lists)
    n_genres = len(vocab)

    X_genre_raw = np.zeros((n_tracks, n_genres), dtype=np.float32)
    for i, genres in enumerate(genre_lists):
        for g in genres:
            j = vocab_index.get(g)
            if j is not None:
                X_genre_raw[i, j] = 1.0

    # Apply smoothing if similarity matrix provided
    X_genre_smoothed = X_genre_raw.copy()
    if genre_sim_path:
        try:
            sim_npz = np.load(genre_sim_path, allow_pickle=True)
            sim_vocab = sim_npz["genre_vocab"].tolist()
            S = sim_npz["S"]
            X_genre_smoothed = smooth_genres(X_genre_raw, S, vocab, sim_vocab)
            logger.info(f"Applied genre smoothing from {genre_sim_path}")
        except Exception as e:
            logger.warning(f"Failed to load genre similarity matrix: {e}")

    return X_genre_raw, X_genre_smoothed


def build_artifacts(args: argparse.Namespace) -> None:
    """Main artifact building workflow."""
    logger.info("Loading tracks with beat3tower features...")
    tracks, features_list = load_tracks_with_beat3tower(args.db_path, args.max_tracks)

    if not tracks:
        raise RuntimeError("No tracks with beat3tower features found")

    logger.info(f"Processing {len(tracks)} tracks...")

    # Extract tower vectors for all segments
    X_rhythm_full, X_timbre_full, X_harmony_full, bpm_list = extract_tower_vectors(
        features_list, "full"
    )
    X_rhythm_start, X_timbre_start, X_harmony_start, _ = extract_tower_vectors(
        features_list, "start"
    )
    X_rhythm_mid, X_timbre_mid, X_harmony_mid, _ = extract_tower_vectors(
        features_list, "mid"
    )
    X_rhythm_end, X_timbre_end, X_harmony_end, _ = extract_tower_vectors(
        features_list, "end"
    )

    logger.info(
        f"Tower dimensions before normalization: "
        f"rhythm={X_rhythm_full.shape[1]}, "
        f"timbre={X_timbre_full.shape[1]}, "
        f"harmony={X_harmony_full.shape[1]}"
    )

    # Configure normalizer
    config = NormalizerConfig(
        clip_sigma=args.clip_sigma,
        use_pca_whitening=not args.no_pca,
        pca_variance_retain=args.pca_variance,
        l2_normalize=True,
        random_seed=args.random_seed,
    )

    # Fit normalizer on full segment (representative of overall distribution)
    logger.info("Fitting normalizer on full-track features...")
    normalizer = Beat3TowerNormalizer(config)
    normalizer.fit(X_rhythm_full, X_timbre_full, X_harmony_full)

    output_dims = normalizer.get_output_dims()
    logger.info(
        f"Tower dimensions after normalization: "
        f"rhythm={output_dims['rhythm']}, "
        f"timbre={output_dims['timbre']}, "
        f"harmony={output_dims['harmony']}"
    )

    # Transform all segments
    logger.info("Normalizing all segments...")
    X_r_full, X_t_full, X_h_full = normalizer.transform(
        X_rhythm_full, X_timbre_full, X_harmony_full
    )
    X_r_start, X_t_start, X_h_start = normalizer.transform(
        X_rhythm_start, X_timbre_start, X_harmony_start
    )
    X_r_mid, X_t_mid, X_h_mid = normalizer.transform(
        X_rhythm_mid, X_timbre_mid, X_harmony_mid
    )
    X_r_end, X_t_end, X_h_end = normalizer.transform(
        X_rhythm_end, X_timbre_end, X_harmony_end
    )

    # Compute calibration statistics
    logger.info("Computing calibration statistics...")
    tower_calibration = compute_tower_calibration_stats(
        X_r_full, X_t_full, X_h_full,
        n_pairs=min(10000, len(tracks) * 5),
        random_seed=args.random_seed,
    )
    logger.info(
        f"Calibration: rhythm(mean={tower_calibration['rhythm']['random_mean']:.4f}, "
        f"std={tower_calibration['rhythm']['random_std']:.4f}), "
        f"timbre(mean={tower_calibration['timbre']['random_mean']:.4f}, "
        f"std={tower_calibration['timbre']['random_std']:.4f}), "
        f"harmony(mean={tower_calibration['harmony']['random_mean']:.4f}, "
        f"std={tower_calibration['harmony']['random_std']:.4f})"
    )

    # Build concatenated sonic matrix (backward compatibility)
    X_sonic = np.hstack([X_r_full, X_t_full, X_h_full])
    X_sonic = l2_normalize(X_sonic)  # L2 normalize the concatenation

    # Similarly for segments
    X_sonic_start = l2_normalize(np.hstack([X_r_start, X_t_start, X_h_start]))
    X_sonic_mid = l2_normalize(np.hstack([X_r_mid, X_t_mid, X_h_mid]))
    X_sonic_end = l2_normalize(np.hstack([X_r_end, X_t_end, X_h_end]))

    # Load genres
    logger.info("Loading genre information...")
    track_ids = [t["track_id"] for t in tracks]
    genre_lists, vocab = load_genres_for_tracks(args.db_path, track_ids)
    X_genre_raw, X_genre_smoothed = build_genre_matrices(
        genre_lists, vocab, args.genre_sim_path
    )
    logger.info(f"Genre vocabulary size: {len(vocab)}")

    # Prepare metadata
    artist_keys = [
        normalize_artist_key(t["norm_artist"], t["track_id"]) for t in tracks
    ]
    track_artists = [t["artist"] for t in tracks]
    track_titles = [t["title"] for t in tracks]

    # Create feature names
    rhythm_names = [f"rhythm_{i:02d}" for i in range(X_r_full.shape[1])]
    timbre_names = [f"timbre_{i:02d}" for i in range(X_t_full.shape[1])]
    harmony_names = [f"harmony_{i:02d}" for i in range(X_h_full.shape[1])]
    feature_names = rhythm_names + timbre_names + harmony_names

    # Save artifact
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving artifact to {out_path}...")
    np.savez(
        out_path,
        # Per-tower embeddings (full)
        X_sonic_rhythm=X_r_full,
        X_sonic_timbre=X_t_full,
        X_sonic_harmony=X_h_full,
        # Per-tower embeddings (start segment)
        X_sonic_rhythm_start=X_r_start,
        X_sonic_timbre_start=X_t_start,
        X_sonic_harmony_start=X_h_start,
        # Per-tower embeddings (mid segment)
        X_sonic_rhythm_mid=X_r_mid,
        X_sonic_timbre_mid=X_t_mid,
        X_sonic_harmony_mid=X_h_mid,
        # Per-tower embeddings (end segment)
        X_sonic_rhythm_end=X_r_end,
        X_sonic_timbre_end=X_t_end,
        X_sonic_harmony_end=X_h_end,
        # Concatenated embeddings (backward compatibility)
        X_sonic=X_sonic,
        X_sonic_start=X_sonic_start,
        X_sonic_mid=X_sonic_mid,
        X_sonic_end=X_sonic_end,
        # Feature metadata
        sonic_feature_names=np.array(feature_names, dtype=object),
        tower_dims=np.array([output_dims['rhythm'], output_dims['timbre'], output_dims['harmony']]),
        # Calibration
        tower_calibration=tower_calibration,
        # Normalizer params (for reproducibility)
        normalizer_params=normalizer.get_params(),
        # BPM array
        bpm_array=np.array(bpm_list, dtype=np.float32),
        # Genre matrices
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=np.array(vocab, dtype=object),
        # Track metadata
        track_ids=np.array(track_ids, dtype=object),
        track_artists=np.array(track_artists, dtype=object),
        track_titles=np.array(track_titles, dtype=object),
        artist_keys=np.array(artist_keys, dtype=object),
        # Build metadata
        build_config={
            'clip_sigma': args.clip_sigma,
            'use_pca_whitening': not args.no_pca,
            'pca_variance_retain': args.pca_variance,
            'random_seed': args.random_seed,
            'extraction_method': 'beat3tower',
        },
    )

    logger.info(
        f"Artifact saved successfully: "
        f"{len(tracks)} tracks, "
        f"{len(vocab)} genres, "
        f"{X_sonic.shape[1]} sonic dims "
        f"({output_dims['rhythm']}+{output_dims['timbre']}+{output_dims['harmony']})"
    )


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        build_artifacts(args)
    except Exception as e:
        logger.error(f"Build failed: {e}")
        raise


if __name__ == "__main__":
    main()
