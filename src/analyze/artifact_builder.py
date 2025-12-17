from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config_loader import Config
from src.similarity_calculator import SimilarityCalculator
from src.features.beat3tower_types import Beat3TowerFeatures

logger = logging.getLogger(__name__)


def _extract_beat3tower_vector(features: Dict[str, Any], segment_key: str = 'full') -> Optional[np.ndarray]:
    """
    Extract beat3tower feature vector from JSON.

    Args:
        features: Sonic features dict
        segment_key: Which segment to extract ('full', 'start', 'mid', 'end')

    Returns:
        137-dimensional beat3tower vector or None if not beat3tower format
    """
    segment_data = features.get(segment_key)
    if not segment_data:
        return None

    # Check if this is beat3tower format
    if segment_data.get('extraction_method') != 'beat3tower':
        return None

    try:
        # Convert to Beat3TowerFeatures and extract vector
        b3t_features = Beat3TowerFeatures.from_dict(segment_data)
        return b3t_features.to_vector()
    except Exception as e:
        logger.debug(f"Failed to extract beat3tower vector: {e}")
        return None


def _derive_schema(layout: Dict[str, Dict[str, Any]], dim: int) -> tuple[list[str], list[str]]:
    names: list[str] = []
    units: list[str] = []

    def _append(name: str, length: int, label_template: str, unit: str) -> None:
        for i in range(length):
            names.append(label_template.format(i + 1))
            units.append(unit)

    mfcc_len = int(layout.get("mfcc_mean", {}).get("length") or 0)
    chroma_len = int(layout.get("chroma", {}).get("length") or 0)
    bpm_len = int(layout.get("bpm", {}).get("length") or 0)
    spec_len = int(layout.get("spectral_centroid", {}).get("length") or 0)

    _append("mfcc", mfcc_len, "mfcc_{:02d}", "mfcc_coeff")
    _append("chroma", chroma_len, "chroma_{:02d}", "chroma_bin")
    _append("bpm", bpm_len, "bpm", "bpm")
    _append("spectral_centroid", spec_len, "spectral_centroid", "hz")

    if len(names) < dim:
        for i in range(len(names), dim):
            names.append(f"dim_{i:02d}")
            units.append("")
    elif len(names) > dim:
        names = names[:dim]
        units = units[:dim]
    return names, units


@dataclass(frozen=True)
class ArtifactBuildResult:
    out_path: Path
    n_tracks: int
    n_genres: int
    stats: dict


def _normalize_artist_key(raw: Optional[str], track_id: Optional[str]) -> str:
    """Normalize artist identifier with fallbacks."""
    import re

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


def _smooth_genres(X: np.ndarray, S: Optional[np.ndarray], vocab: List[str], sim_vocab: Optional[List[str]]) -> np.ndarray:
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


def build_ds_artifacts(
    *,
    db_path: str,
    config_path: str,
    out_path: str | Path,
    genre_sim_path: Optional[str | Path] = None,
    max_tracks: int = 0,
    random_seed: int = 0,
) -> ArtifactBuildResult:
    """
    - Query tracks + sonic_features + norm_artist
    - Use the same sonic feature vector builder used by production (SimilarityCalculator helper)
    - Build X_sonic and if available X_sonic_start/mid/end (segment-aware)
    - Build X_genre_raw from MB/Discogs sources only per DS plan
    - If genre_sim_path provided, compute X_genre_smoothed using S; else identity fallback
    - Save NPZ with required keys, matching what the DS playlist pipeline loader expects
    """
    cfg = Config(config_path)
    rng = np.random.default_rng(random_seed)
    calc = SimilarityCalculator(db_path=db_path, config=cfg.config)
    cursor = calc.conn.cursor()

    limit_clause = ""
    if max_tracks and max_tracks > 0:
        limit_clause = f" LIMIT {int(max_tracks)}"
    # Build select with fallbacks for missing columns
    has_norm_artist = _column_exists(cursor, "tracks", "norm_artist")
    if not _column_exists(cursor, "tracks", "sonic_features"):
        raise RuntimeError("tracks table missing sonic_features column; cannot build artifacts.")
    norm_artist_expr = "norm_artist" if has_norm_artist else "artist"
    cursor.execute(
        f"""
        SELECT track_id, artist, title, album, {norm_artist_expr} as norm_artist, sonic_features
        FROM tracks
        WHERE sonic_features IS NOT NULL
        {limit_clause}
        """
    )
    rows = cursor.fetchall()
    if not rows:
        raise RuntimeError("No tracks with sonic_features found; cannot build artifacts.")

    track_ids: List[str] = []
    artist_keys: List[str] = []
    track_artists: List[str] = []
    track_titles: List[str] = []
    X_sonic: List[np.ndarray] = []
    X_sonic_start: List[np.ndarray] = []
    X_sonic_mid: List[np.ndarray] = []
    X_sonic_end: List[np.ndarray] = []
    genre_lists: List[List[str]] = []

    for row in rows:
        track_id = row["track_id"]
        artist = row["artist"] or ""
        title = row["title"] or ""
        norm_artist = row["norm_artist"] or artist
        raw = row["sonic_features"]
        try:
            features = json.loads(raw)
        except Exception:
            continue

        # Try beat3tower extraction first
        base_vec = _extract_beat3tower_vector(features, 'full')
        if base_vec is not None:
            # Beat3tower format - extract all segments
            start_vec = _extract_beat3tower_vector(features, 'start')
            if start_vec is None:
                start_vec = base_vec
            mid_vec = _extract_beat3tower_vector(features, 'mid')
            if mid_vec is None:
                mid_vec = base_vec
            end_vec = _extract_beat3tower_vector(features, 'end')
            if end_vec is None:
                end_vec = base_vec
        else:
            # Fall back to old extraction method
            base_vec = calc.build_sonic_feature_vector(features)
            if base_vec.size == 0:
                continue
            start_vec = calc.build_sonic_feature_vector_by_segment(features, "start")
            mid_vec = calc.build_sonic_feature_vector_by_segment(features, "mid")
            end_vec = calc.build_sonic_feature_vector_by_segment(features, "end")
            if start_vec.shape != base_vec.shape:
                start_vec = base_vec
            if mid_vec.shape != base_vec.shape:
                mid_vec = base_vec
            if end_vec.shape != base_vec.shape:
                end_vec = base_vec

        genres = calc.get_filtered_combined_genres_for_track(track_id) or []
        # CHANGED: Include tracks even with empty genres (don't continue/skip)
        # Empty genre vectors will have genre_sim=0.0 and be excluded by hard gates if min_genre_similarity > 0
        # This allows sonic-only modes and discover mode penalty scoring to work

        track_ids.append(track_id)
        artist_keys.append(_normalize_artist_key(norm_artist, track_id))
        track_artists.append(artist)
        track_titles.append(title)
        X_sonic.append(base_vec)
        X_sonic_start.append(start_vec)
        X_sonic_mid.append(mid_vec)
        X_sonic_end.append(end_vec)
        genre_lists.append(genres)

    if not track_ids:
        raise RuntimeError("No usable tracks after filtering; artifacts not created.")

    # Filter for consistent sonic dimensions (handle legacy mixed-format data)
    dims = [vec.shape[0] for vec in X_sonic]
    from collections import Counter
    dim_counts = Counter(dims)
    logger.info(f"Sonic dimension distribution: {dict(dim_counts)}")

    # PREFER beat3tower (137 dims) if available, otherwise use most common
    if 137 in dim_counts:
        target_dim = 137
        logger.info(f"Found beat3tower features (137 dims): {dim_counts[137]} tracks - using these")
    else:
        target_dim = dim_counts.most_common(1)[0][0]
        logger.info(f"No beat3tower features found, using most common dimension: {target_dim} ({dim_counts[target_dim]} tracks)")

    filtered_indices = [i for i, vec in enumerate(X_sonic) if vec.shape[0] == target_dim]
    if len(filtered_indices) < len(track_ids):
        dropped = len(track_ids) - len(filtered_indices)
        logger.warning(f"Dropping {dropped} tracks with inconsistent sonic dimensions")

    track_ids = [track_ids[i] for i in filtered_indices]
    artist_keys = [artist_keys[i] for i in filtered_indices]
    track_artists = [track_artists[i] for i in filtered_indices]
    track_titles = [track_titles[i] for i in filtered_indices]
    X_sonic = [X_sonic[i] for i in filtered_indices]
    X_sonic_start = [X_sonic_start[i] for i in filtered_indices]
    X_sonic_mid = [X_sonic_mid[i] for i in filtered_indices]
    X_sonic_end = [X_sonic_end[i] for i in filtered_indices]
    genre_lists = [genre_lists[i] for i in filtered_indices]

    if not track_ids:
        raise RuntimeError("No tracks remain after dimension filtering.")

    # Build genre vocab
    vocab_set: Dict[str, int] = {}
    for genres in genre_lists:
        for g in genres:
            vocab_set[g] = vocab_set.get(g, 0) + 1
    vocab = sorted(vocab_set.keys())
    vocab_index = {g: i for i, g in enumerate(vocab)}

    X_genre_raw = np.zeros((len(track_ids), len(vocab)), dtype=np.float32)
    for i, genres in enumerate(genre_lists):
        for g in genres:
            j = vocab_index.get(g)
            if j is not None:
                X_genre_raw[i, j] = 1.0

    # Optional smoothing
    X_genre_smoothed = X_genre_raw.copy()
    if genre_sim_path:
        sim_npz = np.load(genre_sim_path, allow_pickle=True)
        sim_vocab = sim_npz["genre_vocab"].tolist()
        S = sim_npz["S"]
        X_genre_smoothed = _smooth_genres(X_genre_raw, S, vocab, sim_vocab)

    X_sonic_arr = np.vstack(X_sonic)
    X_sonic_start_arr = np.vstack(X_sonic_start)
    X_sonic_mid_arr = np.vstack(X_sonic_mid)
    X_sonic_end_arr = np.vstack(X_sonic_end)

    layout = calc._sonic_feature_layout or {}
    names, units = _derive_schema(layout, X_sonic_arr.shape[1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X_sonic=X_sonic_arr,
        X_sonic_start=X_sonic_start_arr,
        X_sonic_mid=X_sonic_mid_arr,
        X_sonic_end=X_sonic_end_arr,
        sonic_feature_names=np.array(names, dtype=object),
        sonic_feature_units=np.array(units, dtype=object),
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        track_ids=np.array(track_ids),
        track_artists=np.array(track_artists),
        track_titles=np.array(track_titles),
        artist_keys=np.array(artist_keys),
        genre_vocab=np.array(vocab),
    )

    stats = {
        "tracks_kept": len(track_ids),
        "genres_kept": len(vocab),
        "random_seed": random_seed,
    }
    logger.info(
        "Saved DS artifact to %s tracks=%d genres=%d dims=%s",
        out_path,
        len(track_ids),
        len(vocab),
        np.vstack(X_sonic).shape,
    )
    calc.close()
    return ArtifactBuildResult(out_path=out_path, n_tracks=len(track_ids), n_genres=len(vocab), stats=stats)
def _column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())
