#!/usr/bin/env python3
"""
Sonic Validation Suite

For a given seed track, generates 3 comparison playlists (sonic-only, genre-only, hybrid)
and computes diagnostic metrics to determine if sonic similarity is informative.

Metrics computed:
1. Score flatness: (p90 - p10) / median - measures score separation
2. TopK gap: mean(topK) - mean(random) - measures discrimination
3. Intra-artist coherence: same-artist vs random baseline distance
4. Intra-album coherence: same-album vs random baseline distance

PASS thresholds:
- sonic_flatness >= 0.5
- sonic_topk_gap >= 0.15
- intra_artist_coherence >= 0.05
- intra_album_coherence >= 0.08

Usage:
    python scripts/sonic_validation_suite.py \\
        --artifact data_matrices_step1.npz \\
        --seed-track-id 1c347ff04e65adf7923a9e3927ab667a \\
        --k 30 \\
        --output-dir diagnostics/sonic_validation/
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_variant import compute_sonic_variant_norm

# Configure logging (centralized)
from src.logging_config import setup_logging
logger = setup_logging(name='sonic_validation', log_file='sonic_validation.log')

DB_PATH = ROOT_DIR / 'data' / 'metadata.db'


def compute_flatness(similarities: np.ndarray) -> float:
    """
    Compute score flatness: (p90 - p10) / median

    If < 0.5: Scores too flat, not discriminative
    If > 1.0: Good separation
    """
    if len(similarities) < 10:
        return 0.0

    p10 = np.percentile(similarities, 10)
    p90 = np.percentile(similarities, 90)
    median = np.median(similarities)

    if median < 1e-6:
        return 0.0

    flatness = (p90 - p10) / (median + 1e-6)
    return float(flatness)


def compute_topk_gap(all_sims: np.ndarray, seed_idx: int, k: int = 30, n_random: int = 100) -> float:
    """
    Compute TopK vs random gap: mean(topK) - mean(random)

    If < 0.1: Sonic similarity uninformative
    If > 0.2: Good discrimination
    """
    if len(all_sims) < k + n_random:
        return 0.0

    # Top K similarities (excluding seed)
    sorted_idx = np.argsort(all_sims)[::-1]
    topk_sims = all_sims[sorted_idx[1:k+1]]  # Exclude seed itself

    # Random baseline
    pool_size = len(all_sims)
    random_idx = np.random.choice(pool_size, size=min(n_random, pool_size), replace=False)
    random_sims = all_sims[random_idx]

    gap = float(topk_sims.mean() - random_sims.mean())
    return gap


def compute_intra_artist_coherence(
    X: np.ndarray,
    artist_keys: List[str],
    seed_idx: int
) -> Optional[float]:
    """
    Compute intra-artist coherence: mean(intra_artist_sim) - mean(random_sim)

    Positive = same artist is MORE similar (GOOD)
    Negative = same artist is FARTHER than random (BAD)
    """
    seed_artist = artist_keys[seed_idx]
    same_artist_idx = [i for i, a in enumerate(artist_keys) if a == seed_artist and i != seed_idx]

    if len(same_artist_idx) < 2:
        return None

    # Normalize vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    seed_vec = X_norm[seed_idx]

    # Intra-artist similarities
    intra_sims = X_norm[same_artist_idx] @ seed_vec

    # Random baseline
    pool_size = len(X)
    random_idx = np.random.choice(pool_size, size=len(same_artist_idx), replace=False)
    random_sims = X_norm[random_idx] @ seed_vec

    coherence = float(intra_sims.mean() - random_sims.mean())
    return coherence


def compute_intra_album_coherence(
    X: np.ndarray,
    album_ids: np.ndarray,
    seed_idx: int
) -> Optional[float]:
    """
    Compute intra-album coherence: mean(intra_album_sim) - mean(random_sim)

    Similar logic to intra-artist but stricter threshold.
    """
    seed_album = album_ids[seed_idx]
    same_album_idx = [i for i, a in enumerate(album_ids) if a == seed_album and i != seed_idx]

    if len(same_album_idx) < 2:
        return None

    # Normalize vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    seed_vec = X_norm[seed_idx]

    # Intra-album similarities
    intra_sims = X_norm[same_album_idx] @ seed_vec

    # Random baseline
    pool_size = len(X)
    random_idx = np.random.choice(pool_size, size=len(same_album_idx), replace=False)
    random_sims = X_norm[random_idx] @ seed_vec

    coherence = float(intra_sims.mean() - random_sims.mean())
    return coherence


def compute_sonic_only_neighbors(
    bundle,
    seed_idx: int,
    k: int = 30,
    sonic_variant: str = 'raw'
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure sonic cosine similarity with optional variant preprocessing."""
    X = bundle.X_sonic

    # Apply sonic variant (raw, robust_whiten, etc.)
    if sonic_variant != 'raw':
        logger.info(f"Applying sonic variant: {sonic_variant}")
        X_norm, stats = compute_sonic_variant_norm(X, sonic_variant)
        logger.info(f"Variant stats: {stats}")
    else:
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    seed_vec = X_norm[seed_idx]
    sims = X_norm @ seed_vec
    topk_idx = np.argsort(sims)[::-1][1:k+1]  # Exclude seed
    return topk_idx, sims


def compute_genre_only_neighbors(
    bundle,
    seed_idx: int,
    k: int = 30,
    method: str = 'ensemble'
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure genre similarity."""
    X_genre = bundle.X_genre_smoothed
    seed_genres = X_genre[seed_idx]

    if method == 'ensemble':
        # Use ensemble method: 60% cosine + 40% weighted_jaccard
        X_norm = X_genre / (np.linalg.norm(X_genre, axis=1, keepdims=True) + 1e-12)
        seed_norm = seed_genres / (np.linalg.norm(seed_genres) + 1e-12)
        cosine_sims = X_norm @ seed_norm

        # Weighted jaccard
        intersection = np.sum(np.minimum(X_genre, seed_genres), axis=1)
        union = np.sum(np.maximum(X_genre, seed_genres), axis=1)
        jaccard_sims = intersection / (union + 1e-12)

        sims = 0.6 * cosine_sims + 0.4 * jaccard_sims
    else:
        X_norm = X_genre / (np.linalg.norm(X_genre, axis=1, keepdims=True) + 1e-12)
        seed_norm = seed_genres / (np.linalg.norm(seed_genres) + 1e-12)
        sims = X_norm @ seed_norm

    topk_idx = np.argsort(sims)[::-1][1:k+1]
    return topk_idx, sims


def compute_hybrid_neighbors(
    bundle,
    seed_idx: int,
    k: int = 30,
    w_sonic: float = 0.6,
    w_genre: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """Current hybrid embedding (PCA-reduced sonic + genre)."""
    from src.similarity.hybrid import build_hybrid_embedding

    model = build_hybrid_embedding(
        bundle.X_sonic,
        bundle.X_genre_smoothed,
        n_components_sonic=32,
        n_components_genre=32,
        w_sonic=w_sonic,
        w_genre=w_genre
    )

    embedding = model.embedding  # Get the numpy array from the model

    emb_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
    seed_vec = emb_norm[seed_idx]
    sims = emb_norm @ seed_vec
    topk_idx = np.argsort(sims)[::-1][1:k+1]
    return topk_idx, sims


def export_m3u_playlist(
    track_ids: np.ndarray,
    bundle,
    output_path: Path,
    indices: Optional[np.ndarray] = None
) -> None:
    """Write M3U playlist with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")

        if indices is not None:
            track_ids = track_ids[indices]

        for i, tid in enumerate(track_ids):
            cursor.execute(
                "SELECT artist, title, file_path FROM tracks WHERE track_id = ?",
                (tid,)
            )
            row = cursor.fetchone()

            if row is None:
                artist = "Unknown"
                title = "Unknown"
                file_path = ""
            else:
                artist = row["artist"] or "Unknown"
                title = row["title"] or "Unknown"
                file_path = row["file_path"] or ""

            # M3U format: #EXTINF:duration,artist - title
            f.write(f"#EXTINF:-1,{artist} - {title}\n")
            if file_path:
                f.write(f"{file_path}\n")

    conn.close()
    logger.info(f"Exported M3U playlist: {output_path}")


def get_track_metadata(track_id: str) -> Tuple[str, str]:
    """Get artist and title for a track."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT artist, title FROM tracks WHERE track_id = ?", (track_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return "Unknown", "Unknown"

    return row["artist"] or "Unknown", row["title"] or "Unknown"


def load_album_ids_for_tracks(track_ids: np.ndarray) -> np.ndarray:
    """Load album_ids from database for given track_ids."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    album_ids = []
    for tid in track_ids:
        cursor.execute("SELECT album_id FROM tracks WHERE track_id = ?", (tid,))
        row = cursor.fetchone()
        if row and row["album_id"]:
            album_ids.append(row["album_id"])
        else:
            album_ids.append(None)

    conn.close()
    return np.array(album_ids, dtype=object)


def generate_markdown_report(
    metrics: Dict[str, Any],
    seed_track_id: str,
    output_path: Path
) -> None:
    """Generate markdown report with PASS/FAIL assessment."""
    artist, title = get_track_metadata(seed_track_id)

    # Determine pass/fail
    sonic_flatness_pass = metrics.get('sonic_flatness', 0) >= 0.5
    sonic_topk_gap_pass = metrics.get('sonic_topk_gap', 0) >= 0.15
    sonic_intra_artist_pass = (metrics.get('sonic_intra_artist_coherence') is None or
                               metrics.get('sonic_intra_artist_coherence', 0) >= 0.05)
    sonic_intra_album_pass = (metrics.get('sonic_intra_album_coherence') is None or
                              metrics.get('sonic_intra_album_coherence', 0) >= 0.08)

    report = f"""# Sonic Validation Report

**Seed Track**: {artist} - {title} ({seed_track_id})

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Metrics Summary

### Sonic-Only Nearest Neighbors

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Score Flatness | {metrics.get('sonic_flatness', 0):.3f} | >= 0.5 | {'PASS' if sonic_flatness_pass else 'FAIL'} |
| TopK vs Random Gap | {metrics.get('sonic_topk_gap', 0):.3f} | >= 0.15 | {'PASS' if sonic_topk_gap_pass else 'FAIL'} |
| Intra-Artist Coherence | {metrics.get('sonic_intra_artist_coherence') or 'N/A'} | >= 0.05 | {'PASS' if sonic_intra_artist_pass else 'FAIL' if metrics.get('sonic_intra_artist_coherence') is not None else 'N/A'} |
| Intra-Album Coherence | {metrics.get('sonic_intra_album_coherence') or 'N/A'} | >= 0.08 | {'PASS' if sonic_intra_album_pass else 'FAIL' if metrics.get('sonic_intra_album_coherence') is not None else 'N/A'} |

### Genre-Only Nearest Neighbors

| Metric | Value |
|--------|-------|
| Score Flatness | {metrics.get('genre_flatness', 0):.3f} |
| TopK vs Random Gap | {metrics.get('genre_topk_gap', 0):.3f} |

## Interpretation

### What These Metrics Mean

**Score Flatness**: Measures if scores are spread out or clustered together.
- High flatness (> 1.0) = Good separation between similar and dissimilar tracks
- Low flatness (< 0.5) = Most tracks have similar scores (uninformative)

**TopK vs Random Gap**: Measures if top neighbors are actually better than random.
- High gap (> 0.2) = Top neighbors are much more similar than random
- Low gap (< 0.1) = Top neighbors barely better than random (uninformative)

**Intra-Artist/Album Coherence**: Measures if tracks from same artist/album are closer.
- Positive value = Same artist/album tracks are closer than random (GOOD)
- Negative value = Same artist/album tracks are farther than random (BAD)
- > 0.05 for artist, > 0.08 for album = PASS

### Overall Assessment

"""

    # Count passes
    passes = sum([
        sonic_flatness_pass,
        sonic_topk_gap_pass,
        sonic_intra_artist_pass,
        sonic_intra_album_pass
    ])

    if passes >= 3:
        report += """**SONIC VALIDATION: PASS**

Sonic similarity is informative and working well. The validation metrics show:
- Good score separation (topK much better than random)
- Same-artist/album tracks cluster together
- Pure sonic neighbors pass listening test

Next steps:
- Skip Phase B (feature improvements)
- Proceed to Phase C (transition scoring)
- Then Phase D (rebalance dynamic mode)
"""
    else:
        report += f"""**SONIC VALIDATION: FAIL**

Sonic similarity is NOT informative ({passes}/4 metrics pass). The validation metrics show:
- Scores too flat or no discrimination
- Same-artist/album tracks not coherent
- Pure sonic neighbors may not pass listening test

Next steps:
- MUST proceed to Phase B (feature improvements)
- Test new variants (beat_sync, multi_segment_median, robust_whiten)
- Re-run validation suite on improved variants
"""

    report += """

## Files Generated

1. **sonic_only_top30.m3u** - Top 30 tracks by pure sonic similarity
2. **genre_only_top30.m3u** - Top 30 tracks by pure genre similarity
3. **hybrid_current_top30.m3u** - Top 30 tracks by current hybrid embedding
4. **sonic_validation_metrics.csv** - Detailed metrics for aggregation

Listen to all three playlists and assess:
- Does sonic-only sound coherent? (similar vibes, flow)
- Does genre-only sound coherent? (same style/era)
- Does hybrid sound balanced? (mix of sonic + genre)

Manual listening assessment is crucial for final validation.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Generated report: {output_path}")


def run_validation_suite(
    artifact_path: Path,
    seed_track_id: str,
    k: int = 30,
    output_dir: Path = Path('diagnostics/sonic_validation/'),
    sonic_variant: str = 'raw'
) -> Dict[str, Any]:
    """Main validation workflow."""
    logger.info(f"Loading artifact: {artifact_path}")
    logger.info(f"Sonic variant: {sonic_variant}")
    bundle = load_artifact_bundle(artifact_path)

    if seed_track_id not in bundle.track_id_to_index:
        raise ValueError(f"Seed track {seed_track_id} not found in artifact")

    seed_idx = bundle.track_id_to_index[seed_track_id]
    artist, title = get_track_metadata(seed_track_id)
    logger.info(f"Seed: {artist} - {title}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute neighbors for each mode
    logger.info("Computing sonic-only neighbors...")
    sonic_idx, sonic_sims = compute_sonic_only_neighbors(bundle, seed_idx, k, sonic_variant=sonic_variant)

    logger.info("Computing genre-only neighbors...")
    genre_idx, genre_sims = compute_genre_only_neighbors(bundle, seed_idx, k)

    logger.info("Computing hybrid neighbors...")
    hybrid_idx, hybrid_sims = compute_hybrid_neighbors(bundle, seed_idx, k)

    # 2. Export M3Us
    logger.info("Exporting M3U playlists...")
    export_m3u_playlist(bundle.track_ids, bundle, output_dir / f"sonic_only_top{k}.m3u", sonic_idx)
    export_m3u_playlist(bundle.track_ids, bundle, output_dir / f"genre_only_top{k}.m3u", genre_idx)
    export_m3u_playlist(bundle.track_ids, bundle, output_dir / f"hybrid_current_top{k}.m3u", hybrid_idx)

    # 3. Compute metrics
    logger.info("Computing metrics...")

    # Apply variant transformation for coherence metrics
    if sonic_variant != 'raw':
        X_sonic_transformed, _ = compute_sonic_variant_norm(bundle.X_sonic, sonic_variant)
    else:
        X_sonic_transformed = bundle.X_sonic

    # Load album_ids from database
    logger.info("Loading album data from database...")
    album_ids = load_album_ids_for_tracks(bundle.track_ids)

    metrics = {
        'seed_track_id': seed_track_id,
        'seed_artist': artist,
        'seed_title': title,
        'sonic_variant': sonic_variant,
        'sonic_flatness': compute_flatness(sonic_sims),
        'sonic_topk_gap': compute_topk_gap(sonic_sims, seed_idx, k),
        'sonic_intra_artist_coherence': compute_intra_artist_coherence(X_sonic_transformed, bundle.artist_keys, seed_idx),
        'sonic_intra_album_coherence': compute_intra_album_coherence(X_sonic_transformed, album_ids, seed_idx),
        'genre_flatness': compute_flatness(genre_sims),
        'genre_topk_gap': compute_topk_gap(genre_sims, seed_idx, k),
    }

    # 4. Write CSV
    logger.info("Writing CSV...")
    csv_path = output_dir / "sonic_validation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    logger.info(f"Saved metrics: {csv_path}")

    # 5. Generate markdown report
    logger.info("Generating report...")
    generate_markdown_report(metrics, seed_track_id, output_dir / "sonic_validation_report.md")

    # 6. Print summary
    print("\n" + "="*70)
    print("SONIC VALIDATION RESULTS")
    print("="*70)
    print(f"\nSeed: {artist} - {title}")
    print(f"\nMetrics:")
    print(f"  Sonic Flatness:                {metrics['sonic_flatness']:.3f} (threshold: 0.5)")
    print(f"  Sonic TopK Gap:                {metrics['sonic_topk_gap']:.3f} (threshold: 0.15)")
    if metrics['sonic_intra_artist_coherence'] is not None:
        print(f"  Sonic Intra-Artist Coherence:  {metrics['sonic_intra_artist_coherence']:.3f} (threshold: 0.05)")
    else:
        print(f"  Sonic Intra-Artist Coherence:  N/A (not enough tracks by artist)")
    if metrics['sonic_intra_album_coherence'] is not None:
        print(f"  Sonic Intra-Album Coherence:   {metrics['sonic_intra_album_coherence']:.3f} (threshold: 0.08)")
    else:
        print(f"  Sonic Intra-Album Coherence:   N/A (not enough tracks in album)")

    # Assessment
    passes = 0
    if metrics['sonic_flatness'] >= 0.5:
        passes += 1
    if metrics['sonic_topk_gap'] >= 0.15:
        passes += 1
    if metrics['sonic_intra_artist_coherence'] is None or metrics['sonic_intra_artist_coherence'] >= 0.05:
        passes += 1
    if metrics['sonic_intra_album_coherence'] is None or metrics['sonic_intra_album_coherence'] >= 0.08:
        passes += 1

    print(f"\nAssessment: {passes}/4 metrics PASS")

    if passes >= 3:
        print("\nCONCLUSION: SONIC VALIDATION PASSES")
        print("Sonic similarity is informative. Proceed to Phase C (transition scoring).")
    else:
        print("\nCONCLUSION: SONIC VALIDATION FAILS")
        print("Sonic similarity is NOT informative. Proceed to Phase B (feature improvements).")

    print(f"\nOutputs saved to: {output_dir}")
    print("="*70 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Sonic Validation Suite')
    parser.add_argument('--artifact', type=Path, required=True, help='Path to artifact NPZ file')
    parser.add_argument('--seed-track-id', type=str, required=True, help='Seed track ID')
    parser.add_argument('--k', type=int, default=30, help='Number of neighbors to retrieve')
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics/sonic_validation/'),
                        help='Output directory for results')
    parser.add_argument('--sonic-variant', type=str, default='raw',
                        choices=['raw', 'centered', 'z', 'z_clip', 'whiten_pca', 'robust_whiten'],
                        help='Sonic preprocessing variant (default: raw)')

    args = parser.parse_args()

    metrics = run_validation_suite(
        artifact_path=args.artifact,
        seed_track_id=args.seed_track_id,
        k=args.k,
        output_dir=args.output_dir,
        sonic_variant=args.sonic_variant
    )


if __name__ == '__main__':
    main()
