#!/usr/bin/env python3
"""
Sonic Quality Validation Suite (Phase 0)
=========================================

Validates that sonic features are discriminative (not flat/collapsed).

Metrics computed:
1. Per-dimension variance (are dimensions informative?)
2. Unique vector percentage (are vectors distinct?)
3. Random-pair similarity distribution (baseline)
4. TopK vs Random gap (discrimination power)
5. Within-artist coherence (same artist should be closer)
6. Sonic flatness (p90-p10 spread)

Usage:
    python scripts/validate_sonic_quality.py \\
        --artifact data_matrices.npz \\
        --seeds "track_id1,track_id2" \\
        --output diagnostics/sonic_validation/

    python scripts/validate_sonic_quality.py --help
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.artifacts import load_artifact_bundle, ArtifactBundle
from src.logging_config import setup_logging

logger = setup_logging(name='validate_sonic_quality', log_file='validate_sonic_quality.log')


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_dimension_variance(X: np.ndarray) -> Dict[str, Any]:
    """
    Compute per-dimension variance statistics.

    A healthy embedding should have:
    - All dimensions with variance > 1e-6 (not constant)
    - Median variance > 1e-3 (reasonably spread)
    """
    variances = np.var(X, axis=0)

    return {
        'n_dimensions': X.shape[1],
        'variance_min': float(np.min(variances)),
        'variance_max': float(np.max(variances)),
        'variance_median': float(np.median(variances)),
        'variance_mean': float(np.mean(variances)),
        'n_zero_variance': int(np.sum(variances < 1e-8)),
        'n_low_variance': int(np.sum(variances < 1e-4)),
        'pass': bool(np.min(variances) > 1e-6 and np.median(variances) > 1e-4),
    }


def compute_unique_vectors(X: np.ndarray, precision: int = 4) -> Dict[str, Any]:
    """
    Compute percentage of unique vectors (rounded to precision decimal places).

    If all tracks collapse to similar vectors, unique % will be low.
    """
    # Round to precision and hash
    X_rounded = np.round(X, decimals=precision)

    hashes = set()
    for row in X_rounded:
        h = hashlib.md5(row.tobytes()).hexdigest()
        hashes.add(h)

    unique_pct = len(hashes) / X.shape[0] * 100

    return {
        'n_tracks': X.shape[0],
        'n_unique': len(hashes),
        'unique_pct': float(unique_pct),
        'precision_decimals': precision,
        'pass': bool(unique_pct > 95.0),
    }


def compute_random_similarity_distribution(
    X: np.ndarray,
    n_pairs: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute similarity distribution for random pairs.

    A healthy embedding should have:
    - Mean random similarity well below 1.0 (ideally < 0.8)
    - Non-trivial std (> 0.02)
    """
    rng = np.random.default_rng(seed)
    n_tracks = X.shape[0]

    # L2 normalize for cosine similarity
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # Sample random pairs
    idx1 = rng.integers(0, n_tracks, size=n_pairs)
    idx2 = rng.integers(0, n_tracks, size=n_pairs)

    # Avoid self-pairs
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Compute similarities
    sims = np.sum(X_norm[idx1] * X_norm[idx2], axis=1)

    return {
        'n_pairs': len(sims),
        'mean': float(np.mean(sims)),
        'std': float(np.std(sims)),
        'min': float(np.min(sims)),
        'max': float(np.max(sims)),
        'p10': float(np.percentile(sims, 10)),
        'p25': float(np.percentile(sims, 25)),
        'p50': float(np.percentile(sims, 50)),
        'p75': float(np.percentile(sims, 75)),
        'p90': float(np.percentile(sims, 90)),
        'pass': bool(np.mean(sims) < 0.95 and np.std(sims) > 0.02),
    }


def compute_all_similarities(X: np.ndarray, seed_idx: int) -> np.ndarray:
    """Compute cosine similarity from seed to all tracks."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    seed_vec = X_norm[seed_idx]
    sims = X_norm @ seed_vec
    return sims


def compute_sonic_flatness(sims: np.ndarray) -> Dict[str, Any]:
    """
    Compute sonic flatness = (p90 - p10) / median.

    High flatness means good separation between similar and dissimilar tracks.
    Target: >= 0.5
    """
    p10 = np.percentile(sims, 10)
    p90 = np.percentile(sims, 90)
    median = np.median(sims)

    flatness = (p90 - p10) / (median + 1e-6)

    return {
        'p10': float(p10),
        'p90': float(p90),
        'median': float(median),
        'flatness': float(flatness),
        'pass': bool(flatness >= 0.5),
    }


def compute_topk_gap(
    sims: np.ndarray,
    seed_idx: int,
    k: int = 30,
    n_random: int = 100,
    rng_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute gap between top-K neighbors and random sample.

    Gap = mean(topK) - mean(random)
    Target: >= 0.15
    """
    rng = np.random.default_rng(rng_seed)

    # TopK (exclude seed itself)
    sims_copy = sims.copy()
    sims_copy[seed_idx] = -1.0  # Exclude seed

    sorted_idx = np.argsort(sims_copy)[::-1]
    topk_idx = sorted_idx[:k]
    topk_sims = sims[topk_idx]

    # Random sample
    available_idx = [i for i in range(len(sims)) if i != seed_idx]
    random_idx = rng.choice(available_idx, size=min(n_random, len(available_idx)), replace=False)
    random_sims = sims[random_idx]

    gap = float(np.mean(topk_sims) - np.mean(random_sims))

    return {
        'k': k,
        'topk_mean': float(np.mean(topk_sims)),
        'topk_min': float(np.min(topk_sims)),
        'topk_max': float(np.max(topk_sims)),
        'random_mean': float(np.mean(random_sims)),
        'random_std': float(np.std(random_sims)),
        'gap': gap,
        'pass': bool(gap >= 0.15),
    }


def compute_within_artist_coherence(
    X: np.ndarray,
    artist_keys: np.ndarray,
    seed_idx: int,
    n_random: int = 100,
    rng_seed: int = 42
) -> Dict[str, Any]:
    """
    Compute within-artist vs random separation.

    Same artist tracks should be more similar than random tracks.
    Coherence = mean(intra_artist) - mean(random)
    Target: > 0
    """
    rng = np.random.default_rng(rng_seed)

    seed_artist = str(artist_keys[seed_idx])
    same_artist_idx = [
        i for i in range(len(artist_keys))
        if str(artist_keys[i]) == seed_artist and i != seed_idx
    ]

    if len(same_artist_idx) < 2:
        return {
            'seed_artist': seed_artist,
            'n_same_artist': len(same_artist_idx),
            'intra_artist_mean': None,
            'random_mean': None,
            'coherence': None,
            'pass': None,
            'note': 'Insufficient same-artist tracks for comparison',
        }

    # L2 normalize
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    seed_vec = X_norm[seed_idx]

    # Intra-artist similarities
    intra_sims = [float(np.dot(X_norm[i], seed_vec)) for i in same_artist_idx]

    # Random similarities
    other_idx = [i for i in range(len(artist_keys)) if i != seed_idx and i not in same_artist_idx]
    random_idx = rng.choice(other_idx, size=min(n_random, len(other_idx)), replace=False)
    random_sims = [float(np.dot(X_norm[i], seed_vec)) for i in random_idx]

    coherence = float(np.mean(intra_sims) - np.mean(random_sims))

    return {
        'seed_artist': seed_artist,
        'n_same_artist': len(same_artist_idx),
        'intra_artist_mean': float(np.mean(intra_sims)),
        'intra_artist_std': float(np.std(intra_sims)),
        'random_mean': float(np.mean(random_sims)),
        'random_std': float(np.std(random_sims)),
        'coherence': coherence,
        'pass': bool(coherence > 0.0),
    }


def get_topk_metadata(
    bundle: ArtifactBundle,
    sims: np.ndarray,
    seed_idx: int,
    k: int = 10
) -> List[Dict[str, Any]]:
    """Get metadata for top-K neighbors."""
    sims_copy = sims.copy()
    sims_copy[seed_idx] = -1.0

    sorted_idx = np.argsort(sims_copy)[::-1][:k]

    results = []
    for rank, idx in enumerate(sorted_idx, 1):
        entry = {
            'rank': rank,
            'track_id': str(bundle.track_ids[idx]),
            'similarity': float(sims[idx]),
        }
        if bundle.track_artists is not None:
            entry['artist'] = str(bundle.track_artists[idx])
        if bundle.track_titles is not None:
            entry['title'] = str(bundle.track_titles[idx])
        entry['artist_key'] = str(bundle.artist_keys[idx])
        results.append(entry)

    return results


# ============================================================================
# Main Validation Runner
# ============================================================================

def run_validation_suite(
    artifact_path: Path,
    seed_track_ids: List[str],
    k: int = 30,
    n_random_pairs: int = 10000,
) -> Dict[str, Any]:
    """
    Run full validation suite on artifact.

    Returns:
        Dict with all metrics and pass/fail status
    """
    logger.info(f"Loading artifact from {artifact_path}")
    bundle = load_artifact_bundle(artifact_path)

    X_sonic = bundle.X_sonic
    logger.info(f"Loaded {X_sonic.shape[0]} tracks, {X_sonic.shape[1]} dimensions")

    results = {
        'timestamp': datetime.now().isoformat(),
        'artifact_path': str(artifact_path),
        'n_tracks': X_sonic.shape[0],
        'n_dimensions': X_sonic.shape[1],
        'n_seeds_tested': len(seed_track_ids),
        'global_metrics': {},
        'per_seed': {},
        'summary': {},
    }

    # ---- Global Metrics ----
    logger.info("Computing global metrics...")

    logger.info("  - Dimension variance")
    results['global_metrics']['dimension_variance'] = compute_dimension_variance(X_sonic)

    logger.info("  - Unique vectors")
    results['global_metrics']['unique_vectors'] = compute_unique_vectors(X_sonic)

    logger.info("  - Random similarity distribution")
    results['global_metrics']['random_similarity'] = compute_random_similarity_distribution(
        X_sonic, n_pairs=n_random_pairs
    )

    # ---- Per-Seed Metrics ----
    for seed_id in seed_track_ids:
        logger.info(f"Processing seed: {seed_id[:16]}...")

        if seed_id not in bundle.track_id_to_index:
            logger.warning(f"Seed {seed_id} not found in artifact, skipping")
            results['per_seed'][seed_id] = {'error': 'Track not found in artifact'}
            continue

        seed_idx = bundle.track_id_to_index[seed_id]
        seed_results = {}

        # Get seed metadata
        seed_results['seed_metadata'] = {
            'track_id': seed_id,
            'index': seed_idx,
            'artist_key': str(bundle.artist_keys[seed_idx]),
        }
        if bundle.track_artists is not None:
            seed_results['seed_metadata']['artist'] = str(bundle.track_artists[seed_idx])
        if bundle.track_titles is not None:
            seed_results['seed_metadata']['title'] = str(bundle.track_titles[seed_idx])

        # Compute all similarities
        sims = compute_all_similarities(X_sonic, seed_idx)

        # Metrics
        seed_results['sonic_flatness'] = compute_sonic_flatness(sims)
        seed_results['topk_gap'] = compute_topk_gap(sims, seed_idx, k=k)
        seed_results['within_artist_coherence'] = compute_within_artist_coherence(
            X_sonic, bundle.artist_keys, seed_idx
        )
        seed_results['top10_neighbors'] = get_topk_metadata(bundle, sims, seed_idx, k=10)

        results['per_seed'][seed_id] = seed_results

    # ---- Aggregate Summary ----
    results['summary'] = compute_summary(results)

    return results


def compute_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregate summary and pass/fail status."""
    summary = {
        'global_pass': True,
        'per_seed_pass': True,
        'overall_pass': True,
        'issues': [],
    }

    # Global checks
    gm = results['global_metrics']

    if not gm['dimension_variance']['pass']:
        summary['global_pass'] = False
        summary['issues'].append('FAIL: Dimension variance too low (collapsed features)')

    if not gm['unique_vectors']['pass']:
        summary['global_pass'] = False
        summary['issues'].append(f"FAIL: Only {gm['unique_vectors']['unique_pct']:.1f}% unique vectors")

    if not gm['random_similarity']['pass']:
        summary['global_pass'] = False
        summary['issues'].append(
            f"FAIL: Random similarity mean={gm['random_similarity']['mean']:.4f}, "
            f"std={gm['random_similarity']['std']:.4f} (too flat)"
        )

    # Per-seed checks
    flatness_values = []
    gap_values = []
    coherence_values = []

    for seed_id, seed_data in results['per_seed'].items():
        if 'error' in seed_data:
            continue

        flatness_values.append(seed_data['sonic_flatness']['flatness'])
        gap_values.append(seed_data['topk_gap']['gap'])

        if seed_data['within_artist_coherence']['coherence'] is not None:
            coherence_values.append(seed_data['within_artist_coherence']['coherence'])

        if not seed_data['sonic_flatness']['pass']:
            summary['per_seed_pass'] = False
            summary['issues'].append(
                f"WARN: Seed {seed_id[:8]} flatness={seed_data['sonic_flatness']['flatness']:.3f} < 0.5"
            )

        if not seed_data['topk_gap']['pass']:
            summary['per_seed_pass'] = False
            summary['issues'].append(
                f"WARN: Seed {seed_id[:8]} topK_gap={seed_data['topk_gap']['gap']:.3f} < 0.15"
            )

    # Aggregate stats
    if flatness_values:
        summary['avg_flatness'] = float(np.mean(flatness_values))
        summary['min_flatness'] = float(np.min(flatness_values))

    if gap_values:
        summary['avg_topk_gap'] = float(np.mean(gap_values))
        summary['min_topk_gap'] = float(np.min(gap_values))

    if coherence_values:
        summary['avg_coherence'] = float(np.mean(coherence_values))
        summary['min_coherence'] = float(np.min(coherence_values))

    summary['overall_pass'] = summary['global_pass'] and summary['per_seed_pass']

    return summary


def generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate human-readable markdown report."""
    lines = [
        "# Sonic Quality Validation Report",
        "",
        f"**Generated**: {results['timestamp']}",
        f"**Artifact**: `{results['artifact_path']}`",
        f"**Tracks**: {results['n_tracks']:,}",
        f"**Dimensions**: {results['n_dimensions']}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    summary = results['summary']
    status = "✅ PASS" if summary['overall_pass'] else "❌ FAIL"
    lines.append(f"**Overall Status**: {status}")
    lines.append("")

    if summary['issues']:
        lines.append("### Issues")
        for issue in summary['issues']:
            lines.append(f"- {issue}")
        lines.append("")

    if 'avg_flatness' in summary:
        lines.extend([
            "### Aggregate Metrics",
            "",
            f"| Metric | Value | Target |",
            f"|--------|-------|--------|",
            f"| Avg Flatness | {summary.get('avg_flatness', 'N/A'):.3f} | >= 0.5 |",
            f"| Min Flatness | {summary.get('min_flatness', 'N/A'):.3f} | >= 0.5 |",
            f"| Avg TopK Gap | {summary.get('avg_topk_gap', 'N/A'):.3f} | >= 0.15 |",
            f"| Min TopK Gap | {summary.get('min_topk_gap', 'N/A'):.3f} | >= 0.15 |",
            f"| Avg Coherence | {summary.get('avg_coherence', 'N/A'):.3f} | > 0 |",
            "",
        ])

    # Global metrics
    lines.extend([
        "---",
        "",
        "## Global Metrics",
        "",
    ])

    gm = results['global_metrics']

    # Dimension variance
    dv = gm['dimension_variance']
    dv_status = "✅" if dv['pass'] else "❌"
    lines.extend([
        f"### Dimension Variance {dv_status}",
        "",
        f"- Min: {dv['variance_min']:.2e}",
        f"- Median: {dv['variance_median']:.2e}",
        f"- Max: {dv['variance_max']:.2e}",
        f"- Zero-variance dims: {dv['n_zero_variance']}",
        f"- Low-variance dims: {dv['n_low_variance']}",
        "",
    ])

    # Unique vectors
    uv = gm['unique_vectors']
    uv_status = "✅" if uv['pass'] else "❌"
    lines.extend([
        f"### Unique Vectors {uv_status}",
        "",
        f"- Unique: {uv['n_unique']:,} / {uv['n_tracks']:,} ({uv['unique_pct']:.1f}%)",
        f"- Precision: {uv['precision_decimals']} decimal places",
        "",
    ])

    # Random similarity
    rs = gm['random_similarity']
    rs_status = "✅" if rs['pass'] else "❌"
    lines.extend([
        f"### Random Pair Similarity Distribution {rs_status}",
        "",
        f"- Mean: {rs['mean']:.4f}",
        f"- Std: {rs['std']:.4f}",
        f"- Range: [{rs['min']:.4f}, {rs['max']:.4f}]",
        f"- Percentiles: p10={rs['p10']:.4f}, p50={rs['p50']:.4f}, p90={rs['p90']:.4f}",
        "",
    ])

    # Per-seed results
    lines.extend([
        "---",
        "",
        "## Per-Seed Results",
        "",
    ])

    for seed_id, seed_data in results['per_seed'].items():
        if 'error' in seed_data:
            lines.extend([
                f"### Seed: `{seed_id[:16]}...`",
                f"**Error**: {seed_data['error']}",
                "",
            ])
            continue

        meta = seed_data['seed_metadata']
        artist = meta.get('artist', meta.get('artist_key', 'Unknown'))
        title = meta.get('title', 'Unknown')

        lines.extend([
            f"### Seed: {artist} - {title}",
            f"*Track ID: `{seed_id[:16]}...`*",
            "",
        ])

        # Flatness
        sf = seed_data['sonic_flatness']
        sf_status = "✅" if sf['pass'] else "❌"
        lines.extend([
            f"**Flatness** {sf_status}: {sf['flatness']:.3f} (target >= 0.5)",
            f"- p10={sf['p10']:.4f}, p90={sf['p90']:.4f}, median={sf['median']:.4f}",
            "",
        ])

        # TopK gap
        tg = seed_data['topk_gap']
        tg_status = "✅" if tg['pass'] else "❌"
        lines.extend([
            f"**TopK Gap** {tg_status}: {tg['gap']:.3f} (target >= 0.15)",
            f"- TopK mean={tg['topk_mean']:.4f}, Random mean={tg['random_mean']:.4f}",
            "",
        ])

        # Coherence
        wac = seed_data['within_artist_coherence']
        if wac['coherence'] is not None:
            wac_status = "✅" if wac['pass'] else "❌"
            lines.extend([
                f"**Within-Artist Coherence** {wac_status}: {wac['coherence']:.3f} (target > 0)",
                f"- Same artist ({wac['n_same_artist']} tracks): mean={wac['intra_artist_mean']:.4f}",
                f"- Random: mean={wac['random_mean']:.4f}",
                "",
            ])
        else:
            lines.append(f"**Within-Artist Coherence**: {wac['note']}")
            lines.append("")

        # Top neighbors
        lines.extend([
            "**Top 10 Neighbors**:",
            "",
            "| Rank | Similarity | Artist | Title |",
            "|------|------------|--------|-------|",
        ])
        for neighbor in seed_data['top10_neighbors']:
            artist = neighbor.get('artist', neighbor.get('artist_key', 'Unknown'))
            title = neighbor.get('title', 'Unknown')
            lines.append(
                f"| {neighbor['rank']} | {neighbor['similarity']:.4f} | {artist[:25]} | {title[:30]} |"
            )
        lines.append("")

    # Write report
    report_path = output_path / "validation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"Wrote markdown report to {report_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate sonic feature quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--artifact', '-a',
        type=str,
        required=True,
        help='Path to artifact NPZ file'
    )

    parser.add_argument(
        '--seeds', '-s',
        type=str,
        default='',
        help='Comma-separated list of seed track IDs to test'
    )

    parser.add_argument(
        '--seed-file',
        type=str,
        help='File containing seed track IDs (one per line)'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='Number of top neighbors to analyze (default: 30)'
    )

    parser.add_argument(
        '--n-random-pairs',
        type=int,
        default=10000,
        help='Number of random pairs for distribution (default: 10000)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='diagnostics/sonic_validation/',
        help='Output directory for results'
    )

    parser.add_argument(
        '--random-seeds',
        type=int,
        default=0,
        help='If > 0, select this many random seeds from artifact'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        logger.error(f"Artifact not found: {artifact_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect seed IDs
    seed_ids = []

    if args.seeds:
        seed_ids.extend([s.strip() for s in args.seeds.split(',') if s.strip()])

    if args.seed_file:
        seed_file = Path(args.seed_file)
        if seed_file.exists():
            with open(seed_file, 'r') as f:
                seed_ids.extend([line.strip() for line in f if line.strip()])

    # Random seeds if requested
    if args.random_seeds > 0:
        bundle = load_artifact_bundle(artifact_path)
        rng = np.random.default_rng(42)
        random_idx = rng.choice(len(bundle.track_ids), size=min(args.random_seeds, len(bundle.track_ids)), replace=False)
        seed_ids.extend([str(bundle.track_ids[i]) for i in random_idx])

    if not seed_ids:
        logger.warning("No seed track IDs provided. Using 5 random seeds.")
        bundle = load_artifact_bundle(artifact_path)
        rng = np.random.default_rng(42)
        random_idx = rng.choice(len(bundle.track_ids), size=min(5, len(bundle.track_ids)), replace=False)
        seed_ids = [str(bundle.track_ids[i]) for i in random_idx]

    logger.info(f"Testing {len(seed_ids)} seed tracks")

    # Run validation
    results = run_validation_suite(
        artifact_path=artifact_path,
        seed_track_ids=seed_ids,
        k=args.k,
        n_random_pairs=args.n_random_pairs,
    )

    # Save JSON results
    json_path = output_dir / "validation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote JSON results to {json_path}")

    # Generate markdown report
    generate_markdown_report(results, output_dir)

    # Print summary
    summary = results['summary']
    print("\n" + "=" * 60)
    print("SONIC QUALITY VALIDATION SUMMARY")
    print("=" * 60)

    status = "PASS" if summary['overall_pass'] else "FAIL"
    print(f"\nOverall Status: {status}")

    if 'avg_flatness' in summary:
        print(f"\nAggregate Metrics:")
        print(f"  Avg Flatness:  {summary['avg_flatness']:.3f} (target >= 0.5)")
        print(f"  Avg TopK Gap:  {summary['avg_topk_gap']:.3f} (target >= 0.15)")
        if 'avg_coherence' in summary:
            print(f"  Avg Coherence: {summary['avg_coherence']:.3f} (target > 0)")

    if summary['issues']:
        print(f"\nIssues ({len(summary['issues'])}):")
        for issue in summary['issues'][:5]:
            print(f"  - {issue}")
        if len(summary['issues']) > 5:
            print(f"  ... and {len(summary['issues']) - 5} more")

    print(f"\nFull results: {output_dir}")
    print("=" * 60)

    return 0 if summary['overall_pass'] else 1


if __name__ == "__main__":
    sys.exit(main())
