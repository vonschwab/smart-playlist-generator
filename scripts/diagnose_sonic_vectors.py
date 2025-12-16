#!/usr/bin/env python3
"""
Diagnose if sonic vectors are collapsed or degenerate due to bugs/caching/normalization.

Checks:
1. Per-dimension variance/std across library
2. Count of near-unique vectors
3. Distribution of random-pair cosine sims vs topK sims
4. Evidence of caching/normalization issues
"""

import sys
import sqlite3
import json
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.artifacts import load_artifact_bundle
from src.similarity_calculator import SimilarityCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = ROOT_DIR / 'data' / 'metadata.db'
CONFIG_PATH = ROOT_DIR / 'config.yaml'


def diagnose_vector_degeneracy(artifact_path: str) -> dict:
    """Check if vectors are collapsed/degenerate."""
    logger.info("Loading artifact...")
    bundle = load_artifact_bundle(artifact_path)

    X_sonic = bundle.X_sonic
    N, D = X_sonic.shape
    logger.info(f"Artifact: {N} tracks × {D} dimensions")

    results = {}

    # ============================================================================
    # 1. Per-dimension variance/std
    # ============================================================================
    logger.info("\n[1] Per-Dimension Statistics")
    print("="*80)

    dims_var = np.var(X_sonic, axis=0)
    dims_std = np.std(X_sonic, axis=0)
    dims_mean = np.mean(X_sonic, axis=0)

    var_summary = pd.DataFrame({
        'dim': range(D),
        'mean': dims_mean,
        'std': dims_std,
        'var': dims_var,
        'cv': np.where(dims_mean != 0, dims_std / np.abs(dims_mean), 0)  # coefficient of variation
    })

    print("\nTop 10 dimensions by variance:")
    print(var_summary.nlargest(10, 'var')[['dim', 'mean', 'std', 'var']].to_string(index=False))

    print("\nBottom 10 dimensions by variance:")
    print(var_summary.nsmallest(10, 'var')[['dim', 'mean', 'std', 'var']].to_string(index=False))

    # RED FLAG: If many dimensions have near-zero variance
    zero_var_count = (dims_std < 1e-6).sum()
    print(f"\nDimensions with near-zero std (<1e-6): {zero_var_count}/{D}")
    if zero_var_count > D * 0.1:
        print("  WARNING: >10% of dimensions are constant! Vectors may be collapsed.")

    results['dims_var'] = var_summary
    results['zero_var_count'] = int(zero_var_count)

    # ============================================================================
    # 2. Vector uniqueness
    # ============================================================================
    logger.info("\n[2] Vector Uniqueness")
    print("="*80)

    # Round to 4 decimal places and hash
    X_rounded = np.round(X_sonic, 4)
    vector_hashes = set()
    for i in range(N):
        vec_hash = tuple(X_rounded[i])
        vector_hashes.add(vec_hash)

    unique_count = len(vector_hashes)
    unique_pct = 100 * unique_count / N

    print(f"\nTotal vectors: {N}")
    print(f"Unique vectors (rounded to 4dp): {unique_count}")
    print(f"Uniqueness: {unique_pct:.1f}%")

    if unique_pct < 90:
        print(f"  WARNING: Only {unique_pct:.1f}% unique! Vectors are collapsed.")
    elif unique_pct < 99:
        print(f"  CAUTION: {unique_pct:.1f}% unique. Some collapsing may occur.")
    else:
        print(f"  GOOD: {unique_pct:.1f}% unique vectors.")

    results['unique_vectors'] = int(unique_count)
    results['unique_pct'] = float(unique_pct)

    # ============================================================================
    # 3. Cosine similarity distribution
    # ============================================================================
    logger.info("\n[3] Cosine Similarity Distribution")
    print("="*80)

    # Sample 3 seeds
    seed_indices = np.random.choice(N, size=3, replace=False)

    sim_stats = []

    for seed_idx in seed_indices:
        seed_track_id = bundle.track_ids[seed_idx]

        # Normalize
        X_norm = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
        seed_vec_norm = X_norm[seed_idx]

        # Compute all similarities
        all_sims = X_norm @ seed_vec_norm

        # Exclude self (seed)
        all_sims_no_self = np.concatenate([all_sims[:seed_idx], all_sims[seed_idx+1:]])

        # Top-K
        sorted_idx = np.argsort(all_sims)[::-1]
        topk_sims = all_sims[sorted_idx[1:31]]  # top 30, excluding self

        # Random sample
        random_idx = np.random.choice(len(all_sims_no_self), size=100, replace=False)
        random_sims = all_sims_no_self[random_idx]

        # Statistics
        stats = {
            'seed_idx': seed_idx,
            'seed_track_id': seed_track_id,
            'topk_mean': float(topk_sims.mean()),
            'topk_median': float(np.median(topk_sims)),
            'topk_min': float(topk_sims.min()),
            'topk_max': float(topk_sims.max()),
            'random_mean': float(random_sims.mean()),
            'random_median': float(np.median(random_sims)),
            'random_min': float(random_sims.min()),
            'random_max': float(random_sims.max()),
            'gap': float(topk_sims.mean() - random_sims.mean()),
        }
        sim_stats.append(stats)

    sim_df = pd.DataFrame(sim_stats)
    print("\nCosine Similarity Distribution (3 random seeds):")
    print(sim_df[['seed_idx', 'topk_mean', 'random_mean', 'gap']].to_string(index=False))

    print("\nDetailed TopK vs Random:")
    for idx, row in sim_df.iterrows():
        print(f"\n  Seed {idx} (track {row['seed_track_id'][:8]}...):")
        print(f"    TopK:    mean={row['topk_mean']:.4f}, median={row['topk_median']:.4f}, range=[{row['topk_min']:.4f}, {row['topk_max']:.4f}]")
        print(f"    Random:  mean={row['random_mean']:.4f}, median={row['random_median']:.4f}, range=[{row['random_min']:.4f}, {row['random_max']:.4f}]")
        print(f"    Gap:     {row['gap']:.4f} (need >=0.15 to PASS)")

    results['sim_stats'] = sim_df

    # ============================================================================
    # 4. Check for normalization artifacts
    # ============================================================================
    logger.info("\n[4] Normalization Artifacts Check")
    print("="*80)

    # Check if all vectors have similar norm
    norms = np.linalg.norm(X_sonic, axis=1)
    norm_var = np.var(norms)
    norm_std = np.std(norms)
    norm_mean = np.mean(norms)

    print(f"\nVector norm statistics:")
    print(f"  Mean: {norm_mean:.6f}")
    print(f"  Std:  {norm_std:.6f}")
    print(f"  Var:  {norm_var:.6f}")
    print(f"  CV:   {norm_std/norm_mean:.4f}")

    if norm_std < norm_mean * 0.01:
        print("  WARNING: All vectors have nearly identical norm! May indicate post-normalization.")

    results['norm_mean'] = float(norm_mean)
    results['norm_std'] = float(norm_std)

    # ============================================================================
    # 5. Summary diagnosis
    # ============================================================================
    logger.info("\n[5] Diagnosis Summary")
    print("="*80)

    issues = []

    if zero_var_count > D * 0.1:
        issues.append(f"- CRITICAL: {zero_var_count} dimensions have near-zero variance")

    if unique_pct < 90:
        issues.append(f"- CRITICAL: Only {unique_pct:.1f}% unique vectors")

    if sim_df['gap'].mean() < 0.05:
        issues.append(f"- CRITICAL: TopK gap too low ({sim_df['gap'].mean():.4f} << 0.15)")

    if norm_std < norm_mean * 0.01:
        issues.append("- CAUTION: Vector norms suspiciously uniform")

    if not issues:
        print("\nNo obvious vector degeneracy detected!")
        print("Vectors appear normal. Problem is likely in feature extraction design, not bugs.")
    else:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(issue)

    results['diagnosis'] = "\n".join(issues) if issues else "No obvious issues"

    return results


def main():
    parser = argparse.ArgumentParser(description='Diagnose sonic vector degeneracy')
    parser.add_argument('--artifact', type=str, default='./experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz',
                        help='Path to artifact NPZ file')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Save dimension stats to CSV')

    args = parser.parse_args()

    results = diagnose_vector_degeneracy(args.artifact)

    if args.output_csv:
        results['dims_var'].to_csv(args.output_csv, index=False)
        logger.info(f"Saved dimension stats to {args.output_csv}")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
