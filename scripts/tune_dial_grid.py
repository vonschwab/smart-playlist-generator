#!/usr/bin/env python3
"""
Dial Grid Tuning Harness - Run playlists across parameter combinations for A/B testing.

This script generates playlists for multiple seed tracks across a grid of
dial settings (weights, thresholds, genre methods), exports M3U files with
dial-encoded filenames, and writes structured artifacts for analysis.

Examples:
  # Basic usage with default grid
  python scripts/tune_dial_grid.py --artifact path/to/artifact.npz --seeds seed1,seed2,seed3

  # With seeds file
  python scripts/tune_dial_grid.py --artifact path/to/artifact.npz --seeds-file diagnostics/seeds.txt

  # Custom grid
  python scripts/tune_dial_grid.py --artifact path/to/artifact.npz --seeds seed1 \
      --min-genre-sim 0.2,0.25,0.3 --sonic-weight 0.55,0.65,0.75

  # Full dial grid with M3U export for listening tests
  python scripts/tune_dial_grid.py --artifact path/to/artifact.npz --seeds-file seeds.txt \
      --export-m3u-dir diagnostics/tune_m3u --mode dynamic --length 30
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_artifact import (
    RunArtifact,
    RunArtifactWriter,
    append_to_consolidated_csv,
)
from src.features.artifacts import load_artifact_bundle
from src.playlist.config import default_ds_config
from src.playlist.pipeline import DSPipelineResult, build_run_artifact, generate_playlist_ds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DialSettings:
    """A single combination of dial settings."""
    sonic_weight: float
    genre_weight: float
    min_genre_similarity: float
    genre_method: str
    transition_strictness: str  # baseline, strictish, lenient
    sonic_variant: str
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sonic_weight": self.sonic_weight,
            "genre_weight": self.genre_weight,
            "min_genre_similarity": self.min_genre_similarity,
            "genre_method": self.genre_method,
            "transition_strictness": self.transition_strictness,
            "sonic_variant": self.sonic_variant,
            "mode": self.mode,
        }

    def filename_suffix(self) -> str:
        """Generate a suffix for filenames encoding key dials."""
        return (
            f"sw{int(self.sonic_weight*100)}_"
            f"mgs{int(self.min_genre_similarity*100)}_"
            f"gm-{self.genre_method[:4]}_"
            f"ts-{self.transition_strictness[:4]}"
        )


def build_dial_grid(
    sonic_weights: List[float],
    min_genre_sims: List[float],
    genre_methods: List[str],
    transition_strictnesses: List[str],
    sonic_variants: List[str],
    mode: str,
) -> List[DialSettings]:
    """Build the Cartesian product of all dial combinations."""
    grid = []
    for sw, mgs, gm, ts, sv in product(
        sonic_weights, min_genre_sims, genre_methods, transition_strictnesses, sonic_variants
    ):
        gw = round(1.0 - sw, 2)  # genre_weight = 1 - sonic_weight
        grid.append(DialSettings(
            sonic_weight=sw,
            genre_weight=gw,
            min_genre_similarity=mgs,
            genre_method=gm,
            transition_strictness=ts,
            sonic_variant=sv,
            mode=mode,
        ))
    return grid


def get_overrides_for_strictness(
    strictness: str,
    mode: str,
    base_floor: float,
) -> Dict[str, Any]:
    """
    Convert transition strictness level to config overrides.

    Mode-specific policy:
    - narrow: strict floors (hard), strict genre gate (hard)
    - dynamic: moderate floors (still hard), strict genre gate (hard)
    - discover: lenient floors, genre is soft penalty
    """
    overrides: Dict[str, Any] = {"construct": {}}

    if strictness == "baseline":
        # Use mode defaults (no override)
        pass
    elif strictness == "strictish":
        # SUBSTANTIALLY raise floor to force different candidate selection
        # This WILL exclude many candidates and force different playlist construction
        overrides["construct"]["transition_floor"] = 0.85
        overrides["construct"]["hard_floor"] = True
        logger.info("Strictish mode: raising transition_floor to 0.85 (from %.2f)", base_floor)
    elif strictness == "lenient":
        # Lower floor for more permissive transitions
        overrides["construct"]["transition_floor"] = max(0.15, base_floor - 0.10)
        overrides["construct"]["hard_floor"] = False
    elif strictness == "strict":
        # EXTREMELY strict: only highest-quality transitions
        overrides["construct"]["transition_floor"] = 0.95
        overrides["construct"]["hard_floor"] = True
        logger.info("Strict mode: raising transition_floor to 0.95 (from %.2f)", base_floor)

    return overrides if overrides.get("construct") else {}


def load_seeds(seeds_arg: Optional[str], seeds_file: Optional[Path], artifact_path: Path, n_random: int = 10, random_seed: int = 42) -> List[str]:
    """Load seed track IDs from CLI arg, file, or random sample."""
    if seeds_arg:
        # Comma-separated list
        return [s.strip() for s in seeds_arg.split(",") if s.strip()]

    if seeds_file and seeds_file.exists():
        ext = seeds_file.suffix.lower()
        if ext == ".json":
            data = json.loads(seeds_file.read_text())
            if isinstance(data, dict) and "seeds" in data:
                data = data["seeds"]
            return [str(x) for x in data]
        else:
            # Plain text, one per line
            return [line.strip() for line in seeds_file.read_text().splitlines() if line.strip()]

    # Fall back to random sample
    logger.info(f"No seeds specified, sampling {n_random} random seeds from artifact")
    bundle = load_artifact_bundle(artifact_path)
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(len(bundle.track_ids), size=min(n_random, len(bundle.track_ids)), replace=False)
    return [str(bundle.track_ids[i]) for i in indices]


def run_single_dial(
    seed_id: str,
    dials: DialSettings,
    artifact_path: Path,
    playlist_length: int,
    random_seed: int,
) -> Tuple[Optional[DSPipelineResult], Optional[RunArtifact], float, Optional[str]]:
    """
    Run a single playlist generation with the given dial settings.

    Returns: (result, artifact, runtime_sec, error_msg)
    """
    try:
        t0 = time.perf_counter()

        # Get mode defaults
        cfg = default_ds_config(dials.mode, playlist_len=playlist_length)
        base_floor = cfg.construct.transition_floor

        # Build overrides from dial settings
        overrides = get_overrides_for_strictness(dials.transition_strictness, dials.mode, base_floor)

        result = generate_playlist_ds(
            artifact_path=artifact_path,
            seed_track_id=seed_id,
            num_tracks=playlist_length,
            mode=dials.mode,
            random_seed=random_seed,
            overrides=overrides if overrides else None,
            sonic_variant=dials.sonic_variant,
            # Pass hybrid-level tuning dials
            sonic_weight=dials.sonic_weight,
            genre_weight=dials.genre_weight,
            min_genre_similarity=dials.min_genre_similarity,
            genre_method=dials.genre_method,
        )

        runtime = time.perf_counter() - t0

        # Build artifact
        bundle = load_artifact_bundle(artifact_path)
        # Determine genre_gate_mode based on mode
        genre_gate_mode = "soft" if dials.mode == "discover" else "hard"

        artifact = build_run_artifact(
            result,
            bundle,
            cfg,
            seed_id,
            sonic_weight=dials.sonic_weight,
            genre_weight=dials.genre_weight,
            genre_method=dials.genre_method,
            min_genre_similarity=dials.min_genre_similarity,
            genre_gate_mode=genre_gate_mode,
            sonic_variant=dials.sonic_variant,
        )

        return result, artifact, runtime, None

    except Exception as e:
        logger.error(f"Error running dial {dials.filename_suffix()} for seed {seed_id}: {e}")
        return None, None, 0.0, str(e)


def write_m3u(
    track_ids: List[str],
    artifact_path: Path,
    output_path: Path,
    db_path: Optional[Path] = None,
) -> None:
    """Write M3U playlist file."""
    bundle = load_artifact_bundle(artifact_path)

    # Try to load metadata from database for file paths
    meta: Dict[str, Dict[str, str]] = {}
    if db_path and db_path.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT track_id, file_path, artist, title FROM tracks").fetchall()
            for r in rows:
                if r["track_id"]:
                    meta[str(r["track_id"])] = {
                        "file_path": r["file_path"] or "",
                        "artist": r["artist"] or "",
                        "title": r["title"] or "",
                    }
            conn.close()
        except Exception as e:
            logger.warning(f"Could not load metadata from db: {e}")

    # Fall back to bundle data
    for i, tid in enumerate(bundle.track_ids):
        tid_str = str(tid)
        if tid_str not in meta:
            artist = str(bundle.track_artists[i]) if bundle.track_artists is not None else ""
            title = str(bundle.track_titles[i]) if bundle.track_titles is not None else ""
            meta[tid_str] = {"file_path": tid_str, "artist": artist, "title": title}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    for tid in track_ids:
        m = meta.get(str(tid), {})
        artist = m.get("artist", "")
        title = m.get("title", "")
        info = f"#EXTINF:-1,{artist} - {title}".strip(", -")
        lines.append(info)
        lines.append(m.get("file_path") or str(tid))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def compute_summary_row(
    seed_id: str,
    dials: DialSettings,
    artifact: Optional[RunArtifact],
    runtime: float,
    error: Optional[str],
    result: Optional[DSPipelineResult] = None,
) -> Dict[str, Any]:
    """Build a summary row for the consolidated CSV."""
    row = {
        "seed_track_id": seed_id,
        "timestamp": datetime.now().isoformat(),
        "runtime_sec": round(runtime, 3),
        "error": error or "",
        **dials.to_dict(),
    }

    if artifact and result:
        row.update({
            "playlist_length": artifact.metrics.unique_artists,  # Actually should be settings
            "playlist_length_actual": artifact.settings.playlist_length,

            # Edge metrics
            "edge_hybrid_mean": round(artifact.metrics.edge_hybrid_mean, 4),
            "edge_hybrid_median": round(artifact.metrics.edge_hybrid_median, 4),
            "edge_hybrid_p10": round(artifact.metrics.edge_hybrid_p10, 4),
            "edge_hybrid_min": round(artifact.metrics.edge_hybrid_min, 4),

            "edge_sonic_mean": round(artifact.metrics.edge_sonic_mean, 4),
            "edge_sonic_p10": round(artifact.metrics.edge_sonic_p10, 4),
            "edge_sonic_min": round(artifact.metrics.edge_sonic_min, 4),

            "edge_genre_mean": round(artifact.metrics.edge_genre_mean, 4),
            "edge_genre_p10": round(artifact.metrics.edge_genre_p10, 4),
            "edge_genre_min": round(artifact.metrics.edge_genre_min, 4),

            "edge_transition_mean": round(artifact.metrics.edge_transition_mean, 4),
            "edge_transition_p10": round(artifact.metrics.edge_transition_p10, 4),
            "edge_transition_min": round(artifact.metrics.edge_transition_min, 4),

            # Genre leakage proxies
            "edges_with_very_low_genre": artifact.metrics.edges_with_very_low_genre,
            "tracks_below_genre_threshold": artifact.metrics.tracks_below_genre_threshold,

            # Seed coherence
            "seed_sim_mean": round(artifact.metrics.seed_sim_mean, 4),
            "seed_sim_min": round(artifact.metrics.seed_sim_min, 4),

            # Diversity
            "unique_artists": artifact.metrics.unique_artists,
            "max_artist_percentage": round(artifact.metrics.max_artist_percentage, 4),

            # Constraint violations (should be 0)
            "adjacency_violations": artifact.metrics.adjacency_violations,
            "below_floor_count": artifact.metrics.below_floor_count,

            # Exclusion counters
            "below_similarity_floor": artifact.exclusions.below_similarity_floor,
            "below_genre_similarity": result.stats.get("candidate_pool", {}).get("below_genre_similarity", 0),
            "artist_cap_rejected": artifact.exclusions.artist_cap_rejected,
            "transition_floor_rejected": artifact.exclusions.transition_floor_rejected,
        })

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run dial grid tuning for playlist generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--artifact", required=True, type=Path, help="Path to artifact NPZ file")
    parser.add_argument("--seeds", type=str, help="Comma-separated seed track IDs")
    parser.add_argument("--seeds-file", type=Path, help="File with seed track IDs (one per line or JSON)")
    parser.add_argument("--n-random-seeds", type=int, default=10, help="Number of random seeds if none specified")

    parser.add_argument("--mode", default="dynamic", choices=["narrow", "dynamic", "discover"])
    parser.add_argument("--length", type=int, default=30, help="Playlist length")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for reproducibility")

    # Dial grid arguments
    parser.add_argument(
        "--sonic-weight",
        default="0.55,0.65,0.75",
        help="Comma-separated sonic weights (genre_weight = 1 - sonic_weight)"
    )
    parser.add_argument(
        "--min-genre-sim",
        default="0.20,0.25,0.30",
        help="Comma-separated min_genre_similarity values"
    )
    parser.add_argument(
        "--genre-method",
        default="ensemble,weighted_jaccard",
        help="Comma-separated genre similarity methods"
    )
    parser.add_argument(
        "--transition-strictness",
        default="baseline,strictish",
        help="Comma-separated transition strictness levels (baseline, strictish, lenient, strict)"
    )
    parser.add_argument(
        "--sonic-variant",
        default="raw",
        help="Comma-separated sonic variants (raw, centered, z, z_clip, whiten_pca)"
    )

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/tune_grid"), help="Output directory")
    parser.add_argument("--export-m3u-dir", type=Path, help="Directory to export M3U files for listening")
    parser.add_argument("--db", type=Path, default=Path("data/metadata.db"), help="Database path for M3U metadata")

    args = parser.parse_args()

    # Parse grid parameters
    sonic_weights = [float(x.strip()) for x in args.sonic_weight.split(",")]
    min_genre_sims = [float(x.strip()) for x in args.min_genre_sim.split(",")]
    genre_methods = [x.strip() for x in args.genre_method.split(",")]
    transition_strictnesses = [x.strip() for x in args.transition_strictness.split(",")]
    sonic_variants = [x.strip() for x in args.sonic_variant.split(",")]

    # Build dial grid
    grid = build_dial_grid(
        sonic_weights=sonic_weights,
        min_genre_sims=min_genre_sims,
        genre_methods=genre_methods,
        transition_strictnesses=transition_strictnesses,
        sonic_variants=sonic_variants,
        mode=args.mode,
    )
    logger.info(f"Built dial grid with {len(grid)} combinations")

    # Load seeds
    seeds = load_seeds(args.seeds, args.seeds_file, args.artifact, args.n_random_seeds, args.random_seed)
    logger.info(f"Loaded {len(seeds)} seed tracks")

    # Setup output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_writer = RunArtifactWriter(args.output_dir / "artifacts", enabled=True)
    consolidated_csv = args.output_dir / "consolidated_results.csv"

    # Run grid
    total_runs = len(seeds) * len(grid)
    completed = 0
    summary_rows: List[Dict[str, Any]] = []

    for seed_id in seeds:
        for dials in grid:
            completed += 1
            dial_suffix = dials.filename_suffix()
            logger.info(f"[{completed}/{total_runs}] Seed={seed_id[:8]}... Dials={dial_suffix}")

            result, artifact, runtime, error = run_single_dial(
                seed_id=seed_id,
                dials=dials,
                artifact_path=args.artifact,
                playlist_length=args.length,
                random_seed=args.random_seed,
            )

            # Build summary row
            row = compute_summary_row(seed_id, dials, artifact, runtime, error, result)
            summary_rows.append(row)

            # Write artifact
            if artifact:
                artifact_writer.write(artifact)

            # Write M3U if requested
            if result and args.export_m3u_dir:
                seed_short = seed_id[:8]
                m3u_name = f"{seed_short}__{dial_suffix}.m3u8"
                m3u_path = args.export_m3u_dir / m3u_name
                write_m3u(result.track_ids, args.artifact, m3u_path, args.db)

    # Write consolidated CSV
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(consolidated_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Wrote consolidated results to {consolidated_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("TUNING GRID COMPLETE")
    print("=" * 60)
    print(f"Total runs: {total_runs}")
    print(f"Seeds: {len(seeds)}")
    print(f"Dial combinations: {len(grid)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Consolidated CSV: {consolidated_csv}")
    if args.export_m3u_dir:
        print(f"M3U playlists: {args.export_m3u_dir}")

    # Quick stats
    successful = [r for r in summary_rows if not r.get("error")]
    if successful:
        print(f"\nSuccessful runs: {len(successful)}/{total_runs}")

        # Group by dial settings and show averages
        print("\nAverage edge_genre_min by min_genre_similarity:")
        for mgs in min_genre_sims:
            rows = [r for r in successful if r.get("min_genre_similarity") == mgs]
            if rows:
                avg = sum(r.get("edge_genre_min", 0) for r in rows) / len(rows)
                low_genre_edges = sum(r.get("edges_with_very_low_genre", 0) for r in rows)
                print(f"  min_genre_sim={mgs}: avg_genre_min={avg:.3f}, total_low_genre_edges={low_genre_edges}")


if __name__ == "__main__":
    main()
