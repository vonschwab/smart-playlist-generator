"""
Run Artifact Module - Structured per-run output for playlist tuning.

Emits JSON + CSV artifacts capturing:
- Settings snapshot (mode, weights, thresholds, etc.)
- Per-track data (position, id, artist, seed_sim, etc.)
- Per-edge data (prev→next similarity scores)
- Exclusion counters (why candidates were rejected)
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SettingsSnapshot:
    """Captures all tunable settings for a single playlist run."""
    # Run identification
    run_id: str
    timestamp: str

    # Mode and pipeline
    mode: str  # narrow, dynamic, discover
    pipeline: str  # ds, legacy

    # Seed info
    seed_track_id: str
    seed_artist: str
    seed_title: str

    # Playlist params
    playlist_length: int
    random_seed: int

    # Hybrid weights
    sonic_weight: float
    genre_weight: float

    # Genre settings
    genre_method: str
    min_genre_similarity: float
    genre_gate_mode: str  # "hard" or "soft"

    # Transition settings
    transition_floor: float
    transition_gamma: float
    hard_floor: bool
    center_transitions: bool

    # Construction params (DS pipeline)
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    alpha_schedule: str = "constant"

    # Candidate pool params
    similarity_floor: float = 0.0
    max_pool_size: int = 0

    # Artist constraints
    max_artist_fraction: float = 0.0
    min_gap: int = 0

    # Sonic variant (default: robust_whiten - validated as best-performing)
    sonic_variant: str = "robust_whiten"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackRecord:
    """Per-track data in the final playlist."""
    position: int
    track_id: str
    artist_key: str
    artist_name: str
    title: str
    duration_ms: Optional[int]
    seed_sim: float  # Similarity to seed track
    genres: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['genres'] = ','.join(self.genres) if self.genres else ''
        return d


@dataclass
class EdgeRecord:
    """Per-edge (transition) data between consecutive tracks."""
    position: int  # Edge index (0 = track[0]→track[1])
    prev_track_id: str
    next_track_id: str
    prev_artist: str
    next_artist: str

    # Similarity scores
    sonic_sim: float
    genre_sim: float
    hybrid_sim: float

    # Transition-specific (end→start)
    transition_sim: float  # T score (blended or segment)
    transition_raw: float  # Uncentered segment cos
    transition_centered: float  # Centered segment cos (if applicable)

    # Flags
    below_floor: bool = False
    same_artist: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExclusionCounters:
    """Tracks why candidates were excluded during pool/construction."""
    # Candidate pool phase
    below_similarity_floor: int = 0

    # Genre gate (if hard)
    genre_gate_rejected: int = 0

    # Artist constraints
    artist_cap_rejected: int = 0
    adjacency_rejected: int = 0
    min_gap_rejected: int = 0

    # Transition floor
    transition_floor_rejected: int = 0

    # Total candidates considered
    total_candidates_considered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryMetrics:
    """Aggregated metrics for the playlist."""
    # Edge metrics
    edge_hybrid_mean: float = 0.0
    edge_hybrid_median: float = 0.0
    edge_hybrid_p10: float = 0.0
    edge_hybrid_min: float = 0.0

    edge_sonic_mean: float = 0.0
    edge_sonic_median: float = 0.0
    edge_sonic_p10: float = 0.0
    edge_sonic_min: float = 0.0

    edge_genre_mean: float = 0.0
    edge_genre_median: float = 0.0
    edge_genre_p10: float = 0.0
    edge_genre_min: float = 0.0

    edge_transition_mean: float = 0.0
    edge_transition_median: float = 0.0
    edge_transition_p10: float = 0.0
    edge_transition_min: float = 0.0

    # Genre leakage proxies
    tracks_below_genre_threshold: int = 0
    edges_with_very_low_genre: int = 0  # genre_sim < 0.1

    # Seed coherence
    seed_sim_mean: float = 0.0
    seed_sim_min: float = 0.0

    # Diversity
    unique_artists: int = 0
    max_artist_percentage: float = 0.0

    # Constraint violations (should be 0)
    adjacency_violations: int = 0
    min_gap_violations: int = 0
    cap_violations: int = 0

    # Below floor count
    below_floor_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunArtifact:
    """Complete artifact for a single playlist run."""
    settings: SettingsSnapshot
    tracks: List[TrackRecord]
    edges: List[EdgeRecord]
    exclusions: ExclusionCounters
    metrics: SummaryMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            'settings': self.settings.to_dict(),
            'tracks': [t.to_dict() for t in self.tracks],
            'edges': [e.to_dict() for e in self.edges],
            'exclusions': self.exclusions.to_dict(),
            'metrics': self.metrics.to_dict(),
        }


def compute_summary_metrics(
    tracks: List[TrackRecord],
    edges: List[EdgeRecord],
    min_genre_similarity: float,
) -> SummaryMetrics:
    """Compute aggregated metrics from track and edge data."""
    import numpy as np

    metrics = SummaryMetrics()

    if not edges:
        return metrics

    # Edge arrays
    hybrid_sims = np.array([e.hybrid_sim for e in edges], dtype=float)
    sonic_sims = np.array([e.sonic_sim for e in edges], dtype=float)
    genre_sims = np.array([e.genre_sim for e in edges], dtype=float)
    trans_sims = np.array([e.transition_sim for e in edges], dtype=float)

    # Filter NaN for robust stats
    def safe_stats(arr):
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.mean(valid)),
            float(np.median(valid)),
            float(np.percentile(valid, 10)),
            float(np.min(valid)),
        )

    metrics.edge_hybrid_mean, metrics.edge_hybrid_median, metrics.edge_hybrid_p10, metrics.edge_hybrid_min = safe_stats(hybrid_sims)
    metrics.edge_sonic_mean, metrics.edge_sonic_median, metrics.edge_sonic_p10, metrics.edge_sonic_min = safe_stats(sonic_sims)
    metrics.edge_genre_mean, metrics.edge_genre_median, metrics.edge_genre_p10, metrics.edge_genre_min = safe_stats(genre_sims)
    metrics.edge_transition_mean, metrics.edge_transition_median, metrics.edge_transition_p10, metrics.edge_transition_min = safe_stats(trans_sims)

    # Genre leakage proxies
    valid_genre = genre_sims[np.isfinite(genre_sims)]
    metrics.edges_with_very_low_genre = int(np.sum(valid_genre < 0.1))

    # Seed similarity
    if tracks:
        seed_sims = np.array([t.seed_sim for t in tracks], dtype=float)
        valid_seed = seed_sims[np.isfinite(seed_sims)]
        if len(valid_seed) > 0:
            metrics.seed_sim_mean = float(np.mean(valid_seed))
            metrics.seed_sim_min = float(np.min(valid_seed))

    # Diversity
    artists = [t.artist_key for t in tracks]
    metrics.unique_artists = len(set(artists))
    if artists:
        from collections import Counter
        counts = Counter(artists)
        metrics.max_artist_percentage = max(counts.values()) / len(tracks)

    # Below floor count
    metrics.below_floor_count = sum(1 for e in edges if e.below_floor)

    # Constraint violations
    metrics.adjacency_violations = sum(1 for e in edges if e.same_artist)

    return metrics


def generate_run_id(seed_track_id: str, mode: str, timestamp: Optional[str] = None) -> str:
    """Generate a unique run ID."""
    ts = timestamp or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    # Truncate seed ID for readability
    seed_short = seed_track_id[:8] if len(seed_track_id) > 8 else seed_track_id
    return f"{ts}_{mode}_{seed_short}"


class RunArtifactWriter:
    """Writes run artifacts to disk in JSON and CSV formats."""

    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, artifact: RunArtifact) -> Optional[Dict[str, Path]]:
        """Write artifact to JSON and CSV files. Returns paths if written."""
        if not self.enabled:
            return None

        run_id = artifact.settings.run_id
        paths = {}

        # JSON (complete artifact)
        json_path = self.output_dir / f"{run_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(artifact.to_dict(), f, indent=2, ensure_ascii=False)
        paths['json'] = json_path
        logger.info(f"Wrote run artifact JSON: {json_path}")

        # CSV - tracks
        tracks_csv = self.output_dir / f"{run_id}_tracks.csv"
        if artifact.tracks:
            fieldnames = list(artifact.tracks[0].to_dict().keys())
            with open(tracks_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([t.to_dict() for t in artifact.tracks])
            paths['tracks_csv'] = tracks_csv

        # CSV - edges
        edges_csv = self.output_dir / f"{run_id}_edges.csv"
        if artifact.edges:
            fieldnames = list(artifact.edges[0].to_dict().keys())
            with open(edges_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([e.to_dict() for e in artifact.edges])
            paths['edges_csv'] = edges_csv

        # CSV - summary (single row for easy aggregation)
        summary_csv = self.output_dir / f"{run_id}_summary.csv"
        summary_row = {
            **artifact.settings.to_dict(),
            **artifact.metrics.to_dict(),
            **artifact.exclusions.to_dict(),
        }
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)
        paths['summary_csv'] = summary_csv

        return paths


def append_to_consolidated_csv(
    consolidated_path: Path,
    artifact: RunArtifact,
) -> None:
    """Append a summary row to a consolidated CSV for cross-run comparison."""
    row = {
        **artifact.settings.to_dict(),
        **artifact.metrics.to_dict(),
        **artifact.exclusions.to_dict(),
    }

    write_header = not consolidated_path.exists()
    with open(consolidated_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
