from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.playlist.pipeline import generate_playlist_ds as core_generate_playlist_ds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DsRunResult:
    track_ids: List[str]
    requested: Dict[str, Any]
    effective: Dict[str, Any]
    metrics: Dict[str, Any]
    playlist_stats: Dict[str, Any]


def generate_playlist_ds(
    *,
    artifact_path: str,
    seed_track_id: str,
    mode: str,
    length: int,
    random_seed: int,
    overrides: Optional[Dict[str, Any]] = None,
    enable_logging: bool = False,
    allowed_track_ids: Optional[List[str]] = None,
    excluded_track_ids: Optional[set[str]] = None,
    single_artist: bool = False,
    sonic_variant: Optional[str] = None,
    # Genre similarity parameters
    sonic_weight: Optional[float] = None,
    genre_weight: Optional[float] = None,
    min_genre_similarity: Optional[float] = None,
    genre_method: Optional[str] = None,
) -> DsRunResult:
    """Production-facing wrapper around the DS pipeline."""
    result = core_generate_playlist_ds(
        artifact_path=artifact_path,
        seed_track_id=seed_track_id,
        num_tracks=length,
        mode=mode,
        random_seed=random_seed,
        overrides=overrides,
        allowed_track_ids=allowed_track_ids,
        excluded_track_ids=excluded_track_ids,
        single_artist=single_artist,
        sonic_variant=sonic_variant,
        # Pass through genre similarity parameters
        sonic_weight=sonic_weight,
        genre_weight=genre_weight,
        min_genre_similarity=min_genre_similarity,
        genre_method=genre_method,
    )

    playlist_stats = result.stats.get("playlist", {})
    metrics: Dict[str, Any] = {
        "below_floor": playlist_stats.get("below_floor_count"),
        "min_transition": playlist_stats.get("min_transition"),
        "mean_transition": playlist_stats.get("mean_transition"),
        "gap_sum": playlist_stats.get("gap_sum"),
        "gap_p90": playlist_stats.get("gap_p90"),
        "seed_sim_min": playlist_stats.get("seed_sim_min"),
        "seed_sim_mean": playlist_stats.get("seed_sim_mean"),
        "artist_counts": playlist_stats.get("artist_counts"),
        "distinct_artists": playlist_stats.get("distinct_artists"),
    }

    if enable_logging:
        payload = {
            "pipeline": "ds",
            "mode": mode,
            "seed_track_id": seed_track_id,
            "length": length,
            "random_seed": random_seed,
            "metrics": metrics,
            "effective": result.params_effective,
        }
        logger.info(json.dumps(payload))

    requested = dict(result.params_requested)
    requested.update({"seed_track_id": seed_track_id, "mode": mode, "length": length, "random_seed": random_seed})

    return DsRunResult(
        track_ids=result.track_ids,
        requested=requested,
        effective=result.params_effective,
        metrics=metrics,
        playlist_stats=result.stats,
    )
