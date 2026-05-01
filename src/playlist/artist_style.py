from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.string_utils import normalize_artist_key
from src.playlist.history_analyzer import is_collaboration_of

logger = logging.getLogger(__name__)


def _artist_indices_in_bundle(
    bundle,
    artist_name: str,
    *,
    include_collaborations: bool = False,
) -> List[int]:
    """
    Return bundle indices whose artist matches the named artist.

    Strict mode (default): match by normalized artist key.
    Inclusive mode: also accept tracks whose raw artist string is a
    collaboration containing the named artist (e.g. "Greg Foat & Art Themen",
    "Maston & Greg Foat"). Uses history_analyzer.is_collaboration_of so the
    collab patterns stay defined in one place.
    """
    if bundle.artist_keys is None:
        return []
    artist_key = normalize_artist_key(artist_name)
    raw_artists = getattr(bundle, "track_artists", None)
    indices: List[int] = []
    for i, ak in enumerate(bundle.artist_keys):
        if normalize_artist_key(str(ak)) == artist_key:
            indices.append(i)
            continue
        if include_collaborations and raw_artists is not None:
            try:
                raw = str(raw_artists[i] or "")
            except Exception:
                raw = ""
            if raw and is_collaboration_of(
                collaboration_name=raw, base_artist=artist_name
            ):
                indices.append(i)
    return indices


@dataclass(frozen=True)
class ArtistStyleConfig:
    enabled: bool = False
    cluster_k_min: int = 3
    cluster_k_max: int = 6
    cluster_k_heuristic_enabled: bool = True
    piers_per_cluster: int = 1
    per_cluster_candidate_pool_size: int = 400
    pool_balance_mode: str = "equal"  # equal | proportional_capped
    internal_connector_priority: bool = True
    internal_connector_max_per_segment: int = 2
    medoid_top_k: int = 5
    bridge_floor_strict: float = 0.10    # Phase 1 (ultra-cohesive)
    bridge_floor_narrow: float = 0.05    # Relaxed from 0.08 (Phase 3A)
    bridge_floor_dynamic: float = 0.02   # Relaxed from 0.03 (Phase 3A)
    bridge_weight: float = 0.7
    transition_weight: float = 0.3
    genre_tiebreak_weight: float = 0.05
    # Medoid selection weighting (to avoid interludes/outliers)
    medoid_similarity_weight: float = 0.7  # Weight for sonic similarity to cluster centroid
    medoid_duration_weight: float = 0.3    # Weight for duration typicality (avoid outliers)


def _select_k(track_count: int, cfg: ArtistStyleConfig) -> int:
    """Heuristic for number of clusters."""
    if not cfg.cluster_k_heuristic_enabled:
        return max(cfg.cluster_k_min, min(cfg.cluster_k_max, 3))
    if track_count < 25:
        k = 3
    elif track_count < 60:
        k = 4
    elif track_count < 120:
        k = 5
    else:
        k = 6
    return max(cfg.cluster_k_min, min(cfg.cluster_k_max, k))


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means (cosine via normalized vectors)."""
    n, _ = X.shape
    if n == 0:
        return np.array([], dtype=int), np.empty((0, X.shape[1]))
    # init: random distinct points
    init_idx = rng.choice(n, size=min(k, n), replace=False)
    centroids = X[init_idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(15):
        # assign
        sims = np.dot(X, centroids.T)
        labels = np.argmax(sims, axis=1)
        # update
        new_centroids = []
        for i in range(centroids.shape[0]):
            members = X[labels == i]
            if len(members) == 0:
                new_centroids.append(centroids[i])
            else:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid) + 1e-12
                new_centroids.append(centroid / norm)
        new_centroids = np.stack(new_centroids, axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    return labels, centroids


def _compute_artist_duration_stats(durations_ms: np.ndarray, artist_indices: List[int]) -> Dict[str, float]:
    """
    Compute duration statistics for the artist's catalog for adaptive outlier detection.

    Uses IQR method which is robust to skewed distributions (works for both punk and prog bands).
    """
    if not artist_indices:
        return {}

    artist_durations_sec = durations_ms[artist_indices] / 1000.0

    q1 = float(np.percentile(artist_durations_sec, 25))
    q3 = float(np.percentile(artist_durations_sec, 75))
    iqr = q3 - q1

    return {
        'mean': float(np.mean(artist_durations_sec)),
        'median': float(np.median(artist_durations_sec)),
        'std': float(np.std(artist_durations_sec)),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'min': float(np.min(artist_durations_sec)),
        'max': float(np.max(artist_durations_sec)),
        # Lower fence: Q1 - 1.5*IQR (standard outlier detection)
        'lower_fence': q1 - 1.5 * iqr,
    }


def _duration_outlier_score(duration_sec: float, stats: Dict[str, float]) -> float:
    """
    Calculate how much of an outlier a duration is.

    Returns:
        0.0 = typical duration for this artist
        1.0 = extreme outlier (way too short)

    Uses IQR-based method so it adapts to each artist's typical range.
    For example:
    - Punk band with 1.5 min avg: a 30s track would be an outlier
    - Prog band with 8 min avg: a 30s track would be an outlier
    - But a 1.5 min track would only be an outlier for the prog band
    """
    if not stats:
        return 0.0

    lower_fence = stats['lower_fence']

    # Not an outlier
    if duration_sec >= lower_fence:
        return 0.0

    # Calculate how far below the fence (normalized by IQR)
    iqr = stats['iqr']
    if iqr < 1e-6:  # All tracks same duration (edge case)
        return 0.0

    distance_below = lower_fence - duration_sec
    outlier_magnitude = distance_below / iqr

    # Cap at 1.0 (extreme outlier)
    return min(1.0, outlier_magnitude / 2.0)  # Divided by 2 so 3*IQR below = 1.0


def _medoids_for_cluster(
    X: np.ndarray,
    indices: List[int],
    centroid: np.ndarray,
    bundle_track_ids: Sequence[str],
    per_cluster: int,
    rng: np.random.Generator,
    top_k: int,
    artist_duration_stats: Optional[Dict[str, float]] = None,
    track_durations_ms: Optional[np.ndarray] = None,
    similarity_weight: float = 0.7,
    duration_weight: float = 0.3,
) -> List[int]:
    """
    Select medoids using weighted scoring that penalizes duration outliers.

    Weighting strategy (configurable):
    - similarity_weight (default 70%): similarity to cluster centroid (sonic cohesion)
    - duration_weight (default 30%): duration typicality (avoid extreme outliers)

    This ensures we pick representative tracks that are sonically central
    but not weird interludes/outros. The weights adapt to each artist's catalog,
    so it works for both punk bands (short tracks typical) and prog bands (long tracks typical).
    """
    if not indices:
        return []

    # Base similarity to centroid
    sims = np.dot(X[indices], centroid)

    # Initialize duration weights (default to 1.0 = no penalty)
    duration_weights = np.ones(len(indices))

    # Apply duration outlier penalty if we have duration data
    if artist_duration_stats and track_durations_ms is not None:
        for i, idx in enumerate(indices):
            duration_sec = track_durations_ms[idx] / 1000.0
            outlier_score = _duration_outlier_score(duration_sec, artist_duration_stats)

            # Convert outlier score to weight (1.0 = typical, 0.0 = extreme outlier)
            duration_weights[i] = 1.0 - outlier_score

    # Combined weighted score (configurable weights)
    scores = sims * similarity_weight + duration_weights * duration_weight

    # Select from top-k by combined score
    order = np.argsort(-scores)
    top_k_val = max(1, min(int(top_k), len(indices)))
    medoid_idx = int(order[int(rng.integers(0, top_k_val))])

    medoids: List[int] = [indices[medoid_idx]]

    if per_cluster > 1 and len(indices) > 1:
        # Pick farthest from first medoid to diversify
        # But still respect the quality scores
        first_vec = X[medoids[0]]
        sims2 = np.dot(X[indices], first_vec)

        # Combine diversity (low similarity to first) with quality (high overall score)
        diversity_scores = (1.0 - sims2) * 0.6 + scores * 0.4
        order2 = np.argsort(-diversity_scores)

        for idx in order2:
            cand = indices[int(idx)]
            if cand not in medoids:
                medoids.append(cand)
            if len(medoids) >= per_cluster:
                break

    # Debug logging
    if artist_duration_stats and track_durations_ms is not None:
        medoid_info = []
        for m in medoids:
            cluster_local_idx = indices.index(m)
            dur_sec = track_durations_ms[m] / 1000.0
            score = scores[cluster_local_idx]
            medoid_info.append(f"{bundle_track_ids[m]} ({dur_sec:.0f}s, score={score:.3f})")
        logger.debug("Cluster medoids selected (weighted): %s", medoid_info)
    else:
        logger.debug(
            "Cluster medoids selected: %s",
            [str(bundle_track_ids[i]) for i in medoids],
        )

    return medoids


def cluster_artist_tracks(
    *,
    bundle,
    artist_name: str,
    cfg: ArtistStyleConfig,
    random_seed: int = 0,
    sonic_variant: Optional[str] = None,
    medoid_top_k: int = 1,
    include_collaborations: bool = False,
) -> Tuple[List[List[int]], List[int], List[List[int]], np.ndarray]:
    """Cluster artist tracks in sonic space and return clusters + medoids."""
    track_ids = bundle.track_ids
    if bundle.artist_keys is None:
        raise ValueError("Artifact missing artist_keys for clustering.")
    X_raw = getattr(bundle, "X_sonic", None)
    if X_raw is None:
        raise ValueError("Artifact missing X_sonic for clustering.")
    from src.similarity.sonic_variant import compute_sonic_variant_norm, resolve_sonic_variant

    variant = resolve_sonic_variant(explicit_variant=sonic_variant, config_variant=None)
    X_norm, variant_stats = compute_sonic_variant_norm(X_raw, variant)
    artist_indices = _artist_indices_in_bundle(
        bundle, artist_name, include_collaborations=include_collaborations
    )
    if include_collaborations:
        # Count solo vs collab purely for visibility in the log.
        solo_only = _artist_indices_in_bundle(bundle, artist_name)
        logger.info(
            "Artist style clustering scope: artist=%s solo=%d collab=%d total=%d",
            artist_name, len(solo_only), len(artist_indices) - len(solo_only),
            len(artist_indices),
        )
    if len(artist_indices) < max(3, cfg.cluster_k_min):
        raise ValueError(f"Not enough tracks to cluster for artist {artist_name}")
    k = _select_k(len(artist_indices), cfg)
    rng = np.random.default_rng(random_seed)

    # Compute artist-specific duration statistics for adaptive outlier detection
    artist_duration_stats = None
    if bundle.durations_ms is not None:
        artist_duration_stats = _compute_artist_duration_stats(bundle.durations_ms, artist_indices)
        logger.debug(
            "Artist duration stats for %s: mean=%.1fs, median=%.1fs, Q1=%.1fs, Q3=%.1fs, lower_fence=%.1fs",
            artist_name,
            artist_duration_stats['mean'],
            artist_duration_stats['median'],
            artist_duration_stats['q1'],
            artist_duration_stats['q3'],
            artist_duration_stats['lower_fence'],
        )
    retries = 0
    while True:
        labels, centroids = _kmeans(X_norm[artist_indices], k=k, rng=rng)
        if centroids.shape[0] == 0 or len(set(labels.tolist())) < 2:
            retries += 1
            if k > cfg.cluster_k_min:
                k -= 1
                logger.warning("Cluster retry due to empty/degenerate result; retrying with k=%d", k)
                continue
            raise ValueError(f"Clustering degenerate for artist {artist_name} after retries")
        break
    clusters: List[List[int]] = []
    medoids: List[int] = []
    medoids_by_cluster: List[List[int]] = []
    for c in range(centroids.shape[0]):
        members_local = [artist_indices[i] for i, lab in enumerate(labels) if lab == c]
        if not members_local:
            continue
        clusters.append(members_local)
        medoid_list = _medoids_for_cluster(
            X_norm,
            members_local,
            centroids[c],
            track_ids,
            medoid_top_k,  # Use calculated medoid_top_k based on presence, not cfg.piers_per_cluster
            rng,
            medoid_top_k,
            artist_duration_stats,
            bundle.durations_ms,
            cfg.medoid_similarity_weight,
            cfg.medoid_duration_weight,
        )
        medoids_by_cluster.append(medoid_list)
        medoids.extend(medoid_list)
    logger.info(
        "Artist style clustering: artist=%s k=%d clusters=%d medoids=%d variant=%s dim=%d",
        artist_name,
        k,
        len(clusters),
        len(medoids),
        variant_stats.get("variant", variant),
        int(X_norm.shape[1]),
    )
    # Diagnostics: intra/inter stats
    try:
        intra_sims: List[float] = []
        inter_sims: List[float] = []
        for idx_c, members in enumerate(clusters):
            if len(members) > 1:
                sub = X_norm[members]
                sims = np.dot(sub, sub.T)
                mask = ~np.eye(len(members), dtype=bool)
                intra_sims.extend(sims[mask].tolist())
            for idx_d, others in enumerate(clusters):
                if idx_d <= idx_c or not others:
                    continue
                sims_cross = np.dot(X_norm[members], X_norm[others].T).flatten().tolist()
                inter_sims.extend(sims_cross)
        if intra_sims and inter_sims:
            logger.info(
                "Artist style clustering stats: artist=%s k=%d clusters=%d intra_median=%.3f inter_median=%.3f",
                artist_name,
                k,
                len(clusters),
                float(np.median(intra_sims)),
                float(np.median(inter_sims)),
            )
    except Exception:
        logger.debug("Failed to compute intra/inter cluster stats", exc_info=True)
    return clusters, medoids, medoids_by_cluster, X_norm


def order_clusters(
    medoid_indices: List[int],
    X_norm: np.ndarray,
) -> List[int]:
    """Order medoids with greedy nearest-neighbor to minimize jumps."""
    if not medoid_indices:
        return []
    remaining = set(medoid_indices)
    order: List[int] = []
    current = medoid_indices[0]
    order.append(current)
    remaining.remove(current)
    while remaining:
        sims = {idx: float(np.dot(X_norm[current], X_norm[idx])) for idx in remaining}
        next_idx = max(sims.items(), key=lambda kv: kv[1])[0]
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
    return order


def build_balanced_candidate_pool(
    *,
    bundle,
    cluster_piers: List[List[int]],
    X_norm: np.ndarray,
    per_cluster_size: int,
    pool_balance_mode: str,
    global_floor: Optional[float],
    artist_key: str,
    artist_name: Optional[str] = None,
    include_collaborations: bool = False,
) -> List[str]:
    """Build union of per-cluster sub-pools with balancing."""
    all_candidates: List[int] = []
    track_ids = bundle.track_ids
    total_tracks = X_norm.shape[0]
    # Mask of tracks that count as the seed artist (incl. collaborations when
    # opted in). These are excluded from the *external* candidate pool because
    # they are intended to be piers, not bridge fillers.
    if include_collaborations and artist_name:
        seed_artist_indices = set(
            _artist_indices_in_bundle(
                bundle, artist_name, include_collaborations=True
            )
        )
        mask_artist = np.array(
            [i in seed_artist_indices for i in range(total_tracks)], dtype=bool
        )
    else:
        mask_artist = np.array([
            normalize_artist_key(str(bundle.artist_keys[i])) == artist_key
            if bundle.artist_keys is not None else False
            for i in range(total_tracks)
        ], dtype=bool)
    for cluster_idx, pier_indices in enumerate(cluster_piers):
        if not pier_indices:
            continue
        pier_vecs = X_norm[pier_indices]
        sims = np.max(np.dot(X_norm, pier_vecs.T), axis=1)
        sims[mask_artist] = -1.0  # exclude artist for external pool
        if global_floor is not None:
            sims = np.where(sims < global_floor, -1.0, sims)
        order = np.argsort(-sims)
        take = []
        for idx in order:
            if sims[idx] < 0:
                break
            take.append(int(idx))
            if len(take) >= per_cluster_size:
                break
        all_candidates.extend(take)
        logger.info(
            "Cluster %d: selected %d/%d external candidates",
            cluster_idx,
            len(take),
            per_cluster_size,
        )
    # balancing: dedupe, preserve insertion; equal already enforced by per_cluster_size
    deduped = list(dict.fromkeys(all_candidates))
    return [str(track_ids[i]) for i in deduped]


def get_internal_connectors(
    bundle,
    artist_key: str,
    exclude_indices: List[int],
    global_floor: Optional[float],
    pier_indices: List[int],
    X_norm: np.ndarray,
) -> List[str]:
    track_ids = bundle.track_ids
    exclude_set = set(exclude_indices)
    connectors: List[int] = []
    artist_keys = bundle.artist_keys if bundle.artist_keys is not None else []
    for i, ak in enumerate(artist_keys):
        if normalize_artist_key(str(ak)) != artist_key:
            continue
        if i in exclude_set:
            continue
        if global_floor is not None:
            sim = float(np.max(np.dot(X_norm[i], X_norm[pier_indices].T)))      
            if sim < global_floor:
                continue
        connectors.append(i)
    return [str(track_ids[i]) for i in connectors]
