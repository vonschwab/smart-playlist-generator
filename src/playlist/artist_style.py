from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.string_utils import normalize_artist_key

logger = logging.getLogger(__name__)


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
    bridge_floor_narrow: float = 0.08
    bridge_floor_dynamic: float = 0.03
    bridge_weight: float = 0.7
    transition_weight: float = 0.3
    genre_tiebreak_weight: float = 0.05


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


def _medoids_for_cluster(
    X: np.ndarray,
    indices: List[int],
    centroid: np.ndarray,
    bundle_track_ids: Sequence[str],
    per_cluster: int,
) -> List[int]:
    if not indices:
        return []
    sims = np.dot(X[indices], centroid)
    order = np.argsort(-sims)
    medoids: List[int] = [indices[int(order[0])]]
    if per_cluster > 1 and len(indices) > 1:
        # pick farthest from first medoid to diversify
        first_vec = X[medoids[0]]
        sims2 = np.dot(X[indices], first_vec)
        order2 = np.argsort(sims2)
        for idx in order2:
            cand = indices[int(idx)]
            if cand not in medoids:
                medoids.append(cand)
            if len(medoids) >= per_cluster:
                break
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
) -> Tuple[List[List[int]], List[int], List[List[int]], np.ndarray]:
    """Cluster artist tracks in sonic space and return clusters + medoids."""   
    artist_key = normalize_artist_key(artist_name)
    track_ids = bundle.track_ids
    if bundle.artist_keys is None:
        raise ValueError("Artifact missing artist_keys for clustering.")        
    X_raw = getattr(bundle, "X_sonic", None)
    if X_raw is None:
        raise ValueError("Artifact missing X_sonic for clustering.")
    from src.similarity.sonic_variant import compute_sonic_variant_norm, resolve_sonic_variant

    variant = resolve_sonic_variant(explicit_variant=sonic_variant, config_variant=None)
    X_norm, variant_stats = compute_sonic_variant_norm(X_raw, variant)
    artist_indices = [
        i for i, ak in enumerate(bundle.artist_keys)
        if normalize_artist_key(str(ak)) == artist_key
    ]
    if len(artist_indices) < max(3, cfg.cluster_k_min):
        raise ValueError(f"Not enough tracks to cluster for artist {artist_name}")
    k = _select_k(len(artist_indices), cfg)
    rng = np.random.default_rng(random_seed)
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
            cfg.piers_per_cluster,
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
) -> List[str]:
    """Build union of per-cluster sub-pools with balancing."""
    all_candidates: List[int] = []
    track_ids = bundle.track_ids
    total_tracks = X_norm.shape[0]
    for cluster_idx, pier_indices in enumerate(cluster_piers):
        if not pier_indices:
            continue
        pier_vecs = X_norm[pier_indices]
        sims = np.max(np.dot(X_norm, pier_vecs.T), axis=1)
        mask_artist = np.array([
            normalize_artist_key(str(bundle.artist_keys[i])) == artist_key
            if bundle.artist_keys is not None else False
            for i in range(total_tracks)
        ], dtype=bool)
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
