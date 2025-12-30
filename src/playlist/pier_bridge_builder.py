"""
Pier + Bridge Playlist Builder
==============================

A new playlist ordering strategy where:
- Each seed track is a fixed "pier"
- Bridge segments connect consecutive piers
- No repair pass after ordering

Key features:
- Candidate pool deduped BEFORE ordering (no duplicate songs by normalized artist+title)
- Genre gating stays enabled with hard floors (no relaxation)
- Global used_track_ids prevents duplicates across segments
- One track per artist per segment (provides implicit min_gap without explicit constraints)
- Single seed mode: seed acts as both start AND end pier, creating an arc structure
- Seed artist is allowed in bridges with same constraints as other artists
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.features.artifacts import ArtifactBundle, get_sonic_matrix
from src.title_dedupe import normalize_title_for_dedupe, normalize_artist_key

logger = logging.getLogger(__name__)


@dataclass
class PierBridgeConfig:
    """Configuration for pier + bridge playlist builder."""
    transition_floor: float = 0.20
    initial_neighbors_m: int = 100
    initial_bridge_helpers: int = 50
    max_neighbors_m: int = 400
    max_bridge_helpers: int = 200
    initial_beam_width: int = 20
    max_beam_width: int = 100
    max_expansion_attempts: int = 4
    eta_destination_pull: float = 0.10
    # Transition scoring weights
    weight_end_start: float = 0.70
    weight_mid_mid: float = 0.15
    weight_full_full: float = 0.15


@dataclass
class SegmentDiagnostics:
    """Diagnostics for a single segment."""
    pier_a_id: str
    pier_b_id: str
    target_length: int
    actual_length: int
    pool_size_initial: int
    pool_size_final: int
    expansions: int
    beam_width_used: int
    worst_edge_score: float
    mean_edge_score: float
    success: bool


@dataclass
class PierBridgeResult:
    """Result of pier + bridge playlist construction."""
    track_ids: List[str]
    track_indices: List[int]
    seed_positions: List[int]  # positions of seeds in final playlist
    segment_diagnostics: List[SegmentDiagnostics]
    stats: Dict[str, Any]
    success: bool = True
    failure_reason: Optional[str] = None


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2 normalize each row of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_transition_score(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
) -> float:
    """
    Compute multi-segment transition score from track A to track B.

    score = w_end_start * cos(end(A), start(B))
          + w_mid_mid * cos(mid(A), mid(B))
          + w_full_full * cos(full(A), full(B))
    """
    full_a = X_full[idx_a]
    full_b = X_full[idx_b]

    # Full-full similarity
    sim_full = _cosine_sim(full_a, full_b)

    # End-start similarity (use full as fallback)
    if X_end is not None and X_start is not None:
        sim_end_start = _cosine_sim(X_end[idx_a], X_start[idx_b])
    else:
        sim_end_start = sim_full

    # Mid-mid similarity (use full as fallback)
    if X_mid is not None:
        sim_mid = _cosine_sim(X_mid[idx_a], X_mid[idx_b])
    else:
        sim_mid = sim_full

    return (
        cfg.weight_end_start * sim_end_start +
        cfg.weight_mid_mid * sim_mid +
        cfg.weight_full_full * sim_full
    )


def _compute_bridgeability_score(
    idx_a: int,
    idx_b: int,
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
) -> float:
    """
    Cheap heuristic for how well two seeds can be bridged.
    Uses direct transition similarity plus a term for the distance between them.
    """
    # Direct transition similarity
    if X_end_norm is not None and X_start_norm is not None:
        direct_sim = float(np.dot(X_end_norm[idx_a], X_start_norm[idx_b]))
    else:
        direct_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Full similarity (for overall coherence)
    full_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Combine: favor pairs with good direct transitions
    return 0.6 * direct_sim + 0.4 * full_sim


def _order_seeds_by_bridgeability(
    seed_indices: List[int],
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
) -> List[int]:
    """
    Order seed indices to maximize total bridgeability.
    For <=6 seeds, evaluates all permutations.
    For >6 seeds, uses greedy nearest-neighbor heuristic.
    """
    n = len(seed_indices)
    if n <= 1:
        return seed_indices

    if n <= 6:
        # Exhaustive search for small seed counts
        best_order = None
        best_score = -float('inf')

        for perm in itertools.permutations(seed_indices):
            total_score = 0.0
            for i in range(len(perm) - 1):
                total_score += _compute_bridgeability_score(
                    perm[i], perm[i + 1],
                    X_full_norm, X_start_norm, X_end_norm
                )
            if total_score > best_score:
                best_score = total_score
                best_order = list(perm)

        logger.info("Seed ordering: evaluated %d permutations, best_score=%.4f",
                   math.factorial(n), best_score)
        return best_order or seed_indices
    else:
        # Greedy nearest-neighbor for larger seed counts
        remaining = set(seed_indices)
        # Start with the first seed
        ordered = [seed_indices[0]]
        remaining.remove(seed_indices[0])

        while remaining:
            current = ordered[-1]
            best_next = None
            best_score = -float('inf')

            for candidate in remaining:
                score = _compute_bridgeability_score(
                    current, candidate,
                    X_full_norm, X_start_norm, X_end_norm
                )
                if score > best_score:
                    best_score = score
                    best_next = candidate

            if best_next is not None:
                ordered.append(best_next)
                remaining.remove(best_next)

        logger.info("Seed ordering: greedy heuristic for %d seeds", n)
        return ordered


def _dedupe_candidate_pool(
    pool_indices: List[int],
    bundle: ArtifactBundle,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Deduplicate candidate pool by normalized artist+title.
    Returns deduplicated indices and mapping of norm_key -> chosen index.

    Prefers canonical versions based on version preference scoring.
    """
    from src.title_dedupe import calculate_version_preference_score

    seen: Dict[str, Tuple[int, int]] = {}  # norm_key -> (index, preference_score)

    for idx in pool_indices:
        artist = str(bundle.artist_keys[idx]) if bundle.artist_keys is not None else ""
        title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""

        artist_norm = normalize_artist_key(artist)
        title_norm = normalize_title_for_dedupe(title, mode="loose")
        key = f"{artist_norm}|||{title_norm}"

        # Compute preference score (higher = more canonical)
        pref_score = calculate_version_preference_score(title)

        if key not in seen or pref_score > seen[key][1]:
            seen[key] = (idx, pref_score)

    deduped = [idx for idx, _ in seen.values()]
    norm_to_idx = {key: idx for key, (idx, _) in seen.items()}

    logger.debug("Deduped candidate pool: %d -> %d tracks", len(pool_indices), len(deduped))
    return deduped, norm_to_idx


def _build_segment_candidate_pool(
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    universe_indices: List[int],
    used_track_ids: Set[int],
    neighbors_m: int,
    bridge_helpers: int,
    artist_keys: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Build candidate subset for a segment from pier_a to pier_b.

    Includes:
    - Top M neighbors of pier_a by full similarity
    - Top M neighbors of pier_b by full similarity
    - Top B "bridge helper" tracks by two-sided bridge score

    Only ONE track per artist is allowed in the segment.
    This prevents artist clustering without needing min_gap constraints.
    All artists (including seed artist) follow the same one-per-segment rule.
    """
    # Filter out used tracks
    available = [idx for idx in universe_indices if idx not in used_track_ids]
    if not available:
        return []

    # Compute similarities to both piers
    vec_a = X_full_norm[pier_a]
    vec_b = X_full_norm[pier_b]

    sim_to_a = {}
    sim_to_b = {}
    bridge_score = {}

    for idx in available:
        sim_a = float(np.dot(X_full_norm[idx], vec_a))
        sim_b = float(np.dot(X_full_norm[idx], vec_b))
        sim_to_a[idx] = sim_a
        sim_to_b[idx] = sim_b
        # Bridge score: geometric mean of similarities to both piers
        bridge_score[idx] = math.sqrt(max(0, sim_a) * max(0, sim_b))

    # Top M neighbors of pier_a
    sorted_by_a = sorted(available, key=lambda i: sim_to_a[i], reverse=True)
    neighbors_a = set(sorted_by_a[:neighbors_m])

    # Top M neighbors of pier_b
    sorted_by_b = sorted(available, key=lambda i: sim_to_b[i], reverse=True)
    neighbors_b = set(sorted_by_b[:neighbors_m])

    # Top B bridge helpers
    sorted_by_bridge = sorted(available, key=lambda i: bridge_score[i], reverse=True)
    helpers = set(sorted_by_bridge[:bridge_helpers])

    # Combine all candidates
    combined = neighbors_a | neighbors_b | helpers

    # Dedupe to ONE track per artist (all artists treated equally, including seed artist)
    if artist_keys is not None:
        artist_best: Dict[str, Tuple[int, float]] = {}  # artist -> (idx, score)
        for idx in combined:
            artist = normalize_artist_key(str(artist_keys[idx]))
            score = bridge_score.get(idx, 0.0)

            if artist not in artist_best or score > artist_best[artist][1]:
                artist_best[artist] = (idx, score)

        # Build final pool: one track per artist (normalized)
        deduped: List[int] = [idx for idx, _ in artist_best.values()]

        logger.debug("Segment pool: %d combined -> %d after 1-per-artist dedupe",
                     len(combined), len(deduped))
        return deduped

    return list(combined)


@dataclass
class BeamState:
    """State for beam search."""
    path: List[int]
    score: float
    used: Set[int]


def _beam_search_segment(
    pier_a: int,
    pier_b: int,
    interior_length: int,
    candidates: List[int],
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    beam_width: int,
) -> Optional[List[int]]:
    """
    Constrained beam search to find path from pier_a to pier_b.

    Returns interior track indices (not including piers) or None if no path found.
    """

    if interior_length == 0:
        # Check if direct transition meets floor
        direct_score = _compute_transition_score(
            pier_a, pier_b, X_full, X_start, X_mid, X_end, cfg
        )
        if direct_score >= cfg.transition_floor:
            return []
        else:
            return None

    # Pre-normalize matrices for efficiency
    X_full_norm = _l2_normalize_rows(X_full)
    vec_b_full = X_full_norm[pier_b]

    # Initialize beam with pier_a
    initial_state = BeamState(
        path=[pier_a],
        score=0.0,
        used={pier_a, pier_b},
    )
    beam = [initial_state]

    for step in range(interior_length):
        next_beam: List[BeamState] = []

        for state in beam:
            current = state.path[-1]

            for cand in candidates:
                if cand in state.used:
                    continue

                # Compute transition score
                trans_score = _compute_transition_score(
                    current, cand, X_full, X_start, X_mid, X_end, cfg
                )

                # Hard floor enforcement
                if trans_score < cfg.transition_floor:
                    continue

                # Add heuristic pull toward destination
                dest_pull = cfg.eta_destination_pull * float(np.dot(X_full_norm[cand], vec_b_full))

                new_score = state.score + trans_score + dest_pull
                new_path = state.path + [cand]
                new_used = state.used | {cand}

                next_beam.append(BeamState(
                    path=new_path,
                    score=new_score,
                    used=new_used,
                ))

        if not next_beam:
            return None  # No valid continuations

        # Keep top beam_width states
        next_beam.sort(key=lambda s: s.score, reverse=True)
        beam = next_beam[:beam_width]

    # Final step: connect to pier_b
    best_final: Optional[BeamState] = None
    best_final_score = -float('inf')

    for state in beam:
        last = state.path[-1]
        final_trans = _compute_transition_score(
            last, pier_b, X_full, X_start, X_mid, X_end, cfg
        )

        # Hard floor on final transition
        if final_trans < cfg.transition_floor:
            continue

        total_score = state.score + final_trans
        if total_score > best_final_score:
            best_final_score = total_score
            best_final = state

    if best_final is None:
        return None

    # Return interior tracks (exclude pier_a which is path[0])
    return best_final.path[1:]


def _compute_edge_scores(
    path: List[int],
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
) -> Tuple[float, float]:
    """Compute worst and mean edge scores for a path."""
    if len(path) < 2:
        return (1.0, 1.0)

    scores = []
    for i in range(len(path) - 1):
        score = _compute_transition_score(
            path[i], path[i + 1], X_full, X_start, X_mid, X_end, cfg
        )
        scores.append(score)

    return (min(scores), sum(scores) / len(scores))


def _enforce_min_gap_global(
    indices: List[int],
    artist_keys: Optional[np.ndarray],
    min_gap: int = 1,
) -> Tuple[List[int], int]:
    """
    Drop tracks that would violate a global min_gap across concatenated segments.

    Pier-bridge already enforces one-per-artist per segment, but adjacent
    duplicates can appear at segment boundaries. This pass removes any track
    that would repeat a normalized artist within the last `min_gap` slots.
    """
    if not indices or artist_keys is None or min_gap <= 0:
        return indices, 0

    recent: List[str] = []
    output: List[int] = []
    dropped = 0

    for idx in indices:
        key = normalize_artist_key(str(artist_keys[idx]))
        if key in recent:
            dropped += 1
            continue

        output.append(idx)
        recent.append(key)
        if len(recent) > min_gap:
            recent.pop(0)

    return output, dropped


def build_pier_bridge_playlist(
    *,
    seed_track_ids: List[str],
    total_tracks: int,
    bundle: ArtifactBundle,
    candidate_pool_indices: List[int],
    cfg: Optional[PierBridgeConfig] = None,
    min_genre_similarity: Optional[float] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_method: str = "ensemble",
) -> PierBridgeResult:
    """
    Build playlist using pier + bridge strategy.

    Args:
        seed_track_ids: List of seed track IDs (will become piers)
        total_tracks: Target total playlist length
        bundle: Artifact bundle with sonic features
        candidate_pool_indices: Pre-filtered candidate pool indices
        cfg: Configuration (uses defaults if None)
        min_genre_similarity: Optional genre gate threshold
        X_genre_smoothed: Genre vectors for gating
        genre_method: Genre similarity method

    Returns:
        PierBridgeResult with ordered track IDs and diagnostics
    """
    if cfg is None:
        cfg = PierBridgeConfig()

    num_seeds = len(seed_track_ids)
    if num_seeds == 0:
        raise ValueError("At least one seed is required")
    if num_seeds > total_tracks:
        raise ValueError(f"Number of seeds ({num_seeds}) exceeds total_tracks ({total_tracks})")

    # Resolve seed indices
    seed_indices: List[int] = []
    for tid in seed_track_ids:
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is None:
            raise ValueError(f"Seed track not found in bundle: {tid}")
        seed_indices.append(idx)

    # Remove duplicates while preserving order
    seed_indices = list(dict.fromkeys(seed_indices))
    num_seeds = len(seed_indices)
    seed_id_set = {str(bundle.track_ids[i]) for i in seed_indices}

    logger.info("Pier+Bridge: %d seeds, target %d tracks", num_seeds, total_tracks)

    # Deduplicate candidate pool by artist+title
    deduped_pool, _ = _dedupe_candidate_pool(candidate_pool_indices, bundle)

    # Exclude seed indices from candidate pool
    seed_set = set(seed_indices)
    universe = [idx for idx in deduped_pool if idx not in seed_set]

    logger.info("Pier+Bridge: universe size after dedupe and seed exclusion: %d", len(universe))

    # Get sonic matrices
    X_full = bundle.X_sonic
    X_start = bundle.X_sonic_start
    X_mid = bundle.X_sonic_mid
    X_end = bundle.X_sonic_end

    # Normalize for similarity computations
    X_full_norm = _l2_normalize_rows(X_full)
    X_start_norm = _l2_normalize_rows(X_start) if X_start is not None else None
    X_end_norm = _l2_normalize_rows(X_end) if X_end is not None else None

    # Order seeds by bridgeability
    ordered_seeds = _order_seeds_by_bridgeability(
        seed_indices, X_full_norm, X_start_norm, X_end_norm
    )

    logger.info("Pier+Bridge: seed order = %s",
               [str(bundle.track_ids[i]) for i in ordered_seeds])

    # Handle single seed as both start AND end pier (arc structure)
    # This creates a playlist that starts from seed, explores, and returns to seed-similar sounds
    is_single_seed_arc = (num_seeds == 1)
    if is_single_seed_arc:
        # Duplicate the seed as both start and end pier
        ordered_seeds = [ordered_seeds[0], ordered_seeds[0]]
        num_segments = 1
        total_interior = total_tracks - 1  # Only one seed in final output
        logger.info("Pier+Bridge: single-seed arc mode (seed is both start and end pier)")
    else:
        num_segments = num_seeds - 1
        total_interior = total_tracks - num_seeds

    # Even split with remainder distributed to earlier segments
    base_length = total_interior // num_segments
    remainder = total_interior % num_segments
    segment_lengths = [
        base_length + (1 if i < remainder else 0)
        for i in range(num_segments)
    ]

    logger.info("Pier+Bridge: segment lengths = %s (total_interior=%d)",
               segment_lengths, total_interior)

    # Build segments
    global_used: Set[int] = set(ordered_seeds)  # Seeds are already "used"
    all_segments: List[List[int]] = []
    diagnostics: List[SegmentDiagnostics] = []

    for seg_idx in range(num_segments):
        pier_a = ordered_seeds[seg_idx]
        pier_b = ordered_seeds[seg_idx + 1]
        interior_len = segment_lengths[seg_idx]

        pier_a_id = str(bundle.track_ids[pier_a])
        pier_b_id = str(bundle.track_ids[pier_b])

        logger.info("Building segment %d: %s -> %s (interior=%d)",
                   seg_idx, pier_a_id, pier_b_id, interior_len)

        # Try building with progressively larger pools
        segment_path: Optional[List[int]] = None
        neighbors_m = cfg.initial_neighbors_m
        bridge_helpers = cfg.initial_bridge_helpers
        beam_width = cfg.initial_beam_width
        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0

        for attempt in range(cfg.max_expansion_attempts):
            # Build segment candidate pool (one track per artist, all artists equal)
            segment_candidates = _build_segment_candidate_pool(
                pier_a, pier_b,
                X_full_norm, universe,
                global_used,
                neighbors_m, bridge_helpers,
                artist_keys=bundle.artist_keys,
            )

            if attempt == 0:
                pool_size_initial = len(segment_candidates)
            pool_size_final = len(segment_candidates)

            if len(segment_candidates) < interior_len:
                logger.warning("Segment %d: pool size %d < interior_len %d",
                             seg_idx, len(segment_candidates), interior_len)

            # Run beam search
            segment_path = _beam_search_segment(
                pier_a, pier_b, interior_len,
                segment_candidates,
                X_full, X_start, X_mid, X_end,
                cfg, beam_width,
            )

            if segment_path is not None:
                break

            # Expand for next attempt
            expansions += 1
            neighbors_m = min(neighbors_m * 2, cfg.max_neighbors_m)
            bridge_helpers = min(bridge_helpers * 2, cfg.max_bridge_helpers)
            beam_width = min(beam_width * 2, cfg.max_beam_width)

            logger.info("Segment %d: expanding pool (attempt %d, neighbors=%d, helpers=%d, beam=%d)",
                       seg_idx, attempt + 1, neighbors_m, bridge_helpers, beam_width)

        if segment_path is None:
            logger.error("Segment %d: failed to find valid path after %d attempts",
                        seg_idx, cfg.max_expansion_attempts)
            # Use empty interior as fallback
            segment_path = []

        # Compute edge scores for diagnostics
        full_segment = [pier_a] + segment_path + [pier_b]
        worst_edge, mean_edge = _compute_edge_scores(
            full_segment, X_full, X_start, X_mid, X_end, cfg
        )

        # Record diagnostics
        diagnostics.append(SegmentDiagnostics(
            pier_a_id=pier_a_id,
            pier_b_id=pier_b_id,
            target_length=interior_len,
            actual_length=len(segment_path),
            pool_size_initial=pool_size_initial,
            pool_size_final=pool_size_final,
            expansions=expansions,
            beam_width_used=beam_width,
            worst_edge_score=worst_edge,
            mean_edge_score=mean_edge,
            success=segment_path is not None and len(segment_path) == interior_len,
        ))

        # Commit segment path to used set
        for idx in segment_path:
            global_used.add(idx)

        all_segments.append(full_segment)

    # Concatenate segments
    # First segment: keep full [A, ..., B]
    # Subsequent segments: drop first element (the pier) to avoid duplication
    # Single-seed arc: drop last element (the duplicated seed) to avoid repetition
    final_indices: List[int] = []
    seed_positions: List[int] = []

    if is_single_seed_arc:
        # Single-seed arc: segment is [seed, interior..., seed]
        # Output only [seed, interior...] to avoid duplicate seed at end
        segment = all_segments[0] if all_segments else [ordered_seeds[0]]
        final_indices = segment[:-1]  # Drop the trailing duplicate seed
        seed_positions = [0]  # Seed is at position 0
        logger.info("Pier+Bridge: single-seed arc output: %d tracks (seed at start, arc returns to seed-similar)", len(final_indices))
    else:
        for seg_idx, segment in enumerate(all_segments):
            if seg_idx == 0:
                final_indices.extend(segment)
                seed_positions.append(0)  # First pier
                seed_positions.append(len(final_indices) - 1)  # Second pier
            else:
                # Drop first element (the pier, already included)
                final_indices.extend(segment[1:])
                seed_positions.append(len(final_indices) - 1)  # New pier

    # Convert to track IDs (after enforcing cross-segment min_gap to avoid back-to-back repeats)
    final_indices, dropped = _enforce_min_gap_global(
        final_indices, bundle.artist_keys, min_gap=1
    )
    if dropped:
        logger.debug(
            "Pier+Bridge: dropped %d tracks to enforce cross-segment min_gap", dropped
        )

    final_track_ids = [str(bundle.track_ids[i]) for i in final_indices]

    # Recompute seed positions after any min-gap pruning to keep diagnostics consistent
    seed_positions = [idx for idx, tid in enumerate(final_track_ids) if tid in seed_id_set]
    if len(seed_positions) != (1 if is_single_seed_arc else len(seed_id_set)):
        logger.debug(
            "Pier+Bridge: seed count mismatch after pruning (expected %d, found %d)",
            (1 if is_single_seed_arc else len(seed_id_set)),
            len(seed_positions),
        )

    # Compute overall stats
    actual_num_seeds = 1 if is_single_seed_arc else len(seed_indices)
    stats = {
        "num_seeds": actual_num_seeds,
        "single_seed_arc": is_single_seed_arc,
        "target_tracks": total_tracks,
        "actual_tracks": len(final_indices),
        "universe_size": len(universe),
        "segments_built": len(all_segments),
        "segments_successful": sum(1 for d in diagnostics if d.success),
        "total_expansions": sum(d.expansions for d in diagnostics),
        "config": {
            "transition_floor": cfg.transition_floor,
            "initial_neighbors_m": cfg.initial_neighbors_m,
            "initial_beam_width": cfg.initial_beam_width,
            "eta_destination_pull": cfg.eta_destination_pull,
        },
    }

    logger.info("Pier+Bridge complete: %d tracks, %d segments, %d successful",
               len(final_indices), len(all_segments),
               sum(1 for d in diagnostics if d.success))

    return PierBridgeResult(
        track_ids=final_track_ids,
        track_indices=final_indices,
        seed_positions=seed_positions,
        segment_diagnostics=diagnostics,
        stats=stats,
    )


def generate_pier_bridge_playlist(
    *,
    artifact_path: str,
    seed_track_ids: List[str],
    total_tracks: int,
    mode: str = "dynamic",
    random_seed: int = 0,
    min_genre_similarity: Optional[float] = None,
    genre_method: str = "ensemble",
    transition_floor: Optional[float] = None,
) -> PierBridgeResult:
    """
    High-level entry point for pier+bridge playlist generation.

    Loads artifacts, builds candidate pool, and runs pier+bridge construction.
    """
    from src.features.artifacts import load_artifact_bundle
    from src.playlist.config import default_ds_config
    from src.playlist.candidate_pool import build_candidate_pool
    from src.similarity.hybrid import build_hybrid_embedding
    from src.similarity.sonic_variant import compute_sonic_variant_matrix, resolve_sonic_variant

    bundle = load_artifact_bundle(artifact_path)

    # Validate seeds
    valid_seed_ids = []
    for tid in seed_track_ids:
        if str(tid) in bundle.track_id_to_index:
            valid_seed_ids.append(str(tid))
        else:
            logger.warning("Seed track not found, skipping: %s", tid)

    if not valid_seed_ids:
        raise ValueError("No valid seed tracks found in artifact bundle")

    seed_idx = bundle.track_id_to_index[valid_seed_ids[0]]

    # Build config
    cfg = default_ds_config(mode, playlist_len=total_tracks)

    # Build embedding
    resolved_variant = resolve_sonic_variant()
    X_sonic_for_embed, _ = compute_sonic_variant_matrix(bundle.X_sonic, resolved_variant, l2=False)

    embedding_model = build_hybrid_embedding(
        X_sonic_for_embed,
        bundle.X_genre_smoothed,
        n_components_sonic=32,
        n_components_genre=32,
        w_sonic=0.6,
        w_genre=0.4,
        random_seed=random_seed,
    )

    # Build candidate pool (for genre gating)
    pool = build_candidate_pool(
        seed_idx=seed_idx,
        embedding=embedding_model.embedding,
        artist_keys=bundle.artist_keys,
        cfg=cfg.candidate,
        random_seed=random_seed,
        X_genre_raw=bundle.X_genre_raw if min_genre_similarity else None,
        X_genre_smoothed=bundle.X_genre_smoothed if min_genre_similarity else None,
        min_genre_similarity=min_genre_similarity,
        genre_method=genre_method,
        mode=mode,
    )

    # Build pier config
    pier_cfg = PierBridgeConfig()
    if transition_floor is not None:
        pier_cfg = PierBridgeConfig(transition_floor=transition_floor)
    else:
        pier_cfg = PierBridgeConfig(transition_floor=cfg.construct.transition_floor)

    return build_pier_bridge_playlist(
        seed_track_ids=valid_seed_ids,
        total_tracks=total_tracks,
        bundle=bundle,
        candidate_pool_indices=list(pool.pool_indices),
        cfg=pier_cfg,
        min_genre_similarity=min_genre_similarity,
        X_genre_smoothed=bundle.X_genre_smoothed,
        genre_method=genre_method,
    )
