from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np

from src.features.artifacts import ArtifactBundle, get_sonic_matrix
from src.playlist.candidate_pool import CandidatePoolResult
from src.playlist.config import DSPipelineConfig
from src.similarity.hybrid import HybridEmbeddingModel, cosine_sim_matrix_to_vector, transition_similarity_end_to_start
from src.similarity.sonic_variant import compute_sonic_variant_norm, get_variant_from_env, resolve_sonic_variant, apply_transition_weights
from src.title_dedupe import normalize_title_for_dedupe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaylistResult:
    track_indices: np.ndarray  # ordered indices, length L
    stats: Dict[str, Any]
    params_requested: Dict[str, Any]
    params_effective: Dict[str, Any]


def _alpha_for_position(
    schedule: str,
    start: float,
    mid: float,
    end: float,
    arc_midpoint: float,
    position: int,
    length: int,
) -> float:
    if schedule == "constant" or length <= 1:
        return start
    p = position / max(1, length - 1)
    if p <= arc_midpoint:
        if arc_midpoint == 0:
            return mid
        frac = p / arc_midpoint
        return start + frac * (mid - start)
    frac = (p - arc_midpoint) / max(1e-9, 1 - arc_midpoint)
    return mid + frac * (end - mid)


def _compute_local_sim(
    prev_idx: int,
    cand_indices: np.ndarray,
    emb_norm: np.ndarray,
    X_end: Optional[np.ndarray],
    X_start: Optional[np.ndarray],
    transition_gamma: float,
    rescale01: bool,
) -> np.ndarray:
    if X_end is not None and X_start is not None:
        seg = transition_similarity_end_to_start(X_end, X_start, prev_idx, cand_indices)
        if rescale01:
            seg = np.clip((seg + 1.0) / 2.0, 0.0, 1.0)
        hyb = emb_norm[cand_indices] @ emb_norm[prev_idx]
        return transition_gamma * seg + (1 - transition_gamma) * hyb
    return emb_norm[cand_indices] @ emb_norm[prev_idx]


def _transition_array(
    order: Sequence[int],
    emb_norm: np.ndarray,
    X_end: Optional[np.ndarray],
    X_start: Optional[np.ndarray],
    transition_gamma: float,
    rescale01: bool,
) -> np.ndarray:
    vals = []
    for i in range(1, len(order)):
        prev_idx = order[i - 1]
        cur_idx = order[i]
        if X_end is not None and X_start is not None:
            seg = float(transition_similarity_end_to_start(X_end, X_start, prev_idx, np.array([cur_idx]))[0])
            if rescale01:
                seg = float(np.clip((seg + 1.0) / 2.0, 0.0, 1.0))
            hyb = float(emb_norm[cur_idx] @ emb_norm[prev_idx])
            vals.append(transition_gamma * seg + (1 - transition_gamma) * hyb)
        else:
            vals.append(float(emb_norm[cur_idx] @ emb_norm[prev_idx]))
    return np.array(vals, dtype=float)


def _check_constraints(order: Sequence[int], artist_keys: Sequence[str], max_per_artist: int, min_gap: int) -> Dict[str, int]:
    from src.string_utils import normalize_artist_key

    counts: Dict[str, int] = {}
    adjacency = 0
    gap = 0
    cap = 0
    recent: list[str] = []
    for idx in order:
        # Normalize artist name to handle Unicode correctly (e.g., きゃりーぱみゅぱみゅ)
        artist = normalize_artist_key(str(artist_keys[idx]))
        if recent and artist == recent[-1]:
            adjacency += 1
        if min_gap > 0 and artist in recent[-min_gap:]:
            gap += 1
        counts[artist] = counts.get(artist, 0) + 1
        if counts[artist] > max_per_artist:
            cap += 1
        recent.append(artist)
    return {"adjacency": adjacency, "gap": gap, "cap": cap}


def construct_playlist(
    *,
    seed_idx: int,
    pool: CandidatePoolResult,
    bundle: ArtifactBundle,
    embedding_model: HybridEmbeddingModel,
    cfg: DSPipelineConfig,
    random_seed: int,
    sonic_variant: Optional[str] = None,
    initial_order: Optional[Sequence[int]] = None,
    locked_track_ids: Optional[Set[int]] = None,
    transition_weights: Optional[tuple] = None,
) -> PlaylistResult:
    """
    Greedy construction with constraints and optional repair.

    Args:
        transition_weights: Optional (rhythm, timbre, harmony) weights from config.
                            If None, uses env var or defaults (0.4, 0.35, 0.25).
    """
    rng = np.random.default_rng(random_seed)
    emb = embedding_model.embedding
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    X_start = None
    X_end = None
    X_start_orig = None
    X_end_orig = None
    transition_weight_stats: dict = {}
    if bundle.X_sonic_start is not None and bundle.X_sonic_end is not None:
        X_start_orig = get_sonic_matrix(bundle, "start")
        X_end_orig = get_sonic_matrix(bundle, "end")
        # Apply transition-specific tower weights (rhythm-heavy for BPM flow)
        X_start, start_stats = apply_transition_weights(X_start_orig, config_weights=transition_weights)
        X_end, end_stats = apply_transition_weights(X_end_orig, config_weights=transition_weights)
        transition_weight_stats = {
            "start": start_stats,
            "end": end_stats,
        }
        if start_stats.get("transition_weights_applied"):
            logger.debug(
                "Applied transition weights: rhythm=%.2f timbre=%.2f harmony=%.2f",
                *start_stats.get("transition_weights", (0, 0, 0))
            )
        if X_start.shape[0] != emb.shape[0] or X_end.shape[0] != emb.shape[0]:
            X_start = None
            X_end = None
            X_start_orig = None
            X_end_orig = None
    rescale_transitions = False
    if cfg.construct.center_transitions and X_start is not None and X_end is not None:
        mu_end = X_end.mean(axis=0, keepdims=True)
        mu_start = X_start.mean(axis=0, keepdims=True)
        X_end = X_end - mu_end
        X_start = X_start - mu_start
        rescale_transitions = True

    playlist_len = int(pool.stats.get("target_length", 0)) or (len(pool.pool_indices) + 1)
    max_per_artist = max(1, math.ceil(playlist_len * cfg.construct.max_artist_fraction_final))
    min_gap = cfg.construct.min_gap
    transition_floor = cfg.construct.transition_floor
    hard_floor = cfg.construct.hard_floor
    transition_gamma = cfg.construct.transition_gamma

    from src.string_utils import normalize_artist_key
    track_indices: list[int] = []
    used_counts: Dict[str, int] = {}
    recent_artists: list[str] = []
    used_set: set[int] = set()
    provided_order = initial_order is not None
    locked_ids = set(locked_track_ids or set())
    if provided_order:
        track_indices = list(initial_order)[:playlist_len]
        for idx in track_indices:
            artist = normalize_artist_key(str(bundle.artist_keys[idx]))
            used_counts[artist] = used_counts.get(artist, 0) + 1
            recent_artists.append(artist)
        recent_artists = recent_artists[-max(min_gap, 1) * 2 :] if recent_artists else []
        used_set = set(track_indices)
    else:
        seed_artist = normalize_artist_key(str(bundle.artist_keys[seed_idx]))
        used_counts = {seed_artist: 1}
        recent_artists = [seed_artist]
        track_indices = [seed_idx]
        used_set = {seed_idx}

    seed_sim_lookup = {idx: sim for idx, sim in zip(pool.pool_indices, pool.seed_sim)}

    # Phase C diagnostic counters
    below_floor_count = 0
    gap_sum = 0.0
    gap_values: list[float] = []
    hard_floor_relaxed = False
    fallback_steps = 0
    total_candidates_evaluated = 0
    rejected_by_floor = 0
    penalty_applied_count = 0

    if not provided_order:
        while len(track_indices) < playlist_len:
            remaining = [idx for idx in pool.pool_indices if idx not in used_set]
            if not remaining:
                break
            prev_idx = track_indices[-1]

            best_choice: Optional[int] = None
            best_score = -1e9
            best_local = float("nan")
            best_seed_sim = float("nan")
            alpha_t = _alpha_for_position(
                cfg.construct.alpha_schedule,
                cfg.construct.alpha_start,
                cfg.construct.alpha_mid,
                cfg.construct.alpha_end,
                cfg.construct.arc_midpoint,
                len(track_indices),
                playlist_len,
            )

            for relax in range(4):
                # Enforce constraints with progressive relaxation
                from src.string_utils import normalize_artist_key
                candidate_list: list[int] = []
                for idx in remaining:
                    # Normalize artist name to handle Unicode correctly (e.g., きゃりーぱみゅぱみゅ)
                    artist = normalize_artist_key(str(bundle.artist_keys[idx]))
                    if relax < 2 and artist == recent_artists[-1]:
                        continue
                    if relax < 1 and min_gap > 0 and artist in recent_artists[-min_gap:]:
                        continue
                    cap_limit = max_per_artist + (1 if relax >= 3 else 0)
                    if used_counts.get(artist, 0) >= cap_limit:
                        continue
                    candidate_list.append(idx)
                if not candidate_list:
                    continue

                cand_arr = np.array(candidate_list, dtype=int)
                local_sims = _compute_local_sim(prev_idx, cand_arr, emb_norm, X_end, X_start, transition_gamma, rescale_transitions)

                # Phase C diagnostics: count all candidates evaluated
                total_candidates_evaluated += len(cand_arr)

                # Hard floor pruning if requested
                if hard_floor:
                    mask = local_sims >= transition_floor
                    candidates_below_floor = (~mask).sum()
                    rejected_by_floor += int(candidates_below_floor)

                    if not np.any(mask):
                        hard_floor_relaxed = True
                    else:
                        cand_arr = cand_arr[mask]
                        local_sims = local_sims[mask]
                        if cand_arr.size == 0:
                            continue

                seed_sims = np.array([seed_sim_lookup.get(i, float(emb_norm[i] @ emb_norm[seed_idx])) for i in cand_arr])
                diversity = np.array([1.0 if normalize_artist_key(str(bundle.artist_keys[i])) not in used_counts else 0.0 for i in cand_arr])

                if cand_arr.size == 0:
                    continue

                penalty = np.zeros_like(local_sims)
                if not hard_floor:
                    gaps = np.maximum(0.0, transition_floor - local_sims)
                    penalty = gaps
                    # Phase C diagnostics: count soft penalties applied
                    penalty_applied_count += int((gaps > 0).sum())
                scores = alpha_t * seed_sims + cfg.construct.beta * local_sims + cfg.construct.gamma * diversity - penalty

                # Deterministic tie-breaking
                order = np.lexsort((-cand_arr, -scores))
                scores = scores[order]
                cand_arr = cand_arr[order]
                local_sims = local_sims[order]
                seed_sims = seed_sims[order]

                if scores.size > 0 and scores[0] > best_score:
                    best_score = float(scores[0])
                    best_choice = int(cand_arr[0])
                    best_local = float(local_sims[0])
                    best_seed_sim = float(seed_sims[0])
                    if relax > 0:
                        fallback_steps += 1
                if best_choice is not None:
                    break  # prefer first relaxation level that yields a candidate

            if best_choice is None:
                break

            track_indices.append(best_choice)
            used_set.add(best_choice)
            artist = normalize_artist_key(str(bundle.artist_keys[best_choice]))
            used_counts[artist] = used_counts.get(artist, 0) + 1
            recent_artists.append(artist)
            if len(recent_artists) > max(min_gap, 1) * 2:
                recent_artists = recent_artists[-max(min_gap, 1) * 2 :]

            if not hard_floor and best_local < transition_floor:
                below_floor_count += 1
                gap = max(0.0, transition_floor - best_local)
                gap_sum += gap
                gap_values.append(gap)
            if hard_floor and best_local < transition_floor:
                hard_floor_relaxed = True

    order_array = np.array(track_indices, dtype=int)
    transitions = _transition_array(order_array, emb_norm, X_end, X_start, transition_gamma, rescale_transitions)
    # Recompute artist counts for provided orders to keep stats consistent.
    used_counts = {}
    for idx in order_array:
        artist = normalize_artist_key(str(bundle.artist_keys[idx]))
        used_counts[artist] = used_counts.get(artist, 0) + 1
    artist_counts = {str(bundle.artist_keys[i]): int(used_counts.get(normalize_artist_key(str(bundle.artist_keys[i])), 0)) for i in order_array}
    seed_sim_values = np.array(
        [seed_sim_lookup.get(i, float(emb_norm[i] @ emb_norm[seed_idx])) for i in order_array],
        dtype=float,
    )
    constraint_metrics = _check_constraints(order_array, bundle.artist_keys, max_per_artist, min_gap)
    gap_values_np = np.array(gap_values, dtype=float) if gap_values else np.maximum(0.0, transition_floor - transitions) if transitions.size else np.array([], dtype=float)
    if provided_order or not gap_values:
        gap_sum = float(gap_values_np.sum()) if gap_values_np.size else 0.0
        below_floor_count = int((transitions < transition_floor).sum()) if transitions.size else 0
    # Edge scores for logging
    edge_scores: list[dict[str, float | str]] = []
    if order_array.size >= 2:
        # Precompute norms for sonic / genre
        X_sonic_norm = None
        X_genre_norm = None
        sonic_variant = resolve_sonic_variant(explicit_variant=sonic_variant)
        if bundle.X_sonic is not None:
            X_sonic_norm, sonic_stats = compute_sonic_variant_norm(bundle.X_sonic, sonic_variant)
            if sonic_variant != "raw":
                logger.info(
                    "SONIC_SIM_VARIANT=%s applied for S logging (dim=%d mean_norm=%.6f)",
                    sonic_variant,
                    sonic_stats.get("dim"),
                    sonic_stats.get("mean_norm"),
                )
        if bundle.X_genre_smoothed is not None:
            denom_g = np.linalg.norm(bundle.X_genre_smoothed, axis=1, keepdims=True) + 1e-12
            X_genre_norm = bundle.X_genre_smoothed / denom_g
        transition_vals = transitions.tolist()
        for idx in range(1, len(order_array)):
            prev_idx = order_array[idx - 1]
            cur_idx = order_array[idx]
            t_val = transition_vals[idx - 1] if idx - 1 < len(transition_vals) else float("nan")
            t_center_cos = float("nan")
            t_raw_uncentered = float("nan")
            h_val = float(emb_norm[cur_idx] @ emb_norm[prev_idx])
            if X_end_orig is not None and X_start_orig is not None:
                t_raw_uncentered = float(
                    transition_similarity_end_to_start(
                        X_end_orig, X_start_orig, prev_idx, np.array([cur_idx])
                    )[0]
                )
            if X_end is not None and X_start is not None:
                t_center_cos = float(
                    transition_similarity_end_to_start(
                        X_end, X_start, prev_idx, np.array([cur_idx])
                    )[0]
                )
                if rescale_transitions:
                    t_val = float(np.clip((t_center_cos + 1.0) / 2.0, 0.0, 1.0))
            s_val = float(X_sonic_norm[prev_idx] @ X_sonic_norm[cur_idx]) if X_sonic_norm is not None else float("nan")
            g_val = float(X_genre_norm[prev_idx] @ X_genre_norm[cur_idx]) if X_genre_norm is not None else float("nan")
            edge_scores.append(
                {
                    "prev_id": str(bundle.track_ids[prev_idx]),
                    "cur_id": str(bundle.track_ids[cur_idx]),
                    "T": t_val,
                    "T_used": t_val,
                    "T_centered_cos": t_center_cos,
                    "T_raw_uncentered": t_raw_uncentered,
                    "H": h_val,
                    "S": s_val,
                    "G": g_val,
                }
            )

    params_requested = {
        "mode": cfg.mode,
        "playlist_length": playlist_len,
    }
    params_effective = {
        "max_per_artist_final": max_per_artist,
        "hard_floor": hard_floor,
        "transition_floor": transition_floor,
        "transition_gamma": transition_gamma,
        "center_transitions": bool(rescale_transitions),
        "alpha_schedule": cfg.construct.alpha_schedule,
        "alpha_start": cfg.construct.alpha_start,
        "alpha_mid": cfg.construct.alpha_mid,
        "alpha_end": cfg.construct.alpha_end,
        "arc_midpoint": cfg.construct.arc_midpoint,
        "rng_seed": random_seed,
    }

    stats: Dict[str, Any] = {
        "playlist_length": len(order_array),
        "distinct_artists": len(set(str(bundle.artist_keys[i]) for i in order_array)),
        "max_artist_share": max(used_counts.values()) / len(order_array),
        "artist_counts": used_counts,
        "adjacency_violations": constraint_metrics["adjacency"],
        "min_gap_violations": constraint_metrics["gap"],
        "cap_violations": constraint_metrics["cap"],
        "below_floor_count": below_floor_count,
        "gap_sum": float(gap_sum),
        "gap_mean": float(gap_values_np.mean()) if gap_values_np.size else 0.0,
        "gap_p90": float(np.percentile(gap_values_np, 90)) if gap_values_np.size else 0.0,
        "min_transition": float(np.nanmin(transitions)) if transitions.size else float("nan"),
        "mean_transition": float(np.nanmean(transitions)) if transitions.size else float("nan"),
        "p10_transition": float(np.nanpercentile(transitions, 10)) if transitions.size else float("nan"),
        "p90_transition": float(np.nanpercentile(transitions, 90)) if transitions.size else float("nan"),
        "transition_floor": transition_floor,
        "transition_gamma": transition_gamma,
        "transition_centered": bool(rescale_transitions),
        "seed_sim_min": float(np.nanmin(seed_sim_values)) if seed_sim_values.size else float("nan"),
        "seed_sim_mean": float(np.nanmean(seed_sim_values)) if seed_sim_values.size else float("nan"),
        "hard_floor_relaxed": hard_floor_relaxed,
        "fallback_steps": fallback_steps,
        # Phase C diagnostics (transition scoring validation)
        "total_candidates_evaluated": total_candidates_evaluated,
        "rejected_by_floor": rejected_by_floor,
        "penalty_applied_count": penalty_applied_count,
        "edge_scores": edge_scores,
        "transition_weights": transition_weight_stats.get("start", {}).get("transition_weights"),
    }

    # Optional repair stage (lightweight adaptation of experiment repair)
    def _objective_values(order: np.ndarray) -> Dict[str, float]:
        trans = _transition_array(order, emb_norm, X_end, X_start, transition_gamma, rescale_transitions)
        gaps = np.maximum(0.0, transition_floor - trans) if trans.size else np.array([])
        return {
            "gap_sum": float(gaps.sum()) if gaps.size else 0.0,
            "below_floor": float((trans < transition_floor).sum()) if trans.size else 0.0,
            "min_trans": float(np.nanmin(trans)) if trans.size else float("nan"),
        }

    def _objective_tuple(vals: Dict[str, float]) -> tuple:
        if cfg.repair.objective == "gap_penalty":
            return (vals["gap_sum"], vals["below_floor"], -vals["min_trans"])
        return (vals["below_floor"], vals["gap_sum"], -vals["min_trans"])

    if cfg.repair.enabled and len(order_array) > 2:
        current_order = order_array
        current_vals = _objective_values(current_order)
        current_tuple = _objective_tuple(current_vals)
        repair_iters = 0
        unused_pool = [idx for idx in pool.pool_indices if idx not in current_order and idx not in locked_ids]

        # Build set of normalized titles in current order to prevent duplicate titles
        def _get_title_set(order: np.ndarray) -> set:
            titles = set()
            if bundle.track_titles is not None:
                for idx in order:
                    title = bundle.track_titles[idx] or ""
                    titles.add(normalize_title_for_dedupe(str(title), mode="loose"))
            return titles
        current_titles = _get_title_set(current_order)

        while repair_iters < cfg.repair.max_iters:
            if not unused_pool:
                break
            trans = _transition_array(current_order, emb_norm, X_end, X_start, transition_gamma, rescale_transitions)
            if trans.size == 0:
                break
            worst_edges = np.argsort(trans)[: cfg.repair.max_edges]
            best_candidate_order = None
            best_candidate_tuple = current_tuple
            best_candidate_vals = current_vals
            best_candidate_idx = None

            for edge in worst_edges:
                pos_next = edge + 1
                pos_prev = edge
                for cand in unused_pool:
                    # Check for title duplication before considering this candidate
                    if bundle.track_titles is not None:
                        cand_title = bundle.track_titles[cand] or ""
                        cand_norm = normalize_title_for_dedupe(str(cand_title), mode="loose")
                        if cand_norm in current_titles:
                            continue  # Skip - would introduce duplicate title

                    # substitute_next
                    if cfg.repair.allow_substitute_next and current_order[pos_next] not in locked_ids:
                        order_candidate = current_order.copy()
                        order_candidate[pos_next] = cand
                        check = _check_constraints(order_candidate, bundle.artist_keys, max_per_artist, min_gap)
                        if not any(v > 0 for v in check.values()):
                            vals = _objective_values(order_candidate)
                            tup = _objective_tuple(vals)
                            if tup < best_candidate_tuple:
                                best_candidate_order = order_candidate
                                best_candidate_tuple = tup
                                best_candidate_vals = vals
                                best_candidate_idx = cand
                    # substitute_prev
                    if cfg.repair.allow_substitute_prev and current_order[pos_prev] not in locked_ids:
                        order_candidate = current_order.copy()
                        order_candidate[pos_prev] = cand
                        check = _check_constraints(order_candidate, bundle.artist_keys, max_per_artist, min_gap)
                        if not any(v > 0 for v in check.values()):
                            vals = _objective_values(order_candidate)
                            tup = _objective_tuple(vals)
                            if tup < best_candidate_tuple:
                                best_candidate_order = order_candidate
                                best_candidate_tuple = tup
                                best_candidate_vals = vals
                                best_candidate_idx = cand

            if best_candidate_order is None:
                break
            current_order = best_candidate_order
            current_tuple = best_candidate_tuple
            current_vals = best_candidate_vals
            # Update title set with the new order
            current_titles = _get_title_set(current_order)
            if best_candidate_idx is not None and best_candidate_idx in unused_pool:
                unused_pool.remove(best_candidate_idx)
            repair_iters += 1

        if current_order is not order_array:
            order_array = current_order
            transitions = _transition_array(order_array, emb_norm, X_end, X_start, transition_gamma, rescale_transitions)
            stats.update(
                {
                    "repair_applied": True,
                    "repair_iters": repair_iters,
                    "repair_gap_penalty": current_vals["gap_sum"],
                    "repair_below_floor": current_vals["below_floor"],
                    "repair_objective": cfg.repair.objective,
                }
            )
        else:
            stats["repair_applied"] = False
    else:
        stats["repair_applied"] = False

    return PlaylistResult(
        track_indices=order_array,
        stats=stats,
        params_requested=params_requested,
        params_effective=params_effective,
    )
