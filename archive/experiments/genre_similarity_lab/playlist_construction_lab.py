"""
Playlist construction lab (artist-aware ordering with mode presets).

Consumes:
- Step 1 artifact (data_matrices_step1.npz) for embeddings/metadata.
- Candidate pool npz produced by playlist_candidate_pool_lab.py.

Builds a hybrid embedding, applies artist constraints (caps, min-gap, no
adjacency), and uses a simple beam search to order a playlist balancing
seed cohesion and local transitions.
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def format_float(x: float) -> str:
    """Format float with two decimals."""
    return f"{x:.2f}"


def safe_path(path: Path) -> Path:
    """Append _vN to path stem until it does not exist."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    version = 2
    while True:
        candidate = parent / f"{stem}_v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def default_out_csv(
    args,
    seed_track_id: Optional[str],
    seed_idx: int,
    mode_cfg: Dict[str, float],
    transition_mode: str,
    transition_floor: float,
) -> Path:
    """Build a default CSV path encoding key parameters."""
    artifacts_dir = Path("experiments/genre_similarity_lab/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    seed_identifier = seed_track_id if seed_track_id else f"seedidx{seed_idx}"
    seed_prefix = seed_identifier[:10]
    parts = [
        "playlist",
        args.mode,
        seed_prefix,
        transition_mode,
    ]
    if transition_mode == "segment_sonic":
        parts.append(f"g{format_float(args.transition_gamma)}")
    floor_mode = args.transition_floor_mode
    parts.append(f"floor-{floor_mode}")
    if floor_mode != "off":
        parts.append(f"f{format_float(transition_floor)}")
    parts.append(f"b{mode_cfg['beam']}")
    parts.append(f"L{args.playlist_len}")
    fname = "_".join(parts) + ".csv"
    path = artifacts_dir / fname
    safe = safe_path(path)
    if safe != path:
        logger.info("Auto out-csv exists, using %s", safe)
    return safe

DEFAULT_ARTIFACT = "experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"
DEFAULT_POOL = "experiments/genre_similarity_lab/artifacts/candidate_pool.npz"

MODE = {
    "narrow": {
        "max_artist_frac": 0.20,
        "min_gap": 3,
        "alpha": 0.65,
        "beta": 0.35,
        "div_bonus": 0.02,
        "repeat_bonus": 0.03,
        "avg_tracks_per_artist_target": 1.6,
        "new_artist_penalty": 0.01,
        "repeat_ramp_bonus": 0.03,
        "transition_floor": 0.50,
        "transition_floor_mode_default": "soft",
        "transition_penalty_lambda_default": 0.8,
        "beam": 1,
        "beam_segment_override": 3,
        "alpha_schedule_default": "constant",
        "alpha_start": 0.65,
        "alpha_mid": 0.65,
        "alpha_end": 0.65,
    },
    "dynamic": {
        "max_artist_frac": 0.125,
        "min_gap": 6,
        "alpha": 0.55,
        "beta": 0.45,
        "div_bonus": 0.04,
        "repeat_bonus": 0.02,
        "avg_tracks_per_artist_target": 1.25,
        "new_artist_penalty": 0.00,
        "repeat_ramp_bonus": 0.01,
        "transition_floor": 0.55,
        "transition_floor_mode_default": "hard",
        "transition_penalty_lambda_default": 1.2,
        "beam": 3,
        "beam_segment_override": 5,
        "alpha_schedule_default": "arc",
        "alpha_start": 0.65,
        "alpha_mid": 0.45,
        "alpha_end": 0.60,
    },
    "discover": {
        "max_artist_frac": 0.05,
        "min_gap": 9,
        "alpha": 0.40,
        "beta": 0.60,
        "div_bonus": 0.10,
        "repeat_bonus": 0.00,
        "avg_tracks_per_artist_target": 1.08,
        "new_artist_penalty": 0.00,
        "repeat_ramp_bonus": 0.00,
        "transition_floor": 0.55,
        "beam": 5,
        "beam_segment_override": 5,
        "transition_floor_mode_default": "soft",
        "transition_penalty_lambda_default": 0.8,
        "alpha_schedule_default": "arc",
        "alpha_start": 0.55,
        "alpha_mid": 0.30,
        "alpha_end": 0.45,
    },
}

logger = logging.getLogger(__name__)


def _fit_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    embeddings = pca.fit_transform(X_scaled)
    return embeddings, scaler, pca


def _build_hybrid(
    X_sonic: np.ndarray,
    X_genre: np.ndarray,
    n_components_sonic: int,
    n_components_genre: int,
    w_sonic: float,
    w_genre: float,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    n_comp_sonic = min(n_components_sonic, X_sonic.shape[1], X_sonic.shape[0])
    n_comp_genre = min(n_components_genre, X_genre.shape[1], X_genre.shape[0])
    if n_comp_sonic < n_components_sonic or n_comp_genre < n_components_genre:
        logger.info(
            "Adjusted n_components: sonic=%d, genre=%d (requested sonic=%d, genre=%d)",
            n_comp_sonic,
            n_comp_genre,
            n_components_sonic,
            n_components_genre,
        )
    E_sonic, sonic_scaler, sonic_pca = _fit_pca(X_sonic, n_comp_sonic)
    E_genre, _, _ = _fit_pca(X_genre, n_comp_genre)
    hybrid = np.concatenate([w_sonic * E_sonic, w_genre * E_genre], axis=1)
    # L2 normalize for cosine via dot
    norms = np.linalg.norm(hybrid, axis=1, keepdims=True) + 1e-12
    hybrid_norm = hybrid / norms
    return hybrid, hybrid_norm, sonic_scaler, sonic_pca


def _safe_str(arr: np.ndarray, idx: int) -> str:
    val = arr[idx]
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return val.decode("cp1252", errors="ignore")
    return str(val)


def _normalize_artist_key(raw: Optional[str]) -> str:
    txt = (str(raw) if raw is not None else "").strip().lower()
    if txt in {"", "unknown", "nan", "none"}:
        return ""
    return txt


@dataclass
class BeamState:
    seq: List[int]
    used_counts: Counter
    recent_artists: deque
    score: float


def _load_pool(pool_path: str, track_id_to_idx: Dict[str, int], artist_keys_eff: np.ndarray) -> Dict[str, np.ndarray]:
    """Load pool npz and align indices to artifact."""
    try:
        pool = np.load(pool_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Pool npz not found: {pool_path}", file=sys.stderr)
        sys.exit(1)

    pool_indices = pool["pool_indices"] if "pool_indices" in pool else None
    pool_track_ids = pool["pool_track_ids"] if "pool_track_ids" in pool else None
    if pool_indices is None and pool_track_ids is None:
        print("Pool npz missing pool_indices/pool_track_ids", file=sys.stderr)
        sys.exit(1)
    if pool_indices is None and pool_track_ids is not None:
        mapped = []
        for tid in pool_track_ids:
            tid_str = str(tid)
            if tid_str in track_id_to_idx:
                mapped.append(track_id_to_idx[tid_str])
        pool_indices = np.array(mapped, dtype=int)
    pool_indices = pool_indices.astype(int)
    pool_seed_sim = pool["pool_seed_sim"] if "pool_seed_sim" in pool else None
    if pool_seed_sim is not None:
        pool_seed_sim = np.asarray(pool_seed_sim, dtype=float)
        if pool_seed_sim.shape[0] != pool_indices.shape[0]:
            pool_seed_sim = None
    pool_artist_keys = pool["pool_artist_keys"] if "pool_artist_keys" in pool else artist_keys_eff[pool_indices]
    return {
        "indices": pool_indices,
        "seed_sim": pool_seed_sim,
        "artist_keys": pool_artist_keys,
    }


def _candidate_seed_sim(seed_idx: int, hybrid_norm: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return hybrid_norm[candidates] @ hybrid_norm[seed_idx]


def _feasible_candidates(
    pool_indices: np.ndarray,
    state: BeamState,
    artist_keys: Sequence[str],
    max_per_artist: int,
    min_gap: int,
    relax_level: int,
) -> List[int]:
    """
    relax_level:
      0: strict (no adjacency, honor min_gap, honor max_per_artist)
      1: allow min_gap violation but forbid adjacency, honor max_per_artist
      2: allow min_gap + adjacency, honor max_per_artist
      3: allow min_gap + adjacency, allow max_per_artist+1
    """
    used = set(state.seq)
    last_artist = artist_keys[state.seq[-1]]
    recent = set(list(state.recent_artists)[-min_gap:]) if min_gap > 0 else set()
    results = []
    for idx in pool_indices:
        if idx in used:
            continue
        artist = artist_keys[idx]
        if relax_level < 2 and artist == last_artist:
            continue  # no adjacency
        if relax_level < 1 and artist in recent:
            continue  # min-gap enforced
        count = state.used_counts.get(artist, 0)
        cap = max_per_artist + (1 if relax_level >= 3 else 0)
        if count >= cap:
            continue
        results.append(idx)
    return results


def _transition_score(
    idx: int,
    prev_idx: int,
    seed_sims: Dict[int, float],
    hybrid_norm: np.ndarray,
    artist_keys: Sequence[str],
    used_counts: Counter,
    alpha: float,
    beta: float,
    repeat_bonus: float,
    recent_window: Sequence[str],
    E_start: Optional[np.ndarray],
    E_end: Optional[np.ndarray],
    transition_gamma: float,
) -> Tuple[float, float, float, float, float, float, float]:
    seed_sim = seed_sims.get(idx, float(hybrid_norm[prev_idx] @ hybrid_norm[idx]))  # fallback shouldn’t hit
    local_hyb = float(hybrid_norm[prev_idx] @ hybrid_norm[idx])
    local_seg = float(E_end[prev_idx] @ E_start[idx]) if E_start is not None and E_end is not None else float("nan")
    if E_start is not None and E_end is not None:
        local_sim = transition_gamma * local_seg + (1 - transition_gamma) * local_hyb
    else:
        local_sim = local_hyb
    artist = artist_keys[idx]
    novelty = 1.0 if artist not in used_counts else 0.0
    repeat_ok = 1.0 if (artist in used_counts and artist not in recent_window) else 0.0
    score = alpha * seed_sim + beta * local_sim + repeat_bonus * repeat_ok
    return score, seed_sim, local_sim, novelty, repeat_ok, local_seg, local_hyb


def construct_playlist(
    seed_idx: int,
    pool_indices: np.ndarray,
    seed_sims_pool: np.ndarray,
    artist_keys: np.ndarray,
    hybrid_norm: np.ndarray,
    E_start: Optional[np.ndarray],
    E_end: Optional[np.ndarray],
    transition_gamma: float,
    transition_mode: str,
    transition_floor: Optional[float],
    transition_floor_mode: Optional[str],
    transition_penalty_lambda: Optional[float],
    playlist_len: int,
    mode_cfg: Dict[str, float],
    repeat_bonus: float,
    artist_target: int,
    new_artist_penalty: float,
    novelty_decay: str,
    repeat_ramp_bonus: float,
    beam_source: str,
    alpha_schedule: str,
    alpha_start: float,
    alpha_mid: float,
    alpha_end: float,
    arc_midpoint: float,
) -> Tuple[
    List[int],
    Dict[str, int],
    List[Dict[str, int]],
    List[float],
    List[float],
    List[float],
    List[float],
    int,
    int,
    str,
    float,
    float,
    str,
    int,
    List[float],
    str,
    float,
    float,
    float,
    float,
]:
    max_per_artist = max(1, math.ceil(playlist_len * mode_cfg["max_artist_frac"]))
    min_gap = mode_cfg["min_gap"]
    beam_size = mode_cfg["beam"]
    alpha, beta, div_bonus = mode_cfg["alpha"], mode_cfg["beta"], mode_cfg["div_bonus"]
    if transition_floor is None:
        transition_floor = mode_cfg.get("transition_floor", 0.0)
    if transition_floor_mode is None:
        transition_floor_mode = mode_cfg.get("transition_floor_mode_default", "soft")
    if transition_penalty_lambda is None:
        transition_penalty_lambda = mode_cfg.get("transition_penalty_lambda_default", 0.6)

    seed_sims_dict = {idx: float(sim) for idx, sim in zip(pool_indices, seed_sims_pool)}
    seed_artist = artist_keys[seed_idx]
    init_state = BeamState(
        seq=[seed_idx],
        used_counts=Counter({seed_artist: 1}),
        recent_artists=deque([seed_artist], maxlen=min_gap if min_gap > 0 else None),
        score=0.0,
    )
    beam = [init_state]
    step_stats: List[Dict[str, int]] = []
    chosen_below_floor = 0
    repeat_ramp_applied = 0
    alpha_t_list: List[float] = []

    # Resolve alpha schedule defaults
    if alpha_schedule is None:
        alpha_schedule = mode_cfg.get("alpha_schedule_default", "constant")
    if alpha_start is None:
        alpha_start = mode_cfg.get("alpha_start", mode_cfg.get("alpha", 0.6))
    if alpha_mid is None:
        alpha_mid = mode_cfg.get("alpha_mid", alpha_start)
    if alpha_end is None:
        alpha_end = mode_cfg.get("alpha_end", alpha_start)
    if arc_midpoint is None:
        arc_midpoint = 0.55

    def alpha_for_position(pos: int) -> float:
        # pos is next position index (0-based in seq), apply schedule for transition into this position
        if alpha_schedule == "constant":
            return alpha_start
        # arc schedule
        p = pos / max(1, playlist_len - 1)
        m = arc_midpoint
        if p <= m:
            if m == 0:
                return alpha_mid
            frac = p / m
            return alpha_start + frac * (alpha_mid - alpha_start)
        else:
            denom = max(1e-9, 1 - m)
            frac = (p - m) / denom
            return alpha_mid + frac * (alpha_end - alpha_mid)

    for step in range(1, playlist_len):
        expansions: List[BeamState] = []
        total_candidates_considered = 0
        fallback_used = 0
        below_floor_scored = 0
        floor_relaxed = False
        alpha_t = alpha_for_position(step)
        for state in beam:
            # Try progressively relaxed constraints
            candidates: List[int] = []
            for relax in range(4):
                candidates = _feasible_candidates(
                    pool_indices, state, artist_keys, max_per_artist, min_gap, relax
                )
                if candidates:
                    if relax > 0:
                        fallback_used += 1
                    break
            total_candidates_considered += len(candidates)
            if not candidates:
                continue
            # Score candidates
            prev_idx = state.seq[-1]
            scored: List[Tuple[float, int, float, float, float, float]] = []
            for idx in candidates:
                score, seed_sim, local_sim, _, repeat_ok, local_seg, local_hyb = _transition_score(
                    idx,
                    prev_idx,
                    seed_sims_dict,
                    hybrid_norm,
                    artist_keys,
                    state.used_counts,
                    alpha_t,
                    beta,
                    repeat_bonus,
                    state.recent_artists,
                    E_start,
                    E_end,
                    transition_gamma,
                )
                # Apply transition floor logic for segment mode
                if transition_mode == "segment_sonic" and not math.isnan(local_seg):
                    if transition_floor_mode == "hard" and local_seg < transition_floor:
                        continue
                    if transition_floor_mode == "soft" and local_seg < transition_floor:
                        below_floor_scored += 1
                        penalty = transition_penalty_lambda * max(0.0, transition_floor - local_seg)
                        score -= penalty
                # Adaptive novelty/penalty
                artist = artist_keys[idx]
                distinct_used = len(state.used_counts)
                artist_novelty = 1 if artist not in state.used_counts else 0
                if novelty_decay == "off":
                    novelty_weight = div_bonus
                else:
                    novelty_weight = div_bonus * max(0.0, 1.0 - (distinct_used / max(1, artist_target)))
                novelty_term = novelty_weight * artist_novelty
                penalty_term = 0.0
                if artist_novelty and distinct_used >= artist_target:
                    penalty_term = new_artist_penalty
                ramp = 0.0
                if repeat_ok == 1 and distinct_used >= artist_target:
                    ramp = repeat_ramp_bonus
                    repeat_ramp_applied += 1
                score = score + novelty_term - penalty_term + ramp

                scored.append((score, idx, seed_sim, local_sim, local_seg, local_hyb))
            scored.sort(key=lambda t: -t[0])
            for score, idx, seed_sim, local_sim, local_seg, local_hyb in scored[:50]:
                new_seq = state.seq + [idx]
                new_counts = state.used_counts.copy()
                artist = artist_keys[idx]
                new_counts[artist] += 1
                new_recent = deque(state.recent_artists, maxlen=state.recent_artists.maxlen)
                new_recent.append(artist)
                expansions.append(
                    BeamState(
                        seq=new_seq,
                        used_counts=new_counts,
                        recent_artists=new_recent,
                        score=state.score + score,
                    )
                )
        if not expansions and transition_mode == "segment_sonic" and transition_floor_mode == "hard":
            # Retry without the floor for this step
            floor_relaxed = True
            for state in beam:
                candidates = []
                for relax in range(4):
                    candidates = _feasible_candidates(
                        pool_indices, state, artist_keys, max_per_artist, min_gap, relax
                    )
                    if candidates:
                        if relax > 0:
                            fallback_used += 1
                        break
                total_candidates_considered += len(candidates)
                if not candidates:
                    continue
                prev_idx = state.seq[-1]
                scored = []
                for idx in candidates:
                    score, seed_sim, local_sim, _, _, local_seg, local_hyb = _transition_score(
                        idx,
                        prev_idx,
                        seed_sims_dict,
                        hybrid_norm,
                        artist_keys,
                        state.used_counts,
                        alpha_t,
                        beta,
                        repeat_bonus,
                        state.recent_artists,
                        E_start,
                        E_end,
                        transition_gamma,
                    )
                    scored.append((score, idx, seed_sim, local_sim, local_seg, local_hyb))
                scored.sort(key=lambda t: -t[0])
                for score, idx, seed_sim, local_sim, local_seg, local_hyb in scored[:50]:
                    new_seq = state.seq + [idx]
                    new_counts = state.used_counts.copy()
                    artist = artist_keys[idx]
                    new_counts[artist] += 1
                    new_recent = deque(state.recent_artists, maxlen=state.recent_artists.maxlen)
                    new_recent.append(artist)
                    expansions.append(
                        BeamState(
                            seq=new_seq,
                            used_counts=new_counts,
                            recent_artists=new_recent,
                            score=state.score + score,
                        )
                    )

        if not expansions:
            logger.warning("Beam search terminated early at step %d (no expansions).", step)
            break
        expansions.sort(key=lambda s: -s.score)
        beam = expansions[:beam_size]
        step_stats.append(
            {
                "step": step,
                "candidates": total_candidates_considered,
                "fallback_used": fallback_used,
                "below_floor_scored": below_floor_scored,
                "floor_relaxed": int(floor_relaxed),
            }
        )

    best = beam[0]

    seed_sim_list: List[float] = []
    local_sim_list: List[float] = []
    local_seg_list: List[float] = []
    local_hyb_list: List[float] = []
    for i, idx in enumerate(best.seq):
        if i == 0:
            seed_sim_list.append(1.0)
            local_sim_list.append(float("nan"))
            local_seg_list.append(float("nan"))
            local_hyb_list.append(float("nan"))
            alpha_t_list.append(alpha_for_position(0))
            continue
        prev_idx = best.seq[i - 1]
        seed_sim_list.append(seed_sims_dict.get(idx, float(hybrid_norm[seed_idx] @ hybrid_norm[idx])))
        local_hyb = float(hybrid_norm[prev_idx] @ hybrid_norm[idx])
        local_seg = float(E_end[prev_idx] @ E_start[idx]) if E_start is not None and E_end is not None else float("nan")
        local_used = transition_gamma * local_seg + (1 - transition_gamma) * local_hyb if not np.isnan(local_seg) else local_hyb
        if transition_mode == "segment_sonic" and not np.isnan(local_seg) and local_seg < transition_floor:
            chosen_below_floor += 1
        local_sim_list.append(local_used)
        local_seg_list.append(local_seg)
        local_hyb_list.append(local_hyb)
        alpha_t_list.append(alpha_for_position(i))

    return (
        best.seq,
        best.used_counts,
        step_stats,
        seed_sim_list,
        local_sim_list,
        local_seg_list,
        local_hyb_list,
        chosen_below_floor,
        repeat_ramp_applied,
        transition_floor_mode,
        transition_floor,
        transition_penalty_lambda,
        transition_mode,
        mode_cfg.get("beam", 0),
        beam_source,
        alpha_t_list,
        alpha_schedule,
        alpha_start,
        alpha_mid,
        alpha_end,
        arc_midpoint,
    )


def _compute_transitions(
    order: Sequence[int],
    transition_mode: str,
    hybrid_norm: np.ndarray,
    E_start: Optional[np.ndarray],
    E_end: Optional[np.ndarray],
    transition_gamma: float,
) -> np.ndarray:
    vals = []
    for i in range(1, len(order)):
        prev_idx = order[i - 1]
        cur_idx = order[i]
        local_hyb = float(hybrid_norm[prev_idx] @ hybrid_norm[cur_idx])
        if transition_mode == "segment_sonic" and E_start is not None and E_end is not None:
            local_seg = float(E_end[prev_idx] @ E_start[cur_idx])
            local_used = transition_gamma * local_seg + (1 - transition_gamma) * local_hyb
        else:
            local_seg = float("nan")
            local_used = local_hyb
        vals.append(local_used if not math.isnan(local_used) else local_hyb)
    return np.array(vals, dtype=float)


def _check_constraints(
    order: Sequence[int],
    artist_keys: Sequence[str],
    max_per_artist: int,
    min_gap: int,
) -> Tuple[bool, Dict[str, int]]:
    counts = Counter()
    recent = deque(maxlen=min_gap if min_gap > 0 else None)
    adjacency_viol = 0
    gap_viol = 0
    cap_viol = 0
    for idx in order:
        artist = artist_keys[idx]
        if recent and artist == recent[-1]:
            adjacency_viol += 1
        if min_gap > 0 and artist in recent:
            gap_viol += 1
        counts[artist] += 1
        if counts[artist] > max_per_artist:
            cap_viol += 1
        recent.append(artist)
    ok = adjacency_viol == 0 and gap_viol == 0 and cap_viol == 0 and len(order) == len(set(order))
    return ok, {"adjacency": adjacency_viol, "gap": gap_viol, "cap": cap_viol}


def _objective_stats(
    order: Sequence[int],
    transition_mode: str,
    hybrid_norm: np.ndarray,
    E_start: Optional[np.ndarray],
    E_end: Optional[np.ndarray],
    transition_gamma: float,
    floor_for_repair: float,
    seed_sims_vec: np.ndarray,
) -> Dict[str, float]:
    trans = _compute_transitions(order, transition_mode, hybrid_norm, E_start, E_end, transition_gamma)
    min_trans = float(np.nanmin(trans)) if len(trans) > 0 else float("nan")
    p10 = float(np.nanpercentile(trans, 10)) if len(trans) > 0 else float("nan")
    mean_trans = float(np.nanmean(trans)) if len(trans) > 0 else float("nan")
    if math.isnan(floor_for_repair):
        below_floor = 0
        gap_vals = np.zeros_like(trans)
    else:
        below_floor = int(np.sum(trans < floor_for_repair))
        gap_vals = np.maximum(0.0, floor_for_repair - trans)
    seed_vals = seed_sims_vec[list(order)][1:] if len(order) > 1 else []
    mean_seed = float(np.nanmean(seed_vals)) if len(seed_vals) > 0 else float("nan")
    return {
        "min_trans": min_trans,
        "p10_trans": p10,
        "mean_trans": mean_trans,
        "below_floor": below_floor,
        "mean_seed": mean_seed,
        "gap_sum": float(np.sum(gap_vals)) if len(gap_vals) > 0 else 0.0,
        "gap_mean": float(np.mean(gap_vals)) if len(gap_vals) > 0 else 0.0,
        "gap_p90": float(np.nanpercentile(gap_vals, 90)) if len(gap_vals) > 0 else 0.0,
        "trans_array": trans,
    }


def _objective_tuple(stats: Dict[str, float], objective: str) -> Tuple[float, float, float, float]:
    """
    Lower tuple is better.
    For reduce_below_floor we minimize below_floor, then maximize min/mean transition, then seed mean.
    For lexicographic we maximize min_trans, then minimize below_floor, then maximize mean_trans, then seed mean.
    """
    if objective == "reduce_below_floor":
        return (
            stats["below_floor"],
            -stats["min_trans"],
            -stats["mean_trans"],
            -stats["mean_seed"],
        )
    if objective == "gap_penalty":
        return (
            stats["gap_sum"],
            stats["below_floor"],
            -stats["min_trans"],
            -stats["mean_trans"],
            -stats["mean_seed"],
        )
    # default lexicographic (maximize min, minimize below, maximize mean, seed)
    return (
        -stats["min_trans"],
        stats["below_floor"],
        -stats["mean_trans"],
        -stats["mean_seed"],
    )


def main():
    parser = argparse.ArgumentParser(description="Playlist construction lab (artist-aware ordering).")
    parser.add_argument("--artifact-path", default=DEFAULT_ARTIFACT, help="Path to Step 1 artifact npz.")
    parser.add_argument("--pool-npz", default=DEFAULT_POOL, help="Path to candidate pool npz.")
    parser.add_argument("--seed-track-id", required=True, help="Seed track_id.")
    parser.add_argument("--mode", choices=list(MODE.keys()), default="dynamic", help="Mode presets.")
    parser.add_argument("--playlist-len", type=int, default=30, help="Target playlist length.")
    parser.add_argument("--w-sonic", type=float, default=0.6, help="Hybrid weight for sonic.")
    parser.add_argument("--w-genre", type=float, default=0.4, help="Hybrid weight for genre.")
    parser.add_argument("--n-components-sonic", type=int, default=32, help="PCA components for sonic.")
    parser.add_argument("--n-components-genre", type=int, default=32, help="PCA components for genre.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state (currently unused).")
    parser.add_argument("--out-csv", help="Optional path to save ordered playlist as CSV.")
    parser.add_argument("--out-npz", help="Optional path to save ordered playlist as NPZ.")
    parser.add_argument("--div-bonus", type=float, help="Override div_bonus; default from mode preset.")
    parser.add_argument("--repeat-bonus", type=float, help="Override repeat_bonus; default from mode preset.")
    parser.add_argument(
        "--transition-mode",
        choices=["hybrid", "segment_sonic"],
        default="hybrid",
        help="Use hybrid cosine or segment-based sonic transitions.",
    )
    parser.add_argument("--sonic-start-key", default="X_sonic_start", help="NPZ key for start-segment sonic features.")
    parser.add_argument("--sonic-end-key", default="X_sonic_end", help="NPZ key for end-segment sonic features.")
    parser.add_argument(
        "--transition-gamma",
        type=float,
        default=1.0,
        help="Blend weight for segment_sonic vs hybrid in local_sim (1.0=segment only).",
    )
    parser.add_argument(
        "--transition-floor",
        type=float,
        help="Floor for segment local_sim (default from mode).",
    )
    parser.add_argument(
        "--transition-floor-mode",
        choices=["soft", "hard", "off"],
        default=None,
        help="How to apply transition floor in segment mode.",
    )
    parser.add_argument(
        "--transition-penalty-lambda",
        type=float,
        default=None,
        help="Penalty weight for soft floor when segment_sonic transitions are below floor.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        help="Override beam width (default from mode).",
    )
    parser.add_argument(
        "--beam-segment-override",
        type=int,
        default=3,
        help="Beam width to use automatically when transition_mode=segment_sonic (if --beam not provided).",
    )
    parser.add_argument(
        "--artist-target",
        type=int,
        help="Target distinct artists; default computed from mode avg_tracks_per_artist_target.",
    )
    parser.add_argument(
        "--new-artist-penalty",
        type=float,
        help="Penalty applied when adding new artists after target is reached (default from mode).",
    )
    parser.add_argument(
        "--novelty-decay",
        choices=["linear", "off"],
        default="linear",
        help="How the novelty bonus decays as distinct artists approach the target.",
    )
    parser.add_argument(
        "--alpha-schedule",
        choices=["constant", "arc"],
        help="Seed cohesion weight schedule; default from mode preset.",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        help="Alpha at start of playlist (default from mode preset).",
    )
    parser.add_argument(
        "--alpha-mid",
        type=float,
        help="Alpha at midpoint (for arc schedule; default from mode preset).",
    )
    parser.add_argument(
        "--alpha-end",
        type=float,
        help="Alpha at end of playlist (default from mode preset).",
    )
    parser.add_argument(
        "--arc-midpoint",
        type=float,
        help="Fraction of playlist where alpha_mid applies (default 0.55 if not overridden).",
    )
    parser.add_argument(
        "--repair-pass",
        choices=["off", "auto", "on"],
        default="auto",
        help="Post-pass repair to improve transitions without breaking constraints.",
    )
    parser.add_argument(
        "--repair-max-iters",
        type=int,
        default=20,
        help="Maximum repair iterations.",
    )
    parser.add_argument(
        "--repair-window",
        type=int,
        default=6,
        help="Repair considers swaps/relocates within +/- this window around worst edge.",
    )
    parser.add_argument(
        "--repair-ops",
        default="swap,relocate",
        help="Comma-separated ops to try (swap,relocate).",
    )
    parser.add_argument(
        "--repair-objective",
        choices=["maximin", "lexicographic", "reduce_below_floor", "gap_penalty"],
        default="lexicographic",
        help="Repair objective ranking.",
    )
    parser.add_argument(
        "--repair-log-top",
        type=int,
        default=5,
        help="Log top N candidate repairs per iteration.",
    )
    parser.add_argument(
        "--repair-target-floor",
        type=float,
        help="Override floor used by repair objective (default: construction floor).",
    )
    parser.add_argument(
        "--repair-max-edges",
        type=int,
        default=5,
        help="Number of worst edges to target per iteration (when below floor).",
    )
    parser.add_argument(
        "--repair-trigger",
        choices=["auto", "below_floor", "always"],
        default="auto",
        help="When to invoke repair.",
    )
    parser.add_argument(
        "--repair-substitute-topk",
        type=int,
        default=80,
        help="Top-K candidates to consider for substitute_next repair op.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for deterministic tie-breaking.",
    )
    parser.add_argument(
        "--repair-assert-monotonic",
        action="store_true",
        help="Assert that below_floor never worsens when using reduce_below_floor objective.",
    )
    parser.add_argument(
        "--list-sonic-keys",
        action="store_true",
        help="List artifact keys containing 'X_sonic' and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    try:
        art = np.load(args.artifact_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Artifact not found: {args.artifact_path}", file=sys.stderr)
        sys.exit(1)

    if args.list_sonic_keys:
        keys = [k for k in art.files if "X_sonic" in k]
        print("Sonic-related keys in artifact:")
        for k in keys:
            print(f"  {k}")
        return

    X_sonic = art["X_sonic"]
    X_genre = art["X_genre_smoothed"] if "X_genre_smoothed" in art else art["X_genre"]
    track_ids = art["track_ids"]
    artist_names = art["artist_names"] if "artist_names" in art else art["track_artists"]
    track_titles = art["track_titles"]
    if "artist_keys_effective" in art:
        artist_keys = art["artist_keys_effective"]
    elif "artist_keys" in art:
        artist_keys = art["artist_keys"]
    else:
        artist_keys = np.array([""] * len(track_ids))

    # Normalize artist keys quickly for safety
    artist_keys = np.array([_normalize_artist_key(a) or f"unknown:{_safe_str(track_ids, i)}" for i, a in enumerate(artist_keys)], dtype=object)

    # Map track_id -> idx
    track_id_to_idx = {str(tid): i for i, tid in enumerate(track_ids)}
    if args.seed_track_id not in track_id_to_idx:
        print(f"Seed track_id not found in artifact: {args.seed_track_id}", file=sys.stderr)
        sys.exit(1)
    seed_idx = track_id_to_idx[args.seed_track_id]

    hybrid, hybrid_norm, sonic_scaler, sonic_pca = _build_hybrid(
        X_sonic,
        X_genre,
        args.n_components_sonic,
        args.n_components_genre,
        args.w_sonic,
        args.w_genre,
    )

    # Segment embeddings (optional)
    E_start = None
    E_end = None
    transition_mode = args.transition_mode
    if args.transition_mode == "segment_sonic":
        try:
            X_start = art[args.sonic_start_key]
            X_end = art[args.sonic_end_key]
            if X_start.shape[0] != X_sonic.shape[0] or X_end.shape[0] != X_sonic.shape[0]:
                raise ValueError("Segment feature row counts do not match X_sonic")
            X_start_scaled = sonic_scaler.transform(X_start)
            X_end_scaled = sonic_scaler.transform(X_end)
            E_start_raw = sonic_pca.transform(X_start_scaled)
            E_end_raw = sonic_pca.transform(X_end_scaled)
            E_start = E_start_raw / (np.linalg.norm(E_start_raw, axis=1, keepdims=True) + 1e-12)
            E_end = E_end_raw / (np.linalg.norm(E_end_raw, axis=1, keepdims=True) + 1e-12)
            logger.info(
                "Loaded segment sonic features start=%s end=%s; using segment_sonic transitions",
                args.sonic_start_key,
                args.sonic_end_key,
            )
        except KeyError:
            logger.warning(
                "Segment sonic keys not found (%s, %s); falling back to hybrid transitions",
                args.sonic_start_key,
                args.sonic_end_key,
            )
            transition_mode = "hybrid"
        except Exception as exc:
            logger.warning(
                "Segment sonic loading failed (%s); falling back to hybrid transitions", exc
            )
            transition_mode = "hybrid"

    pool_data = _load_pool(args.pool_npz, track_id_to_idx, artist_keys)
    pool_indices = pool_data["indices"]
    # Seed sims
    if pool_data["seed_sim"] is not None:
        seed_sims_pool = pool_data["seed_sim"]
    else:
        seed_sims_pool = _candidate_seed_sim(seed_idx, hybrid_norm, pool_indices)

    mode_cfg = MODE[args.mode].copy()
    beam_source = "mode_default"
    if args.div_bonus is not None:
        mode_cfg["div_bonus"] = args.div_bonus
    if args.repeat_bonus is not None:
        mode_cfg["repeat_bonus"] = args.repeat_bonus
    if args.beam is not None:
        mode_cfg["beam"] = args.beam
        beam_auto_override = False
        beam_source = "cli"
    elif args.transition_mode == "segment_sonic":
        seg_override = (
            args.beam_segment_override
            if args.beam_segment_override is not None
            else mode_cfg.get("beam_segment_override", mode_cfg["beam"])
        )
        mode_cfg["beam"] = seg_override
        beam_auto_override = True
        beam_source = "segment_override"
    else:
        beam_auto_override = False
        beam_source = "mode_default"
    mode_cfg["max_per_artist"] = max(1, math.ceil(args.playlist_len * mode_cfg["max_artist_frac"]))
    transition_floor = args.transition_floor if args.transition_floor is not None else MODE[args.mode]["transition_floor"]
    alpha_schedule_effective = args.alpha_schedule or mode_cfg.get("alpha_schedule_default", "constant")
    alpha_start_effective = (
        args.alpha_start if args.alpha_start is not None else mode_cfg.get("alpha_start", mode_cfg["alpha"])
    )
    alpha_mid_effective = args.alpha_mid if args.alpha_mid is not None else mode_cfg.get("alpha_mid", alpha_start_effective)
    alpha_end_effective = args.alpha_end if args.alpha_end is not None else mode_cfg.get("alpha_end", alpha_start_effective)
    arc_midpoint_effective = args.arc_midpoint if args.arc_midpoint is not None else 0.55

    print("=== Mode Config ===")
    print(
        f"mode={args.mode} | L={args.playlist_len} | max_per_artist={mode_cfg['max_per_artist']} | "
        f"min_gap={mode_cfg['min_gap']} | alpha={mode_cfg['alpha']} | beta={mode_cfg['beta']} | "
        f"div_bonus={mode_cfg['div_bonus']} | repeat_bonus={mode_cfg['repeat_bonus']} | beam={mode_cfg['beam']}"
    )
    floor_mode_effective = args.transition_floor_mode or mode_cfg.get("transition_floor_mode_default", "soft")
    penalty_lambda_effective = (
        args.transition_penalty_lambda
        if args.transition_penalty_lambda is not None
        else mode_cfg.get("transition_penalty_lambda_default", 0.6)
    )
    floor_for_repair = args.repair_target_floor if args.repair_target_floor is not None else transition_floor
    print(
        f"Transition mode: {transition_mode} (gamma={args.transition_gamma}, "
        f"floor={transition_floor}, floor_mode={floor_mode_effective}, penalty_lambda={penalty_lambda_effective}, "
        f"beam_auto_override={beam_auto_override})"
    )
    print(
        f"Alpha schedule: {alpha_schedule_effective} "
        f"(start={alpha_start_effective:.2f}, mid={alpha_mid_effective:.2f}, "
        f"end={alpha_end_effective:.2f}, arc_midpoint={arc_midpoint_effective:.2f})"
    )
    print(f"Pool size: {len(pool_indices)} candidates")
    print()
    artist_target = args.artist_target if args.artist_target is not None else min(
        args.playlist_len, math.ceil(args.playlist_len / MODE[args.mode].get("avg_tracks_per_artist_target", 1.0))
    )
    new_artist_penalty = args.new_artist_penalty if args.new_artist_penalty is not None else MODE[args.mode].get(
        "new_artist_penalty", 0.0
    )
    repeat_ramp_bonus = MODE[args.mode].get("repeat_ramp_bonus", 0.0)

    seq, used_counts, step_stats, seed_sim_list, local_sim_list, local_seg_list, local_hyb_list, chosen_below_floor, repeat_ramp_applied, floor_mode_effective, transition_floor, penalty_lambda_effective, transition_mode, mode_cfg["beam"], beam_source, alpha_t_list, alpha_schedule_effective, alpha_start_effective, alpha_mid_effective, alpha_end_effective, arc_midpoint_effective = construct_playlist(
        seed_idx,
        pool_indices,
        seed_sims_pool,
        artist_keys,
        hybrid_norm,
        E_start if transition_mode == "segment_sonic" else None,
        E_end if transition_mode == "segment_sonic" else None,
        args.transition_gamma,
        transition_mode,
        transition_floor if transition_mode == "segment_sonic" else 0.0,
        floor_mode_effective,
        penalty_lambda_effective,
        args.playlist_len,
        mode_cfg,
        repeat_bonus=mode_cfg["repeat_bonus"],
        artist_target=artist_target,
        new_artist_penalty=new_artist_penalty,
        novelty_decay=args.novelty_decay,
        repeat_ramp_bonus=repeat_ramp_bonus,
        beam_source=beam_source,
        alpha_schedule=alpha_schedule_effective,
        alpha_start=alpha_start_effective,
        alpha_mid=alpha_mid_effective,
        alpha_end=alpha_end_effective,
        arc_midpoint=arc_midpoint_effective,
    )

    sample_positions = [1, 5, 10, 15, args.playlist_len - 1]
    print("Alpha_t samples:")
    for pos in sample_positions:
        if pos < len(alpha_t_list):
            print(f"  t={pos:>2}: {alpha_t_list[pos]:.3f}")
    print()

    # Post-pass repair
    repair_applied = False
    repair_iters_used = 0
    repair_delta_min = 0.0
    repair_delta_below = 0
    repair_ops_raw = args.repair_ops
    if args.repair_objective == "reduce_below_floor" and args.repair_ops == "swap,relocate":
        repair_ops_raw = "substitute_next"
    repair_ops_allowed = {op.strip() for op in repair_ops_raw.split(",") if op.strip()}
    seed_sims_vec = hybrid_norm @ hybrid_norm[seed_idx]
    pool_set = set(pool_indices.tolist())
    repair_max_edges = args.repair_max_edges
    if args.repair_objective == "gap_penalty" and args.repair_max_edges == 5 and args.mode == "discover":
        repair_max_edges = 8

    def recompute_lists(order: List[int]) -> Tuple[List[float], List[float], List[float], List[float], int]:
        seed_list: List[float] = []
        local_used_list: List[float] = []
        local_seg_list: List[float] = []
        local_hyb_list: List[float] = []
        chosen_below = 0
        for i, idx in enumerate(order):
            if i == 0:
                seed_list.append(1.0)
                local_used_list.append(float("nan"))
                local_seg_list.append(float("nan"))
                local_hyb_list.append(float("nan"))
                continue
            prev_idx = order[i - 1]
            seed_list.append(float(seed_sims_vec[idx]))
            local_hyb = float(hybrid_norm[prev_idx] @ hybrid_norm[idx])
            if transition_mode == "segment_sonic" and E_start is not None and E_end is not None:
                local_seg = float(E_end[prev_idx] @ E_start[idx])
                local_used = args.transition_gamma * local_seg + (1 - args.transition_gamma) * local_hyb
                if floor_mode_effective != "off" and local_seg < transition_floor:
                    chosen_below += 1
            else:
                local_seg = float("nan")
                local_used = local_hyb
            local_used_list.append(local_used)
            local_seg_list.append(local_seg)
            local_hyb_list.append(local_hyb)
        return seed_list, local_used_list, local_seg_list, local_hyb_list, chosen_below

    def apply_swap(order: List[int], i: int, j: int) -> List[int]:
        new_order = order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return new_order

    def apply_relocate(order: List[int], i: int, j: int) -> List[int]:
        if i == j:
            return order
        new_order = order.copy()
        elem = new_order.pop(i)
        new_order.insert(j, elem)
        return new_order

    current_stats = _objective_stats(
        seq,
        transition_mode,
        hybrid_norm,
        E_start if transition_mode == "segment_sonic" else None,
        E_end if transition_mode == "segment_sonic" else None,
        args.transition_gamma,
        floor_for_repair if transition_mode == "segment_sonic" else float("nan"),
        seed_sims_vec,
    )
    def _should_repair() -> bool:
        if args.repair_pass == "off":
            return False
        if args.repair_pass == "on":
            return True
        # auto
        if transition_mode == "segment_sonic":
            if args.repair_trigger == "auto":
                return current_stats["min_trans"] < floor_for_repair or chosen_below_floor > 0
            if args.repair_trigger == "below_floor":
                return current_stats["min_trans"] < floor_for_repair
            if args.repair_trigger == "always":
                return True
        else:
            if args.repair_trigger == "always":
                return True
            if args.repair_trigger in {"auto", "below_floor"}:
                return current_stats["min_trans"] < (floor_for_repair if not math.isnan(floor_for_repair) else 0.55)
        return False

    if _should_repair():
        print("Repair pass enabled (initial min_trans "
              f"{current_stats['min_trans']:.3f}, below_floor={current_stats['below_floor']}, mean_trans={current_stats['mean_trans']:.3f}); "
              f"construction_floor={transition_floor} {floor_mode_effective}, repair_floor={floor_for_repair}")
        best_order = seq
        best_stats = current_stats
        L = len(seq)
        for itr in range(args.repair_max_iters):
            trans = best_stats["trans_array"]
            if len(trans) == 0:
                break
            below_idx = [i for i, v in enumerate(trans) if v < floor_for_repair]
            if below_idx:
                worst_edges = sorted(below_idx, key=lambda i: trans[i])[: repair_max_edges]
            else:
                worst_edges = [int(np.nanargmin(trans))]
            worst_val = float(trans[worst_edges[0]])
            positions_set = set()
            for we in worst_edges:
                pos_start = max(1, we - args.repair_window)
                pos_end = min(L - 1, we + args.repair_window + 1)
                positions_set.update(range(pos_start, pos_end + 1))
            positions = sorted(p for p in positions_set if p < L)
            candidates = []
            # generate ops
            if "swap" in repair_ops_allowed:
                for i in positions:
                    for j in positions:
                        if i >= L or j >= L or i == j:
                            continue
                        candidates.append(("swap", i, j))
            if "relocate" in repair_ops_allowed:
                for i in positions:
                    for j in positions:
                        if i >= L or j >= L or i == j:
                            continue
                        candidates.append(("relocate", i, j))
            if "substitute_next" in repair_ops_allowed:
                # replace next node in worst edge (t+1) with unused candidate from pool
                unused = list(pool_set - set(best_order))
                if unused:
                    for worst_idx in worst_edges:
                        tpos = worst_idx + 1
                        A = best_order[worst_idx]
                        B_pos = tpos
                        C = best_order[worst_idx + 2] if worst_idx + 2 < L else None
                        # precompute anchor vectors
                        A_end = E_end[A] if E_end is not None else None
                        C_start = E_start[C] if (C is not None and E_start is not None) else None
                        if A_end is not None:
                            sims1 = A_end @ E_start[unused].T
                            top_indices = np.argsort(sims1)[::-1][: args.repair_substitute_topk]
                            cand_list = [unused[k] for k in top_indices]
                        else:
                            cand_list = unused[: args.repair_substitute_topk]
                        if C_start is not None and E_end is not None:
                            sims2 = E_end[unused] @ C_start
                            top2 = np.argsort(sims2)[::-1][: args.repair_substitute_topk]
                            cand_list = list({*cand_list, *[unused[k] for k in top2]})
                        for x in cand_list:
                            candidates.append(("substitute_next", B_pos, x))
            if "substitute_prev" in repair_ops_allowed and E_start is not None and E_end is not None:
                unused = list(pool_set - set(best_order))
                if unused:
                    for worst_idx in worst_edges:
                        if worst_idx >= L:
                            continue
                        A_pos = worst_idx
                        A = best_order[A_pos]
                        B = best_order[worst_idx + 1] if worst_idx + 1 < L else None
                        P = best_order[worst_idx - 1] if worst_idx - 1 >= 0 else None
                        if B is None:
                            continue
                        B_start = E_start[B]
                        P_end = E_end[P] if P is not None else None
                        if B_start is None:
                            continue
                        cand_list = []
                        if P_end is not None:
                            sims_prev = P_end @ E_start[unused].T
                            top_prev = np.argsort(sims_prev)[::-1][: args.repair_substitute_topk]
                            cand_list.extend([unused[k] for k in top_prev])
                        sims_next = E_end[unused] @ B_start
                        top_next = np.argsort(sims_next)[::-1][: args.repair_substitute_topk]
                        cand_list.extend([unused[k] for k in top_next])
                        cand_list = list(dict.fromkeys(cand_list))  # dedupe, preserve order
                        for x in cand_list:
                            candidates.append(("substitute_prev", A_pos, x))
            scored_candidates = []
            for op, i, j in candidates:
                if op == "swap":
                    order2 = apply_swap(best_order, i, j)
                else:
                    if op == "relocate":
                        order2 = apply_relocate(best_order, i, j)
                    elif op == "substitute_next":
                        # i is position, j is replacement track idx
                        if i >= L:
                            continue
                        if j in best_order:
                            continue
                        order2 = best_order.copy()
                        order2[i] = j
                    else:
                        continue
                ok, _viol = _check_constraints(order2, artist_keys, mode_cfg["max_per_artist"], mode_cfg["min_gap"])
                if not ok:
                    continue
                stats2 = _objective_stats(
                    order2,
                    transition_mode,
                    hybrid_norm,
                    E_start if transition_mode == "segment_sonic" else None,
                    E_end if transition_mode == "segment_sonic" else None,
                    args.transition_gamma,
                    floor_for_repair if transition_mode == "segment_sonic" else float("nan"),
                    seed_sims_vec,
                )
                key = _objective_tuple(stats2, args.repair_objective)
                scored_candidates.append((key, op, i, j, stats2))
            if not scored_candidates:
                print(f"Repair iteration {itr}: no feasible candidates; stopping.")
                break
                scored_candidates.sort()
            best_key, best_op, best_i, best_j, cand_stats = scored_candidates[0]
            current_key = _objective_tuple(best_stats, args.repair_objective)
            if best_key < current_key:
                print(
                    f"Repair iter {itr}: targeting edges {worst_edges} (repair_floor={floor_for_repair}) "
                    f"| current min={best_stats['min_trans']:.3f} below={best_stats['below_floor']} mean={best_stats['mean_trans']:.3f}"
                )
                print(
                    f"  applying {best_op}({best_i},{best_j}) -> min {best_stats['min_trans']:.3f}->{cand_stats['min_trans']:.3f} "
                    f"below {best_stats['below_floor']}->{cand_stats['below_floor']} mean {best_stats['mean_trans']:.3f}->{cand_stats['mean_trans']:.3f} "
                    f"gap_sum {best_stats.get('gap_sum',0):.3f}->{cand_stats.get('gap_sum',0):.3f} "
                    f"seed_mean {best_stats['mean_seed']:.3f}->{cand_stats['mean_seed']:.3f}"
                )
                if args.repair_log_top > 0:
                    for rank, entry in enumerate(scored_candidates[: args.repair_log_top], start=1):
                        key, op, i, j, st = entry
                        print(
                            f"    cand #{rank}: {op}({i},{j}) min={st['min_trans']:.3f} "
                            f"below={st['below_floor']} mean={st['mean_trans']:.3f} seed_mean={st['mean_seed']:.3f}"
                        )
                if best_op == "swap":
                    best_order = apply_swap(best_order, best_i, best_j)
                elif best_op == "relocate":
                    best_order = apply_relocate(best_order, best_i, best_j)
                elif best_op == "substitute_next":
                    best_order = best_order.copy()
                    best_order[best_i] = best_j
                elif best_op == "substitute_prev":
                    best_order = best_order.copy()
                    best_order[best_i] = best_j
                if args.repair_assert_monotonic and args.repair_objective == "reduce_below_floor":
                    if cand_stats["below_floor"] > best_stats["below_floor"]:
                        raise RuntimeError(
                            f"Repair monotonicity violated: {best_stats['below_floor']}->{cand_stats['below_floor']}"
                        )
                best_stats = cand_stats
                repair_applied = True
                repair_iters_used += 1
            else:
                print(f"Repair iter {itr}: no improving candidates; stopping.")
                break
        if repair_applied:
            repair_delta_min = best_stats["min_trans"] - current_stats["min_trans"]
            repair_delta_below = best_stats["below_floor"] - current_stats["below_floor"]
            # Recompute lists for final order
            seq = best_order
            seed_sim_list, local_sim_list, local_seg_list, local_hyb_list, chosen_below_floor = recompute_lists(seq)
            print(
                f"Repair complete: iters={repair_iters_used}, min_trans {current_stats['min_trans']:.3f}->{best_stats['min_trans']:.3f}, "
                f"below_floor {current_stats['below_floor']}->{best_stats['below_floor']} (repair_floor={floor_for_repair})"
            )
        else:
            print("Repair evaluated but no changes applied.")

    print("=== Feasibility per step ===")
    for stat in step_stats:
        print(
            f"step={stat['step']:>3} | candidates={stat['candidates']:>5} | fallbacks={stat['fallback_used']:>3}"
            + (
                f" | below_floor_scored={stat.get('below_floor_scored',0):>3} | floor_relaxed={stat.get('floor_relaxed',0)}"
                if transition_mode == "segment_sonic"
                else ""
            )
        )
    print()

    # Final diagnostics
    artist_counter = Counter(artist_keys[i] for i in seq)
    adjacency_viol = 0
    gap_viol = 0
    min_gap = mode_cfg["min_gap"]
    recent = deque(maxlen=min_gap if min_gap > 0 else None)
    for idx in seq:
        artist = artist_keys[idx]
        if recent and artist == recent[-1]:
            adjacency_viol += 1
        if min_gap > 0 and artist in recent:
            gap_viol += 1
        recent.append(artist)
    max_artist_frac_actual = max(artist_counter.values()) / len(seq)
    print("=== Playlist Summary ===")
    print(f"Length: {len(seq)} (target {args.playlist_len})")
    print(f"Distinct artists: {len(artist_counter)}")
    print(f"Max artist fraction: {max_artist_frac_actual:.3f}")
    print(f"Adjacency violations: {adjacency_viol}")
    print(f"Min-gap violations: {gap_viol}")
    print(f"Mean seed_sim (excl seed=1): {np.nanmean(seed_sim_list[1:]):.3f}")
    print(f"Mean local_sim: {np.nanmean(local_sim_list[1:]):.3f}")
    print()

    top_artists = artist_counter.most_common(15)
    print("Top artists:")
    for artist, count in top_artists:
        print(f"  {artist:<25} {count}")
    print()

    # Diversity metrics
    counts = np.array(list(artist_counter.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    effective_artists = float(np.exp(entropy))
    # Gini coefficient
    sorted_counts = np.sort(counts)
    n_art = len(sorted_counts)
    gini = (
        (np.sum((2 * np.arange(1, n_art + 1) - n_art - 1) * sorted_counts))
        / (n_art * np.sum(sorted_counts) + 1e-12)
    )
    print("Diversity:")
    print(f"  Distinct artists: {len(artist_counter)}")
    print(f"  Effective artists (perplexity): {effective_artists:.2f}")
    print(f"  Gini (0=even, 1=uneven): {gini:.3f}")
    print()

    # Seed/local stats
    seed_vals = np.array(seed_sim_list[1:], dtype=float)
    local_vals = np.array(local_sim_list[1:], dtype=float)
    def _stats(arr: np.ndarray) -> str:
        return (
            f"min={np.nanmin(arr):.3f}, p10={np.nanpercentile(arr,10):.3f}, "
            f"median={np.nanmedian(arr):.3f}, mean={np.nanmean(arr):.3f}, "
            f"p90={np.nanpercentile(arr,90):.3f}, max={np.nanmax(arr):.3f}"
        )
    print("Seed similarity stats:")
    print(f"  {_stats(seed_vals)}")
    print("Local similarity stats (used):")
    print(f"  {_stats(local_vals)}")
    if transition_mode == "segment_sonic":
        local_seg_vals = np.array(local_seg_list[1:], dtype=float)
        local_hyb_vals = np.array(local_hyb_list[1:], dtype=float)
        print("Local segment stats:")
        print(f"  {_stats(local_seg_vals)}")
        print("Local hybrid stats:")
        print(f"  {_stats(local_hyb_vals)}")
    print()

    # Seed drift: running mean embedding vs seed
    drift_points = [5, 10, 15, len(seq)]
    seed_vec = hybrid_norm[seed_idx]
    print("Seed drift (cosine of running mean vs seed):")
    for k in drift_points:
        if k > len(seq):
            continue
        mean_vec = np.mean(hybrid_norm[seq[:k]], axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
        cos = float(np.dot(mean_vec, seed_vec))
        print(f"  first {k:>2} tracks: {cos:.3f}")
    print()

    if transition_mode == "segment_sonic":
        print(f"Transitions chosen below floor: {chosen_below_floor}")
        print(f"Local segment min={np.nanmin(local_seg_list[1:]):.3f}, p10={np.nanpercentile(local_seg_list[1:],10):.3f}")

    # Transitions preview
    print("First 15 transitions:")
    for i in range(min(15, len(seq) - 1)):
        cur = seq[i]
        nxt = seq[i + 1]
        extra = ""
        if transition_mode == "segment_sonic":
            extra = f" | local_seg={local_seg_list[i+1]:.3f} | local_hyb={local_hyb_list[i+1]:.3f}"
        print(
            f"{i:>2} | {_safe_str(artist_names, cur)} - {_safe_str(track_titles, cur)}"
            f" -> {_safe_str(artist_names, nxt)} - {_safe_str(track_titles, nxt)}"
            f" | seed_sim_next={seed_sim_list[i+1]:.3f} | local_sim={local_sim_list[i+1]:.3f}{extra}"
        )

    if transition_mode == "segment_sonic":
        # Worst 10 transitions by used local_sim
        pairs = []
        for i in range(1, len(seq)):
            pairs.append(
                (
                    local_sim_list[i],
                    local_seg_list[i],
                    local_hyb_list[i],
                    seed_sim_list[i],
                    seq[i - 1],
                    seq[i],
                )
            )
        pairs.sort(key=lambda t: t[0])
        print("\nWorst 10 transitions (by used local_sim):")
        for idx, (lu, ls, lh, ss, prev_idx, nxt_idx) in enumerate(pairs[:10], start=1):
            print(
                f"{idx:>2} | {_safe_str(artist_names, prev_idx)} - {_safe_str(track_titles, prev_idx)}"
                f" -> {_safe_str(artist_names, nxt_idx)} - {_safe_str(track_titles, nxt_idx)}"
                f" | local_seg={ls:.3f} | local_hyb={lh:.3f} | local_used={lu:.3f} | seed_sim_next={ss:.3f}"
            )

    # Save CSV
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "pos",
                    "track_id",
                    "artist_key",
                    "artist",
                    "title",
                    "seed_sim",
                    "local_sim_segment",
                    "local_sim_hybrid",
                    "local_sim_used",
                    "step_score",
                    "alpha_t_used",
                    "repair_applied",
                    "repair_iters_used",
                    "random_seed_used",
                ]
            )
            for pos, idx in enumerate(seq, start=1):
                artist = _safe_str(artist_names, idx)
                title = _safe_str(track_titles, idx)
                track_id = _safe_str(track_ids, idx)
                seed_sim = seed_sim_list[pos - 1]
                local_used = local_sim_list[pos - 1]
                local_seg = local_seg_list[pos - 1]
                local_hyb = local_hyb_list[pos - 1]
                alpha_t_used = alpha_t_list[pos - 1] if pos - 1 < len(alpha_t_list) else mode_cfg["alpha"]
                if pos == 1:
                    step_score = seed_sim
                else:
                    prev_artists = [artist_keys[j] for j in seq[: pos - 1]]
                    novelty = 1.0 if artist_keys[idx] not in prev_artists else 0.0
                    recent = prev_artists[-mode_cfg["min_gap"] :] if mode_cfg["min_gap"] > 0 else []
                    repeat_ok = 1.0 if (artist_keys[idx] in prev_artists and artist_keys[idx] not in recent) else 0.0
                    step_score = (
                        alpha_t_used * seed_sim
                        + mode_cfg["beta"] * local_used
                        + mode_cfg["div_bonus"] * novelty
                        + mode_cfg["repeat_bonus"] * repeat_ok
                    )
                writer.writerow(
                    [
                        pos,
                        track_id,
                        artist_keys[idx],
                        artist,
                        title,
                        seed_sim,
                        local_seg,
                        local_hyb,
                        local_used,
                        step_score,
                        alpha_t_used,
                        int(repair_applied),
                        repair_iters_used,
                        args.random_seed if args.random_seed is not None else "",
                    ]
                )
        print(f"\nSaved playlist CSV to {out_path}")

    if args.out_npz:
        out_path = Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "mode": args.mode,
            "playlist_len": args.playlist_len,
            "weights": {"w_sonic": args.w_sonic, "w_genre": args.w_genre},
            "components": {"sonic": args.n_components_sonic, "genre": args.n_components_genre},
            "alpha_schedule": {
                "schedule": alpha_schedule_effective,
                "start": alpha_start_effective,
                "mid": alpha_mid_effective,
                "end": alpha_end_effective,
                "arc_midpoint": arc_midpoint_effective,
            },
        }
        np.savez(
            out_path,
            seq_indices=np.array(seq, dtype=int),
            track_ids=np.array([_safe_str(track_ids, i) for i in seq]),
            artist_keys=np.array([artist_keys[i] for i in seq]),
            seed_sims=np.array(seed_sim_list, dtype=float),
            local_sims=np.array(local_sim_list, dtype=float),
            params_json=json.dumps(params),
            step_stats=np.array(step_stats, dtype=object),
            artist_counts=np.array(list(artist_counter.items()), dtype=object),
            local_sim_segment=np.array(local_seg_list, dtype=float),
            local_sim_hybrid=np.array(local_hyb_list, dtype=float),
            alpha_t_used=np.array(alpha_t_list, dtype=float),
        )
        print(f"Saved playlist NPZ to {out_path}")


if __name__ == "__main__":
    main()
