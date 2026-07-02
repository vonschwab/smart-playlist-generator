"""Single-position replacement candidate scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.playlist.bpm_axis import bpm_log_distance
from src.playlist.transition_metrics import TransitionMetricContext, score_transition_edge


SUPPORTED_MODES = ("best", "different_pace", "different_genre", "different_sound")
DEFAULT_T_MIN = 0.20
DEFAULT_FILTER_K = 50


@dataclass(frozen=True)
class ReplacementContext:
    X_sonic: np.ndarray
    X_full: np.ndarray
    X_start: Optional[np.ndarray]
    X_end: Optional[np.ndarray]
    X_mid: Optional[np.ndarray]
    X_genre_smoothed: Optional[np.ndarray]
    perceptual_bpm: Optional[np.ndarray]
    tempo_stability: Optional[np.ndarray]
    track_ids: np.ndarray
    artist_keys: np.ndarray
    candidate_pool_indices: np.ndarray
    idf_weights: Optional[np.ndarray] = None
    transition_metric_context: Optional[TransitionMetricContext] = None
    transition_floor: float = DEFAULT_T_MIN


def _eligible_candidate_indices(
    ctx: ReplacementContext,
    *,
    prev_idx: int,
    next_idx: int,
    current_idx: int,
    playlist_indices: Sequence[int],
) -> np.ndarray:
    pool = np.asarray(ctx.candidate_pool_indices, dtype=int)
    if pool.size == 0:
        return pool

    excluded = set(int(i) for i in playlist_indices) | {int(current_idx)}
    prev_artist = str(ctx.artist_keys[int(prev_idx)])
    next_artist = str(ctx.artist_keys[int(next_idx)])

    keep: list[int] = []
    for raw_idx in pool:
        idx = int(raw_idx)
        if idx in excluded:
            continue
        artist = str(ctx.artist_keys[idx])
        if artist == prev_artist or artist == next_artist:
            continue
        keep.append(idx)
    return np.asarray(keep, dtype=int)


def _transition_quality(
    ctx: ReplacementContext,
    *,
    prev_idx: int,
    cand_idx: int,
    next_idx: int,
) -> tuple[float, float]:
    if ctx.transition_metric_context is not None:
        t_prev = float(score_transition_edge(ctx.transition_metric_context, int(prev_idx), int(cand_idx))["T"])
        t_next = float(score_transition_edge(ctx.transition_metric_context, int(cand_idx), int(next_idx))["T"])
        return t_prev, t_next

    def _cos01(a_idx: int, b_idx: int) -> float:
        a = np.asarray(ctx.X_full[int(a_idx)], dtype=float)
        b = np.asarray(ctx.X_full[int(b_idx)], dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.clip((float(np.dot(a, b) / denom) + 1.0) / 2.0, 0.0, 1.0))

    return _cos01(prev_idx, cand_idx), _cos01(cand_idx, next_idx)


def _safe_cosine_divergence(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - np.dot(a, b) / denom)


def _genre_divergence(ctx: ReplacementContext, *, cand_idx: int, current_idx: int) -> float:
    if ctx.X_genre_smoothed is None:
        return 0.0
    a = np.asarray(ctx.X_genre_smoothed[int(current_idx)], dtype=float)
    b = np.asarray(ctx.X_genre_smoothed[int(cand_idx)], dtype=float)
    if ctx.idf_weights is not None:
        weights = np.asarray(ctx.idf_weights, dtype=float)
        a = a * weights
        b = b * weights
    return _safe_cosine_divergence(a, b)


def _pace_divergence(ctx: ReplacementContext, *, cand_idx: int, current_idx: int) -> float:
    if ctx.perceptual_bpm is not None:
        cand_bpm = float(ctx.perceptual_bpm[int(cand_idx)])
        current_bpm = float(ctx.perceptual_bpm[int(current_idx)])
        if np.isfinite(cand_bpm) and np.isfinite(current_bpm) and cand_bpm > 0 and current_bpm > 0:
            return float(bpm_log_distance(cand_bpm, current_bpm))
    # No usable BPM: no pace signal (the tower rhythm axis was removed in SP-B).
    return 0.0


def _sound_divergence(ctx: ReplacementContext, *, cand_idx: int, current_idx: int) -> float:
    # Full-sonic cosine divergence on the loaded (muq) matrix. The old tower
    # "color" carving was meaningless on a no-tower embedding.
    a = np.asarray(ctx.X_sonic[int(current_idx)], dtype=float)
    b = np.asarray(ctx.X_sonic[int(cand_idx)], dtype=float)
    return _safe_cosine_divergence(a, b)


def find_replacement_candidates(
    *,
    prev_idx: int,
    next_idx: int,
    current_idx: int,
    playlist_indices: Sequence[int],
    ctx: ReplacementContext,
    mode: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown replacement mode: '{mode}'. Supported: {SUPPORTED_MODES}")

    eligible = _eligible_candidate_indices(
        ctx,
        prev_idx=prev_idx,
        next_idx=next_idx,
        current_idx=current_idx,
        playlist_indices=playlist_indices,
    )
    if eligible.size == 0:
        return []

    scored: list[tuple[int, float, float, float]] = []
    for raw_idx in eligible:
        idx = int(raw_idx)
        t_prev, t_next = _transition_quality(ctx, prev_idx=prev_idx, cand_idx=idx, next_idx=next_idx)
        mean_t = 0.5 * (t_prev + t_next)
        if t_prev < float(ctx.transition_floor) or t_next < float(ctx.transition_floor):
            continue
        scored.append((idx, t_prev, t_next, mean_t))

    if not scored:
        return []

    scored.sort(key=lambda item: item[3], reverse=True)
    if mode == "best":
        chosen = scored[: int(top_k)]
    else:
        divergence_func = {
            "different_pace": _pace_divergence,
            "different_genre": _genre_divergence,
            "different_sound": _sound_divergence,
        }[mode]
        rerank_pool = scored[: max(DEFAULT_FILTER_K, int(top_k))]
        with_divergence = [
            (idx, t_prev, t_next, mean_t, divergence_func(ctx, cand_idx=idx, current_idx=current_idx))
            for idx, t_prev, t_next, mean_t in rerank_pool
        ]
        with_divergence.sort(key=lambda item: (item[4], item[3]), reverse=True)
        chosen = [(idx, t_prev, t_next, mean_t) for idx, t_prev, t_next, mean_t, _ in with_divergence[: int(top_k)]]

    result: list[dict[str, Any]] = []
    for idx, t_prev, t_next, mean_t in chosen:
        entry: dict[str, Any] = {
            "index": int(idx),
            "track_id": str(ctx.track_ids[int(idx)]),
            "artist_key": str(ctx.artist_keys[int(idx)]),
            "t_prev": float(t_prev),
            "t_next": float(t_next),
            "mean_t": float(mean_t),
        }
        if ctx.perceptual_bpm is not None:
            bpm = float(ctx.perceptual_bpm[int(idx)])
            if np.isfinite(bpm):
                entry["perceptual_bpm"] = bpm
        result.append(entry)
    return result
