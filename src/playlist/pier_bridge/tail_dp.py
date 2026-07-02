"""Segment tail re-optimization (tail-DP) — the c-tail landing fix.

The beam picks the last interior slots effectively blind to the landing edge
(spec 2026-07-02: 6/9 segments left >0.2 in-pool min-edge unused). This module
re-opens the last min(2, interior) slots after the segment is finalized and
exactly maximizes the window min-edge over the segment pool. Pure functions —
the builder supplies constraints via a callback. Never-worse by construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TailSwap:
    new_tail: tuple[int, ...]
    old_min: float
    new_min: float


def _calibrate(ctx, arr: np.ndarray) -> np.ndarray:
    """Vectorized twin of vec._calibrate_transition_cos, honoring ctx flags."""
    if not ctx.center_transitions:
        return np.asarray(arr, dtype=np.float64)
    z = float(ctx.calib_gain) * (np.asarray(arr, dtype=np.float64) - float(ctx.calib_center)) / float(ctx.calib_scale)
    return 1.0 / (1.0 + np.exp(-z))


def batch_T(ctx, src_indices: Sequence[int], dst_indices: Sequence[int]) -> np.ndarray:
    """Calibrated blended transition T from each src to each dst.

    Exactly mirrors score_transition_edge: per-component calibration THEN the
    weighted blend. The end->start component falls back to full-full whenever
    EITHER X_end or X_start is absent (not independently substituted); mid
    falls back to full-full when X_mid is absent.
    """
    src = np.asarray(list(src_indices), dtype=int)
    dst = np.asarray(list(dst_indices), dtype=int)
    X_full = ctx.X_full
    ff = X_full[src] @ X_full[dst].T
    if ctx.X_end is not None and ctx.X_start is not None:
        es = ctx.X_end[src] @ ctx.X_start[dst].T
    else:
        es = ff
    mm = ctx.X_mid[src] @ ctx.X_mid[dst].T if ctx.X_mid is not None else ff
    return (
        float(ctx.weight_end_start) * _calibrate(ctx, es)
        + float(ctx.weight_mid_mid) * _calibrate(ctx, mm)
        + float(ctx.weight_full_full) * _calibrate(ctx, ff)
    )


def optimize_segment_tail(
    ctx,
    *,
    segment_path: list[int],
    pier_a: int,
    pier_b: int,
    candidates: Sequence[int],
    epsilon: float,
    is_allowed_pair: Callable[[int, int], bool],
    max_pairs_checked: int = 50,
) -> Optional[TailSwap]:
    """Best allowed re-fill of the last min(2, len(path)) interior slots.

    Returns None when nothing beats the existing tail by >= epsilon (never-
    worse), when the path/candidates are empty, or on internal error (logged).
    """
    try:
        path = [int(i) for i in segment_path]
        cand = [int(c) for c in candidates if int(c) != int(pier_b)]
        if not path or not cand:
            return None
        window = min(2, len(path))
        prefix_end = int(path[-(window + 1)]) if len(path) > window else int(pier_a)

        if window == 1:
            t_in = batch_T(ctx, [prefix_end], cand)[0]      # prefix -> x
            t_out = batch_T(ctx, cand, [pier_b])[:, 0]      # x -> pier_b
            scores = np.minimum(t_in, t_out)
            old_min = float(np.min(batch_T(ctx, [prefix_end, path[-1]],
                                           [path[-1], pier_b])[[0, 1], [0, 1]]))
            order = np.argsort(-scores, kind="stable")
            for rank in order[: int(max_pairs_checked)]:
                x = cand[int(rank)]
                if float(scores[rank]) < old_min + float(epsilon):
                    return None
                if is_allowed_pair(x, x):
                    return TailSwap((x,), old_min, float(scores[rank]))
            return None

        # window == 2
        t_in = batch_T(ctx, [prefix_end], cand)[0]          # prefix -> x
        t_mid = batch_T(ctx, cand, cand)                    # x -> y
        t_out = batch_T(ctx, cand, [pier_b])[:, 0]          # y -> pier_b
        M = np.minimum(np.minimum(t_in[:, None], t_mid), t_out[None, :])
        np.fill_diagonal(M, -np.inf)

        ex, ey = path[-2], path[-1]
        old_edges = [
            batch_T(ctx, [prefix_end], [ex])[0, 0],
            batch_T(ctx, [ex], [ey])[0, 0],
            batch_T(ctx, [ey], [pier_b])[0, 0],
        ]
        old_min = float(min(old_edges))

        flat_order = np.argsort(-M, axis=None, kind="stable")
        for flat in flat_order[: int(max_pairs_checked)]:
            xi, yi = np.unravel_index(int(flat), M.shape)
            score = float(M[xi, yi])
            if not np.isfinite(score) or score < old_min + float(epsilon):
                return None
            x, y = cand[int(xi)], cand[int(yi)]
            if is_allowed_pair(x, y):
                return TailSwap((x, y), old_min, score)
        return None
    except Exception:
        logger.warning("tail_dp: internal error; keeping original segment tail", exc_info=True)
        return None
