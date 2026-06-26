from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.playlist.pier_bridge.vec import _calibrate_transition_cos, _l2_normalize_rows

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionMetricContext:
    """Prepared matrices and weights for the final edge transition metric."""

    X_full: np.ndarray
    X_start: Optional[np.ndarray]
    X_mid: Optional[np.ndarray]
    X_end: Optional[np.ndarray]
    X_sonic_norm: np.ndarray
    X_genre_norm: Optional[np.ndarray] = None
    X_hybrid_norm: Optional[np.ndarray] = None
    center_transitions: bool = False
    weight_end_start: float = 0.70
    weight_mid_mid: float = 0.15
    weight_full_full: float = 0.15
    transition_gamma: Optional[float] = None
    # Calibrated-sigmoid rescale params (used when center_transitions=True).
    # Fixed constants derived once from the library cosine band; see
    # docs/superpowers/specs/2026-06-25-sonic-centered-transition-design.md.
    calib_center: float = 0.32
    calib_scale: float = 0.0625
    calib_gain: float = 1.0


def _norm_optional(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mat is None:
        return None
    return _l2_normalize_rows(mat)


def _center_optional(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mat is None:
        return None
    return mat - mat.mean(axis=0, keepdims=True)


def _safe_dot(mat_a: Optional[np.ndarray], mat_b: Optional[np.ndarray], idx_a: int, idx_b: int) -> float:
    if mat_a is None or mat_b is None:
        return float("nan")
    try:
        return float(np.dot(mat_a[int(idx_a)], mat_b[int(idx_b)]))
    except Exception:
        return float("nan")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False



def build_transition_metric_context(
    *,
    X_sonic: np.ndarray,
    X_start: Optional[np.ndarray] = None,
    X_mid: Optional[np.ndarray] = None,
    X_end: Optional[np.ndarray] = None,
    X_genre: Optional[np.ndarray] = None,
    center_transitions: bool = False,
    transition_weights: Optional[tuple[float, float, float]] = None,
    sonic_variant: Optional[str] = None,
    transition_gamma: Optional[float] = None,
    embedding_random_seed: Optional[int] = None,
    weight_end_start: float = 0.70,
    weight_mid_mid: float = 0.15,
    weight_full_full: float = 0.15,
    calib_center: float = 0.32,
    calib_scale: float = 0.0625,
    calib_gain: float = 1.0,
) -> TransitionMetricContext:
    """Build the shared transition metric context from raw artifact matrices."""

    from src.similarity.sonic_variant import (
        apply_transition_weights,
        compute_sonic_variant_norm,
        resolve_sonic_variant,
    )

    variant = resolve_sonic_variant(explicit_variant=sonic_variant, config_variant=None)
    X_sonic_norm, _ = compute_sonic_variant_norm(X_sonic, variant)

    X_full_tr, _ = apply_transition_weights(X_sonic, config_weights=transition_weights)
    X_start_tr = None
    X_mid_tr = None
    X_end_tr = None
    if X_start is not None:
        X_start_tr, _ = apply_transition_weights(X_start, config_weights=transition_weights)
    if X_mid is not None:
        X_mid_tr, _ = apply_transition_weights(X_mid, config_weights=transition_weights)
    if X_end is not None:
        X_end_tr, _ = apply_transition_weights(X_end, config_weights=transition_weights)

    if center_transitions:
        X_full_tr = _center_optional(X_full_tr)
        X_start_tr = _center_optional(X_start_tr)
        X_mid_tr = _center_optional(X_mid_tr)
        X_end_tr = _center_optional(X_end_tr)

    X_genre_norm = _norm_optional(X_genre)
    X_hybrid_norm = None
    if X_genre is not None:
        try:
            from src.similarity.hybrid import build_hybrid_embedding

            emb_model = build_hybrid_embedding(
                X_sonic,
                X_genre,
                n_components_sonic=32,
                n_components_genre=32,
                w_sonic=1.0,
                w_genre=1.0,
                random_seed=embedding_random_seed or 0,
            )
            X_hybrid_norm = _l2_normalize_rows(emb_model.embedding)
        except Exception as exc:  # pragma: no cover - defensive diagnostic
            logger.debug("Transition metric: failed to build hybrid context (%s)", exc)

    return TransitionMetricContext(
        X_full=_l2_normalize_rows(X_full_tr),
        X_start=_norm_optional(X_start_tr),
        X_mid=_norm_optional(X_mid_tr),
        X_end=_norm_optional(X_end_tr),
        X_sonic_norm=X_sonic_norm,
        X_genre_norm=X_genre_norm,
        X_hybrid_norm=X_hybrid_norm,
        center_transitions=bool(center_transitions),
        weight_end_start=float(weight_end_start),
        weight_mid_mid=float(weight_mid_mid),
        weight_full_full=float(weight_full_full),
        transition_gamma=(float(transition_gamma) if transition_gamma is not None else None),
        calib_center=float(calib_center),
        calib_scale=float(calib_scale),
        calib_gain=float(calib_gain),
    )


def score_transition_edge(context: TransitionMetricContext, prev_idx: int, cur_idx: int) -> dict:
    """Score one directed playlist edge with the shared final-edge metric."""

    prev_idx = int(prev_idx)
    cur_idx = int(cur_idx)

    sim_full_raw = _safe_dot(context.X_full, context.X_full, prev_idx, cur_idx)
    sim_end_start_raw = _safe_dot(context.X_end, context.X_start, prev_idx, cur_idx)
    if not _finite(sim_end_start_raw):
        sim_end_start_raw = sim_full_raw
    sim_mid_raw = _safe_dot(context.X_mid, context.X_mid, prev_idx, cur_idx)
    if not _finite(sim_mid_raw):
        sim_mid_raw = sim_full_raw

    t_raw = (
        float(context.weight_end_start) * float(sim_end_start_raw)
        + float(context.weight_mid_mid) * float(sim_mid_raw)
        + float(context.weight_full_full) * float(sim_full_raw)
    )
    if context.center_transitions:
        def _r(x: float) -> float:
            return _calibrate_transition_cos(
                x,
                center=context.calib_center,
                scale=context.calib_scale,
                gain=context.calib_gain,
            )

        t_val = (
            float(context.weight_end_start) * _r(sim_end_start_raw)
            + float(context.weight_mid_mid) * _r(sim_mid_raw)
            + float(context.weight_full_full) * _r(sim_full_raw)
        )
    else:
        t_val = t_raw

    s_val = _safe_dot(context.X_sonic_norm, context.X_sonic_norm, prev_idx, cur_idx)
    g_val = _safe_dot(context.X_genre_norm, context.X_genre_norm, prev_idx, cur_idx)
    h_val = _safe_dot(context.X_hybrid_norm, context.X_hybrid_norm, prev_idx, cur_idx)
    if not _finite(h_val):
        h_val = s_val

    edge = {
        "T": float(t_val),
        "T_raw": float(t_raw),
        "T_centered_cos": float(sim_end_start_raw),
        "H": float(h_val),
        "S": float(s_val),
        "floor": None,
        "gamma": context.transition_gamma,
    }
    edge["G"] = float(g_val) if _finite(g_val) else None
    return edge


def is_broken_transition(
    edge: dict,
    transition_floor: float,
    centered_cos_floor: Optional[float] = None,
) -> bool:
    """Return True only when an edge is catastrophically anti-aligned.

    Roam-only: the ``transition_floor`` HARD GATE is removed. With a
    discriminating ``T`` (the calibrated sigmoid), the beam objective and roam's
    worst-edge minimax already prefer good edges by *optimization* — eliminating
    candidates on a `T` floor only adds cascade/budget risk (north star #5 +
    the 90 s ceiling). The ``transition_floor`` parameter is retained for the
    legacy-cascade callers that still pass it; it no longer gates. (The cascade
    plumbing itself is swept by the roam-promotion, not here.)

    The ``-0.5`` ``centered_cos_floor`` safety stays: it gates the *raw*
    end→start cosine (`T_centered_cos`), catching an edge that is actively
    *opposite* (essentially never fires in the anisotropic MERT space).
    """

    if centered_cos_floor is not None:
        centered = edge.get("T_centered_cos")
        if _finite(centered) and float(centered) < float(centered_cos_floor):
            return True
    return False
