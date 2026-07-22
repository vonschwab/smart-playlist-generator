from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.playlist.pier_bridge.vec import _calibrate_transition_cos, _l2_normalize_rows

logger = logging.getLogger(__name__)


# Per-variant transition-cosine calibration (center, scale). The realistic
# centered end->start cosine band differs by sonic embedding, so the rescale
# sigmoid's center/scale MUST track the active variant or it saturates. Bands
# derived on the full 41k artifact via
# scripts/research/calibrate_transition_sigmoid.py <variant>.
#   muq:  band p1/p50/p99 ~ 0.32/0.55/0.87   (contrastive; runs hot)
TRANSITION_CALIB_BY_VARIANT: dict[str, tuple[float, float]] = {
    "muq": (0.594, 0.092),
}
# muq is the sole registered variant (SP-B removed MERT/towers); a None
# variant (defensive only — the loader always stamps one) maps to muq.
_DEFAULT_CALIB_VARIANT = "muq"


def resolve_transition_calib(
    variant: Optional[str],
    *,
    override: Optional[tuple[float, ...]] = None,
) -> tuple[float, float, float]:
    """Resolve (center, scale, gain) transition-sigmoid params for ``variant``.

    Priority: an explicit ``override`` (center, scale[, gain]) wins, for tuning;
    otherwise the per-variant band table. A ``None``/empty variant maps to the
    muq band. An unknown variant with no override RAISES — a configured sonic
    space the transition rescale can't calibrate is a startup error, not a
    silent fallback that would saturate every edge (the project's #1 failure
    mode).
    """
    if override is not None:
        center, scale = float(override[0]), float(override[1])
        gain = float(override[2]) if len(override) > 2 else 1.0
        return center, scale, gain
    key = (variant or _DEFAULT_CALIB_VARIANT).strip().lower()
    if key in TRANSITION_CALIB_BY_VARIANT:
        center, scale = TRANSITION_CALIB_BY_VARIANT[key]
        return float(center), float(scale), 1.0
    raise ValueError(
        f"No transition calibration for sonic variant {variant!r}. Derive its band by fitting a "
        f"transition sigmoid for {key!r} and add an entry to "
        "TRANSITION_CALIB_BY_VARIANT, or pass an explicit override — a configured sonic space "
        "the transition rescale can't calibrate is a startup error, not a silent fallback."
    )


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
    # Fixed constants derived once from the muq cosine band; see
    # docs/superpowers/specs/2026-06-25-sonic-centered-transition-design.md.
    calib_center: float = 0.594
    calib_scale: float = 0.092
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
    transition_gamma: Optional[float] = None,
    embedding_random_seed: Optional[int] = None,
    weight_end_start: float = 0.70,
    weight_mid_mid: float = 0.15,
    weight_full_full: float = 0.15,
    calib_center: float = 0.594,
    calib_scale: float = 0.092,
    calib_gain: float = 1.0,
) -> TransitionMetricContext:
    """Build the shared transition metric context from raw artifact matrices."""

    # One sonic space (muq): plain L2-normalized cosine, no tower transforms.
    X_sonic_norm = _l2_normalize_rows(X_sonic)

    X_full_tr = X_sonic
    X_start_tr = X_start
    X_mid_tr = X_mid
    X_end_tr = X_end

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
