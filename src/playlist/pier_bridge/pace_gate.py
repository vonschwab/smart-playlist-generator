"""BPM/onset/energy moving-target pace gates for pier-bridge beam search.

The rhythm-tower-axis gate (cosine similarity on a PCA rhythm sub-vector) was
removed in SP-B: under muq (no tower decomposition) it had already fallen
back to the perceptual-BPM band permanently, so BPM/onset bands are now the
only pace-gate path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.playlist.sonic_axes import interpolate_axis_vector
from src.playlist.bpm_axis import interpolate_log_bpm, bpm_log_distance as _bpm_log_distance


def compute_step_log_bpm_target(
    bpm_a: float,
    bpm_b: float,
    *,
    step: int,
    segment_length: int,
) -> float:
    """Return log-space interpolated target BPM at beam step `step`.

    Uses geometric mean interpolation: midpoint of 60 and 240 is 120, not 150.
    """
    if int(segment_length) <= 0:
        return float(bpm_a)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_log_bpm(float(bpm_a), float(bpm_b), t=t)


def bpm_fallback_max_log_distance(pace_bridge_floor: float) -> float:
    """Map a legacy rhythm-cosine bridge-floor tightness onto a perceptual-BPM
    log-distance cap.

    Since SP-B, BPM/onset bands are the only pace-gate path; this ladder picks
    the BPM cap matching a configured ``pace_bridge_floor`` when no explicit
    ``bpm_bridge_max_log_distance`` is set. The ladder mirrors PACE_MODE_PRESETS
    (bridge_floor → bpm_bridge_max_log_distance) so it stays consistent with
    the pace_mode presets.
    """
    from src.playlist.mode_presets import PACE_MODE_PRESETS

    ladder = sorted(
        (
            (float(p["bridge_floor"]), float(p["bpm_bridge_max_log_distance"]))
            for p in PACE_MODE_PRESETS.values()
            if float(p.get("bridge_floor", 0.0)) > 0.0
            and np.isfinite(float(p.get("bpm_bridge_max_log_distance", float("inf"))))
        ),
        reverse=True,
    )
    for floor, distance in ladder:
        if float(pace_bridge_floor) >= floor:
            return distance
    # Below the loosest preset floor: use the loosest finite cap.
    return ladder[-1][1] if ladder else 0.85


def compute_step_log_onset_target(
    onset_a: float,
    onset_b: float,
    *,
    step: int,
    segment_length: int,
) -> float:
    """Log-space interpolated onset-rate target at beam step `step`.

    Onset rate is a positive event-density rate, so it interpolates in
    log-space exactly like BPM (geometric mean at the midpoint).
    """
    if int(segment_length) <= 0:
        return float(onset_a)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_log_bpm(float(onset_a), float(onset_b), t=t)


def filter_candidates_by_onset_target(
    *,
    candidate_indices,
    onset_rate: np.ndarray,
    target_onset: float,
    max_log_distance: float,
) -> list:
    """Drop candidates whose onset-rate log-distance to target exceeds the cap.

    Candidates with NaN onset_rate bypass the gate (graceful coverage gap).
    No tempo-stability bypass: onset density is meaningful regardless of tempo
    tracking stability (unlike BPM).
    """
    if not np.isfinite(float(max_log_distance)):
        return list(candidate_indices)
    indices = list(candidate_indices)
    if not indices:
        return []
    cand_onset = onset_rate[indices]
    bypass = np.isnan(cand_onset)
    distances = _bpm_log_distance(cand_onset, float(target_onset))
    pass_mask = bypass | (distances <= float(max_log_distance))
    return [idx for idx, ok in zip(indices, pass_mask) if bool(ok)]


def filter_candidates_by_bpm_target(
    *,
    candidate_indices,
    perceptual_bpm: np.ndarray,
    tempo_stability,
    target_bpm: float,
    max_log_distance: float,
    stability_min: float = 0.5,
) -> list:
    """Drop candidates whose perceptual BPM is too far from target_bpm.

    Bypasses candidates with NaN BPM or low tempo_stability.
    """
    if not np.isfinite(float(max_log_distance)):
        return list(candidate_indices)
    indices = list(candidate_indices)
    if not indices:
        return []
    cand_bpm = perceptual_bpm[indices]
    bypass = np.isnan(cand_bpm)
    if tempo_stability is not None:
        cand_stab = np.asarray(tempo_stability)[indices]
        bypass = bypass | (cand_stab < float(stability_min))
    distances = _bpm_log_distance(cand_bpm, float(target_bpm))
    pass_mask = bypass | (distances <= float(max_log_distance))
    return [idx for idx, ok in zip(indices, pass_mask) if bool(ok)]


def compute_step_energy_target(
    e_a: np.ndarray,
    e_b: np.ndarray,
    *,
    step: int,
    segment_length: int,
) -> np.ndarray:
    """Linear pier->pier energy target at beam step `step` (energy is a linear scale)."""
    if int(segment_length) <= 0:
        return np.asarray(e_a, dtype=float)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_axis_vector(e_a, e_b, t)


def compute_energy_pace_penalty(
    energy_matrix: Optional[np.ndarray],
    *,
    current: int,
    cand: int,
    pier_a: int,
    pier_b: int,
    step: int,
    segment_length: int,
    step_cap: float,
    step_strength: float,
    arc_band: float,
    arc_strength: float,
) -> float:
    """SOFT pace penalty (>= 0) for placing `cand` after `current`.

    Two terms: adjacent-step cap (energy distance current->cand) and arc-band
    (distance from the interpolated pier->pier target). NaN/None -> 0.0.
    NEVER raises and NEVER signals exclusion — callers only subtract this.
    """
    if energy_matrix is None:
        return 0.0
    penalty = 0.0
    e_cand = energy_matrix[int(cand)]
    if not np.all(np.isfinite(e_cand)):
        return 0.0
    # adjacent-step cap
    if step_strength > 0.0:
        e_cur = energy_matrix[int(current)]
        if np.all(np.isfinite(e_cur)):
            d_step = float(np.linalg.norm(e_cand - e_cur))
            if d_step > step_cap:
                penalty += step_strength * (d_step - step_cap)
    # arc-band
    if arc_strength > 0.0:
        e_a, e_b = energy_matrix[int(pier_a)], energy_matrix[int(pier_b)]
        if np.all(np.isfinite(e_a)) and np.all(np.isfinite(e_b)):
            target = compute_step_energy_target(e_a, e_b, step=step, segment_length=segment_length)
            d_arc = float(np.linalg.norm(e_cand - target))
            if d_arc > arc_band:
                penalty += arc_strength * (d_arc - arc_band)
    return penalty
