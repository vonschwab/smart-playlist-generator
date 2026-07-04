"""Perceptual BPM resolution and log-space distance metrics.

The raw `primary_bpm` from beat tracking is ambiguous at 2:1 ratios — a
slow track with strong half-time feel may be detected at twice its "felt"
tempo, and vice versa. The extractor flags this via `half_tempo_likely`
and `double_tempo_likely`. We use those flags to recover the perceptual
tempo before comparing.

Comparison happens in log-space: a 2:1 BPM ratio is a distance of 1.0
("one octave"), reflecting how dramatically different 70 BPM and 140 BPM
feel even though they're harmonically related.
"""
from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def resolve_perceptual_bpm(
    primary_bpm: float,
    *,
    half_tempo_likely: bool,
    double_tempo_likely: bool,
) -> float:
    """Resolve perceived tempo from detected tempo + half/double flags."""
    if bool(half_tempo_likely) and not bool(double_tempo_likely):
        return float(primary_bpm) * 2.0
    if bool(double_tempo_likely) and not bool(half_tempo_likely):
        return float(primary_bpm) / 2.0
    return float(primary_bpm)


def bpm_log_distance(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """|log2(a / b)|. Returns inf for non-positive inputs."""
    # Scalar fast path (the per-candidate beam hot call, ~4M/gen). Skips the
    # np.asarray + two np.where array allocations that dominate a scalar call.
    # Keeps np.log2 (no math-lib swap) so it is bit-identical to the array path
    # below — float(a)/float(b) and np.log2 produce the identical bits a 0-d
    # array would; only the wasted array machinery is gone.
    if isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)):
        if a <= 0 or b <= 0:
            return np.inf
        return abs(float(np.log2(a / b)))
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    invalid = (a_arr <= 0) | (b_arr <= 0)
    safe_a = np.where(invalid, 1.0, a_arr)
    safe_b = np.where(invalid, 1.0, b_arr)
    dist = np.abs(np.log2(safe_a / safe_b))
    return np.where(invalid, np.inf, dist)


def interpolate_log_bpm(
    bpm_a: float,
    bpm_b: float,
    *,
    t: float,
) -> float:
    """Log-space interpolation. t=0 → bpm_a, t=1 → bpm_b, t=0.5 → geometric mean."""
    if bpm_a <= 0 or bpm_b <= 0:
        return float(bpm_a if t <= 0.5 else bpm_b)
    log_a = np.log2(bpm_a)
    log_b = np.log2(bpm_b)
    t_clamped = max(0.0, min(1.0, float(t)))
    return float(2.0 ** ((1.0 - t_clamped) * log_a + t_clamped * log_b))
