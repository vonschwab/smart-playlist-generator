"""Surgical tower-weighted sonic rebuild.

Recompute ONLY the sonic matrices of an existing artifact as per-tower
``sqrt(weight) * L2(tower)``, from per-tower matrices already stored in the npz.
Genre matrices and track order are copied byte-identical, so the dense-genre
sidecar stays valid and no DB / audio is read. See
docs/superpowers/specs/2026-06-01-sonic-tower-weighted-fix-design.md.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float64)
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return mat / norms


def tower_weighted_from_towers(
    rhythm: np.ndarray,
    timbre: np.ndarray,
    harmony: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    """Per-tower L2-normalise, scale each by sqrt(weight), concatenate.

    Each output row's per-tower sub-vector has norm sqrt(weight), so the tower
    weighting is applied exactly and is invariant to the towers' raw scales.
    """
    w_r, w_t, w_h = (float(w) for w in weights)
    scales = np.sqrt(np.array([w_r, w_t, w_h], dtype=np.float64))
    out = np.concatenate(
        [
            scales[0] * _l2_rows(rhythm),
            scales[1] * _l2_rows(timbre),
            scales[2] * _l2_rows(harmony),
        ],
        axis=1,
    )
    return out.astype(np.float32)


def build_tower_weighted_arrays(
    data: Mapping[str, np.ndarray],
    weights: Tuple[float, float, float],
) -> dict:
    """Return a full set of npz arrays with sonic matrices rebuilt tower_weighted.

    Copies every key from ``data`` unchanged, then overwrites:
      - adds ``X_sonic_tower_weighted`` (full) — the variant key the loader selects
      - overwrites ``X_sonic_start`` / ``_mid`` / ``_end`` (loader reads these directly)
      - sets ``X_sonic_variant`` = "tower_weighted", ``X_sonic_pre_scaled`` = True
    The raw ``X_sonic`` key and all genre/track/per-tower keys are left intact.
    """
    out = {k: data[k] for k in data.files}  # type: ignore[attr-defined]

    out["X_sonic_tower_weighted"] = tower_weighted_from_towers(
        data["X_sonic_rhythm"], data["X_sonic_timbre"], data["X_sonic_harmony"], weights
    )
    for seg in ("start", "mid", "end"):
        out[f"X_sonic_{seg}"] = tower_weighted_from_towers(
            data[f"X_sonic_rhythm_{seg}"],
            data[f"X_sonic_timbre_{seg}"],
            data[f"X_sonic_harmony_{seg}"],
            weights,
        )
    out["X_sonic_variant"] = np.array("tower_weighted")
    out["X_sonic_pre_scaled"] = np.array(True)
    return out
