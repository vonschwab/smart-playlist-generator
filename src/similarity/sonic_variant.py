from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

_cache: Dict[Tuple[int, str, bool], Tuple[np.ndarray, dict]] = {}
_ALLOWED = {"raw", "centered", "z", "z_clip", "whiten_pca", "robust_whiten"}


def _normalize_variant_name(name: Optional[str]) -> str:
    # Default to robust_whiten (validated as best-performing in beat3tower tests)
    if not name:
        return "robust_whiten"
    variant = str(name).strip().lower()
    if variant not in _ALLOWED:
        return "robust_whiten"
    return variant


def resolve_sonic_variant(explicit_variant: Optional[str] = None, config_variant: Optional[str] = None) -> str:
    """
    Pick variant with priority: explicit (CLI) -> env (SONIC_SIM_VARIANT) -> config value -> raw.
    Unknown values fall back to raw.
    """
    if explicit_variant:
        return _normalize_variant_name(explicit_variant)
    env_val = os.getenv("SONIC_SIM_VARIANT")
    if env_val:
        return _normalize_variant_name(env_val)
    return _normalize_variant_name(config_variant)


def get_variant_from_env() -> str:
    return resolve_sonic_variant(None, None)


def _variant_transform(X_sonic: np.ndarray, variant: str) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    variant = _normalize_variant_name(variant)
    # raw: cosine on original X_sonic
    # centered: subtract global mean
    # z: per-dimension z-score
    # z_clip: z-score then clip to [-3, 3]
    # whiten_pca: z-score then PCA whitening
    # robust_whiten: robust scaling (median/IQR) then PCA whitening
    mean = X_sonic.mean(axis=0, keepdims=True)
    std_used = None
    if variant == "raw":
        mat = X_sonic
    elif variant == "centered":
        mat = X_sonic - mean
    elif variant in {"z", "z_clip"}:
        std = X_sonic.std(axis=0, keepdims=True)
        std_safe = np.where(std < 1e-12, 1.0, std)
        mat = (X_sonic - mean) / std_safe
        std_used = std_safe
        if variant == "z_clip":
            mat = np.clip(mat, -3.0, 3.0)
    elif variant == "whiten_pca":
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaled = scaler.fit_transform(X_sonic)
        pca = PCA(whiten=True)
        mat = pca.fit_transform(scaled)
        std_used = scaler.scale_[np.newaxis, :]
    elif variant == "robust_whiten":
        # Robust scaling: uses median and IQR instead of mean/std (resistant to outliers)
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        scaled = scaler.fit_transform(X_sonic)
        # PCA whitening: decorrelates features and normalizes variance
        pca = PCA(whiten=True)
        mat = pca.fit_transform(scaled)
        std_used = scaler.scale_[np.newaxis, :]
    else:  # pragma: no cover - defensive fallback
        mat = X_sonic
    stats = {
        "variant": variant,
        "dim": int(X_sonic.shape[1]),
    }
    if std_used is not None:
        stats["mean_std"] = float(np.mean(std_used))
        stats["min_std"] = float(np.min(std_used))
        stats["max_std"] = float(np.max(std_used))
    # Let downstream callers know if the matrix is already centred / scaled / whitened
    # so they can avoid re-standardising and erasing the variant effect.
    stats["pre_scaled"] = variant != "raw"
    return mat, stats, std_used


def compute_sonic_variant_matrix(X_sonic: np.ndarray, variant: str = "robust_whiten", *, l2: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Apply variant transform (raw/centered/z/z_clip/whiten_pca) with optional L2 row normalisation.
    Caches per (id(X_sonic), variant, l2) to avoid redundant work.
    """
    variant = _normalize_variant_name(variant)
    key = (id(X_sonic), variant, bool(l2))
    if key in _cache:
        return _cache[key]
    mat, stats, _ = _variant_transform(X_sonic, variant)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    if l2:
        mat = mat / norms
    stats = {**stats, "mean_norm": float(np.mean(norms))}
    _cache[key] = (mat, stats)
    return mat, stats


def compute_sonic_variant_norm(X_sonic: np.ndarray, variant: str = "robust_whiten") -> Tuple[np.ndarray, dict]:
    """
    Return row-normalized sonic matrix per variant (raw/centered/z/...), with lightweight stats.
    """
    return compute_sonic_variant_matrix(X_sonic, variant, l2=True)
