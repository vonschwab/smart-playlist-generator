from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

_cache: Dict[Tuple[int, str, bool], Tuple[np.ndarray, dict]] = {}
_ALLOWED = {
    "raw",
    "centered",
    "z",
    "z_clip",
    "whiten_pca",
    "robust_whiten",
    "tower_l2",
    "tower_robust",
    "tower_iqr",
    "tower_weighted",
    "tower_pca",
}


def _get_tower_weights(
    config_weights: Optional[tuple[float, float, float]] = None
) -> tuple[float, float, float]:
    """
    Return (rhythm, timbre, harmony) weights for tower_weighted/tower_pca.
    Priority: config_weights (passed) > SONIC_TOWER_WEIGHTS env > default.
    """
    default = (0.2, 0.5, 0.3)
    # Priority 1: explicit config override
    if config_weights is not None:
        if len(config_weights) == 3 and all(w >= 0 for w in config_weights):
            total = sum(config_weights)
            if total > 0:
                return (config_weights[0] / total, config_weights[1] / total, config_weights[2] / total)
    # Priority 2: environment variable
    env_val = os.getenv("SONIC_TOWER_WEIGHTS")
    if not env_val:
        return default
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    try:
        weights = tuple(float(p) for p in parts)
    except ValueError:
        return default
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        return default
    total = sum(weights)
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def _get_transition_weights(
    config_weights: Optional[tuple[float, float, float]] = None
) -> tuple[float, float, float]:
    """
    Return (rhythm, timbre, harmony) weights for transition scoring (end→start).
    Priority: config_weights (passed) > SONIC_TRANSITION_WEIGHTS env > default.
    Default prioritizes rhythm for smooth BPM/tempo flow between tracks.
    """
    default = (0.4, 0.35, 0.25)  # Rhythm-heavy for transitions
    # Priority 1: explicit config override
    if config_weights is not None:
        if len(config_weights) == 3 and all(w >= 0 for w in config_weights):
            total = sum(config_weights)
            if total > 0:
                return (config_weights[0] / total, config_weights[1] / total, config_weights[2] / total)
    # Priority 2: environment variable
    env_val = os.getenv("SONIC_TRANSITION_WEIGHTS")
    if not env_val:
        return default
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    try:
        weights = tuple(float(p) for p in parts)
    except ValueError:
        return default
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        return default
    total = sum(weights)
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def _get_tower_pca_dims(
    config_dims: Optional[tuple[int, int, int]] = None
) -> tuple[int, int, int]:
    """
    Return (rhythm, timbre, harmony) PCA component counts for tower_pca.
    Priority: config_dims (passed) > SONIC_TOWER_PCA env > default.
    """
    default = (8, 16, 8)
    # Priority 1: explicit config override
    if config_dims is not None:
        if len(config_dims) == 3 and all(d > 0 for d in config_dims):
            return tuple(int(d) for d in config_dims)
    # Priority 2: environment variable
    env_val = os.getenv("SONIC_TOWER_PCA")
    if not env_val:
        return default
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError:
        return default
    if any(d <= 0 for d in dims):
        return default
    return dims


def _normalize_variant_name(name: Optional[str]) -> str:
    # Default to tower_pca (current production variant)
    if not name:
        return "tower_pca"
    variant = str(name).strip().lower()
    if variant not in _ALLOWED:
        return "tower_pca"
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
    # tower_l2: per-tower L2 normalize (rhythm/timbre/harmony), then concatenate
    # tower_robust: per-tower robust scaling (median/IQR), then concatenate
    # tower_iqr: per-tower robust scaling without centering (IQR only), then concatenate
    # tower_weighted: per-tower L2 normalize, apply weights, then concatenate
    # tower_pca: per-tower StandardScaler + PCA, apply weights, then concatenate
    mean = X_sonic.mean(axis=0, keepdims=True)
    std_used = None
    tower_dims = None
    tower_fallback = False
    tower_pca_dims = None
    tower_pca_requested = None
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
    elif variant == "tower_l2":
        try:
            from src.features.beat3tower_types import (
                HarmonyTowerFeatures,
                RhythmTowerFeatures,
                TimbreTowerFeatures,
            )
            tower_dims = (
                RhythmTowerFeatures.n_features(),
                TimbreTowerFeatures.n_features(),
                HarmonyTowerFeatures.n_features(),
            )
        except Exception:
            tower_dims = None

        if tower_dims and X_sonic.shape[1] == sum(tower_dims):
            r_dim, t_dim, h_dim = tower_dims
            r_slice = slice(0, r_dim)
            t_slice = slice(r_dim, r_dim + t_dim)
            h_slice = slice(r_dim + t_dim, r_dim + t_dim + h_dim)

            def _l2_rows(arr: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                return arr / norms

            r_mat = _l2_rows(X_sonic[:, r_slice])
            t_mat = _l2_rows(X_sonic[:, t_slice])
            h_mat = _l2_rows(X_sonic[:, h_slice])
            mat = np.concatenate([r_mat, t_mat, h_mat], axis=1)
        else:
            mat = X_sonic
            tower_fallback = True
    elif variant == "tower_robust":
        try:
            from src.features.beat3tower_types import (
                HarmonyTowerFeatures,
                RhythmTowerFeatures,
                TimbreTowerFeatures,
            )
            tower_dims = (
                RhythmTowerFeatures.n_features(),
                TimbreTowerFeatures.n_features(),
                HarmonyTowerFeatures.n_features(),
            )
        except Exception:
            tower_dims = None

        if tower_dims and X_sonic.shape[1] == sum(tower_dims):
            r_dim, t_dim, h_dim = tower_dims
            r_slice = slice(0, r_dim)
            t_slice = slice(r_dim, r_dim + t_dim)
            h_slice = slice(r_dim + t_dim, r_dim + t_dim + h_dim)

            r_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
            t_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
            h_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))

            r_mat = r_scaler.fit_transform(X_sonic[:, r_slice])
            t_mat = t_scaler.fit_transform(X_sonic[:, t_slice])
            h_mat = h_scaler.fit_transform(X_sonic[:, h_slice])
            mat = np.concatenate([r_mat, t_mat, h_mat], axis=1)
        else:
            mat = X_sonic
            tower_fallback = True
    elif variant == "tower_iqr":
        try:
            from src.features.beat3tower_types import (
                HarmonyTowerFeatures,
                RhythmTowerFeatures,
                TimbreTowerFeatures,
            )
            tower_dims = (
                RhythmTowerFeatures.n_features(),
                TimbreTowerFeatures.n_features(),
                HarmonyTowerFeatures.n_features(),
            )
        except Exception:
            tower_dims = None

        if tower_dims and X_sonic.shape[1] == sum(tower_dims):
            r_dim, t_dim, h_dim = tower_dims
            r_slice = slice(0, r_dim)
            t_slice = slice(r_dim, r_dim + t_dim)
            h_slice = slice(r_dim + t_dim, r_dim + t_dim + h_dim)

            r_scaler = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0))
            t_scaler = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0))
            h_scaler = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0))

            r_mat = r_scaler.fit_transform(X_sonic[:, r_slice])
            t_mat = t_scaler.fit_transform(X_sonic[:, t_slice])
            h_mat = h_scaler.fit_transform(X_sonic[:, h_slice])
            mat = np.concatenate([r_mat, t_mat, h_mat], axis=1)
        else:
            mat = X_sonic
            tower_fallback = True
    elif variant == "tower_weighted":
        try:
            from src.features.beat3tower_types import (
                HarmonyTowerFeatures,
                RhythmTowerFeatures,
                TimbreTowerFeatures,
            )
            tower_dims = (
                RhythmTowerFeatures.n_features(),
                TimbreTowerFeatures.n_features(),
                HarmonyTowerFeatures.n_features(),
            )
        except Exception:
            tower_dims = None

        if tower_dims and X_sonic.shape[1] == sum(tower_dims):
            r_dim, t_dim, h_dim = tower_dims
            r_slice = slice(0, r_dim)
            t_slice = slice(r_dim, r_dim + t_dim)
            h_slice = slice(r_dim + t_dim, r_dim + t_dim + h_dim)

            def _l2_rows(arr: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                return arr / norms

            r_mat = _l2_rows(X_sonic[:, r_slice])
            t_mat = _l2_rows(X_sonic[:, t_slice])
            h_mat = _l2_rows(X_sonic[:, h_slice])

            w_r, w_t, w_h = _get_tower_weights()
            scales = np.sqrt(np.array([w_r, w_t, w_h], dtype=float))
            mat = np.concatenate(
                [scales[0] * r_mat, scales[1] * t_mat, scales[2] * h_mat],
                axis=1,
            )
        else:
            mat = X_sonic
            tower_fallback = True
    elif variant == "tower_pca":
        try:
            from src.features.beat3tower_types import (
                HarmonyTowerFeatures,
                RhythmTowerFeatures,
                TimbreTowerFeatures,
            )
            tower_dims = (
                RhythmTowerFeatures.n_features(),
                TimbreTowerFeatures.n_features(),
                HarmonyTowerFeatures.n_features(),
            )
        except Exception:
            tower_dims = None

        if tower_dims and X_sonic.shape[1] == sum(tower_dims):
            r_dim, t_dim, h_dim = tower_dims
            r_slice = slice(0, r_dim)
            t_slice = slice(r_dim, r_dim + t_dim)
            h_slice = slice(r_dim + t_dim, r_dim + t_dim + h_dim)

            r_req, t_req, h_req = _get_tower_pca_dims()

            def _pca_tower(mat: np.ndarray, n_req: int) -> np.ndarray:
                n_samples, n_features = mat.shape
                n_components = max(1, min(n_req, n_features, n_samples))
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaled = scaler.fit_transform(mat)
                pca = PCA(n_components=n_components, random_state=0, whiten=True)
                return pca.fit_transform(scaled)

            r_pca = _pca_tower(X_sonic[:, r_slice], r_req)
            t_pca = _pca_tower(X_sonic[:, t_slice], t_req)
            h_pca = _pca_tower(X_sonic[:, h_slice], h_req)

            w_r, w_t, w_h = _get_tower_weights()
            scales = np.sqrt(np.array([w_r, w_t, w_h], dtype=float))
            mat = np.concatenate(
                [scales[0] * r_pca, scales[1] * t_pca, scales[2] * h_pca],
                axis=1,
            )
            tower_pca_dims = (r_pca.shape[1], t_pca.shape[1], h_pca.shape[1])
            tower_pca_requested = (r_req, t_req, h_req)
        else:
            mat = X_sonic
            tower_fallback = True
    else:  # pragma: no cover - defensive fallback
        mat = X_sonic
    stats = {
        "variant": variant,
        "dim": int(X_sonic.shape[1]),
    }
    if variant in {"tower_l2", "tower_robust", "tower_iqr", "tower_weighted", "tower_pca"}:
        stats["tower_dims"] = tower_dims
        stats["tower_fallback"] = tower_fallback
    if variant in {"tower_weighted", "tower_pca"}:
        stats["tower_weights"] = _get_tower_weights()
    if variant == "tower_pca":
        stats["tower_pca_dims"] = tower_pca_dims
        stats["tower_pca_requested"] = tower_pca_requested
    if std_used is not None:
        stats["mean_std"] = float(np.mean(std_used))
        stats["min_std"] = float(np.min(std_used))
        stats["max_std"] = float(np.max(std_used))
    # Let downstream callers know if the matrix is already centred / scaled / whitened
    # so they can avoid re-standardising and erasing the variant effect.
    pre_scaled = variant != "raw"
    if variant in {"tower_l2", "tower_robust", "tower_iqr", "tower_weighted", "tower_pca"} and tower_fallback:
        pre_scaled = False
    stats["pre_scaled"] = pre_scaled
    return mat, stats, std_used


def compute_sonic_variant_matrix(X_sonic: np.ndarray, variant: str = "tower_pca", *, l2: bool = False) -> Tuple[np.ndarray, dict]:
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


def compute_sonic_variant_norm(X_sonic: np.ndarray, variant: str = "tower_pca") -> Tuple[np.ndarray, dict]:
    """
    Return row-normalized sonic matrix per variant (raw/centered/z/...), with lightweight stats.
    """
    return compute_sonic_variant_matrix(X_sonic, variant, l2=True)


def apply_transition_weights(
    X: np.ndarray,
    config_weights: Optional[tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Apply transition-specific tower weights to a beat3tower matrix.

    Uses config_weights or SONIC_TRANSITION_WEIGHTS (default 0.4/0.35/0.25)
    to prioritize rhythm for smooth BPM transitions between tracks.

    Args:
        X: (N, 137) beat3tower feature matrix (start or end segment)
        config_weights: Optional (rhythm, timbre, harmony) weights from config

    Returns:
        Tuple of (weighted_matrix, stats_dict)
    """
    stats: dict = {}

    # Check for beat3tower dimensions
    if X.shape[1] != 137:
        stats["transition_weights_applied"] = False
        stats["reason"] = f"non-beat3tower dims ({X.shape[1]})"
        return X, stats

    # Tower boundaries for beat3tower (137 dims)
    r_slice = slice(0, 21)      # rhythm: 21 dims
    t_slice = slice(21, 104)    # timbre: 83 dims
    h_slice = slice(104, 137)   # harmony: 33 dims

    # L2 normalize each tower independently
    def _l2_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    r_mat = _l2_rows(X[:, r_slice])
    t_mat = _l2_rows(X[:, t_slice])
    h_mat = _l2_rows(X[:, h_slice])

    # Apply transition weights (rhythm-heavy for BPM flow)
    w_r, w_t, w_h = _get_transition_weights(config_weights)
    scales = np.sqrt(np.array([w_r, w_t, w_h], dtype=float))

    weighted = np.concatenate(
        [scales[0] * r_mat, scales[1] * t_mat, scales[2] * h_mat],
        axis=1,
    )

    stats["transition_weights_applied"] = True
    stats["transition_weights"] = (w_r, w_t, w_h)
    stats["tower_dims"] = (21, 83, 33)

    return weighted, stats
