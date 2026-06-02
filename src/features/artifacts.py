from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

from src.genre.artifact_identity import dense_sidecar_mismatch_reason

logger = logging.getLogger(__name__)

Segment = Literal["full", "start", "mid", "end"]


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_path: Path
    track_ids: np.ndarray  # (N,)
    artist_keys: np.ndarray  # (N,)
    track_artists: Optional[np.ndarray]  # (N,) display-only, may be missing
    track_titles: Optional[np.ndarray]  # (N,) display-only, may be missing
    X_sonic: np.ndarray  # (N, Ds)
    X_sonic_start: Optional[np.ndarray]  # (N, Ds_seg) if present
    X_sonic_mid: Optional[np.ndarray]
    X_sonic_end: Optional[np.ndarray]
    X_genre_raw: np.ndarray  # (N, G)
    X_genre_smoothed: np.ndarray  # (N, G)
    genre_vocab: np.ndarray  # (G,)
    track_id_to_index: Dict[str, int]  # built mapping
    sonic_feature_names: Optional[np.ndarray] = None
    sonic_feature_units: Optional[np.ndarray] = None
    durations_ms: Optional[np.ndarray] = None  # (N,) track durations in milliseconds, optional
    # Optional: precomputed variant + raw matrix metadata
    X_sonic_raw: Optional[np.ndarray] = None
    sonic_variant: Optional[str] = None
    sonic_pre_scaled: bool = False
    # Optional: dense PMI-SVD genre embedding (from sidecar NPZ)
    X_genre_dense: Optional[np.ndarray] = None   # (N, dim) L2-normalized
    genre_emb: Optional[np.ndarray] = None       # (V, dim) vocabulary embedding


def _require_keys(npz: np.lib.npyio.NpzFile, required: Dict[str, str]) -> None:
    missing = [k for k in required if k not in npz]
    if missing:
        raise ValueError(f"Artifact missing required keys: {missing}")


def _ensure_first_dim(name: str, arr: Optional[np.ndarray], expected: int) -> None:
    if arr is None:
        return
    if arr.shape[0] != expected:
        raise ValueError(f"{name} first dimension {arr.shape[0]} does not match expected {expected}")


def load_artifact_bundle(path: str | Path) -> ArtifactBundle:
    """Load NPZ, validate required keys, build track_id_to_index, and return bundle.

    Cached: a single playlist generation calls this 3-7 times with the same
    artifact path, decoding ~20 MB of matrices each time. The path-keyed
    cache (maxsize=2 to handle primary + dev/test artifacts) collapses
    those into one decode per distinct path. Bundles are treated as
    read-only by all call sites.

    To force a re-read (e.g. after rebuilding artifacts on disk), call
    `load_artifact_bundle.cache_clear()`.
    """
    return _load_artifact_bundle_cached(Path(path))


load_artifact_bundle.cache_clear = lambda: _load_artifact_bundle_cached.cache_clear()  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=2)
def _load_artifact_bundle_cached(artifact_path: Path) -> ArtifactBundle:
    data = np.load(artifact_path, allow_pickle=True)

    required_keys = {
        "track_ids",
        "artist_keys",
        "X_sonic",
        "X_genre_raw",
        "X_genre_smoothed",
        "genre_vocab",
    }
    _require_keys(data, required_keys)

    track_ids = data["track_ids"]
    artist_keys = data["artist_keys"]
    track_artists = data["track_artists"] if "track_artists" in data else data.get("artist_names")
    track_titles = data["track_titles"] if "track_titles" in data else None
    X_sonic_raw = data["X_sonic"]
    X_sonic_start = data["X_sonic_start"] if "X_sonic_start" in data else None
    X_sonic_mid = data["X_sonic_mid"] if "X_sonic_mid" in data else None
    X_sonic_end = data["X_sonic_end"] if "X_sonic_end" in data else None
    X_genre_raw = data["X_genre_raw"]
    X_genre_smoothed = data["X_genre_smoothed"]
    genre_vocab = data["genre_vocab"]
    sonic_feature_names = data["sonic_feature_names"] if "sonic_feature_names" in data else None
    sonic_feature_units = data["sonic_feature_units"] if "sonic_feature_units" in data else None
    durations_ms = data["durations_ms"] if "durations_ms" in data else None

    # Prefer precomputed variant matrix when present
    sonic_variant = None
    sonic_pre_scaled = False
    if "X_sonic_variant" in data:
        try:
            sonic_variant = str(data["X_sonic_variant"].item())
        except Exception:
            sonic_variant = str(data["X_sonic_variant"])
    if sonic_variant:
        variant_key = f"X_sonic_{sonic_variant}"
        if variant_key in data:
            X_sonic = data[variant_key]
            sonic_pre_scaled = True
            logger.info("Using precomputed sonic variant '%s' from artifact key %s", sonic_variant, variant_key)
        else:
            X_sonic = X_sonic_raw
            logger.warning(
                "Artifact declared sonic_variant=%s but missing key %s; falling back to X_sonic raw.",
                sonic_variant,
                variant_key,
            )
    else:
        X_sonic = X_sonic_raw

    # --- Optional dense genre embedding sidecar ---
    X_genre_dense: Optional[np.ndarray] = None
    genre_emb: Optional[np.ndarray] = None
    _DEFAULT_SIDECAR_DIM = 64
    _sidecar_path = artifact_path.parent / f"{artifact_path.stem}_genre_emb_dim{_DEFAULT_SIDECAR_DIM}.npz"
    if _sidecar_path.exists():
        try:
            with np.load(_sidecar_path, allow_pickle=True) as _sc:
                reason = dense_sidecar_mismatch_reason(artifact=data, sidecar=_sc)
                if reason is None:
                    X_genre_dense = _sc["X_genre_dense"].astype(np.float32)
                    genre_emb = _sc["genre_emb"].astype(np.float32)
            if reason is None:
                logger.info(
                    "Loaded dense genre sidecar: %s | X_genre_dense=%s",
                    _sidecar_path.name,
                    X_genre_dense.shape,
                )
            else:
                logger.warning(
                    "Genre embedding sidecar %s %s - ignoring. "
                    "Re-run scripts/build_genre_embedding.py to rebuild.",
                    _sidecar_path.name,
                    reason,
                )
        except Exception as exc:
            logger.warning("Could not load genre embedding sidecar %s: %s", _sidecar_path.name, exc)

    N = track_ids.shape[0]
    # Validate aligned shapes
    aligned = {
        "artist_keys": artist_keys,
        "track_artists": track_artists,
        "track_titles": track_titles,
        "durations_ms": durations_ms,
        "X_sonic": X_sonic,
        "X_sonic_start": X_sonic_start,
        "X_sonic_mid": X_sonic_mid,
        "X_sonic_end": X_sonic_end,
        "X_genre_raw": X_genre_raw,
        "X_genre_smoothed": X_genre_smoothed,
        "X_sonic_raw": X_sonic_raw,
    }
    for name, arr in aligned.items():
        _ensure_first_dim(name, arr if isinstance(arr, np.ndarray) else None, N)

    # Duplicate track ids are not allowed
    track_id_to_index: Dict[str, int] = {}
    for idx, raw_tid in enumerate(track_ids):
        tid = str(raw_tid)
        if tid in track_id_to_index:
            raise ValueError(f"Duplicate track_id detected: {tid}")
        track_id_to_index[tid] = idx

    logger.info(
        "Loaded artifact %s | tracks=%d | X_sonic=%s | X_genre_raw=%s | X_genre_smoothed=%s",
        artifact_path,
        N,
        getattr(X_sonic, "shape", None),
        getattr(X_genre_raw, "shape", None),
        getattr(X_genre_smoothed, "shape", None),
    )

    return ArtifactBundle(
        artifact_path=artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_sonic_start=X_sonic_start,
        X_sonic_mid=X_sonic_mid,
        X_sonic_end=X_sonic_end,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        sonic_feature_names=sonic_feature_names,
        sonic_feature_units=sonic_feature_units,
        durations_ms=durations_ms,
        track_id_to_index=track_id_to_index,
        X_sonic_raw=X_sonic_raw,
        sonic_variant=sonic_variant,
        sonic_pre_scaled=sonic_pre_scaled,
        X_genre_dense=X_genre_dense,
        genre_emb=genre_emb,
    )


def get_sonic_matrix(bundle: ArtifactBundle, segment: Segment) -> np.ndarray:
    """Return X_sonic_* if present else fall back to X_sonic for missing segments."""
    if segment == "full":
        return bundle.X_sonic
    if segment == "start" and bundle.X_sonic_start is not None:
        return bundle.X_sonic_start
    if segment == "mid" and bundle.X_sonic_mid is not None:
        return bundle.X_sonic_mid
    if segment == "end" and bundle.X_sonic_end is not None:
        return bundle.X_sonic_end
    return bundle.X_sonic
