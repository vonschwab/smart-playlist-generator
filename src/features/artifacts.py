from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from src.genre.artifact_identity import dense_sidecar_mismatch_reason

logger = logging.getLogger(__name__)

Segment = Literal["full", "start", "mid", "end"]

# The v4.1 tower-weight alignment: transition_weights == tower_weights ==
# rhythm 0.20 / timbre 0.50 / harmony 0.30. Anything else is an intentional
# user override (see validate_tower_knobs).
DEFAULT_TOWER_TRANSITION_WEIGHTS: Tuple[float, float, float] = (0.20, 0.50, 0.30)

# Process-wide config override (config.yaml: artifacts.sonic_variant_override).
# When set, it wins over the variant the artifact declares via its
# X_sonic_variant key. Published at config-load time by src.config_loader.Config
# and the GUI worker's load_config_with_overrides.
_sonic_variant_override: Optional[str] = None


def set_sonic_variant_override(value: Optional[str]) -> None:
    """Publish ``artifacts.sonic_variant_override`` to the artifact loader.

    Called at config-load time. Changing the value invalidates the bundle
    cache so the next ``load_artifact_bundle`` re-resolves the variant. If the
    override names a variant whose ``X_sonic_{variant}`` key is missing from
    the artifact, loading raises — a configured knob that cannot act is a
    startup error, never a silent fallback.
    """
    global _sonic_variant_override
    norm = str(value).strip() if value is not None and str(value).strip() else None
    if norm != _sonic_variant_override:
        _sonic_variant_override = norm
        _load_artifact_bundle_cached.cache_clear()


def get_sonic_variant_override() -> Optional[str]:
    """Return the process-wide ``artifacts.sonic_variant_override`` value."""
    return _sonic_variant_override


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
    # Authoritative per-tower blend split (rhythm, timbre, harmony) — sums to X_sonic
    # width. The source of truth for slicing X_sonic into perceptual axes; None for
    # legacy artifacts that predate the key.
    tower_dims: Optional[Tuple[int, int, int]] = None
    # Optional: dense PMI-SVD genre embedding (from sidecar NPZ)
    X_genre_dense: Optional[np.ndarray] = None   # (N, dim) L2-normalized
    genre_emb: Optional[np.ndarray] = None       # (V, dim) vocabulary embedding
    # Optional: layered genre graph shadow matrices (from layered_shadow artifacts)
    X_genre_leaf_idf: Optional[np.ndarray] = None
    X_genre_family: Optional[np.ndarray] = None
    X_genre_bridge: Optional[np.ndarray] = None
    X_facet: Optional[np.ndarray] = None
    genre_leaf_vocab: Optional[np.ndarray] = None
    genre_family_vocab: Optional[np.ndarray] = None
    genre_bridge_vocab: Optional[np.ndarray] = None
    facet_vocab: Optional[np.ndarray] = None
    genre_graph_taxonomy_version: Optional[np.ndarray] = None
    genre_graph_sidecar_fingerprint: Optional[np.ndarray] = None


def _require_keys(npz: np.lib.npyio.NpzFile, required: Dict[str, str]) -> None:
    missing = [k for k in required if k not in npz]
    if missing:
        raise ValueError(f"Artifact missing required keys: {missing}")


def _ensure_first_dim(name: str, arr: Optional[np.ndarray], expected: int) -> None:
    if arr is None:
        return
    if arr.shape[0] != expected:
        raise ValueError(f"{name} first dimension {arr.shape[0]} does not match expected {expected}")


def load_artifact_bundle(
    path: str | Path, sonic_variant_override: Optional[str] = None
) -> ArtifactBundle:
    """Load NPZ, validate required keys, build track_id_to_index, and return bundle.

    Cached: a single playlist generation calls this 3-7 times with the same
    artifact path, decoding ~20 MB of matrices each time. The path-keyed
    cache (maxsize=2 to handle primary + dev/test artifacts) collapses
    those into one decode per distinct path. Bundles are treated as
    read-only by all call sites.

    ``sonic_variant_override``: explicit ``artifacts.sonic_variant_override``
    value; when None (the default) the process-wide value published via
    ``set_sonic_variant_override`` applies, so all call sites resolve the
    same sonic space without threading the override explicitly.

    To force a re-read (e.g. after rebuilding artifacts on disk), call
    `load_artifact_bundle.cache_clear()`.
    """
    override = (
        sonic_variant_override
        if sonic_variant_override is not None
        else _sonic_variant_override
    )
    return _load_artifact_bundle_cached(Path(path), override)


load_artifact_bundle.cache_clear = lambda: _load_artifact_bundle_cached.cache_clear()  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=2)
def _load_artifact_bundle_cached(
    artifact_path: Path, sonic_variant_override: Optional[str] = None
) -> ArtifactBundle:
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
    X_genre_leaf_idf = data["X_genre_leaf_idf"] if "X_genre_leaf_idf" in data else None
    X_genre_family = data["X_genre_family"] if "X_genre_family" in data else None
    X_genre_bridge = data["X_genre_bridge"] if "X_genre_bridge" in data else None
    X_facet = data["X_facet"] if "X_facet" in data else None
    genre_leaf_vocab = data["genre_leaf_vocab"] if "genre_leaf_vocab" in data else None
    genre_family_vocab = data["genre_family_vocab"] if "genre_family_vocab" in data else None
    genre_bridge_vocab = data["genre_bridge_vocab"] if "genre_bridge_vocab" in data else None
    facet_vocab = data["facet_vocab"] if "facet_vocab" in data else None
    genre_graph_taxonomy_version = (
        data["genre_graph_taxonomy_version"] if "genre_graph_taxonomy_version" in data else None
    )
    genre_graph_sidecar_fingerprint = (
        data["genre_graph_sidecar_fingerprint"] if "genre_graph_sidecar_fingerprint" in data else None
    )
    sonic_feature_names = data["sonic_feature_names"] if "sonic_feature_names" in data else None
    sonic_feature_units = data["sonic_feature_units"] if "sonic_feature_units" in data else None
    durations_ms = data["durations_ms"] if "durations_ms" in data else None

    tower_dims: Optional[Tuple[int, int, int]] = None
    if "tower_dims" in data:
        try:
            td = tuple(int(v) for v in np.asarray(data["tower_dims"]).ravel())
            if len(td) == 3:
                tower_dims = td  # type: ignore[assignment]
        except (TypeError, ValueError):
            tower_dims = None

    # Prefer precomputed variant matrix when present. The config override
    # (artifacts.sonic_variant_override) wins over the artifact-declared
    # X_sonic_variant; a missing override key is a hard error.
    declared_variant: Optional[str] = None
    if "X_sonic_variant" in data:
        try:
            declared_variant = str(data["X_sonic_variant"].item())
        except Exception:
            declared_variant = str(data["X_sonic_variant"])
    sonic_variant = sonic_variant_override or declared_variant
    sonic_pre_scaled = False
    if sonic_variant_override:
        variant_key = f"X_sonic_{sonic_variant_override}"
        if variant_key not in data:
            raise ValueError(
                f"artifacts.sonic_variant_override='{sonic_variant_override}' is configured "
                f"but artifact {artifact_path} has no '{variant_key}' key. A configured knob "
                "that cannot act is a startup error — remove the override or fold the "
                "variant into the artifact first."
            )
        X_sonic = data[variant_key]
        sonic_pre_scaled = True
        logger.info(
            "Sonic variant override active: artifacts.sonic_variant_override=%s wins over "
            "artifact-declared variant %r (using key %s)",
            sonic_variant_override,
            declared_variant,
            variant_key,
        )
    elif declared_variant:
        variant_key = f"X_sonic_{declared_variant}"
        if variant_key in data:
            X_sonic = data[variant_key]
            sonic_pre_scaled = True
            logger.info("Using precomputed sonic variant '%s' from artifact key %s", declared_variant, variant_key)
        else:
            X_sonic = X_sonic_raw
            logger.warning(
                "Artifact declared sonic_variant=%s but missing key %s; falling back to X_sonic raw.",
                declared_variant,
                variant_key,
            )
    else:
        X_sonic = X_sonic_raw

    # Variant-aware start/mid/end resolution: when a variant matrix is in
    # effect, prefer X_sonic_{variant}_{start|mid|end}; fall back to the
    # legacy segment keys (with an INFO log) when the variant-specific
    # segment keys are absent. Legacy artifacts (no variant) are untouched.
    if sonic_pre_scaled and sonic_variant:
        missing_seg_keys: list[str] = []
        seg_legacy = {"start": X_sonic_start, "mid": X_sonic_mid, "end": X_sonic_end}
        seg_resolved: Dict[str, Optional[np.ndarray]] = {}
        for seg, legacy_arr in seg_legacy.items():
            seg_key = f"X_sonic_{sonic_variant}_{seg}"
            if seg_key in data:
                seg_resolved[seg] = data[seg_key]
            else:
                seg_resolved[seg] = legacy_arr
                missing_seg_keys.append(seg_key)
        X_sonic_start = seg_resolved["start"]
        X_sonic_mid = seg_resolved["mid"]
        X_sonic_end = seg_resolved["end"]
        if missing_seg_keys and any(
            arr is not None for arr in (X_sonic_start, X_sonic_mid, X_sonic_end)
        ):
            logger.info(
                "Sonic variant '%s': segment keys %s absent from artifact; using legacy "
                "X_sonic_start/mid/end segment matrices.",
                sonic_variant,
                missing_seg_keys,
            )

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
        "X_genre_leaf_idf": X_genre_leaf_idf,
        "X_genre_family": X_genre_family,
        "X_genre_bridge": X_genre_bridge,
        "X_facet": X_facet,
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
        tower_dims=tower_dims,
        X_genre_dense=X_genre_dense,
        genre_emb=genre_emb,
        X_genre_leaf_idf=X_genre_leaf_idf,
        X_genre_family=X_genre_family,
        X_genre_bridge=X_genre_bridge,
        X_facet=X_facet,
        genre_leaf_vocab=genre_leaf_vocab,
        genre_family_vocab=genre_family_vocab,
        genre_bridge_vocab=genre_bridge_vocab,
        facet_vocab=facet_vocab,
        genre_graph_taxonomy_version=genre_graph_taxonomy_version,
        genre_graph_sidecar_fingerprint=genre_graph_sidecar_fingerprint,
    )


def _is_default_transition_weights(weights: Tuple[float, float, float]) -> bool:
    """True when ``weights`` normalize to the 0.20/0.50/0.30 default split."""
    try:
        vals = tuple(float(v) for v in weights)
    except (TypeError, ValueError):
        return False
    if len(vals) != 3 or any(v < 0 for v in vals):
        return False
    total = sum(vals)
    if total <= 0:
        return False
    return all(
        abs(v / total - d) <= 1e-6
        for v, d in zip(vals, DEFAULT_TOWER_TRANSITION_WEIGHTS)
    )


def _variant_lacks_tower_split(bundle: ArtifactBundle) -> bool:
    """True when an active sonic variant has no rhythm/timbre/harmony split.

    Only variant-active bundles qualify: legacy artifacts with no declared
    variant keep their current behavior. A variant has a tower split when the
    artifact's ``tower_dims`` matches the blend width, or the matrix is the
    raw 137-dim beat3tower blend (hardcoded slices in apply_transition_weights).
    """
    if not bundle.sonic_pre_scaled or not bundle.sonic_variant:
        return False
    X = bundle.X_sonic
    if X is None:
        return False
    blend_dim = int(X.shape[1])
    if blend_dim == 137:
        return False
    td = bundle.tower_dims
    if td is not None and len(td) == 3 and sum(int(v) for v in td) == blend_dim:
        return False
    return True


def validate_tower_knobs(
    bundle: ArtifactBundle,
    transition_weights: Optional[Tuple[float, float, float]],
) -> None:
    """Guard tower-style knobs against no-tower sonic variants (e.g. mert).

    A no-tower variant is a single space: ``transition_weights`` /
    ``tower_weights`` cannot act on it (beam and reporter both score plain
    cosine, so the v4.1 beam/reporter alignment holds by construction).

    - default (0.20/0.50/0.30) or unset weights → INFO log that the knobs are
      inert for this variant;
    - non-default weights → raise (a configured knob that cannot act is a
      startup error, not a silent no-op).
    """
    if not _variant_lacks_tower_split(bundle):
        return
    variant = str(bundle.sonic_variant)
    blend_dim = int(bundle.X_sonic.shape[1]) if bundle.X_sonic is not None else -1
    if transition_weights is None or _is_default_transition_weights(transition_weights):
        logger.info(
            "Sonic variant '%s' has no rhythm/timbre/harmony tower split "
            "(blend dim=%d, tower_dims=%r): tower_weights/transition_weights are "
            "inert for this variant; transitions score plain cosine in the variant space.",
            variant,
            blend_dim,
            bundle.tower_dims,
        )
        return
    raise ValueError(
        f"transition_weights={tuple(float(v) for v in transition_weights)} are configured "
        f"(non-default) but the active sonic variant '{variant}' has no tower split "
        f"(X_sonic dim {blend_dim}, tower_dims={bundle.tower_dims!r}); tower-style weights "
        "cannot act on a single-space variant. Remove the non-default transition_weights "
        f"(default {DEFAULT_TOWER_TRANSITION_WEIGHTS}) or switch back to a tower variant."
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
