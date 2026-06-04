"""Bundle restriction phase — slice an ArtifactBundle by allowed/excluded track ids.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split). Pure
numpy index math; no orchestration logic, no audit emission. Always
exempts the primary seed and any anchor seeds from exclusion so they
can still serve as piers.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from src.features.artifacts import ArtifactBundle

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_TRACK_ID_LIMIT = 25_000


def _allowed_track_id_limit() -> int:
    raw = os.environ.get("PLAYLIST_DS_ALLOWED_TRACK_ID_LIMIT")
    if raw is None:
        return DEFAULT_ALLOWED_TRACK_ID_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid PLAYLIST_DS_ALLOWED_TRACK_ID_LIMIT=%r; using default %d",
            raw,
            DEFAULT_ALLOWED_TRACK_ID_LIMIT,
        )
        return DEFAULT_ALLOWED_TRACK_ID_LIMIT


def restrict_bundle(
    bundle: ArtifactBundle,
    seed_track_id: str,
    seed_idx: int,
    *,
    anchor_seed_ids: List[str],
    allowed_track_ids: Optional[List[str]],
    excluded_track_ids: Optional[set],
    allowed_track_ids_set: Optional[set],
) -> tuple[ArtifactBundle, int, Optional[set]]:
    """Restrict ``bundle`` to the requested track-id subset.

    Two branches:
      * If ``allowed_track_ids`` is non-empty: clamp to those plus the
        seed/anchor exempt set; if ``excluded_track_ids`` is also given,
        apply both (intersection).
      * Else if ``excluded_track_ids`` is non-empty: mask out those tracks
        from the full bundle, exempting seed + anchors.
      * Otherwise: return the bundle unchanged.

    Returns ``(possibly_restricted_bundle, new_seed_idx,
    new_allowed_track_ids_set)``. The returned ``allowed_track_ids_set`` is
    augmented with the exempt-pier set in the first branch so downstream
    invariant checks pass.

    Raises ``ValueError`` if the restriction empties the bundle or if the
    allowed set exceeds the configured safety limit.
    """
    if allowed_track_ids:
        allowed_indices: List[int] = []
        allowed_track_ids_set = {str(tid) for tid in allowed_track_ids}
        for tid in allowed_track_ids_set:
            idx = bundle.track_id_to_index.get(tid)
            if idx is not None:
                allowed_indices.append(idx)

        # Always include seed + anchor seeds (piers), even if not explicitly in allowed list.
        exempt_ids = {str(seed_track_id)}
        if anchor_seed_ids:
            exempt_ids.update(str(sid) for sid in anchor_seed_ids)
        for tid in exempt_ids:
            idx = bundle.track_id_to_index.get(tid)
            if idx is not None:
                allowed_indices.append(idx)

        # Add exempt_ids to allowed_track_ids_set for final enforcement check
        allowed_track_ids_set.update(exempt_ids)

        applied_excluded = 0
        if excluded_track_ids:
            excluded_set = {str(t) for t in excluded_track_ids}
            kept: List[int] = []
            for idx in allowed_indices:
                tid = str(bundle.track_ids[int(idx)])
                if tid in exempt_ids:
                    kept.append(idx)
                    continue
                if tid in excluded_set:
                    applied_excluded += 1
                    continue
                kept.append(idx)
            allowed_indices = kept

        allowed_indices = sorted(set(int(i) for i in allowed_indices))
        N_allowed = len(allowed_indices)
        if N_allowed == 0:
            raise ValueError("No allowed track_ids were found in artifact bundle.")
        limit = _allowed_track_id_limit()
        if N_allowed > limit:
            raise ValueError(
                f"Allowed track_id set too large ({N_allowed} > {limit}); refusing DS run."
            )

        bundle = _slice_bundle(bundle, allowed_indices)
        new_seed_idx = bundle.track_id_to_index[str(seed_track_id)]
        logger.info(
            "DS pipeline bundle restricted: N=%d X_sonic=%s X_genre_smoothed=%s X_genre_dense=%s",
            N_allowed,
            getattr(bundle.X_sonic, "shape", None),
            getattr(bundle.X_genre_smoothed, "shape", None),
            getattr(bundle.X_genre_dense, "shape", None),
        )
        if excluded_track_ids and (os.environ.get("PLAYLIST_DIAG_RECENCY") or os.environ.get("PLAYLIST_DIAG_POOL")):
            logger.info(
                "DS bundle clamp+exclude: allowed_ids=%d excluded_ids=%d applied_excluded=%d final_N=%d",
                len(allowed_track_ids_set),
                len(excluded_track_ids),
                applied_excluded,
                N_allowed,
            )
        return bundle, new_seed_idx, allowed_track_ids_set

    if excluded_track_ids:
        total_tracks = int(bundle.track_ids.shape[0])
        mask_keep: List[bool] = []
        applied_excluded = 0
        exempt_ids = {str(seed_track_id)}
        if anchor_seed_ids:
            exempt_ids.update(str(sid) for sid in anchor_seed_ids)
        for tid in bundle.track_ids:
            sid = str(tid)
            if sid in exempt_ids:
                mask_keep.append(True)
                continue
            if sid in excluded_track_ids:
                applied_excluded += 1
                mask_keep.append(False)
                continue
            mask_keep.append(True)
        mask_arr = np.array(mask_keep, dtype=bool)
        if not np.any(mask_arr):
            raise ValueError("Excluded set removed all tracks; cannot run DS.")
        allowed_indices = np.nonzero(mask_arr)[0].tolist()
        if seed_idx not in allowed_indices:
            allowed_indices = sorted(set(allowed_indices + [seed_idx]))

        # Note: matches the legacy behavior that did not pass durations_ms here.
        # Preserved verbatim during the Tier-1.5 extraction.
        bundle = _slice_bundle(bundle, allowed_indices, include_durations=False)
        new_seed_idx = bundle.track_id_to_index[str(seed_track_id)]
        if os.environ.get("PLAYLIST_DIAG_RECENCY") or os.environ.get("PLAYLIST_DIAG_POOL"):
            logger.info(
                "DS candidate pool after exclusions: total=%d requested_excluded=%d applied_excluded=%d final_pool=%d",
                total_tracks,
                len(excluded_track_ids),
                applied_excluded,
                len(allowed_indices),
            )
        return bundle, new_seed_idx, allowed_track_ids_set

    # No restriction requested.
    return bundle, seed_idx, allowed_track_ids_set


def _slice_bundle(
    bundle: ArtifactBundle,
    indices: List[int],
    *,
    include_durations: bool = True,
) -> ArtifactBundle:
    """Slice every aligned matrix in ``bundle`` to ``indices``."""
    def _opt(arr):
        return None if arr is None else arr[indices]

    kwargs = dict(
        artifact_path=bundle.artifact_path,
        track_ids=bundle.track_ids[indices],
        artist_keys=bundle.artist_keys[indices],
        track_artists=_opt(bundle.track_artists),
        track_titles=_opt(bundle.track_titles),
        X_sonic=bundle.X_sonic[indices],
        X_sonic_start=_opt(bundle.X_sonic_start),
        X_sonic_mid=_opt(bundle.X_sonic_mid),
        X_sonic_end=_opt(bundle.X_sonic_end),
        X_genre_raw=bundle.X_genre_raw[indices],
        X_genre_smoothed=bundle.X_genre_smoothed[indices],
        X_genre_dense=_opt(bundle.X_genre_dense),
        genre_vocab=bundle.genre_vocab,
        track_id_to_index={str(tid): i for i, tid in enumerate(bundle.track_ids[indices])},
        # Scalar/metadata fields — not row-indexed; must be forwarded explicitly
        # because ArtifactBundle is a frozen dataclass (defaults don't carry over).
        sonic_variant=bundle.sonic_variant,
        sonic_pre_scaled=bundle.sonic_pre_scaled,
        tower_dims=bundle.tower_dims,
        genre_emb=bundle.genre_emb,
    )
    if include_durations:
        kwargs["durations_ms"] = _opt(bundle.durations_ms)
    return ArtifactBundle(**kwargs)
