"""
Playlist reporting and diagnostics module.

This module provides functions for generating detailed reports about playlists,
including edge score computation and comprehensive statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import os
import numpy as np

from src.playlist.utils import sanitize_for_logging
from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_variant import resolve_sonic_variant, compute_sonic_variant_norm
from src.similarity.hybrid import transition_similarity_end_to_start

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for playlist report generation."""
    verbose: bool = False
    include_edge_scores: bool = True
    include_percentiles: bool = True
    verbose_edges: bool = False


def format_duration(duration_ms: int) -> str:
    """
    Convert milliseconds to human-readable duration format.

    Args:
        duration_ms: Duration in milliseconds

    Returns:
        Formatted duration string (e.g., "45.2 minutes")
    """
    total_minutes = duration_ms / 1000 / 60
    return f"{total_minutes:.1f} minutes"


def log_recency_edge_diff(
    *,
    before_tracks: List[Dict[str, Any]],
    after_tracks: List[Dict[str, Any]]
) -> None:
    """
    Diagnostic: log adjacency changes introduced by recency filtering (no behavior change).

    Args:
        before_tracks: Track list before recency filtering
        after_tracks: Track list after recency filtering
    """
    diag_enabled = bool(os.environ.get("PLAYLIST_DIAG_RECENCY"))
    if not diag_enabled and not logger.isEnabledFor(logging.DEBUG):
        return
    before_ids = [str(t.get("rating_key")) for t in before_tracks]
    after_ids = [str(t.get("rating_key")) for t in after_tracks]
    removed = [tid for tid in before_ids if tid not in after_ids]
    before_edges = list(zip(before_ids, before_ids[1:]))
    after_edges = list(zip(after_ids, after_ids[1:]))
    new_edges = [e for e in after_edges if e not in before_edges]
    logger.info(
        "Recency adjacency diag: before=%d after=%d removed=%d new_edges=%d",
        len(before_ids),
        len(after_ids),
        len(removed),
        len(new_edges),
    )
    if removed:
        logger.info("Recency removed ids: %s", removed[:10])
    if new_edges:
        logger.info("Recency new edges (sample): %s", new_edges[:10])


def compute_edge_scores_from_artifact(
    *,
    tracks: List[Dict[str, Any]],
    artifact_path: Optional[str],
    config_sonic_variant: Optional[str],
    transition_floor: Optional[float] = None,
    transition_gamma: Optional[float] = None,
    embedding_random_seed: Optional[int] = None,
    center_transitions: bool = False,
    verbose: bool = False,
    sonic_variant: Optional[str] = None,
    last_ds_report: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-edge scores (T/S/G) for the final playlist order using artifact matrices.

    Args:
        tracks: List of track dictionaries
        artifact_path: Path to artifact bundle
        config_sonic_variant: Sonic variant from configuration
        transition_floor: Minimum transition score threshold
        transition_gamma: Blend weight for transition scoring
        embedding_random_seed: Random seed for embedding
        center_transitions: Whether to center transition embeddings
        verbose: Enable verbose logging
        sonic_variant: Explicit sonic variant override
        last_ds_report: Last DS report dictionary (for storing baseline)

    Returns:
        List of edge score dictionaries
    """
    if not artifact_path or not tracks or len(tracks) < 2:
        return []
    try:
        bundle = load_artifact_bundle(artifact_path)
    except Exception as exc:
        logger.error("Failed to load artifact for edge logging (%s)", exc, exc_info=True)
        return []

    # Map rating_key -> index
    idxs = []
    missing = 0
    missing_ids: List[str] = []
    for t in tracks:
        tid = str(t.get("rating_key"))
        idx = bundle.track_id_to_index.get(tid)
        if idx is None:
            missing += 1
            missing_ids.append(tid)
            idxs.append(None)
        else:
            idxs.append(idx)
    if missing:
        logger.error("Missing %d track ids in artifact; edge scores may be incomplete.", missing)
        if verbose:
            sample = missing_ids[:5]
            logger.info(
                "Missing track id samples: %s",
                [(tid, tid in bundle.track_id_to_index) for tid in sample],
            )

    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    X_start = getattr(bundle, "X_sonic_start", None)
    X_end = getattr(bundle, "X_sonic_end", None)
    X_start_orig = X_start
    X_end_orig = X_end
    rescale_transitions = False
    if center_transitions and X_start is not None and X_end is not None:
        mu_end = X_end.mean(axis=0, keepdims=True)
        mu_start = X_start.mean(axis=0, keepdims=True)
        X_end = X_end - mu_end
        X_start = X_start - mu_start
        rescale_transitions = True
    emb_norm = None
    try:
        # Rebuild the hybrid embedding used by DS so transition scores match constructor logic.
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
        emb = emb_model.embedding
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    except Exception as exc:
        logger.warning("Edge logging: failed to build hybrid embedding (%s)", exc)

    def _norm(mat):
        if mat is None:
            return None
        denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / denom

    sonic_variant = resolve_sonic_variant(explicit_variant=sonic_variant, config_variant=config_sonic_variant)
    sonic_norm = None
    if X_sonic is not None:
        sonic_norm, sonic_stats = compute_sonic_variant_norm(X_sonic, sonic_variant)
        if sonic_variant != "raw":
            logger.info(
                "SONIC_SIM_VARIANT=%s applied for edge logging (dim=%d mean_norm=%.6f)",
                sonic_variant,
                sonic_stats.get("dim"),
                sonic_stats.get("mean_norm"),
            )
    genre_norm = _norm(X_genre)
    gamma = 1.0 if transition_gamma is None else float(transition_gamma)

    edge_scores: List[Dict[str, Any]] = []
    missing_edge_pairs: List[Tuple[str, str]] = []

    # Determine valid bounds for indices
    max_idx = 0
    if X_sonic is not None:
        max_idx = X_sonic.shape[0] - 1
    elif X_genre is not None:
        max_idx = X_genre.shape[0] - 1

    for i in range(1, len(tracks)):
        prev_idx = idxs[i - 1]
        cur_idx = idxs[i]
        if prev_idx is None or cur_idx is None:
            if verbose:
                missing_edge_pairs.append(
                    (
                        str(tracks[i - 1].get("rating_key")),
                        str(tracks[i].get("rating_key")),
                    )
                )
            continue
        # Check bounds
        if prev_idx < 0 or prev_idx > max_idx or cur_idx < 0 or cur_idx > max_idx:
            if verbose:
                missing_edge_pairs.append(
                    (
                        str(tracks[i - 1].get("rating_key")),
                        str(tracks[i].get("rating_key")),
                    )
                )
            logger.warning(
                "Track indices out of bounds: prev_idx=%d cur_idx=%d (valid range 0-%d)",
                prev_idx, cur_idx, max_idx
            )
            continue
        t_raw_uncentered = float("nan")
        if X_end_orig is not None and X_start_orig is not None:
            try:
                t_raw_uncentered = float(
                    transition_similarity_end_to_start(X_end_orig, X_start_orig, prev_idx, np.array([cur_idx]))[0]
                )
            except Exception:
                t_raw_uncentered = float("nan")
        t_centered_cos = float("nan")
        h_val = float("nan")
        if emb_norm is not None:
            h_val = float(emb_norm[prev_idx] @ emb_norm[cur_idx])
        t_used = t_raw_uncentered
        if X_end is not None and X_start is not None:
            try:
                t_centered_cos = float(
                    transition_similarity_end_to_start(X_end, X_start, prev_idx, np.array([cur_idx]))[0]
                )
            except Exception:
                t_centered_cos = float("nan")
        if rescale_transitions and np.isfinite(t_centered_cos):
            t_used = float(np.clip((t_centered_cos + 1.0) / 2.0, 0.0, 1.0))
        if (not rescale_transitions) and np.isfinite(t_raw_uncentered) and np.isfinite(h_val):
            t_used = gamma * t_raw_uncentered + (1 - gamma) * h_val
        elif (not rescale_transitions) and np.isfinite(h_val):
            t_used = h_val
        elif (not rescale_transitions) and np.isfinite(t_raw_uncentered):
            t_used = t_raw_uncentered
        elif rescale_transitions and np.isfinite(h_val) and not np.isfinite(t_used):
            t_used = h_val
        if rescale_transitions and not np.isfinite(t_used) and np.isfinite(t_raw_uncentered):
            t_used = float(np.clip((t_raw_uncentered + 1.0) / 2.0, 0.0, 1.0))
        if (t_used != t_used) and sonic_norm is not None:
            # Fallback to sonic cosine if transition segments unavailable
            t_used = float(sonic_norm[prev_idx] @ sonic_norm[cur_idx])
        s_val = float("nan")
        if sonic_norm is not None:
            s_val = float(sonic_norm[prev_idx] @ sonic_norm[cur_idx])
        g_val = float("nan")
        if genre_norm is not None:
            g_val = float(genre_norm[prev_idx] @ genre_norm[cur_idx])
        edge_scores.append(
            {
                "prev_id": tracks[i - 1].get("rating_key"),
                "cur_id": tracks[i].get("rating_key"),
                "prev_idx": prev_idx,
                "cur_idx": cur_idx,
                "T": t_used,
                "T_used": t_used,
                "T_raw": t_raw_uncentered,
                "T_raw_uncentered": t_raw_uncentered,
                "T_centered_cos": t_centered_cos,
                "H": h_val,
                "S": s_val,
                "G": g_val,
                "floor": transition_floor,
                "gamma": transition_gamma,
            }
        )

    if verbose:
        # Sample from the full artifact candidate space to understand baseline similarity levels.
        cand_indices = list(range(int(getattr(X_sonic, "shape", [0])[0] or 0)))
        if not cand_indices:
            cand_indices = sorted({idx for idx in idxs if idx is not None})
        if len(cand_indices) >= 2:
            rng = np.random.default_rng(0)
            sample_size = min(2000, max(1, len(cand_indices) * 2))
            sample_pairs = rng.choice(cand_indices, size=(sample_size, 2), replace=True)

            def _pct(arr: Optional[np.ndarray]):
                if arr is None or arr.size == 0:
                    return None
                return {
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                    "p99": float(np.percentile(arr, 99)),
                }

            base_s = None
            base_g = None
            base_t_raw = None
            base_t_center = None
            base_t_used = None
            if sonic_norm is not None:
                base_s = np.array([float(sonic_norm[a] @ sonic_norm[b]) for a, b in sample_pairs], dtype=float)
            if genre_norm is not None:
                base_g = np.array([float(genre_norm[a] @ genre_norm[b]) for a, b in sample_pairs], dtype=float)
            if X_end is not None and X_start is not None:
                vals_raw = []
                vals_center = []
                vals_used = []
                for a, b in sample_pairs:
                    seg_val = float(transition_similarity_end_to_start(X_end, X_start, a, np.array([b]))[0])
                    vals_center.append(seg_val)
                    if rescale_transitions:
                        vals_used.append(float(np.clip((seg_val + 1.0) / 2.0, 0.0, 1.0)))
                    vals_raw.append(
                        float(
                            transition_similarity_end_to_start(
                                X_end_orig if X_end_orig is not None else X_end,
                                X_start_orig if X_start_orig is not None else X_start,
                                a,
                                np.array([b]),
                            )[0]
                        )
                    )
                    if emb_norm is not None and not rescale_transitions:
                        hyb = float(emb_norm[a] @ emb_norm[b])
                        vals_used.append(gamma * seg_val + (1 - gamma) * hyb)
                base_t_raw = np.array(vals_raw, dtype=float)
                base_t_center = np.array(vals_center, dtype=float)
                if vals_used:
                    base_t_used = np.array(vals_used, dtype=float)
            elif emb_norm is not None:
                vals_used = [float(emb_norm[a] @ emb_norm[b]) for a, b in sample_pairs]
                base_t_used = np.array(vals_used, dtype=float)
            baseline = {
                "S": _pct(base_s),
                "G": _pct(base_g),
                "T": _pct(base_t_used),
                "T_raw": _pct(base_t_raw),
                "T_centered_cos": _pct(base_t_center),
                "T_centered_rescaled": _pct(base_t_used) if rescale_transitions else None,
            }
            if last_ds_report is not None:
                last_ds_report["baseline"] = baseline

        idx_counts = {}
        for idx in idxs:
            if idx is None:
                continue
            idx_counts[idx] = idx_counts.get(idx, 0) + 1
        if idx_counts:
            unique_indices_used = len(idx_counts)
            max_repeat = max(idx_counts.values())
            repeated = sum(1 for v in idx_counts.values() if v > 1)
            logger.info(
                "Edge mapping summary: unique_indices=%d repeated=%d max_repeat=%d",
                unique_indices_used,
                repeated,
                max_repeat,
            )
            if sonic_norm is not None:
                per_dim_std = np.std(sonic_norm, axis=0)
                mean_std = float(per_dim_std.mean())
                if mean_std < 1e-3:
                    logger.warning("X_sonic appears nearly constant (mean per-dim std=%.6f)", mean_std)
        if missing_edge_pairs and verbose:
            sample_pairs = missing_edge_pairs[:5]
            logger.info(
                "Edges skipped due to missing indices: %d sample ids=%s",
                len(missing_edge_pairs),
                [
                    (
                        p,
                        c,
                        p in bundle.track_id_to_index,
                        c in bundle.track_id_to_index,
                    )
                    for p, c in sample_pairs
                ],
            )

    return edge_scores


def print_playlist_report(
    *,
    tracks: List[Dict[str, Any]],
    artist_name: str = None,
    dynamic: bool = False,
    verbose_edges: bool = False,
    last_ds_report: Optional[Dict[str, Any]] = None,
    last_scope: Optional[str] = None,
    last_ds_mode: Optional[str] = None,
):
    """
    Print detailed track report showing how each track was selected.

    Args:
        tracks: Final playlist tracks
        artist_name: Name of seed artist (if applicable)
        dynamic: Whether dynamic mode was used
        verbose_edges: Whether to print per-edge scores when available (DS)
        last_ds_report: Last DS report dictionary
        last_scope: Last scope value
        last_ds_mode: Last DS mode value
    """
    logger.info("=" * 80)
    # Pipeline context summary if available
    pipeline_ctx = []
    pipeline_ctx.append("pipeline=ds")
    if last_scope is not None:
        pipeline_ctx.append(f"scope={last_scope}")
    if last_ds_mode is not None:
        pipeline_ctx.append(f"mode={last_ds_mode}")
    if artist_name:
        pipeline_ctx.append(f'seed="{sanitize_for_logging(artist_name)}"')
    ctx_str = " | ".join(pipeline_ctx)
    logger.info(f"PLAYLIST TRACKLIST  ({ctx_str})")
    logger.info("=" * 80)

    # Group tracks by source for statistics
    source_counts = {'sonic': 0, 'genre': 0, 'unknown': 0}
    edge_scores_for_logging: List[Dict[str, Any]] = []
    edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if last_ds_report:
        report = last_ds_report or {}
        playlist_stats = report.get("playlist_stats") or {}
        playlist_stats_playlist = playlist_stats.get("playlist") or {}
        edge_scores_for_logging = (
            playlist_stats_playlist.get("edge_scores")
            or playlist_stats.get("edge_scores")
            or report.get("edge_scores")
            or []
        )
        last_ds_report["edge_scores"] = edge_scores_for_logging
        edge_map = {(str(e.get("prev_id")), str(e.get("cur_id"))): e for e in edge_scores_for_logging}
        if verbose_edges:
            missing_hydration = [t for t in tracks if not t.get("artist") or not t.get("title")]
            if missing_hydration:
                logger.info(
                    "Hydration gaps: %d tracks missing artist/title; samples=%s",
                    len(missing_hydration),
                    [t.get("rating_key") for t in missing_hydration[:5]],
                )
            if edge_scores_for_logging and len(edge_scores_for_logging) != max(0, len(tracks) - 1):
                logger.error(
                    "Edge score count mismatch (edges=%d, expected=%d); skipping per-edge printing.",
                    len(edge_scores_for_logging),
                    max(0, len(tracks) - 1),
                )
                edge_map = {}

    for i, track in enumerate(tracks, 1):
        raw_artist = str(track.get('artist', 'Unknown') or 'Unknown')
        raw_title = str(track.get('title', 'Unknown') or 'Unknown')
        artist = sanitize_for_logging(raw_artist)
        title = sanitize_for_logging(raw_title)
        source = track.get('source', 'unknown')
        track_id = track.get("rating_key") or track.get("id") or track.get("track_id")
        file_path = track.get("file") or track.get("path") or track.get("file_path")

        sanitized = (artist != raw_artist) or (title != raw_title)
        if sanitized and logger.isEnabledFor(logging.DEBUG):
            # Keep raw values ASCII-safe for console logs.
            raw_artist_dbg = raw_artist.encode("unicode_escape", errors="backslashreplace").decode("ascii")
            raw_title_dbg = raw_title.encode("unicode_escape", errors="backslashreplace").decode("ascii")
            logger.debug(
                "Track metadata sanitized: idx=%d id=%s file=%s raw_artist=%s raw_title=%s safe_artist=%s safe_title=%s",
                i,
                track_id,
                file_path,
                raw_artist_dbg,
                raw_title_dbg,
                artist,
                title,
            )

        # Count by source
        source_counts[source] = source_counts.get(source, 0) + 1

        # Format: Track 01: Artist - Title
        marker = f" [id={track_id}]" if sanitized and track_id is not None else ""
        logger.info(f"Track {i:02d}: {artist} - {title}{marker}")
        # Verbose per-edge scores (DS only)
        if verbose_edges and last_ds_report and i > 1:
            prev_id = str(tracks[i - 2].get("rating_key"))
            cur_id = str(tracks[i - 1].get("rating_key"))
            edge = edge_map.get((prev_id, cur_id))
            if edge:
                import math

                t_val = edge.get("T", float("nan"))
                t_raw = edge.get("T_raw", float("nan"))
                t_center_cos = edge.get("T_centered_cos", float("nan"))
                h_val = edge.get("H", float("nan"))
                s_val = edge.get("S", float("nan"))
                g_val = edge.get("G", float("nan"))
                raw_part = ""
                if isinstance(t_raw, (int, float)) and not math.isnan(t_raw):
                    raw_part = f" (raw={t_raw:.3f})"
                center_part = ""
                if isinstance(t_center_cos, (int, float)) and not math.isnan(t_center_cos):
                    center_part = f" center_cos={t_center_cos:.3f}"
                h_part = f"  H={h_val:.3f}" if isinstance(h_val, (int, float)) and not math.isnan(h_val) else "  H=n/a"
                logger.info(
                    "    edge %02d->%02d: T=%.3f%s%s%s  S=%.3f  G=%.3f",
                    i - 1,
                    i,
                    t_val if t_val == t_val else float("nan"),
                    raw_part,
                    center_part,
                    h_part,
                    s_val if s_val == s_val else float("nan"),
                    g_val if g_val == g_val else float("nan"),
                )
            else:
                logger.error("Missing edge score for %s->%s", prev_id, cur_id)

    # Print summary statistics
    logger.info("=" * 80)
    logger.info("PLAYLIST STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total tracks: {len(tracks)}")
    logger.info(f"Unique artists: {len(set(t.get('artist') for t in tracks))}")

    if last_ds_report:
        import math

        report = last_ds_report or {}
        metrics = report.get("metrics") or {}
        playlist_stats = report.get("playlist_stats") or {}
        playlist_stats_playlist = playlist_stats.get("playlist") or {}
        edge_scores = (
            playlist_stats_playlist.get("edge_scores")
            or playlist_stats.get("edge_scores")
            or report.get("edge_scores")
            or []
        )
        report["edge_scores"] = edge_scores
        baseline = report.get("baseline") or {}
        transition_floor_val = (
            metrics.get("transition_floor")
            or playlist_stats_playlist.get("transition_floor")
            or playlist_stats.get("transition_floor")
            or report.get("transition_floor")
        )
        gamma_val = report.get("transition_gamma") or playlist_stats_playlist.get("transition_gamma")
        transition_centered = bool(
            playlist_stats_playlist.get("transition_centered")
            or playlist_stats.get("transition_centered")
            or report.get("transition_centered")
        )
        tracks_by_id = {str(t.get("rating_key")): t for t in tracks}
        if verbose_edges:
            def _nan_count(vals):
                return sum(1 for v in vals if isinstance(v, float) and math.isnan(v))

            logger.info(
                "Edge score arrays: tracks=%d edges=%d missing_T=%d missing_S=%d missing_G=%d",
                len(tracks),
                len(edge_scores),
                _nan_count([e.get("T", float("nan")) for e in edge_scores]),
                _nan_count([e.get("S", float("nan")) for e in edge_scores]),
                _nan_count([e.get("G", float("nan")) for e in edge_scores]),
            )
            if len(edge_scores) != max(0, len(tracks) - 1):
                logger.error(
                    "Edge score count mismatch (edges=%d, expected=%d); skipping edge logging for missing scores.",
                    len(edge_scores),
                    max(0, len(tracks) - 1),
                )
            if edge_scores and all((isinstance(e.get("T", float("nan")), float) and math.isnan(e.get("T"))) for e in edge_scores):
                logger.error("Edge scores contain only NaN values; skipping edge logging.")
                edge_scores = []
            # Index sanity for first 3 and bottom 3
            if edge_scores:
                samples = edge_scores[:3] + sorted(edge_scores, key=lambda e: e.get("T", float("nan")))[:3]
                for e in samples:
                    logger.info(
                        "Edge idx map: prev_idx=%s cur_idx=%s prev=%s - %s -> %s - %s T=%.3f",
                        e.get("prev_idx"),
                        e.get("cur_idx"),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("artist", "")),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("title", "")),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("artist", "")),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("title", "")),
                        e.get("T", float("nan")),
                    )
            if baseline:
                def _fmt_pct(stats: Optional[dict]) -> str:
                    if not stats:
                        return "n/a"
                    return f"p50={stats.get('p50')} p90={stats.get('p90')} p99={stats.get('p99')}"

                logger.info("Baseline (artifact) percentiles: T=%s | S=%s | G=%s",
                            _fmt_pct(baseline.get("T_centered_rescaled") if transition_centered else baseline.get("T")),
                            _fmt_pct(baseline.get("S")),
                            _fmt_pct(baseline.get("G")))
                if transition_centered:
                    logger.info(
                        "Baseline centered: T_centered_cos=%s | T_centered_rescaled=%s",
                        _fmt_pct(baseline.get("T_centered_cos")),
                        _fmt_pct(baseline.get("T_centered_rescaled")),
                    )
            if edge_scores:
                idxs = [e.get("prev_idx") for e in edge_scores] + [e.get("cur_idx") for e in edge_scores]
                idxs = [i for i in idxs if i is not None]
                if idxs:
                    idx_counts = {}
                    for i in idxs:
                        idx_counts[i] = idx_counts.get(i, 0) + 1
                    unique_indices_used = len(idx_counts)
                    max_repeat = max(idx_counts.values())
                    repeated = sum(1 for v in idx_counts.values() if v > 1)
                    logger.info(
                        "Chain index-degree summary: unique_nodes=%d nodes_with_degree>1=%d max_degree=%d (expected for contiguous chain)",
                        unique_indices_used,
                        repeated,
                        max_repeat,
                    )
        if not edge_scores:
            logger.info("Edge score summaries: none (edges=0)")
            return
        logger.info("Edge score summaries:")

        def _summ(arr):
            import math

            if not arr:
                return None
            arr_np = np.array([v for v in arr if isinstance(v, (int, float)) and not math.isnan(v)], dtype=float)
            if arr_np.size == 0:
                return None
            return {
                "mean": float(np.nanmean(arr_np)),
                "p10": float(np.nanpercentile(arr_np, 10)),
                "p50": float(np.nanpercentile(arr_np, 50)),
                "p90": float(np.nanpercentile(arr_np, 90)),
                "p99": float(np.nanpercentile(arr_np, 99)),
                "min": float(np.nanmin(arr_np)),
            }

        def _fmt(val: Optional[float]) -> str:
            if val is None:
                return "n/a"
            try:
                if math.isnan(val):
                    return "n/a"
            except Exception:
                return "n/a"
            return f"{val:.3f}"

        t_list = [e.get("T", float("nan")) for e in edge_scores]
        t_center_cos_list = [e.get("T_centered_cos", float("nan")) for e in edge_scores]
        s_list = [e.get("S", float("nan")) for e in edge_scores]
        g_list = [e.get("G", float("nan")) for e in edge_scores]
        t_raw_list = [e.get("T_raw", float("nan")) for e in edge_scores]
        h_list = [e.get("H", float("nan")) for e in edge_scores]
        t_stats = _summ(t_list)
        s_stats = _summ(s_list)
        g_stats = _summ(g_list)
        t_raw_stats = _summ(t_raw_list)
        h_stats = _summ(h_list)
        t_center_stats = _summ(t_center_cos_list)
        if t_stats:
            t_clean = [v for v in t_list if isinstance(v, (int, float)) and not np.isnan(v)]
            below_floor_count = None
            if transition_floor_val is not None and t_clean:
                below_floor_count = sum(1 for v in t_clean if v < transition_floor_val)
            logger.info(
                "  T transition: mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s  below_floor=%s (floor=%s gamma=%s centered=%s)",
                _fmt(t_stats.get("mean")),
                _fmt(t_stats.get("p10")),
                _fmt(t_stats.get("p50")),
                _fmt(t_stats.get("p90")),
                _fmt(t_stats.get("p99")),
                _fmt(t_stats.get("min")),
                below_floor_count if below_floor_count is not None else (metrics.get("below_floor") if metrics else None),
                transition_floor_val,
                gamma_val,
                transition_centered,
            )
        if transition_centered and t_center_stats:
            logger.info(
                "  T_centered_cos: mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(t_center_stats.get("mean")),
                _fmt(t_center_stats.get("p10")),
                _fmt(t_center_stats.get("p50")),
                _fmt(t_center_stats.get("p90")),
                _fmt(t_center_stats.get("p99")),
                _fmt(t_center_stats.get("min")),
            )
        if os.environ.get("PLAYLIST_DIAG_SONIC"):
            def _spread(stats: Optional[dict]) -> str:
                if not stats:
                    return "n/a"
                p90 = stats.get("p90")
                p10 = stats.get("p10")
                if p90 is None or p10 is None:
                    return "n/a"
                return _fmt(p90 - p10)
            logger.info(
                "Sonic diag: H(p50=%s p90=%s spread=%s) S(p50=%s p90=%s spread=%s)",
                _fmt(h_stats.get("p50") if h_stats else None),
                _fmt(h_stats.get("p90") if h_stats else None),
                _spread(h_stats),
                _fmt(s_stats.get("p50") if s_stats else None),
                _fmt(s_stats.get("p90") if s_stats else None),
                _spread(s_stats),
            )
        if verbose_edges and t_raw_stats:
            logger.info(
                "  T_raw (end->start): mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(t_raw_stats.get("mean")),
                _fmt(t_raw_stats.get("p10")),
                _fmt(t_raw_stats.get("p50")),
                _fmt(t_raw_stats.get("p90")),
                _fmt(t_raw_stats.get("p99")),
                _fmt(t_raw_stats.get("min")),
            )
        if verbose_edges and h_stats:
            logger.info(
                "  H hybrid:     mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(h_stats.get("mean")),
                _fmt(h_stats.get("p10")),
                _fmt(h_stats.get("p50")),
                _fmt(h_stats.get("p90")),
                _fmt(h_stats.get("p99")),
                _fmt(h_stats.get("min")),
            )
        if s_stats:
            logger.info(
                "  S sonic:      mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(s_stats.get("mean")),
                _fmt(s_stats.get("p10")),
                _fmt(s_stats.get("p50")),
                _fmt(s_stats.get("p90")),
                _fmt(s_stats.get("p99")),
                _fmt(s_stats.get("min")),
            )
        if g_stats:
            logger.info(
                "  G genre:      mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(g_stats.get("mean")),
                _fmt(g_stats.get("p10")),
                _fmt(g_stats.get("p50")),
                _fmt(g_stats.get("p90")),
                _fmt(g_stats.get("p99")),
                _fmt(g_stats.get("min")),
            )
        if verbose_edges and baseline:
            logger.info("Baseline vs playlist percentiles:")
            def _log_percentiles(label: str, playlist_vals: Optional[dict], base_vals: Optional[dict]) -> None:
                if not playlist_vals and not base_vals:
                    return
                logger.info(
                    "  %s playlist p50=%s p90=%s p99=%s | baseline p50=%s p90=%s p99=%s",
                    label,
                    _fmt(playlist_vals.get("p50") if playlist_vals else None),
                    _fmt(playlist_vals.get("p90") if playlist_vals else None),
                    _fmt(playlist_vals.get("p99") if playlist_vals else None),
                    _fmt((base_vals or {}).get("p50") if base_vals else None),
                    _fmt((base_vals or {}).get("p90") if base_vals else None),
                    _fmt((base_vals or {}).get("p99") if base_vals else None),
                )
            baseline_t = baseline.get("T_centered_rescaled") if transition_centered else baseline.get("T")
            _log_percentiles("T", t_stats, baseline_t)
            _log_percentiles("T_raw", t_raw_stats, baseline.get("T_raw"))
            _log_percentiles("S", s_stats, baseline.get("S"))
            _log_percentiles("G", g_stats, baseline.get("G"))
        # Weakest transitions
        if edge_scores:
            weakest = sorted(
                [e for e in edge_scores if isinstance(e.get("T"), (int, float)) and not np.isnan(e.get("T"))],
                key=lambda e: e.get("T", float("inf")),
            )[:3]
            logger.info("Weakest transitions (bottom 3 by T):")
            for idx, e in enumerate(weakest, 1):
                logger.info(
                    "  #%s  T=%.3f  S=%.3f  G=%.3f  %s - %s (idx=%s) -> %s - %s (idx=%s)",
                    str(idx).zfill(2),
                    e.get("T", float("nan")),
                    e.get("S", float("nan")),
                    e.get("G", float("nan")),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("artist", "")),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("title", "")),
                    e.get("prev_idx"),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("artist", "")),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("title", "")),
                    e.get("cur_idx"),
                )
    else:
        if dynamic:
            logger.info(f"  - Sonic similarity: {source_counts.get('sonic', 0)} tracks")
            logger.info(f"  - Genre-based: {source_counts.get('genre', 0)} tracks")
        else:
            logger.info(f"  - Sonic similarity: {source_counts.get('sonic', 0)} tracks")

    if artist_name:
        seed_norm = (artist_name or "").strip().casefold()
        norm_artists = [(t.get("artist") or "").strip().casefold() for t in tracks]
        seed_count = sum(1 for a in norm_artists if a == seed_norm)
        seed_pct = (seed_count / len(tracks)) * 100 if tracks else 0
        logger.debug("Seed artist normalized: %s | first3 track artists: %s", seed_norm, norm_artists[:3])
        logger.info(f"  - Seed artist ({sanitize_for_logging(artist_name)}): {seed_count} tracks ({seed_pct:.1f}%)")

    # Duration info
    total_duration_ms = sum((track.get('duration') or 0) for track in tracks)
    total_minutes = total_duration_ms / 1000 / 60
    logger.info(f"Total duration: {total_minutes:.1f} minutes")
    logger.info("=" * 80)
