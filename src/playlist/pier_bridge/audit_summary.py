"""Audit summary + bridgeability scoring (Tier-3.1 PR-4).

Extracted from pier_bridge_builder.py. Two read-only helpers:

  * _summarize_candidates_for_audit — builds per-candidate diagnostic rows
    plus distribution summaries (used by the audit emitter inside
    build_pier_bridge_playlist).
  * _compute_bridgeability_score — cheap heuristic for pair-seed ordering
    (used by _order_seeds_by_bridgeability, to be extracted in PR-6).

Neither is part of the algorithmic hot path; both are telemetry / pre-pass
helpers. They take a PierBridgeConfig in but only read fields — no mutation.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Set

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.pier_bridge.config import (
    PierBridgeConfig,
    _compute_transition_score_raw_and_transformed,
)
from src.playlist.pier_bridge.metrics import _dist
from src.string_utils import sanitize_for_logging


def _summarize_candidates_for_audit(
    *,
    candidates: list[int],
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    X_full_tr_norm: np.ndarray,
    X_start_tr_norm: Optional[np.ndarray],
    X_mid_tr_norm: Optional[np.ndarray],
    X_end_tr_norm: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    bundle: ArtifactBundle,
    internal_connector_indices: Optional[Set[int]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Optional[float]]]]:
    if not candidates:
        return [], {}

    cand_sorted = sorted(set(int(i) for i in candidates))
    sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
    sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])

    # Progress diagnostics: projection onto the AB direction in the same sonic
    # similarity space used for endpoint sims.
    vec_a_full = X_full_norm[pier_a]
    vec_b_full = X_full_norm[pier_b]
    d = vec_b_full - vec_a_full
    denom_progress = float(np.dot(d, d))
    progress_active = bool(math.isfinite(denom_progress) and denom_progress > 1e-12)

    sim_a_vals: list[float] = []
    sim_b_vals: list[float] = []
    hmean_vals: list[float] = []
    progress_vals: list[float] = []
    tmin_vals: list[float] = []
    t_a_raw_vals: list[float] = []
    t_b_raw_vals: list[float] = []
    t_a_vals: list[float] = []
    t_b_vals: list[float] = []
    gmin_vals: list[float] = []
    g_a_vals: list[float] = []
    g_b_vals: list[float] = []

    genre_vec_a = X_genre_norm[pier_a] if X_genre_norm is not None else None
    genre_vec_b = X_genre_norm[pier_b] if X_genre_norm is not None else None

    rows: list[dict[str, Any]] = []
    for cand in cand_sorted:
        keys = identity_keys_for_index(bundle, int(cand))
        sim_a = float(sim_to_a[cand])
        sim_b = float(sim_to_b[cand])
        denom = sim_a + sim_b
        hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom

        progress_t = None
        if progress_active:
            t_raw = float(np.dot((X_full_norm[cand] - vec_a_full), d) / denom_progress)
            if math.isfinite(t_raw):
                progress_t = float(max(0.0, min(1.0, t_raw)))
                progress_vals.append(float(progress_t))

        t_a_raw, t_a = _compute_transition_score_raw_and_transformed(
            pier_a, cand, X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm, X_end_tr_norm, cfg
        )
        t_b_raw, t_b = _compute_transition_score_raw_and_transformed(
            cand, pier_b, X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm, X_end_tr_norm, cfg
        )
        t_min = min(t_a, t_b)

        g_a = float("nan")
        g_b = float("nan")
        g_min = float("nan")
        if genre_vec_a is not None and genre_vec_b is not None:
            g_a = float(np.dot(genre_vec_a, X_genre_norm[cand]))  # type: ignore[index]
            g_b = float(np.dot(X_genre_norm[cand], genre_vec_b))  # type: ignore[index]
            g_min = min(g_a, g_b) if math.isfinite(g_a) and math.isfinite(g_b) else float("nan")

        final = cfg.weight_bridge * hmean + cfg.weight_transition * t_min
        if math.isfinite(g_min) and cfg.genre_tiebreak_weight:
            final += float(cfg.genre_tiebreak_weight) * float(g_min)
        if (
            cfg.genre_penalty_strength > 0
            and math.isfinite(g_min)
            and float(g_min) < float(cfg.genre_penalty_threshold)
        ):
            final *= (1.0 - float(cfg.genre_penalty_strength))

        artist = (
            str(bundle.track_artists[cand])
            if bundle.track_artists is not None
            else (str(bundle.artist_keys[cand]) if bundle.artist_keys is not None else "")
        )
        title = str(bundle.track_titles[cand]) if bundle.track_titles is not None else ""
        rows.append(
            {
                "track_id": str(bundle.track_ids[cand]),
                "artist": sanitize_for_logging(artist),
                "title": sanitize_for_logging(title),
                "artist_key": keys.artist_key,
                "title_key": keys.title_key,
                "progress_t": (round(float(progress_t), 3) if progress_t is not None else None),
                "simA": round(sim_a, 3),
                "simB": round(sim_b, 3),
                "hmean": round(hmean, 3),
                "bridge_sim": round(hmean, 3),
                "T_min": round(float(t_min), 3),
                "G_min": (round(float(g_min), 3) if math.isfinite(g_min) else None),
                "final": round(float(final), 3),
                "internal": bool(internal_connector_indices and cand in internal_connector_indices),
            }
        )

        sim_a_vals.append(sim_a)
        sim_b_vals.append(sim_b)
        hmean_vals.append(hmean)
        tmin_vals.append(float(t_min))
        t_a_raw_vals.append(float(t_a_raw))
        t_b_raw_vals.append(float(t_b_raw))
        t_a_vals.append(float(t_a))
        t_b_vals.append(float(t_b))
        if math.isfinite(g_min):
            gmin_vals.append(float(g_min))
        if math.isfinite(g_a):
            g_a_vals.append(float(g_a))
        if math.isfinite(g_b):
            g_b_vals.append(float(g_b))

    rows = sorted(rows, key=lambda r: (-float(r.get("final") or 0.0), str(r.get("track_id", ""))))[: max(0, int(top_k))]

    dists: dict[str, dict[str, Optional[float]]] = {
        "simA": _dist(sim_a_vals),
        "simB": _dist(sim_b_vals),
        "hmean": _dist(hmean_vals),
        "progress_t": _dist(progress_vals),
        "T_min": _dist(tmin_vals),
        "T_raw_pierA_to_cand": _dist(t_a_raw_vals),
        "T_raw_cand_to_pierB": _dist(t_b_raw_vals),
        "T_pierA_to_cand": _dist(t_a_vals),
        "T_cand_to_pierB": _dist(t_b_vals),
    }
    if X_genre_norm is not None:
        dists["G_min"] = _dist(gmin_vals)
        dists["G_pierA_to_cand"] = _dist(g_a_vals)
        dists["G_cand_to_pierB"] = _dist(g_b_vals)
    return rows, dists


def _compute_bridgeability_score(
    idx_a: int,
    idx_b: int,
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
) -> float:
    """
    Cheap heuristic for how well two seeds can be bridged.
    Uses direct transition similarity plus a term for the distance between them.
    """
    # Direct transition similarity
    if X_end_norm is not None and X_start_norm is not None:
        direct_sim = float(np.dot(X_end_norm[idx_a], X_start_norm[idx_b]))
    else:
        direct_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Full similarity (for overall coherence)
    full_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Combine: favor pairs with good direct transitions
    return 0.6 * direct_sim + 0.4 * full_sim
