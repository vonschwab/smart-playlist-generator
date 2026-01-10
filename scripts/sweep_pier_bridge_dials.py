#!/usr/bin/env python3
"""
Sweep pier-bridge dials and summarize run-audit outcomes.

This script is testing-only: it does not modify production logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import os
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import Config
from src.features.artifacts import load_artifact_bundle
from src.playlist.config import default_ds_config, resolve_pier_bridge_tuning
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from src.playlist.pier_bridge_builder import PierBridgeConfig
from src.playlist.scoring.transition_scoring import compute_transition_score
from src.similarity.sonic_variant import (
    apply_transition_weights,
    compute_sonic_variant_matrix,
    resolve_sonic_variant,
)


@dataclass
class Scenario:
    name: str
    seed_track_id: str
    anchor_seed_ids: list[str]
    length: int = 30
    mode: Optional[str] = None


SCENARIOS: dict[str, Scenario] = {
    "bowie": Scenario(
        name="bowie",
        seed_track_id="9dbf29a650b43f5ab7e0e836428000e1",
        anchor_seed_ids=[
            "9dbf29a650b43f5ab7e0e836428000e1",  # Life On Mars?
            "55979af56e425c605d2b8df5c6e1b5d7",  # Warszawa
            "00e9f570e6a9839e2844dd8410242fce",  # Let's Dance
            "328656b5878f1f293a4e981a0b3d961c",  # Suffragette City
        ],
        length=30,
        mode=None,
    ),
}


@dataclass(frozen=True)
class PosthocContext:
    bundle: Any
    X_full_norm: np.ndarray
    X_full_tr_norm: np.ndarray
    X_start_tr_norm: Optional[np.ndarray]
    X_mid_tr_norm: Optional[np.ndarray]
    X_end_tr_norm: Optional[np.ndarray]
    X_genre_norm: Optional[np.ndarray]
    center_transitions: bool
    weight_end_start: float
    weight_mid_mid: float
    weight_full_full: float


def _parse_json_block(text: str, header: str) -> dict[str, Any]:
    marker = f"## {header}"
    idx = text.find(marker)
    if idx == -1:
        return {}
    block = text[idx:]
    start = block.find("```json")
    if start == -1:
        return {}
    start = start + len("```json")
    end = block.find("```", start)
    if end == -1:
        return {}
    raw = block[start:end].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _parse_summary_stats(text: str) -> dict[str, Any]:
    marker = "### summary_stats"
    idx = text.find(marker)
    if idx == -1:
        return {}
    block = text[idx:]
    start = block.find("```json")
    if start == -1:
        return {}
    start = start + len("```json")
    end = block.find("```", start)
    if end == -1:
        return {}
    raw = block[start:end].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _parse_tracklist(text: str) -> list[str]:
    marker = "### tracklist"
    idx = text.find(marker)
    if idx == -1:
        return []
    lines = text[idx:].splitlines()
    rows: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("| pos"):
            in_table = True
            continue
        if in_table and line.startswith("| ---"):
            continue
        if in_table:
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 4:
                continue
            track_id = parts[1].strip("`")
            rows.append(track_id)
    return rows


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def _parse_transition_weights(ds_cfg: dict[str, Any]) -> Optional[tuple[float, float, float]]:
    raw = ds_cfg.get("transition_weights", {})
    if isinstance(raw, dict):
        try:
            return (
                float(raw.get("rhythm", 0.4)),
                float(raw.get("timbre", 0.35)),
                float(raw.get("harmony", 0.25)),
            )
        except Exception:
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except Exception:
            return None
    return None


def _resolve_transition_weights_from_overrides(overrides: dict[str, Any]) -> Optional[tuple[float, float, float]]:
    raw = overrides.get("transition_weights")
    if isinstance(raw, dict):
        try:
            return (
                float(raw.get("rhythm", 0.4)),
                float(raw.get("timbre", 0.35)),
                float(raw.get("harmony", 0.25)),
            )
        except Exception:
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except Exception:
            return None
    return None


def _build_pier_bridge_config(
    *,
    mode: str,
    length: int,
    overrides: dict[str, Any],
    ds_cfg: dict[str, Any],
    beam_width: Optional[int],
) -> Optional[PierBridgeConfig]:
    if beam_width is None:
        return None
    cfg = default_ds_config(mode, playlist_len=length, overrides=ds_cfg)
    tuning, _ = resolve_pier_bridge_tuning(
        mode=cfg.mode,
        similarity_floor=float(cfg.candidate.similarity_floor),
        overrides=overrides,
    )
    transition_weights = _resolve_transition_weights_from_overrides(overrides)
    resolved_variant = resolve_sonic_variant(explicit_variant=None, config_variant=None)
    pb_cfg = PierBridgeConfig(
        transition_floor=float(tuning.transition_floor),
        bridge_floor=float(tuning.bridge_floor),
        center_transitions=cfg.construct.center_transitions,
        transition_weights=transition_weights,
        sonic_variant=resolved_variant,
        weight_bridge=float(tuning.weight_bridge),
        weight_transition=float(tuning.weight_transition),
        genre_tiebreak_weight=float(tuning.genre_tiebreak_weight),
        genre_penalty_threshold=float(tuning.genre_penalty_threshold),
        genre_penalty_strength=float(tuning.genre_penalty_strength),
    )
    return replace(
        pb_cfg,
        initial_beam_width=int(beam_width),
        max_beam_width=int(beam_width),
    )


def _build_posthoc_context(artifact_path: str, ds_cfg: dict[str, Any]) -> PosthocContext:
    bundle = load_artifact_bundle(artifact_path)
    variant = resolve_sonic_variant(explicit_variant=None, config_variant=None)
    X_full_variant, _ = compute_sonic_variant_matrix(bundle.X_sonic, variant, l2=False)
    X_full_norm = _l2_normalize_rows(X_full_variant)

    transition_weights = _parse_transition_weights(ds_cfg)
    X_full_tr, _ = apply_transition_weights(bundle.X_sonic, config_weights=transition_weights)
    X_start_tr = (
        apply_transition_weights(bundle.X_sonic_start, config_weights=transition_weights)[0]
        if bundle.X_sonic_start is not None
        else None
    )
    X_mid_tr = (
        apply_transition_weights(bundle.X_sonic_mid, config_weights=transition_weights)[0]
        if bundle.X_sonic_mid is not None
        else None
    )
    X_end_tr = (
        apply_transition_weights(bundle.X_sonic_end, config_weights=transition_weights)[0]
        if bundle.X_sonic_end is not None
        else None
    )

    center_transitions = bool(ds_cfg.get("constraints", {}).get("center_transitions", False))
    if center_transitions:
        X_full_tr = X_full_tr - X_full_tr.mean(axis=0, keepdims=True)
        if X_start_tr is not None:
            X_start_tr = X_start_tr - X_start_tr.mean(axis=0, keepdims=True)
        if X_mid_tr is not None:
            X_mid_tr = X_mid_tr - X_mid_tr.mean(axis=0, keepdims=True)
        if X_end_tr is not None:
            X_end_tr = X_end_tr - X_end_tr.mean(axis=0, keepdims=True)

    X_full_tr_norm = _l2_normalize_rows(X_full_tr)
    X_start_tr_norm = _l2_normalize_rows(X_start_tr) if X_start_tr is not None else None
    X_mid_tr_norm = _l2_normalize_rows(X_mid_tr) if X_mid_tr is not None else None
    X_end_tr_norm = _l2_normalize_rows(X_end_tr) if X_end_tr is not None else None
    X_genre_norm = None
    if bundle.X_genre_smoothed is not None:
        denom_g = np.linalg.norm(bundle.X_genre_smoothed, axis=1, keepdims=True) + 1e-12
        X_genre_norm = bundle.X_genre_smoothed / denom_g

    pb_defaults = PierBridgeConfig()
    return PosthocContext(
        bundle=bundle,
        X_full_norm=X_full_norm,
        X_full_tr_norm=X_full_tr_norm,
        X_start_tr_norm=X_start_tr_norm,
        X_mid_tr_norm=X_mid_tr_norm,
        X_end_tr_norm=X_end_tr_norm,
        X_genre_norm=X_genre_norm,
        center_transitions=center_transitions,
        weight_end_start=float(pb_defaults.weight_end_start),
        weight_mid_mid=float(pb_defaults.weight_mid_mid),
        weight_full_full=float(pb_defaults.weight_full_full),
    )


def _tracklist_hash(track_ids: list[str]) -> Optional[str]:
    if not track_ids:
        return None
    digest = hashlib.sha256("|".join(track_ids).encode("utf-8")).hexdigest()
    return digest


def _track_ids_to_indices(track_ids: list[str], bundle: Any) -> tuple[Optional[list[int]], list[str]]:
    missing: list[str] = []
    indices: list[int] = []
    for tid in track_ids:
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is None:
            missing.append(str(tid))
        else:
            indices.append(int(idx))
    if missing:
        return None, missing
    return indices, []


def _locate_piers(track_ids: list[str], pier_ids: list[str]) -> tuple[Optional[list[tuple[int, str]]], list[str]]:
    positions: list[tuple[int, str]] = []
    missing: list[str] = []
    for pier_id in pier_ids:
        hits = [i for i, tid in enumerate(track_ids) if tid == pier_id]
        if not hits:
            missing.append(pier_id)
            continue
        positions.append((hits[0], pier_id))
    if missing:
        return None, missing
    positions.sort(key=lambda x: x[0])
    return positions, []


def _segment_positions(track_ids: list[str], pier_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    pier_positions, missing = _locate_piers(track_ids, pier_ids)
    if not pier_positions:
        return [], missing
    segments: list[dict[str, Any]] = []
    for idx in range(len(pier_positions) - 1):
        pos_a, pier_a = pier_positions[idx]
        pos_b, pier_b = pier_positions[idx + 1]
        if pos_b <= pos_a:
            continue
        interior = list(range(pos_a + 1, pos_b))
        segments.append(
            {
                "pier_a": pier_a,
                "pier_b": pier_b,
                "pos_a": pos_a,
                "pos_b": pos_b,
                "interior_positions": interior,
            }
        )
    return segments, []


def _compute_raw_sonic_metrics(
    indices: list[int],
    segments: list[dict[str, Any]],
    ctx: PosthocContext,
) -> dict[str, Optional[float]]:
    if len(indices) < 2:
        return {
            "raw_sonic_sim_mean": None,
            "raw_sonic_sim_min": None,
            "bridge_raw_sonic_sim_mean": None,
            "bridge_raw_sonic_sim_min": None,
        }

    sims: list[float] = []
    for i in range(1, len(indices)):
        sims.append(
            float(
                compute_transition_score(
                    indices[i - 1],
                    indices[i],
                    ctx.X_full_tr_norm,
                    ctx.X_start_tr_norm,
                    ctx.X_mid_tr_norm,
                    ctx.X_end_tr_norm,
                    ctx.weight_end_start,
                    ctx.weight_mid_mid,
                    ctx.weight_full_full,
                    ctx.center_transitions,
                )
            )
        )

    bridge_sims: list[float] = []
    for seg in segments:
        for pos in range(seg["pos_a"], seg["pos_b"]):
            if pos + 1 >= len(indices):
                continue
            bridge_sims.append(
                float(
                    compute_transition_score(
                        indices[pos],
                        indices[pos + 1],
                        ctx.X_full_tr_norm,
                        ctx.X_start_tr_norm,
                        ctx.X_mid_tr_norm,
                        ctx.X_end_tr_norm,
                        ctx.weight_end_start,
                        ctx.weight_mid_mid,
                        ctx.weight_full_full,
                        ctx.center_transitions,
                    )
                )
            )

    return {
        "raw_sonic_sim_mean": float(np.mean(sims)) if sims else None,
        "raw_sonic_sim_min": float(np.min(sims)) if sims else None,
        "bridge_raw_sonic_sim_mean": float(np.mean(bridge_sims)) if bridge_sims else None,
        "bridge_raw_sonic_sim_min": float(np.min(bridge_sims)) if bridge_sims else None,
    }


def _step_fraction(step_idx: int, steps: int) -> float:
    if steps <= 0:
        return 0.0
    return (step_idx + 1) / float(steps + 1)


def _progress_target_arc(step_idx: int, steps: int) -> float:
    if steps <= 0:
        return 0.0
    frac = _step_fraction(step_idx, steps)
    return 0.5 - 0.5 * math.cos(math.pi * frac)


def _compute_pacing_metrics(
    indices: list[int],
    segments: list[dict[str, Any]],
    ctx: PosthocContext,
) -> dict[str, Optional[float]]:
    if not segments:
        return {
            "p50_arc_dev": None,
            "p90_arc_dev": None,
            "max_jump": None,
            "monotonic_violations": None,
            "mean_seg_distance": None,
            "min_seg_distance": None,
        }

    devs: list[float] = []
    max_jump: Optional[float] = None
    monotonic_violations = 0
    seg_distances: list[float] = []
    eps = 1e-6

    for seg in segments:
        if not seg["interior_positions"]:
            continue
        try:
            idx_a = indices[seg["pos_a"]]
            idx_b = indices[seg["pos_b"]]
        except IndexError:
            continue

        vec_a = ctx.X_full_norm[idx_a]
        vec_b = ctx.X_full_norm[idx_b]
        ab = vec_b - vec_a
        denom = float(np.dot(ab, ab))
        if not math.isfinite(denom) or denom <= 1e-12:
            continue
        seg_distances.append(float(math.sqrt(denom)))

        t_vals: list[float] = []
        steps = len(seg["interior_positions"])
        for step_idx, pos in enumerate(seg["interior_positions"]):
            if pos >= len(indices):
                continue
            vec_x = ctx.X_full_norm[indices[pos]]
            t_raw = float(np.dot((vec_x - vec_a), ab) / denom)
            if not math.isfinite(t_raw):
                continue
            t = float(max(0.0, min(1.0, t_raw)))
            t_vals.append(t)
            target = _progress_target_arc(step_idx, steps)
            devs.append(abs(t - target))

        last_t: Optional[float] = None
        for t in t_vals:
            if last_t is not None:
                jump = t - last_t
                if max_jump is None or jump > max_jump:
                    max_jump = jump
                if t < last_t - eps:
                    monotonic_violations += 1
            last_t = t

    if not devs:
        return {
            "p50_arc_dev": None,
            "p90_arc_dev": None,
            "max_jump": None,
            "monotonic_violations": None,
            "mean_seg_distance": float(np.mean(seg_distances)) if seg_distances else None,
            "min_seg_distance": float(np.min(seg_distances)) if seg_distances else None,
        }

    dev_arr = np.array(devs, dtype=float)
    return {
        "p50_arc_dev": float(np.percentile(dev_arr, 50)),
        "p90_arc_dev": float(np.percentile(dev_arr, 90)),
        "max_jump": float(max_jump) if max_jump is not None else None,
        "monotonic_violations": int(monotonic_violations),
        "mean_seg_distance": float(np.mean(seg_distances)) if seg_distances else None,
        "min_seg_distance": float(np.min(seg_distances)) if seg_distances else None,
    }


def _compute_genre_metrics(
    indices: list[int],
    segments: list[dict[str, Any]],
    ctx: PosthocContext,
) -> dict[str, Optional[float]]:
    if not segments or ctx.X_genre_norm is None:
        return {
            "genre_target_sim_mean": None,
            "genre_target_sim_p50": None,
            "genre_target_sim_p90": None,
            "genre_target_delta_mean": None,
        }

    sims: list[float] = []
    deltas: list[float] = []
    for seg in segments:
        if not seg["interior_positions"]:
            continue
        try:
            idx_a = indices[seg["pos_a"]]
            idx_b = indices[seg["pos_b"]]
        except IndexError:
            continue
        g_a = ctx.X_genre_norm[idx_a]
        g_b = ctx.X_genre_norm[idx_b]
        if float(np.linalg.norm(g_a)) <= 1e-8 or float(np.linalg.norm(g_b)) <= 1e-8:
            continue
        steps = len(seg["interior_positions"])
        prev_sim: Optional[float] = None
        for step_idx, pos in enumerate(seg["interior_positions"]):
            if pos >= len(indices):
                continue
            frac = _step_fraction(step_idx, steps)
            g_target = (1.0 - frac) * g_a + frac * g_b
            norm = float(np.linalg.norm(g_target))
            if norm <= 1e-12:
                continue
            g_target = g_target / norm
            sim = float(np.dot(ctx.X_genre_norm[indices[pos]], g_target))
            if not math.isfinite(sim):
                continue
            sims.append(sim)
            if prev_sim is not None:
                deltas.append(abs(sim - prev_sim))
            prev_sim = sim

    if not sims:
        return {
            "genre_target_sim_mean": None,
            "genre_target_sim_p50": None,
            "genre_target_sim_p90": None,
            "genre_target_delta_mean": None,
        }

    sims_arr = np.array(sims, dtype=float)
    return {
        "genre_target_sim_mean": float(np.mean(sims_arr)),
        "genre_target_sim_p50": float(np.percentile(sims_arr, 50)),
        "genre_target_sim_p90": float(np.percentile(sims_arr, 90)),
        "genre_target_delta_mean": float(np.mean(deltas)) if deltas else None,
    }


def _transition_series(indices: list[int], ctx: PosthocContext) -> list[float]:
    sims: list[float] = []
    for i in range(1, len(indices)):
        sims.append(
            float(
                compute_transition_score(
                    indices[i - 1],
                    indices[i],
                    ctx.X_full_tr_norm,
                    ctx.X_start_tr_norm,
                    ctx.X_mid_tr_norm,
                    ctx.X_end_tr_norm,
                    ctx.weight_end_start,
                    ctx.weight_mid_mid,
                    ctx.weight_full_full,
                    ctx.center_transitions,
                )
            )
        )
    return sims


def _arc_dev_by_position(
    indices: list[int],
    segments: list[dict[str, Any]],
    ctx: PosthocContext,
) -> dict[int, float]:
    dev_by_pos: dict[int, float] = {}
    for seg in segments:
        if not seg["interior_positions"]:
            continue
        idx_a = indices[seg["pos_a"]]
        idx_b = indices[seg["pos_b"]]
        vec_a = ctx.X_full_norm[idx_a]
        vec_b = ctx.X_full_norm[idx_b]
        ab = vec_b - vec_a
        denom = float(np.dot(ab, ab))
        if not math.isfinite(denom) or denom <= 1e-12:
            continue
        steps = len(seg["interior_positions"])
        for step_idx, pos in enumerate(seg["interior_positions"]):
            if pos >= len(indices):
                continue
            vec_x = ctx.X_full_norm[indices[pos]]
            t_raw = float(np.dot((vec_x - vec_a), ab) / denom)
            if not math.isfinite(t_raw):
                continue
            t = float(max(0.0, min(1.0, t_raw)))
            target = _progress_target_arc(step_idx, steps)
            dev_by_pos[pos] = abs(t - target)
    return dev_by_pos


def _format_track(track_id: str, bundle: Any) -> str:
    idx = bundle.track_id_to_index.get(str(track_id))
    if idx is None:
        return track_id
    artist = None
    title = None
    if getattr(bundle, "track_artists", None) is not None:
        artist = str(bundle.track_artists[idx])
    if getattr(bundle, "track_titles", None) is not None:
        title = str(bundle.track_titles[idx])
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return title
    return track_id


def _self_test() -> None:
    # Progress computation sanity (A->B, midpoint gives t~0.5).
    a = np.array([0.0, 0.0], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)
    x = np.array([0.5, 0.0], dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = float(np.dot((x - a), ab) / denom)
    assert abs(t - 0.5) < 1e-6

    # Arc target curve endpoints sanity.
    t0 = _progress_target_arc(0, 10)
    t9 = _progress_target_arc(9, 10)
    assert 0.0 <= t0 <= 0.2
    assert 0.8 <= t9 <= 1.0

    # Percentile stability.
    vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    assert abs(np.percentile(vals, 50) - 0.25) < 1e-6


def _safe_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return None


def _z_scores(values: list[Optional[float]]) -> list[Optional[float]]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return [None for _ in values]
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    std = math.sqrt(var)
    if std <= 1e-9:
        return [None for _ in values]
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append((v - mean) / std)
    return out


def _bucket_arc_strength(enabled: Optional[bool], weight: Optional[float]) -> str:
    if not enabled:
        return "off"
    if weight is None or weight <= 0:
        return "off"
    if weight <= 0.10:
        return "low"
    if weight <= 0.30:
        return "mid"
    if weight <= 1.00:
        return "high"
    return "extreme"


def _bucket_jumpiness(max_step: Optional[float]) -> str:
    if max_step is None:
        return "unbounded"
    if max_step <= 0.15:
        return "tight"
    if max_step <= 0.30:
        return "moderate"
    return "loose"


def _sample_sweep(
    sweep: list[dict[str, Any]],
    *,
    sample_n: int,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    if sample_n <= 0:
        return sweep, {}, warnings

    by_label = {cfg["label"]: cfg for cfg in sweep}
    selected: dict[str, dict[str, Any]] = {}

    def _add(cfg: Optional[dict[str, Any]]) -> None:
        if not cfg:
            return
        selected[cfg["label"]] = cfg

    # Always include baseline if present.
    if "baseline" in by_label:
        _add(by_label["baseline"])

    def _get(cfg: dict[str, Any], path: list[str]) -> Any:
        cur = cfg
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    # Ensure endpoints for numeric dials.
    numeric_paths = [
        (["progress_arc", "weight"], "progress_arc_weight"),
        (["progress_arc", "tolerance"], "progress_arc_tolerance"),
        (["progress_arc", "max_step"], "progress_arc_max_step"),
    ]
    for path, _ in numeric_paths:
        values = []
        for cfg in sweep:
            val = _get(cfg, path)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                values.append(float(val))
        if values:
            vmin = min(values)
            vmax = max(values)
            for target in (vmin, vmax):
                candidates = [cfg for cfg in sweep if _get(cfg, path) == target]
                if candidates:
                    _add(sorted(candidates, key=lambda c: c["label"])[0])

    # Ensure all categorical values appear at least once.
    categorical_paths = [
        (["progress_arc", "loss"], "progress_arc_loss"),
        (["progress_arc", "autoscale", "enabled"], "progress_arc_autoscale_enabled"),
        (["genre", "tie_break_band"], "genre_tie_break_band"),
        (["beam_width"], "beam_width"),
    ]
    for path, _ in categorical_paths:
        values = []
        for cfg in sweep:
            values.append(_get(cfg, path))
        for val in sorted(set(values), key=lambda v: str(v)):
            candidates = [cfg for cfg in sweep if _get(cfg, path) == val]
            if candidates:
                _add(sorted(candidates, key=lambda c: c["label"])[0])

    required = list(selected.values())
    if len(required) > sample_n:
        warnings.append(
            f"Requested sample_n={sample_n} but required endpoints={len(required)}; using endpoints only."
        )
        return required, {"sample_n": len(required), "sample_seed": sample_seed}, warnings

    remaining = [cfg for cfg in sweep if cfg["label"] not in selected]
    rng = random.Random(sample_seed)
    rng.shuffle(remaining)
    needed = max(0, sample_n - len(required))
    sampled = required + remaining[:needed]
    return sampled, {"sample_n": sample_n, "sample_seed": sample_seed}, warnings


def _select_shortlist(
    sweep_full: list[dict[str, Any]],
    *,
    shortlist_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not shortlist_csv.exists():
        raise SystemExit(f"Shortlist CSV not found: {shortlist_csv}")
    rows = list(csv.DictReader(shortlist_csv.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise SystemExit(f"Shortlist CSV is empty: {shortlist_csv}")

    def _safe_float_str(val: Any) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    pacing_rows = [
        r
        for r in rows
        if _safe_float_str(r.get("p90_arc_dev")) is not None
        and _safe_float_str(r.get("max_jump")) is not None
    ]
    pacing_rows = sorted(
        pacing_rows,
        key=lambda r: (_safe_float_str(r.get("p90_arc_dev")), _safe_float_str(r.get("max_jump"))),
    )
    balanced_rows = [
        r for r in rows if _safe_float_str(r.get("composite_score")) is not None
    ]
    balanced_rows = sorted(
        balanced_rows, key=lambda r: _safe_float_str(r.get("composite_score")), reverse=True
    )

    labels = {"baseline"}
    labels.update(r.get("label") for r in pacing_rows[:10] if r.get("label"))
    labels.update(r.get("label") for r in balanced_rows[:10] if r.get("label"))

    label_map = {cfg["label"]: cfg for cfg in sweep_full}
    selected = []
    for label in sorted(labels):
        cfg = label_map.get(label)
        if cfg is None:
            warnings.append(f"Shortlist label missing from sweep: {label}")
            continue
        selected.append(cfg)

    return selected, {"mode": "shortlist", "source": str(shortlist_csv)}, warnings


def _jaccard(a: list[str], b: list[str]) -> Optional[float]:
    if not a or not b:
        return None
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return None
    return len(sa & sb) / float(len(sa | sb))


def _build_sweep_matrix() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    configs.append(
        {
            "label": "baseline",
            "progress_arc": {"enabled": False},
            "genre": {"tie_break_band": None},
            "soft_genre_penalty_strength": None,
            "beam_width": None,
        }
    )

    weights = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0]
    tolerances = [0.00, 0.05, 0.08, 0.12, 0.20]
    losses = ["abs", "huber"]
    max_steps = [None, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
    autoscale_enabled = [False, True]
    tie_breaks = [None, 0.02]
    beam_widths = [None, 10, 25, 50]

    for weight in weights:
        for tol in tolerances:
            for loss in losses:
                for max_step in max_steps:
                    for autoscale in autoscale_enabled:
                        for tie_break in tie_breaks:
                            for beam_width in beam_widths:
                                cfg = {
                                    "label": f"arc_w{weight}_t{tol}_{loss}_m{max_step}_auto{autoscale}_tb{tie_break}_beam{beam_width}",
                                    "progress_arc": {
                                        "enabled": True,
                                        "weight": weight,
                                        "tolerance": tol,
                                        "loss": loss,
                                        "shape": "arc",
                                        "huber_delta": 0.10,
                                        "max_step": max_step,
                                        "max_step_mode": "penalty",
                                        "max_step_penalty": 0.25,
                                        "autoscale": {
                                            "enabled": autoscale,
                                            "min_distance": 0.05,
                                            "distance_scale": 0.50,
                                            "per_step_scale": True,
                                        },
                                    },
                                    "genre": {"tie_break_band": tie_break},
                                    "soft_genre_penalty_strength": None,
                                    "beam_width": beam_width,
                                }
                                configs.append(cfg)
    return configs


def _build_overrides(config: dict[str, Any], audit_out_dir: str) -> dict[str, Any]:
    pier_bridge: dict[str, Any] = {
        "audit_run": {
            "enabled": True,
            "out_dir": audit_out_dir,
            "include_top_k": 25,
            "max_bytes": 350000,
            "write_on_success": True,
            "write_on_failure": True,
        },
    }
    progress_arc = config.get("progress_arc")
    if isinstance(progress_arc, dict):
        pier_bridge["progress_arc"] = progress_arc
    genre_cfg = config.get("genre")
    if isinstance(genre_cfg, dict) and genre_cfg.get("tie_break_band") is not None:
        pier_bridge["genre"] = {"tie_break_band": genre_cfg.get("tie_break_band")}
    if config.get("soft_genre_penalty_strength") is not None:
        pier_bridge["soft_genre_penalty_strength"] = config.get("soft_genre_penalty_strength")
    return {"pier_bridge": pier_bridge}


def _extract_metrics(audit_path: Path) -> dict[str, Any]:
    text = audit_path.read_text(encoding="utf-8", errors="replace")
    summary = _parse_summary_stats(text)
    pool = _parse_json_block(text, "3) Pool / Gating Summary")
    track_ids = _parse_tracklist(text)
    metrics = {
        "min_transition": _safe_float(summary.get("min_transition")),
        "mean_transition": _safe_float(summary.get("mean_transition")),
        "below_floor_count": summary.get("below_floor_count"),
        "soft_genre_penalty_hits": summary.get("soft_genre_penalty_hits"),      
        "soft_genre_penalty_edges_scored": summary.get("soft_genre_penalty_edges_scored"),
        "candidate_pool_size": None,
        "candidate_pool_after_dedupe": None,
        "genre_cache_hit_rate": None,
    }
    pool_stats = pool.get("candidate_pool_stats", {}) if isinstance(pool, dict) else {}
    if isinstance(pool_stats, dict):
        metrics["candidate_pool_size"] = pool_stats.get("pool_size")
    if isinstance(pool, dict):
        metrics["candidate_pool_after_dedupe"] = pool.get("candidate_pool_indices_after_dedupe")
    if "genre_cache" in text:
        metrics["genre_cache_missing"] = False
    else:
        metrics["genre_cache_missing"] = True
    return metrics, track_ids


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted(k for k in rows[0].keys() if not k.startswith("_"))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cleaned = {k: v for k, v in row.items() if not k.startswith("_")}
            writer.writerow(cleaned)


def _unique_hashes(rows: list[dict[str, Any]]) -> set[str]:
    hashes = {str(r.get("tracklist_hash")) for r in rows if r.get("tracklist_hash")}
    return {h for h in hashes if h and h != "None"}


def _group_hashes_by_key(rows: list[dict[str, Any]], key: str) -> list[tuple[str, int]]:
    groups: dict[str, set[str]] = {}
    for row in rows:
        bucket = str(row.get(key))
        h = row.get("tracklist_hash")
        if not h:
            continue
        groups.setdefault(bucket, set()).add(str(h))
    items = [(k, len(v)) for k, v in groups.items()]
    return sorted(items, key=lambda x: (-x[1], x[0]))


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    top_configs: dict[str, list[dict[str, Any]]],
    *,
    baseline_by_seed: dict[str, list[str]],
    posthoc_ctx: PosthocContext,
    anchor_seed_ids: list[str],
    sample_info: Optional[dict[str, Any]],
    total_grid_size: int,
) -> None:
    lines: list[str] = []
    lines.append("# Pier-Bridge Dial Sweep")
    lines.append("")
    lines.append(f"- total_grid_size: {total_grid_size}")
    if sample_info:
        if sample_info.get("mode") == "shortlist":
            lines.append(f"- sampled: False (shortlist source={sample_info.get('source')})")
        else:
            lines.append(f"- sampled: True (n={sample_info.get('sample_n')}, seed={sample_info.get('sample_seed')})")
    else:
        lines.append("- sampled: False")
    lines.append("")
    if not rows:
        lines.append("No sweep results.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    missing_cache = any(row.get("genre_cache_missing") for row in rows)
    missing_posthoc = any(row.get("posthoc_missing") for row in rows)
    if missing_cache or missing_posthoc:
        lines.append("## Missing Metrics")
        lines.append("")
        if missing_cache:
            lines.append("- Genre cache hit rate is not present in the run audit markdown output.")
        if missing_posthoc:
            lines.append("- Some runs could not resolve track IDs against the artifact bundle.")
        lines.append("")

    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- raw_sonic_sim_*: mean/min transition similarity on final ordering using the same transition scoring space.")
    lines.append("- bridge_raw_sonic_*: same as above, restricted to bridge edges between consecutive piers.")
    lines.append("- p50_arc_dev/p90_arc_dev: absolute deviation from the arc target curve using post-hoc progress projection.")
    lines.append("- max_jump: max forward progress jump within a bridge segment (post-hoc).")
    lines.append("- monotonic_violations: count of backward steps in progress within segments.")
    lines.append("- genre_target_sim_*: post-hoc similarity to linear genre target per step (mean/p50/p90).")
    lines.append("- genre_target_delta_mean: mean absolute change in genre target similarity per step (smoothness).")
    lines.append("- composite_score: z-normalized blend of raw_sonic_sim_mean/min, p90_arc_dev, max_jump, and runtime.")
    lines.append("")

    lines.append(f"## Sweep Results ({len(_unique_hashes(rows))} unique tracklists)")
    lines.append("")
    lines.append("| label | seed | raw_sonic_sim_mean | raw_sonic_sim_min | bridge_raw_sonic_mean | bridge_raw_sonic_min | p90_arc_dev | max_jump | mono_viol | genre_target_sim_mean | genre_target_delta_mean | overlap | hash | runtime_s |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for row in rows:
        lines.append(
            "| {label} | {seed} | {raw_mean} | {raw_min} | {bridge_mean} | {bridge_min} | {p90} | {max_jump} | {mono} | {genre_mean} | {genre_delta} | {overlap} | `{hash}` | {runtime_s} |".format(
                label=row.get("label"),
                seed=row.get("seed_track_id"),
                raw_mean=row.get("raw_sonic_sim_mean"),
                raw_min=row.get("raw_sonic_sim_min"),
                bridge_mean=row.get("bridge_raw_sonic_sim_mean"),
                bridge_min=row.get("bridge_raw_sonic_sim_min"),
                p90=row.get("p90_arc_dev"),
                max_jump=row.get("max_jump"),
                mono=row.get("monotonic_violations"),
                genre_mean=row.get("genre_target_sim_mean"),
                genre_delta=row.get("genre_target_delta_mean"),
                overlap=row.get("overlap_jaccard"),
                hash=row.get("tracklist_hash"),
                runtime_s=row.get("runtime_s"),
            )
        )
    lines.append("")

    def _render_top(title: str, items: list[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for item in items:
            lines.append(
                "- {label}: raw_sonic_sim_mean={raw_sonic_sim_mean}, raw_sonic_sim_min={raw_sonic_sim_min}, p90_arc_dev={p90_arc_dev}, max_jump={max_jump}, overlap={overlap_jaccard}".format(
                    **item
                )
            )
        lines.append("")

    _render_top("Top Configs: Smoothness", top_configs.get("smoothness", []))
    _render_top("Top Configs: Pacing", top_configs.get("pacing", []))
    _render_top("Top Configs: Balanced", top_configs.get("balanced", []))

    lines.append("## Separation Table")
    lines.append("")
    lines.append("| dial | value | unique_tracklists |")
    lines.append("| --- | --- | ---: |")
    for dial_key in [
        "progress_arc_weight",
        "progress_arc_loss",
        "progress_arc_max_step",
        "progress_arc_tolerance",
        "progress_arc_autoscale_enabled",
        "genre_tie_break_band",
        "beam_width",
        "arc_strength_bucket",
        "jumpiness_bucket",
    ]:
        for value, count in _group_hashes_by_key(rows, dial_key)[:6]:
            lines.append(f"| {dial_key} | {value} | {count} |")
    lines.append("")

    lines.append("## What Changed vs Baseline")
    lines.append("")
    rows_by_seed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_seed.setdefault(str(row.get("seed_track_id")), []).append(row)
    for seed_id, seed_rows in rows_by_seed.items():
        baseline_ids = baseline_by_seed.get(seed_id)
        if not baseline_ids:
            continue
        lines.append(f"### seed {seed_id}")
        baseline_hash = next((r.get("tracklist_hash") for r in seed_rows if r.get("label") == "baseline"), None)
        lines.append(f"- baseline hash: `{baseline_hash}`")
        lines.append("")
        for row in seed_rows:
            if row.get("tracklist_hash") == baseline_hash:
                continue
            track_ids = row.get("_track_ids") or []
            indices, missing = _track_ids_to_indices(track_ids, posthoc_ctx.bundle)
            base_indices, _ = _track_ids_to_indices(baseline_ids, posthoc_ctx.bundle)
            if indices is None or base_indices is None:
                continue
            segments, _ = _segment_positions(track_ids, anchor_seed_ids)
            base_segments, _ = _segment_positions(baseline_ids, anchor_seed_ids)
            cur_trans = _transition_series(indices, posthoc_ctx)
            base_trans = _transition_series(base_indices, posthoc_ctx)
            cur_dev = _arc_dev_by_position(indices, segments, posthoc_ctx)
            base_dev = _arc_dev_by_position(base_indices, base_segments, posthoc_ctx)
            diffs = []
            for i in range(min(len(track_ids), len(baseline_ids))):
                if track_ids[i] != baseline_ids[i]:
                    diffs.append(i)
            if len(track_ids) != len(baseline_ids):
                diffs.extend(range(min(len(track_ids), len(baseline_ids)), max(len(track_ids), len(baseline_ids))))
            if not diffs:
                continue
            lines.append(f"- hash `{row.get('tracklist_hash')}` label `{row.get('label')}` overlap={row.get('overlap_jaccard')}")
            for pos in diffs[:20]:
                base_id = baseline_ids[pos] if pos < len(baseline_ids) else None
                cur_id = track_ids[pos] if pos < len(track_ids) else None
                base_label = _format_track(base_id, posthoc_ctx.bundle) if base_id else "missing"
                cur_label = _format_track(cur_id, posthoc_ctx.bundle) if cur_id else "missing"
                base_t = base_trans[pos - 1] if pos > 0 and pos - 1 < len(base_trans) else None
                cur_t = cur_trans[pos - 1] if pos > 0 and pos - 1 < len(cur_trans) else None
                base_d = base_dev.get(pos)
                cur_d = cur_dev.get(pos)
                lines.append(
                    f"  - pos {pos+1}: {base_label} -> {cur_label} | base_T={base_t} cur_T={cur_t} base_dev={base_d} cur_dev={cur_d}"
                )
            lines.append("")

    lines.append("## UI Dial Proposal")
    lines.append("")
    lines.append("The following presets are derived from sweep outcomes. If pacing metrics are missing, use smoothness + overlap until pacing metrics are available.")
    lines.append("")
    presets = top_configs.get("presets", [])
    if presets:
        lines.append("| preset | label | weight | tolerance | loss | max_step | autoscale | tie_break_band |")
        lines.append("| --- | --- | ---: | ---: | --- | ---: | --- | ---: |")
        for p in presets:
            lines.append(
                "| {preset} | {label} | {weight} | {tolerance} | {loss} | {max_step} | {autoscale} | {tie_break_band} |".format(
                    **p
                )
            )
    else:
        lines.append("- No preset mapping generated.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    numeric_fields = [
        "raw_sonic_sim_mean",
        "raw_sonic_sim_min",
        "bridge_raw_sonic_sim_mean",
        "bridge_raw_sonic_sim_min",
        "p50_arc_dev",
        "p90_arc_dev",
        "max_jump",
        "monotonic_violations",
        "genre_target_sim_mean",
        "genre_target_sim_p50",
        "genre_target_sim_p90",
        "genre_target_delta_mean",
        "overlap_jaccard",
        "runtime_s",
        "below_floor_count",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("label")), []).append(row)
    agg_rows: list[dict[str, Any]] = []
    for label, items in grouped.items():
        first = items[0]
        agg: dict[str, Any] = {
            "label": label,
            "count_runs": len(items),
            "unique_tracklists": len({r.get("tracklist_hash") for r in items if r.get("tracklist_hash")}),
            "progress_arc_enabled": first.get("progress_arc_enabled"),
            "progress_arc_weight": first.get("progress_arc_weight"),
            "progress_arc_tolerance": first.get("progress_arc_tolerance"),
            "progress_arc_loss": first.get("progress_arc_loss"),
            "progress_arc_max_step": first.get("progress_arc_max_step"),
            "progress_arc_autoscale_enabled": first.get("progress_arc_autoscale_enabled"),
            "genre_tie_break_band": first.get("genre_tie_break_band"),
            "beam_width": first.get("beam_width"),
            "arc_strength_bucket": first.get("arc_strength_bucket"),
            "jumpiness_bucket": first.get("jumpiness_bucket"),
        }
        for field in numeric_fields:
            vals = [r.get(field) for r in items if isinstance(r.get(field), (int, float))]
            if not vals:
                agg[f"{field}_mean"] = None
                agg[f"{field}_std"] = None
            else:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                agg[f"{field}_mean"] = float(mean)
                agg[f"{field}_std"] = float(math.sqrt(var))
        agg_rows.append(agg)
    return agg_rows


def _write_markdown_agg(path: Path, rows: list[dict[str, Any]], *, sample_info: Optional[dict[str, Any]], total_grid_size: int) -> None:
    lines: list[str] = []
    lines.append("# Pier-Bridge Dial Sweep (Aggregated)")
    lines.append("")
    lines.append(f"- total_grid_size: {total_grid_size}")
    if sample_info:
        if sample_info.get("mode") == "shortlist":
            lines.append(f"- sampled: False (shortlist source={sample_info.get('source')})")
        else:
            lines.append(f"- sampled: True (n={sample_info.get('sample_n')}, seed={sample_info.get('sample_seed')})")
    else:
        lines.append("- sampled: False")
    lines.append("")
    if not rows:
        lines.append("No aggregated results.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Aggregated Results")
    lines.append("")
    lines.append("| label | runs | unique_hashes | raw_mean | raw_min | p90_dev | max_jump | overlap | runtime_s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {label} | {runs} | {unique} | {raw_mean} | {raw_min} | {p90} | {jump} | {overlap} | {runtime} |".format(
                label=row.get("label"),
                runs=row.get("count_runs"),
                unique=row.get("unique_tracklists"),
                raw_mean=row.get("raw_sonic_sim_mean_mean"),
                raw_min=row.get("raw_sonic_sim_min_mean"),
                p90=row.get("p90_arc_dev_mean"),
                jump=row.get("max_jump_mean"),
                overlap=row.get("overlap_jaccard_mean"),
                runtime=row.get("runtime_s_mean"),
            )
        )
    lines.append("")
    baseline = next((r for r in rows if r.get("label") == "baseline"), None)
    if baseline:
        lines.append("## Stability Shortlists")
        lines.append("")
        lines.append("### Best Pacing (mean p90, then std p90)")
        lines.append("")
        pacing_sorted = sorted(
            rows,
            key=lambda r: (
                r.get("p90_arc_dev_mean") if r.get("p90_arc_dev_mean") is not None else 1e9,
                r.get("p90_arc_dev_std") if r.get("p90_arc_dev_std") is not None else 1e9,
            ),
        )[:10]
        for row in pacing_sorted:
            lines.append(
                "- {label}: p90_mean={p90m}, p90_std={p90s}, max_jump_mean={jm}, bridge_min_mean={bm}".format(
                    label=row.get("label"),
                    p90m=row.get("p90_arc_dev_mean"),
                    p90s=row.get("p90_arc_dev_std"),
                    jm=row.get("max_jump_mean"),
                    bm=row.get("bridge_raw_sonic_sim_min_mean"),
                )
            )
        lines.append("")

        lines.append("### Balanced (beats baseline on mean p90, mean max_jump, mean bridge_min)")
        lines.append("")
        balanced = []
        for row in rows:
            if row.get("label") == "baseline":
                continue
            if (
                row.get("p90_arc_dev_mean") is not None
                and row.get("max_jump_mean") is not None
                and row.get("bridge_raw_sonic_sim_min_mean") is not None
                and baseline.get("p90_arc_dev_mean") is not None
                and baseline.get("max_jump_mean") is not None
                and baseline.get("bridge_raw_sonic_sim_min_mean") is not None
            ):
                if (
                    row["p90_arc_dev_mean"] < baseline["p90_arc_dev_mean"]
                    and row["max_jump_mean"] < baseline["max_jump_mean"]
                    and row["bridge_raw_sonic_sim_min_mean"] > baseline["bridge_raw_sonic_sim_min_mean"]
                ):
                    balanced.append(row)
        balanced = sorted(
            balanced,
            key=lambda r: (r.get("p90_arc_dev_mean"), r.get("max_jump_mean")),
        )[:10]
        for row in balanced:
            lines.append(
                "- {label}: p90_mean={p90m}, max_jump_mean={jm}, bridge_min_mean={bm}, overlap_mean={ov}".format(
                    label=row.get("label"),
                    p90m=row.get("p90_arc_dev_mean"),
                    jm=row.get("max_jump_mean"),
                    bm=row.get("bridge_raw_sonic_sim_min_mean"),
                    ov=row.get("overlap_jaccard_mean"),
                )
            )
        if not balanced:
            lines.append("- None (no configs beat baseline on all three criteria).")
        lines.append("")

        lines.append("### Most Seed-Unstable (highest p90 std, then max_jump std)")
        lines.append("")
        unstable = sorted(
            rows,
            key=lambda r: (
                r.get("p90_arc_dev_std") if r.get("p90_arc_dev_std") is not None else -1.0,
                r.get("max_jump_std") if r.get("max_jump_std") is not None else -1.0,
            ),
            reverse=True,
        )[:10]
        for row in unstable:
            lines.append(
                "- {label}: p90_std={p90s}, max_jump_std={jm}, unique_hashes={uh}".format(
                    label=row.get("label"),
                    p90s=row.get("p90_arc_dev_std"),
                    jm=row.get("max_jump_std"),
                    uh=row.get("unique_tracklists"),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep pier-bridge dial configs.")
    parser.add_argument("--scenario", default="bowie", help="Scenario name (default: bowie).")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for determinism.")
    parser.add_argument("--runs-per-config", type=int, default=1, help="Runs per config.")
    parser.add_argument("--max-runs", type=int, default=None, help="Cap total runs.")
    parser.add_argument("--filter", type=str, default=None, help="Substring filter for labels.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running.")
    parser.add_argument("--self-test", action="store_true", help="Run basic self-tests and exit.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed track IDs.")
    parser.add_argument("--sample", type=int, default=None, help="Deterministically sample N configs.")
    parser.add_argument("--sample-seed", type=int, default=1337, help="Seed for deterministic sampling.")
    parser.add_argument("--shortlist", action="store_true", help="Run shortlist configs from a prior CSV.")
    parser.add_argument("--shortlist-path", type=str, default="docs/diagnostics/pier_bridge_dial_sweep.csv", help="CSV path for shortlist selection.")
    parser.add_argument("--out-dir", type=str, default="docs/diagnostics", help="Output directory.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("Self-test: OK")
        return 0

    cfg = Config()
    ds_cfg = cfg.get("playlists", "ds_pipeline", default={}) or {}
    artifact_path = ds_cfg.get("artifact_path")
    if not artifact_path:
        raise SystemExit("Missing playlists.ds_pipeline.artifact_path in config.yaml")

    scenario = SCENARIOS.get(args.scenario)
    if scenario is None:
        raise SystemExit(f"Unknown scenario: {args.scenario}")

    mode = scenario.mode or ds_cfg.get("mode", "dynamic")
    posthoc_ctx = _build_posthoc_context(artifact_path, ds_cfg)
    sweep_full = _build_sweep_matrix()
    total_grid_size = len(sweep_full)
    sweep = sweep_full
    if args.filter:
        sweep = [c for c in sweep if args.filter in c["label"]]
    sample_info = None
    sample_warnings: list[str] = []
    if args.shortlist:
        sweep, sample_info, sample_warnings = _select_shortlist(
            sweep_full,
            shortlist_csv=Path(args.shortlist_path),
        )
    elif args.sample:
        sweep, sample_info, sample_warnings = _sample_sweep(
            sweep,
            sample_n=int(args.sample),
            sample_seed=int(args.sample_seed),
        )
    if args.max_runs:
        sweep = sweep[: max(0, int(args.max_runs))]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "run_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Planned runs: {len(sweep)}")
        for cfg_item in sweep:
            print(cfg_item["label"])
        return 0

    rows: list[dict[str, Any]] = []
    baseline_by_seed: dict[str, list[str]] = {}
    warnings: list[str] = []
    warnings.extend(sample_warnings)

    seeds = [scenario.seed_track_id]
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    valid_seeds: list[str] = []
    for seed in seeds:
        if posthoc_ctx.bundle.track_id_to_index.get(str(seed)) is not None:
            valid_seeds.append(seed)
            continue
        if str(seed).isdigit():
            idx = int(str(seed))
            if 0 <= idx < len(posthoc_ctx.bundle.track_ids):
                resolved = str(posthoc_ctx.bundle.track_ids[idx])
                warnings.append(f"Seed {seed} resolved by index to track_id {resolved}.")
                valid_seeds.append(resolved)
                continue
        warnings.append(f"Seed {seed} not found in artifact; skipping.")
    if not valid_seeds:
        raise SystemExit("No valid seeds found in artifact.")
    seeds = valid_seeds
    for cfg_item in sweep:
        for run_idx in range(args.runs_per_config):
            for seed_track_id in seeds:
                overrides = _build_overrides(cfg_item, str(audit_dir))
                beam_width = cfg_item.get("beam_width")
                pb_cfg = _build_pier_bridge_config(
                    mode=mode,
                    length=scenario.length,
                    overrides=overrides,
                    ds_cfg=ds_cfg,
                    beam_width=beam_width,
                )
                t0 = time.perf_counter()
                result = generate_playlist_ds(
                    artifact_path=artifact_path,
                    seed_track_id=seed_track_id,
                    anchor_seed_ids=scenario.anchor_seed_ids,
                    mode=mode,
                    length=scenario.length,
                    random_seed=int(args.random_seed),
                    overrides=overrides,
                    dry_run=True,
                    artist_playlist=True,
                    artist_style_enabled=False,
                    pier_bridge_config=pb_cfg,
                )
                runtime = time.perf_counter() - t0

                audit_path = result.playlist_stats.get("playlist", {}).get("audit_path")
                if audit_path:
                    audit_path = Path(audit_path)
                else:
                    audit_path = None

                metrics = {}
                track_ids = []
                if audit_path and audit_path.exists():
                    metrics, track_ids = _extract_metrics(audit_path)
                else:
                    metrics = {
                        "min_transition": None,
                        "mean_transition": None,
                        "below_floor_count": None,
                        "soft_genre_penalty_hits": None,
                        "soft_genre_penalty_edges_scored": None,
                        "candidate_pool_size": None,
                        "candidate_pool_after_dedupe": None,
                        "genre_cache_hit_rate": None,
                        "genre_cache_missing": True,
                    }

                if cfg_item["label"] == "baseline" and seed_track_id not in baseline_by_seed:
                    baseline_by_seed[seed_track_id] = track_ids
                overlap_j = _jaccard(baseline_by_seed.get(seed_track_id, []), track_ids)
                tracklist_hash = _tracklist_hash(track_ids)

                posthoc_missing = False
                indices, missing_ids = _track_ids_to_indices(track_ids, posthoc_ctx.bundle)
                if indices is None:
                    posthoc_missing = True
                    warnings.append(
                        f"{cfg_item['label']} missing {len(missing_ids)} track ids from artifact."
                    )
                    segments = []
                    raw_metrics = {
                        "raw_sonic_sim_mean": None,
                        "raw_sonic_sim_min": None,
                        "bridge_raw_sonic_sim_mean": None,
                        "bridge_raw_sonic_sim_min": None,
                    }
                    pacing_metrics = {
                        "p50_arc_dev": None,
                        "p90_arc_dev": None,
                        "max_jump": None,
                        "monotonic_violations": None,
                        "mean_seg_distance": None,
                        "min_seg_distance": None,
                    }
                    genre_metrics = {
                        "genre_target_sim_mean": None,
                        "genre_target_sim_p50": None,
                        "genre_target_sim_p90": None,
                        "genre_target_delta_mean": None,
                    }
                else:
                    segments, missing_piers = _segment_positions(track_ids, scenario.anchor_seed_ids)
                    if missing_piers:
                        posthoc_missing = True
                        warnings.append(
                            f"{cfg_item['label']} missing pier ids {missing_piers} in tracklist."
                        )
                    raw_metrics = _compute_raw_sonic_metrics(indices, segments, posthoc_ctx)
                    pacing_metrics = _compute_pacing_metrics(indices, segments, posthoc_ctx)
                    genre_metrics = _compute_genre_metrics(indices, segments, posthoc_ctx)

                row = {
                    "label": cfg_item["label"],
                    "run_index": run_idx,
                    "mode": mode,
                    "audit_path": str(audit_path) if audit_path else None,
                    "runtime_s": round(runtime, 3),
                    "overlap_jaccard": overlap_j,
                    "tracklist_hash": tracklist_hash,
                    "posthoc_missing": posthoc_missing,
                    "seed_track_id": seed_track_id,
                    "beam_width": beam_width,
                    "_track_ids": track_ids,
                }
                row.update(metrics)
                row.update(raw_metrics)
                row.update(pacing_metrics)
                row.update(genre_metrics)
                row["progress_arc_enabled"] = cfg_item.get("progress_arc", {}).get("enabled")
                row["progress_arc_weight"] = cfg_item.get("progress_arc", {}).get("weight")
                row["progress_arc_tolerance"] = cfg_item.get("progress_arc", {}).get("tolerance")
                row["progress_arc_loss"] = cfg_item.get("progress_arc", {}).get("loss")
                row["progress_arc_huber_delta"] = cfg_item.get("progress_arc", {}).get("huber_delta")
                row["progress_arc_max_step"] = cfg_item.get("progress_arc", {}).get("max_step")
                row["progress_arc_max_step_mode"] = cfg_item.get("progress_arc", {}).get("max_step_mode")
                row["progress_arc_autoscale_enabled"] = cfg_item.get("progress_arc", {}).get("autoscale", {}).get("enabled")
                row["progress_arc_autoscale_per_step_scale"] = cfg_item.get("progress_arc", {}).get("autoscale", {}).get("per_step_scale")
                row["genre_tie_break_band"] = cfg_item.get("genre", {}).get("tie_break_band") if cfg_item.get("genre") else None
                row["arc_strength_bucket"] = _bucket_arc_strength(
                    row.get("progress_arc_enabled"), row.get("progress_arc_weight")
                )
                row["jumpiness_bucket"] = _bucket_jumpiness(row.get("progress_arc_max_step"))
                if sample_info:
                    row["sample_n"] = sample_info.get("sample_n")
                    row["sample_seed"] = sample_info.get("sample_seed")

                rows.append(row)

    mean_vals = [row.get("raw_sonic_sim_mean") for row in rows]
    min_vals = [row.get("raw_sonic_sim_min") for row in rows]
    p90_vals = [row.get("p90_arc_dev") for row in rows]
    jump_vals = [row.get("max_jump") for row in rows]
    runtime_vals = [row.get("runtime_s") for row in rows]

    mean_z = _z_scores([_safe_float(v) for v in mean_vals])
    min_z = _z_scores([_safe_float(v) for v in min_vals])
    p90_z = _z_scores([_safe_float(v) for v in p90_vals])
    jump_z = _z_scores([_safe_float(v) for v in jump_vals])
    runtime_z = _z_scores([_safe_float(v) for v in runtime_vals])

    for i, row in enumerate(rows):
        score = 0.0
        used = 0
        if mean_z[i] is not None:
            score += 1.0 * mean_z[i]
            used += 1
        if min_z[i] is not None:
            score += 0.5 * min_z[i]
            used += 1
        if jump_z[i] is not None:
            score -= 1.0 * jump_z[i]
            used += 1
        if p90_z[i] is not None:
            score -= 0.5 * p90_z[i]
            used += 1
        if runtime_z[i] is not None:
            score -= 0.25 * runtime_z[i]
            used += 1
        row["composite_score"] = round(score, 4) if used else None

    def _top_by_key(key_func, reverse: bool = True) -> list[dict[str, Any]]:
        items = [r for r in rows if r.get("raw_sonic_sim_mean") is not None]
        return sorted(items, key=key_func, reverse=reverse)[:5]

    top_configs = {
        "smoothness": _top_by_key(
            lambda r: (
                r.get("raw_sonic_sim_min") if r.get("raw_sonic_sim_min") is not None else -1.0,
                r.get("raw_sonic_sim_mean") if r.get("raw_sonic_sim_mean") is not None else -1.0,
            ),
            reverse=True,
        ),
        "pacing": _top_by_key(
            lambda r: (
                -(r.get("p90_arc_dev") if r.get("p90_arc_dev") is not None else 1e9),
                -(r.get("max_jump") if r.get("max_jump") is not None else 1e9),
            ),
            reverse=True,
        ),
        "balanced": _top_by_key(
            lambda r: (r.get("composite_score") or -1e9),
            reverse=True,
        ),
    }

    presets: list[dict[str, Any]] = []
    if rows:
        def _pick(label: str, row: dict[str, Any]) -> None:
            presets.append(
                {
                    "preset": label,
                    "label": row.get("label"),
                    "weight": row.get("progress_arc_weight"),
                    "tolerance": row.get("progress_arc_tolerance"),
                    "loss": row.get("progress_arc_loss"),
                    "max_step": row.get("progress_arc_max_step"),
                    "autoscale": row.get("progress_arc_autoscale_enabled"),
                    "tie_break_band": row.get("genre_tie_break_band"),
                }
            )

        smooth = top_configs["smoothness"][0] if top_configs["smoothness"] else rows[0]
        balanced = top_configs["balanced"][0] if top_configs["balanced"] else rows[0]
        pacing = top_configs["pacing"][0] if top_configs["pacing"] else rows[0]
        rail = pacing
        _pick("Loose", smooth)
        _pick("Balanced", balanced)
        _pick("Guided", pacing)
        _pick("Rail", rail)
    top_configs["presets"] = presets

    csv_path = out_dir / "pier_bridge_dial_sweep.csv"
    md_path = out_dir / "pier_bridge_dial_sweep.md"
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        rows,
        top_configs,
        baseline_by_seed=baseline_by_seed,
        posthoc_ctx=posthoc_ctx,
        anchor_seed_ids=scenario.anchor_seed_ids,
        sample_info=sample_info,
        total_grid_size=total_grid_size,
    )
    agg_rows = _aggregate_rows(rows)
    agg_csv_path = out_dir / "pier_bridge_dial_sweep_agg.csv"
    agg_md_path = out_dir / "pier_bridge_dial_sweep_agg.md"
    _write_csv(agg_csv_path, agg_rows)
    _write_markdown_agg(
        agg_md_path,
        agg_rows,
        sample_info=sample_info,
        total_grid_size=total_grid_size,
    )
    print(str(md_path))
    print(str(csv_path))
    print(str(agg_md_path))
    print(str(agg_csv_path))
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
