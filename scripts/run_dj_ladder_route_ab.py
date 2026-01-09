#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import Config
from src.playlist.config import default_ds_config, resolve_pier_bridge_tuning
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from src.playlist.pier_bridge_builder import PierBridgeConfig
from src.similarity.sonic_variant import resolve_sonic_variant

AUDIT_DIR = REPO_ROOT / "docs" / "diagnostics" / "run_audits"
OUT_DOC = REPO_ROOT / "docs" / "diagnostics" / "dj_ladder_route_ab.md"

SCENARIO = {
    "name": "dj_stress_indie_ladder",
    "seed_track_id": "99cbf799b947abc1efadeff958fa3a86",  # Destroyer - The Space Race
    "anchor_seed_ids": [
        "99cbf799b947abc1efadeff958fa3a86",  # Destroyer - The Space Race
        "22d617435daa61958c67624fca3dbc23",  # Wilco - Spiders (Kidsmoke)
        "f4282083af4d8149e32c76c2eccea5bc",  # Early Day Miners - The Union Trade
        "7cf9096e6cc377f4d38d4af7d8bdca69",  # North Americans - Bleeding Heart Tetra
        "b6bae06cc55006d73afa712edf0a28d2",  # MJ Lenderman - Rudolph
        "9cc4ace8edc409d5e94a08bfdf2a3615",  # Elliott Smith - Needle In The Hay
    ],
    "length": 30,
    "mode": "dynamic",
}

RUNS = [
    {"label": "linear", "route_shape": "linear", "smoothed": False},
    {"label": "ladder_onehot", "route_shape": "ladder", "smoothed": False},
    {"label": "ladder_smoothed", "route_shape": "ladder", "smoothed": True},
]


def _load_sweep_helpers():
    import importlib.util

    sweep_path = REPO_ROOT / "scripts" / "sweep_pier_bridge_dials.py"
    spec = importlib.util.spec_from_file_location("sweep_helpers", sweep_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load sweep helpers from {sweep_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _build_pier_bridge_config(
    mode: str, length: int, ds_cfg: dict[str, Any]
) -> PierBridgeConfig:
    cfg = default_ds_config(mode, playlist_len=length, overrides=ds_cfg)
    tuning, _ = resolve_pier_bridge_tuning(
        mode=cfg.mode,
        similarity_floor=float(cfg.candidate.similarity_floor),
        overrides={},
    )
    resolved_variant = resolve_sonic_variant(explicit_variant=None, config_variant=None)
    pb_cfg = PierBridgeConfig(
        transition_floor=float(tuning.transition_floor),
        bridge_floor=float(tuning.bridge_floor),
        center_transitions=cfg.construct.center_transitions,
        transition_weights=None,
        sonic_variant=resolved_variant,
        weight_bridge=float(tuning.weight_bridge),
        weight_transition=float(tuning.weight_transition),
        genre_tiebreak_weight=float(tuning.genre_tiebreak_weight),
        genre_penalty_threshold=float(tuning.genre_penalty_threshold),
        genre_penalty_strength=float(tuning.genre_penalty_strength),
    )
    return replace(pb_cfg, initial_beam_width=pb_cfg.initial_beam_width, max_beam_width=pb_cfg.max_beam_width)


def _build_overrides(*, route_shape: str, smoothed: bool) -> dict[str, Any]:
    pier_bridge: dict[str, Any] = {
        "audit_run": {
            "enabled": True,
            "out_dir": str(AUDIT_DIR),
            "include_top_k": 25,
            "max_bytes": 350000,
            "write_on_success": True,
            "write_on_failure": True,
        },
        "segment_pool_max": 80,
        "max_segment_pool_max": 80,
        "dj_bridging": {
            "enabled": True,
            "route_shape": str(route_shape),
            "pooling": {
                "strategy": "dj_union",
                "debug_compare_baseline": True,
            },
            "ladder": {
                "use_smoothed_waypoint_vectors": bool(smoothed),
            },
        },
    }
    return {"pier_bridge": pier_bridge}


def _hash_tracklist(track_ids: list[str]) -> str:
    digest = hashlib.sha1()
    for tid in track_ids:
        digest.update(str(tid).encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()[:12]


def _parse_pool_counts_by_segment(text: str) -> list[dict[str, Any]]:
    import re

    pool_counts: list[dict[str, Any]] = []
    cur_seg = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("### Segment "):
            match = re.match(r"### Segment\\s+(\\d+)", line)
            if match:
                cur_seg = int(match.group(1))
        if line == "**pool_counts**":
            start = None
            end = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "```json":
                    start = j + 1
                    continue
                if start is not None and lines[j].strip() == "```":
                    end = j
                    break
            if start is not None and end is not None:
                raw = "\n".join(lines[start:end]).strip()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                payload["_segment_index"] = cur_seg
                pool_counts.append(payload)
                i = end
        i += 1
    return pool_counts


def _aggregate_pool_counts(rows: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    if not rows:
        return {
            "pool_overlap_jaccard": None,
            "pool_overlap_union_only": None,
            "chosen_from_local_count": None,
            "chosen_from_genre_count": None,
            "chosen_from_toward_count": None,
        }

    def _mean(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _sum(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not vals:
            return None
        return float(sum(vals))

    return {
        "pool_overlap_jaccard": _mean("pool_overlap_jaccard"),
        "pool_overlap_union_only": _mean("pool_overlap_union_only"),
        "chosen_from_local_count": _sum("chosen_from_local_count"),
        "chosen_from_genre_count": _sum("chosen_from_genre_count"),
        "chosen_from_toward_count": _sum("chosen_from_toward_count"),
    }


def _segment_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Optional[float]]:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    if not vals:
        return {"min": None, "median": None, "max": None}
    arr = np.array(vals, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.percentile(arr, 50)),
        "max": float(np.max(arr)),
    }


def _extract_segment_waypoints(playlist_stats: dict[str, Any]) -> list[dict[str, Any]]:
    segments = playlist_stats.get("playlist", {}).get("segment_diagnostics") or []
    rows: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        rows.append(
            {
                "segment": int(idx),
                "route_shape": seg.get("route_shape"),
                "ladder_waypoint_labels": seg.get("ladder_waypoint_labels") or [],
                "ladder_waypoint_count": seg.get("ladder_waypoint_count"),
                "ladder_waypoint_vector_mode": seg.get("ladder_waypoint_vector_mode"),
                "ladder_waypoint_vector_stats": seg.get("ladder_waypoint_vector_stats") or [],
            }
        )
    return rows


def _extract_ladder_warnings(playlist_stats: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = playlist_stats.get("playlist", {}).get("warnings") or []
    if not isinstance(warnings, list):
        return []
    ladder_types = {
        "genre_ladder_unavailable",
        "genre_ladder_label_unmapped",
        "genre_ladder_smoothed_fallback",
        "genre_missing",
    }
    out: list[dict[str, Any]] = []
    for entry in warnings:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") in ladder_types:
            out.append(entry)
    return out


def _resolve_waypoint_mode(segments: list[dict[str, Any]]) -> str:
    modes = {
        str(seg.get("ladder_waypoint_vector_mode"))
        for seg in segments
        if seg.get("ladder_waypoint_vector_mode") is not None
    }
    modes.discard("None")
    if not modes:
        return "na"
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def main() -> int:
    cfg = Config()
    ds_cfg = cfg.get("playlists", "ds_pipeline", default={}) or {}
    artifact_path = ds_cfg.get("artifact_path")
    if not artifact_path:
        raise ValueError("Missing playlists.artifact_path in config")

    sweep = _load_sweep_helpers()
    posthoc_ctx = sweep._build_posthoc_context(str(artifact_path), ds_cfg)
    pb_cfg = _build_pier_bridge_config(SCENARIO["mode"], SCENARIO["length"], ds_cfg)

    rows: list[dict[str, Any]] = []
    waypoint_details: dict[str, list[dict[str, Any]]] = {}
    segment_summaries: dict[str, dict[str, Optional[float]]] = {}

    for run in RUNS:
        overrides = _build_overrides(route_shape=run["route_shape"], smoothed=run["smoothed"])
        t0 = time.perf_counter()
        result = generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id=SCENARIO["seed_track_id"],
            anchor_seed_ids=list(SCENARIO["anchor_seed_ids"]),
            mode=SCENARIO["mode"],
            length=int(SCENARIO["length"]),
            random_seed=1337,
            overrides=overrides,
            dry_run=True,
            artist_playlist=True,
            artist_style_enabled=False,
            pier_bridge_config=pb_cfg,
        )
        runtime = time.perf_counter() - t0

        track_ids = list(result.track_ids or [])
        tracklist_hash = _hash_tracklist(track_ids) if track_ids else None

        audit_path = result.playlist_stats.get("playlist", {}).get("audit_path")
        audit_text = ""
        if audit_path:
            audit_file = Path(audit_path)
            if audit_file.exists():
                audit_text = audit_file.read_text(encoding="utf-8", errors="replace")

        pool_counts = _parse_pool_counts_by_segment(audit_text) if audit_text else []
        pool_metrics = _aggregate_pool_counts(pool_counts)
        overlap_summary = _segment_summary(pool_counts, "pool_overlap_jaccard")
        segment_summaries[run["label"]] = overlap_summary

        segments, _ = sweep._segment_positions(track_ids, list(SCENARIO["anchor_seed_ids"]))
        indices, _ = sweep._track_ids_to_indices(track_ids, posthoc_ctx.bundle)
        raw_metrics = sweep._compute_raw_sonic_metrics(indices or [], segments, posthoc_ctx)
        pacing_metrics = sweep._compute_pacing_metrics(indices or [], segments, posthoc_ctx)
        genre_metrics = sweep._compute_genre_metrics(indices or [], segments, posthoc_ctx)

        waypoint_details[run["label"]] = _extract_segment_waypoints(result.playlist_stats)
        ladder_warnings = _extract_ladder_warnings(result.playlist_stats)

        rows.append(
            {
                "label": run["label"],
                "route_shape": run["route_shape"],
                "smoothed": run["smoothed"],
                "tracklist_hash": tracklist_hash,
                "waypoint_vector_mode": _resolve_waypoint_mode(waypoint_details[run["label"]]),
                "bridge_raw_sonic_sim_mean": raw_metrics.get("bridge_raw_sonic_sim_mean"),
                "bridge_raw_sonic_sim_min": raw_metrics.get("bridge_raw_sonic_sim_min"),
                "p90_arc_dev": pacing_metrics.get("p90_arc_dev"),
                "max_jump": pacing_metrics.get("max_jump"),
                "genre_target_sim_mean": genre_metrics.get("genre_target_sim_mean"),
                "genre_target_delta_mean": genre_metrics.get("genre_target_delta_mean"),
                "runtime_s": float(runtime),
                "pool_overlap_jaccard": pool_metrics.get("pool_overlap_jaccard"),
                "chosen_from_local_count": pool_metrics.get("chosen_from_local_count"),
                "chosen_from_genre_count": pool_metrics.get("chosen_from_genre_count"),
                "chosen_from_toward_count": pool_metrics.get("chosen_from_toward_count"),
                "ladder_warnings": ladder_warnings,
            }
        )

    lines: list[str] = []
    lines.append("# DJ Ladder Route A/B (Indie Scenario)")
    lines.append("")
    lines.append("- variants: linear, ladder_onehot, ladder_smoothed")
    lines.append(f"- scenario: {SCENARIO['name']}")
    lines.append(f"- seed_track_id: {SCENARIO['seed_track_id']}")
    lines.append(f"- anchors: {', '.join(SCENARIO['anchor_seed_ids'])}")
    lines.append(f"- length: {SCENARIO['length']}")
    lines.append("")
    lines.append("| label | route_shape | waypoint_vector_mode | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | chosen_from_local_count | chosen_from_genre_count | chosen_from_toward_count | pool_overlap_jaccard |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {label} | {route_shape} | {waypoint_vector_mode} | `{tracklist_hash}` | {bridge_raw_sonic_sim_mean:.4f} | {bridge_raw_sonic_sim_min:.4f} | {p90_arc_dev:.4f} | {max_jump:.4f} | {chosen_from_local_count} | {chosen_from_genre_count} | {chosen_from_toward_count} | {pool_overlap_jaccard} |".format(
                label=row["label"],
                route_shape=str(row["route_shape"]),
                waypoint_vector_mode=str(row.get("waypoint_vector_mode") or "na"),
                tracklist_hash=row.get("tracklist_hash") or "missing",
                bridge_raw_sonic_sim_mean=float(row.get("bridge_raw_sonic_sim_mean") or 0.0),
                bridge_raw_sonic_sim_min=float(row.get("bridge_raw_sonic_sim_min") or 0.0),
                p90_arc_dev=float(row.get("p90_arc_dev") or 0.0),
                max_jump=float(row.get("max_jump") or 0.0),
                chosen_from_local_count=(
                    "{:.0f}".format(row["chosen_from_local_count"])
                    if row.get("chosen_from_local_count") is not None
                    else "na"
                ),
                chosen_from_genre_count=(
                    "{:.0f}".format(row["chosen_from_genre_count"])
                    if row.get("chosen_from_genre_count") is not None
                    else "na"
                ),
                chosen_from_toward_count=(
                    "{:.0f}".format(row["chosen_from_toward_count"])
                    if row.get("chosen_from_toward_count") is not None
                    else "na"
                ),
                pool_overlap_jaccard=(
                    "{:.4f}".format(row["pool_overlap_jaccard"])
                    if row.get("pool_overlap_jaccard") is not None
                    else "na"
                ),
            )
        )

    lines.append("")
    lines.append("## Waypoint Labels By Segment")
    for label, segments in waypoint_details.items():
        lines.append("")
        lines.append(f"### {label}")
        if not segments:
            lines.append("- no segment diagnostics available")
            continue
        for seg in segments:
            lines.append(
                "- segment {segment}: route_shape={route_shape} mode={mode} waypoint_count={count} labels={labels}".format(
                    segment=seg.get("segment"),
                    route_shape=seg.get("route_shape"),
                    mode=seg.get("ladder_waypoint_vector_mode"),
                    count=seg.get("ladder_waypoint_count"),
                    labels=", ".join(seg.get("ladder_waypoint_labels") or []),
                )
            )
            stats = seg.get("ladder_waypoint_vector_stats") or []
            if stats:
                preview = []
                for entry in stats[:3]:
                    top_labels = entry.get("top_labels") or []
                    top_parts = []
                    for item in top_labels[:3]:
                        if not isinstance(item, dict):
                            continue
                        weight = item.get("weight")
                        try:
                            weight_str = "{:.2f}".format(float(weight))
                        except Exception:
                            weight_str = "na"
                        top_parts.append(f"{item.get('label')}:{weight_str}")
                    top_preview = ", ".join(top_parts)
                    preview.append(f"{entry.get('label')}[{top_preview}]")
                lines.append(f"  - smoothed_top3: {', '.join(preview)}")

    lines.append("")
    lines.append("## Ladder Warnings")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['label']}")
        warnings = row.get("ladder_warnings") or []
        if not warnings:
            lines.append("- none")
            continue
        for entry in warnings[:10]:
            entry_type = entry.get("type") if isinstance(entry, dict) else "unknown"
            msg = entry.get("message") if isinstance(entry, dict) else ""
            lines.append(f"- {entry_type}: {msg}")

    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
