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
OUT_DOC = REPO_ROOT / "docs" / "diagnostics" / "dj_connector_bias_ab_indie.md"

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


def _build_overrides(
    *,
    waypoint_weight: float,
    connectors_enabled: bool,
) -> dict[str, Any]:
    pier_bridge: dict[str, Any] = {
        "audit_run": {
            "enabled": True,
            "out_dir": str(AUDIT_DIR),
            "include_top_k": 25,
            "max_bytes": 350000,
            "write_on_success": True,
            "write_on_failure": True,
        },
        "dj_bridging": {
            "enabled": True,
            "route_shape": "linear",
            "waypoint_weight": float(waypoint_weight),
            "pooling": {
                "strategy": "dj_union",
                "debug_compare_baseline": True,
            },
            "connectors": {
                "enabled": bool(connectors_enabled),
                "max_connectors": 3,
                "use_only_when_far": True,
                "far_threshold": 0.50,
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
            "dj_connectors_injected_count": None,
            "dj_connectors_chosen_count": None,
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
        "dj_connectors_injected_count": _sum("dj_connectors_injected_count"),
        "dj_connectors_chosen_count": _sum("dj_connectors_chosen_count"),
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


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = _load_sweep_helpers()
    cfg = Config()
    ds_cfg = cfg.get("playlists", "ds_pipeline", default={}) or {}
    artifact_path = ds_cfg.get("artifact_path")
    if not artifact_path:
        raise SystemExit("Missing playlists.ds_pipeline.artifact_path in config.yaml")
    posthoc_ctx = sweep._build_posthoc_context(str(artifact_path), ds_cfg)
    pb_cfg = _build_pier_bridge_config(SCENARIO["mode"], SCENARIO["length"], ds_cfg)

    runs = [
        {"label": "A_connectors_off_low", "connectors": False, "waypoint_weight": 0.05},
        {"label": "A_connectors_off_high", "connectors": False, "waypoint_weight": 0.30},
        {"label": "B_connectors_on_low", "connectors": True, "waypoint_weight": 0.05},
        {"label": "B_connectors_on_high", "connectors": True, "waypoint_weight": 0.30},
    ]

    rows: list[dict[str, Any]] = []
    segment_summaries: dict[str, dict[str, Any]] = {}
    for run in runs:
        overrides = _build_overrides(
            waypoint_weight=float(run["waypoint_weight"]),
            connectors_enabled=bool(run["connectors"]),
        )
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
        injected_summary = _segment_summary(pool_counts, "dj_connectors_injected_count")
        chosen_summary = _segment_summary(pool_counts, "dj_connectors_chosen_count")
        segment_summaries[run["label"]] = {
            "overlap": overlap_summary,
            "injected": injected_summary,
            "chosen": chosen_summary,
        }

        segments, _ = sweep._segment_positions(track_ids, list(SCENARIO["anchor_seed_ids"]))
        indices, _ = sweep._track_ids_to_indices(track_ids, posthoc_ctx.bundle)
        raw_metrics = sweep._compute_raw_sonic_metrics(indices or [], segments, posthoc_ctx)
        pacing_metrics = sweep._compute_pacing_metrics(indices or [], segments, posthoc_ctx)
        genre_metrics = sweep._compute_genre_metrics(indices or [], segments, posthoc_ctx)

        rows.append(
            {
                "label": run["label"],
                "connectors": run["connectors"],
                "waypoint_weight": run["waypoint_weight"],
                "tracklist_hash": tracklist_hash,
                "bridge_raw_sonic_sim_mean": raw_metrics.get("bridge_raw_sonic_sim_mean"),
                "bridge_raw_sonic_sim_min": raw_metrics.get("bridge_raw_sonic_sim_min"),
                "p90_arc_dev": pacing_metrics.get("p90_arc_dev"),
                "max_jump": pacing_metrics.get("max_jump"),
                "genre_target_sim_mean": genre_metrics.get("genre_target_sim_mean"),
                "genre_target_delta_mean": genre_metrics.get("genre_target_delta_mean"),
                "runtime_s": float(runtime),
                "pool_overlap_jaccard": pool_metrics.get("pool_overlap_jaccard"),
                "dj_connectors_injected_count": pool_metrics.get("dj_connectors_injected_count"),
                "dj_connectors_chosen_count": pool_metrics.get("dj_connectors_chosen_count"),
            }
        )

    lines: list[str] = []
    lines.append("# DJ Connector Bias A/B (Indie Scenario)")
    lines.append("")
    lines.append(f"- scenario: {SCENARIO['name']}")
    lines.append(f"- seed_track_id: {SCENARIO['seed_track_id']}")
    lines.append(f"- anchors: {', '.join(SCENARIO['anchor_seed_ids'])}")
    lines.append(f"- length: {SCENARIO['length']}")
    lines.append("")
    lines.append("| label | connectors | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | genre_target_sim_mean | genre_target_delta_mean | runtime_s | pool_overlap_jaccard | connectors_injected_count | connectors_chosen_count |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {label} | {connectors} | {waypoint_weight:.2f} | `{tracklist_hash}` | {bridge_raw_sonic_sim_mean:.4f} | {bridge_raw_sonic_sim_min:.4f} | {p90_arc_dev:.4f} | {max_jump:.4f} | {genre_target_sim_mean:.4f} | {genre_target_delta_mean:.4f} | {runtime_s:.2f} | {pool_overlap_jaccard} | {dj_connectors_injected_count} | {dj_connectors_chosen_count} |".format(
                label=row["label"],
                connectors=str(row["connectors"]).lower(),
                waypoint_weight=float(row["waypoint_weight"]),
                tracklist_hash=row.get("tracklist_hash") or "missing",
                bridge_raw_sonic_sim_mean=float(row.get("bridge_raw_sonic_sim_mean") or 0.0),
                bridge_raw_sonic_sim_min=float(row.get("bridge_raw_sonic_sim_min") or 0.0),
                p90_arc_dev=float(row.get("p90_arc_dev") or 0.0),
                max_jump=float(row.get("max_jump") or 0.0),
                genre_target_sim_mean=float(row.get("genre_target_sim_mean") or 0.0),
                genre_target_delta_mean=float(row.get("genre_target_delta_mean") or 0.0),
                runtime_s=float(row.get("runtime_s") or 0.0),
                pool_overlap_jaccard=(
                    "{:.4f}".format(row["pool_overlap_jaccard"])
                    if row.get("pool_overlap_jaccard") is not None
                    else "na"
                ),
                dj_connectors_injected_count=(
                    "{:.0f}".format(row["dj_connectors_injected_count"])
                    if row.get("dj_connectors_injected_count") is not None
                    else "na"
                ),
                dj_connectors_chosen_count=(
                    "{:.0f}".format(row["dj_connectors_chosen_count"])
                    if row.get("dj_connectors_chosen_count") is not None
                    else "na"
                ),
            )
        )
    lines.append("")
    lines.append("## Per-Segment Summary (min/median/max)")
    for label, summary in segment_summaries.items():
        overlap = summary["overlap"]
        injected = summary["injected"]
        chosen = summary["chosen"]
        lines.append("")
        lines.append(f"### {label}")
        lines.append(
            "- pool_overlap_jaccard: min={min:.4f} median={median:.4f} max={max:.4f}".format(
                min=overlap["min"] or 0.0,
                median=overlap["median"] or 0.0,
                max=overlap["max"] or 0.0,
            )
        )
        lines.append(
            "- connectors_injected_count: min={min:.1f} median={median:.1f} max={max:.1f}".format(
                min=injected["min"] or 0.0,
                median=injected["median"] or 0.0,
                max=injected["max"] or 0.0,
            )
        )
        lines.append(
            "- connectors_chosen_count: min={min:.1f} median={median:.1f} max={max:.1f}".format(
                min=chosen["min"] or 0.0,
                median=chosen["median"] or 0.0,
                max=chosen["max"] or 0.0,
            )
        )

    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
