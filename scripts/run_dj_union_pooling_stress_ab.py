#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

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
OUT_DOC = REPO_ROOT / "docs" / "diagnostics" / "dj_union_pooling_stress_ab.md"

SCENARIOS = [
    {
        "name": "dj_stress_charli",
        "seed_track_id": "97bf21610ba410fba544940267ebe0c5",  # Charli XCX - forever
        "anchor_seed_ids": [
            "97bf21610ba410fba544940267ebe0c5",  # Charli XCX - forever
            "5a33b78a22826c00be01634ffbf22d9e",  # Solange - My Skin My Logo
            "e8dd7b07d07a1ded4c95983db39aacec",  # Channel Tres - Topdown
            "5352c2182087b0ec214332f85772557a",  # Jessy Lanza - Anyone Around
        ],
        "length": 30,
        "mode": "dynamic",
        "out_doc": OUT_DOC,
    },
    {
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
        "out_doc": REPO_ROOT / "docs" / "diagnostics" / "dj_union_pooling_stress_ab_indie.md",
    },
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


def _build_overrides(
    *,
    waypoint_weight: float,
    pooling_strategy: str,
    segment_pool_max: int,
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
        "segment_pool_max": int(segment_pool_max),
        "max_segment_pool_max": int(segment_pool_max),
        "dj_bridging": {
            "enabled": True,
            "route_shape": "linear",
            "waypoint_weight": float(waypoint_weight),
            "pooling": {
                "strategy": str(pooling_strategy),
                "debug_compare_baseline": True,
            },
        },
    }
    return {"pier_bridge": pier_bridge}


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


def _run_scenario(
    scenario: dict[str, Any],
    *,
    sweep,
    posthoc_ctx,
    pb_cfg: PierBridgeConfig,
    artifact_path: str,
    stress_pool_max: int,
) -> None:
    runs = [
        {"label": "A_baseline_low", "pooling": "baseline", "waypoint_weight": 0.05},
        {"label": "A_baseline_high", "pooling": "baseline", "waypoint_weight": 0.30},
        {"label": "B_union_low", "pooling": "dj_union", "waypoint_weight": 0.05},
        {"label": "B_union_high", "pooling": "dj_union", "waypoint_weight": 0.30},
    ]
    rows: list[dict[str, Any]] = []
    segment_summaries: dict[str, dict[str, Any]] = {}

    for run in runs:
        overrides = _build_overrides(
            waypoint_weight=float(run["waypoint_weight"]),
            pooling_strategy=str(run["pooling"]),
            segment_pool_max=int(stress_pool_max),
        )
        t0 = time.perf_counter()
        result = generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id=scenario["seed_track_id"],
            anchor_seed_ids=list(scenario["anchor_seed_ids"]),
            mode=scenario["mode"],
            length=int(scenario["length"]),
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
        local_summary = _segment_summary(pool_counts, "chosen_from_local_count")
        toward_summary = _segment_summary(pool_counts, "chosen_from_toward_count")
        genre_summary = _segment_summary(pool_counts, "chosen_from_genre_count")
        segment_summaries[run["label"]] = {
            "overlap": overlap_summary,
            "local": local_summary,
            "toward": toward_summary,
            "genre": genre_summary,
        }

        segments, _ = sweep._segment_positions(track_ids, list(scenario["anchor_seed_ids"]))
        indices, _ = sweep._track_ids_to_indices(track_ids, posthoc_ctx.bundle)
        raw_metrics = sweep._compute_raw_sonic_metrics(indices or [], segments, posthoc_ctx)
        pacing_metrics = sweep._compute_pacing_metrics(indices or [], segments, posthoc_ctx)
        genre_metrics = sweep._compute_genre_metrics(indices or [], segments, posthoc_ctx)

        rows.append(
            {
                "label": run["label"],
                "pooling": run["pooling"],
                "waypoint_weight": run["waypoint_weight"],
                "segment_pool_max": stress_pool_max,
                "tracklist_hash": tracklist_hash,
                "bridge_raw_sonic_sim_mean": raw_metrics.get("bridge_raw_sonic_sim_mean"),
                "bridge_raw_sonic_sim_min": raw_metrics.get("bridge_raw_sonic_sim_min"),
                "p90_arc_dev": pacing_metrics.get("p90_arc_dev"),
                "max_jump": pacing_metrics.get("max_jump"),
                "genre_target_sim_mean": genre_metrics.get("genre_target_sim_mean"),
                "genre_target_delta_mean": genre_metrics.get("genre_target_delta_mean"),
                "runtime_s": float(runtime),
                "pool_overlap_jaccard": pool_metrics.get("pool_overlap_jaccard"),
                "union_only_count": pool_metrics.get("pool_overlap_union_only"),
                "chosen_from_local_count": pool_metrics.get("chosen_from_local_count"),
                "chosen_from_genre_count": pool_metrics.get("chosen_from_genre_count"),
                "chosen_from_toward_count": pool_metrics.get("chosen_from_toward_count"),
            }
        )

    lines: list[str] = []
    lines.append("# DJ Union Pooling Stress A/B")
    lines.append("")
    lines.append(f"- scenario: {scenario['name']}")
    lines.append(f"- seed_track_id: {scenario['seed_track_id']}")
    lines.append(f"- anchors: {', '.join(scenario['anchor_seed_ids'])}")
    lines.append(f"- length: {scenario['length']}")
    lines.append(f"- stress_segment_pool_max: {stress_pool_max}")
    lines.append("")
    lines.append("| label | pooling | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | genre_target_sim_mean | genre_target_delta_mean | runtime_s | pool_overlap_jaccard | union_only_count | chosen_from_local_count | chosen_from_genre_count | chosen_from_toward_count |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {label} | {pooling} | {waypoint_weight:.2f} | `{tracklist_hash}` | {bridge_raw_sonic_sim_mean:.4f} | {bridge_raw_sonic_sim_min:.4f} | {p90_arc_dev:.4f} | {max_jump:.4f} | {genre_target_sim_mean:.4f} | {genre_target_delta_mean:.4f} | {runtime_s:.2f} | {pool_overlap_jaccard} | {union_only_count} | {chosen_from_local_count} | {chosen_from_genre_count} | {chosen_from_toward_count} |".format(
                label=row["label"],
                pooling=row["pooling"],
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
                union_only_count=(
                    "{:.1f}".format(row["union_only_count"])
                    if row.get("union_only_count") is not None
                    else "na"
                ),
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
            )
        )
    lines.append("")
    lines.append("## Per-Segment Summary (min/median/max)")
    for label, summary in segment_summaries.items():
        overlap = summary["overlap"]
        local = summary["local"]
        toward = summary["toward"]
        genre = summary["genre"]
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
            "- chosen_from_local_count: min={min:.1f} median={median:.1f} max={max:.1f}".format(
                min=local["min"] or 0.0,
                median=local["median"] or 0.0,
                max=local["max"] or 0.0,
            )
        )
        lines.append(
            "- chosen_from_toward_count: min={min:.1f} median={median:.1f} max={max:.1f}".format(
                min=toward["min"] or 0.0,
                median=toward["median"] or 0.0,
                max=toward["max"] or 0.0,
            )
        )
        lines.append(
            "- chosen_from_genre_count: min={min:.1f} median={median:.1f} max={max:.1f}".format(
                min=genre["min"] or 0.0,
                median=genre["median"] or 0.0,
                max=genre["max"] or 0.0,
            )
        )

    scenario["out_doc"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = _load_sweep_helpers()
    cfg = Config()
    ds_cfg = cfg.get("playlists", "ds_pipeline", default={}) or {}
    artifact_path = ds_cfg.get("artifact_path")
    if not artifact_path:
        raise SystemExit("Missing playlists.ds_pipeline.artifact_path in config.yaml")
    stress_pool_max = 80
    for scenario in SCENARIOS:
        posthoc_ctx = sweep._build_posthoc_context(str(artifact_path), ds_cfg)
        pb_cfg = _build_pier_bridge_config(scenario["mode"], scenario["length"], ds_cfg)
        _run_scenario(
            scenario,
            sweep=sweep,
            posthoc_ctx=posthoc_ctx,
            pb_cfg=pb_cfg,
            artifact_path=str(artifact_path),
            stress_pool_max=stress_pool_max,
        )


if __name__ == "__main__":
    main()
