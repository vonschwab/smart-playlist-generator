#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import Config
from src.playlist.config import default_ds_config, resolve_pier_bridge_tuning
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from src.playlist.pier_bridge_builder import PierBridgeConfig
from src.similarity.sonic_variant import resolve_sonic_variant

OUT_DOC = REPO_ROOT / "docs" / "diagnostics" / "dj_relaxation_micro_pier_demo.md"

SCENARIO = {
    "name": "dj_stress_indie_ladder",
    "seed_track_id": "99cbf799b947abc1efadeff958fa3a86",
    "anchor_seed_ids": [
        "99cbf799b947abc1efadeff958fa3a86",
        "22d617435daa61958c67624fca3dbc23",
        "f4282083af4d8149e32c76c2eccea5bc",
        "7cf9096e6cc377f4d38d4af7d8bdca69",
        "b6bae06cc55006d73afa712edf0a28d2",
        "9cc4ace8edc409d5e94a08bfdf2a3615",
    ],
    "length": 30,
    "mode": "dynamic",
}

SCENARIOS = [
    {
        "label": "infeasible_by_design",
        "overrides": {
            "pier_bridge": {
                "segment_pool_max": 40,
                "max_segment_pool_max": 40,
                "bridge_floor_dynamic": 0.15,
                "dj_bridging": {
                    "enabled": True,
                    "route_shape": "ladder",
                    "relaxation": {
                        "enabled": True,
                        "max_attempts": 4,
                        "emit_warnings": True,
                        "allow_floor_relaxation": True,
                    },
                    "micro_piers": {
                        "enabled": True,
                        "candidate_source": "both",
                        "top_k": 10,
                        "max": 1,
                    },
                },
            }
        },
    },
    {
        "label": "metadata_missing",
        "overrides": {
            "pier_bridge": {
                "dj_bridging": {
                    "enabled": True,
                    "route_shape": "ladder",
                    "ladder": {
                        "min_label_weight": 0.9,
                    },
                    "relaxation": {
                        "enabled": True,
                        "max_attempts": 4,
                        "emit_warnings": True,
                        "allow_floor_relaxation": True,
                    },
                },
            }
        },
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


def _build_pier_bridge_config(mode: str, length: int, ds_cfg: dict[str, Any]) -> PierBridgeConfig:
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


def _extract_relaxation_summary(playlist_stats: dict[str, Any]) -> list[dict[str, Any]]:
    segments = playlist_stats.get("playlist", {}).get("segment_diagnostics") or []
    summaries: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        summaries.append(
            {
                "segment_index": idx,
                "relaxation_success_attempt": seg.get("relaxation_success_attempt"),
                "relaxation_attempts": seg.get("relaxation_attempts") or [],
                "micro_pier_used": seg.get("micro_pier_used"),
                "micro_pier_track_id": seg.get("micro_pier_track_id"),
            }
        )
    return summaries


def _extract_warnings(playlist_stats: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = playlist_stats.get("playlist", {}).get("warnings") or []
    out: list[dict[str, Any]] = []
    for entry in warnings:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") in {
            "dj_relaxation_attempts",
            "micro_pier_fallback",
            "micro_pier_used",
            "genre_missing",
            "genre_ladder_unavailable",
        }:
            out.append(entry)
    return out


def main() -> int:
    cfg = Config()
    ds_cfg = cfg.get("playlists", "ds_pipeline", default={}) or {}
    artifact_path = ds_cfg.get("artifact_path")
    if not artifact_path:
        raise ValueError("Missing playlists.ds_pipeline.artifact_path in config")

    sweep = _load_sweep_helpers()
    posthoc_ctx = sweep._build_posthoc_context(str(artifact_path), ds_cfg)
    pb_cfg = _build_pier_bridge_config(SCENARIO["mode"], SCENARIO["length"], ds_cfg)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        overrides = scenario["overrides"]
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

        segments, _ = sweep._segment_positions(track_ids, list(SCENARIO["anchor_seed_ids"]))
        indices, _ = sweep._track_ids_to_indices(track_ids, posthoc_ctx.bundle)
        raw_metrics = sweep._compute_raw_sonic_metrics(indices or [], segments, posthoc_ctx)
        pacing_metrics = sweep._compute_pacing_metrics(indices or [], segments, posthoc_ctx)

        rows.append(
            {
                "label": scenario["label"],
                "tracklist_hash": tracklist_hash,
                "bridge_raw_sonic_sim_mean": raw_metrics.get("bridge_raw_sonic_sim_mean"),
                "bridge_raw_sonic_sim_min": raw_metrics.get("bridge_raw_sonic_sim_min"),
                "p90_arc_dev": pacing_metrics.get("p90_arc_dev"),
                "max_jump": pacing_metrics.get("max_jump"),
                "runtime_s": float(runtime),
                "relaxation_summary": _extract_relaxation_summary(result.playlist_stats),
                "warnings": _extract_warnings(result.playlist_stats),
            }
        )

    lines: list[str] = []
    lines.append("# DJ Relaxation + Micro-Pier Demo")
    lines.append("")
    lines.append(f"- scenario: {SCENARIO['name']}")
    lines.append(f"- seed_track_id: {SCENARIO['seed_track_id']}")
    lines.append(f"- anchors: {', '.join(SCENARIO['anchor_seed_ids'])}")
    lines.append(f"- length: {SCENARIO['length']}")
    lines.append("")
    lines.append("| label | tracklist_hash | bridge_raw_sonic_sim_mean | bridge_raw_sonic_sim_min | p90_arc_dev | max_jump | runtime_s |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {label} | `{tracklist_hash}` | {bridge_raw_sonic_sim_mean:.4f} | {bridge_raw_sonic_sim_min:.4f} | {p90_arc_dev:.4f} | {max_jump:.4f} | {runtime_s:.2f} |".format(
                label=row["label"],
                tracklist_hash=row.get("tracklist_hash") or "missing",
                bridge_raw_sonic_sim_mean=float(row.get("bridge_raw_sonic_sim_mean") or 0.0),
                bridge_raw_sonic_sim_min=float(row.get("bridge_raw_sonic_sim_min") or 0.0),
                p90_arc_dev=float(row.get("p90_arc_dev") or 0.0),
                max_jump=float(row.get("max_jump") or 0.0),
                runtime_s=float(row.get("runtime_s") or 0.0),
            )
        )

    lines.append("")
    lines.append("## Relaxation Summary")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['label']}")
        segments = row.get("relaxation_summary") or []
        if not segments:
            lines.append("- no segment diagnostics")
            continue
        for seg in segments:
            lines.append(
                "- segment {idx}: success_attempt={attempt} micro_pier_used={micro}".format(
                    idx=seg.get("segment_index"),
                    attempt=seg.get("relaxation_success_attempt"),
                    micro=seg.get("micro_pier_used"),
                )
            )
            attempts = seg.get("relaxation_attempts") or []
            if attempts:
                first = attempts[0]
                lines.append(
                    "  - first_attempt: label={label} changes={changes}".format(
                        label=first.get("label"),
                        changes=", ".join(first.get("changes") or []),
                    )
                )

    lines.append("")
    lines.append("## Warnings")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['label']}")
        warnings = row.get("warnings") or []
        if not warnings:
            lines.append("- none")
            continue
        for entry in warnings[:10]:
            lines.append(f"- {entry.get('type')}: {entry.get('message')}")

    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
