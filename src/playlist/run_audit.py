from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class RunAuditConfig:
    enabled: bool = False
    out_dir: str = "docs/run_audits"
    include_top_k: int = 25
    max_bytes: int = 350000
    write_on_success: bool = True
    write_on_failure: bool = True


@dataclass(frozen=True)
class InfeasibleHandlingConfig:
    enabled: bool = False
    strategy: str = "backoff"
    min_bridge_floor: float = 0.0
    backoff_steps: tuple[float, ...] = ()
    max_attempts_per_segment: int = 8
    widen_search_on_backoff: bool = True
    extra_neighbors_m: int = 200
    extra_bridge_helpers: int = 100
    extra_beam_width: int = 50
    extra_expansion_attempts: int = 2


@dataclass(frozen=True)
class RunAuditContext:
    timestamp_utc: str
    run_id: str
    ds_mode: str
    seed_track_id: str
    seed_artist: Optional[str]
    dry_run: bool
    artifact_path: str
    sonic_variant: Optional[str]
    allowed_ids_count: int
    pool_source: Optional[str] = None
    artist_style_enabled: bool = False
    artist_playlist: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunAuditEvent:
    kind: str
    ts_utc: str
    payload: dict[str, Any] = field(default_factory=dict)


def parse_run_audit_config(raw: Any) -> RunAuditConfig:
    if not isinstance(raw, dict):
        return RunAuditConfig()
    return RunAuditConfig(
        enabled=bool(raw.get("enabled", False)),
        out_dir=str(raw.get("out_dir", "docs/run_audits")),
        include_top_k=int(raw.get("include_top_k", 25)),
        max_bytes=int(raw.get("max_bytes", 350000)),
        write_on_success=bool(raw.get("write_on_success", True)),
        write_on_failure=bool(raw.get("write_on_failure", True)),
    )


def parse_infeasible_handling_config(raw: Any) -> InfeasibleHandlingConfig:
    if not isinstance(raw, dict):
        return InfeasibleHandlingConfig()
    steps_raw = raw.get("backoff_steps", ())
    steps: list[float] = []
    if isinstance(steps_raw, (list, tuple)):
        for v in steps_raw:
            if isinstance(v, (int, float)):
                steps.append(float(v))
    return InfeasibleHandlingConfig(
        enabled=bool(raw.get("enabled", False)),
        strategy=str(raw.get("strategy", "backoff")),
        min_bridge_floor=float(raw.get("min_bridge_floor", 0.0)),
        backoff_steps=tuple(steps),
        max_attempts_per_segment=int(raw.get("max_attempts_per_segment", 8)),
        widen_search_on_backoff=bool(raw.get("widen_search_on_backoff", True)),
        extra_neighbors_m=int(raw.get("extra_neighbors_m", 200)),
        extra_bridge_helpers=int(raw.get("extra_bridge_helpers", 100)),
        extra_beam_width=int(raw.get("extra_beam_width", 50)),
        extra_expansion_attempts=int(raw.get("extra_expansion_attempts", 2)),
    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dist_summary(values: Iterable[float]) -> dict[str, Optional[float]]:
    import numpy as np

    vals = [float(v) for v in values if v is not None]  # type: ignore[comparison-overlap]
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return {"min": None, "p05": None, "p50": None, "p95": None, "max": None}
    arr = np.array(vals, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _fmt_dist(name: str, dist: dict[str, Optional[float]]) -> str:
    def f(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    return f"- {name}: min={f(dist['min'])} p05={f(dist['p05'])} p50={f(dist['p50'])} p95={f(dist['p95'])} max={f(dist['max'])}"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), indent=2, ensure_ascii=False, sort_keys=True)


def write_markdown_report(
    *,
    context: RunAuditContext,
    events: list[RunAuditEvent],
    path: str | Path,
    max_bytes: int,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# DS Run Audit: {context.run_id}")
    lines.append("")
    lines.append("## 1) Run Metadata")
    lines.append(f"- timestamp_utc: `{context.timestamp_utc}`")
    lines.append(f"- run_id: `{context.run_id}`")
    lines.append(f"- ds_mode: `{context.ds_mode}`")
    lines.append(f"- seed_track_id: `{context.seed_track_id}`")
    lines.append(f"- seed_artist: `{context.seed_artist or ''}`")
    lines.append(f"- dry_run: `{context.dry_run}`")
    lines.append(f"- artifact_path: `{context.artifact_path}`")
    lines.append(f"- sonic_variant: `{context.sonic_variant or ''}`")
    lines.append(f"- allowed_ids_count: `{context.allowed_ids_count}`")
    if context.pool_source is not None:
        lines.append(f"- pool_source: `{context.pool_source}`")
    lines.append(f"- artist_style_enabled: `{context.artist_style_enabled}`")   
    lines.append(f"- artist_playlist: `{context.artist_playlist}`")
    if context.extra:
        lines.append("")
        lines.append("### extra")
        lines.append("```json")
        lines.append(_safe_json(context.extra))
        lines.append("```")

    preflight = next((e for e in events if e.kind == "preflight"), None)
    if preflight is not None:
        lines.append("")
        lines.append("## 2) Effective Tuning")
        tuning = preflight.payload.get("tuning", {})
        sources = preflight.payload.get("tuning_sources", {})
        lines.append("### values")
        lines.append("```json")
        lines.append(_safe_json(tuning))
        lines.append("```")
        lines.append("### override sources")
        lines.append("```json")
        lines.append(_safe_json(sources))
        lines.append("```")

        lines.append("")
        lines.append("## 3) Pool / Gating Summary")
        pool_summary = preflight.payload.get("pool_summary", {})
        lines.append("```json")
        lines.append(_safe_json(pool_summary))
        lines.append("```")

        recency = preflight.payload.get("recency", {})
        ds_inputs = preflight.payload.get("ds_inputs", {})
        if isinstance(recency, dict) or isinstance(ds_inputs, dict):
            lines.append("")
            lines.append("## 3b) Recency (Pre-Order)")
            lines.append("```json")
            lines.append(_safe_json({"ds_inputs": ds_inputs, "recency": recency}))
            lines.append("```")

        style_summary = preflight.payload.get("style_summary")
        if isinstance(style_summary, dict):
            lines.append("")
            lines.append("### style_aware_summary")
            lines.append("```json")
            lines.append(_safe_json(style_summary))
            lines.append("```")

    # Group segment attempts
    segment_attempts: dict[int, list[RunAuditEvent]] = {}
    segment_results: dict[int, RunAuditEvent] = {}
    for ev in events:
        if ev.kind == "segment_attempt":
            seg = int(ev.payload.get("segment_index", -1))
            segment_attempts.setdefault(seg, []).append(ev)
        if ev.kind in {"segment_success", "segment_failure"}:
            seg = int(ev.payload.get("segment_index", -1))
            segment_results[seg] = ev

    seg_ids = sorted(k for k in segment_attempts.keys() if k >= 0)
    if seg_ids:
        lines.append("")
        lines.append("## 4) Segment Diagnostics")

    for seg in seg_ids:
        attempts = sorted(segment_attempts.get(seg, []), key=lambda e: int(e.payload.get("attempt_number", 0)))
        header = attempts[0].payload.get("segment_header") if attempts else None
        lines.append("")
        lines.append(f"### Segment {seg}{f' — {header}' if header else ''}")
        for att in attempts:
            attempt_no = int(att.payload.get("attempt_number", 0))
            bridge_floor = att.payload.get("bridge_floor")
            widened = bool(att.payload.get("widened", False))
            lines.append("")
            lines.append(f"#### Attempt {attempt_no} (bridge_floor={bridge_floor} widened={widened})")

            seg_strategy = att.payload.get("segment_pool_strategy")
            seg_pool_max = att.payload.get("segment_pool_max")
            beam_width = att.payload.get("beam_width")
            expansion_attempts = att.payload.get("expansion_attempts")
            if any(v is not None for v in [seg_strategy, seg_pool_max, beam_width, expansion_attempts]):
                lines.append("")
                lines.append(f"- segment_pool_strategy: `{seg_strategy}`")
                lines.append(f"- segment_pool_max: `{seg_pool_max}`")
                lines.append(f"- beam_width: `{beam_width}`")
                lines.append(f"- expansion_attempts: `{expansion_attempts}`")

            pool_counts = att.payload.get("pool_counts", {})
            lines.append("")
            lines.append("**pool_counts**")
            lines.append("```json")
            lines.append(_safe_json(pool_counts))
            lines.append("```")

            dists = att.payload.get("distributions", {})
            if isinstance(dists, dict) and dists:
                lines.append("")
                lines.append("**distributions**")
                for name, dist in dists.items():
                    if isinstance(dist, dict):
                        lines.append(_fmt_dist(str(name), dist))

            penalty = att.payload.get("soft_genre_penalty", {})
            if isinstance(penalty, dict) and penalty:
                lines.append("")
                lines.append("**soft_genre_penalty**")
                lines.append("```json")
                lines.append(_safe_json(penalty))
                lines.append("```")

            reason = att.payload.get("reason")
            if reason:
                lines.append("")
                lines.append(f"**reason**: {reason}")

            top = att.payload.get("top_candidates", [])
            if isinstance(top, list) and top:
                lines.append("")
                lines.append(f"**top_candidates (top {len(top)})**")
                lines.append("")
                sample = top[0] if isinstance(top[0], dict) else {}
                has_identity = isinstance(sample, dict) and (
                    "artist_key" in sample or "title_key" in sample or "progress_t" in sample
                )
                if has_identity:
                    lines.append(
                        "| rank | track_id | artist | title | artist_key | title_key | progress_t | simA | simB | bridge_sim | T_min | G_min | final | internal |"
                    )
                    lines.append(
                        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                    )
                    for i, row in enumerate(top, 1):
                        bridge_sim = row.get("bridge_sim", row.get("hmean", ""))
                        lines.append(
                            "| {rank} | `{track_id}` | {artist} | {title} | {artist_key} | {title_key} | {progress_t} | {simA} | {simB} | {bridge_sim} | {tmin} | {gmin} | {final} | {internal} |".format(
                                rank=i,
                                track_id=str(row.get("track_id", "")),
                                artist=str(row.get("artist", "")).replace("|", "\\|"),
                                title=str(row.get("title", "")).replace("|", "\\|"),
                                artist_key=str(row.get("artist_key", "")).replace("|", "\\|"),
                                title_key=str(row.get("title_key", "")).replace("|", "\\|"),
                                progress_t=str(row.get("progress_t", "")),
                                simA=str(row.get("simA", "")),
                                simB=str(row.get("simB", "")),
                                bridge_sim=str(bridge_sim),
                                tmin=str(row.get("T_min", "")),
                                gmin=str(row.get("G_min", "")),
                                final=str(row.get("final", "")),
                                internal=str(row.get("internal", False)),
                            )
                        )
                else:
                    lines.append("| rank | track_id | artist | title | simA | simB | hmean | T_min | G_min | final | internal |")
                    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                    for i, row in enumerate(top, 1):
                        lines.append(
                            "| {rank} | `{track_id}` | {artist} | {title} | {simA} | {simB} | {hmean} | {tmin} | {gmin} | {final} | {internal} |".format(
                                rank=i,
                                track_id=str(row.get("track_id", "")),
                                artist=str(row.get("artist", "")).replace("|", "\\|"),
                                title=str(row.get("title", "")).replace("|", "\\|"),
                                simA=str(row.get("simA", "")),
                                simB=str(row.get("simB", "")),
                                hmean=str(row.get("hmean", "")),
                                tmin=str(row.get("T_min", "")),
                                gmin=str(row.get("G_min", "")),
                                final=str(row.get("final", "")),
                                internal=str(row.get("internal", False)),
                            )
                        )

        res = segment_results.get(seg)
        if res is not None:
            lines.append("")
            lines.append(f"**segment_result**: `{res.kind}`")
            if res.payload.get("bridge_floor_used") is not None:
                lines.append(f"- bridge_floor_used: `{res.payload.get('bridge_floor_used')}`")
            if res.payload.get("failure_reason"):
                lines.append(f"- failure_reason: {res.payload.get('failure_reason')}")

    final = next((e for e in events if e.kind in {"final_success", "final_failure"}), None)
    if final is not None:
        lines.append("")
        lines.append(f"## 5) Final Result ({final.kind})")
        if final.kind == "final_success":
            pov = final.payload.get("post_order_validation")
            pof = final.payload.get("post_order_filters_applied")
            tracks = final.payload.get("playlist_tracks", [])
            if isinstance(tracks, list) and tracks:
                lines.append("")
                lines.append("### tracklist")
                lines.append("")
                lines.append("| pos | track_id | artist | title |")
                lines.append("| ---: | --- | --- | --- |")
                for i, t in enumerate(tracks, 1):
                    lines.append(
                        "| {pos} | `{track_id}` | {artist} | {title} |".format(
                            pos=i,
                            track_id=str(t.get("track_id", "")),
                            artist=str(t.get("artist", "")).replace("|", "\\|"),
                            title=str(t.get("title", "")).replace("|", "\\|"),
                        )
                    )
            weakest = final.payload.get("weakest_edges", [])
            if isinstance(weakest, list) and weakest:
                lines.append("")
                lines.append("### weakest_edges")
                lines.append("")
                lines.append("| rank | prev_id | cur_id | T | S | G |")
                lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
                for i, e in enumerate(weakest, 1):
                    lines.append(
                        "| {rank} | `{prev}` | `{cur}` | {t} | {s} | {g} |".format(
                            rank=i,
                            prev=str(e.get("prev_id", "")),
                            cur=str(e.get("cur_id", "")),
                            t=str(e.get("T", "")),
                            s=str(e.get("S", "")),
                            g=str(e.get("G", "")),
                        )
                    )
            summary = final.payload.get("summary_stats", {})
            if isinstance(summary, dict) and summary:
                lines.append("")
                lines.append("### summary_stats")
                lines.append("```json")
                lines.append(_safe_json(summary))
                lines.append("```")

            if isinstance(pov, dict) or isinstance(pof, list):
                lines.append("")
                lines.append("### post_order_validation")
                lines.append("```json")
                lines.append(_safe_json({"post_order_filters_applied": pof, "post_order_validation": pov}))
                lines.append("```")
        else:
            lines.append("")
            lines.append("```json")
            lines.append(_safe_json(final.payload))
            lines.append("```")

    md = "\n".join(lines) + "\n"
    raw = md.encode("utf-8", errors="replace")
    if max_bytes and len(raw) > int(max_bytes):
        suffix = "\n\n[TRUNCATED: report exceeded max_bytes]\n"
        keep = max(0, int(max_bytes) - len(suffix.encode("utf-8")))
        md = raw[:keep].decode("utf-8", errors="ignore") + suffix

    out_path.write_text(md, encoding="utf-8")
    return out_path
