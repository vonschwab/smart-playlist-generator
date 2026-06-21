"""Helpers for parsing and summarizing Analyze Library reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ANALYZE_LIBRARY_ACTION_LABELS: dict[str, str] = {
    "scan": "Scan library",
    "genres": "Update genres",
    "discogs": "Update Discogs genres",
    "sonic": "Update sonic features",
    "genre-sim": "Build genre similarity",
    "artifacts": "Build DS artifacts",
    "energy": "Energy scan",
    "genre-embedding": "Build genre embedding",
    "verify": "Verify outputs",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_analyze_library_report(report_path: str | Path) -> dict[str, Any]:
    """Parse ``analyze_run_report.json`` into compact GUI-friendly data."""
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    stages_payload = report.get("stages") or {}
    stages: list[dict[str, Any]] = []
    ran_stages = 0
    skipped_stages = 0
    errors_count = 0
    verify_issues: list[str] = []

    for name, stage_report in stages_payload.items():
        stage_report = stage_report or {}
        decision = str(stage_report.get("decision") or "-")
        if decision in {"ran", "forced"}:
            ran_stages += 1
        elif decision == "skipped":
            skipped_stages += 1

        stage_errors = _safe_int(stage_report.get("errors_count"), 0)
        errors_count += stage_errors

        result = stage_report.get("result") or {}
        issues = result.get("issues") if isinstance(result, dict) else None
        if name == "verify" and isinstance(issues, list):
            verify_issues = [str(issue) for issue in issues]

        stages.append(
            {
                "name": name,
                "decision": decision,
                "reason": stage_report.get("reason") or "",
                "duration_sec": _safe_float(stage_report.get("duration_sec"), 0.0),
                "processed_count": stage_report.get("processed_count"),
                "pending_estimate": stage_report.get("pending_estimate"),
                "errors_count": stage_errors,
                "throughput": stage_report.get("throughput"),
            }
        )

    result: dict[str, Any] = {
        "run_id": report.get("run_id") or "",
        "report_path": str(path),
        "out_dir": str(report.get("out_dir") or ""),
        "total_duration_sec": _safe_float(report.get("total_duration_sec"), 0.0),
        "total_stages": len(stages),
        "ran_stages": ran_stages,
        "skipped_stages": skipped_stages,
        "errors_count": errors_count,
        "verify_issues": verify_issues,
        "stages": stages,
    }
    result["summary"] = format_analyze_library_summary(result)
    return result


def parse_analyze_library_paused(report_path: str | Path) -> tuple[str, str] | None:
    """If the run stopped at a resumable checkpoint, return (stage, reason).

    ``run_pipeline`` sets ``report["paused"]`` and records the stage's decision as
    ``"paused"`` (e.g. enrich hit a Claude rate window). Returns None for a normal
    completed run.
    """
    path = Path(report_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None
    if not report.get("paused"):
        return None
    stage = str(report.get("paused_stage") or "")
    stage_report = (report.get("stages") or {}).get(stage) or {}
    reason = str(stage_report.get("reason") or "resumable checkpoint")
    return stage, reason


def format_analyze_library_summary(result: dict[str, Any]) -> str:
    """Return a concise one-line Analyze Library summary for tables/status bars."""
    total = _safe_int(result.get("total_stages"), 0)
    ran = _safe_int(result.get("ran_stages"), 0)
    skipped = _safe_int(result.get("skipped_stages"), 0)
    verify_issues = result.get("verify_issues") or []
    issue_count = len(verify_issues) if isinstance(verify_issues, list) else 0
    duration = _safe_float(result.get("total_duration_sec"), 0.0)
    return f"{total} stages: {ran} ran, {skipped} skipped, {issue_count} verify issues, {duration:.1f}s"


def format_analyze_library_action_label(stage: str) -> str:
    """Return the user-facing label for an Analyze Library stage action."""
    return ANALYZE_LIBRARY_ACTION_LABELS.get(stage, _format_stage_label(stage))


def format_analyze_library_action_list(stages: Sequence[str]) -> str:
    """Return a readable stage sequence for GUI previews, logs, and status."""
    return " -> ".join(format_analyze_library_action_label(stage) for stage in stages)


def build_analyze_library_readout(
    result: dict[str, Any],
    *,
    status: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Build a compact, GUI-friendly Analyze Library readout."""
    result = result or {}
    status_text = str(status or "").replace("_", " ").title() or "Unknown"
    failed = status_text.lower() == "failed"
    summary = str(result.get("summary") or "").strip()
    headline = "Analyze Library failed" if failed else summary
    if not headline:
        headline = "Analyze Library complete" if status_text.lower() == "success" else "Analyze Library status"

    verify_issues = result.get("verify_issues") or []
    if not isinstance(verify_issues, list):
        verify_issues = [str(verify_issues)]
    errors_count = _safe_int(result.get("errors_count"), 0)

    metrics: list[tuple[str, str]] = [("Status", status_text)]
    total_stages = result.get("total_stages")
    if total_stages is not None:
        metrics.append(("Stages", str(_safe_int(total_stages))))
    if result.get("ran_stages") is not None:
        metrics.append(("Ran", str(_safe_int(result.get("ran_stages")))))
    if result.get("skipped_stages") is not None:
        metrics.append(("Skipped", str(_safe_int(result.get("skipped_stages")))))
    metrics.append(("Verify issues", str(len(verify_issues))))
    metrics.append(("Errors", str(errors_count)))
    if result.get("total_duration_sec") is not None:
        metrics.append(("Duration", f"{_safe_float(result.get('total_duration_sec')):.1f}s"))

    attention: list[str] = []
    if error_message:
        attention.append(str(error_message))
    attention.extend(str(issue) for issue in verify_issues)
    if errors_count and not verify_issues:
        attention.append(f"{errors_count} stage errors reported")

    return {
        "headline": headline,
        "metrics": metrics,
        "attention": attention,
    }


def format_analyze_library_attention_summary(
    result: dict[str, Any],
    *,
    status: str = "",
    error_message: str = "",
    max_items: int = 2,
) -> str:
    """Return a concise Needs Attention line for Analyze Library GUI surfaces."""
    result = result or {}
    readout = build_analyze_library_readout(
        result,
        status=status,
        error_message=error_message,
    )
    attention = [str(item) for item in readout.get("attention") or [] if str(item)]
    if not attention:
        return ""

    verify_issues = result.get("verify_issues") or []
    if not isinstance(verify_issues, list):
        verify_issues = [str(verify_issues)]

    shown = attention[:max(1, max_items)]
    detail = "; ".join(shown)
    remaining = len(attention) - len(shown)
    if remaining > 0:
        detail += f"; +{remaining} more"

    if verify_issues:
        issue_word = "issue" if len(verify_issues) == 1 else "issues"
        return f"Needs Attention: {len(verify_issues)} verify {issue_word} - {detail}"
    return f"Needs Attention: {detail}"


def _format_stage_label(stage: str) -> str:
    labels = {
        "genre-sim": "Genre Similarity",
    }
    return labels.get(stage, stage.replace("-", " ").replace("_", " ").title())


def parse_analyze_library_stage_progress(
    message: str,
    stage_order: list[str],
) -> dict[str, Any] | None:
    """Parse structured Analyze Library stage log lines into GUI progress."""
    if "stage=" not in message or "decision=" not in message:
        return None

    fields: dict[str, str] = {}
    for part in message.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()

    stage = fields.get("stage")
    decision = fields.get("decision")
    if not stage or not decision or stage not in stage_order:
        return None

    index = stage_order.index(stage) + 1
    total = len(stage_order)
    reason = fields.get("reason") or decision
    label = _format_stage_label(stage)

    if "processed" in fields or "elapsed_s" in fields:
        parts = [decision]
        processed = fields.get("processed")
        if processed and processed != "-":
            parts.append(f"processed {processed}")
        errors = fields.get("errors")
        if errors is not None:
            parts.append(f"errors {errors}")
        elapsed = fields.get("elapsed_s")
        if elapsed and elapsed != "-":
            parts.append(f"elapsed {elapsed}s")
        detail = f"Stage {index}/{total} - {label}: finished ({'; '.join(parts)})"
    else:
        state = "skipped" if decision == "skipped" else "running"
        parts = [reason]
        pending = fields.get("pending")
        if pending:
            parts.append(f"pending {pending}")
        detail = f"Stage {index}/{total} - {label}: {state} ({'; '.join(parts)})"

    return {
        "stage": stage,
        "current": index,
        "total": total,
        "detail": detail,
    }
