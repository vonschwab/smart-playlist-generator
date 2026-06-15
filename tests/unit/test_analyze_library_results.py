"""Tests for Analyze Library result parsing and summaries."""

import json

from src.playlist.analyze_library_results import (
    build_analyze_library_readout,
    format_analyze_library_action_list,
    format_analyze_library_attention_summary,
    format_analyze_library_summary,
    parse_analyze_library_paused,
    parse_analyze_library_report,
    parse_analyze_library_stage_progress,
)


def test_parse_analyze_library_paused_returns_stage_and_reason(tmp_path):
    report_path = tmp_path / "analyze_run_report.json"
    report_path.write_text(
        json.dumps(
            {
                "paused": True,
                "paused_stage": "enrich",
                "stages": {
                    "enrich": {"decision": "paused", "reason": "Claude rate window"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert parse_analyze_library_paused(report_path) == ("enrich", "Claude rate window")


def test_parse_analyze_library_paused_none_for_completed_run(tmp_path):
    report_path = tmp_path / "analyze_run_report.json"
    report_path.write_text(
        json.dumps({"stages": {"scan": {"decision": "ran"}}}), encoding="utf-8"
    )
    assert parse_analyze_library_paused(report_path) is None
    # Missing file is treated as "not paused", never an exception.
    assert parse_analyze_library_paused(tmp_path / "nope.json") is None


def test_parse_analyze_library_report_builds_stage_summary(tmp_path):
    report_path = tmp_path / "analyze_run_report.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "out_dir": "data/artifacts/beat3tower_32k",
                "total_duration_sec": 31.25,
                "stages": {
                    "scan": {
                        "decision": "ran",
                        "processed_count": 20,
                        "errors_count": 0,
                        "duration_sec": 1.5,
                        "throughput": 13.3,
                    },
                    "genres": {
                        "decision": "skipped",
                        "reason": "fingerprint_unchanged",
                        "duration_sec": 0.1,
                        "processed_count": 0,
                        "errors_count": 0,
                    },
                    "verify": {
                        "decision": "ran",
                        "duration_sec": 0.5,
                        "processed_count": None,
                        "errors_count": 2,
                        "result": {"issues": ["missing_manifest", "stale_artifact"]},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = parse_analyze_library_report(report_path)

    assert result["report_path"] == str(report_path)
    assert result["total_stages"] == 3
    assert result["ran_stages"] == 2
    assert result["skipped_stages"] == 1
    assert result["errors_count"] == 2
    assert result["verify_issues"] == ["missing_manifest", "stale_artifact"]
    assert result["stages"][0]["name"] == "scan"
    assert result["stages"][0]["processed_count"] == 20


def test_format_analyze_library_summary_is_human_readable():
    summary = format_analyze_library_summary(
        {
            "total_stages": 7,
            "ran_stages": 3,
            "skipped_stages": 4,
            "verify_issues": [],
            "total_duration_sec": 31.25,
        }
    )

    assert summary == "7 stages: 3 ran, 4 skipped, 0 verify issues, 31.2s"


def test_format_analyze_library_action_list_uses_gui_labels():
    stage_text = format_analyze_library_action_list(
        ["sonic", "genre-sim", "artifacts"]
    )

    assert (
        stage_text
        == "Update sonic features -> Build genre similarity -> Build DS artifacts"
    )


def test_build_analyze_library_readout_flags_attention_items():
    readout = build_analyze_library_readout(
        {
            "summary": "3 stages: 2 ran, 1 skipped, 2 verify issues, 31.2s",
            "total_stages": 3,
            "ran_stages": 2,
            "skipped_stages": 1,
            "errors_count": 2,
            "verify_issues": ["missing_manifest", "stale_artifact"],
            "total_duration_sec": 31.2,
        },
        status="SUCCESS",
    )

    assert readout["headline"] == "3 stages: 2 ran, 1 skipped, 2 verify issues, 31.2s"
    assert ("Verify issues", "2") in readout["metrics"]
    assert "missing_manifest" in readout["attention"][0]


def test_build_analyze_library_readout_handles_failed_without_report_data():
    readout = build_analyze_library_readout(
        {},
        status="FAILED",
        error_message="Worker exited unexpectedly",
    )

    assert readout["headline"] == "Analyze Library failed"
    assert ("Status", "Failed") in readout["metrics"]
    assert "Worker exited unexpectedly" in readout["attention"]


def test_format_analyze_library_attention_summary_prioritizes_verify_issues():
    summary = format_analyze_library_attention_summary(
        {
            "verify_issues": ["missing_manifest", "stale_artifact", "bad_counts"],
            "errors_count": 0,
        },
        status="SUCCESS",
    )

    assert (
        summary
        == "Needs Attention: 3 verify issues - missing_manifest; stale_artifact; +1 more"
    )


def test_format_analyze_library_attention_summary_handles_crash_message():
    summary = format_analyze_library_attention_summary(
        {},
        status="FAILED",
        error_message="Worker exited unexpectedly",
    )

    assert summary == "Needs Attention: Worker exited unexpectedly"


def test_parse_analyze_library_stage_progress_from_cli_log():
    progress = parse_analyze_library_stage_progress(
        "run_id=abc | stage=sonic | decision=ran | reason=required | pending=42",
        ["scan", "genres", "discogs", "sonic", "genre-sim", "artifacts", "verify"],
    )

    assert progress == {
        "stage": "sonic",
        "current": 4,
        "total": 7,
        "detail": "Stage 4/7 - Sonic: running (required; pending 42)",
    }


def test_parse_analyze_library_stage_progress_marks_completion():
    progress = parse_analyze_library_stage_progress(
        "run_id=abc | stage=verify | decision=ran | reason=required | processed=- | elapsed_s=0.50 | throughput=- | errors=0",
        ["scan", "genres", "discogs", "sonic", "genre-sim", "artifacts", "verify"],
    )

    assert progress == {
        "stage": "verify",
        "current": 7,
        "total": 7,
        "detail": "Stage 7/7 - Verify: finished (ran; errors 0; elapsed 0.50s)",
    }


def test_parse_analyze_library_stage_progress_formats_skips():
    progress = parse_analyze_library_stage_progress(
        "run_id=abc | stage=genre-sim | decision=skipped | reason=fingerprint_same | pending=unknown",
        ["scan", "genres", "discogs", "sonic", "genre-sim", "artifacts", "verify"],
    )

    assert progress == {
        "stage": "genre-sim",
        "current": 5,
        "total": 7,
        "detail": "Stage 5/7 - Genre Similarity: skipped (fingerprint_same; pending unknown)",
    }
