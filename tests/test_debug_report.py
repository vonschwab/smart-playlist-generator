from pathlib import Path

from src.playlist_gui.diagnostics.report import build_debug_report
from src.playlist_gui.diagnostics.checks import CheckResult


def test_debug_report_redacts_tokens(tmp_path: Path):
    log_file = tmp_path / "log.txt"
    log_file.write_text("token=ABC123\nok line\n", encoding="utf-8")

    report = build_debug_report(
        base_config_path="C:/Users/user/config.yaml",
        preset_label="Base",
        mode="Artist",
        artist="Secret Band",
        worker_status="running",
        last_job_summary="Scan done",
        last_job_error="",
        readiness=[CheckResult("config", True, "ok")],
        gui_log_path=log_file,
        worker_events=["token=XYZ"],
    )

    assert "ABC123" not in report
    assert "XYZ" not in report
    assert "Debug Report" in report


def test_debug_report_save_redacted(tmp_path: Path):
    log_file = tmp_path / "log.txt"
    log_file.write_text("Authorization: Bearer SECRET\n", encoding="utf-8")
    report = build_debug_report(
        base_config_path="C:/cfg.yaml",
        preset_label="Base",
        mode="Artist",
        artist="Singer",
        worker_status="running",
        last_job_summary="ok",
        last_job_error="token=123",
        readiness=[CheckResult("config", True, "ok")],
        gui_log_path=log_file,
        worker_events=["password=bad"],
    )
    out = tmp_path / "report.txt"
    out.write_text(report, encoding="utf-8")
    saved = out.read_text(encoding="utf-8")
    assert "SECRET" not in saved
    assert "token=123" not in saved
    assert "password=bad" not in saved
