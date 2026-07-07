import logging
import sqlite3
from pathlib import Path

import scripts.analyze_library as analyze
from src.logging_utils import ProgressLogger


def _write_config(tmp_path: Path, db_path: Path, analyze_log_dir: Path | None = None) -> Path:
    music_dir = tmp_path / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    text = (
        "library:\n"
        f"  database_path: {db_path}\n"
        f"  music_directory: {music_dir}\n"
        "openai:\n"
        "  api_key: test-key\n"
    )
    if analyze_log_dir is not None:
        text += (
            "logging:\n"
            "  analyze_logs:\n"
            "    enabled: true\n"
            f"    dir: {analyze_log_dir.as_posix()}\n"
            "    retention_days: 30\n"
            "    level: DEBUG\n"
        )
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _make_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            file_path TEXT,
            file_mtime_ns INTEGER,
            file_modified INTEGER,
            musicbrainz_id TEXT,
            mbid_status TEXT,
            sonic_features TEXT,
            sonic_failed_at INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def test_analyze_logging_plan_and_summary(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    config_path = _write_config(tmp_path, db_path)
    out_dir = tmp_path / "artifacts"

    def _stub_scan(ctx):
        if getattr(ctx["args"], "progress", True):
            prog = ProgressLogger(
                logging.getLogger("analyze_library.stub_scan"),
                total=2,
                label="stub_scan",
                interval_s=0,
                every_n=1,
                verbose_each=False,
            )
            prog.update(detail="a")
            prog.update(detail="b")
            prog.finish()
        return {"total": 2, "scan_total": 2, "orphaned": {}}

    def _stub_verify(ctx):
        return {"skipped": False, "issues": []}

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan, "verify": _stub_verify})

    args = analyze.parse_args(
        [
            "--config",
            str(config_path),
            "--db-path",
            str(db_path),
            "--stages",
            "scan,verify",
            "--out-dir",
            str(out_dir),
            "--log-file",
            str(tmp_path / "analyze.log"),
        ]
    )

    caplog.set_level(logging.INFO)
    analyze.run_pipeline(args)

    text = caplog.text
    assert "Analyze run start" in text
    assert "run_id=" in text
    assert "stage=scan" in text and "decision=" in text and "reason=" in text
    assert "stage=verify" in text
    assert "RUN RECAP" in text


def test_analyze_verbose_enables_per_item_debug(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    config_path = _write_config(tmp_path, db_path)
    out_dir = tmp_path / "artifacts"

    def _stub_scan(ctx):
        prog = ProgressLogger(
            logging.getLogger("analyze_library.stub_verbose"),
            total=3,
            label="verbose_scan",
            interval_s=0,
            every_n=1,
            verbose_each=bool(getattr(ctx["args"], "verbose", False)),
        )
        prog.update(detail="one")
        prog.update(detail="two")
        prog.update(detail="three")
        prog.finish()
        return {"total": 3, "scan_total": 3, "orphaned": {}}

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan})

    args = analyze.parse_args(
        [
            "--config",
            str(config_path),
            "--db-path",
            str(db_path),
            "--stages",
            "scan",
            "--out-dir",
            str(out_dir),
            "--verbose",
            "--log-file",
            str(tmp_path / "analyze_verbose.log"),
        ]
    )

    caplog.set_level(logging.DEBUG)
    analyze.run_pipeline(args)

    debug_lines = [rec.message for rec in caplog.records if rec.levelno == logging.DEBUG and "verbose_scan item" in rec.message]
    assert len(debug_lines) >= 3


def test_scan_modified_breakdown_in_report(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    config_path = _write_config(tmp_path, db_path)
    out_dir = tmp_path / "artifacts"

    def _stub_scan(ctx):
        return {
            "total": 1,
            "scan_total": 1,
            "orphaned": {},
            "modified_reasons": {"stat_changed": 1, "new_not_in_db": 0},
            "modified_examples": {"stat_changed": ["song.mp3"]},
        }

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan})

    args = analyze.parse_args(
        [
            "--config",
            str(config_path),
            "--db-path",
            str(db_path),
            "--stages",
            "scan",
            "--out-dir",
            str(out_dir),
            "--log-file",
            str(tmp_path / "analyze_scan.log"),
        ]
    )

    caplog.set_level(logging.INFO)
    analyze.run_pipeline(args)

    report_path = out_dir / "analyze_run_report.json"
    data = report_path.read_text(encoding="utf-8")
    assert "modified_reasons" in data
    assert "stat_changed" in data
    assert "scan modified breakdown" in caplog.text


def test_run_pipeline_can_suppress_console_logging_for_worker_stdout(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    config_path = _write_config(tmp_path, db_path)
    out_dir = tmp_path / "artifacts"
    log_path = tmp_path / "analyze_worker.log"

    def _stub_scan(ctx):
        return {"total": 1, "scan_total": 1, "orphaned": {}}

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan})

    args = analyze.parse_args(
        [
            "--config",
            str(config_path),
            "--db-path",
            str(db_path),
            "--stages",
            "scan",
            "--out-dir",
            str(out_dir),
            "--log-file",
            str(log_path),
        ]
    )

    analyze.run_pipeline(args, console_logging=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Analyze run start" in log_path.read_text(encoding="utf-8")


def test_stage_muq_logs_model_load_and_heartbeat(tmp_path, monkeypatch, caplog):
    import numpy as np
    import src.analyze.muq_runner as muq_runner
    import src.analyze.track_paths as track_paths

    config_path = _write_config(tmp_path, tmp_path / "metadata.db")
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        track_paths, "load_paths",
        lambda db_path: {"t1": "/a", "t2": "/b", "t3": "/c"},
    )
    monkeypatch.setattr(
        muq_runner, "build_muq_embedder",
        lambda device="cpu", torch_threads=0: (lambda p: np.ones(4, np.float32)),
    )

    args = analyze.parse_args(["--config", str(config_path)])
    args.force = False
    args.limit = None
    args.progress = True
    args.progress_interval = 0.0   # force a heartbeat on every item
    args.progress_every = 1
    args.verbose = False

    ctx = {
        "config_path": str(config_path),
        "args": args,
        "out_dir": out_dir,
        "db_path": str(tmp_path / "metadata.db"),
    }

    caplog.set_level(logging.INFO, logger="analyze_library")
    result = analyze.stage_muq(ctx)

    assert result["ok"] == 3
    assert "MuQ-MuLan loaded in" in caplog.text
    assert "muq:" in caplog.text          # per-item heartbeat summary
    assert "muq complete" in caplog.text  # ProgressLogger.finish() line


def test_run_pipeline_checks_cancellation_between_stages(tmp_path, monkeypatch):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    config_path = _write_config(tmp_path, db_path)
    out_dir = tmp_path / "artifacts"
    stages_run = []

    class CancelledByTest(Exception):
        pass

    def _stub_scan(ctx):
        stages_run.append("scan")
        return {"total": 1, "scan_total": 1, "orphaned": {}}

    def _stub_verify(ctx):
        stages_run.append("verify")
        return {"skipped": False, "issues": []}

    checks = {"count": 0}

    def _cancel_after_first_stage():
        checks["count"] += 1
        if checks["count"] > 2:
            raise CancelledByTest()

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan, "verify": _stub_verify})

    args = analyze.parse_args(
        [
            "--config",
            str(config_path),
            "--db-path",
            str(db_path),
            "--stages",
            "scan,verify",
            "--out-dir",
            str(out_dir),
            "--log-file",
            str(tmp_path / "analyze_cancel.log"),
        ]
    )

    try:
        analyze.run_pipeline(args, cancellation_check=_cancel_after_first_stage)
    except CancelledByTest:
        pass
    else:
        raise AssertionError("Expected cancellation to stop the pipeline")

    assert stages_run == ["scan"]


def test_run_pipeline_writes_per_run_analyze_log(tmp_path, monkeypatch):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    alog_dir = tmp_path / "logs_analyze"
    config_path = _write_config(tmp_path, db_path, analyze_log_dir=alog_dir)
    out_dir = tmp_path / "artifacts"

    def _stub_scan(ctx):
        return {"total": 1, "scan_total": 1, "orphaned": {}}

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan})

    args = analyze.parse_args(
        [
            "--config", str(config_path),
            "--db-path", str(db_path),
            "--stages", "scan",
            "--out-dir", str(out_dir),
        ]
    )  # NOTE: no --log-file -> per-run path is used

    analyze.run_pipeline(args, console_logging=False)

    logs = list(alog_dir.glob("*.log"))
    assert len(logs) == 1
    assert logs[0].name.endswith(".log")
    assert "Analyze run start" in logs[0].read_text(encoding="utf-8")
