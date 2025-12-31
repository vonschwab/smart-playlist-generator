import logging
import sqlite3
from pathlib import Path

import scripts.analyze_library as analyze
from src.logging_utils import ProgressLogger


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    music_dir = tmp_path / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "library:\n"
        f"  database_path: {db_path}\n"
        f"  music_directory: {music_dir}\n"
        "openai:\n"
        "  api_key: test-key\n",
        encoding="utf-8",
    )
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
