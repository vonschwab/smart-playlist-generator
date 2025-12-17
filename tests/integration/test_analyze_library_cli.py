import json
from pathlib import Path

import pytest

import scripts.analyze_library as analyze


def write_min_config(tmp_path: Path, db_path: Path) -> Path:
    config = {
        "library": {"database_path": str(db_path)},
        "openai": {"api_key": "dummy", "model": "gpt-4o-mini"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(json.dumps(config))
    return cfg_path


def test_dry_run_prints_plan(capsys, tmp_path):
    cfg = write_min_config(tmp_path, tmp_path / "meta.db")
    analyze.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "Plan" in out
    assert "stages:" in out


def test_stage_order_monkeypatched(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db_path.touch()
    cfg = write_min_config(tmp_path, db_path)

    calls = []

    def make_stage(name):
        def _stage(ctx):
            calls.append(name)
            return {"ok": True}

        return _stage

    monkeypatch.setitem(analyze.STAGE_FUNCS, "genres", make_stage("genres"))
    monkeypatch.setitem(analyze.STAGE_FUNCS, "artifacts", make_stage("artifacts"))

    analyze.main(
        [
            "--config",
            str(cfg),
            "--stages",
            "genres,artifacts",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert calls == ["genres", "artifacts"]

