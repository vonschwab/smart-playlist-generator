import sqlite3
import yaml
from pathlib import Path

from src.playlist_gui.diagnostics.checks import run_checks


def test_run_checks_with_temp_db(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    db_path = tmp_path / "metadata.db"
    # write config
    cfg = {
        "library": {"database_path": str(db_path)},
        "playlists": {"ds_pipeline": {"artifact_path": str(tmp_path / "artifact.npz")}},
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    # create db with tracks table
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tracks (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO tracks(id) VALUES (1)")
    conn.commit()
    conn.close()

    # create artifact file
    (tmp_path / "artifact.npz").write_text("x")

    results = run_checks(str(cfg_path), None)
    assert any(r.name == "config" and r.ok for r in results)
    assert any(r.name == "database" and r.ok for r in results)
    assert any(r.name == "artifacts" and r.ok for r in results)
