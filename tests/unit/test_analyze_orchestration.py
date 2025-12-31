import json
import sqlite3
from pathlib import Path

import numpy as np
import scripts.analyze_library as analyze


def _make_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = {
        "library": {
            "database_path": str(db_path),
            "music_directory": str(tmp_path / "music"),
        },
        "openai": {"api_key": "test-key"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "library:\n"
        f"  database_path: {cfg['library']['database_path']}\n"
        f"  music_directory: {cfg['library']['music_directory']}\n"
        "openai:\n"
        f"  api_key: {cfg['openai']['api_key']}\n",
        encoding="utf-8",
    )
    return config_path


def _make_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tracks (
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
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute(
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, artist TEXT, title TEXT, discogs_status TEXT)"
    )
    conn.execute(
        "INSERT INTO tracks (track_id, file_path, file_mtime_ns, file_modified, sonic_features) VALUES (?,?,?,?,?)",
        ("t1", "path1", 1, 1, "{}"),
    )
    conn.execute("INSERT INTO track_genres VALUES (?,?,?)", ("t1", "rock", "file"))
    conn.commit()
    return conn


def _stub_stage_genre_sim(ctx):
    return {"skipped": False}


def _stub_stage_artifacts(ctx):
    out_dir = ctx["out_dir"]
    out_path = out_dir / "data_matrices_step1.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        track_ids=np.array(["t1"]),
        artist_keys=np.array(["a1"]),
        track_artists=np.array(["a1"]),
        track_titles=np.array(["title"]),
        X_sonic=np.zeros((1, 2)),
        X_genre_raw=np.zeros((1, 1)),
        X_genre_smoothed=np.zeros((1, 1)),
        genre_vocab=np.array(["rock"]),
        X_sonic_raw=np.zeros((1, 2)),
    )
    fp = analyze.compute_stage_fingerprint(ctx, "artifacts")
    manifest = analyze._write_artifact_manifest(out_dir, fp, ctx.get("config_hash", ""), {})
    return {"path": str(out_path), "fingerprint": fp, "manifest": str(manifest), "skipped": False}


def test_stage_gating_and_manifest(tmp_path, monkeypatch):
    db_path = tmp_path / "metadata.db"
    conn = _make_db(db_path)
    conn.close()
    config_path = _make_config(tmp_path, db_path)
    out_dir = tmp_path / "artifacts"

    stub_map = {
        "genre-sim": _stub_stage_genre_sim,
        "artifacts": _stub_stage_artifacts,
        "verify": analyze.stage_verify,
    }
    monkeypatch.setattr(analyze, "STAGE_FUNCS", stub_map)

    args = analyze.parse_args(
        [
            "--config",
            str(config_path),
            "--db-path",
            str(db_path),
            "--stages",
            "genre-sim,artifacts,verify",
            "--out-dir",
            str(out_dir),
        ]
    )

    # First run: stages run
    rc1 = analyze.run_pipeline(args)
    assert rc1 == 0
    report_path = out_dir / "analyze_run_report.json"
    report1 = json.loads(report_path.read_text(encoding="utf-8"))
    assert report1["stages"]["genre-sim"]["decision"] == "ran"
    assert report1["stages"]["artifacts"]["decision"] == "ran"
    assert (out_dir / "artifact_manifest.json").exists()

    # Second run: unchanged fingerprints → skip
    rc2 = analyze.run_pipeline(args)
    assert rc2 == 0
    report2 = json.loads(report_path.read_text(encoding="utf-8"))
    assert report2["stages"]["genre-sim"]["decision"] == "skipped"
    assert report2["stages"]["artifacts"]["decision"] == "skipped"
    assert report2["stages"]["verify"]["decision"] == "skipped"

    # Change inputs: new genre row should flip fingerprints and rerun
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO track_genres VALUES (?,?,?)", ("t1", "pop", "file"))
    conn.commit()
    conn.close()

    rc3 = analyze.run_pipeline(args)
    assert rc3 == 0
    report3 = json.loads(report_path.read_text(encoding="utf-8"))
    assert report3["stages"]["genre-sim"]["decision"] == "ran"
    assert report3["stages"]["artifacts"]["decision"] == "ran"
