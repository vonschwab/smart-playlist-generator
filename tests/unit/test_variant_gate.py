import yaml
from scripts.analyze_library import _variant_gate, _sonic_fold_settings

def _cfg(tmp_path, variant):
    p = tmp_path / "c.yaml"
    if variant is None:
        p.write_text("{}", encoding="utf-8")   # no override key at all
    else:
        p.write_text(yaml.safe_dump({"artifacts": {"sonic_variant_override": variant}}), encoding="utf-8")
    return str(p)

def test_active_variant_runs_inactive_skips(tmp_path):
    muq_cfg = _cfg(tmp_path, "muq")
    assert _variant_gate(muq_cfg, "muq") is None                # muq active -> muq runs
    assert _variant_gate(muq_cfg, "mert") is not None           # muq active -> mert skips
    mert_cfg = _cfg(tmp_path, "mert")
    assert _variant_gate(mert_cfg, "mert") is None
    assert _variant_gate(mert_cfg, "muq") is not None

def test_default_variant_is_muq(tmp_path):
    # No override key at all -> the active variant defaults to muq (SP-B).
    _, active = _sonic_fold_settings(_cfg(tmp_path, None))
    assert active == "muq"


def test_unknown_variant_warns_loudly(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "muq2"), "muq")   # a typo
    assert any("not a recognized sonic variant" in r.getMessage() for r in caplog.records)


def test_known_variant_does_not_warn(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "muq"), "muq")
    assert not any("not a recognized" in r.getMessage() for r in caplog.records)


def test_tower_weighted_override_now_warns(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "tower_weighted"), "muq")
    assert any("not a recognized sonic variant" in r.getMessage() for r in caplog.records)


def _write_artifact_and_db(tmp_path, track_ids, variant_stamp):
    """Minimal artifact + metadata.db pair for exercising stage_verify directly."""
    import sqlite3

    import numpy as np

    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "data_matrices_step1.npz",
        track_ids=np.array(track_ids, dtype=object),
        artist_keys=np.array(["a"] * len(track_ids)),
        track_artists=np.array(["a"] * len(track_ids)),
        track_titles=np.array(["x"] * len(track_ids)),
        X_sonic=np.zeros((len(track_ids), 3), np.float32),
        X_sonic_raw=np.zeros((len(track_ids), 3), np.float32),
        X_genre_raw=np.zeros((len(track_ids), 1), np.float32),
        X_genre_smoothed=np.zeros((len(track_ids), 1), np.float32),
        genre_vocab=np.array(["rock"]),
        X_sonic_variant=np.array(variant_stamp),
    )
    # The guard only checks existence of the ACTIVE variant's sidecar — an empty
    # stub is enough to trigger it (SP-B re-key: was mert_sidecar.npz).
    (out_dir / "muq_sidecar.npz").write_bytes(b"")

    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tracks (track_id TEXT, file_path TEXT)")
    for tid in track_ids:
        conn.execute("INSERT INTO tracks VALUES (?, ?)", (tid, "x"))
    conn.commit()
    conn.close()
    return out_dir, str(db_path)


def test_verify_flags_sonic_variant_mismatch(tmp_path):
    """verify guard: active variant (default muq) + its sidecar present, but the
    artifact stamped with a different variant, is a loud failure. Ported from the
    deleted tests/unit/test_analyze_mert_stage.py and re-keyed off muq_sidecar.npz
    (was mert_sidecar.npz) now that muq is the active/default sonic variant."""
    import sqlite3
    from argparse import Namespace

    from scripts.analyze_library import stage_verify

    track_ids = ["t1", "t2"]
    out_dir, db_path = _write_artifact_and_db(tmp_path, track_ids, "tower_weighted")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    args = Namespace(
        force=False, limit=None, dry_run=False, progress=False, verbose=False,
        progress_interval=15.0, progress_every=500, max_tracks=0, workers="auto",
    )
    ctx = {
        # No config file at all -> _sonic_fold_settings defaults to (True, "muq").
        "config_path": str(tmp_path / "config.yaml"),
        "db_path": db_path,
        "out_dir": out_dir,
        "args": args,
        "conn": conn,
        "config_hash": "test",
        "library_root": str(tmp_path),
        "genres_dirty": False, "sonic_dirty": False,
        "artifacts_dirty": False, "force_stage": False,
    }
    result = stage_verify(ctx)
    conn.close()

    assert "sonic_variant_mismatch" in result["issues"]
