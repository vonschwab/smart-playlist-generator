"""Unit tests for the analyze_library MERT stage (learned sonic embedding extraction)."""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf

import scripts.analyze_library as al
from scripts.extract_mert_sidecar import ShardStore

EMB_DIM = 768


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _write_wav(path: Path, seconds: float = 2.0, sr: int = 24000) -> str:
    """A tiny real WAV in tmp_path (test fixture — not a library file)."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(path, y, sr)
    return str(path)


def _metadata_db(tmp_path: Path, track_files: dict[str, str]) -> str:
    """Minimal metadata.db with a tracks table mapping track_id -> file_path."""
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,"
        " album_id TEXT, title TEXT, file_path TEXT)"
    )
    for tid, fp in track_files.items():
        conn.execute(
            "INSERT INTO tracks VALUES (?, 'Artist', 'Album', 'alb1', ?, ?)",
            (tid, tid, fp),
        )
    conn.commit()
    conn.close()
    return str(db)


def _write_artifact(out_dir: Path, track_ids: list[str]) -> Path:
    """Fake production artifact npz carrying only track_ids (all the stage reads)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data_matrices_step1.npz"
    np.savez(path, track_ids=np.array(track_ids, dtype=object))
    return path


def _ctx(tmp_path: Path, db_path: str, out_dir: Path, **arg_overrides):
    """Minimal stage ctx mirroring run_pipeline's construction."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    defaults = dict(
        force=False, limit=None, dry_run=False, progress=False, verbose=False,
        progress_interval=15.0, progress_every=500, max_tracks=0, workers="auto",
    )
    defaults.update(arg_overrides)
    args = Namespace(**defaults)
    return {
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


class _CountingEmbedder:
    """Fake embedder honoring the Callable[[np.ndarray], np.ndarray] interface."""

    def __init__(self):
        self.calls = 0

    def __call__(self, y: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full(EMB_DIM, 0.5, dtype=np.float32)


def _install_fake_embedder(monkeypatch) -> _CountingEmbedder:
    emb = _CountingEmbedder()
    monkeypatch.setattr(al, "_build_mert_embedder", lambda device, torch_threads: emb)
    return emb


def _seed_done(out_dir: Path, track_ids: list[str]) -> None:
    """Pre-populate manifest + a shard so `track_ids` count as already extracted."""
    store = ShardStore(out_dir / "mert_shards", shard_size=len(track_ids) or 1)
    zero = np.zeros(EMB_DIM, np.float32)
    for tid in track_ids:
        store.add(tid, zero, zero, zero)
    store.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Stage behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_mert_stage_skips_when_manifest_complete(tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts"
    _write_artifact(out_dir, ["t1", "t2"])
    _seed_done(out_dir, ["t1", "t2"])
    db = _metadata_db(tmp_path, {"t1": "x", "t2": "x"})

    def explode(device, torch_threads):
        raise AssertionError("embedder must not be built when nothing is pending")

    monkeypatch.setattr(al, "_build_mert_embedder", explode)

    ctx = _ctx(tmp_path, db, out_dir)
    result = al.stage_mert(ctx)
    ctx["conn"].close()

    assert result["skipped"] is True
    assert result["pending"] == 0


def test_mert_stage_processes_pending_tracks(tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts"
    _write_artifact(out_dir, ["t1", "t2", "t3"])
    _seed_done(out_dir, ["t1"])  # t1 already extracted
    wav2 = _write_wav(tmp_path / "t2.wav")
    wav3 = _write_wav(tmp_path / "t3.wav")
    db = _metadata_db(tmp_path, {"t1": "ignored", "t2": wav2, "t3": wav3})
    emb = _install_fake_embedder(monkeypatch)

    ctx = _ctx(tmp_path, db, out_dir)
    result = al.stage_mert(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["pending"] == 2
    assert result["ok"] == 2
    assert result["failed"] == 0
    # 2 s tracks use one replicated window -> exactly one embed call per track
    assert emb.calls == 2
    # merged sidecar contains all three tracks exactly once
    sidecar = np.load(result["sidecar"], allow_pickle=True)
    assert sorted(str(t) for t in sidecar["track_ids"]) == ["t1", "t2", "t3"]
    assert sidecar["emb_start"].shape == (3, EMB_DIM)


def test_mert_stage_respects_limit(tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts"
    _write_artifact(out_dir, ["t1", "t2", "t3"])
    files = {tid: _write_wav(tmp_path / f"{tid}.wav") for tid in ("t1", "t2", "t3")}
    db = _metadata_db(tmp_path, files)
    emb = _install_fake_embedder(monkeypatch)

    ctx = _ctx(tmp_path, db, out_dir, limit=1)
    result = al.stage_mert(ctx)
    ctx["conn"].close()

    assert result["pending"] == 1
    assert result["ok"] == 1
    assert emb.calls == 1


def test_mert_stage_force_re_extracts(tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts"
    _write_artifact(out_dir, ["t1", "t2"])
    _seed_done(out_dir, ["t1", "t2"])  # everything already in manifest
    files = {tid: _write_wav(tmp_path / f"{tid}.wav") for tid in ("t1", "t2")}
    db = _metadata_db(tmp_path, files)
    emb = _install_fake_embedder(monkeypatch)

    ctx = _ctx(tmp_path, db, out_dir, force=True)
    result = al.stage_mert(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["pending"] == 2
    assert emb.calls == 2


def test_mert_stage_failure_does_not_crash_run(tmp_path, monkeypatch):
    out_dir = tmp_path / "artifacts"
    _write_artifact(out_dir, ["good", "bad"])
    good = _write_wav(tmp_path / "good.wav")
    corrupt = tmp_path / "bad.wav"
    corrupt.write_text("this is not audio", encoding="utf-8")
    db = _metadata_db(tmp_path, {"good": good, "bad": str(corrupt)})
    _install_fake_embedder(monkeypatch)

    ctx = _ctx(tmp_path, db, out_dir)
    result = al.stage_mert(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["ok"] == 1
    assert result["failed"] == 1
    manifest = json.loads(
        (out_dir / "mert_shards" / "manifest.json").read_text(encoding="utf-8")
    )
    assert "bad" in manifest["failed"]
    assert "good" in manifest["done"]


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint / ordering
# ─────────────────────────────────────────────────────────────────────────────


def test_mert_fingerprint_changes_on_new_track(tmp_path):
    out_dir = tmp_path / "artifacts"
    _write_artifact(out_dir, ["t1", "t2"])
    db = _metadata_db(tmp_path, {"t1": "x", "t2": "x"})

    ctx = _ctx(tmp_path, db, out_dir)
    fp_before = al.compute_stage_fingerprint(ctx, "mert")
    _write_artifact(out_dir, ["t1", "t2", "t3"])  # new track lands in the artifact
    fp_after = al.compute_stage_fingerprint(ctx, "mert")
    ctx["conn"].close()

    assert fp_before != fp_after


def test_mert_stage_registered_after_sonic():
    order = al.STAGE_ORDER_DEFAULT
    assert "mert" in order
    assert "mert" in al.STAGE_FUNCS
    # sonic populates first; MERT is a separate sidecar consumed at fold time
    assert order.index("mert") == order.index("sonic") + 1
    assert order.index("mert") < order.index("artifacts")
