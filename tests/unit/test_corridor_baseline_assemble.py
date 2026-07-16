import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cb_assemble", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "assemble_baseline.py")
assemble = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assemble)


# ---- to_posix ----------------------------------------------------------------

def test_to_posix_normalizes_windows_backslashes():
    assert assemble.to_posix(r"C:\Users\Dylan\Desktop\PG3_SAT2\docs\x.json") == \
        "C:/Users/Dylan/Desktop/PG3_SAT2/docs/x.json"


def test_to_posix_leaves_posix_paths_unchanged():
    assert assemble.to_posix("docs/corridor_baseline/x.json") == "docs/corridor_baseline/x.json"


# ---- sha256_file (small synthetic file -- never the real 507MB artifact) ----

def test_sha256_file_streaming_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "blob.bin"
    p.write_bytes(b"corridor baseline test bytes" * 1000)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert assemble.sha256_file(p, chunk_size=17) == expected  # odd chunk size exercises the loop boundary


# ---- read_json: missing-input failure ----------------------------------------

def test_read_json_missing_file_raises_naming_the_file(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match=r"does_not_exist\.json"):
        assemble.read_json(missing)


def test_read_json_reads_existing(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert assemble.read_json(p) == {"a": 1}


# ---- read_perturbation_docstring ----------------------------------------------

def test_read_perturbation_docstring_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        assemble.read_perturbation_docstring(tmp_path / "nope.py")


def test_read_perturbation_docstring_reads_module_docstring(tmp_path):
    p = tmp_path / "fake_perturb.py"
    p.write_text('"""Fake perturbation rules.\n\nbool -> flipped\n"""\nX = 1\n', encoding="utf-8")
    doc = assemble.read_perturbation_docstring(p)
    assert doc.startswith("Fake perturbation rules.")


def test_read_perturbation_docstring_no_docstring_raises(tmp_path):
    p = tmp_path / "no_doc.py"
    p.write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no module docstring"):
        assemble.read_perturbation_docstring(p)


def test_read_perturbation_docstring_real_file_is_readable():
    # Exercises the real perturb.py without importing it (ast-parse only).
    doc = assemble.read_perturbation_docstring()
    assert "Perturbation rules" in doc


# ---- assemble_meta (pure) -----------------------------------------------------

def _fake_artifact_info():
    return {"path": "data/artifacts/beat3tower_32k/data_matrices_step1.npz", "sha256": "deadbeef", "size_bytes": 123}


def _fake_db_info():
    return {"path": "data/metadata.db", "track_count": 43036}


def test_assemble_meta_shape():
    meta = assemble.assemble_meta(
        captured_on_commit="abc123",
        branch_tip="def456",
        captured_date="2026-07-16",
        artifact_info=_fake_artifact_info(),
        db_info=_fake_db_info(),
        corpus=["SADE", "Bill Evans Trio"],
        detents=["home", "open"],
        sweep_cells=[("Bill Evans Trio", "open"), ("Swirlies", "home")],
        perturbation_rules="bool -> flipped",
    )
    assert meta["captured_on_commit"] == "abc123"
    assert meta["branch_tip"] == "def456"
    assert meta["captured_date"] == "2026-07-16"
    assert meta["artifact"] == _fake_artifact_info()
    assert meta["db"] == _fake_db_info()
    assert meta["corpus"] == ["SADE", "Bill Evans Trio"]
    assert meta["detents"] == ["home", "open"]
    # tuples in sweep_cells become lists -- json.dumps would do this anyway,
    # but doing it in assemble_meta means the returned python dict already
    # matches the serialized shape (no surprise at json.dumps time).
    assert meta["sweep_cells"] == [["Bill Evans Trio", "open"], ["Swirlies", "home"]]
    assert meta["perturbation_rules"] == "bool -> flipped"


def test_assemble_meta_does_not_mutate_inputs():
    corpus = ["SADE"]
    sweep_cells = [("Bill Evans Trio", "open")]
    assemble.assemble_meta(
        captured_on_commit="a", branch_tip="b", captured_date="2026-07-16",
        artifact_info=_fake_artifact_info(), db_info=_fake_db_info(),
        corpus=corpus, detents=["home"], sweep_cells=sweep_cells, perturbation_rules="x",
    )
    assert corpus == ["SADE"]
    assert sweep_cells == [("Bill Evans Trio", "open")]


# ---- assemble_baseline (pure) --------------------------------------------------

def test_assemble_baseline_top_level_shape():
    meta = {"captured_on_commit": "abc"}
    baseline = assemble.assemble_baseline(
        meta=meta, transforms={"t": 1}, corpus={"c": 2}, knob_sweep={"k": 3},
    )
    assert baseline == {"meta": meta, "transforms": {"t": 1}, "corpus": {"c": 2}, "knob_sweep": {"k": 3}}
    assert set(baseline.keys()) == {"meta", "transforms", "corpus", "knob_sweep"}


# ---- compute_artifact_info / compute_db_info: missing-path failure ------------

def test_compute_artifact_info_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sonic artifact"):
        assemble.compute_artifact_info(tmp_path / "nope.npz")


def test_compute_db_info_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="library DB"):
        assemble.compute_db_info(tmp_path / "nope.db")


def test_compute_artifact_info_small_file(tmp_path):
    p = tmp_path / "artifact.npz"
    p.write_bytes(b"npz-stub-bytes")
    info = assemble.compute_artifact_info(p)
    assert info["size_bytes"] == len(b"npz-stub-bytes")
    assert info["path"] == assemble.to_posix(p)
    assert len(info["sha256"]) == 64


def test_compute_db_info_reads_track_count(tmp_path):
    import sqlite3
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO tracks VALUES (?)", [("t1",), ("t2",), ("t3",)])
    conn.commit()
    conn.close()
    info = assemble.compute_db_info(db_path)
    assert info["track_count"] == 3
    assert info["path"] == assemble.to_posix(db_path)
