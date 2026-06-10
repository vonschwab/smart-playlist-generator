"""Tests for the genre-similarity source flag (Stage 3 of the taxonomy integration).

playlists.ds_pipeline.genre_similarity.source selects which generator writes
genre_similarity_matrix.npz at artifact-build time:
  cooccurrence (default) — legacy Jaccard from library tag co-occurrence
  graph                  — derived from the SP3a layered taxonomy

Provenance lives in the NPZ stats ("source") so flipping the flag forces a
rebuild instead of silently reusing a matrix built the other way.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from src.config_loader import Config
from src.genre.graph_adapter import load_graph_adapter
from src.genre.graph_similarity import (
    build_graph_similarity,
    npz_similarity_source,
    save_graph_similarity_npz,
)


def _write_config(tmp_path: Path, source: str | None) -> str:
    cfg: dict = {
        "library": {"database_path": str(tmp_path / "unused.db")},
        "playlists": {"ds_pipeline": {}},
    }
    if source is not None:
        cfg["playlists"]["ds_pipeline"]["genre_similarity"] = {"source": source}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


# --- config getter -----------------------------------------------------------


def test_genre_similarity_source_defaults_to_cooccurrence(tmp_path: Path):
    cfg = Config(_write_config(tmp_path, None))
    assert cfg.get_ds_genre_similarity_source() == "cooccurrence"


def test_genre_similarity_source_graph(tmp_path: Path):
    cfg = Config(_write_config(tmp_path, "graph"))
    assert cfg.get_ds_genre_similarity_source() == "graph"


def test_genre_similarity_source_invalid_falls_back(tmp_path: Path):
    cfg = Config(_write_config(tmp_path, "quantum"))
    assert cfg.get_ds_genre_similarity_source() == "cooccurrence"


# --- NPZ provenance ----------------------------------------------------------


def test_graph_builder_stamps_source_in_stats():
    result = build_graph_similarity(load_graph_adapter())
    assert result.stats["source"] == "graph"


def test_npz_similarity_source_reads_provenance(tmp_path: Path):
    out = tmp_path / "sim.npz"
    save_graph_similarity_npz(build_graph_similarity(load_graph_adapter()), out)
    assert npz_similarity_source(out) == "graph"


def test_npz_similarity_source_legacy_npz_defaults_cooccurrence(tmp_path: Path):
    out = tmp_path / "legacy.npz"
    np.savez(
        out,
        genre_vocab=np.array(["a", "b"], dtype=object),
        S=np.eye(2, dtype=np.float32),
        stats={"genres_kept": 2},
    )
    assert npz_similarity_source(out) == "cooccurrence"


def test_npz_similarity_source_missing_file_is_none(tmp_path: Path):
    assert npz_similarity_source(tmp_path / "nope.npz") is None


# --- analyze_library stage ---------------------------------------------------


def _stage_ctx(tmp_path: Path, *, source: str, force: bool = False) -> dict:
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(exist_ok=True)
    return {
        "out_dir": out_dir,
        "db_path": str(tmp_path / "unused.db"),
        "config_path": _write_config(tmp_path, source),
        "args": SimpleNamespace(force=force),
    }


def test_stage_genre_sim_graph_mode_writes_provenance_tagged_npz(tmp_path: Path):
    from scripts.analyze_library import stage_genre_sim

    ctx = _stage_ctx(tmp_path, source="graph")
    result = stage_genre_sim(ctx)
    assert result["skipped"] is False
    out_path = Path(result["path"])
    assert out_path.name == "genre_similarity_matrix.npz"
    assert npz_similarity_source(out_path) == "graph"

    data = np.load(out_path, allow_pickle=True)
    vocab = [str(g) for g in data["genre_vocab"]]
    assert "shoegaze" in vocab
    assert data["S"].shape == (len(vocab), len(vocab))


def test_stage_genre_sim_graph_mode_skips_when_fresh(tmp_path: Path):
    from scripts.analyze_library import stage_genre_sim

    ctx = _stage_ctx(tmp_path, source="graph")
    assert stage_genre_sim(ctx)["skipped"] is False
    assert stage_genre_sim(ctx)["skipped"] is True


def test_stage_genre_sim_rebuilds_on_source_mismatch(tmp_path: Path, monkeypatch):
    import scripts.analyze_library as al

    # Existing matrix was built by the graph generator…
    ctx = _stage_ctx(tmp_path, source="graph")
    assert al.stage_genre_sim(ctx)["skipped"] is False

    # …but the config now asks for cooccurrence: the stage must not reuse it.
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("no db in this test")

    monkeypatch.setattr(al, "build_genre_similarity_matrix", fake_build)
    result = al.stage_genre_sim(_stage_ctx(tmp_path, source="cooccurrence"))
    assert calls, "stale graph-built matrix was silently reused"
    assert result["skipped"] is True and "no db" in result["reason"]


# --- GUI worker resolver -----------------------------------------------------


def test_worker_genre_sim_path_none_by_default(tmp_path: Path):
    from src.playlist_gui.worker import _resolve_worker_genre_sim_path

    config = {"playlists": {"ds_pipeline": {}}}
    artifact = tmp_path / "artifacts" / "data_matrices_step1.npz"
    assert _resolve_worker_genre_sim_path(config, str(artifact)) is None


def test_worker_genre_sim_path_graph_builds_and_returns(tmp_path: Path):
    from src.playlist_gui.worker import _resolve_worker_genre_sim_path

    config = {"playlists": {"ds_pipeline": {"genre_similarity": {"source": "graph"}}}}
    artifact = tmp_path / "artifacts" / "data_matrices_step1.npz"
    resolved = _resolve_worker_genre_sim_path(config, str(artifact))
    assert resolved is not None
    path = Path(resolved)
    assert path.parent == artifact.parent
    assert npz_similarity_source(path) == "graph"
    # Second call reuses the fresh matrix (same path, still graph-tagged).
    assert _resolve_worker_genre_sim_path(config, str(artifact)) == resolved
