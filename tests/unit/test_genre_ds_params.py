"""Unit tests for resolve_genre_ds_params — the single source of truth for the
genre gate + hybrid weights the DS pipeline needs.

This logic used to live inline in playlist_generator (the orchestrator) and was
NOT reproduced by the gui_fidelity harness, so harness-based genre experiments
silently ran with the genre gate OFF. Extracting it into one function lets both
the orchestrator and the harness call the same resolution — no drift.
"""
from __future__ import annotations

from src.playlist.genre_ds_params import resolve_genre_ds_params


def _cfg(**genre):
    return {"genre_similarity": genre}


def test_genre_enabled_dynamic_reads_config():
    cfg = _cfg(enabled=True, min_genre_similarity=0.40, sonic_weight=0.52,
              weight=0.48, method="ensemble")
    out = resolve_genre_ds_params(cfg, "dynamic")
    assert out["min_genre_similarity"] == 0.40
    assert out["sonic_weight"] == 0.52
    assert out["genre_weight"] == 0.48
    assert out["genre_method"] == "ensemble"


def test_narrow_mode_uses_narrow_floor():
    cfg = _cfg(enabled=True, min_genre_similarity=0.40,
               min_genre_similarity_narrow=0.42)
    out = resolve_genre_ds_params(cfg, "narrow")
    assert out["min_genre_similarity"] == 0.42


def test_narrow_floor_ignored_in_dynamic():
    cfg = _cfg(enabled=True, min_genre_similarity=0.40,
               min_genre_similarity_narrow=0.42)
    out = resolve_genre_ds_params(cfg, "dynamic")
    assert out["min_genre_similarity"] == 0.40


def test_genre_disabled_zeros_genre_weight():
    cfg = _cfg(enabled=False, sonic_weight=0.7)
    out = resolve_genre_ds_params(cfg, "dynamic")
    assert out["min_genre_similarity"] is None
    assert out["genre_weight"] == 0.0
    assert out["sonic_weight"] == 0.7
    assert out["genre_method"] is None


def test_defaults_when_keys_absent():
    out = resolve_genre_ds_params({"genre_similarity": {}}, "dynamic")
    assert out["min_genre_similarity"] == 0.30
    assert out["sonic_weight"] == 0.50
    assert out["genre_weight"] == 0.50
    assert out["genre_method"] == "ensemble"


def test_sonic_only_without_mode_overrides_disables_genre():
    cfg = _cfg(enabled=True, min_genre_similarity=0.40)
    out = resolve_genre_ds_params(cfg, "sonic_only")
    assert out["min_genre_similarity"] is None
    assert out["genre_weight"] == 0.0
    assert out["sonic_weight"] == 1.0


def test_sonic_only_with_mode_overrides_keeps_genre():
    cfg = _cfg(enabled=True, min_genre_similarity=0.40)
    cfg["genre_mode"] = "narrow"  # mode override active
    out = resolve_genre_ds_params(cfg, "sonic_only")
    assert out["min_genre_similarity"] == 0.40
