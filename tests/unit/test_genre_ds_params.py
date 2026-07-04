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


def test_cohesion_mode_does_not_alter_genre_floor():
    """The genre floor is owned by genre_mode's preset (baked into genre_cfg by
    apply_mode_presets). Cohesion mode must never swap in a different floor —
    the 2026-07-04 slider eval caught cohesion=narrow silently raising the
    genre gate 0.25 -> 0.40 (docs/run_audits/slider_differentiation_2026-07-04)."""
    cfg = _cfg(enabled=True, min_genre_similarity=0.25,
               min_genre_similarity_narrow=0.40)  # legacy key: must be ignored
    for cohesion_mode in ("strict", "narrow", "dynamic", "discover"):
        out = resolve_genre_ds_params(cfg, cohesion_mode)
        assert out["min_genre_similarity"] == 0.25, cohesion_mode


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
