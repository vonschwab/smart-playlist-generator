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


def test_admission_percentile_resolved_from_genre_cfg():
    """Fix 2 (2026-07-04): the genre-mode admission percentile rides genre_cfg
    (written by apply_mode_presets from GENRE_MODE_PRESETS) and is resolved
    here like the other genre-gate params, keyed by genre_mode NOT cohesion."""
    cfg = _cfg(enabled=True, min_genre_similarity=0.25, admission_percentile=0.60)
    out = resolve_genre_ds_params(cfg, "dynamic")
    assert out["genre_admission_percentile"] == 0.60


def test_admission_percentile_none_when_absent_or_disabled():
    out = resolve_genre_ds_params(_cfg(enabled=True, min_genre_similarity=0.25), "dynamic")
    assert out["genre_admission_percentile"] is None
    out = resolve_genre_ds_params(_cfg(enabled=False, admission_percentile=0.60), "dynamic")
    assert out["genre_admission_percentile"] is None


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


# ── Phase 1 Task 5 req 0: raw genre_mode passthrough ────────────────────────
#
# The corridor path's genre-mode-keyed relevance mask (Phase 1 Task 4) needs
# the raw playlists.genre_mode STRING, not just the numbers apply_mode_presets
# resolves it to. apply_mode_presets reads the key but never deletes it, so
# it survives on playlists_cfg for this function to read straight through.

def test_genre_mode_passthrough():
    cfg = _cfg(enabled=True, min_genre_similarity=0.30)
    cfg["genre_mode"] = "strict"
    out = resolve_genre_ds_params(cfg, "dynamic")
    assert out["genre_mode"] == "strict"


def test_genre_mode_none_when_absent():
    out = resolve_genre_ds_params(_cfg(enabled=True, min_genre_similarity=0.30), "dynamic")
    assert out["genre_mode"] is None


def test_genre_mode_passthrough_independent_of_cohesion_mode():
    """genre_mode is read verbatim off playlists_cfg -- unlike min_genre_similarity
    (see test_cohesion_mode_does_not_alter_genre_floor), cohesion `mode` was never
    involved in resolving it, so there's no coupling to pin against, only that it's
    the same string regardless of which cohesion mode is passed."""
    cfg = _cfg(enabled=True, min_genre_similarity=0.30)
    cfg["genre_mode"] = "discover"
    for cohesion_mode in ("strict", "narrow", "dynamic", "discover", "sonic_only"):
        out = resolve_genre_ds_params(cfg, cohesion_mode)
        assert out["genre_mode"] == "discover", cohesion_mode
