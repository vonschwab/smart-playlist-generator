"""Single source of truth for the genre-similarity params the DS pipeline needs.

The DS pipeline (`generate_playlist_ds`) takes the genre gate + hybrid weights as
explicit parameters (``min_genre_similarity``, ``sonic_weight``, ``genre_weight``,
``genre_method``) — they are NOT carried in the ``overrides`` dict. The orchestrator
(`playlist_generator`) resolves them from ``playlists.genre_similarity`` config and
passes them. Any other caller (e.g. the gui_fidelity test harness) MUST resolve them
the same way, or it silently runs with the genre gate off. This function is that
shared resolver so the orchestrator and the harness cannot drift.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_genre_ds_params(playlists_cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Resolve genre gate + hybrid weights from ``playlists_cfg`` for cohesion ``mode``.

    Mirrors playlist_generator's inline resolution exactly. ``mode`` is the cohesion
    mode (the DS pipeline ``mode`` arg). Returns a dict with keys
    ``sonic_weight``, ``genre_weight``, ``min_genre_similarity``, ``genre_method``
    suitable for splatting into ``generate_playlist_ds(...)``.
    """
    genre_cfg = playlists_cfg.get("genre_similarity", {}) or {}
    genre_enabled = genre_cfg.get("enabled", True)

    # The floor is owned entirely by the genre preset already baked into
    # genre_cfg (keyed by genre_mode). Cohesion ``mode`` must not alter it —
    # the legacy min_genre_similarity_narrow swap keyed on cohesion=narrow
    # was removed 2026-07-04 (slider-differentiation eval: it silently raised
    # the genre gate 0.25 -> 0.40 during pure cohesion sweeps).
    min_genre_sim: Optional[float] = (
        genre_cfg.get("min_genre_similarity", 0.30) if genre_enabled else None
    )

    # Fix 2 (2026-07-04): per-genre-mode adaptive admission percentile
    # (positive-mass, sparse flat gate) — the live genre gate; the absolute
    # floor above is the rollback path (acts only when this is None/0).
    genre_admission_percentile: Optional[float] = None
    if genre_enabled:
        _gap = genre_cfg.get("admission_percentile")
        if _gap is not None:
            genre_admission_percentile = float(_gap)

    genre_method: Optional[str] = genre_cfg.get("method", "ensemble") if genre_enabled else None
    sonic_weight: Optional[float] = genre_cfg.get("sonic_weight", 0.50) if genre_enabled else None
    genre_weight: Optional[float] = genre_cfg.get("weight", 0.50) if genre_enabled else None
    if not genre_enabled:
        genre_weight = 0.0
        sonic_weight = genre_cfg.get("sonic_weight", 1.0) or 1.0

    mode_overrides_active = bool(
        playlists_cfg.get("genre_mode") or playlists_cfg.get("sonic_mode")
    )
    if mode == "sonic_only" and not mode_overrides_active:
        min_genre_sim = None
        genre_method = None
        genre_weight = 0.0
        sonic_weight = 1.0
        genre_admission_percentile = None

    return {
        "sonic_weight": sonic_weight,
        "genre_weight": genre_weight,
        "min_genre_similarity": min_genre_sim,
        "genre_method": genre_method,
        "genre_admission_percentile": genre_admission_percentile,
    }
