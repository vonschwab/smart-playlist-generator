"""Single source of truth for the genre-similarity params the DS pipeline needs.

The DS pipeline (`generate_playlist_ds`) takes the genre gate + hybrid weights as
explicit parameters (``min_genre_similarity``, ``sonic_weight``, ``genre_weight``,
``genre_method``, ``genre_mode``) — they are NOT carried in the ``overrides`` dict.
The orchestrator (`playlist_generator`) resolves them from ``playlists.genre_similarity``
config and passes them. Any other caller (e.g. the gui_fidelity test harness) MUST
resolve them the same way, or it silently runs with the genre gate off. This function
is that shared resolver so the orchestrator and the harness cannot drift.

``genre_mode`` (Phase 1 Task 5, corridor-pooling): the raw ``playlists.genre_mode``
string, passed straight through to ``build_pier_bridge_playlist`` for the corridor
path's genre-mode-keyed relevance mask (Phase 1 Task 4). Every other key here is a
NUMBER genre_mode resolves to (via ``mode_presets.apply_mode_presets``); this is the
one seam that carries the raw mode string itself.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_genre_ds_params(playlists_cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Resolve genre gate + hybrid weights from ``playlists_cfg`` for cohesion ``mode``.

    Mirrors playlist_generator's inline resolution exactly. ``mode`` is the cohesion
    mode (the DS pipeline ``mode`` arg). Returns a dict with keys
    ``sonic_weight``, ``genre_weight``, ``min_genre_similarity``, ``genre_method``,
    ``genre_admission_percentile``, ``genre_mode`` suitable for splatting into
    ``generate_playlist_ds(...)``.
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

    # The raw genre_mode string (Phase 1 Task 5 corridor-pooling reseat req 0):
    # apply_mode_presets reads playlists_cfg["genre_mode"] to derive the
    # min_genre_similarity/admission_percentile numbers resolved above, but
    # never deletes the key -- it survives on playlists_cfg unchanged. This is
    # the ONLY seam that carries the raw mode string (not the numbers it
    # resolves to) out to a DS-pipeline caller, which is what the corridor
    # path's build_pier_bridge_playlist(genre_mode=...) kwarg needs (see its
    # docstring). Resolved here, not re-derived, so this function stays the
    # single source of truth for every genre-mode-derived value.
    genre_mode: Optional[str] = playlists_cfg.get("genre_mode")

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
        "genre_mode": genre_mode,
    }
