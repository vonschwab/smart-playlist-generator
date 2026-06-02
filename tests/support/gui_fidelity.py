"""GUI-fidelity test harness for playlist generation.

WHY THIS EXISTS
---------------
Most "playlist bugs" we chase turn out to be the *test* diverging from how the
real GUI generates. A direct ``generate_playlist_ds(overrides={...})`` call does
NOT inherit config.yaml's defaults — missing keys silently fall to dataclass
defaults that differ from production (e.g. ``disallow_pier_artists_in_interiors``
is False by default but True in config.yaml; ``artist_identity.enabled`` is False
by default but True in config.yaml; ``min_gap`` falls to a mode default instead of
the "Artist Gap" slider value). Tests written that way pass or fail for reasons
that have nothing to do with production behavior.

This harness reproduces the EXACT config-resolution chain the GUI worker uses,
so any test built on it inherits production config automatically:

    UIStateModel (the GUI knobs)
      -> policy.derive_runtime_config()        # slider -> overrides mapping
      -> policy.merge_overrides()
      -> worker.load_config_with_overrides()   # MERGE WITH config.yaml  <- the bit tests skip
      -> playlist_generator.build_ds_overrides()
      -> ds_pipeline_runner.generate_playlist_ds()

The non-negotiable design rule: this file REUSES the production functions above.
It must never reimplement the slider->config mapping, or the drift just moves here.

SCOPE (intentional):
  * Covers generation LOGIC at full config fidelity (the class of bugs we keep hitting).
  * Seeds mode (and artist mode when piers are supplied) — fast, artifact-level, no DB / no Last.fm.
  * Does NOT cover the Qt widget layer, nor the DB-clustering / Last.fm-recency layers
    (those need the full ``handle_generate_playlist`` worker entry — a separate, heavier tier).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence

from src.playlist_gui.ui_state import UIStateModel
from src.playlist_gui.policy import derive_runtime_config, merge_overrides
from src.playlist_gui.worker import load_config_with_overrides
from src.playlist_generator import build_ds_overrides
from src.playlist.ds_pipeline_runner import generate_playlist_ds

DEFAULT_CONFIG = "config.yaml"


def gui_ui_state(**overrides: Any) -> UIStateModel:
    """Build a UIStateModel for seeds-mode generation with production-like defaults.

    Pass GUI-visible knobs as kwargs, e.g.
        gui_ui_state(cohesion_mode="narrow", genre_mode="narrow",
                     sonic_mode="narrow", pace_mode="narrow", artist_spacing="strong")
    """
    base = UIStateModel(mode="seeds")
    return replace(base, **overrides)


def resolve_gui_overrides(
    ui: UIStateModel,
    *,
    config_path: str = DEFAULT_CONFIG,
    seed_artist_keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Resolve the EXACT ds-pipeline overrides the GUI worker would send for ``ui``.

    Walks the real chain so config.yaml defaults (disallow_pier_artists,
    artist_identity, etc.) are present. Returns the dict accepted by
    ``generate_playlist_ds(overrides=...)``.
    """
    decisions = derive_runtime_config(ui, seed_artist_keys=seed_artist_keys)
    overrides = merge_overrides({}, decisions.overrides)
    merged = load_config_with_overrides(config_path, overrides)
    ds_cfg = (merged.get("playlists", {}) or {}).get("ds_pipeline", {}) or {}
    return build_ds_overrides(ds_cfg)


def resolved_artifact_path(config_path: str = DEFAULT_CONFIG) -> str:
    merged = load_config_with_overrides(config_path, {})
    ds_cfg = (merged.get("playlists", {}) or {}).get("ds_pipeline", {}) or {}
    return str(ds_cfg.get("artifact_path") or "data/artifacts/beat3tower_32k/data_matrices_step1.npz")


def generate_like_gui(
    *,
    seeds: Sequence[str],
    ui: Optional[UIStateModel] = None,
    config_path: str = DEFAULT_CONFIG,
    artifact_path: Optional[str] = None,
    length: Optional[int] = None,
    random_seed: int = 0,
    **ui_kwargs: Any,
):
    """Generate a seeds-mode playlist exactly as the GUI worker resolves config.

    ``seeds`` are bundle track_ids used as piers. Extra kwargs build the UIStateModel
    (cohesion_mode, genre_mode, sonic_mode, pace_mode, artist_spacing, ...).
    Returns the DsRunResult from generate_playlist_ds.
    """
    ui = ui or gui_ui_state(**ui_kwargs)
    ds_overrides = resolve_gui_overrides(ui, config_path=config_path)
    art = artifact_path or resolved_artifact_path(config_path)
    seeds = list(seeds)
    return generate_playlist_ds(
        artifact_path=art,
        seed_track_id=seeds[0],
        anchor_seed_ids=seeds,
        mode=ui.cohesion_mode,
        pace_mode=ui.pace_mode,
        length=length if length is not None else int(ui.track_count),
        random_seed=random_seed,
        overrides=ds_overrides,
        artist_style_enabled=False,
        artist_playlist=False,
    )


# ── Assertion helpers ───────────────────────────────────────────────────────

def artist_at_positions(bundle, track_ids) -> list[str]:
    ti = bundle.track_id_to_index
    return [str(bundle.track_artists[ti[str(t)]]) for t in track_ids if str(t) in ti]


def find_min_gap_violations(artists: Sequence[str], min_gap: int) -> list[tuple[int, int, str, int]]:
    """Return (pos_i, pos_j, artist, gap) for same-artist pairs closer than min_gap (1-indexed)."""
    out: list[tuple[int, int, str, int]] = []
    for i in range(len(artists)):
        for j in range(i + 1, min(i + min_gap, len(artists))):
            if artists[i].lower() == artists[j].lower():
                out.append((i + 1, j + 1, artists[i], j - i))
    return out


def assert_min_gap(bundle, track_ids, min_gap: int) -> None:
    artists = artist_at_positions(bundle, track_ids)
    violations = find_min_gap_violations(artists, min_gap)
    assert not violations, f"min_gap={min_gap} violations: {violations}"
