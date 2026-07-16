"""Pure perturbation logic for the 'no knob goes inert' sweep (contract §
'Automated completeness net'). No engine imports — unit-testable in isolation.

Perturbation rules (recorded in feature_baseline.json meta):
  bool  -> flipped
  C-term fields (soft scoring strengths/weights, contract Category C) -> 0.0
  float -> x1.5; 0.0 -> 0.3; *percentile*/*_floor* fields halve instead when
           x1.5 would exceed 1.0
  int   -> +1
  str / list / dict / None -> SKIP (recorded with reason; mode strings are
           covered by the dial audit / Category E, not this sweep)
"""
from __future__ import annotations

from typing import Any

SKIP = object()

_PREFIX_MAP = {
    "candidate_pool.": "playlists.ds_pipeline.candidate_pool.",
    "playlist.": "playlists.ds_pipeline.pier_bridge.",
}
_UNMAPPED_PREFIXES = ("sonic_variant.", "embedding.")

# The effective blob nests most PierBridgeConfig fields one level deeper
# (playlist.pier_config.<x>) than config.yaml stores them
# (playlists.ds_pipeline.pier_bridge.<x> — flat, no "pier_config" key).
# Verified against a real "Bill Evans Trio"/open effective blob (Step 5):
# without this strip, all ~180 playlist.pier_config.* leaves round-tripped
# to a nonexistent playlists.ds_pipeline.pier_bridge.pier_config.* path.
_INFIX_STRIP = {
    "playlist.": "pier_config.",
}

# Contract Category C soft-term knobs -> perturb to 0.0 ("term fires" differential).
C_TERM_FIELDS = frozenset({
    "duration_penalty_weight",          # C1
    "seed_character_strength",          # C2
    "popularity_penalty_strength",      # C3
    "local_sonic_edge_penalty_strength",  # C4
    "soft_genre_penalty_strength",      # C5
    "genre_tiebreak_weight",            # C6
    "genre_pair_penalty",               # C7
    "progress_penalty_weight",          # C8
    "bpm_bridge_soft_penalty_strength",   # C9
    "onset_bridge_soft_penalty_strength",  # C9
    "instrumental_penalty_weight",      # C10
    "weight_bridge",                    # C11
    "weight_transition",                # C11
})


def flatten_leaves(blob: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in blob.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_leaves(v, f"{path}."))
        else:
            out[path] = v
    return out


def config_path_for(blob_path: str) -> str | None:
    if blob_path.startswith(_UNMAPPED_PREFIXES):
        return None
    for pfx, target in _PREFIX_MAP.items():
        if blob_path.startswith(pfx):
            suffix = blob_path[len(pfx):]
            infix = _INFIX_STRIP.get(pfx)
            if infix and suffix.startswith(infix):
                suffix = suffix[len(infix):]
            return target + suffix
    return None


def perturb_value(field_name: str, value: Any) -> Any:
    leaf = field_name.rsplit(".", 1)[-1]
    if isinstance(value, bool):
        return not value
    if leaf in C_TERM_FIELDS and isinstance(value, (int, float)) and value != 0:
        return 0.0
    if isinstance(value, float):
        if value == 0.0:
            return 0.3
        if ("percentile" in leaf or leaf.endswith("_floor")) and value * 1.5 > 1.0:
            return value * 0.5
        return value * 1.5
    if isinstance(value, int):
        return value + 1
    return SKIP
