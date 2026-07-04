"""
Preset modes for genre and sonic similarity.

Provides simple mode-based configuration instead of requiring users to tune
individual weights and thresholds. Each mode maps to a set of optimized
parameters for different playlist generation strategies.

Usage:
    genre_settings = resolve_genre_mode("dynamic")
    sonic_settings = resolve_sonic_mode("narrow")
"""
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

COHESIVE_BROAD_FILTERS = ["rock", "indie", "alternative", "pop"]


# ============================================================================
# GENRE SIMILARITY MODE PRESETS
# ============================================================================

GENRE_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # genre_admission_percentile (Fix 2, 2026-07-04): per-genre-mode adaptive
    # floor over the POSITIVE genre-sim mass (candidate_pool sparse gate) — the
    # live gate, replacing the absolute min_genre_similarity floor whose fixed
    # thresholds collapsed strict==narrow / dynamic==discover on the post-sonic
    # eligible set (slider-differentiation eval 2026-07-04). The absolute floor
    # keys are kept as the rollback path (act only when percentile is 0/unset).
    "strict": {
        "enabled": True,
        "weight": 0.80,
        "sonic_weight": 0.20,
        "min_genre_similarity": 0.50,  # rollback-only; inert while percentile > 0
        "genre_admission_percentile": 0.75,
        "genre_idf_enabled": True,
        "description": "Ultra-tight genre coherence - stay within seed genre",
        "use_case": "Highly cohesive playlists with minimal genre variation",
    },
    "narrow": {
        "enabled": True,
        "weight": 0.65,
        "sonic_weight": 0.35,
        "min_genre_similarity": 0.40,  # rollback-only; inert while percentile > 0
        "genre_admission_percentile": 0.60,
        "genre_idf_enabled": True,
        "description": "Stay close to seed genre with some flexibility",
        "use_case": "Familiar playlists that stay within genre boundaries",
    },
    "dynamic": {
        "enabled": True,
        "weight": 0.50,
        "sonic_weight": 0.50,
        "min_genre_similarity": 0.25,  # rollback-only; inert while percentile > 0
        "genre_admission_percentile": 0.40,
        "genre_idf_enabled": True,
        "description": "Balanced genre exploration (default)",
        "use_case": "Standard playlists with balanced genre/sonic weighting",
    },
    "discover": {
        "enabled": True,
        "weight": 0.35,
        "sonic_weight": 0.65,
        "min_genre_similarity": 0.20,  # rollback-only; inert while percentile > 0
        "genre_admission_percentile": 0.20,
        "genre_idf_enabled": False,  # Exploration mode: don't reward narrow tag matches
        "description": "Genre-adjacent exploration - venture into related genres",
        "use_case": "Exploratory playlists that cross genre boundaries",
    },
    "off": {
        "enabled": False,
        "weight": 0.0,
        "sonic_weight": 1.0,
        "min_genre_similarity": None,
        "genre_admission_percentile": 0.0,
        "genre_idf_enabled": True,
        "description": "Sonic-only mode - ignore genre completely",
        "use_case": "Pure audio similarity, disregard genre tags",
    },
}


# ============================================================================
# SONIC SIMILARITY MODE PRESETS
# ============================================================================

SONIC_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "enabled": True,
        "weight": 0.85,
        "candidate_pool_multiplier": 0.6,
        "min_sonic_similarity": 0.28,  # legacy absolute floor, INERT under MuQ — sonic_admission_percentile (below) replaces it at runtime (candidate_pool.py:658-666); applies only if percentile==0 (no shipped preset). MERT-era value, not MuQ-calibrated.
        # Adaptive percentile floor (Task 1): admits top (1-p) fraction of
        # the seed's sonic similarity distribution.  Initial conservative values —
        # calibration eval-gate sets finals.  0.0 = off (legacy absolute floor).
        "sonic_admission_percentile": 0.75,
        # Never-starve backstop (Task 3): lower bound on pool size after all
        # admission filters.  Initial values — calibration eval-gate sets finals.
        "min_pool_size": 12,
        "description": "Ultra-tight sonic matching - very similar sound",
        "use_case": "Extremely cohesive sound with minimal variation",
    },
    "narrow": {
        "enabled": True,
        "weight": 0.70,
        "candidate_pool_multiplier": 0.8,
        "min_sonic_similarity": 0.18,  # legacy absolute floor, INERT (see 'strict' above); MERT-era value, not MuQ-calibrated.
        "sonic_admission_percentile": 0.60,
        "min_pool_size": 16,
        "description": "Strict sonic coherence - familiar sound",
        "use_case": "Cohesive playlists with consistent sonic character",
    },
    "dynamic": {
        "enabled": True,
        "weight": 0.50,
        "candidate_pool_multiplier": 1.0,
        "min_sonic_similarity": 0.08,  # legacy absolute floor, INERT (see 'strict' above); MERT-era value, not MuQ-calibrated.
        "sonic_admission_percentile": 0.40,
        "min_pool_size": 20,
        "description": "Balanced sonic flow (default)",
        "use_case": "Standard playlists with moderate sonic variation",
    },
    "discover": {
        "enabled": True,
        "weight": 0.35,
        "candidate_pool_multiplier": 1.2,
        "min_sonic_similarity": 0.00,  # legacy absolute floor, INERT (see 'strict' above); MERT-era value, not MuQ-calibrated.
        "sonic_admission_percentile": 0.20,
        "min_pool_size": 24,
        "description": "Broader sonic palette - varied textures",
        "use_case": "Exploratory playlists with diverse sonic textures",
    },
    "off": {
        "enabled": False,
        "weight": 0.0,
        "candidate_pool_multiplier": None,
        "min_sonic_similarity": None,
        "sonic_admission_percentile": 0.0,  # off = no percentile gate
        "min_pool_size": 0,  # off = no backstop
        "description": "Genre-only mode - ignore sonic similarity",
        "use_case": "Match by genre tags only, disregard audio features",
    },
}


# ============================================================================
# PACE (RHYTHM/TEMPO) MODE PRESETS
# ============================================================================

PACE_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": 0.30,
        "bpm_bridge_max_log_distance": 0.40,
        "bpm_trust_min_onset_rate": 0.5,
        "onset_admission_max_log_distance": 0.30,
        "onset_bridge_max_log_distance": 0.40,
        "bpm_bridge_soft_penalty_strength": 0.50,
        "onset_bridge_soft_penalty_strength": 0.50,
        # Energy arc + rescue: calibrated 2026-06-20 (Task 5, worst-edge gate PASS k=20).
        # Rescue re-admits up to 20 arousal-spanning candidates rejected by rhythm bands,
        # so energy can arc even when BPM gates constrain the pool.
        "energy_step_cap": 1.0,
        "energy_step_strength": 0.2,
        "energy_arc_band": 0.5,
        "energy_arc_strength": 0.3,
        "pace_rescue_k_energy": 20,
        "description": "Tight tempo fidelity - stay anchored to seed pace",
        "use_case": "Slow/meditative seeds; mood-locked playlists",
    },
    "narrow": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": 0.50,
        "bpm_bridge_max_log_distance": 0.60,
        "bpm_trust_min_onset_rate": 0.5,
        "onset_admission_max_log_distance": 0.50,
        "onset_bridge_max_log_distance": 0.60,
        "bpm_bridge_soft_penalty_strength": 0.40,
        "onset_bridge_soft_penalty_strength": 0.40,
        # Energy arc + rescue: calibrated 2026-06-20 (Task 5, worst-edge gate PASS k=5).
        # k=10 FAILED on WIDE_SWING (worst_edge_sonic dropped -0.064, beyond -0.05 gate).
        "energy_step_cap": 1.5,
        "energy_step_strength": 0.15,
        "energy_arc_band": 0.4,
        "energy_arc_strength": 0.2,
        "pace_rescue_k_energy": 5,
        "description": "Moderate tempo anchoring",
        "use_case": "Consistent energy with some flex",
    },
    "dynamic": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": 0.75,
        "bpm_bridge_max_log_distance": 0.85,
        "bpm_trust_min_onset_rate": 0.5,
        "onset_admission_max_log_distance": 0.75,
        "onset_bridge_max_log_distance": 0.85,
        "bpm_bridge_soft_penalty_strength": 0.30,
        "onset_bridge_soft_penalty_strength": 0.30,
        # Energy arc: 0.0 (unevaluated — dynamic arc was shipped without worst-edge gate;
        # set to 0.0 until a full eval-gated calibration run is completed and authorized).
        "energy_step_cap": 0.0,
        "energy_step_strength": 0.0,
        "energy_arc_band": 0.0,
        "energy_arc_strength": 0.0,
        "pace_rescue_k_energy": 0,
        "description": "Gentle pace anchoring - catches double-time, allows natural drift",
        "use_case": "General-purpose default; varied playlists with sensible tempo coherence",
    },
    "off": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": float("inf"),
        "bpm_bridge_max_log_distance": float("inf"),
        "bpm_trust_min_onset_rate": 0.0,
        "onset_admission_max_log_distance": float("inf"),
        "onset_bridge_max_log_distance": float("inf"),
        "bpm_bridge_soft_penalty_strength": 0.0,
        "onset_bridge_soft_penalty_strength": 0.0,
        # No energy arc or rescue: pace=off means rhythm is fully disabled.
        "energy_step_cap": 0.0,
        "energy_step_strength": 0.0,
        "energy_arc_band": 0.0,
        "energy_arc_strength": 0.0,
        "pace_rescue_k_energy": 0,
        "description": "No pace constraint - rhythm contributes via sonic embedding only",
        "use_case": "Multi-tempo playlists; no explicit tempo gating",
    },
}


# ============================================================================
# QUICK PRESET COMBINATIONS
# ============================================================================

QUICK_PRESETS: Dict[str, Dict[str, str]] = {
    "balanced": {
        "genre": "dynamic",
        "sonic": "dynamic",
        "description": "Default - balanced genre and sonic exploration",
    },
    "tight": {
        "genre": "strict",
        "sonic": "strict",
        "description": "Ultra-cohesive - same genre and sound",
    },
    "exploratory": {
        "genre": "discover",
        "sonic": "discover",
        "description": "Maximum exploration - adventurous genre and sonic variety",
    },
    "sonic_only": {
        "genre": "off",
        "sonic": "dynamic",
        "description": "Pure sonic similarity - ignore genre tags",
    },
    "genre_only": {
        "genre": "dynamic",
        "sonic": "off",
        "description": "Pure genre matching - ignore sonic features",
    },
    "varied_sound": {
        "genre": "narrow",
        "sonic": "discover",
        "description": "Same genre with varied sonic textures",
    },
    "sonic_thread": {
        "genre": "discover",
        "sonic": "narrow",
        "description": "Tight sound through adjacent genres",
    },
}


# ============================================================================
# MODE RESOLUTION FUNCTIONS
# ============================================================================

def resolve_genre_mode(mode: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve genre mode string to configuration settings.

    Args:
        mode: Genre mode name (strict/narrow/dynamic/discover/off)
        overrides: Optional dict of settings to override preset values

    Returns:
        Dictionary of genre similarity configuration settings

    Raises:
        ValueError: If mode is not recognized

    Examples:
        >>> settings = resolve_genre_mode("dynamic")
        >>> settings["weight"]
        0.5
        >>> settings = resolve_genre_mode("narrow", {"weight": 0.70})
        >>> settings["weight"]
        0.7
    """
    mode = mode.lower()
    if mode not in GENRE_MODE_PRESETS:
        valid_modes = ", ".join(GENRE_MODE_PRESETS.keys())
        raise ValueError(f"Unknown genre mode: '{mode}'. Valid modes: {valid_modes}")

    # Start with preset
    settings = GENRE_MODE_PRESETS[mode].copy()

    # Apply overrides if provided
    if overrides:
        settings.update(overrides)
        logger.info(f"Genre mode '{mode}' with overrides: {overrides}")
    else:
        logger.info(f"Genre mode '{mode}': {settings['description']}")

    return settings


def resolve_sonic_mode(mode: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve sonic mode string to configuration settings.

    Args:
        mode: Sonic mode name (strict/narrow/dynamic/discover/off)
        overrides: Optional dict of settings to override preset values

    Returns:
        Dictionary of sonic similarity configuration settings

    Raises:
        ValueError: If mode is not recognized

    Examples:
        >>> settings = resolve_sonic_mode("narrow")
        >>> settings["weight"]
        0.7
        >>> settings = resolve_sonic_mode("dynamic", {"weight": 0.60})
        >>> settings["weight"]
        0.6
    """
    mode = mode.lower()
    if mode not in SONIC_MODE_PRESETS:
        valid_modes = ", ".join(SONIC_MODE_PRESETS.keys())
        raise ValueError(f"Unknown sonic mode: '{mode}'. Valid modes: {valid_modes}")

    # Start with preset
    settings = SONIC_MODE_PRESETS[mode].copy()

    # Apply overrides if provided
    if overrides:
        settings.update(overrides)
        logger.info(f"Sonic mode '{mode}' with overrides: {overrides}")
    else:
        logger.info(f"Sonic mode '{mode}': {settings['description']}")

    return settings


def resolve_pace_mode(mode: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve pace mode string to rhythm-axis gate settings."""
    mode = (mode or "dynamic").lower()
    if mode not in PACE_MODE_PRESETS:
        valid_modes = ", ".join(PACE_MODE_PRESETS.keys())
        raise ValueError(f"Unknown pace mode: '{mode}'. Valid modes: {valid_modes}")

    settings = {
        key: value
        for key, value in PACE_MODE_PRESETS[mode].items()
        if key not in {"description", "use_case"}
    }
    if overrides:
        settings.update(overrides)
        logger.info(f"Pace mode '{mode}' with overrides: {overrides}")
    else:
        logger.info(f"Pace mode '{mode}': {PACE_MODE_PRESETS[mode]['description']}")
    return settings


def apply_mode_presets(playlists_cfg: Dict[str, Any]) -> None:
    """
    Apply genre_mode/sonic_mode presets to a playlists config dictionary.

    This is the single source of truth for mode-driven gates/weights.
    """
    if not playlists_cfg:
        return

    genre_mode = playlists_cfg.get("genre_mode")
    sonic_mode = playlists_cfg.get("sonic_mode")
    if not genre_mode and not sonic_mode:
        return

    genre_cfg = playlists_cfg.setdefault("genre_similarity", {})
    ds_cfg = playlists_cfg.setdefault("ds_pipeline", {})
    candidate_pool = ds_cfg.setdefault("candidate_pool", {})

    genre_enabled = bool(genre_cfg.get("enabled", True))
    genre_weight = float(genre_cfg.get("weight", 0.50))
    sonic_weight = float(genre_cfg.get("sonic_weight", 0.50))
    min_genre_sim = genre_cfg.get("min_genre_similarity")
    min_sonic_similarity = candidate_pool.get("min_sonic_similarity")
    genre_idf_enabled: Optional[bool] = None  # None means "leave unset" (no mode specified)

    if genre_mode:
        genre_settings = resolve_genre_mode(genre_mode)
        genre_enabled = bool(genre_settings["enabled"])
        genre_weight = float(genre_settings["weight"])
        min_genre_sim = genre_settings.get("min_genre_similarity")
        genre_idf_enabled = bool(genre_settings.get("genre_idf_enabled", True))
        # Fix 2 (2026-07-04): write the genre-mode admission percentile into
        # genre_cfg for genre_ds_params to resolve. Respect an explicit user
        # value (mirrors the sonic_admission_percentile pattern below).
        if "admission_percentile" not in genre_cfg:
            _gap_preset = genre_settings.get("genre_admission_percentile")
            if _gap_preset is not None:
                genre_cfg["admission_percentile"] = float(_gap_preset)
        if (
            str(genre_mode).strip().lower() in {"strict", "narrow"}
            and "broad_filters" not in candidate_pool
        ):
            candidate_pool["broad_filters"] = list(COHESIVE_BROAD_FILTERS)
        if not sonic_mode:
            sonic_weight = float(genre_settings["sonic_weight"])

    if sonic_mode:
        sonic_settings = resolve_sonic_mode(sonic_mode)
        if sonic_settings["enabled"]:
            sonic_weight = float(sonic_settings["weight"])
            min_sonic_similarity = sonic_settings.get("min_sonic_similarity")
        else:
            sonic_weight = 0.0
            genre_weight = 1.0
            min_sonic_similarity = None
            if not genre_mode:
                genre_enabled = True
        if not genre_mode:
            genre_weight = 0.0 if sonic_weight == 0.0 else max(0.0, 1.0 - sonic_weight)

        # Write sonic_admission_percentile preset value to pier_bridge config so
        # pipeline/core.py picks it up via pb_overrides (mirrors the genre
        # admission percentile pattern in config.example.yaml).
        _sap_preset = sonic_settings.get("sonic_admission_percentile", 0.0)
        if _sap_preset is not None:
            pier_bridge_cfg = ds_cfg.setdefault("pier_bridge", {})
            # Only write if not already explicitly overridden by the user.
            if "sonic_admission_percentile" not in pier_bridge_cfg:
                pier_bridge_cfg["sonic_admission_percentile"] = float(_sap_preset)

        # Write min_pool_size preset value to pier_bridge config so
        # pipeline/core.py picks it up via pb_overrides (same pattern as
        # sonic_admission_percentile above).
        _mps_preset = sonic_settings.get("min_pool_size", 0)
        if _mps_preset is not None:
            pier_bridge_cfg = ds_cfg.setdefault("pier_bridge", {})
            if "min_pool_size" not in pier_bridge_cfg:
                pier_bridge_cfg["min_pool_size"] = int(_mps_preset)

    if genre_mode and not genre_enabled:
        genre_weight = 0.0
        min_genre_sim = None
        sonic_weight = 1.0

    if genre_mode and sonic_mode and genre_enabled and sonic_weight > 0.0:
        total = float(sonic_weight) + float(genre_weight)
        if total > 0:
            sonic_weight = float(sonic_weight) / total
            genre_weight = float(genre_weight) / total

    genre_cfg["enabled"] = bool(genre_enabled)
    genre_cfg["weight"] = float(genre_weight)
    genre_cfg["sonic_weight"] = float(sonic_weight)
    if min_genre_sim is not None:
        genre_cfg["min_genre_similarity"] = float(min_genre_sim)
    else:
        genre_cfg.pop("min_genre_similarity", None)
    # min_genre_similarity_narrow removed 2026-07-04: its only reader was the
    # cohesion-keyed swap in genre_ds_params (deleted — cross-axis coupling bug).
    genre_cfg.pop("min_genre_similarity_narrow", None)

    if min_sonic_similarity is not None:
        candidate_pool["min_sonic_similarity"] = float(min_sonic_similarity)
    else:
        candidate_pool["min_sonic_similarity"] = None

    if genre_idf_enabled is not None:
        candidate_pool["genre_idf_enabled"] = genre_idf_enabled


def resolve_quick_preset(preset: str) -> Tuple[str, str]:
    """
    Resolve quick preset name to (genre_mode, sonic_mode) tuple.

    Args:
        preset: Quick preset name (balanced/tight/exploratory/etc.)

    Returns:
        Tuple of (genre_mode, sonic_mode)

    Raises:
        ValueError: If preset is not recognized

    Examples:
        >>> resolve_quick_preset("balanced")
        ('dynamic', 'dynamic')
        >>> resolve_quick_preset("tight")
        ('strict', 'strict')
    """
    preset = preset.lower()
    if preset not in QUICK_PRESETS:
        valid_presets = ", ".join(QUICK_PRESETS.keys())
        raise ValueError(f"Unknown quick preset: '{preset}'. Valid presets: {valid_presets}")

    preset_config = QUICK_PRESETS[preset]
    logger.info(f"Quick preset '{preset}': {preset_config['description']}")

    return preset_config["genre"], preset_config["sonic"]


def get_mode_description(genre_mode: str, sonic_mode: str) -> str:
    """
    Get a human-readable description of the combined genre/sonic mode configuration.

    Args:
        genre_mode: Genre mode name
        sonic_mode: Sonic mode name

    Returns:
        String description of what this mode combination does

    Examples:
        >>> get_mode_description("strict", "strict")
        'Ultra-tight genre coherence + Ultra-tight sonic matching'
        >>> get_mode_description("dynamic", "dynamic")
        'Balanced genre exploration + Balanced sonic flow'
    """
    genre_desc = GENRE_MODE_PRESETS.get(genre_mode.lower(), {}).get("description", "Unknown")
    sonic_desc = SONIC_MODE_PRESETS.get(sonic_mode.lower(), {}).get("description", "Unknown")

    return f"{genre_desc} + {sonic_desc}"


def validate_mode_combination(genre_mode: str, sonic_mode: str) -> bool:
    """
    Validate that a genre/sonic mode combination is valid.

    Args:
        genre_mode: Genre mode name
        sonic_mode: Sonic mode name

    Returns:
        True if valid, False otherwise

    Note:
        The only invalid combination is genre=off + sonic=off (no similarity at all)
    """
    if genre_mode.lower() == "off" and sonic_mode.lower() == "off":
        logger.error("Invalid mode combination: both genre and sonic cannot be 'off'")
        return False
    return True


def list_available_modes() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available modes.

    Returns:
        Dictionary with 'genre', 'sonic', and 'quick_presets' keys containing mode info
    """
    return {
        "genre": {
            mode: {
                "description": config["description"],
                "use_case": config["use_case"],
            }
            for mode, config in GENRE_MODE_PRESETS.items()
        },
        "sonic": {
            mode: {
                "description": config["description"],
                "use_case": config["use_case"],
            }
            for mode, config in SONIC_MODE_PRESETS.items()
        },
        "quick_presets": {
            preset: config["description"]
            for preset, config in QUICK_PRESETS.items()
        },
    }


# ============================================================================
# MODE COMPARISON UTILITIES
# ============================================================================

def compare_modes(genre_mode: str, sonic_mode: str) -> Dict[str, Any]:
    """
    Get detailed comparison of the resolved settings for a mode combination.

    Args:
        genre_mode: Genre mode name
        sonic_mode: Sonic mode name

    Returns:
        Dictionary with resolved settings and analysis
    """
    genre_settings = resolve_genre_mode(genre_mode)
    sonic_settings = resolve_sonic_mode(sonic_mode)

    # Calculate effective weights (genre_weight + sonic_weight should = 1.0)
    genre_weight = genre_settings.get("weight", 0.0)
    sonic_weight_from_genre = genre_settings.get("sonic_weight", 0.0)
    sonic_weight_from_sonic = sonic_settings.get("weight", 0.0)

    # Determine which weight to use (genre preset's sonic_weight takes precedence)
    effective_sonic_weight = sonic_weight_from_genre if genre_settings["enabled"] else sonic_weight_from_sonic

    return {
        "genre_mode": genre_mode,
        "sonic_mode": sonic_mode,
        "description": get_mode_description(genre_mode, sonic_mode),
        "weights": {
            "genre": genre_weight,
            "sonic": effective_sonic_weight,
            "total": genre_weight + effective_sonic_weight,
        },
        "genre_settings": genre_settings,
        "sonic_settings": sonic_settings,
        "playlist_character": _infer_playlist_character(genre_mode, sonic_mode),
    }


def _infer_playlist_character(genre_mode: str, sonic_mode: str) -> str:
    """Infer the overall character of playlists generated with this mode combination."""
    if genre_mode == "strict" and sonic_mode == "strict":
        return "Ultra-cohesive - tightest possible matching"
    elif genre_mode == "off" and sonic_mode != "off":
        return "Sonic-only - pure audio similarity"
    elif genre_mode != "off" and sonic_mode == "off":
        return "Genre-only - pure genre tag matching"
    elif "strict" in (genre_mode, sonic_mode):
        return "Very cohesive with strong constraints"
    elif "discover" in (genre_mode, sonic_mode) and "narrow" in (genre_mode, sonic_mode):
        return "Interesting contrast - exploration in one dimension, coherence in other"
    elif genre_mode == "discover" and sonic_mode == "discover":
        return "Maximum exploration - adventurous"
    elif genre_mode == "dynamic" and sonic_mode == "dynamic":
        return "Balanced - standard playlist generation"
    else:
        return "Custom mode combination"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "GENRE_MODE_PRESETS",
    "SONIC_MODE_PRESETS",
    "PACE_MODE_PRESETS",
    "QUICK_PRESETS",
    "apply_mode_presets",
    "resolve_genre_mode",
    "resolve_sonic_mode",
    "resolve_pace_mode",
    "resolve_quick_preset",
    "get_mode_description",
    "validate_mode_combination",
    "list_available_modes",
    "compare_modes",
]
