"""
Settings Schema - Defines all configurable settings for the GUI

This schema-driven approach makes it easy to add new controls:
1. Add a SettingSpec to SETTINGS_SCHEMA
2. The advanced_panel.py will automatically render the control
3. The config_model.py will handle merging/validation

Normalization groups ensure weights sum to 1.0 within each group.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


class SettingType(Enum):
    """Types of settings supported by the GUI"""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    CHOICE = "choice"


@dataclass
class SettingSpec:
    """
    Specification for a single configurable setting.

    Attributes:
        key_path: Dot-separated path in config (e.g., "playlists.ds_pipeline.tower_weights.rhythm")
        label: Human-readable label for the UI
        setting_type: Type of the setting (int, float, bool, etc.)
        group: UI grouping for organizing controls
        default: Default value if not in config
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        step: Step size for sliders (for numeric types)
        tooltip: Short help text shown on hover
        description: Detailed explanation shown in expandable info section
        normalize_group: Name of normalization group (weights in same group sum to 1.0)
        choices: List of valid choices (for CHOICE type)
        display_as_percent: If True, display float as percentage (0-100)
    """
    key_path: str
    label: str
    setting_type: SettingType
    group: str
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    tooltip: Optional[str] = None
    description: Optional[str] = None
    normalize_group: Optional[str] = None
    choices: Optional[List[str]] = None
    display_as_percent: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Settings Schema Definition
# ─────────────────────────────────────────────────────────────────────────────

SETTINGS_SCHEMA: List[SettingSpec] = [
    # ─────────────────────────────────────────────────────────────────────────
    # Playlist Counts
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.count",
        label="Playlists per batch",
        setting_type=SettingType.INT,
        group="Playlist Settings",
        default=8,
        min_value=1,
        max_value=20,
        step=1,
        tooltip="Number of playlists to generate in batch mode",
        description="When generating from listening history, this controls how many separate playlists are created in one batch. Each playlist will have different seed tracks from your recent listening."
    ),
    SettingSpec(
        key_path="playlists.tracks_per_playlist",
        label="Tracks per playlist",
        setting_type=SettingType.INT,
        group="Playlist Settings",
        default=30,
        min_value=10,
        max_value=100,
        step=5,
        tooltip="Target number of tracks per playlist",
        description="The target length for each generated playlist. The actual count may vary slightly based on available candidates and constraint enforcement."
    ),
    SettingSpec(
        key_path="playlists.seed_count",
        label="Seed count",
        setting_type=SettingType.INT,
        group="Playlist Settings",
        default=5,
        min_value=1,
        max_value=20,
        step=1,
        tooltip="Number of seed tracks from listening history",
        description="How many of your recently played tracks are used as 'seeds' to find similar music. More seeds = more variety in the playlist, but potentially less cohesion. For focused playlists, try 2-3 seeds."
    ),
    SettingSpec(
        key_path="playlists.similar_per_seed",
        label="Similar tracks per seed",
        setting_type=SettingType.INT,
        group="Playlist Settings",
        default=20,
        min_value=5,
        max_value=50,
        step=5,
        tooltip="How many similar tracks to find per seed",
        description="For each seed track, this many candidates are found. Higher values give more options but take longer. The candidates are pooled and then the best are selected for the final playlist."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid Embedding Weights (must sum to 1.0)
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.embedding.sonic_weight",
        label="Sonic weight",
        setting_type=SettingType.FLOAT,
        group="Hybrid Weights",
        default=0.60,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="embedding_weights",
        tooltip="Weight for sonic (audio) similarity",
        description="Controls how much the actual audio characteristics (rhythm, timbre, harmony) influence track selection vs. genre metadata. Higher sonic weight = playlists based more on how songs actually sound. Lower = relies more on genre tags."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.embedding.genre_weight",
        label="Genre weight",
        setting_type=SettingType.FLOAT,
        group="Hybrid Weights",
        default=0.40,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="embedding_weights",
        tooltip="Weight for genre similarity",
        description="Controls how much genre metadata (from MusicBrainz/Discogs) influences track selection. Higher values keep playlists within genre boundaries. Lower values allow cross-genre exploration based on sonic similarity."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Tower Weights (Candidate Selection - must sum to 1.0)
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.tower_weights.rhythm",
        label="Rhythm",
        setting_type=SettingType.FLOAT,
        group="Tower Weights (Candidate Selection)",
        default=0.20,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="tower_weights",
        tooltip="Weight for rhythm/tempo features",
        description="Rhythm features capture tempo, beat patterns, and rhythmic complexity. Increase for dance music or when BPM consistency matters. Includes onset detection, beat tracking, and tempo estimation."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.tower_weights.timbre",
        label="Timbre",
        setting_type=SettingType.FLOAT,
        group="Tower Weights (Candidate Selection)",
        default=0.50,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="tower_weights",
        tooltip="Weight for timbre/texture features",
        description="Timbre features capture the 'texture' and 'color' of sound - instrumentation, production style, vocal characteristics. This is usually the most important feature for finding sonically similar tracks. Based on MFCCs and spectral analysis."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.tower_weights.harmony",
        label="Harmony",
        setting_type=SettingType.FLOAT,
        group="Tower Weights (Candidate Selection)",
        default=0.30,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="tower_weights",
        tooltip="Weight for harmony/key features",
        description="Harmony features capture musical key, chord progressions, and tonal content. Important for DJ-style mixing or when you want harmonically compatible tracks. Based on chroma and tonnetz analysis."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Transition Weights (must sum to 1.0)
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.transition_weights.rhythm",
        label="Rhythm",
        setting_type=SettingType.FLOAT,
        group="Transition Weights",
        default=0.40,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="transition_weights",
        tooltip="Weight for rhythm in transitions",
        description="When scoring track-to-track transitions, this controls how much rhythm/tempo matching matters. The system compares the END of track A to the START of track B. Higher = smoother tempo flow."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.transition_weights.timbre",
        label="Timbre",
        setting_type=SettingType.FLOAT,
        group="Transition Weights",
        default=0.35,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="transition_weights",
        tooltip="Weight for timbre in transitions",
        description="Controls how much timbral similarity matters for transitions. Matching timbre between song endings and beginnings creates smoother listening experiences."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.transition_weights.harmony",
        label="Harmony",
        setting_type=SettingType.FLOAT,
        group="Transition Weights",
        default=0.25,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        normalize_group="transition_weights",
        tooltip="Weight for harmony in transitions",
        description="Controls how much harmonic/key compatibility matters for transitions. Higher values avoid jarring key changes between consecutive tracks."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate Pool
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.candidate_pool.similarity_floor",
        label="Similarity floor",
        setting_type=SettingType.FLOAT,
        group="Candidate Pool",
        default=0.20,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip="Minimum similarity to include in pool",
        description="Tracks must be at least this similar to the seed to be considered. Lower = larger pool with more variety (discovery mode). Higher = smaller pool with tighter similarity (focused mode). Try 0.10-0.15 for discovery, 0.30+ for focused."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.candidate_pool.min_sonic_similarity_narrow",
        label="Sonic floor (narrow mode)",
        setting_type=SettingType.FLOAT,
        group="Candidate Pool",
        default=0.10,
        min_value=-1.0,
        max_value=1.0,
        step=0.05,
        tooltip="Hard sonic similarity floor for narrow mode",
        description="Candidates with sonic similarity below this value are rejected before scoring when running narrow mode. Default 0.10 to prevent negative/low sonic matches from entering the pool."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.candidate_pool.min_sonic_similarity_dynamic",
        label="Sonic floor (dynamic mode)",
        setting_type=SettingType.FLOAT,
        group="Candidate Pool",
        default=0.00,
        min_value=-1.0,
        max_value=1.0,
        step=0.05,
        tooltip="Hard sonic similarity floor for dynamic mode",
        description="Candidates with sonic similarity below this value are rejected before scoring when running dynamic mode. Default 0.00 blocks negatives while allowing borderline neutral matches."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.candidate_pool.max_pool_size",
        label="Max pool size",
        setting_type=SettingType.INT,
        group="Candidate Pool",
        default=1200,
        min_value=100,
        max_value=5000,
        step=100,
        tooltip="Maximum candidates to consider",
        description="Limits how many candidate tracks are considered before final selection. Larger pools give more options but take longer. For small libraries, 500 is fine. For large libraries, 1200-2000 works well."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.candidate_pool.max_artist_fraction",
        label="Max artist fraction",
        setting_type=SettingType.FLOAT,
        group="Candidate Pool",
        default=0.125,
        min_value=0.0,
        max_value=0.5,
        step=0.025,
        tooltip="Max tracks per artist as fraction",
        description="Limits how much any single artist can dominate the playlist. 0.125 means max 12.5% from one artist (about 4 tracks in a 30-track playlist). Lower = more variety, higher = allows more focus on favorite artists."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.scoring.alpha",
        label="Alpha (seed similarity)",
        setting_type=SettingType.FLOAT,
        group="Scoring",
        default=0.55,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip="Weight for seed similarity",
        description="Controls how much each track should resemble the original seed tracks. Higher alpha = tracks stay close to seeds (focused). Lower alpha = allows more drift from seeds (discovery). Formula: score = alpha*seed_sim + beta*transition + gamma*diversity"
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.scoring.beta",
        label="Beta (transition similarity)",
        setting_type=SettingType.FLOAT,
        group="Scoring",
        default=0.55,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip="Weight for transition quality",
        description="Controls how much transition quality matters when selecting the next track. Higher beta = prioritizes smooth track-to-track flow. Lower beta = allows more abrupt changes between tracks."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.scoring.gamma",
        label="Gamma (diversity bonus)",
        setting_type=SettingType.FLOAT,
        group="Scoring",
        default=0.04,
        min_value=0.0,
        max_value=0.2,
        step=0.01,
        tooltip="Bonus for artist diversity",
        description="Adds a bonus for tracks by artists not yet in the playlist. Higher gamma = more artist variety. Keep this small (0.02-0.08) as it's additive. Too high and it overrides sonic quality."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.scoring.alpha_schedule",
        label="Alpha schedule",
        setting_type=SettingType.CHOICE,
        group="Scoring",
        default="arc",
        choices=["constant", "arc"],
        tooltip="How alpha varies over playlist",
        description="'constant' uses the same alpha throughout. 'arc' varies alpha by position: higher at start (stay close to seed), lower in middle (explore), higher at end (return home). Arc creates a narrative journey."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Constraints
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.constraints.min_gap",
        label="Min artist gap",
        setting_type=SettingType.INT,
        group="Constraints",
        default=6,
        min_value=1,
        max_value=20,
        step=1,
        tooltip="Minimum tracks between same artist",
        description="Prevents the same artist from appearing too close together. A gap of 6 means at least 6 other tracks must play before an artist repeats. Higher = more spread out. Lower = allows back-to-back from prolific artists."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.constraints.transition_floor",
        label="Transition floor",
        setting_type=SettingType.FLOAT,
        group="Constraints",
        default=0.20,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip="Minimum transition quality",
        description="Transitions below this similarity are considered 'bad'. With hard_floor=true, they're rejected. Higher values enforce smoother flow but may limit options in diverse libraries."
    ),
    SettingSpec(
        key_path="playlists.ds_pipeline.constraints.hard_floor",
        label="Hard floor",
        setting_type=SettingType.BOOL,
        group="Constraints",
        default=True,
        tooltip="Reject vs penalize bad transitions",
        description="When true, tracks with below-floor transitions are rejected entirely. When false, they're heavily penalized but still possible. Use 'false' if playlists are too short or you need more flexibility."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Genre Similarity
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.genre_similarity.enabled",
        label="Enable genre similarity",
        setting_type=SettingType.BOOL,
        group="Genre Similarity",
        default=True,
        tooltip="Use genre metadata",
        description="When enabled, genre tags from MusicBrainz and Discogs influence track selection. Disable for pure sonic similarity (ignores genre labels). Useful for cross-genre exploration."
    ),
    SettingSpec(
        key_path="playlists.genre_similarity.weight",
        label="Genre weight",
        setting_type=SettingType.FLOAT,
        group="Genre Similarity",
        default=0.50,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip="Genre influence in scoring",
        description="How much genre similarity contributes to overall track scoring. Higher = stay within genre, lower = allow cross-genre mixing based on sonic similarity."
    ),
    SettingSpec(
        key_path="playlists.genre_similarity.min_genre_similarity",
        label="Min genre similarity",
        setting_type=SettingType.FLOAT,
        group="Genre Similarity",
        default=0.30,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip="Minimum genre match threshold",
        description="Tracks must share at least this much genre similarity with the seed. Lower = more cross-genre exploration. Higher = stricter genre matching. 0.30 is a good balance."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # DS Pipeline Mode
    # ─────────────────────────────────────────────────────────────────────────
    SettingSpec(
        key_path="playlists.ds_pipeline.mode",
        label="Pipeline mode",
        setting_type=SettingType.CHOICE,
        group="Pipeline",
        default="dynamic",
        choices=["narrow", "dynamic", "discover", "sonic_only"],
        tooltip="Overall pipeline behavior",
        description="Presets that adjust multiple settings: 'narrow' = focused, stay close to seed; 'dynamic' = balanced mix; 'discover' = explore further from seed; 'sonic_only' = ignore genres entirely, pure audio matching."
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # Repair Pass - REMOVED
    # The pier-bridge system eliminates the need for repair passes.
    # Playlist ordering is now determined by beam-search bridging between
    # seed piers, with no post-hoc repair.
    # ─────────────────────────────────────────────────────────────────────────
]


def get_settings_by_group() -> dict:
    """Group settings by their UI group for easier rendering."""
    groups = {}
    for spec in SETTINGS_SCHEMA:
        if spec.group not in groups:
            groups[spec.group] = []
        groups[spec.group].append(spec)
    return groups


def get_normalize_groups() -> dict:
    """Get all normalization groups and their member settings."""
    groups = {}
    for spec in SETTINGS_SCHEMA:
        if spec.normalize_group:
            if spec.normalize_group not in groups:
                groups[spec.normalize_group] = []
            groups[spec.normalize_group].append(spec)
    return groups


def get_setting_by_key(key_path: str) -> Optional[SettingSpec]:
    """Find a setting specification by its key path."""
    for spec in SETTINGS_SCHEMA:
        if spec.key_path == key_path:
            return spec
    return None


# Secret patterns to filter from Advanced Panel
_SECRET_KEY_PATTERNS = (
    "api_key", "token", "secret", "password", "credential",
    "plex", "discogs", "lastfm", "bearer"
)


def is_secret_setting(spec: SettingSpec) -> bool:
    """Check if a setting should be treated as a secret (hidden from UI)."""
    key_lower = spec.key_path.lower()
    return any(pattern in key_lower for pattern in _SECRET_KEY_PATTERNS)


def get_visible_settings() -> List[SettingSpec]:
    """Get all settings that should be visible in the Advanced Panel (excludes secrets)."""
    return [spec for spec in SETTINGS_SCHEMA if not is_secret_setting(spec)]


def get_visible_settings_by_group() -> dict:
    """Get visible settings grouped by their UI group (excludes secrets)."""
    groups = {}
    for spec in SETTINGS_SCHEMA:
        if is_secret_setting(spec):
            continue
        if spec.group not in groups:
            groups[spec.group] = []
        groups[spec.group].append(spec)
    return groups


def get_group_key_paths(group_name: str) -> List[str]:
    """Get all key_paths for settings in a specific group."""
    groups = get_settings_by_group()
    if group_name not in groups:
        return []
    return [spec.key_path for spec in groups[group_name]]
