"""
Configuration Loader - Manages YAML configuration and environment variables
"""
import yaml
import os
from typing import Any, Optional


class Config:
    """Configuration manager for Playlist Generator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        """Validate required configuration fields"""
        required_fields = [
            ('library', 'database_path'),
            ('openai', 'api_key')
        ]

        for section, field in required_fields:
            if section not in self.config:
                raise ValueError(f"Missing configuration section: {section}")
            if field not in self.config[section]:
                raise ValueError(f"Missing configuration field: {section}.{field}")

            # Check if value is placeholder or empty
            value = self.config[section][field]
            if not value or str(value).startswith('YOUR_'):
                raise ValueError(f"Please set {section}.{field} in {self.config_path}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default
        return self.config[section].get(key, default)
    
    @property
    def library_database_path(self) -> str:
        """Get library database path"""
        return self.config['library']['database_path']

    @property
    def library_music_directory(self) -> str:
        """Get music directory path"""
        return self.config['library'].get('music_directory', 'E:\\MUSIC')
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key (with environment variable override)"""
        return os.getenv('OPENAI_API_KEY') or self.config['openai']['api_key']

    @property
    def openai_model(self) -> str:
        """Get OpenAI model"""
        return self.config['openai'].get('model', 'gpt-4o-mini')

    @property
    def lastfm_api_key(self) -> str:
        """Get Last.FM API key (with environment variable override)"""
        if 'lastfm' not in self.config or 'api_key' not in self.config.get('lastfm', {}):
            return ''
        return os.getenv('LASTFM_API_KEY') or self.config['lastfm']['api_key']

    @property
    def lastfm_username(self) -> str:
        """Get Last.FM username (with environment variable override)"""
        if 'lastfm' not in self.config or 'username' not in self.config.get('lastfm', {}):
            return ''
        return os.getenv('LASTFM_USERNAME') or self.config['lastfm']['username']

    @property
    def lastfm_history_days(self) -> int:
        """Get Last.FM history days"""
        return self.config.get('lastfm', {}).get('history_days', 90)

    @property
    def min_duration_minutes(self) -> int:
        """Get minimum playlist duration in minutes"""
        return self.config.get('playlists', {}).get('min_duration_minutes', 90)

    @property
    def min_track_duration_seconds(self) -> int:
        """Get minimum track duration in seconds (filter out short tracks)"""
        return self.config.get('playlists', {}).get('min_track_duration_seconds', 90)

    @property
    def max_track_duration_seconds(self) -> int:
        """Get maximum track duration in seconds (filter out overly long tracks)"""
        return self.config.get('playlists', {}).get('max_track_duration_seconds', 720)

    @property
    def recently_played_filter_enabled(self) -> bool:
        """Check if recently played filtering is enabled"""
        return self.config.get('playlists', {}).get('recently_played_filter', {}).get('enabled', True)

    @property
    def recently_played_lookback_days(self) -> int:
        """Get recently played filter lookback window in days (0 = all history)"""
        return self.config.get('playlists', {}).get('recently_played_filter', {}).get('lookback_days', 0)

    @property
    def recently_played_min_playcount(self) -> int:
        """Get minimum playcount threshold for recently played filter"""
        return self.config.get('playlists', {}).get('recently_played_filter', {}).get('min_playcount_threshold', 0)

    @property
    def max_tracks_per_artist(self) -> int:
        """Get maximum tracks per artist in a playlist"""
        return self.config.get('playlists', {}).get('max_tracks_per_artist', 3)

    @property
    def artist_window_size(self) -> int:
        """Get window size for artist frequency limiting"""
        return self.config.get('playlists', {}).get('artist_window_size', 8)

    @property
    def max_artist_per_window(self) -> int:
        """Get maximum times an artist can appear in the window"""
        return self.config.get('playlists', {}).get('max_artist_per_window', 1)

    @property
    def min_seed_artist_ratio(self) -> float:
        """Get minimum ratio of seed artist tracks (e.g., 0.125 = 1/8)"""
        return self.config.get('playlists', {}).get('min_seed_artist_ratio', 0.125)

    # Dynamic mode settings
    @property
    def dynamic_sonic_ratio(self) -> float:
        """Get percentage of tracks from sonic similarity in dynamic mode"""
        return self.config.get('playlists', {}).get('dynamic_mode', {}).get('sonic_ratio', 0.6)

    @property
    def dynamic_genre_ratio(self) -> float:
        """Get percentage of tracks from genre matching in dynamic mode"""
        return self.config.get('playlists', {}).get('dynamic_mode', {}).get('genre_ratio', 0.4)

    # Similarity settings
    @property
    def similarity_min_threshold(self) -> float:
        """Get minimum similarity threshold for pairing (0.0-1.0)"""
        return self.config.get('playlists', {}).get('similarity', {}).get('min_threshold', 0.5)

    @property
    def similarity_artist_direct_match(self) -> float:
        """Get similarity score for direct artist match"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_direct_match', 0.9)

    @property
    def similarity_artist_shared_base(self) -> float:
        """Get base similarity score for shared similar artists"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_shared_base', 0.4)

    @property
    def similarity_artist_shared_increment(self) -> float:
        """Get score increment per common similar artist"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_shared_increment', 0.05)

    @property
    def similarity_artist_shared_max(self) -> float:
        """Get maximum similarity score for shared similar artists"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_shared_max', 0.7)

    # Limits
    @property
    def limit_similar_tracks(self) -> int:
        """Get number of similar tracks to fetch per seed"""
        return self.config.get('playlists', {}).get('limits', {}).get('similar_tracks', 50)

    @property
    def limit_similar_artists(self) -> int:
        """Get number of similar artists to fetch"""
        return self.config.get('playlists', {}).get('limits', {}).get('similar_artists', 30)

    @property
    def limit_extension_base(self) -> int:
        """Get base number of tracks when extending playlist"""
        return self.config.get('playlists', {}).get('limits', {}).get('extension_base', 10)

    @property
    def limit_extension_increment(self) -> int:
        """Get additional tracks per extension attempt"""
        return self.config.get('playlists', {}).get('limits', {}).get('extension_increment', 20)

    # Instrumentation / artifact emission settings
    @property
    def emit_run_artifacts(self) -> bool:
        """Check if run artifact emission is enabled"""
        return self.config.get('playlists', {}).get('instrumentation', {}).get('emit_run_artifacts', False)

    @property
    def run_artifact_dir(self) -> str:
        """Get directory for run artifacts"""
        return self.config.get('playlists', {}).get('instrumentation', {}).get('artifact_dir', 'diagnostics/run_artifacts')

    # Title deduplication settings
    @property
    def title_dedupe_enabled(self) -> bool:
        """Check if title deduplication is enabled"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('enabled', True)

    @property
    def title_dedupe_threshold(self) -> int:
        """Get fuzzy match threshold for title deduplication (0-100)"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('threshold', 92)

    @property
    def title_dedupe_mode(self) -> str:
        """Get title deduplication mode ('strict' or 'loose')"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('mode', 'loose')

    @property
    def title_dedupe_short_title_min_len(self) -> int:
        """Get minimum title length for fuzzy matching (shorter titles require exact match)"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('short_title_min_len', 6)

    # ─────────────────────────────────────────────────────────────────────────
    # DS Pipeline Tuning Parameters
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ds_pipeline(self, *keys, default=None):
        """Helper to access nested ds_pipeline config values."""
        val = self.config.get('playlists', {}).get('ds_pipeline', {})
        for key in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(key, default)
            if val is default:
                return default
        return val

    @property
    def ds_tower_weights(self) -> tuple:
        """Tower weights (rhythm, timbre, harmony) for candidate selection."""
        tw = self._get_ds_pipeline('tower_weights', default={})
        return (
            tw.get('rhythm', 0.20),
            tw.get('timbre', 0.50),
            tw.get('harmony', 0.30),
        )

    @property
    def ds_transition_weights(self) -> tuple:
        """Transition weights (rhythm, timbre, harmony) for end→start scoring."""
        tw = self._get_ds_pipeline('transition_weights', default={})
        return (
            tw.get('rhythm', 0.40),
            tw.get('timbre', 0.35),
            tw.get('harmony', 0.25),
        )

    @property
    def ds_tower_pca_dims(self) -> tuple:
        """PCA dimensions (rhythm, timbre, harmony) per tower."""
        dims = self._get_ds_pipeline('tower_pca_dims', default={})
        return (
            dims.get('rhythm', 8),
            dims.get('timbre', 16),
            dims.get('harmony', 8),
        )

    @property
    def ds_embedding_sonic_components(self) -> int:
        """PCA dimensions for sonic features in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'sonic_components', default=32)

    @property
    def ds_embedding_genre_components(self) -> int:
        """PCA dimensions for genre features in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'genre_components', default=32)

    @property
    def ds_embedding_sonic_weight(self) -> float:
        """Weight for sonic similarity in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'sonic_weight', default=0.60)

    @property
    def ds_embedding_genre_weight(self) -> float:
        """Weight for genre similarity in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'genre_weight', default=0.40)

    @property
    def ds_candidate_similarity_floor(self) -> float:
        """Minimum sonic similarity for candidate pool."""
        return self._get_ds_pipeline('candidate_pool', 'similarity_floor', default=0.20)

    @property
    def ds_candidate_max_pool_size(self) -> int:
        """Maximum candidates before construction phase."""
        return self._get_ds_pipeline('candidate_pool', 'max_pool_size', default=1200)

    @property
    def ds_candidate_max_artist_fraction(self) -> float:
        """Max fraction of playlist from one artist."""
        return self._get_ds_pipeline('candidate_pool', 'max_artist_fraction', default=0.125)

    @property
    def ds_scoring_alpha(self) -> float:
        """Seed similarity weight in scoring."""
        return self._get_ds_pipeline('scoring', 'alpha', default=0.55)

    @property
    def ds_scoring_beta(self) -> float:
        """Transition similarity weight in scoring."""
        return self._get_ds_pipeline('scoring', 'beta', default=0.55)

    @property
    def ds_scoring_gamma(self) -> float:
        """Artist diversity bonus in scoring."""
        return self._get_ds_pipeline('scoring', 'gamma', default=0.04)

    @property
    def ds_scoring_alpha_schedule(self) -> str:
        """Alpha schedule: 'constant' or 'arc'."""
        return self._get_ds_pipeline('scoring', 'alpha_schedule', default='arc')

    @property
    def ds_scoring_alpha_start(self) -> float:
        """Alpha at playlist start (if arc schedule)."""
        return self._get_ds_pipeline('scoring', 'alpha_start', default=0.65)

    @property
    def ds_scoring_alpha_mid(self) -> float:
        """Alpha at playlist midpoint (if arc schedule)."""
        return self._get_ds_pipeline('scoring', 'alpha_mid', default=0.45)

    @property
    def ds_scoring_alpha_end(self) -> float:
        """Alpha at playlist end (if arc schedule)."""
        return self._get_ds_pipeline('scoring', 'alpha_end', default=0.60)

    @property
    def ds_scoring_arc_midpoint(self) -> float:
        """Where midpoint falls in playlist (0.0-1.0)."""
        return self._get_ds_pipeline('scoring', 'arc_midpoint', default=0.55)

    @property
    def ds_constraints_min_gap(self) -> int:
        """Minimum positions between same artist."""
        return self._get_ds_pipeline('constraints', 'min_gap', default=6)

    @property
    def ds_constraints_hard_floor(self) -> bool:
        """Reject (True) vs penalize (False) below floor."""
        return self._get_ds_pipeline('constraints', 'hard_floor', default=True)

    @property
    def ds_constraints_transition_floor(self) -> float:
        """Minimum acceptable transition similarity."""
        return self._get_ds_pipeline('constraints', 'transition_floor', default=0.20)

    @property
    def ds_constraints_center_transitions(self) -> bool:
        """Whether to center X_start/X_end matrices."""
        return self._get_ds_pipeline('constraints', 'center_transitions', default=True)

    @property
    def ds_repair_enabled(self) -> bool:
        """Whether repair pass is enabled."""
        return self._get_ds_pipeline('repair', 'enabled', default=True)

    @property
    def ds_repair_max_iters(self) -> int:
        """Maximum repair iterations."""
        return self._get_ds_pipeline('repair', 'max_iters', default=5)

    @property
    def ds_repair_max_edges(self) -> int:
        """Maximum edges to fix per iteration."""
        return self._get_ds_pipeline('repair', 'max_edges', default=5)

    @property
    def ds_repair_objective(self) -> str:
        """Repair objective: 'gap_penalty' or 'below_floor_first'."""
        return self._get_ds_pipeline('repair', 'objective', default='gap_penalty')

    def get_ds_tuning_dict(self) -> dict:
        """Return all DS tuning parameters as a dict for pipeline consumption."""
        return {
            'tower_weights': self.ds_tower_weights,
            'transition_weights': self.ds_transition_weights,
            'tower_pca_dims': self.ds_tower_pca_dims,
            'embedding': {
                'sonic_components': self.ds_embedding_sonic_components,
                'genre_components': self.ds_embedding_genre_components,
                'sonic_weight': self.ds_embedding_sonic_weight,
                'genre_weight': self.ds_embedding_genre_weight,
            },
            'candidate_pool': {
                'similarity_floor': self.ds_candidate_similarity_floor,
                'max_pool_size': self.ds_candidate_max_pool_size,
                'max_artist_fraction': self.ds_candidate_max_artist_fraction,
            },
            'scoring': {
                'alpha': self.ds_scoring_alpha,
                'beta': self.ds_scoring_beta,
                'gamma': self.ds_scoring_gamma,
                'alpha_schedule': self.ds_scoring_alpha_schedule,
                'alpha_start': self.ds_scoring_alpha_start,
                'alpha_mid': self.ds_scoring_alpha_mid,
                'alpha_end': self.ds_scoring_alpha_end,
                'arc_midpoint': self.ds_scoring_arc_midpoint,
            },
            'constraints': {
                'min_gap': self.ds_constraints_min_gap,
                'hard_floor': self.ds_constraints_hard_floor,
                'transition_floor': self.ds_constraints_transition_floor,
                'center_transitions': self.ds_constraints_center_transitions,
            },
            'repair': {
                'enabled': self.ds_repair_enabled,
                'max_iters': self.ds_repair_max_iters,
                'max_edges': self.ds_repair_max_edges,
                'objective': self.ds_repair_objective,
            },
        }

    def __repr__(self) -> str:
        """String representation (hides sensitive data)"""
        return f"Config(database={self.library_database_path}, lastfm_user={self.lastfm_username})"


# Example usage
if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    try:
        config = Config()
        logger.info(f"Configuration loaded successfully: {config}")
        logger.info(f"Database path: {config.library_database_path}")
        logger.info(f"Music directory: {config.library_music_directory}")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
