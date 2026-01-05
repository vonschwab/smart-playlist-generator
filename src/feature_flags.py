"""Feature flag system for safe refactoring transitions.

This module provides a centralized feature flag system that allows gradual
migration from legacy code to refactored implementations. All flags default
to False (legacy behavior) for maximum safety.

Usage:
    from src.feature_flags import FeatureFlags

    # Load from config
    config = Config("config.yaml")
    flags = FeatureFlags(config.config)

    # Check flags
    if flags.use_unified_genre_normalization():
        # Use new implementation
        ...
    else:
        # Use legacy implementation
        ...

Configuration:
    Add to config.yaml under 'experimental' section:

    experimental:
      use_unified_genre_normalization: false
      use_unified_artist_normalization: false
      ...
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Feature flag system for safe refactoring transitions.

    All flags default to False (legacy behavior) unless explicitly enabled
    in configuration. This ensures that refactoring doesn't break existing
    functionality unless intentionally activated.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize feature flags from configuration.

        Args:
            config: Configuration dictionary (usually from Config.config)
        """
        self.flags = config.get('experimental', {})

        # Log active flags on initialization
        active_flags = [name for name, value in self.flags.items() if value]
        if active_flags:
            logger.info(f"Active feature flags: {', '.join(active_flags)}")
        else:
            logger.debug("No experimental features enabled (all flags default to legacy)")

    # ==================================================================================
    # Genre/Artist Normalization Flags
    # ==================================================================================

    def use_unified_genre_normalization(self) -> bool:
        """Use unified genre normalization module (Phase 2.1).

        When enabled, uses `src.genre.normalize_unified` instead of legacy
        `genre_normalization.py` and `genre/normalize.py` modules.

        Returns:
            True if unified normalization should be used
        """
        return self.flags.get('use_unified_genre_normalization', False)

    def use_unified_artist_normalization(self) -> bool:
        """Use unified artist normalization module (Phase 2.2).

        When enabled, uses consolidated artist normalization in `string_utils.py`
        with all features (ensemble suffix handling, collaboration parsing).

        Returns:
            True if unified artist normalization should be used
        """
        return self.flags.get('use_unified_artist_normalization', False)

    # ==================================================================================
    # Similarity Calculation Flags
    # ==================================================================================

    def use_unified_genre_similarity(self) -> bool:
        """Use unified genre similarity module (Phase 2.3).

        When enabled, uses `src.similarity.genre.GenreSimilarityCalculator`
        instead of multiple scattered implementations.

        Returns:
            True if unified genre similarity should be used
        """
        return self.flags.get('use_unified_genre_similarity', False)

    # ==================================================================================
    # Pier-Bridge Refactoring Flags
    # ==================================================================================

    def use_extracted_pier_bridge_scoring(self) -> bool:
        """Use extracted pier-bridge scoring modules (Phase 3.1).

        When enabled, uses:
        - `src.playlist.scoring.transition_scoring`
        - `src.playlist.scoring.bridge_scoring`
        - `src.playlist.scoring.constraints`

        Instead of inline scoring logic in `pier_bridge_builder.py`.

        Returns:
            True if extracted scoring modules should be used
        """
        return self.flags.get('use_extracted_pier_bridge_scoring', False)

    def use_extracted_segment_pool(self) -> bool:
        """Use extracted segment pool builder (Phase 3.2).

        When enabled, uses `src.playlist.segment_pool_builder.SegmentCandidatePoolBuilder`
        instead of inline pool building logic.

        Returns:
            True if extracted segment pool builder should be used
        """
        return self.flags.get('use_extracted_segment_pool', False)

    def use_extracted_pier_bridge_diagnostics(self) -> bool:
        """Use extracted pier-bridge diagnostics (Phase 3.3).

        When enabled, uses `src.playlist.pier_bridge_diagnostics.PierBridgeDiagnosticsCollector`
        with dependency injection instead of inline diagnostics.

        Returns:
            True if extracted diagnostics should be used
        """
        return self.flags.get('use_extracted_pier_bridge_diagnostics', False)

    # ==================================================================================
    # Playlist Generation Flags
    # ==================================================================================

    def use_new_candidate_generator(self) -> bool:
        """Use enhanced candidate generator (Phase 4.2).

        When enabled, uses refactored `src.playlist.candidate_generator.CandidatePoolGenerator`
        with cleaner mode separation.

        Returns:
            True if new candidate generator should be used
        """
        return self.flags.get('use_new_candidate_generator', False)

    def use_playlist_factory(self) -> bool:
        """Use playlist factory with strategy pattern (Phase 4.1).

        When enabled, uses `src.playlist.playlist_factory.PlaylistFactory`
        with separate strategy classes for each mode (artist, genre, batch, history).

        Returns:
            True if playlist factory should be used
        """
        return self.flags.get('use_playlist_factory', False)

    def use_filtering_pipeline(self) -> bool:
        """Use composable filter pipeline (Phase 4.3).

        When enabled, uses `src.playlist.filtering.FilterPipeline` with
        Chain of Responsibility pattern instead of inline filtering.

        Returns:
            True if filtering pipeline should be used
        """
        return self.flags.get('use_filtering_pipeline', False)

    def use_history_repository(self) -> bool:
        """Use listening history repository (Phase 4.4).

        When enabled, uses `src.repositories.listening_history_repository.ListeningHistoryRepository`
        with dependency injection instead of direct client access.

        Returns:
            True if history repository should be used
        """
        return self.flags.get('use_history_repository', False)

    # ==================================================================================
    # Pipeline Refactoring Flags
    # ==================================================================================

    def use_config_resolver(self) -> bool:
        """Use explicit config resolution (Phase 5.1).

        When enabled, uses `src.playlist.config_resolver.ConfigResolver`
        with explicit precedence rules instead of 6-level fallback chains.

        Returns:
            True if config resolver should be used
        """
        return self.flags.get('use_config_resolver', False)

    def use_pipeline_builder(self) -> bool:
        """Use DS pipeline builder pattern (Phase 5.2).

        When enabled, uses `src.playlist.ds_pipeline_builder.DSPipelineBuilder`
        with fluent API instead of direct construction.

        Returns:
            True if pipeline builder should be used
        """
        return self.flags.get('use_pipeline_builder', False)

    def use_variant_cache(self) -> bool:
        """Use sonic variant caching (Phase 5.3).

        When enabled, uses `src.similarity.variant_cache.VariantCache`
        to avoid recomputing sonic variants for same artifacts.

        Returns:
            True if variant caching should be used
        """
        return self.flags.get('use_variant_cache', False)

    # ==================================================================================
    # Configuration Facades
    # ==================================================================================

    def use_typed_config(self) -> bool:
        """Use typed configuration facades (Phase 2.4).

        When enabled, uses `src.config.playlist_config.PlaylistGenerationConfig`
        with dataclasses instead of raw dict access.

        Returns:
            True if typed config should be used
        """
        return self.flags.get('use_typed_config', False)

    # ==================================================================================
    # Utility Methods
    # ==================================================================================

    def is_any_enabled(self) -> bool:
        """Check if any experimental features are enabled.

        Returns:
            True if at least one flag is enabled
        """
        return any(self.flags.values())

    def get_active_flags(self) -> Dict[str, bool]:
        """Get dictionary of all active (enabled) flags.

        Returns:
            Dictionary of flag_name: True for all enabled flags
        """
        return {name: value for name, value in self.flags.items() if value}

    def get_all_flags(self) -> Dict[str, bool]:
        """Get dictionary of all flags with their values.

        Returns:
            Dictionary of all flags (enabled and disabled)
        """
        return dict(self.flags)

    def __repr__(self) -> str:
        """String representation showing active flags."""
        active = self.get_active_flags()
        if not active:
            return "FeatureFlags(no active flags)"
        return f"FeatureFlags(active: {list(active.keys())})"
