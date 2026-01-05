"""
DS Pipeline Builder
===================

Extracted from pipeline.py (Phase 5.2).

This module provides a builder pattern for DS pipeline configuration,
simplifying the complex parameter list and making the API more maintainable.

Usage:
    pipeline = (DSPipelineBuilder()
        .with_artifacts("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
        .with_seed("d97b2f9e9f9c6c56e09135ecf9c30876")
        .with_mode("dynamic")
        .with_num_tracks(30)
        .with_random_seed(42)
        .build())

    result = pipeline.execute()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.playlist.pier_bridge_builder import PierBridgeConfig

logger = logging.getLogger(__name__)


@dataclass
class DSPipelineRequest:
    """Request for DS pipeline execution with all configuration."""

    # Required parameters
    artifact_path: str | Path
    seed_track_id: str
    num_tracks: int
    mode: str
    random_seed: int

    # Optional configuration
    overrides: Optional[Dict[str, Any]] = None
    allowed_track_ids: Optional[List[str]] = None
    excluded_track_ids: Optional[Set[str]] = None
    single_artist: bool = False
    sonic_variant: Optional[str] = None
    anchor_seed_ids: Optional[List[str]] = None
    pier_bridge_config: Optional[PierBridgeConfig] = None

    # Hybrid-level tuning dials
    sonic_weight: Optional[float] = None
    genre_weight: Optional[float] = None
    min_genre_similarity: Optional[float] = None
    genre_method: Optional[str] = None

    # Advanced parameters
    allowed_track_ids_set: Optional[Set[str]] = None
    internal_connector_ids: Optional[List[str]] = None
    internal_connector_max_per_segment: int = 0
    internal_connector_priority: bool = True

    # Audit and dry-run
    dry_run: bool = False
    pool_source: Optional[str] = None
    artist_style_enabled: bool = False
    artist_playlist: bool = False
    audit_context_extra: Optional[Dict[str, Any]] = None


class DSPipelineBuilder:
    """Builder for DS pipeline with fluent API.

    Simplifies construction of DS pipeline requests by providing
    a chainable interface for configuration.

    Example:
        builder = (DSPipelineBuilder()
            .with_artifacts(artifact_path)
            .with_seed(seed_track_id)
            .with_mode("dynamic")
            .with_num_tracks(30)
            .with_random_seed(42)
            .with_sonic_weight(0.8)
            .with_genre_weight(0.2))

        request = builder.build()

        # Then execute with pipeline
        from src.playlist.pipeline import generate_playlist_ds
        result = generate_playlist_ds(**request.__dict__)
    """

    def __init__(self):
        """Initialize builder with default values."""
        self._artifact_path: Optional[str | Path] = None
        self._seed_track_id: Optional[str] = None
        self._num_tracks: int = 30
        self._mode: str = "dynamic"
        self._random_seed: int = 42

        # Optional parameters
        self._overrides: Optional[Dict[str, Any]] = None
        self._allowed_track_ids: Optional[List[str]] = None
        self._excluded_track_ids: Optional[Set[str]] = None
        self._single_artist: bool = False
        self._sonic_variant: Optional[str] = None
        self._anchor_seed_ids: Optional[List[str]] = None
        self._pier_bridge_config: Optional[PierBridgeConfig] = None

        # Hybrid tuning
        self._sonic_weight: Optional[float] = None
        self._genre_weight: Optional[float] = None
        self._min_genre_similarity: Optional[float] = None
        self._genre_method: Optional[str] = None

        # Advanced
        self._allowed_track_ids_set: Optional[Set[str]] = None
        self._internal_connector_ids: Optional[List[str]] = None
        self._internal_connector_max_per_segment: int = 0
        self._internal_connector_priority: bool = True

        # Audit
        self._dry_run: bool = False
        self._pool_source: Optional[str] = None
        self._artist_style_enabled: bool = False
        self._artist_playlist: bool = False
        self._audit_context_extra: Optional[Dict[str, Any]] = None

    def with_artifacts(self, path: str | Path) -> DSPipelineBuilder:
        """Set artifact bundle path.

        Args:
            path: Path to artifact bundle (.npz file)

        Returns:
            Self for chaining
        """
        self._artifact_path = path
        return self

    def with_seed(self, track_id: str) -> DSPipelineBuilder:
        """Set seed track ID.

        Args:
            track_id: Primary seed track ID

        Returns:
            Self for chaining
        """
        self._seed_track_id = track_id
        return self

    def with_num_tracks(self, num: int) -> DSPipelineBuilder:
        """Set number of tracks in playlist.

        Args:
            num: Target playlist length

        Returns:
            Self for chaining
        """
        self._num_tracks = num
        return self

    def with_mode(self, mode: str) -> DSPipelineBuilder:
        """Set playlist mode.

        Args:
            mode: Playlist mode (dynamic, narrow, sonic_only, etc.)

        Returns:
            Self for chaining
        """
        self._mode = mode
        return self

    def with_random_seed(self, seed: int) -> DSPipelineBuilder:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value

        Returns:
            Self for chaining
        """
        self._random_seed = seed
        return self

    def with_overrides(self, overrides: Dict[str, Any]) -> DSPipelineBuilder:
        """Set configuration overrides.

        Args:
            overrides: Configuration override dictionary

        Returns:
            Self for chaining
        """
        self._overrides = overrides
        return self

    def with_allowed_tracks(self, track_ids: List[str]) -> DSPipelineBuilder:
        """Set allowed track IDs.

        Args:
            track_ids: List of allowed track IDs

        Returns:
            Self for chaining
        """
        self._allowed_track_ids = track_ids
        self._allowed_track_ids_set = set(track_ids) if track_ids else None
        return self

    def with_excluded_tracks(self, track_ids: Set[str]) -> DSPipelineBuilder:
        """Set excluded track IDs.

        Args:
            track_ids: Set of excluded track IDs

        Returns:
            Self for chaining
        """
        self._excluded_track_ids = track_ids
        return self

    def with_single_artist(self, enabled: bool = True) -> DSPipelineBuilder:
        """Enable single-artist constraint.

        Args:
            enabled: Whether to enforce single artist

        Returns:
            Self for chaining
        """
        self._single_artist = enabled
        return self

    def with_sonic_variant(self, variant: str) -> DSPipelineBuilder:
        """Set sonic variant.

        Args:
            variant: Sonic variant name (tower_pca, beat3tower, etc.)

        Returns:
            Self for chaining
        """
        self._sonic_variant = variant
        return self

    def with_anchor_seeds(self, seed_ids: List[str]) -> DSPipelineBuilder:
        """Set anchor seed IDs (multi-seed playlists).

        Args:
            seed_ids: List of anchor seed track IDs

        Returns:
            Self for chaining
        """
        self._anchor_seed_ids = seed_ids
        return self

    def with_pier_bridge_config(self, config: PierBridgeConfig) -> DSPipelineBuilder:
        """Set pier-bridge configuration.

        Args:
            config: PierBridgeConfig instance

        Returns:
            Self for chaining
        """
        self._pier_bridge_config = config
        return self

    def with_sonic_weight(self, weight: float) -> DSPipelineBuilder:
        """Set sonic similarity weight.

        Args:
            weight: Sonic weight (0.0-1.0)

        Returns:
            Self for chaining
        """
        self._sonic_weight = weight
        return self

    def with_genre_weight(self, weight: float) -> DSPipelineBuilder:
        """Set genre similarity weight.

        Args:
            weight: Genre weight (0.0-1.0)

        Returns:
            Self for chaining
        """
        self._genre_weight = weight
        return self

    def with_min_genre_similarity(self, threshold: float) -> DSPipelineBuilder:
        """Set minimum genre similarity threshold.

        Args:
            threshold: Minimum genre similarity

        Returns:
            Self for chaining
        """
        self._min_genre_similarity = threshold
        return self

    def with_genre_method(self, method: str) -> DSPipelineBuilder:
        """Set genre similarity method.

        Args:
            method: Genre similarity method (jaccard, cosine, etc.)

        Returns:
            Self for chaining
        """
        self._genre_method = method
        return self

    def with_internal_connectors(
        self,
        connector_ids: List[str],
        max_per_segment: int = 0,
        priority: bool = True
    ) -> DSPipelineBuilder:
        """Set internal connector configuration.

        Args:
            connector_ids: List of internal connector track IDs
            max_per_segment: Maximum connectors per segment
            priority: Whether to prioritize connectors

        Returns:
            Self for chaining
        """
        self._internal_connector_ids = connector_ids
        self._internal_connector_max_per_segment = max_per_segment
        self._internal_connector_priority = priority
        return self

    def with_dry_run(self, enabled: bool = True) -> DSPipelineBuilder:
        """Enable dry-run mode.

        Args:
            enabled: Whether to run in dry-run mode

        Returns:
            Self for chaining
        """
        self._dry_run = enabled
        return self

    def with_artist_style(self, enabled: bool = True) -> DSPipelineBuilder:
        """Enable artist style clustering.

        Args:
            enabled: Whether to use artist style clustering

        Returns:
            Self for chaining
        """
        self._artist_style_enabled = enabled
        return self

    def with_artist_playlist(self, enabled: bool = True) -> DSPipelineBuilder:
        """Mark as artist playlist.

        Args:
            enabled: Whether this is an artist playlist

        Returns:
            Self for chaining
        """
        self._artist_playlist = enabled
        return self

    def with_audit_context(self, context: Dict[str, Any]) -> DSPipelineBuilder:
        """Set audit context.

        Args:
            context: Extra audit context data

        Returns:
            Self for chaining
        """
        self._audit_context_extra = context
        return self

    def build(self) -> DSPipelineRequest:
        """Build DSPipelineRequest with validation.

        Returns:
            Configured DSPipelineRequest

        Raises:
            ValueError: If required parameters not set
        """
        # Validate required parameters
        if self._artifact_path is None:
            raise ValueError("Artifact path is required (use with_artifacts)")
        if self._seed_track_id is None:
            raise ValueError("Seed track ID is required (use with_seed)")

        return DSPipelineRequest(
            artifact_path=self._artifact_path,
            seed_track_id=self._seed_track_id,
            num_tracks=self._num_tracks,
            mode=self._mode,
            random_seed=self._random_seed,
            overrides=self._overrides,
            allowed_track_ids=self._allowed_track_ids,
            excluded_track_ids=self._excluded_track_ids,
            single_artist=self._single_artist,
            sonic_variant=self._sonic_variant,
            anchor_seed_ids=self._anchor_seed_ids,
            pier_bridge_config=self._pier_bridge_config,
            sonic_weight=self._sonic_weight,
            genre_weight=self._genre_weight,
            min_genre_similarity=self._min_genre_similarity,
            genre_method=self._genre_method,
            allowed_track_ids_set=self._allowed_track_ids_set,
            internal_connector_ids=self._internal_connector_ids,
            internal_connector_max_per_segment=self._internal_connector_max_per_segment,
            internal_connector_priority=self._internal_connector_priority,
            dry_run=self._dry_run,
            pool_source=self._pool_source,
            artist_style_enabled=self._artist_style_enabled,
            artist_playlist=self._artist_playlist,
            audit_context_extra=self._audit_context_extra,
        )
