"""
Base Strategy for Playlist Generation
=====================================

Extracted from playlist_generator.py (Phase 4.1).

This module defines the abstract base class for playlist generation
strategies and the request/result data structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlaylistRequest:
    """Request for playlist generation.

    Contains all parameters needed to generate a playlist,
    with mode-specific fields optional.
    """

    mode: str
    """Playlist generation mode (artist, genre, batch, history, etc.)."""

    num_tracks: int
    """Target number of tracks in playlist."""

    config: Dict[str, Any]
    """Configuration dictionary from Config object."""

    # Mode-specific parameters
    artist: Optional[str] = None
    """Artist name for artist-based playlists."""

    genre: Optional[str] = None
    """Genre tag for genre-based playlists."""

    seed_track: Optional[str] = None
    """Seed track ID for seed-based playlists."""

    history_days: Optional[int] = None
    """Number of days of history to analyze."""

    batch_size: Optional[int] = None
    """Number of playlists to generate in batch mode."""

    # Additional optional parameters
    name_prefix: Optional[str] = None
    """Custom name prefix for generated playlist."""

    overrides: Optional[Dict[str, Any]] = None
    """Runtime configuration overrides."""


@dataclass
class PlaylistResult:
    """Result of playlist generation.

    Contains the generated playlist, metadata, and diagnostics.
    """

    track_ids: List[str]
    """Ordered list of track IDs in playlist."""

    name: str
    """Playlist name."""

    mode: str
    """Mode used to generate playlist."""

    seed_info: Dict[str, Any] = field(default_factory=dict)
    """Information about seeds used."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    """Quality metrics (diversity, coherence, etc.)."""

    diagnostics: Dict[str, Any] = field(default_factory=dict)
    """Detailed diagnostics from generation process."""

    success: bool = True
    """Whether generation succeeded."""

    failure_reason: Optional[str] = None
    """Reason for failure if success=False."""


class PlaylistGenerationStrategy(ABC):
    """Abstract base class for playlist generation strategies.

    Each strategy implements a specific mode of playlist generation
    (artist-based, genre-based, batch, etc.).

    Subclasses must implement:
    - can_handle(): Check if strategy can handle a request
    - execute(): Generate playlist for a request
    """

    def __init__(self, config: Any):
        """Initialize strategy with configuration.

        Args:
            config: Configuration object (typically Config instance)
        """
        self.config = config

    @abstractmethod
    def can_handle(self, request: PlaylistRequest) -> bool:
        """Check if this strategy can handle the request.

        Args:
            request: Playlist generation request

        Returns:
            True if this strategy can handle the request
        """
        pass

    @abstractmethod
    def execute(self, request: PlaylistRequest) -> PlaylistResult:
        """Execute playlist generation.

        Args:
            request: Playlist generation request

        Returns:
            PlaylistResult with generated playlist and metadata

        Raises:
            ValueError: If request is invalid
            RuntimeError: If generation fails
        """
        pass

    def _create_failure_result(
        self,
        request: PlaylistRequest,
        reason: str,
    ) -> PlaylistResult:
        """Create a failure result.

        Args:
            request: Original request
            reason: Failure reason

        Returns:
            PlaylistResult with success=False
        """
        return PlaylistResult(
            track_ids=[],
            name="",
            mode=request.mode,
            success=False,
            failure_reason=reason,
        )
