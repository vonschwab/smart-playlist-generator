"""
Playlist Factory
================

Extracted from playlist_generator.py (Phase 4.1).

This module provides a factory for creating playlists using the
strategy pattern. Different playlist generation modes are handled
by specialized strategy implementations.
"""

from __future__ import annotations

import logging
from typing import Any, List

from .strategies.base_strategy import (
    PlaylistGenerationStrategy,
    PlaylistRequest,
    PlaylistResult,
)

logger = logging.getLogger(__name__)


class PlaylistFactory:
    """Factory for creating playlists via strategy pattern.

    The factory maintains a registry of strategies and delegates
    playlist generation to the appropriate strategy based on the
    request mode.

    Usage:
        factory = PlaylistFactory(config)
        factory.register_strategy(ArtistPlaylistStrategy(config))
        factory.register_strategy(GenrePlaylistStrategy(config))

        request = PlaylistRequest(mode="artist", num_tracks=30, config=config, artist="Bill Evans")
        result = factory.create(request)
    """

    def __init__(self, config: Any):
        """Initialize playlist factory.

        Args:
            config: Configuration object (typically Config instance)
        """
        self.config = config
        self.strategies: List[PlaylistGenerationStrategy] = []
        self.logger = logging.getLogger(__name__)

    def register_strategy(self, strategy: PlaylistGenerationStrategy) -> None:
        """Register a playlist generation strategy.

        Args:
            strategy: Strategy to register
        """
        self.strategies.append(strategy)
        self.logger.debug(f"Registered strategy: {strategy.__class__.__name__}")

    def create(self, request: PlaylistRequest) -> PlaylistResult:
        """Create playlist by delegating to appropriate strategy.

        Args:
            request: Playlist generation request

        Returns:
            PlaylistResult with generated playlist

        Raises:
            ValueError: If no strategy can handle the request
        """
        # Find matching strategy
        for strategy in self.strategies:
            if strategy.can_handle(request):
                self.logger.info(
                    f"Creating playlist: mode={request.mode} "
                    f"tracks={request.num_tracks} "
                    f"strategy={strategy.__class__.__name__}"
                )
                return strategy.execute(request)

        # No strategy found
        error = f"No strategy can handle request with mode '{request.mode}'"
        self.logger.error(error)
        raise ValueError(error)

    def get_supported_modes(self) -> List[str]:
        """Get list of supported playlist generation modes.

        Returns:
            List of mode names (requires strategies to have 'mode' attribute)
        """
        modes = []
        for strategy in self.strategies:
            # Try to get mode from strategy
            if hasattr(strategy, 'mode'):
                modes.append(strategy.mode)
            else:
                modes.append(strategy.__class__.__name__)
        return modes
