"""
Playlist Generation Strategies
==============================

Extracted from playlist_generator.py (Phase 4.1).

This module provides strategy pattern implementations for different
playlist generation modes.

Available strategies:
- ArtistPlaylistStrategy: Artist-based playlists with style clustering
- GenrePlaylistStrategy: Genre-based playlists
- BatchPlaylistStrategy: Batch generation from listening history
- HistoryPlaylistStrategy: History-based playlists

Public API:
-----------
Base:
    PlaylistGenerationStrategy
    PlaylistRequest
    PlaylistResult

Concrete Strategies:
    ArtistPlaylistStrategy
    GenrePlaylistStrategy
    BatchPlaylistStrategy
    HistoryPlaylistStrategy

Factory:
    PlaylistFactory
"""

from .base_strategy import (
    PlaylistGenerationStrategy,
    PlaylistRequest,
    PlaylistResult,
)

__all__ = [
    "PlaylistGenerationStrategy",
    "PlaylistRequest",
    "PlaylistResult",
]
