"""
Playlist ordering module for track sequencing.

This module provides algorithms for ordering tracks to maximize sonic cohesion:
- Greedy nearest-neighbor ordering
- TSP-based optimization
- Transition matrix building

Migrated from src/playlist_generator.py ordering methods (Phase 7).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import numpy as np

from src.string_utils import normalize_song_title

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderingConfig:
    """Configuration for playlist ordering."""
    method: str = "tsp"  # "tsp" or "greedy"
    limit_similar_tracks: int = 30
    artist_window_size: int = 8
    use_intelligent_selection: bool = True  # Select best-connected subset before TSP


@dataclass(frozen=True)
class OrderingResult:
    """
    Result of ordering operation.

    Attributes:
        ordered_tracks: Tracks in optimal sequence
        stats: Diagnostic statistics (avg_similarity, method_used, elapsed_time, etc.)
    """
    ordered_tracks: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


def order_by_sequential_similarity(
    *,
    tracks: List[Dict[str, Any]],
    seeds: List[Dict[str, Any]],
    library_client,
    limit_similar_tracks: int = 30,
) -> OrderingResult:
    """
    Order tracks using greedy nearest-neighbor approach.

    In dynamic mode, this ensures genre-matched songs are interleaved based on
    their sonic similarity rather than clustered at the end.

    Uses a greedy nearest-neighbor approach starting from seed tracks.

    Args:
        tracks: Tracks to order
        seeds: Seed tracks to start from
        library_client: Library client for similarity lookups
        limit_similar_tracks: Max similar tracks to fetch per track

    Returns:
        OrderingResult with ordered tracks and statistics
    """
    if not tracks:
        return OrderingResult(ordered_tracks=[], stats={})

    # Separate sonic and genre tracks for smarter ordering
    sonic_tracks = [t for t in tracks if t.get('source') == 'sonic']
    genre_tracks = [t for t in tracks if t.get('source') == 'genre']
    other_tracks = [t for t in tracks if t.get('source') not in ['sonic', 'genre']]

    has_genre_tracks = len(genre_tracks) > 0

    if has_genre_tracks:
        logger.info(f"   Ordering {len(sonic_tracks)} sonic + {len(genre_tracks)} genre-matched tracks (interleaving by similarity)")

    # Start with the first seed track
    ordered = [seeds[0]] if seeds else [tracks[0]]

    # Build remaining pool (sonic + other first, we'll interleave genre tracks)
    remaining_sonic = [t for t in sonic_tracks + other_tracks if t.get('rating_key') != ordered[0].get('rating_key')]
    remaining_genre = list(genre_tracks) if has_genre_tracks else []

    # Track normalized titles to prevent duplicates
    used_titles = {normalize_song_title(ordered[0].get('title', ''))}
    used_keys = {ordered[0].get('rating_key')}

    logger.info(f"   Starting sequential ordering with: {ordered[0].get('artist')} - {ordered[0].get('title')}")

    # OPTIMIZATION: Pre-fetch similarity data for all tracks to avoid repeated API calls
    logger.info(f"   Building similarity cache for {len(tracks)} tracks...")
    similarity_cache = {}
    all_track_keys = [t.get('rating_key') for t in tracks if t.get('rating_key')]

    for track_key in all_track_keys:
        try:
            similar = library_client.get_similar_tracks(track_key, limit=limit_similar_tracks)
            similarity_cache[track_key] = similar if similar else []
        except Exception as e:
            logger.debug(f"   Failed to fetch similarity for track {track_key}: {e}")
            similarity_cache[track_key] = []

    logger.info(f"   Cache complete: {len(similarity_cache)} tracks indexed")

    # Track ordering statistics for summary
    ordering_stats = {
        'similar_matches': 0,
        'sonic_fallbacks': 0,
        'genre_fallbacks': 0,
        'no_api_fallbacks': 0
    }

    # Greedily add the most similar track to the last track
    # We'll consider both sonic and genre tracks, picking the most similar
    while remaining_sonic or remaining_genre:
        last_track = ordered[-1]
        last_track_key = last_track.get('rating_key')

        # Look up similar tracks from pre-computed cache
        similar_to_last = similarity_cache.get(last_track_key, [])

        next_track = None
        next_track_source = None

        if similar_to_last:
            # Build a map of rating_key to distance score
            distance_map = {t.get('rating_key'): t.get('distance') for t in similar_to_last if t.get('distance') is not None}
            similar_keys = {t.get('rating_key') for t in similar_to_last}

            # Find the BEST similar track from BOTH sonic and genre pools
            # This ensures genre tracks are interleaved when they're actually similar
            best_track = None
            best_distance = float('inf')
            best_source_pool = None

            # Check sonic tracks first
            for track in remaining_sonic:
                if track.get('rating_key') in similar_keys:
                    track_title_normalized = normalize_song_title(track.get('title', ''))
                    if track_title_normalized in used_titles:
                        continue

                    distance = distance_map.get(track.get('rating_key'), float('inf'))
                    if distance < best_distance:
                        best_distance = distance
                        best_track = track
                        best_source_pool = 'sonic'

            # Also check genre tracks - they might be more similar!
            for track in remaining_genre:
                if track.get('rating_key') in similar_keys:
                    track_title_normalized = normalize_song_title(track.get('title', ''))
                    if track_title_normalized in used_titles:
                        continue

                    distance = distance_map.get(track.get('rating_key'), float('inf'))
                    if distance < best_distance:
                        best_distance = distance
                        best_track = track
                        best_source_pool = 'genre'

            if best_track:
                next_track = best_track
                next_track_source = best_source_pool
                ordering_stats['similar_matches'] += 1
            else:
                # No similar track found in either pool, use fallback from sonic first
                fallback = None
                fallback_source = None

                # Try sonic pool first
                for i, track in enumerate(remaining_sonic):
                    track_title_normalized = normalize_song_title(track.get('title', ''))
                    if track_title_normalized not in used_titles:
                        fallback = remaining_sonic.pop(i)
                        fallback_source = 'sonic'
                        break

                # If no sonic available, try genre pool
                if not fallback:
                    for i, track in enumerate(remaining_genre):
                        track_title_normalized = normalize_song_title(track.get('title', ''))
                        if track_title_normalized not in used_titles:
                            fallback = remaining_genre.pop(i)
                            fallback_source = 'genre'
                            break

                # Last resort: take first available
                if not fallback:
                    if remaining_sonic:
                        fallback = remaining_sonic.pop(0)
                        fallback_source = 'sonic'
                    elif remaining_genre:
                        fallback = remaining_genre.pop(0)
                        fallback_source = 'genre'

                if fallback:
                    next_track = fallback
                    next_track_source = fallback_source
                    if fallback_source == 'sonic':
                        ordering_stats['sonic_fallbacks'] += 1
                    else:
                        ordering_stats['genre_fallbacks'] += 1
        else:
            # Can't get similar tracks, use fallback strategy
            fallback = None
            fallback_source = None

            # Try sonic pool first
            for i, track in enumerate(remaining_sonic):
                track_title_normalized = normalize_song_title(track.get('title', ''))
                if track_title_normalized not in used_titles:
                    fallback = remaining_sonic.pop(i)
                    fallback_source = 'sonic'
                    break

            # If no sonic available, try genre pool
            if not fallback:
                for i, track in enumerate(remaining_genre):
                    track_title_normalized = normalize_song_title(track.get('title', ''))
                    if track_title_normalized not in used_titles:
                        fallback = remaining_genre.pop(i)
                        fallback_source = 'genre'
                        break

            # Last resort
            if not fallback:
                if remaining_sonic:
                    fallback = remaining_sonic.pop(0)
                    fallback_source = 'sonic'
                elif remaining_genre:
                    fallback = remaining_genre.pop(0)
                    fallback_source = 'genre'

            if fallback:
                next_track = fallback
                next_track_source = fallback_source
                ordering_stats['no_api_fallbacks'] += 1

        # Add the selected track to ordered list and remove from appropriate pool
        if next_track:
            ordered.append(next_track)
            used_titles.add(normalize_song_title(next_track.get('title', '')))
            used_keys.add(next_track.get('rating_key'))

            # Remove from the appropriate pool
            if next_track_source == 'sonic':
                if next_track in remaining_sonic:
                    remaining_sonic.remove(next_track)
            elif next_track_source == 'genre':
                if next_track in remaining_genre:
                    remaining_genre.remove(next_track)
        else:
            # Safety: break if we can't find any more tracks
            break

    # Log ordering summary
    total_ordered = len(ordered) - 1  # Exclude the seed track
    if total_ordered > 0:
        logger.info(f"   Ordering complete: {ordering_stats['similar_matches']} similar matches, "
                   f"{ordering_stats['sonic_fallbacks'] + ordering_stats['genre_fallbacks']} random selections")

    return OrderingResult(
        ordered_tracks=ordered,
        stats=ordering_stats,
    )


