"""
Diversity enforcement module for playlist generation.

This module enforces artist diversity constraints to prevent playlist monotony.

Migrated from src/playlist_generator.py diversity methods (Phase 6).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import Counter
import logging
from src.playlist.utils import safe_get_artist_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiversityConfig:
    """Configuration for diversity enforcement."""
    max_per_artist: int = 6
    min_gap: int = 0  # minimum tracks between same artist
    window_size: int = 8  # for window-based frequency limits
    max_per_window: int = 2


@dataclass(frozen=True)
class DiversityResult:
    """
    Result of diversity enforcement.

    Attributes:
        diversified_tracks: Tracks with diversity constraints applied
        stats: Diagnostic statistics (artist counts, violations corrected, etc.)
    """
    diversified_tracks: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


def diversify_by_artist_cap(
    *,
    tracks: List[Dict[str, Any]],
    max_per_artist: int,
) -> List[Dict[str, Any]]:
    """
    Ensure no artist has more than max_per_artist tracks.

    Args:
        tracks: Playlist tracks
        max_per_artist: Maximum tracks per artist

    Returns:
        Tracks with artist cap enforced
    """
    artist_counts = Counter()
    diversified = []

    for track in tracks:
        artist_key = safe_get_artist_key(track)
        if artist_counts[artist_key] < max_per_artist:
            diversified.append(track)
            artist_counts[artist_key] += 1

    logger.info(f"Diversified {len(tracks)} -> {len(diversified)} tracks (max {max_per_artist} per artist)")
    return diversified


def remove_consecutive_artists(
    *,
    tracks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reorder tracks to ensure no artist plays twice in a row.

    Uses greedy reordering to minimize consecutive artist occurrences.

    Args:
        tracks: Ordered playlist tracks

    Returns:
        Reordered tracks without consecutive same artist
    """
    if len(tracks) <= 1:
        return tracks

    result = [tracks[0]]
    remaining = tracks[1:]
    last_artist_key = safe_get_artist_key(result[0])

    while remaining:
        # Find the first track with a different artist
        next_track = None
        for i, track in enumerate(remaining):
            if safe_get_artist_key(track) != last_artist_key:
                next_track = remaining.pop(i)
                break

        if next_track:
            result.append(next_track)
            last_artist_key = safe_get_artist_key(next_track)
        else:
            # All remaining tracks are by the same artist
            # Just add them (can't avoid consecutive in this case)
            result.extend(remaining)
            break

    return result


def enforce_artist_window(
    *,
    tracks: List[Dict[str, Any]],
    window_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Ensure artist doesn't appear too frequently in sliding windows.

    Ensures no artist appears within window_size tracks of their last appearance.
    When violations found, attempts to swap with later tracks to fix.

    Args:
        tracks: Ordered playlist tracks
        window_size: Size of sliding window

    Returns:
        Tracks with window constraints enforced
    """
    result = []
    artist_positions = {}  # artist -> list of positions in result
    skipped = 0

    for i, track in enumerate(tracks):
        artist_key = safe_get_artist_key(track)
        artist_label = track.get('artist', 'Unknown')

        # Check if artist appeared within window
        if artist_key in artist_positions:
            recent_positions = [p for p in artist_positions[artist_key] if len(result) - p <= window_size]

            if recent_positions:
                # Violation detected - try to find a swap candidate
                swapped = False

                # Look ahead for a valid track to swap with
                for swap_idx in range(i + 1, min(i + 10, len(tracks))):
                    swap_artist_key = safe_get_artist_key(tracks[swap_idx])

                    # Check if swap candidate would be valid
                    if swap_artist_key not in artist_positions:
                        swap_valid = True
                    else:
                        swap_recent = [p for p in artist_positions[swap_artist_key] if len(result) - p <= window_size]
                        swap_valid = len(swap_recent) == 0

                    if swap_valid:
                        # Swap tracks
                        tracks[i], tracks[swap_idx] = tracks[swap_idx], tracks[i]
                        track = tracks[i]
                        artist_key = safe_get_artist_key(track)
                        artist_label = track.get('artist', 'Unknown')
                        swapped = True
                        logger.debug(f"Swapped position {i} with {swap_idx} to fix window violation")
                        break

                if not swapped:
                    # No valid swap found - skip this track
                    logger.debug(f"Skipping {artist_label} - {track.get('title')} due to window violation (no valid swap)")
                    skipped += 1
                    continue

        # Add track to result
        result.append(track)

        # Track artist position
        if artist_key not in artist_positions:
            artist_positions[artist_key] = []
        artist_positions[artist_key].append(len(result) - 1)

    if skipped > 0:
        logger.info(f"Enforced artist window: {skipped} tracks skipped")

    return result


def limit_artist_frequency_in_window(
    *,
    tracks: List[Dict[str, Any]],
    window_size: int,
    max_per_window: int,
) -> List[Dict[str, Any]]:
    """
    Limit artist frequency within sliding windows.

    More granular than enforce_artist_window - allows explicit max per window.
    This version preserves the interleaving of sonic/genre tracks by considering
    source when there are multiple valid candidates.

    Args:
        tracks: Ordered playlist tracks
        window_size: Size of sliding window
        max_per_window: Maximum artist occurrences per window

    Returns:
        Tracks with frequency limits enforced
    """
    if len(tracks) <= window_size:
        return tracks

    # Start with empty result and process all tracks
    result = []
    remaining = list(tracks)

    logger.info(f"   Enforcing artist diversity: max {max_per_window} per {window_size}-track window")

    # Track source distribution to avoid clustering
    recent_sources = []  # Track last few sources to avoid long runs of same source
    diversity_fallback_count = 0  # Count times we couldn't satisfy constraints

    # Process remaining tracks
    while remaining:
        # Look at the last (window_size - 1) tracks to determine what can be added next
        window = result[-(window_size - 1):]
        artist_counts = Counter([safe_get_artist_key(t) for t in window])

        # Find a track that won't violate the constraint
        # Prefer alternating between sources when possible to avoid clustering
        next_track = None
        best_candidate_idx = None

        # Check what source we should prefer (alternate to avoid runs)
        prefer_source = None
        if len(recent_sources) >= 2:
            # If last 2 tracks were the same source, prefer different source
            if recent_sources[-1] == recent_sources[-2] and recent_sources[-1] in ['sonic', 'genre']:
                prefer_source = 'genre' if recent_sources[-1] == 'sonic' else 'sonic'

        # First pass: try to find a track with preferred source
        if prefer_source:
            for i, track in enumerate(remaining):
                artist = safe_get_artist_key(track)
                source = track.get('source')
                # Check if adding this track would violate the constraint
                if artist_counts.get(artist, 0) < max_per_window and source == prefer_source:
                    best_candidate_idx = i
                    next_track = track
                    break

        # Second pass: if no preferred source found, take first valid track
        if not next_track:
            for i, track in enumerate(remaining):
                artist = safe_get_artist_key(track)
                # Check if adding this track would violate the constraint
                if artist_counts.get(artist, 0) < max_per_window:
                    best_candidate_idx = i
                    next_track = track
                    break

        if next_track:
            remaining.pop(best_candidate_idx)
            result.append(next_track)

            # Track source for interleaving
            source = next_track.get('source', 'unknown')
            recent_sources.append(source)
            if len(recent_sources) > 3:
                recent_sources.pop(0)  # Keep only last 3
        else:
            # No valid track found - take the best option (least frequent in window)
            # This is a fallback that should rarely happen
            diversity_fallback_count += 1
            next_track = remaining.pop(0)
            result.append(next_track)

            source = next_track.get('source', 'unknown')
            recent_sources.append(source)
            if len(recent_sources) > 3:
                recent_sources.pop(0)

    # Log diversity summary
    if diversity_fallback_count > 0:
        logger.info(f"   Diversity check: {diversity_fallback_count} constraint fallbacks (tight windows)")

    return result


def apply_diversity_constraints(
    *,
    tracks: List[Dict[str, Any]],
    config: DiversityConfig,
) -> DiversityResult:
    """
    Apply all configured diversity constraints.

    Constraints are applied in this order:
    1. Artist cap
    2. Remove consecutive artists
    3. Window-based frequency limiting

    Args:
        tracks: Tracks to diversify
        config: Diversity configuration

    Returns:
        DiversityResult with diversified tracks and statistics
    """
    current_tracks = list(tracks)
    initial_count = len(current_tracks)
    stats = {}

    # 1. Apply artist cap
    if config.max_per_artist > 0:
        current_tracks = diversify_by_artist_cap(
            tracks=current_tracks,
            max_per_artist=config.max_per_artist,
        )
        stats['artist_cap'] = {
            'before': initial_count,
            'after': len(current_tracks),
            'removed': initial_count - len(current_tracks),
        }

    # 2. Remove consecutive artists
    if config.min_gap > 0:
        before = len(current_tracks)
        current_tracks = remove_consecutive_artists(tracks=current_tracks)
        stats['consecutive_artists_removed'] = before == len(current_tracks)

    # 3. Window-based frequency limiting
    if config.window_size > 0 and config.max_per_window > 0:
        before = len(current_tracks)
        current_tracks = limit_artist_frequency_in_window(
            tracks=current_tracks,
            window_size=config.window_size,
            max_per_window=config.max_per_window,
        )
        stats['window_frequency'] = {
            'before': before,
            'after': len(current_tracks),
            'removed': before - len(current_tracks),
        }

    stats['overall'] = {
        'initial': initial_count,
        'final': len(current_tracks),
        'total_removed': initial_count - len(current_tracks),
    }

    return DiversityResult(
        diversified_tracks=current_tracks,
        stats=stats,
    )
