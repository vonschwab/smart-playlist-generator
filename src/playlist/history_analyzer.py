"""
Listening history analysis module.

This module analyzes listening history (Last.fm or local) to identify top artists
and select diverse seed tracks for playlist generation.

Migrated from src/playlist_generator.py history analysis methods (Phase 4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Counter as CounterType, Set
from collections import Counter, defaultdict
import random
import logging

from src.playlist.utils import safe_get_artist_key
from src.string_utils import normalize_artist_key
from src.title_dedupe import normalize_title_for_dedupe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryAnalysisConfig:
    """Configuration for history analysis."""
    seed_count: int = 5
    artist_count: int = 10
    include_collaborations: bool = False


@dataclass(frozen=True)
class HistoryAnalysisResult:
    """
    Result of history analysis.

    Attributes:
        seed_tracks: Selected seed tracks with artist diversity
        top_artists: Dict mapping artist name to their tracks
        stats: Diagnostic statistics
    """
    seed_tracks: List[Dict[str, Any]]
    top_artists: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


def analyze_top_artists(
    *,
    history: List[Dict[str, Any]],
    artist_count: int,
    include_collaborations: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze listening history to get top played artists and their tracks.

    Args:
        history: Play history (list of tracks with play counts)
        artist_count: Number of top artists to return
        include_collaborations: Include collaboration tracks for artists

    Returns:
        Dict mapping artist name to list of their tracks
    """
    logger.info(f"Analyzing top {artist_count} artists from listening history...")

    # Group tracks by normalized artist key
    artist_tracks = defaultdict(list)
    display_counts = defaultdict(Counter)
    for track in history:
        artist = track.get('artist')
        if artist:
            artist_key = normalize_artist_key(artist)
            if not artist_key:
                continue
            artist_tracks[artist_key].append(track)
            display_counts[artist_key][artist] += 1

    # Sort artists by total play count
    artist_play_counts = {
        artist_key: sum(t.get('play_count', 1) for t in tracks)
        for artist_key, tracks in artist_tracks.items()
    }

    top_artists = sorted(artist_play_counts.items(), key=lambda x: x[1], reverse=True)[:artist_count]

    result = {}
    for artist_key, total_plays in top_artists:
        exact_match_tracks = artist_tracks[artist_key]
        artist = display_counts[artist_key].most_common(1)[0][0]

        # Check if we should include collaborations
        if include_collaborations and len(exact_match_tracks) < 4:
            logger.info(f"  {artist}: Only {len(exact_match_tracks)} exact match tracks, searching for collaborations...")

            # Search for collaboration tracks
            collaboration_tracks = []
            for other_key, other_tracks in artist_tracks.items():
                other_display = display_counts[other_key].most_common(1)[0][0]
                if is_collaboration_of(collaboration_name=other_display, base_artist=artist):
                    collaboration_tracks.extend(other_tracks)
                    logger.info(f"    Found collaboration: {other_display} ({len(other_tracks)} tracks)")

            # Combine exact matches and collaborations
            combined_tracks = exact_match_tracks + collaboration_tracks
            logger.info(f"  {artist}: {total_plays} total plays across {len(combined_tracks)} tracks ({len(exact_match_tracks)} solo, {len(collaboration_tracks)} collaborations)")
            result[artist] = combined_tracks
        else:
            logger.info(f"  {artist}: {total_plays} total plays across {len(exact_match_tracks)} tracks")
            result[artist] = exact_match_tracks

    return result


def select_diverse_seeds(
    *,
    play_counts: CounterType,
    track_metadata: Dict[str, Any],
    count: int,
) -> List[Dict[str, Any]]:
    """
    Select seed tracks ensuring artist diversity.

    Ensures no artist dominates the seed selection.

    Args:
        play_counts: Counter of track_id -> play count
        track_metadata: Mapping of track_id -> track dict
        count: Number of seeds to select

    Returns:
        Diverse seed tracks
    """
    seeds = []
    used_artists = set()

    # First pass: one track per artist (by play count)
    for key, play_count in play_counts.most_common():
        track = track_metadata[key]
        artist = track.get('artist', 'Unknown')
        artist_key = safe_get_artist_key(track)
        title = track.get('title', 'Unknown Track')

        if artist_key not in used_artists:
            seed_track = {
                **track,
                'play_count': play_count
            }
            seeds.append(seed_track)
            used_artists.add(artist_key)

            logger.info(f"  Selected seed: {artist} - {title} ({play_count} plays)")

            if len(seeds) >= count:
                break

    # Second pass: fill remaining slots if needed (allow duplicate artists)
    if len(seeds) < count:
        for key, play_count in play_counts.most_common():
            if len(seeds) >= count:
                break

            track = {**track_metadata[key], 'play_count': play_count}
            if track not in seeds:
                artist = track.get('artist', 'Unknown')
                title = track.get('title', 'Unknown Track')
                seeds.append(track)
                logger.info(f"  Selected seed: {artist} - {title} ({play_count} plays)")

    return seeds[:count]


def get_seed_tracks_for_artist(
    *,
    artist: str,
    tracks: List[Dict[str, Any]],
    seed_count: int = 4,
) -> List[Dict[str, Any]]:
    """
    Get seed tracks for a specific artist.

    Selects most played track plus random sampling from top 20.
    Ensures no duplicate titles (different releases of the same song) are selected.

    Args:
        artist: Artist name
        tracks: Artist's tracks with play counts
        seed_count: Number of seeds to return

    Returns:
        Seed tracks for this artist (title-deduplicated)
    """
    if not tracks:
        return []

    # Sort by play count
    sorted_tracks = sorted(tracks, key=lambda x: x.get('play_count', 0), reverse=True)

    seeds: List[Dict[str, Any]] = []
    used_titles: Set[str] = set()  # Normalized titles to avoid duplicates

    def _is_title_duplicate(title: str) -> bool:
        """Check if this title (normalized) is already in seeds."""
        norm = normalize_title_for_dedupe(title or "", mode="loose")
        return norm in used_titles

    def _add_seed(track: Dict[str, Any], reason: str) -> bool:
        """Add track to seeds if not a title duplicate. Returns True if added."""
        title = track.get('title', '')
        norm = normalize_title_for_dedupe(title or "", mode="loose")
        if norm in used_titles:
            logger.debug(f"    Skipping duplicate title: {artist} - {title}")
            return False
        used_titles.add(norm)
        seeds.append(track)
        logger.info(f"    {reason}: {artist} - {title} ({track.get('play_count', 0)} plays)")
        return True

    # Top 1 most played track (guaranteed)
    if len(sorted_tracks) > 0:
        _add_seed(sorted_tracks[0], "Top played")

    # Random tracks from top 20 (excluding the #1 already selected)
    if len(sorted_tracks) > 1 and len(seeds) < seed_count:
        # Get tracks 2-20 (or fewer if not enough tracks), excluding title duplicates
        top_20_pool = [
            t for t in sorted_tracks[1:min(20, len(sorted_tracks))]
            if not _is_title_duplicate(t.get('title', ''))
        ]

        # Shuffle and pick to fill remaining slots
        random.shuffle(top_20_pool)
        for track in top_20_pool:
            if len(seeds) >= seed_count:
                break
            _add_seed(track, "Random from top 20")
    elif len(sorted_tracks) <= 1:
        logger.warning(f"    Only {len(sorted_tracks)} track(s) available for {artist}")

    return seeds


def is_collaboration_of(
    *,
    collaboration_name: str,
    base_artist: str,
) -> bool:
    """
    Check if a collaboration name includes the base artist.

    Handles variations like "Artist feat. Other" or "Artist & Other".

    Args:
        collaboration_name: Full collaboration name
        base_artist: Base artist name to check for

    Returns:
        True if collaboration includes base artist
    """
    # Handle None values
    if not collaboration_name or not base_artist:
        return False

    # Normalize for comparison (case-insensitive)
    collab_lower = collaboration_name.lower()
    base_lower = base_artist.lower()

    # Not a collaboration if exact match
    if collab_lower == base_lower:
        return False

    # Check if base artist appears in the collaboration name
    if base_lower not in collab_lower:
        return False

    # Common collaboration patterns
    collaboration_patterns = [
        ' & ', ' and ', ' with ', ' featuring ', ' feat. ', ' feat ', ' ft. ', ' ft ',
        ', ', ' + ', ' / ',
        ' trio', ' quartet', ' quintet', ' sextet', ' ensemble', ' orchestra',
        ' band', ' group'
    ]

    # Check if any collaboration pattern appears
    for pattern in collaboration_patterns:
        if pattern in collab_lower:
            return True

    return False


def analyze_listening_history(
    *,
    history: List[Dict[str, Any]],
    config: HistoryAnalysisConfig,
) -> HistoryAnalysisResult:
    """
    Full history analysis pipeline.

    Analyzes history, selects top artists, and picks diverse seeds.

    Args:
        history: Play history
        config: Analysis configuration

    Returns:
        HistoryAnalysisResult with seeds, top artists, and stats
    """
    logger.info("Analyzing listening history")

    # Count plays by rating key
    play_counts = Counter()
    track_metadata = {}

    for track in history:
        key = track.get('rating_key')
        if key:
            play_counts[key] += 1
            track_metadata[key] = track

    # Get top played tracks with artist diversity
    seeds = select_diverse_seeds(
        play_counts=play_counts,
        track_metadata=track_metadata,
        count=config.seed_count,
    )

    for seed in seeds:
        logger.info(f"Seed track: {seed['artist']} - {seed['title']} ({seed['play_count']} plays)")

    # Optionally analyze top artists
    top_artists = {}
    if config.artist_count > 0:
        top_artists = analyze_top_artists(
            history=history,
            artist_count=config.artist_count,
            include_collaborations=config.include_collaborations,
        )

    stats = {
        'history_size': len(history),
        'unique_tracks': len(play_counts),
        'seed_count': len(seeds),
        'artist_count': len(top_artists),
    }

    return HistoryAnalysisResult(
        seed_tracks=seeds,
        top_artists=top_artists,
        stats=stats,
    )
