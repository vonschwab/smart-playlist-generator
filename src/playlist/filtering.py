"""
Track filtering module for playlist generation.

This module provides filtering functions to remove tracks based on various criteria:
- Duration (too long)
- Recently played (from local history or Last.fm scrobbles)
- Seed preservation (ensure seed tracks remain in playlist)

Migrated from src/playlist_generator.py filtering methods (Phase 3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from ..string_utils import normalize_artist_key, normalize_match_string
from .utils import safe_get_artist_key

logger = logging.getLogger(__name__)

RECENCY_STAGE_CANDIDATE_POOL = "candidate_pool"


def _assert_recency_stage(stage: str) -> None:
    if stage != RECENCY_STAGE_CANDIDATE_POOL:
        raise ValueError(f"Recency filter must not run after ordering (stage={stage})")


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for track filtering operations."""
    max_duration_seconds: Optional[int] = 720  # 12 minutes
    recency_lookback_days: int = 14
    preserve_seed_tracks: bool = True
    recently_played_filter_enabled: bool = True
    recently_played_min_playcount: int = 0


@dataclass(frozen=True)
class FilterResult:
    """
    Result of filtering operation with diagnostics.

    Attributes:
        filtered_tracks: Tracks that passed all filters
        stats: Diagnostic statistics (counts, matches, etc.)
    """
    filtered_tracks: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


def filter_by_duration(
    *,
    tracks: List[Dict[str, Any]],
    max_duration_seconds: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Remove tracks above maximum duration.

    Args:
        tracks: List of track dictionaries
        max_duration_seconds: Maximum duration in seconds (None to skip filtering)

    Returns:
        Filtered list of tracks
    """
    if not max_duration_seconds:
        return tracks

    max_duration_ms = max_duration_seconds * 1000

    filtered = [t for t in tracks if (t.get('duration') or 0) <= max_duration_ms]

    logger.debug(f"Duration filter: {len(tracks)} -> {len(filtered)} tracks (max={max_duration_seconds}s)")

    return filtered


def filter_by_recently_played(
    *,
    tracks: List[Dict[str, Any]],
    play_history: List[Dict[str, Any]],
    lookback_days: int,
    min_playcount: int = 0,
    exempt_tracks: Optional[List[Dict[str, Any]]] = None,
    stage: str,
) -> FilterResult:
    """
    Filter out recently played tracks based on local history.

    Args:
        tracks: Candidate tracks to filter
        play_history: Local play history
        lookback_days: How many days back to consider (0 = all history)
        min_playcount: Minimum play count threshold for filtering
        exempt_tracks: Tracks to exempt from filtering (e.g., seeds)

    Returns:
        FilterResult with filtered tracks and statistics
    """
    _assert_recency_stage(stage)

    # Build set of tracks to filter based on configuration
    played_keys = set()
    play_counts = defaultdict(int)

    # Count plays per track
    for track in play_history:
        key = track.get('rating_key')
        if key:
            play_counts[key] += 1

    # Apply filtering rules
    if lookback_days > 0:
        # Filter only tracks played within lookback window
        cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())

        for track in play_history:
            key = track.get('rating_key')
            timestamp = (
                track.get('timestamp')
                or track.get('last_played')
                or track.get('lastfm_timestamp')
                or 0
            )

            if key and timestamp >= cutoff_timestamp:
                # Only filter if playcount threshold met
                if play_counts[key] >= min_playcount:
                    played_keys.add(key)
    else:
        # Filter all history (default behavior)
        for track in play_history:
            key = track.get('rating_key')
            if key and play_counts[key] >= min_playcount:
                played_keys.add(key)

    # Build set of exempt track keys
    exempt_keys: Set[str] = set()
    if exempt_tracks:
        exempt_keys = {t.get("rating_key") for t in exempt_tracks if t.get("rating_key")}

    # Filter out played tracks (except exempt tracks)
    filtered = [t for t in tracks if t.get('rating_key') not in played_keys or t.get('rating_key') in exempt_keys]

    logger.info(
        "stage=%s | Local recency exclusions: before=%d after=%d excluded=%d lookback_days=%d min_playcount=%d history_size=%d exempt=%d",
        stage,
        len(tracks),
        len(filtered),
        len(tracks) - len(filtered),
        lookback_days,
        min_playcount,
        len(play_history),
        len(exempt_keys),
    )
    logger.debug(
        "Filtering details: history_size=%d played_keys=%d candidates_before=%d candidates_after=%d still_present=%d",
        len(play_history),
        len(played_keys),
        len(tracks),
        len(filtered),
        sum(1 for t in filtered if t.get('rating_key') in played_keys and t.get('rating_key') not in exempt_keys),
    )

    if played_keys and not exempt_keys and len(filtered) == len(tracks):
        logger.debug("Filter check: exclusion set non-empty but no candidates removed (possible key mismatch).")

    stats = {
        'history_size': len(play_history),
        'played_keys': len(played_keys),
        'before': len(tracks),
        'after': len(filtered),
        'removed': len(tracks) - len(filtered),
    }

    return FilterResult(filtered_tracks=filtered, stats=stats)


def filter_by_scrobbles(
    *,
    tracks: List[Dict[str, Any]],
    scrobbles: List[Dict[str, Any]],
    lookback_days: int,
    exempt_tracks: Optional[List[Dict[str, Any]]] = None,
    sample_limit: int = 5,
    stage: str,
) -> FilterResult:
    """
    Filter candidates using Last.FM scrobbles.

    Uses artist::title matching to identify recently played tracks.
    Ignores mbid for consistency between Last.fm and library.

    Args:
        tracks: Candidate tracks to filter
        scrobbles: Last.fm scrobbles
        lookback_days: How many days back to consider
        exempt_tracks: Tracks to exempt from filtering (e.g., seeds)
        sample_limit: Number of filtered examples to log

    Returns:
        FilterResult with filtered tracks and statistics
    """
    _assert_recency_stage(stage)

    if not scrobbles or lookback_days <= 0:
        return FilterResult(filtered_tracks=tracks, stats={"skipped": True})

    cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())

    def _key_for_track(track: Dict[str, Any]) -> Optional[str]:
        """Generate artist::title key for matching (ignore mbid)."""
        artist = track.get("artist")
        title = track.get("title")
        if not artist or not title:
            return None
        artist_key = normalize_artist_key(artist)
        if not artist_key:
            return None
        return f"{artist_key}::{normalize_match_string(title)}"

    scrobble_keys = set()
    exempt_keys = set()

    if exempt_tracks:
        for t in exempt_tracks:
            k = _key_for_track(t)
            if k:
                exempt_keys.add(k)

    for s in scrobbles:
        ts = s.get("timestamp", 0)
        if ts == 0 or ts < cutoff_timestamp:
            continue
        k = _key_for_track(s)
        if k:
            scrobble_keys.add(k)

    if not scrobble_keys:
        logger.info(
            "stage=%s | Last.fm recency exclusions: before=%d after=%d excluded=%d lookback_days=%d scrobbles=%d keys=%d exempt=%d (skipped=no_usable_keys)",
            stage,
            len(tracks),
            len(tracks),
            0,
            lookback_days,
            len(scrobbles),
            0,
            len(exempt_keys),
        )
        return FilterResult(filtered_tracks=tracks, stats={"skipped": True, "reason": "no_usable_keys"})

    filtered = []
    filtered_out = []
    for t in tracks:
        k = _key_for_track(t)
        if k and k in scrobble_keys and k not in exempt_keys:
            filtered_out.append(t)
            continue
        filtered.append(t)

    logger.info(
        "stage=%s | Last.fm recency exclusions: before=%d after=%d excluded=%d lookback_days=%d scrobbles=%d keys=%d exempt=%d",
        stage,
        len(tracks),
        len(filtered),
        len(filtered_out),
        lookback_days,
        len(scrobbles),
        len(scrobble_keys),
        len(exempt_keys),
    )

    if filtered_out and logger.isEnabledFor(logging.DEBUG):
        from .utils import sanitize_for_logging
        sample = [f"{sanitize_for_logging(t.get('artist',''))} - {sanitize_for_logging(t.get('title',''))}" for t in filtered_out[:sample_limit]]
        logger.debug("Filtered examples: %s", sample)

    if not filtered_out and logger.isEnabledFor(logging.DEBUG):
        reason = "no key overlap"
        if not tracks:
            reason = "no candidates"
        elif all(_key_for_track(t) is None for t in tracks):
            reason = "candidates missing artist/title"
        logger.debug("Scrobble recency filter made no changes (%s)", reason)

    stats = {
        'scrobbles': len(scrobbles),
        'scrobble_keys': len(scrobble_keys),
        'before': len(tracks),
        'after': len(filtered),
        'removed': len(filtered_out),
        'lookback_days': lookback_days,
    }

    return FilterResult(filtered_tracks=filtered, stats=stats)


def ensure_seed_tracks_present(
    *,
    seed_tracks: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    target_length: int,
) -> List[Dict[str, Any]]:
    """
    Ensure all seed tracks are kept in the final playlist.

    Missing seeds are inserted at positions that avoid immediate same-artist
    adjacency when possible. Length is capped at target_length, preserving all
    seeds when there is room.

    Args:
        seed_tracks: Original seed tracks
        candidates: Current candidate list (may be missing seeds)
        target_length: Target playlist length

    Returns:
        Candidate list with seeds restored
    """
    if not seed_tracks:
        return candidates[:target_length] if target_length > 0 else list(candidates)

    def _key(track: Dict[str, Any]) -> Tuple[str, str, str]:
        """Generate unique key for track (rating_key or artist::title)."""
        rid = str(track.get("rating_key") or "").strip()
        if rid:
            return ("id", rid, "")
        artist = normalize_artist_key(track.get("artist", ""))
        title = normalize_match_string(track.get("title", ""))
        return ("at", artist, title)

    result: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()

    # Deduplicate candidates while preserving order
    for t in candidates:
        k = _key(t)
        if k in seen:
            continue
        seen.add(k)
        result.append(t)

    seed_keys = {_key(s) for s in seed_tracks}

    # Insert missing seeds at positions avoiding same-artist adjacency
    for seed in seed_tracks:
        sk = _key(seed)
        if sk in seen:
            continue  # already present

        seed_artist = safe_get_artist_key(seed)
        insert_at = None

        # Find best insertion point (avoiding same-artist adjacency)
        for i in range(len(result) + 1):
            prev_artist = safe_get_artist_key(result[i - 1]) if i > 0 else None
            next_artist = safe_get_artist_key(result[i]) if i < len(result) else None
            if (prev_artist is None or prev_artist != seed_artist) and (next_artist is None or next_artist != seed_artist):
                insert_at = i
                break

        if insert_at is None:
            insert_at = len(result)

        result.insert(insert_at, seed)
        seen.add(sk)

    # Cap length, preferring to keep seeds
    desired_len = max(target_length, len(seed_tracks)) if target_length > 0 else len(result)
    if len(result) > desired_len:
        # Drop from the end, preferring to drop non-seed tracks
        i = len(result) - 1
        while len(result) > desired_len and i >= 0:
            if _key(result[i]) not in seed_keys:
                result.pop(i)
            i -= 1
        result = result[:desired_len]

    return result


def apply_filters(
    *,
    tracks: List[Dict[str, Any]],
    config: FilterConfig,
    play_history: Optional[List[Dict[str, Any]]] = None,
    scrobbles: Optional[List[Dict[str, Any]]] = None,
    seed_tracks: Optional[List[Dict[str, Any]]] = None,
) -> FilterResult:
    """
    Apply all configured filters in sequence.

    Filters are applied in this order:
    1. Duration filtering
    2. Recently played (local history)
    3. Recently played (Last.fm scrobbles)
    4. Seed preservation (if target length specified)

    Args:
        tracks: Tracks to filter
        config: Filter configuration
        play_history: Optional local play history
        scrobbles: Optional Last.fm scrobbles
        seed_tracks: Optional seed tracks to preserve

    Returns:
        FilterResult with filtered tracks and cumulative statistics
    """
    cumulative_stats = {}
    current_tracks = list(tracks)
    initial_count = len(current_tracks)

    # 1. Duration filtering
    current_tracks = filter_by_duration(
        tracks=current_tracks,
        max_duration_seconds=config.max_duration_seconds,
    )
    cumulative_stats['duration_filter'] = {
        'before': initial_count,
        'after': len(current_tracks),
        'removed': initial_count - len(current_tracks),
    }

    # 2. Local history filtering
    if play_history and config.recently_played_filter_enabled:
        before = len(current_tracks)
        result = filter_by_recently_played(
            tracks=current_tracks,
            play_history=play_history,
            lookback_days=config.recency_lookback_days,
            min_playcount=config.recently_played_min_playcount,
            exempt_tracks=seed_tracks if config.preserve_seed_tracks else None,
            stage=RECENCY_STAGE_CANDIDATE_POOL,
        )
        current_tracks = result.filtered_tracks
        cumulative_stats['local_history_filter'] = result.stats

    # 3. Last.fm scrobbles filtering
    if scrobbles:
        before = len(current_tracks)
        result = filter_by_scrobbles(
            tracks=current_tracks,
            scrobbles=scrobbles,
            lookback_days=config.recency_lookback_days,
            exempt_tracks=seed_tracks if config.preserve_seed_tracks else None,
            stage=RECENCY_STAGE_CANDIDATE_POOL,
        )
        current_tracks = result.filtered_tracks
        cumulative_stats['scrobbles_filter'] = result.stats

    # Final stats
    cumulative_stats['overall'] = {
        'initial': initial_count,
        'final': len(current_tracks),
        'total_removed': initial_count - len(current_tracks),
    }

    logger.info(f"Overall filtering: {initial_count} -> {len(current_tracks)} tracks")

    return FilterResult(filtered_tracks=current_tracks, stats=cumulative_stats)
