"""
Candidate track generation module for playlist building.

This module generates candidate tracks using:
- Sonic similarity analysis
- Genre-based discovery (dynamic mode)
- Title deduplication (fuzzy matching)
- Duration filtering

Migrated from src/playlist_generator.py candidate methods (Phase 8).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import logging

from src.string_utils import normalize_song_title, normalize_genre
from src.title_dedupe import TitleDedupeTracker
from src.playlist import scoring
from src.playlist.utils import sanitize_for_logging, safe_get_artist_key
from src.string_utils import normalize_artist_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for candidate generation."""
    limit_per_seed: int = 30  # Similar tracks per seed
    pool_target: int = 60  # Target pool size (playlist size + buffer)
    artist_cap: int = 6  # Max tracks per artist
    use_genre_discovery: bool = False  # Enable dynamic mode (sonic + genre)
    sonic_ratio: float = 0.6  # Sonic tracks ratio in dynamic mode
    genre_ratio: float = 0.4  # Genre tracks ratio in dynamic mode
    min_track_duration_seconds: int = 47  # Filter short tracks (hard minimum)
    max_track_duration_seconds: int = 720  # Filter long tracks (hard maximum)
    # Title deduplication settings
    title_dedupe_enabled: bool = True
    title_dedupe_threshold: float = 0.85
    title_dedupe_mode: str = "fuzzy"
    title_dedupe_short_title_min_len: int = 10


@dataclass(frozen=True)
class CandidateResult:
    """
    Result of candidate generation.

    Attributes:
        candidates: Generated candidate tracks
        stats: Diagnostic statistics (sonic_count, genre_count, filtered_counts, etc.)
    """
    candidates: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


def build_seed_title_set(
    *,
    seeds: List[Dict[str, Any]],
) -> Set[str]:
    """
    Build normalized set of seed titles for duplicate filtering.

    Args:
        seeds: Seed tracks

    Returns:
        Set of normalized seed titles
    """
    seed_titles = {normalize_song_title(seed.get('title', '')) for seed in seeds}
    seed_titles.discard('')  # Remove empty strings
    return seed_titles


def collect_sonic_candidates(
    *,
    seeds: List[Dict[str, Any]],
    seed_titles: Set[str],
    library_client,
    candidate_per_seed: int,
    min_duration_ms: int,
    max_duration_ms: int,
    title_dedupe_tracker: Optional[TitleDedupeTracker] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Collect sonic similarity candidates across seeds with filtering.

    Args:
        seeds: Seed tracks
        seed_titles: Set of normalized seed titles (for duplicate exclusion)
        library_client: Library client for sonic similarity lookups
        candidate_per_seed: Number of candidates to fetch per seed
        min_duration_ms: Minimum track duration (milliseconds)
        max_duration_ms: Maximum track duration (milliseconds)
        title_dedupe_tracker: Optional tracker for fuzzy title deduplication

    Returns:
        (candidates, filtered_counts) where filtered_counts has keys:
            'short', 'long', 'dupe_title', 'fuzzy_dupe'
    """
    all_candidates: Dict[str, Dict[str, Any]] = {}
    filtered_counts = {
        'short': 0,
        'long': 0,
        'dupe_title': 0,
        'fuzzy_dupe': 0,
    }

    # Pre-populate tracker with seed titles if enabled
    if title_dedupe_tracker and title_dedupe_tracker.enabled:
        for seed in seeds:
            seed_artist = seed.get('artist', '')
            seed_title = seed.get('title', '')
            if seed_artist and seed_title:
                title_dedupe_tracker.add(seed_artist, seed_title)

    for seed in seeds:
        seed_id = seed.get('rating_key')
        if not seed_id:
            continue

        # Get sonic similarity candidates
        similar = library_client.get_similar_tracks_sonic_only(
            seed_id, limit=candidate_per_seed, min_similarity=0.1
        )

        # Filter long tracks
        before_long = len(similar)
        similar = [t for t in similar if (t.get('duration') or 0) <= max_duration_ms]
        filtered_counts['long'] += max(0, before_long - len(similar))

        weight = seed.get('play_count', 1)

        for track in similar:
            track_key = track.get('rating_key')
            track_artist = track.get('artist', '')
            track_title = track.get('title', '')

            # Skip tracks with same title as seed tracks (exact duplicates)
            track_title_normalized = normalize_song_title(track_title)
            if track_title_normalized in seed_titles:
                filtered_counts['dupe_title'] += 1
                continue

            # Fuzzy title deduplication (check against already-accepted candidates)
            if title_dedupe_tracker and title_dedupe_tracker.enabled:
                is_dup, matched = title_dedupe_tracker.is_duplicate(
                    track_artist, track_title, debug=logger.isEnabledFor(logging.DEBUG)
                )
                if is_dup:
                    filtered_counts['fuzzy_dupe'] += 1
                    logger.debug(
                        f"Title dedupe: skipping '{track_artist} - {track_title}' "
                        f"(fuzzy match to '{matched}')"
                    )
                    continue

            # Filter short tracks (interludes, skits, etc.)
            track_duration = track.get('duration') or 0
            if min_duration_ms > 0 and track_duration < min_duration_ms:
                filtered_counts['short'] += 1
                continue

            # If track already exists, keep the one with better similarity score
            existing = all_candidates.get(track_key)
            if existing and existing.get('similarity_score', 0) >= track.get('similarity_score', 0):
                continue

            # Add to tracker for future duplicate detection
            if title_dedupe_tracker and title_dedupe_tracker.enabled:
                title_dedupe_tracker.add(track_artist, track_title)

            # Enrich track with seed context
            track['seed_artist'] = seed.get('artist')
            track['seed_title'] = seed.get('title')
            track['seed_rating_key'] = seed_id
            track['weight'] = weight
            track['source'] = 'sonic'

            all_candidates[track_key] = track

    return list(all_candidates.values()), filtered_counts


# NOTE: The full implementations of these functions are substantial (~400+ lines combined)
# For Phase 8 Part 2, we're adding complete implementations extracted from playlist_generator.py

def generate_candidates_dynamic(
    *,
    seeds: List[Dict[str, Any]],
    library_client,
    metadata_client,
    config: CandidateConfig,
) -> CandidateResult:
    """
    Generate tracks using dynamic mode: mix sonic similarity with genre-based discovery
    60% from sonic analysis, 40% from genre matching

    Args:
        seeds: List of seed tracks (must have 'genres' field)
        limit_per_seed: Override for similar_per_seed (useful for extending playlists)

    Returns:
        Mixed list of sonically similar and genre-matched tracks
    """
    similar_per_seed = config.limit_per_seed
    min_track_duration_ms = config.min_track_duration_seconds * 1000

    # Calculate how many tracks from each source (from config)
    sonic_per_seed = int(similar_per_seed * config.sonic_ratio)
    genre_per_seed = int(similar_per_seed * config.genre_ratio)

    logger.info(f"  Target: {sonic_per_seed} sonic + {genre_per_seed} genre-based tracks per seed")

    # Part 1: Get sonic similarity tracks (existing logic)
    sonic_tracks = []
    seen_keys = set()
    filtered_short_count = 0
    filtered_long_count = 0
    filtered_fuzzy_dupe_count = 0

    # Build set of normalized seed titles to filter out
    seed_titles = build_seed_title_set(seeds=seeds)

    # Build set of seed artists to exclude from similarity results
    seed_artists = {safe_get_artist_key(seed) for seed in seeds}
    seed_artists.discard("")

    # Create title dedupe tracker if enabled
    title_dedupe_tracker = TitleDedupeTracker(
        threshold=config.title_dedupe_threshold,
        mode=config.title_dedupe_mode,
        short_title_min_len=config.title_dedupe_short_title_min_len,
        enabled=config.title_dedupe_enabled,
    )

    # Pre-populate tracker with seed titles
    if title_dedupe_tracker.enabled:
        for seed in seeds:
            seed_artist = seed.get('artist', '')
            seed_title = seed.get('title', '')
            if seed_artist and seed_title:
                title_dedupe_tracker.add(seed_artist, seed_title)

    for seed in seeds:
        key = seed.get('rating_key')
        if not key:
            continue

        similar = library_client.get_similar_tracks(key, limit=sonic_per_seed)
        before_long = len(similar)
        similar = [t for t in similar if (t.get('duration') or 0) <= max_duration_ms]
        filtered_long_count += max(0, before_long - len(similar))
        weight = seed.get('play_count', 1)

        for track in similar:
            track_key = track.get('rating_key')

            if track_key in seen_keys:
                continue

            # Skip tracks by seed artists
            track_artist_key = safe_get_artist_key(track)
            if track_artist_key in seed_artists:
                continue

            # Skip tracks with same title as seeds
            track_title = track.get('title', '')
            track_title_normalized = normalize_song_title(track_title)
            if track_title_normalized in seed_titles:
                continue

            # Fuzzy title deduplication
            if title_dedupe_tracker.enabled:
                is_dup, matched = title_dedupe_tracker.is_duplicate(
                    track.get('artist', ''), track_title,
                    debug=logger.isEnabledFor(logging.DEBUG)
                )
                if is_dup:
                    filtered_fuzzy_dupe_count += 1
                    logger.debug(
                        f"Title dedupe (dynamic/sonic): skipping '{track.get('artist')} - {track_title}' "
                        f"(fuzzy match to '{matched}')"
                    )
                    continue

            # Filter short tracks
            track_duration = track.get('duration') or 0
            if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                filtered_short_count += 1
                continue

            # Add to tracker for future duplicate detection
            if title_dedupe_tracker.enabled:
                title_dedupe_tracker.add(track.get('artist', ''), track_title)

            seen_keys.add(track_key)
            track['seed_artist'] = seed.get('artist')
            track['seed_title'] = seed.get('title')
            track['weight'] = weight
            track['source'] = 'sonic'

            sonic_tracks.append(track)

    logger.info(f"  Sonic similarity: {len(sonic_tracks)} tracks (filtered {filtered_fuzzy_dupe_count} fuzzy dupes)")

    # Part 2: Get genre-based tracks using metadata database
    genre_tracks = []

    # Extract all genres from seeds
    all_genres = set()
    for seed in seeds:
        seed_genres = seed.get('genres', [])
        if seed_genres:
            # Take top 3 genres from each seed
            all_genres.update(seed_genres[:3])

    if not all_genres:
        logger.info("  No genres available for genre-based discovery, using sonic only")
        return sonic_tracks

    seed_genre_list = sorted(list(all_genres))
    logger.info(f"  Seed genres: {', '.join(seed_genre_list)}")

    # Use metadata database if available, otherwise fall back to local metadata
    if metadata_client:
        cursor = metadata_client.conn.cursor()
        normalized_seed_genres = [normalize_genre(g) for g in all_genres]
        placeholders = ",".join("?" * len(normalized_seed_genres))
        cursor.execute(f"""
            SELECT artist, GROUP_CONCAT(genre) AS genres, COUNT(*) AS match_count
            FROM artist_genres
            WHERE genre IN ({placeholders})
            GROUP BY artist
            ORDER BY match_count DESC
        """, tuple(normalized_seed_genres))

        artist_genre_scores = []
        for row in cursor.fetchall():
            artist_name = row['artist']
            # Skip seed artists
            artist_key = normalize_artist_key(artist_name)
            if any(safe_get_artist_key(seed) == artist_key for seed in seeds):
                continue

            matching_genres = row['genres'].split(',') if row['genres'] else []
            artist_genre_scores.append({
                'artist': artist_name,
                'matching_genres': matching_genres,
                'match_count': row['match_count'],
                'all_tags': matching_genres
            })

        logger.info(f"  Found {len(artist_genre_scores)} artists with matching genres")

        # Log top matching artists
        logger.info(f"  Top genre matches:")
        for artist_score in artist_genre_scores[:10]:
            genres_str = ', '.join(artist_score['matching_genres'])
            logger.info(f"    - {sanitize_for_logging(artist_score['artist'])}: {artist_score['match_count']} matches ({genres_str})")

        # Select tracks from top-matching artists (1 track per artist for diversity)
        target_genre_tracks = genre_per_seed * len(seeds)
        selected_artists = set()

        for artist_score in artist_genre_scores:
            if len(genre_tracks) >= target_genre_tracks:
                break

            artist_name = artist_score['artist']
            artist_key = normalize_artist_key(artist_name)

            # Get tracks from this artist (from metadata DB to get rating keys)
            matched_tracks = metadata_client.get_tracks_by_artist(artist_name, limit=5)

            # Find first track that's not already used
            for matched in matched_tracks:
                track_key = matched.get('track_id')

                # Skip if already in sonic tracks or genre tracks
                if track_key in seen_keys:
                    continue

                # Skip seeds
                matched_title = matched.get('title', '')
                track_title_normalized = normalize_song_title(matched_title)
                if track_title_normalized in seed_titles:
                    continue

                # Fuzzy title deduplication
                if title_dedupe_tracker.enabled:
                    is_dup, dup_matched = title_dedupe_tracker.is_duplicate(
                        artist_name, matched_title,
                        debug=logger.isEnabledFor(logging.DEBUG)
                    )
                    if is_dup:
                        filtered_fuzzy_dupe_count += 1
                        logger.debug(
                            f"Title dedupe (dynamic/genre-meta): skipping '{artist_name} - {matched_title}' "
                            f"(fuzzy match to '{dup_matched}')"
                        )
                        continue

                # Get full track data from library (includes duration)
                full_track_data = library_client.get_track_by_key(track_key)
                if not full_track_data:
                    continue

                # Hard filter: Check duration constraints
                track_duration_ms = full_track_data.get('duration', 0)
                min_duration_ms = int(config.min_track_duration_seconds * 1000)
                max_duration_ms = int(config.max_track_duration_seconds * 1000)

                # Skip if duration outside acceptable range
                if track_duration_ms > 0:
                    if track_duration_ms < min_duration_ms:
                        logger.debug(f"Skipping {full_track_data.get('title')} - too short ({track_duration_ms}ms < {min_duration_ms}ms)")
                        continue
                    if track_duration_ms > max_duration_ms:
                        logger.debug(f"Skipping {full_track_data.get('title')} - too long ({track_duration_ms}ms > {max_duration_ms}ms)")
                        continue

                # Found a good track - add it with full library data
                track = {
                    'rating_key': track_key,
                    'title': full_track_data.get('title'),
                    'artist': full_track_data.get('artist'),
                    'album': full_track_data.get('album'),
                    'duration': full_track_data.get('duration', 0),
                    'uri': f"/library/metadata/{track_key}",
                    'source': 'genre',
                    'weight': matched.get('weight', 1.0),
                    'matched_genres': artist_score['matching_genres']  # Store for reporting
                }

                # Add to tracker for future duplicate detection
                if title_dedupe_tracker.enabled:
                    title_dedupe_tracker.add(full_track_data.get('artist', ''), full_track_data.get('title', ''))

                seen_keys.add(track_key)
                genre_tracks.append(track)
                selected_artists.add(artist_key)

                # Only take 1 track per artist
                break

        logger.info(f"  Genre-based discovery: {len(genre_tracks)} tracks from {len(selected_artists)} artists")

    else:
        # Fallback to local library metadata (original implementation)
        logger.info("  Metadata database not available, falling back to local library genre metadata")

        # Get all tracks from library (cached)
        all_library_tracks = library_client.get_all_tracks()

        # Filter tracks by genre and diversify by artist
        artist_track_counts = {}

        for track in all_library_tracks:
            track_key = track.get('rating_key')

            # Skip if already in sonic tracks
            if track_key in seen_keys:
                continue

            # Skip seeds
            track_title = track.get('title', '')
            track_title_normalized = normalize_song_title(track_title)
            if track_title_normalized in seed_titles:
                continue

            # Fuzzy title deduplication
            track_artist = track.get('artist', '')
            if title_dedupe_tracker.enabled:
                is_dup, dup_matched = title_dedupe_tracker.is_duplicate(
                    track_artist, track_title,
                    debug=logger.isEnabledFor(logging.DEBUG)
                )
                if is_dup:
                    filtered_fuzzy_dupe_count += 1
                    logger.debug(
                        f"Title dedupe (dynamic/genre-fallback): skipping '{track_artist} - {track_title}' "
                        f"(fuzzy match to '{dup_matched}')"
                    )
                    continue

            # Filter tracks by duration (hard limits)
            track_duration = track.get('duration') or 0
            if track_duration > 0:
                if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                    continue
                if max_track_duration_ms > 0 and track_duration > max_track_duration_ms:
                    filtered_long_count += 1
                    continue

            # Check if track's genre matches any seed genres
            track_genre = track.get('genre', '')
            if not track_genre:
                continue

            # Match if any seed genre is in the track's genre (with normalization)
            track_genre_normalized = normalize_genre(track_genre)
            genre_match = False
            for seed_genre in all_genres:
                seed_genre_normalized = normalize_genre(seed_genre)
                if seed_genre_normalized in track_genre_normalized:
                    genre_match = True
                    break

            if not genre_match:
                continue

            # Limit tracks per artist to increase diversity
            artist_key = safe_get_artist_key(track)
            artist_track_counts[artist_key] = artist_track_counts.get(artist_key, 0) + 1

            # Max 2 tracks per artist in genre pool
            if artist_track_counts[artist_key] > 2:
                continue

            # Add to tracker for future duplicate detection
            if title_dedupe_tracker.enabled:
                title_dedupe_tracker.add(track_artist, track_title)

            seen_keys.add(track_key)
            track['source'] = 'genre'
            track['weight'] = 1
            genre_tracks.append(track)

            # Stop once we have enough genre tracks
            target_genre_tracks = genre_per_seed * len(seeds)
            if len(genre_tracks) >= target_genre_tracks:
                break

        logger.info(f"  Genre-based discovery: {len(genre_tracks)} tracks from {len(artist_track_counts)} artists")

    # Combine sonic and genre tracks
    all_tracks = sonic_tracks + genre_tracks

    if filtered_short_count > 0:
        logger.info(f"  Filtered out {filtered_short_count} short tracks (< {config.min_track_duration_seconds}s)")
    if filtered_long_count > 0:
        logger.info(f"  Filtered out {filtered_long_count} long tracks (> {config.max_track_duration_seconds}s)")
    if filtered_fuzzy_dupe_count > 0:
        logger.info(f"  Filtered out {filtered_fuzzy_dupe_count} fuzzy duplicate titles")

    logger.info(f"  Total: {len(all_tracks)} tracks ({len(sonic_tracks)} sonic + {len(genre_tracks)} genre)")



def generate_candidates(
    *,
    seeds: List[Dict[str, Any]],
    library_client,
    similarity_calculator,
    metadata_client,
    config: CandidateConfig,
) -> CandidateResult:
    """
    Generate candidate tracks using sonic-first pipeline.
    
    This is a complete implementation extracted from PlaylistGenerator.generate_similar_tracks()
    Adapted to use explicit dependency injection and return CandidateResult.

    Pipeline:
    1. Sonic-only discovery per seed (no genre filtering)
    2. Merge/dedup and apply per-artist cap
    3. Filter by genre threshold via hybrid similarity  
    4. Rank by hybrid score and keep a buffered pool (target + buffer)

    Args:
        seeds: Seed tracks
        library_client: Library client for similarity lookups
        similarity_calculator: Similarity calculator for hybrid scoring
        metadata_client: Metadata client (for dynamic mode delegation)
        config: Candidate generation configuration

    Returns:
        CandidateResult with sonic-first candidates and stats
    """
    # Dynamic mode: delegate to generate_candidates_dynamic
    if config.use_genre_discovery:
        logger.info("Finding similar tracks (60% sonic, 40% genre-based)...")
        return generate_candidates_dynamic(
            seeds=seeds,
            library_client=library_client,
            metadata_client=metadata_client,
            config=config,
        )

    # Sonic-first mode
    logger.info("Finding similar tracks (sonic-first pipeline)...")

    min_duration_ms = config.min_track_duration_seconds * 1000
    max_duration_ms = config.max_track_duration_seconds * 1000

    seed_titles = build_seed_title_set(seeds=seeds)

    # Create title dedupe tracker
    title_dedupe_tracker = TitleDedupeTracker(
        threshold=config.title_dedupe_threshold,
        mode=config.title_dedupe_mode,
        short_title_min_len=config.title_dedupe_short_title_min_len,
        enabled=config.title_dedupe_enabled,
    )

    # Collect sonic candidates
    candidates, filtered_counts = collect_sonic_candidates(
        seeds=seeds,
        seed_titles=seed_titles,
        library_client=library_client,
        candidate_per_seed=config.limit_per_seed,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        title_dedupe_tracker=title_dedupe_tracker,
    )

    logger.info(
        "Sonic-only pool: %s candidates (filtered %s short, %s long, %s exact dupe, %s fuzzy dupe)",
        len(candidates),
        filtered_counts['short'],
        filtered_counts['long'],
        filtered_counts['dupe_title'],
        filtered_counts.get('fuzzy_dupe', 0),
    )

    # Apply artist cap
    capped_candidates = scoring.cap_candidates_by_artist(
        candidates=candidates,
        artist_cap=config.artist_cap,
        limit=config.pool_target * 2,
        score_key="similarity_score",
    )
    logger.info(f"After artist cap: {len(capped_candidates)} candidates")

    # Score genre and hybrid similarity
    genre_pass, genre_fail = scoring.score_genre_and_hybrid(
        candidates=capped_candidates,
        similarity_calculator=similarity_calculator,
        genre_method=similarity_calculator.genre_method,
    )
    logger.info(f"After genre filter/hybrid scoring: {len(genre_pass)} candidates")

    if genre_fail:
        for t, gsim in genre_fail[:5]:
            logger.debug(f"Genre-filtered: {t.get('artist')} - {t.get('title')} (genre_sim={gsim:.3f})")

    # Finalize pool (sort by hybrid score, enforce artist cap, trim to target)
    final_pool = scoring.finalize_pool(
        candidates=genre_pass,
        artist_cap=config.artist_cap,
        pool_target=config.pool_target,
        score_key="hybrid_score",
    )

    logger.info(f"Selected {len(final_pool)} candidates for ordering (target pool: {config.pool_target})")

    return CandidateResult(
        candidates=final_pool,
        stats={
            'sonic_count': len(candidates),
            'genre_count': 0,
            'filtered': filtered_counts,
            'genre_pass_count': len(genre_pass),
            'genre_fail_count': len(genre_fail),
        }
    )
