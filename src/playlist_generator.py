"""
Playlist Generator - Core logic for creating AI-powered playlists
"""
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import random
import logging
import time
import numpy as np
from .artist_cache import ArtistSimilarityCache
from .artist_utils import extract_primary_artist
from .similarity_calculator import SimilarityCalculator
from .string_utils import normalize_genre, normalize_song_title

logger = logging.getLogger(__name__)


def sanitize_for_logging(text: str) -> str:
    """Sanitize text for Windows console logging by replacing unencodable characters"""
    if not text:
        return text
    try:
        # Try to encode with cp1252 (Windows console encoding)
        text.encode('cp1252')
        return text
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Replace unencodable characters with '?'
        return text.encode('cp1252', errors='replace').decode('cp1252')


def safe_get_artist(track: Dict[str, Any], lowercase: bool = True) -> str:
    """
    Safely get artist name from a track dictionary with None-safe fallback

    Args:
        track: Track dictionary
        lowercase: Whether to convert to lowercase

    Returns:
        Artist name (empty string if None or missing)
    """
    artist = track.get('artist') or ''
    return artist.lower() if lowercase and artist else artist


def _convert_seconds_to_ms(seconds: Optional[int]) -> int:
    """Helper to convert seconds to milliseconds with None safety."""
    if seconds is None:
        return 0
    try:
        return int(seconds) * 1000
    except (TypeError, ValueError):
        return 0



class PlaylistGenerator:
    """Generates playlists based on listening history and similarity"""

    def __init__(self, library_client, config, lastfm_client=None, track_matcher=None, metadata_client=None):
        self.library = library_client
        self.config = config
        self.lastfm = lastfm_client
        self.matcher = track_matcher
        self.metadata = metadata_client  # Metadata database for genre lookups
        # Expose similarity calculator if provided by the library client; fall back to new instance
        self.similarity_calc = getattr(library_client, 'similarity_calc', None)
        if self.similarity_calc is None:
            db_path = None
            try:
                db_path = self.config.get('library', {}).get('database_path', 'data/metadata.db')
            except Exception as e:
                logger.debug(f"Falling back to default metadata path after config read failure: {e}")
                db_path = 'data/metadata.db'
            self.similarity_calc = SimilarityCalculator(db_path=db_path, config=self.config)
        self.genre_similarity_cache = {}  # Cache for Last.FM similar tags

        # Initialize persistent artist similarity cache
        cache_expiry = self.config.get('playlists', 'cache_expiry_days', 30)
        self.artist_cache = ArtistSimilarityCache(expiry_days=cache_expiry)

        # Clean up expired entries on startup
        self.artist_cache.clear_expired()

    def _ensure_similarity_calculator(self) -> None:
        """Ensure similarity calculator exists (fallback to local init)."""
        if getattr(self, 'similarity_calc', None):
            return
        self.similarity_calc = getattr(self.library, 'similarity_calc', None)
        if self.similarity_calc is None:
            try:
                db_path = self.config.get('library', {}).get('database_path', 'data/metadata.db')
            except Exception as e:
                logger.debug(f"Falling back to default metadata path after config read failure: {e}")
                db_path = 'data/metadata.db'
            self.similarity_calc = SimilarityCalculator(db_path=db_path, config=self.config)

    def _similarity_config(self, limit_per_seed: Optional[int]) -> Dict[str, Any]:
        """Derive similarity generation settings (caps, pool sizes, per-seed limits)."""
        target_playlist_size = self.config.get('playlists', 'tracks_per_playlist', default=30)
        candidate_per_seed = limit_per_seed if limit_per_seed is not None else self.config.get('playlists', 'similar_per_seed', default=30)
        buffer_size = max(2, int(target_playlist_size * 1.0))  # 100% buffer
        pool_target = target_playlist_size + buffer_size
        artist_cap = max(6, getattr(self.config, 'max_tracks_per_artist', 6))
        return {
            'target_playlist_size': target_playlist_size,
            'candidate_per_seed': candidate_per_seed,
            'pool_target': pool_target,
            'artist_cap': artist_cap,
        }

    def _filter_long_tracks(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove tracks above maximum duration (seconds value from config)."""
        max_duration_seconds = self.config.get('playlists', 'max_track_duration_seconds', default=720)
        if not max_duration_seconds:
            return candidates
        max_duration_ms = _convert_seconds_to_ms(max_duration_seconds)
        return [c for c in candidates if (c.get('duration') or 0) <= max_duration_ms]

    def _build_seed_title_set(self, seeds: List[Dict[str, Any]]) -> set:
        """Normalize seed titles once for duplicate-title filtering."""
        seed_titles = {normalize_song_title(seed.get('title', '')) for seed in seeds}
        seed_titles.discard('')
        return seed_titles

    def _cap_candidates_by_artist(
        self, candidates: List[Dict[str, Any]], artist_cap: int, limit: int
    ) -> List[Dict[str, Any]]:
        """Apply artist cap and truncate to limit based on sonic score ordering."""
        sorted_candidates = sorted(candidates, key=lambda t: t.get('similarity_score', 0), reverse=True)
        capped = []
        artist_counts = Counter()
        for track in sorted_candidates:
            artist = (track.get('artist') or '').lower()
            if artist_counts[artist] >= artist_cap:
                continue
            artist_counts[artist] += 1
            capped.append(track)
            if len(capped) >= limit:
                break
        return capped

    def _score_genre_and_hybrid(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], float]]]:
        """Compute genre and hybrid scores; partition into pass/fail."""
        genre_pass = []
        genre_fail = []
        for track in candidates:
            seed_id = track.get('seed_rating_key')
            track_id = track.get('rating_key')
            if not seed_id or not track_id:
                continue

            seed_genres = self.similarity_calc._get_combined_genres(seed_id)
            cand_genres = self.similarity_calc._get_combined_genres(track_id)
            genre_sim = (
                self.similarity_calc.genre_calc.calculate_similarity(
                    seed_genres, cand_genres, method=self.similarity_calc.genre_method
                )
                if seed_genres and cand_genres
                else 0.0
            )

            hybrid = self.similarity_calc.calculate_hybrid_similarity(seed_id, track_id)
            if hybrid is None or hybrid <= 0:
                genre_fail.append((track, genre_sim))
                continue

            track['hybrid_score'] = hybrid
            track['genre_sim'] = genre_sim
            genre_pass.append(track)

        return genre_pass, genre_fail

    def _finalize_pool(
        self, candidates: List[Dict[str, Any]], artist_cap: int, pool_target: int
    ) -> List[Dict[str, Any]]:
        """Sort by hybrid score, enforce artist cap, and trim to pool_target."""
        final_pool = []
        artist_counts = Counter()
        for track in sorted(candidates, key=lambda t: t.get('hybrid_score', 0), reverse=True):
            artist = (track.get('artist') or '').lower()
            if artist_counts[artist] >= artist_cap:
                continue
            artist_counts[artist] += 1
            final_pool.append(track)
            if len(final_pool) >= pool_target:
                break
        return final_pool

    def _collect_sonic_candidates(
        self,
        seeds: List[Dict[str, Any]],
        seed_titles: set,
        candidate_per_seed: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Collect sonic-only candidates across seeds with basic filtering.

        Returns:
            (candidates, filtered_counts)
        """
        min_track_duration_ms = self.config.min_track_duration_seconds * 1000
        all_candidates: Dict[str, Dict[str, Any]] = {}
        filtered_short = 0
        filtered_long = 0
        filtered_dupe_title = 0

        for seed in seeds:
            seed_id = seed.get('rating_key')
            if not seed_id:
                continue

            similar = self.library.get_similar_tracks_sonic_only(seed_id, limit=candidate_per_seed, min_similarity=0.1)
            before_long = len(similar)
            similar = self._filter_long_tracks(similar)
            filtered_long += max(0, before_long - len(similar))

            weight = seed.get('play_count', 1)

            for track in similar:
                track_key = track.get('rating_key')
                # Skip tracks with same title as seed tracks (e.g., remasters, live versions)
                track_title_normalized = normalize_song_title(track.get('title', ''))
                if track_title_normalized in seed_titles:
                    filtered_dupe_title += 1
                    continue

                # Filter out short tracks (interludes, skits, etc.)
                track_duration = track.get('duration') or 0
                if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                    filtered_short += 1
                    continue

                existing = all_candidates.get(track_key)
                if existing and existing.get('similarity_score', 0) >= track.get('similarity_score', 0):
                    continue

                track['seed_artist'] = seed.get('artist')
                track['seed_title'] = seed.get('title')
                track['seed_rating_key'] = seed_id
                track['weight'] = weight
                track['source'] = 'sonic'
                all_candidates[track_key] = track

        return list(all_candidates.values()), {
            'short': filtered_short,
            'long': filtered_long,
            'dupe_title': filtered_dupe_title,
        }
    
    def analyze_listening_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze listening history to identify top artists/tracks

        Args:
            history: List of played tracks

        Returns:
            List of seed tracks (most frequently played)
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
        seed_count = self.config.get('playlists', 'seed_count', default=5)
        seeds = self._select_diverse_seeds(play_counts, track_metadata, seed_count)

        for seed in seeds:
            logger.info(f"Seed track: {seed['artist']} - {seed['title']} ({seed['play_count']} plays)")

        return seeds

    def _select_diverse_seeds(self, play_counts: Counter, track_metadata: Dict,
                              count: int) -> List[Dict[str, Any]]:
        """
        Select seed tracks ensuring artist diversity

        Args:
            play_counts: Counter of plays by rating key
            track_metadata: Metadata for each track
            count: Number of seeds to select

        Returns:
            List of diverse seed tracks
        """
        seeds = []
        used_artists = set()

        # First pass: one track per artist (by play count)
        for key, play_count in play_counts.most_common():
            track = track_metadata[key]
            artist = track.get('artist', 'Unknown')
            title = track.get('title', 'Unknown Track')

            if artist not in used_artists:
                seed_track = {
                    **track,
                    'play_count': play_count
                }
                seeds.append(seed_track)
                used_artists.add(artist)

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
    
    def generate_similar_tracks(self, seeds: List[Dict[str, Any]], dynamic: bool = False, limit_per_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate list of similar tracks based on seeds.

        Pipeline:
          1) Sonic-only discovery per seed (no genre filtering)
          2) Merge/dedup and apply per-artist cap
          3) Filter by genre threshold via hybrid similarity
          4) Rank by hybrid score and keep a buffered pool (target + 10%)

        Args:
            seeds: List of seed tracks
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)
            limit_per_seed: Override for similar_per_seed (useful for extending playlists)

        Returns:
            List of similar tracks (deduplicated and filtered for duplicate titles)
        """
        # Ensure similarity calculator is available (fallback if not injected)
        self._ensure_similarity_calculator()

        if dynamic:
            logger.info("Finding similar tracks (60% sonic, 40% genre-based)...")
            return self._generate_similar_tracks_dynamic(seeds, limit_per_seed=limit_per_seed)
        else:
            logger.info("Finding similar tracks (sonic-first pipeline)...")

        sim_cfg = self._similarity_config(limit_per_seed)

        seed_titles = self._build_seed_title_set(seeds)
        candidates, filtered_counts = self._collect_sonic_candidates(
            seeds, seed_titles, sim_cfg['candidate_per_seed']
        )

        logger.info(
            "Sonic-only pool: %s candidates (filtered %s short, %s long, %s duplicate titles)",
            len(candidates),
            filtered_counts['short'],
            filtered_counts['long'],
            filtered_counts['dupe_title'],
        )

        capped_candidates = self._cap_candidates_by_artist(
            candidates, sim_cfg['artist_cap'], sim_cfg['pool_target'] * 2
        )
        logger.info(f"After artist cap: {len(capped_candidates)} candidates")

        genre_pass, genre_fail = self._score_genre_and_hybrid(capped_candidates)
        logger.info(f"After genre filter/hybrid scoring: {len(genre_pass)} candidates")
        if genre_fail:
            for t, gsim in genre_fail[:5]:
                logger.debug(f"Genre-filtered: {t.get('artist')} - {t.get('title')} (genre_sim={gsim:.3f})")

        final_pool = self._finalize_pool(genre_pass, sim_cfg['artist_cap'], sim_cfg['pool_target'])
        logger.info(f"Selected {len(final_pool)} candidates for ordering (target pool: {sim_cfg['pool_target']})")
        return final_pool

    def _generate_similar_tracks_dynamic(self, seeds: List[Dict[str, Any]], limit_per_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate tracks using dynamic mode: mix sonic similarity with genre-based discovery
        60% from sonic analysis, 40% from genre matching

        Args:
            seeds: List of seed tracks (must have 'genres' field)
            limit_per_seed: Override for similar_per_seed (useful for extending playlists)

        Returns:
            Mixed list of sonically similar and genre-matched tracks
        """
        similar_per_seed = limit_per_seed if limit_per_seed is not None else self.config.get('playlists', 'similar_per_seed', default=10)
        min_track_duration_ms = self.config.min_track_duration_seconds * 1000

        # Calculate how many tracks from each source (from config)
        sonic_per_seed = int(similar_per_seed * self.config.dynamic_sonic_ratio)
        genre_per_seed = int(similar_per_seed * self.config.dynamic_genre_ratio)

        logger.info(f"  Target: {sonic_per_seed} sonic + {genre_per_seed} genre-based tracks per seed")

        # Part 1: Get sonic similarity tracks (existing logic)
        sonic_tracks = []
        seen_keys = set()
        filtered_short_count = 0
        filtered_long_count = 0

        # Build set of normalized seed titles to filter out
        seed_titles = {normalize_song_title(seed.get('title', '')) for seed in seeds}
        seed_titles.discard('')

        # Build set of seed artists to exclude from similarity results
        seed_artists = {seed.get('artist', '').lower() for seed in seeds}
        seed_artists.discard('')

        for seed in seeds:
            key = seed.get('rating_key')
            if not key:
                continue

            similar = self.library.get_similar_tracks(key, limit=sonic_per_seed)
            before_long = len(similar)
            similar = self._filter_long_tracks(similar)
            filtered_long_count += max(0, before_long - len(similar))
            weight = seed.get('play_count', 1)

            for track in similar:
                track_key = track.get('rating_key')

                if track_key in seen_keys:
                    continue

                # Skip tracks by seed artists
                track_artist = track.get('artist', '').lower()
                if track_artist in seed_artists:
                    continue

                # Skip tracks with same title as seeds
                track_title_normalized = normalize_song_title(track.get('title', ''))
                if track_title_normalized in seed_titles:
                    continue

                # Filter short tracks
                track_duration = track.get('duration') or 0
                if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
                    filtered_short_count += 1
                    continue

                seen_keys.add(track_key)
                track['seed_artist'] = seed.get('artist')
                track['seed_title'] = seed.get('title')
                track['weight'] = weight
                track['source'] = 'sonic'

                sonic_tracks.append(track)

        logger.info(f"  Sonic similarity: {len(sonic_tracks)} tracks")

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
        if self.metadata:
            # Get all artists and their genres from metadata database
            cursor = self.metadata.conn.cursor()
            cursor.execute("""
                SELECT artist_name, lastfm_tags
                FROM artists
                WHERE lastfm_tags IS NOT NULL AND lastfm_tags != ''
            """)

            # Calculate genre overlap for each artist
            # Normalize seed genres for matching (aggressive normalization)
            normalized_seed_genres = set(normalize_genre(g) for g in all_genres)

            artist_genre_scores = []
            for row in cursor.fetchall():
                artist_name = row['artist_name']
                # Parse JSON array of tags
                import json
                try:
                    artist_tags_raw = json.loads(row['lastfm_tags']) if row['lastfm_tags'] else []
                except (json.JSONDecodeError, TypeError):
                    # Fallback to comma-split if not JSON
                    artist_tags_raw = row['lastfm_tags'].split(',') if row['lastfm_tags'] else []

                # Normalize artist tags (aggressive normalization)
                artist_tags_normalized = set(normalize_genre(tag) for tag in artist_tags_raw)

                # Skip seed artists
                if any(safe_get_artist(seed) == artist_name.lower() for seed in seeds):
                    continue

                # Calculate how many seed genres this artist has (case-insensitive)
                matching_genres = artist_tags_normalized & normalized_seed_genres
                if matching_genres:
                    artist_genre_scores.append({
                        'artist': artist_name,
                        'matching_genres': list(matching_genres),
                        'match_count': len(matching_genres),
                        'all_tags': artist_tags_raw
                    })

            # Sort by match count (most overlapping genres first)
            artist_genre_scores.sort(key=lambda x: x['match_count'], reverse=True)

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

                # Get tracks from this artist (from metadata DB to get rating keys)
                matched_tracks = self.metadata.get_tracks_by_artist(artist_name, limit=5)

                # Find first track that's not already used
                for matched in matched_tracks:
                    track_key = matched.get('track_id')

                    # Skip if already in sonic tracks or genre tracks
                    if track_key in seen_keys:
                        continue

                    # Skip seeds
                    track_title_normalized = normalize_song_title(matched.get('title', ''))
                    if track_title_normalized in seed_titles:
                        continue

                    # Get full track data from library (includes duration)
                    full_track_data = self.library.get_track_by_key(track_key)
                    if not full_track_data:
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

                    seen_keys.add(track_key)
                    genre_tracks.append(track)
                    selected_artists.add(artist_name)

                    # Only take 1 track per artist
                    break

            logger.info(f"  Genre-based discovery: {len(genre_tracks)} tracks from {len(selected_artists)} artists")

        else:
            # Fallback to local library metadata (original implementation)
            logger.info("  Metadata database not available, falling back to local library genre metadata")

            # Get all tracks from library (cached)
            all_library_tracks = self.library.get_all_tracks()

            # Filter tracks by genre and diversify by artist
            artist_track_counts = {}

            for track in all_library_tracks:
                track_key = track.get('rating_key')

                # Skip if already in sonic tracks
                if track_key in seen_keys:
                    continue

                # Skip seeds
                track_title_normalized = normalize_song_title(track.get('title', ''))
                if track_title_normalized in seed_titles:
                    continue

                # Filter short tracks
                track_duration = track.get('duration') or 0
                if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
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
                artist = track.get('artist', 'Unknown')
                artist_track_counts[artist] = artist_track_counts.get(artist, 0) + 1

                # Max 2 tracks per artist in genre pool
                if artist_track_counts[artist] > 2:
                    continue

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
            logger.info(f"  Filtered out {filtered_short_count} short tracks (< {self.config.min_track_duration_seconds}s)")
        if filtered_long_count > 0:
            logger.info(f"  Filtered out {filtered_long_count} long tracks (> {self.config.max_track_duration_seconds}s)")

        logger.info(f"  Total: {len(all_tracks)} tracks ({len(sonic_tracks)} sonic + {len(genre_tracks)} genre)")

        return all_tracks

    def filter_tracks(self, tracks: List[Dict[str, Any]],
                     history: List[Dict[str, Any]],
                     exempt_tracks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Filter out recently played tracks to keep playlist fresh

        Args:
            tracks: List of candidate tracks
            history: Play history to exclude
            exempt_tracks: Optional list of tracks exempt from filtering (e.g., seed tracks)

        Returns:
            Filtered list of tracks
        """
        # Check if filtering is enabled
        if not self.config.recently_played_filter_enabled:
            logger.info("Recently played filtering is disabled")
            return tracks

        # Get filter configuration
        lookback_days = self.config.recently_played_lookback_days
        min_playcount = self.config.recently_played_min_playcount

        # Build set of tracks to filter based on configuration
        played_keys = set()
        play_counts = defaultdict(int)

        # Count plays per track
        for track in history:
            key = track.get('rating_key')
            if key:
                play_counts[key] += 1

        # Apply filtering rules
        if lookback_days > 0:
            # Filter only tracks played within lookback window
            from datetime import datetime, timedelta
            cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())

            for track in history:
                key = track.get('rating_key')
                timestamp = track.get('timestamp', 0)

                if key and timestamp >= cutoff_timestamp:
                    # Only filter if playcount threshold not met
                    if play_counts[key] >= min_playcount:
                        played_keys.add(key)
        else:
            # Filter all history (default behavior)
            for track in history:
                key = track.get('rating_key')
                if key and play_counts[key] >= min_playcount:
                    played_keys.add(key)

        # Build set of exempt track keys
        exempt_keys = set()
        if exempt_tracks:
            exempt_keys = {t.get('rating_key') for t in exempt_tracks if t.get('rating_key')}
            logger.info(f"  Exempting {len(exempt_keys)} seed tracks from filtering")

        # Filter out played tracks (except exempt tracks)
        filtered = [t for t in tracks if t.get('rating_key') not in played_keys or t.get('rating_key') in exempt_keys]

        filter_msg = f"Filtered {len(tracks)} -> {len(filtered)} tracks"
        if lookback_days > 0:
            filter_msg += f" (removed tracks played in last {lookback_days} days"
        else:
            filter_msg += f" (removed recently played tracks"
        if min_playcount > 0:
            filter_msg += f", playcount >= {min_playcount})"
        else:
            filter_msg += ")"

        logger.info(filter_msg)
        return filtered
    
    def diversify_tracks(self, tracks: List[Dict[str, Any]],
                        max_per_artist: int = None) -> List[Dict[str, Any]]:
        """
        Ensure playlist has diversity (don't let one artist dominate)

        Args:
            tracks: List of tracks
            max_per_artist: Maximum tracks per artist (uses config if None)

        Returns:
            Diversified track list
        """
        if max_per_artist is None:
            max_per_artist = self.config.max_tracks_per_artist

        artist_counts = Counter()
        diversified = []

        for track in tracks:
            artist = track.get('artist', 'Unknown')

            if artist_counts[artist] < max_per_artist:
                diversified.append(track)
                artist_counts[artist] += 1

        logger.info(f"Diversified {len(tracks)} -> {len(diversified)} tracks (max {max_per_artist} per artist)")
        return diversified
    
    def _is_collaboration_of(self, collaboration_name: str, base_artist: str) -> bool:
        """
        Check if a collaboration name includes the base artist

        Args:
            collaboration_name: Full artist name (potentially a collaboration)
            base_artist: Base artist name to check for

        Returns:
            True if collaboration_name is a collaboration including base_artist
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

    def _analyze_top_artists_from_history(self, history: List[Dict[str, Any]],
                                          artist_count: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze listening history to get top played artists and their tracks

        Args:
            history: List of played tracks
            artist_count: Number of top artists to return

        Returns:
            Dict mapping artist name to list of their played tracks
        """
        logger.info(f"Analyzing top {artist_count} artists from listening history...")

        # Group tracks by artist
        artist_tracks = defaultdict(list)
        for track in history:
            artist = track.get('artist')
            if artist:
                artist_tracks[artist].append(track)

        # Sort artists by total play count
        artist_play_counts = {
            artist: sum(t.get('play_count', 1) for t in tracks)
            for artist, tracks in artist_tracks.items()
        }

        top_artists = sorted(artist_play_counts.items(), key=lambda x: x[1], reverse=True)[:artist_count]

        result = {}
        for artist, total_plays in top_artists:
            exact_match_tracks = artist_tracks[artist]

            # Check if we have enough tracks (need at least 4 for seed selection)
            if len(exact_match_tracks) < 4:
                logger.info(f"  {artist}: Only {len(exact_match_tracks)} exact match tracks, searching for collaborations...")

                # Search for collaboration tracks
                collaboration_tracks = []
                for other_artist, other_tracks in artist_tracks.items():
                    if self._is_collaboration_of(other_artist, artist):
                        collaboration_tracks.extend(other_tracks)
                        logger.info(f"    Found collaboration: {other_artist} ({len(other_tracks)} tracks)")

                # Combine exact matches and collaborations
                combined_tracks = exact_match_tracks + collaboration_tracks
                logger.info(f"  {artist}: {total_plays} total plays across {len(combined_tracks)} tracks ({len(exact_match_tracks)} solo, {len(collaboration_tracks)} collaborations)")
                result[artist] = combined_tracks
            else:
                logger.info(f"  {artist}: {total_plays} total plays across {len(exact_match_tracks)} tracks")
                result[artist] = exact_match_tracks

        return result

    def _get_seed_tracks_for_artist(self, artist: str, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get seed tracks for an artist: 1 top played + random from top 20

        Args:
            artist: Artist name
            tracks: List of tracks by this artist

        Returns:
            List of seed tracks (count from config.seed_count)
        """
        if not tracks:
            return []

        # Get seed count from config
        seed_count = self.config.get('playlists', 'seed_count', default=5)

        # Sort by play count
        sorted_tracks = sorted(tracks, key=lambda x: x.get('play_count', 0), reverse=True)

        seeds = []

        # Top 1 most played track (guaranteed)
        if len(sorted_tracks) > 0:
            track = sorted_tracks[0]
            logger.info(f"    Top played: {artist} - {track.get('title')} ({track.get('play_count', 0)} plays)")
            seeds.append(track)

        # Random tracks from top 20 (excluding the #1 already selected)
        if len(sorted_tracks) > 1:
            # Get tracks 2-20 (or fewer if not enough tracks)
            top_20_pool = sorted_tracks[1:min(20, len(sorted_tracks))]

            # Pick (seed_count - 1) random from this pool (or fewer if not enough)
            num_random = min(seed_count - 1, len(top_20_pool))
            random_picks = random.sample(top_20_pool, num_random)

            for track in random_picks:
                logger.info(f"    Random from top 20: {artist} - {track.get('title')} ({track.get('play_count', 0)} plays)")
                seeds.append(track)
        else:
            logger.warning(f"    Only {len(sorted_tracks)} track(s) available for {artist}")

        return seeds

    def create_playlist_batch(self, count: int, dynamic: bool = False) -> List[List[Dict[str, Any]]]:
        """
        Create multiple playlists with single seed artist per playlist

        Args:
            count: Number of playlists to create
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            List of playlist dictionaries with tracks and metadata
        """
        logger.info(f"="*70)
        logger.info(f"Creating {count} playlists (1 seed artist per playlist)")
        if dynamic:
            logger.info("Dynamic mode enabled: 60% sonic similarity + 40% genre-based discovery")
        logger.info(f"="*70)

        # Get listening history - prefer Last.FM if available
        if self.lastfm and self.matcher:
            history = self._get_lastfm_history()
        else:
            history = self._get_local_history()

        if not history:
            logger.warning("No listening history found")
            return []

        # We need exactly 'count' artists (one per playlist)
        artists_needed = count

        logger.info(f"\nNeed {artists_needed} artists (one per playlist)")
        logger.info(f"Each artist will provide 4 seed tracks\n")

        # Get top played artists and their tracks
        top_artists_tracks = self._analyze_top_artists_from_history(history, artists_needed)

        if len(top_artists_tracks) < artists_needed:
            logger.warning(f"Only found {len(top_artists_tracks)} artists, need {artists_needed}")
            return []

        # Get seed tracks for each artist (2 most played + 2 random from top 10)
        logger.info(f"\n{'='*70}")
        logger.info("Selecting seed tracks for each artist:")
        logger.info(f"{'='*70}\n")

        artist_seeds = {}
        for artist, tracks in top_artists_tracks.items():
            logger.info(f"  Artist: {artist}")
            seeds = self._get_seed_tracks_for_artist(artist, tracks)
            if seeds:
                artist_seeds[artist] = seeds

        # Fetch genre tags for artists
        logger.info(f"\n{'='*70}")
        logger.info("Fetching genre tags for artists:")
        logger.info(f"{'='*70}\n")

        for artist in artist_seeds.keys():
            if self.lastfm:
                genres = self.lastfm.get_artist_tags(artist)
                # Add genres to all seed tracks for this artist
                for seed in artist_seeds[artist]:
                    seed['genres'] = genres
                if genres:
                    logger.info(f"  {artist}: {', '.join(genres)}")

        # Create playlists from single artists
        playlists = self._create_playlists_from_single_artists(artist_seeds, history, dynamic=dynamic)

        return playlists

    def create_playlist_for_artist(self, artist_name: str, track_count: int = 30, dynamic: bool = False, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a single playlist for a specific artist without requiring listening history

        Args:
            artist_name: Name of the artist to create playlist for
            track_count: Target number of tracks in playlist

        Returns:
            Playlist dictionary with tracks and metadata, or None if unable to create
        """
        logger.info(f"Creating playlist for artist: {artist_name}")

        # Get all tracks by this artist from local library
        all_library_tracks = self.library.get_all_tracks()

        # Filter to just this artist (case-insensitive, exact match)
        artist_tracks = [t for t in all_library_tracks
                        if safe_get_artist(t) == artist_name.lower()]

        # If we don't have enough tracks (or zero), search for collaborations
        if len(artist_tracks) < 4:
            logger.info(f"Artist has only {len(artist_tracks)} exact match tracks, searching for collaborations...")

            # Search for collaboration tracks
            collaboration_tracks = [t for t in all_library_tracks
                                  if self._is_collaboration_of(t.get('artist', ''), artist_name)]

            if collaboration_tracks:
                # Group by collaboration artist name for logging
                collab_artists = {}
                for track in collaboration_tracks:
                    collab_artist = track.get('artist', '')
                    if collab_artist not in collab_artists:
                        collab_artists[collab_artist] = 0
                    collab_artists[collab_artist] += 1

                for collab_artist, count in collab_artists.items():
                    logger.info(f"  Found collaboration: {collab_artist} ({count} tracks)")

                # Combine exact matches and collaborations
                artist_tracks.extend(collaboration_tracks)
                logger.info(f"Found {len(artist_tracks)} total tracks ({len(artist_tracks) - len(collaboration_tracks)} solo, {len(collaboration_tracks)} collaborations)")
            else:
                logger.warning(f"Artist has only {len(artist_tracks)} tracks and no collaborations found, need at least 4")
                return None

        if len(artist_tracks) < 4:
            logger.warning(f"Artist has only {len(artist_tracks)} total tracks (including collaborations), need at least 4")
            return None

        logger.info(f"Using {len(artist_tracks)} tracks for {artist_name}")

        # Add play count (0 for all, since we don't have history)
        for track in artist_tracks:
            track['play_count'] = 0

        # Get 4 seed tracks (2 most played + 2 random from top 10)
        # Since we don't have play counts, just pick 4 random tracks
        import random
        seed_tracks = random.sample(artist_tracks, min(4, len(artist_tracks)))

        seed_titles = [track.get('title') for track in seed_tracks]
        logger.info(f"Seeds ({len(seed_tracks)}): {', '.join(seed_titles)}")

        # Fetch genre tags if Last.FM is available
        if self.lastfm:
            genres = self.lastfm.get_artist_tags(artist_name)
            for seed in seed_tracks:
                seed['genres'] = genres
            if genres:
                logger.info(f"Genres: {', '.join(genres)}")

        # Generate similar tracks using similarity engine
        # Use larger pool for TSP optimization (2-3x target count)
        tsp_pool_multiplier = 2.5  # Gather 2.5x more candidates for TSP to optimize
        limit_per_seed = int((track_count * tsp_pool_multiplier) / len(seed_tracks)) if seed_tracks else 20
        similar_tracks = self.generate_similar_tracks(seed_tracks, dynamic=dynamic, limit_per_seed=limit_per_seed)
        logger.info(f"  Generated pool of {len(similar_tracks)} candidates (target: {track_count}, multiplier: {tsp_pool_multiplier}x)")

        # Keep the full pool; artist/window limits and caps are enforced downstream
        diverse_tracks = similar_tracks

        # Ensure minimum seed artist representation (configurable ratio)
        min_seed_tracks = max(4, int(track_count * self.config.min_seed_artist_ratio))
        seed_artist_tracks = [t for t in diverse_tracks if safe_get_artist(t) == artist_name.lower()]

        additional_count = 0
        if len(seed_artist_tracks) < min_seed_tracks:
            additional_needed = min_seed_tracks - len(seed_artist_tracks)
            # Get more tracks from the artist
            available_tracks = [t for t in artist_tracks
                              if t.get('rating_key') not in {track.get('rating_key') for track in diverse_tracks}]
            if available_tracks:
                additional = available_tracks[:additional_needed]
                diverse_tracks.extend(additional)
                additional_count = len(additional)

        logger.info(f"  Assembled {len(diverse_tracks)} candidates ({len(seed_artist_tracks) + additional_count} from {artist_name})")

        # Optimize track order using TSP (end→beginning transitions)
        # Intelligent selection happens inside TSP method
        final_tracks = self._optimize_playlist_order_tsp(
            diverse_tracks,
            seed_tracks,
            verbose=verbose,
            target_count=track_count  # Enable intelligent candidate selection
        )

        # Limit artist frequency in windows (configurable)
        final_tracks = self._limit_artist_frequency_in_window(final_tracks)

        # Note: No naive trimming needed - intelligent selection happens before TSP

        # Ensure playlist meets minimum duration
        min_duration_ms = self.config.min_duration_minutes * 60 * 1000  # Convert minutes to milliseconds
        total_duration_ms = sum((track.get('duration') or 0) for track in final_tracks)

        # If playlist is shorter than minimum, add more similar tracks
        if total_duration_ms < min_duration_ms:
            logger.info(f"  Playlist is {total_duration_ms/1000/60:.1f} min, extending to {self.config.min_duration_minutes} min...")

            # Track all used rating keys to avoid duplicates
            used_keys = {track.get('rating_key') for track in final_tracks}
            used_titles = {normalize_song_title(track.get('title', '')) for track in final_tracks}

            # Keep fetching more similar tracks until we meet duration
            attempts = 0
            max_attempts = 10

            while total_duration_ms < min_duration_ms and attempts < max_attempts:
                # Generate more similar tracks (use higher limit to get more variety)
                # Increase limit with each attempt to get progressively more tracks
                extended_limit = self.config.limit_extension_base + (attempts * self.config.limit_extension_increment)
                more_similar = self.generate_similar_tracks(seed_tracks, dynamic=dynamic, limit_per_seed=extended_limit)

                # Filter out duplicates and already-used tracks
                new_tracks = []
                for track in more_similar:
                    track_key = track.get('rating_key')
                    track_title_normalized = normalize_song_title(track.get('title', ''))

                    if track_key not in used_keys and track_title_normalized not in used_titles:
                        new_tracks.append(track)
                        used_keys.add(track_key)
                        used_titles.add(track_title_normalized)

                if not new_tracks:
                    logger.info(f"  No more unique tracks available, stopping at {total_duration_ms/1000/60:.1f} min")
                    break

                # Keep full set; diversity enforced by window/artist checks below

                # Add tracks until we reach minimum duration
                for track in new_tracks:
                    if total_duration_ms >= min_duration_ms:
                        break

                    # Check artist frequency constraint for this new track
                    # Look at last (window_size - 1) tracks
                    window = final_tracks[-(8 - 1):]
                    artist_counts = {}
                    for t in window:
                        artist = t.get('artist')
                        artist_counts[artist] = artist_counts.get(artist, 0) + 1

                    # Only add if it doesn't violate constraint
                    track_artist = track.get('artist')
                    if artist_counts.get(track_artist, 0) < 1:  # max 1 per 8-song window
                        final_tracks.append(track)
                        total_duration_ms += (track.get('duration') or 0)

                attempts += 1

            logger.info(f"  Extended playlist to {len(final_tracks)} tracks ({total_duration_ms/1000/60:.1f} min)")

        if not final_tracks:
            logger.warning(f"No tracks generated for {artist_name}")
            return None

        # Log final duration
        total_minutes = total_duration_ms / 1000 / 60
        logger.info(f"  Playlist duration: {total_minutes:.1f} minutes")

        # Count seed artist tracks in final playlist
        seed_count = len([t for t in final_tracks if safe_get_artist(t) == artist_name.lower()])
        seed_percentage = (seed_count / len(final_tracks)) * 100 if final_tracks else 0

        logger.info(f"  Final: {len(final_tracks)} tracks from {len(set(t.get('artist') for t in final_tracks))} artists")
        logger.info(f"  Seed artist ({artist_name}): {seed_count} tracks ({seed_percentage:.1f}%)")

        # Print detailed track report
        self._print_playlist_report(final_tracks, artist_name=artist_name, dynamic=dynamic)

        return {
            'title': artist_name,
            'artists': (artist_name,),
            'genres': seed_tracks[0].get('genres', []) if seed_tracks else [],
            'tracks': final_tracks
        }

    def _pair_artists_by_similarity(self, artists: List[str], num_pairs: int,
                                     min_similarity: float = None) -> List[Tuple[str, str]]:
        """
        Pair artists by similarity, removing most dissimilar pair first

        Args:
            artists: List of artist names
            num_pairs: Number of pairs to create
            min_similarity: Minimum similarity threshold for pairing (0.0-1.0)

        Returns:
            List of artist pairs (tuples)
        """
        if min_similarity is None:
            min_similarity = self.config.similarity_min_threshold

        logger.info(f"\n{'='*70}")
        logger.info(f"Pairing {len(artists)} artists by similarity:")
        logger.info(f"{'='*70}\n")

        if len(artists) < num_pairs * 2:
            logger.warning(f"Not enough artists to create {num_pairs} pairs")
            num_pairs = len(artists) // 2

        # Build similarity matrix between all artists
        # Create dummy seeds with just artist names for similarity calculation
        artist_seeds_dummy = [{'artist': a, 'genres': []} for a in artists]

        # Fetch similar artists for each
        for seed in artist_seeds_dummy:
            artist = seed['artist']
            similar = self._get_similar_artists(artist)
            seed['similar_artists'] = similar

        # Calculate pairwise similarity
        similarity_matrix = {}
        for i, seed1 in enumerate(artist_seeds_dummy):
            for j, seed2 in enumerate(artist_seeds_dummy):
                if i >= j:
                    continue
                sim = self._calculate_artist_similarity(seed1, seed2)
                similarity_matrix[(i, j)] = sim

        # Find and remove most dissimilar pair
        if len(artists) > num_pairs * 2:
            sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1])
            worst_pair_idx, worst_sim = sorted_pairs[0]
            idx1, idx2 = worst_pair_idx
            artist1, artist2 = artists[idx1], artists[idx2]

            logger.info(f">> REMOVING MOST DISSIMILAR PAIR:")
            logger.info(f"   {artist1} <-> {artist2} (similarity = {worst_sim:.2f})")
            logger.info(f"   This ensures all remaining pairs are cohesive\n")

            # Remove both artists from the pool
            artists = [a for i, a in enumerate(artists) if i not in [idx1, idx2]]

        # Pair up remaining artists by highest similarity
        logger.info("Creating artist pairs by similarity:\n")
        pairs = []
        used = set()

        # Build new similarity matrix with current artist indices
        new_similarity_matrix = {}
        for i in range(len(artists)):
            for j in range(i + 1, len(artists)):
                # Calculate similarity between these two artists
                seed1 = {'artist': artists[i], 'genres': []}
                seed2 = {'artist': artists[j], 'genres': []}
                seed1['similar_artists'] = self._get_similar_artists(artists[i])
                seed2['similar_artists'] = self._get_similar_artists(artists[j])

                sim = self._calculate_artist_similarity(seed1, seed2)
                new_similarity_matrix[(i, j)] = sim

        similarity_scores = [(sim, i, j) for (i, j), sim in new_similarity_matrix.items()]

        # Sort by similarity (highest first)
        similarity_scores.sort(reverse=True)

        logger.info(f"Pairing artists (minimum similarity = {min_similarity}):\n")

        for sim, i, j in similarity_scores:
            if i not in used and j not in used and len(pairs) < num_pairs:
                # Only pair if similarity meets threshold
                if sim >= min_similarity:
                    pairs.append((artists[i], artists[j]))
                    used.add(i)
                    used.add(j)
                    logger.info(f"  Pair {len(pairs)}: {artists[i]} <-> {artists[j]} (similarity = {sim:.2f})")
                else:
                    # Log rejected pairs that don't meet threshold
                    logger.debug(f"  Rejected: {artists[i]} <-> {artists[j]} (similarity = {sim:.2f} < {min_similarity})")

        if len(pairs) < num_pairs:
            logger.warning(f"\nOnly created {len(pairs)} pairs (requested {num_pairs})")
            logger.warning(f"Could not find enough artist pairs with similarity >= {min_similarity}")
            logger.warning(f"Consider increasing artist pool or lowering similarity threshold\n")
        else:
            logger.info(f"\nCreated {len(pairs)} artist pairs (all with similarity >= {min_similarity})\n")

        return pairs

    def _generate_playlist_title(self, artist1: str, artist2: str, genres: List[str]) -> str:
        """
        Generate playlist title based on artist(s) and configured format

        Args:
            artist1: First artist name (or only artist for single-artist playlists)
            artist2: Second artist name (empty string for single-artist playlists)
            genres: Combined genre tags

        Returns:
            Generated playlist title (without prefix)
        """
        name_format = self.config.get('playlists', 'name_format', 'artists')

        # Determine if single or paired artists
        is_single_artist = not artist2 or artist2 == ""

        if name_format == 'artists':
            # Format: Artist1 or Artist1 + Artist2
            if is_single_artist:
                return artist1
            else:
                return f"{artist1} + {artist2}"

        elif name_format == 'genres':
            # Format: Dominant Genre Style
            if genres:
                # Take first two genres
                genre_text = " & ".join(genres[:2]).title()
                return f"{genre_text} Mix"
            else:
                # Fallback to artist names if no genres
                if is_single_artist:
                    return artist1
                else:
                    return f"{artist1} + {artist2}"

        elif name_format == 'hybrid':
            # Format: Artist1 (Genre) or Artist1 + Artist2 (Genre)
            if genres:
                genre_text = genres[0].title()
                if is_single_artist:
                    return f"{artist1} ({genre_text})"
                else:
                    return f"{artist1} + {artist2} ({genre_text})"
            else:
                if is_single_artist:
                    return artist1
                else:
                    return f"{artist1} + {artist2}"

        else:
            # Default to artists
            if is_single_artist:
                return artist1
            else:
                return f"{artist1} + {artist2}"

    def _create_playlists_from_pairs(self, artist_pairs: List[Tuple[str, str]],
                                    artist_seeds: Dict[str, List[Dict[str, Any]]],
                                    history: List[Dict[str, Any]],
                                    dynamic: bool = False) -> List[Dict[str, Any]]:
        """
        Create playlists from artist pairs using their seed tracks

        Args:
            artist_pairs: List of (artist1, artist2) tuples
            artist_seeds: Dict mapping artist name to their seed tracks
            history: Play history for filtering

        Returns:
            List of playlist dictionaries with tracks and metadata
        """
        logger.info(f"{'='*70}")
        logger.info(f"Generating playlists from artist pairs:")
        logger.info(f"{'='*70}\n")

        playlists = []

        for idx, (artist1, artist2) in enumerate(artist_pairs, 1):
            logger.info(f"Playlist {idx}: {artist1} + {artist2}")

            # Get seed tracks for both artists (4 total)
            seeds = artist_seeds.get(artist1, []) + artist_seeds.get(artist2, [])

            for seed in seeds:
                logger.info(f"  Seed: {seed.get('artist')} - {seed.get('title')}")

            # Collect all genres from seeds
            all_genres = []
            for seed in seeds:
                all_genres.extend(seed.get('genres', []))
            # Get unique genres while preserving order
            seen = set()
            unique_genres = [g for g in all_genres if not (g in seen or seen.add(g))]

            # Generate similar tracks using similarity engine
            similar_tracks = self.generate_similar_tracks(seeds, dynamic=dynamic)
            logger.info(f"  Generated {len(similar_tracks)} similar tracks")

            # Add seed tracks to the candidate pool (they're guaranteed to be in playlist)
            all_tracks = seeds + similar_tracks
            logger.info(f"  Total candidates: {len(all_tracks)} tracks ({len(seeds)} seeds + {len(similar_tracks)} similar)")

            # Filter out recently played (seeds are exempt)
            fresh_tracks = self.filter_tracks(all_tracks, history, exempt_tracks=seeds)
            logger.info(f"  After filtering: {len(fresh_tracks)} fresh tracks")

            # Diversify
            diverse_tracks = self.diversify_tracks(fresh_tracks)
            logger.info(f"  After diversification: {len(diverse_tracks)} tracks")

            # Optimize track order using TSP (end→beginning transitions)
            final_tracks = self._optimize_playlist_order_tsp(diverse_tracks, seeds)
            logger.info(f"  TSP optimization complete: {len(final_tracks)} tracks")

            # Limit artist frequency in 8-song windows (max 1 per window)
            final_tracks = self._limit_artist_frequency_in_window(final_tracks, window_size=8, max_per_window=1)
            logger.info(f"  After limiting artist frequency: {len(final_tracks)} tracks")

            # Ensure playlist meets minimum duration
            min_duration_ms = self.config.min_duration_minutes * 60 * 1000  # Convert minutes to milliseconds
            total_duration_ms = 0
            duration_limited_tracks = []

            for track in final_tracks:
                duration_limited_tracks.append(track)
                total_duration_ms += (track.get('duration') or 0)

                # Stop once we've reached the minimum duration
                if total_duration_ms >= min_duration_ms:
                    break

            final_tracks = duration_limited_tracks

            # Log final duration
            total_minutes = total_duration_ms / 1000 / 60
            logger.info(f"  Playlist duration: {total_minutes:.1f} minutes")

            if final_tracks:
                # Generate title for this playlist
                title = self._generate_playlist_title(artist1, artist2, unique_genres)

                # Print detailed track report
                self._print_playlist_report(final_tracks, artist_name=f"{artist1} + {artist2}", dynamic=dynamic)

                playlists.append({
                    'title': title,
                    'artists': (artist1, artist2),
                    'genres': unique_genres,
                    'tracks': final_tracks
                })
                logger.info(f"  Title: {title}")
                logger.info(f"  Final: {len(final_tracks)} tracks from {len(set(t.get('artist') for t in final_tracks))} artists\n")
            else:
                logger.warning(f"  No tracks generated for this pair\n")

        return playlists

    def _create_playlists_from_single_artists(self, artist_seeds: Dict[str, List[Dict[str, Any]]],
                                              history: List[Dict[str, Any]],
                                              dynamic: bool = False) -> List[Dict[str, Any]]:
        """
        Create playlists from single artists with advanced requirements:
        - At least 1/8 of tracks must be by the seed artist
        - Tracks ordered by sequential sonic similarity
        - No artist plays twice in a row

        Args:
            artist_seeds: Dict mapping artist name to their seed tracks
            history: Play history for filtering
            dynamic: Enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            List of playlist dictionaries with tracks and metadata
        """
        logger.info(f"{'='*70}")
        logger.info(f"Generating playlists from single artists:")
        logger.info(f"{'='*70}\n")

        playlists = []

        for idx, (artist, seeds) in enumerate(artist_seeds.items(), 1):
            logger.info(f"Playlist {idx}: {artist}")

            for seed in seeds:
                logger.info(f"  Seed: {artist} - {seed.get('title')}")

            # Collect genres from seeds
            all_genres = []
            for seed in seeds:
                all_genres.extend(seed.get('genres', []))
            seen = set()
            unique_genres = [g for g in all_genres if not (g in seen or seen.add(g))]

            # Generate similar tracks using similarity engine
            similar_tracks = self.generate_similar_tracks(seeds, dynamic=dynamic)
            logger.info(f"  Generated {len(similar_tracks)} similar tracks")

            # Add seed tracks to the candidate pool (they're guaranteed to be in playlist)
            all_tracks = seeds + similar_tracks
            logger.info(f"  Total candidates: {len(all_tracks)} tracks ({len(seeds)} seeds + {len(similar_tracks)} similar)")

            # Filter out recently played (seeds are exempt)
            fresh_tracks = self.filter_tracks(all_tracks, history, exempt_tracks=seeds)
            logger.info(f"  After filtering: {len(fresh_tracks)} fresh tracks")

            # Diversify tracks
            diverse_tracks = self.diversify_tracks(fresh_tracks)
            logger.info(f"  After diversification: {len(diverse_tracks)} tracks")

            # Ensure seed artist quota (configurable ratio)
            target_playlist_size = self.config.get('playlists', 'tracks_per_playlist', 30)
            min_seed_artist_tracks = max(4, int(target_playlist_size * self.config.min_seed_artist_ratio))

            seed_artist_tracks = [t for t in diverse_tracks if t.get('artist') == artist]
            logger.info(f"  Seed artist tracks in pool: {len(seed_artist_tracks)}")

            # If not enough seed artist tracks, add them from history
            if len(seed_artist_tracks) < min_seed_artist_tracks:
                logger.info(f"  Need {min_seed_artist_tracks - len(seed_artist_tracks)} more {artist} tracks")
                additional_tracks = self._get_additional_artist_tracks(artist, history, diverse_tracks,
                                                                       min_seed_artist_tracks - len(seed_artist_tracks))
                diverse_tracks.extend(additional_tracks)
                logger.info(f"  Added {len(additional_tracks)} additional {artist} tracks")

            # Optimize track order using TSP (end→beginning transitions)
            final_tracks = self._optimize_playlist_order_tsp(diverse_tracks, seeds)
            logger.info(f"  TSP optimization complete: {len(final_tracks)} tracks")

            # Limit artist frequency in 8-song windows (max 1 per window)
            final_tracks = self._limit_artist_frequency_in_window(final_tracks, window_size=8, max_per_window=1)
            logger.info(f"  After limiting artist frequency: {len(final_tracks)} tracks")

            # Ensure playlist meets minimum duration
            min_duration_ms = self.config.min_duration_minutes * 60 * 1000  # Convert minutes to milliseconds
            total_duration_ms = 0
            duration_limited_tracks = []

            for track in final_tracks:
                duration_limited_tracks.append(track)
                total_duration_ms += (track.get('duration') or 0)

                # Stop once we've reached the minimum duration
                if total_duration_ms >= min_duration_ms:
                    break

            final_tracks = duration_limited_tracks

            # Log final duration
            total_minutes = total_duration_ms / 1000 / 60
            logger.info(f"  Playlist duration: {total_minutes:.1f} minutes")

            # Count final seed artist representation
            final_seed_artist_count = sum(1 for t in final_tracks if t.get('artist') == artist)
            seed_artist_percentage = (final_seed_artist_count / len(final_tracks) * 100) if final_tracks else 0

            if final_tracks:
                # Generate title
                title = self._generate_playlist_title(artist, "", unique_genres)

                # Print detailed track report
                self._print_playlist_report(final_tracks, artist_name=artist, dynamic=dynamic)

                playlists.append({
                    'title': title,
                    'artists': (artist,),
                    'genres': unique_genres,
                    'tracks': final_tracks
                })
                logger.info(f"  Title: {title}")
                logger.info(f"  Final: {len(final_tracks)} tracks from {len(set(t.get('artist') for t in final_tracks))} artists")
                logger.info(f"  Seed artist ({artist}): {final_seed_artist_count} tracks ({seed_artist_percentage:.1f}%)\n")
            else:
                logger.warning(f"  No tracks generated for {artist}\n")

        return playlists

    def _get_additional_artist_tracks(self, artist: str, history: List[Dict[str, Any]],
                                      existing_tracks: List[Dict[str, Any]],
                                      count: int) -> List[Dict[str, Any]]:
        """
        Get additional tracks from the seed artist to meet quota

        Args:
            artist: Artist name
            history: Listening history
            existing_tracks: Tracks already in the playlist
            count: Number of additional tracks needed

        Returns:
            List of additional artist tracks
        """
        existing_keys = {t.get('rating_key') for t in existing_tracks}
        artist_tracks_from_history = [t for t in history if t.get('artist') == artist
                                      and t.get('rating_key') not in existing_keys]

        # Sort by play count
        artist_tracks_from_history.sort(key=lambda x: x.get('play_count', 0), reverse=True)

        # Return up to 'count' tracks
        return artist_tracks_from_history[:count]

    def _order_by_sequential_similarity(self, tracks: List[Dict[str, Any]],
                                        seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Order tracks so each track is sonically similar to adjacent tracks

        In dynamic mode, this ensures genre-matched songs are interleaved based on
        their sonic similarity rather than clustered at the end.

        Uses a greedy nearest-neighbor approach starting from seed tracks

        Args:
            tracks: Tracks to order
            seeds: Seed tracks to start from

        Returns:
            Ordered list of tracks
        """
        if not tracks:
            return []

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
                similar = self.library.get_similar_tracks(track_key, limit=self.config.limit_similar_tracks)
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

        return ordered

    def _remove_consecutive_artists(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reorder tracks to ensure no artist plays twice in a row

        Args:
            tracks: Ordered tracks

        Returns:
            Tracks with no consecutive artists
        """
        if len(tracks) <= 1:
            return tracks

        result = [tracks[0]]
        remaining = tracks[1:]

        while remaining:
            last_artist = result[-1].get('artist')

            # Find the first track with a different artist
            next_track = None
            for i, track in enumerate(remaining):
                if track.get('artist') != last_artist:
                    next_track = remaining.pop(i)
                    break

            if next_track:
                result.append(next_track)
            else:
                # All remaining tracks are by the same artist
                # Just add them (can't avoid consecutive in this case)
                result.extend(remaining)
                break

        return result

    def _build_transition_matrix(self, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Build distance matrix for TSP optimization

        Calculates transition quality (end→beginning similarity) between all track pairs.
        Lower distance = better transition.

        Args:
            tracks: List of candidate tracks

        Returns:
            NxN distance matrix (numpy array)
        """
        n = len(tracks)
        distance_matrix = np.zeros((n, n))

        logger.info(f"Building transition matrix for {n} tracks...")
        start_time = time.time()

        # Get SimilarityCalculator from library client
        similarity_calc = self.library.similarity_calc

        for i in range(n):
            for j in range(n):
                if i == j:
                    # No self-loops
                    distance_matrix[i][j] = float('inf')
                else:
                    # Calculate transition similarity (end of i → beginning of j)
                    track_i_key = tracks[i].get('rating_key')
                    track_j_key = tracks[j].get('rating_key')

                    if not track_i_key or not track_j_key:
                        distance_matrix[i][j] = 1.0  # Max distance if missing keys
                        continue

                    try:
                        similarity = similarity_calc.calculate_transition_similarity(
                            track_i_key,
                            track_j_key
                        )

                        if similarity is None:
                            # If calculation failed, use maximum distance
                            distance_matrix[i][j] = 1.0
                        else:
                            # Convert similarity to distance (0.0-1.0)
                            distance_matrix[i][j] = 1.0 - similarity

                    except Exception as e:
                        logger.debug(f"Error calculating transition {i}→{j}: {e}")
                        distance_matrix[i][j] = 1.0

        elapsed = time.time() - start_time
        logger.info(f"Transition matrix built in {elapsed:.2f}s ({n*n} comparisons)")

        return distance_matrix

    def _select_best_connected_tracks(self, tracks: List[Dict[str, Any]],
                                      transition_matrix: np.ndarray,
                                      target_count: int,
                                      seed_tracks: List[Dict[str, Any]],
                                      verbose: bool = False) -> tuple:
        """
        Select tracks with best overall transition connectivity

        Instead of optimizing all tracks then trimming, this selects the subset
        of tracks that have the highest average transition quality with each other.

        Strategy:
        1. Calculate bidirectional connectivity score for each track
        2. Bonus for seed artist tracks (preserve representation)
        3. Select top N tracks
        4. Build new transition matrix for selected subset

        Args:
            tracks: Full candidate pool
            transition_matrix: NxN distance matrix for all candidates
            target_count: Number of tracks to select
            seed_tracks: Seed tracks (for artist bonus)
            verbose: Enable detailed logging

        Returns:
            (selected_tracks, subset_transition_matrix)
        """
        n = len(tracks)

        # If pool is already at or below target, no selection needed
        if n <= target_count:
            logger.debug(f"Pool size ({n}) <= target ({target_count}), using all tracks")
            return tracks, transition_matrix

        logger.info(f"Selecting best {target_count} connected tracks from pool of {n}...")

        connectivity_scores = []

        # Calculate connectivity score for each track
        for i in range(n):
            # Outgoing: How well this track transitions TO others
            outgoing_distances = [transition_matrix[i][j] for j in range(n) if i != j]
            outgoing_sim = sum(1.0 - d for d in outgoing_distances) / len(outgoing_distances)

            # Incoming: How well others transition TO this track
            incoming_distances = [transition_matrix[j][i] for j in range(n) if i != j]
            incoming_sim = sum(1.0 - d for d in incoming_distances) / len(incoming_distances)

            # Combined connectivity score (bidirectional)
            base_score = (outgoing_sim + incoming_sim) / 2.0

            # Bonus for seed artist tracks (ensure representation)
            is_seed_artist = any(
                tracks[i].get('artist', '').lower() == seed.get('artist', '').lower()
                for seed in seed_tracks
            )

            if is_seed_artist:
                bonus = 0.1  # 10% bonus for seed artist
            else:
                bonus = 0.0

            final_score = base_score + bonus
            connectivity_scores.append((i, final_score, tracks[i]))

        # Sort by score (highest connectivity first)
        connectivity_scores.sort(key=lambda x: x[1], reverse=True)

        # Verbose logging: Show selection scores
        if verbose:
            logger.info("=" * 80)
            logger.info("CANDIDATE SELECTION SCORES")
            logger.info("=" * 80)

            # Show top candidates
            display_count = min(15, len(connectivity_scores))
            for i, (idx, score, track) in enumerate(connectivity_scores[:display_count]):
                artist = track.get('artist', 'Unknown')[:25]
                title = track.get('title', 'Unknown')[:35]
                selected = "✓ SELECTED" if i < target_count else "  skipped"
                logger.info(f"{selected} | Score: {score:.3f} | {artist} - {title}...")

            if len(connectivity_scores) > display_count:
                logger.info(f"  ... and {len(connectivity_scores) - display_count} more candidates")

            logger.info("=" * 80)

        # Select top N tracks
        selected_indices = [idx for idx, score, track in connectivity_scores[:target_count]]
        selected_tracks = [track for idx, score, track in connectivity_scores[:target_count]]

        logger.info(f"Selected {len(selected_tracks)} tracks (avg connectivity: "
                   f"{sum(score for _, score, _ in connectivity_scores[:target_count]) / target_count:.3f})")

        # Build subset transition matrix for selected tracks
        subset_size = len(selected_indices)
        subset_matrix = np.zeros((subset_size, subset_size))

        for i, orig_i in enumerate(selected_indices):
            for j, orig_j in enumerate(selected_indices):
                subset_matrix[i][j] = transition_matrix[orig_i][orig_j]

        return selected_tracks, subset_matrix

    def _enforce_artist_window(self, tracks: List[Dict[str, Any]], window_size: int = 8) -> List[Dict[str, Any]]:
        """
        Enforce artist window constraint on track sequence

        Ensures no artist appears within window_size tracks of their last appearance.
        When violations found, attempts to swap with later tracks to fix.

        Args:
            tracks: Ordered track list (from TSP or other ordering)
            window_size: Number of tracks to check back

        Returns:
            Track list with artist window violations removed/fixed
        """
        result = []
        artist_positions = {}  # artist -> list of positions in result
        skipped = 0

        for i, track in enumerate(tracks):
            artist = track.get('artist', 'Unknown')

            # Check if artist appeared within window
            if artist in artist_positions:
                recent_positions = [p for p in artist_positions[artist] if len(result) - p <= window_size]

                if recent_positions:
                    # Violation detected - try to find a swap candidate
                    swapped = False

                    # Look ahead for a valid track to swap with
                    for swap_idx in range(i + 1, min(i + 10, len(tracks))):
                        swap_artist = tracks[swap_idx].get('artist', 'Unknown')

                        # Check if swap candidate would be valid
                        if swap_artist not in artist_positions:
                            swap_valid = True
                        else:
                            swap_recent = [p for p in artist_positions[swap_artist] if len(result) - p <= window_size]
                            swap_valid = len(swap_recent) == 0

                        if swap_valid:
                            # Swap tracks
                            tracks[i], tracks[swap_idx] = tracks[swap_idx], tracks[i]
                            track = tracks[i]
                            artist = track.get('artist', 'Unknown')
                            swapped = True
                            logger.debug(f"Swapped position {i} with {swap_idx} to fix window violation")
                            break

                    if not swapped:
                        # No valid swap found - skip this track
                        logger.debug(f"Skipping {artist} - {track.get('title')} due to window violation (no valid swap)")
                        skipped += 1
                        continue

            # Add track to result
            result.append(track)

            # Track artist position
            if artist not in artist_positions:
                artist_positions[artist] = []
            artist_positions[artist].append(len(result) - 1)

        if skipped > 0:
            logger.info(f"Enforced artist window: {skipped} tracks skipped")

        return result

    def _optimize_playlist_order_tsp(self, tracks: List[Dict[str, Any]], seeds: List[Dict[str, Any]],
                                     verbose: bool = False, target_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Optimize track ordering using TSP (Traveling Salesman Problem)

        Finds the sequence that maximizes overall sonic cohesion by comparing
        end→beginning transitions between all tracks.

        Args:
            tracks: Pool of candidate tracks (should be larger than needed)
            seeds: Seed tracks (used to determine starting point)
            verbose: Enable detailed logging
            target_count: Target number of tracks (enables intelligent selection)

        Returns:
            Optimally ordered tracks with artist window constraint enforced
        """
        if not tracks:
            return []

        if len(tracks) == 1:
            return tracks

        logger.info(f"Optimizing track order using TSP for {len(tracks)} candidates...")
        start_time = time.time()

        try:
            # Import TSP solver
            from python_tsp.heuristics import solve_tsp_simulated_annealing
            from python_tsp.exact import solve_tsp_dynamic_programming
        except ImportError:
            logger.error("python-tsp not installed. Run: pip install python-tsp")
            logger.warning("Falling back to greedy ordering")
            return self._order_by_sequential_similarity(tracks, seeds)

        # Build transition distance matrix for full candidate pool
        full_matrix = self._build_transition_matrix(tracks)

        # Intelligent candidate selection: Select best-connected subset before TSP
        if target_count and len(tracks) > target_count:
            selected_tracks, matrix_to_use = self._select_best_connected_tracks(
                tracks,
                full_matrix,
                target_count,
                seeds,
                verbose=verbose
            )
            tracks_to_optimize = selected_tracks
        else:
            # No selection needed - pool is already at/below target
            tracks_to_optimize = tracks
            matrix_to_use = full_matrix

        distance_matrix = matrix_to_use

        # Choose TSP algorithm based on problem size
        if len(tracks_to_optimize) <= 15:
            # Exact solution for small playlists
            logger.info("Using exact TSP solver (dynamic programming)")
            try:
                permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
            except Exception as e:
                logger.warning(f"Exact TSP failed: {e}, falling back to heuristic")
                permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
        else:
            # Heuristic for larger playlists
            logger.info("Using heuristic TSP solver (simulated annealing)")
            permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

        # Reorder selected tracks according to TSP solution
        optimized_tracks = [tracks_to_optimize[i] for i in permutation]

        # Calculate average transition quality
        avg_distance = distance / (len(tracks_to_optimize) - 1) if len(tracks_to_optimize) > 1 else 0
        avg_similarity = 1.0 - avg_distance

        elapsed = time.time() - start_time
        logger.info(f"TSP optimization complete in {elapsed:.2f}s")
        logger.info(f"Average transition similarity: {avg_similarity:.3f}")

        # Verbose: Log individual transitions
        if verbose and len(optimized_tracks) > 1:
            logger.info("=" * 80)
            logger.info("TRANSITION QUALITY BREAKDOWN")
            logger.info("=" * 80)
            for i in range(len(optimized_tracks) - 1):
                from_idx = permutation[i]
                to_idx = permutation[i + 1]
                transition_distance = distance_matrix[from_idx][to_idx]
                transition_similarity = 1.0 - transition_distance

                from_track = optimized_tracks[i]
                to_track = optimized_tracks[i + 1]

                logger.info(f"Track {i+1:2d} → {i+2:2d}: {transition_similarity:.3f}  |  "
                           f"{from_track.get('artist')} - {from_track.get('title')[:30]}... → "
                           f"{to_track.get('artist')} - {to_track.get('title')[:30]}...")
            logger.info("=" * 80)

        # Enforce artist window constraint
        final_tracks = self._enforce_artist_window(optimized_tracks, window_size=self.config.artist_window_size)

        logger.info(f"Final playlist: {len(final_tracks)} tracks after constraint enforcement")

        return final_tracks

    def _limit_artist_frequency_in_window(self, tracks: List[Dict[str, Any]], window_size: int = None, max_per_window: int = None) -> List[Dict[str, Any]]:
        """
        Ensure no artist appears more than max_per_window times within any window_size span

        This version preserves the interleaving of sonic/genre tracks by considering
        source when there are multiple valid candidates.

        Args:
            tracks: Ordered tracks
            window_size: Size of the sliding window (uses config if None)
            max_per_window: Maximum appearances per artist in window (uses config if None)

        Returns:
            Tracks with artist frequency constraint enforced
        """
        if window_size is None:
            window_size = self.config.artist_window_size
        if max_per_window is None:
            max_per_window = self.config.max_artist_per_window

        if len(tracks) <= window_size:
            return tracks

        from collections import Counter

        # Start with empty result and process all tracks (including first window)
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
            artist_counts = Counter([t.get('artist') for t in window])

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
                    artist = track.get('artist')
                    source = track.get('source')
                    # Check if adding this track would violate the constraint
                    if artist_counts.get(artist, 0) < max_per_window and source == prefer_source:
                        best_candidate_idx = i
                        next_track = track
                        break

            # Second pass: if no preferred source found, take first valid track
            if not next_track:
                for i, track in enumerate(remaining):
                    artist = track.get('artist')
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

    def _get_diverse_seed_pool(self, history: List[Dict[str, Any]],
                               pool_size: int) -> List[Dict[str, Any]]:
        """Get a larger pool of diverse seed tracks with genre clustering"""
        play_counts = Counter()
        track_metadata = {}

        for track in history:
            key = track.get('rating_key')
            if key:
                play_counts[key] += 1
                track_metadata[key] = track

        # Get diverse seeds
        seeds = self._select_diverse_seeds(play_counts, track_metadata, pool_size)

        # Fetch genres for each seed artist if Last.FM client available
        if self.lastfm:
            logger.info("Fetching genre tags for seed artists...")
            for seed in seeds:
                artist = seed.get('artist')
                if artist:
                    genres = self.lastfm.get_artist_tags(artist)
                    seed['genres'] = genres
                    if genres:
                        logger.info(f"  {artist}: {', '.join(genres)}")
                    else:
                        logger.warning(f"  {artist}: NO GENRES FOUND")

        return seeds

    def _get_similar_artists(self, artist_name: str) -> List[str]:
        """
        Get similar artists from Last.FM (with persistent disk caching)

        Args:
            artist_name: Artist name to find similar artists for

        Returns:
            List of similar artist names
        """
        if not self.lastfm:
            return []

        # Check persistent cache first
        cached_result = self.artist_cache.get_similar_artists(artist_name)
        if cached_result is not None:
            return cached_result

        # Cache miss - query Last.FM for similar artists
        similar_artists_data = self.lastfm.get_similar_artists(artist_name, limit=self.config.limit_similar_artists)
        similar_artist_names = [a['name'] for a in similar_artists_data]

        # Save to persistent cache
        self.artist_cache.set_similar_artists(artist_name, similar_artist_names)

        if similar_artist_names:
            logger.info(f"    Last.FM similar artists for '{artist_name}': {', '.join(similar_artist_names[:5])}...")
        else:
            logger.warning(f"    No similar artists found for '{artist_name}'")

        # Rate limiting (only for API calls, not cache hits)
        time.sleep(0.2)

        return similar_artist_names

    def _get_similar_genres(self, genre: str) -> List[str]:
        """
        Get similar genres from Last.FM (with caching)

        Args:
            genre: Genre name to find similar genres for

        Returns:
            List of similar genre names
        """
        if not self.lastfm:
            return []

        if genre in self.genre_similarity_cache:
            return self.genre_similarity_cache[genre]

        # Query Last.FM for similar tags
        similar_tags = self.lastfm.get_similar_tags(genre, limit=10)
        similar_genre_names = [tag['name'] for tag in similar_tags]

        # Cache the result
        self.genre_similarity_cache[genre] = similar_genre_names

        if similar_genre_names:
            logger.debug(f"    Last.FM: '{genre}' is similar to: {', '.join(similar_genre_names[:5])}")

        # Rate limiting
        time.sleep(0.2)

        return similar_genre_names

    def _calculate_artist_similarity(self, seed1: Dict[str, Any],
                                     seed2: Dict[str, Any]) -> float:
        """
        Calculate artist similarity score using Last.FM artist.getSimilar + genre overlap

        Args:
            seed1: First seed track with artist and genres
            seed2: Second seed track with artist and genres

        Returns:
            Similarity score (0.0 to 1.0)
        """
        artist1 = seed1.get('artist', 'Unknown')
        artist2 = seed2.get('artist', 'Unknown')

        logger.debug(f"\n  Comparing: '{artist1}' <-> '{artist2}'")

        # Strategy 1: Check if artists appear in each other's similar artist lists (HIGHEST confidence)
        if self.lastfm:
            similar_to_artist1 = self._get_similar_artists(artist1)
            similar_to_artist2 = self._get_similar_artists(artist2)

            # Normalize artist names for comparison (lowercase, strip)
            similar_to_artist1_normalized = {a.lower().strip() for a in similar_to_artist1}
            similar_to_artist2_normalized = {a.lower().strip() for a in similar_to_artist2}
            artist1_normalized = artist1.lower().strip()
            artist2_normalized = artist2.lower().strip()

            # Check if artist2 is in artist1's similar list (or vice versa)
            if artist2_normalized in similar_to_artist1_normalized:
                logger.info(f"  >> DIRECT MATCH: '{artist2}' is in '{artist1}' similar artists -> similarity = {self.config.similarity_artist_direct_match}")
                return self.config.similarity_artist_direct_match
            elif artist1_normalized in similar_to_artist2_normalized:
                logger.info(f"  >> DIRECT MATCH: '{artist1}' is in '{artist2}' similar artists -> similarity = {self.config.similarity_artist_direct_match}")
                return self.config.similarity_artist_direct_match

            # Strategy 2: Check if they share common similar artists (MEDIUM confidence)
            common_similar = similar_to_artist1_normalized & similar_to_artist2_normalized
            if common_similar:
                # More common artists = higher similarity
                similarity = min(
                    self.config.similarity_artist_shared_max,
                    self.config.similarity_artist_shared_base + (len(common_similar) * self.config.similarity_artist_shared_increment)
                )
                logger.info(f"  -- SHARED SIMILAR ARTISTS: {len(common_similar)} in common {list(common_similar)[:3]} -> similarity = {similarity:.2f}")
                return similarity

        # Strategy 3: Fall back to genre overlap (LOW-MEDIUM confidence)
        genres1 = set(seed1.get('genres', []))
        genres2 = set(seed2.get('genres', []))

        if genres1 and genres2:
            direct_overlap = len(genres1 & genres2)
            if direct_overlap > 0:
                # Jaccard similarity for direct overlap
                jaccard = direct_overlap / len(genres1 | genres2)
                shared = list(genres1 & genres2)
                logger.debug(f"  -- Genre overlap: {shared} -> similarity = {jaccard:.2f}")
                return jaccard

        logger.debug(f"  XX NO SIMILARITY FOUND -> similarity = 0.0")
        return 0.0  # No similarity found

    def _create_diverse_playlists(self, all_seeds: List[Dict[str, Any]],
                                  history: List[Dict[str, Any]],
                                  count: int) -> List[List[Dict[str, Any]]]:
        """
        Create distinct playlists using genre-clustered seed subsets

        Args:
            all_seeds: Large pool of seed tracks (with genres)
            history: Play history for filtering
            count: Number of playlists to create

        Returns:
            List of distinct playlists
        """
        # Cluster seeds by genre
        genre_clusters = self._cluster_seeds_by_genre(all_seeds, count)

        playlists = []

        for i, cluster_seeds in enumerate(genre_clusters, 1):
            if not cluster_seeds:
                logger.warning(f"No seeds in cluster {i}")
                continue

            # Log genre info for this cluster
            cluster_genres = set()
            for seed in cluster_seeds:
                cluster_genres.update(seed.get('genres', [])[:2])

            logger.info(f"Playlist {i} cluster genres: {', '.join(list(cluster_genres)[:5])}")
            logger.info(f"Playlist {i} seed artists: {[s['artist'] for s in cluster_seeds]}")

            # Generate similar tracks for this seed subset
            similar_tracks = self.generate_similar_tracks(cluster_seeds)

            # Add seed tracks to the candidate pool (they're guaranteed to be in playlist)
            all_tracks = cluster_seeds + similar_tracks

            # Filter out recently played (seeds are exempt)
            fresh_tracks = self.filter_tracks(all_tracks, history, exempt_tracks=cluster_seeds)

            # Diversify
            diverse_tracks = self.diversify_tracks(fresh_tracks)

            if diverse_tracks:
                playlists.append(diverse_tracks)
                logger.info(f"Playlist {i}: {len(diverse_tracks)} tracks from {len(set(t.get('artist') for t in diverse_tracks))} artists")
            else:
                logger.warning(f"No tracks for playlist {i}")

        return playlists

    def _cluster_seeds_by_genre(self, seeds: List[Dict[str, Any]],
                                num_clusters: int) -> List[List[Dict[str, Any]]]:
        """
        Cluster seeds by genre similarity using Last.FM similarity data

        Args:
            seeds: List of seed tracks with genre tags
            num_clusters: Number of clusters (playlists) to create

        Returns:
            List of seed lists (one per cluster)
        """
        logger.info(f"Clustering {len(seeds)} seeds into {num_clusters} genre-similar groups...")

        seed_count_per_playlist = self.config.get('playlists', 'seed_count', default=5)

        # If we don't have enough seeds, just split evenly
        if len(seeds) < num_clusters * 2:
            logger.warning("Not enough seeds for proper clustering, using simple split")
            return self._simple_split_seeds(seeds, num_clusters, seed_count_per_playlist)

        # Build similarity matrix between all seeds
        logger.info("="*60)
        logger.info("Building artist similarity matrix using Last.FM...")
        logger.info("="*60)
        similarity_matrix = {}
        for i, seed1 in enumerate(seeds):
            for j, seed2 in enumerate(seeds):
                if i >= j:
                    continue
                similarity = self._calculate_artist_similarity(seed1, seed2)
                similarity_matrix[(i, j)] = similarity

        logger.info(f"\nSimilarity matrix complete - {len(similarity_matrix)} pairs calculated")
        logger.info("="*60)

        # Greedy clustering: Start with most similar seeds
        clusters = []
        assigned = set()

        # Sort seed pairs by similarity (highest first)
        sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)

        logger.info("Starting greedy clustering (similarity threshold = 0.2):")
        logger.info(f"Top 10 most similar pairs:")
        for (i, j), sim in sorted_pairs[:10]:
            logger.info(f"  {seeds[i]['artist']} <-> {seeds[j]['artist']}: {sim:.2f}")

        # Build initial clusters from highly similar seeds
        for (i, j), similarity in sorted_pairs:
            if i in assigned or j in assigned:
                continue

            if similarity >= 0.2:  # Threshold for considering seeds similar
                # Find which cluster to add to, or create new one
                cluster_found = False
                for cluster_idx, cluster in enumerate(clusters):
                    if len(cluster) < seed_count_per_playlist:
                        # Check if both seeds are similar to existing cluster members
                        cluster_compatible = True
                        for existing_idx in cluster:
                            # Check compatibility with cluster
                            sim_i = similarity_matrix.get((min(i, existing_idx), max(i, existing_idx)), 0.0)
                            sim_j = similarity_matrix.get((min(j, existing_idx), max(j, existing_idx)), 0.0)
                            if sim_i < 0.2 or sim_j < 0.2:
                                cluster_compatible = False
                                break

                        if cluster_compatible:
                            logger.info(f"Adding pair to cluster {cluster_idx+1}: {seeds[i]['artist']} + {seeds[j]['artist']} (sim={similarity:.2f})")
                            cluster.append(i)
                            cluster.append(j)
                            assigned.add(i)
                            assigned.add(j)
                            cluster_found = True
                            break

                if not cluster_found and len(clusters) < num_clusters:
                    # Create new cluster
                    logger.info(f"Creating NEW cluster {len(clusters)+1} with: {seeds[i]['artist']} + {seeds[j]['artist']} (sim={similarity:.2f})")
                    clusters.append([i, j])
                    assigned.add(i)
                    assigned.add(j)

        # Fill remaining slots in clusters with compatible seeds
        logger.info("Filling remaining cluster slots with best-fit seeds:")
        for cluster_idx, cluster in enumerate(clusters):
            while len(cluster) < seed_count_per_playlist and len(assigned) < len(seeds):
                best_candidate = None
                best_avg_similarity = 0.0

                for idx in range(len(seeds)):
                    if idx in assigned:
                        continue

                    # Calculate average similarity to cluster members
                    similarities = []
                    for cluster_member_idx in cluster:
                        sim = similarity_matrix.get((min(idx, cluster_member_idx), max(idx, cluster_member_idx)), 0.0)
                        similarities.append(sim)

                    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                    if avg_sim > best_avg_similarity:
                        best_avg_similarity = avg_sim
                        best_candidate = idx

                if best_candidate is not None:
                    cluster_members = [seeds[i]['artist'] for i in cluster]
                    logger.info(f"Adding to cluster {cluster_idx+1}: {seeds[best_candidate]['artist']} (avg_sim={best_avg_similarity:.2f} to {cluster_members[:2]}...)")
                    cluster.append(best_candidate)
                    assigned.add(best_candidate)
                else:
                    break

        # Create any remaining clusters with unassigned seeds
        unassigned = [i for i in range(len(seeds)) if i not in assigned]
        while unassigned and len(clusters) < num_clusters:
            new_cluster = []
            for _ in range(min(seed_count_per_playlist, len(unassigned))):
                if unassigned:
                    new_cluster.append(unassigned.pop(0))
            if new_cluster:
                clusters.append(new_cluster)

        # Convert indices back to seed objects
        seed_clusters = [[seeds[idx] for idx in cluster] for cluster in clusters]

        # Log cluster composition
        for i, cluster in enumerate(seed_clusters, 1):
            if cluster:
                cluster_genres = Counter()
                for seed in cluster:
                    for genre in seed.get('genres', ['unknown'])[:3]:
                        cluster_genres[genre] += 1
                logger.info(f"Cluster {i} genres: {dict(cluster_genres.most_common(5))}")
                logger.debug(f"Cluster {i} artists: {[s.get('artist') for s in cluster]}")

        return seed_clusters

    def _simple_split_seeds(self, seeds: List[Dict[str, Any]], num_clusters: int,
                           seeds_per_cluster: int) -> List[List[Dict[str, Any]]]:
        """Fallback: Simple split when not enough seeds for clustering"""
        clusters = []
        for i in range(0, len(seeds), seeds_per_cluster):
            cluster = seeds[i:i + seeds_per_cluster]
            if cluster:
                clusters.append(cluster)
            if len(clusters) >= num_clusters:
                break
        return clusters

    def _get_lastfm_history(self) -> List[Dict[str, Any]]:
        """
        Get listening history from Last.FM and match to library

        Returns:
            List of matched tracks with play counts
        """
        logger.info("Fetching Last.FM listening history")

        # Get history from Last.FM
        history_days = self.config.lastfm_history_days
        lastfm_tracks = self.lastfm.get_recent_tracks(days=history_days)

        if not lastfm_tracks:
            logger.warning("No Last.FM history found")
            return []

        logger.info(f"Retrieved {len(lastfm_tracks)} tracks from Last.FM")

        # Match to library
        matched_tracks = self.matcher.match_lastfm_to_library(lastfm_tracks)

        if not matched_tracks:
            logger.warning("No tracks could be matched to library")
            return []

        # Aggregate play counts
        aggregated = self.matcher.aggregate_play_counts(matched_tracks)

        return aggregated

    def _get_local_history(self) -> List[Dict[str, Any]]:
        """
        Get listening history from local library (fallback method)

        Returns:
            List of tracks from play history
        """
        logger.info("Fetching local listening history (Last.FM not available)")

        history_days = self.config.get('playlists', 'history_days', default=14)
        history = self.library.get_play_history(None, days=history_days)

        return history
    
    def _split_into_playlists(self, tracks: List[Dict[str, Any]], 
                             count: int) -> List[List[Dict[str, Any]]]:
        """
        Split tracks into multiple playlists
        
        Args:
            tracks: Available tracks
            count: Number of playlists to create
            
        Returns:
            List of track lists
        """
        tracks_per_playlist = self.config.get('playlists', 'tracks_per_playlist', default=30)
        
        # Sort by weight (higher weight = more likely based on listening)
        sorted_tracks = sorted(tracks, key=lambda x: x.get('weight', 0), reverse=True)
        
        playlists = []
        
        for i in range(count):
            # Create playlist with different strategies
            if i == 0:
                # First playlist: Top weighted tracks
                playlist = sorted_tracks[:tracks_per_playlist]
            elif i == 1:
                # Second playlist: More diverse selection
                playlist = self._select_diverse(sorted_tracks, tracks_per_playlist)
            else:
                # Additional playlists: Random sampling
                population = sorted_tracks[:len(sorted_tracks)//2]
                sample_size = min(tracks_per_playlist, len(population))
                playlist = random.sample(population, sample_size) if sample_size > 0 else []
            
            if playlist:
                playlists.append(playlist)
                logger.info(f"Playlist {i+1}: {len(playlist)} tracks")
        
        return playlists
    
    def _select_diverse(self, tracks: List[Dict[str, Any]], 
                       count: int) -> List[Dict[str, Any]]:
        """
        Select tracks ensuring artist diversity
        
        Args:
            tracks: Available tracks
            count: Number to select
            
        Returns:
            Diverse selection of tracks
        """
        selected = []
        used_artists = set()
        
        # First pass: one track per artist
        for track in tracks:
            artist = track.get('artist')
            if artist not in used_artists:
                selected.append(track)
                used_artists.add(artist)
                
                if len(selected) >= count:
                    break
        
        # Second pass: fill remaining slots
        if len(selected) < count:
            remaining = [t for t in tracks if t not in selected]
            needed = count - len(selected)
            selected.extend(remaining[:needed])
        
        return selected

    def _print_playlist_report(self, tracks: List[Dict[str, Any]], artist_name: str = None, dynamic: bool = False):
        """
        Print detailed track report showing how each track was selected

        Args:
            tracks: Final playlist tracks
            artist_name: Name of seed artist (if applicable)
            dynamic: Whether dynamic mode was used
        """
        logger.info("=" * 80)
        logger.info("PLAYLIST TRACKLIST")
        logger.info("=" * 80)

        # Group tracks by source for statistics
        source_counts = {'sonic': 0, 'genre': 0, 'unknown': 0}

        for i, track in enumerate(tracks, 1):
            artist = sanitize_for_logging(track.get('artist', 'Unknown'))
            title = sanitize_for_logging(track.get('title', 'Unknown'))
            source = track.get('source', 'unknown')

            # Count by source
            source_counts[source] = source_counts.get(source, 0) + 1

            # Format: Track 01: Artist - Title
            logger.info(f"Track {i:02d}: {artist} - {title}")

        # Print summary statistics
        logger.info("=" * 80)
        logger.info("PLAYLIST STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total tracks: {len(tracks)}")
        logger.info(f"Unique artists: {len(set(t.get('artist') for t in tracks))}")

        if dynamic:
            logger.info(f"  - Sonic similarity: {source_counts.get('sonic', 0)} tracks")
            logger.info(f"  - Genre-based: {source_counts.get('genre', 0)} tracks")
        else:
            logger.info(f"  - Sonic similarity: {source_counts.get('sonic', 0)} tracks")

        if artist_name:
            seed_count = len([t for t in tracks if safe_get_artist(t) == artist_name.lower()])
            seed_pct = (seed_count / len(tracks)) * 100 if tracks else 0
            logger.info(f"  - Seed artist ({sanitize_for_logging(artist_name)}): {seed_count} tracks ({seed_pct:.1f}%)")

        # Duration info
        total_duration_ms = sum((track.get('duration') or 0) for track in tracks)
        total_minutes = total_duration_ms / 1000 / 60
        logger.info(f"Total duration: {total_minutes:.1f} minutes")
        logger.info("=" * 80)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Playlist Generator module loaded")
