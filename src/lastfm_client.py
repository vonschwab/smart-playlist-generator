"""
Last.FM API Client - Fetches listening history and user data
"""
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import time
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .rate_limiter import RateLimiter
from .retry_helper import retry_with_backoff

logger = logging.getLogger(__name__)


class LastFMClient:
    """Client for interacting with Last.FM API"""

    BASE_URL = "http://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str, username: str):
        """
        Initialize Last.FM client

        Args:
            api_key: Last.FM API key
            username: Last.FM username
        """
        self.api_key = api_key
        self.username = username
        self.session = requests.Session()

        # Rate limiter for Last.FM API (5 requests per second max per Last.FM ToS)
        self.rate_limiter = RateLimiter(calls_per_second=5.0)

        logger.info(f"Initialized Last.FM client for user: {username}")

    def _make_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to Last.FM API with intelligent retry logic

        Args:
            method: API method name
            params: Additional parameters

        Returns:
            JSON response or None on error
        """
        request_params = {
            'method': method,
            'api_key': self.api_key,
            'format': 'json',
            'user': self.username,
            **params
        }

        # Custom retry logic with longer delays for 5xx errors
        max_retries = 5
        initial_delay = 1.0

        for attempt in range(max_retries):
            self.rate_limiter.wait()  # Enforce rate limit
            try:
                response = self.session.get(self.BASE_URL, params=request_params, timeout=10)

                # Special handling for 500+ errors
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                        logger.warning(
                            f"Last.FM returned {response.status_code} (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay:.1f}s for page {params.get('page', 'N/A')}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Last.FM API returned {response.status_code} after {max_retries} retries (page {params.get('page', 'N/A')})")
                        return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Last.FM request timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Last.FM request timeout after {max_retries} retries")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Last.FM API request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Last.FM API request failed after {max_retries} retries: {e}")
                    return None

        return None

    def _load_cache(self, days: int) -> Optional[List[Dict[str, Any]]]:
        """Load cached recent tracks if within the requested window."""
        try:
            conn = sqlite3.connect("data/metadata.db")
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS lastfm_cache (
                    username TEXT PRIMARY KEY,
                    fetched_at INTEGER,
                    days INTEGER,
                    payload TEXT
                )
                """
            )
            cur.execute(
                "SELECT fetched_at, days, payload FROM lastfm_cache WHERE username=?",
                (self.username,),
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            fetched_at, cached_days, payload = row
            # Use cache if it covers at least the requested window and is not older than that window
            age_seconds = time.time() - fetched_at
            if cached_days >= days and age_seconds <= days * 86400:
                logger.info("Last.FM cache hit for %s (age=%.1f days, cached_days=%d)", self.username, age_seconds/86400, cached_days)
                return json.loads(payload)
            return None
        except Exception:
            return None

    def _save_cache(self, days: int, tracks: List[Dict[str, Any]]) -> None:
        """Persist recent tracks cache."""
        try:
            conn = sqlite3.connect("data/metadata.db")
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS lastfm_cache (
                    username TEXT PRIMARY KEY,
                    fetched_at INTEGER,
                    days INTEGER,
                    payload TEXT
                )
                """
            )
            cur.execute(
                "REPLACE INTO lastfm_cache (username, fetched_at, days, payload) VALUES (?, ?, ?, ?)",
                (self.username, int(time.time()), days, json.dumps(tracks)),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.debug("Failed to save Last.FM cache", exc_info=True)

    def get_recent_tracks(self, days: int = 90, limit: int = 200, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get recent listening history (with parallel page fetching)

        Args:
            days: Number of days to look back
            limit: Maximum tracks per page (Last.FM max is 200)

        Returns:
            List of track dictionaries with artist, title, album, timestamp
        """
        def _dedupe_and_sort(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen = set()
            unique = []
            for t in tracks:
                key = (
                    (t.get('artist') or '').strip().lower(),
                    (t.get('title') or '').strip().lower(),
                    (t.get('album') or '').strip().lower(),
                    t.get('timestamp') or 0,
                )
                if key in seen:
                    continue
                seen.add(key)
                unique.append(t)
            unique.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            return unique

        def _fetch(from_timestamp: int) -> List[Dict[str, Any]]:
            logger.info("Fetching Last.FM history since %s (days=%d)", datetime.utcfromtimestamp(from_timestamp).isoformat(), days)

            # Fetch first page to get total page count
            logger.debug("Fetching page 1/1")
            first_page_data = self._make_request('user.getrecenttracks', {
                'from': from_timestamp,
                'limit': limit,
                'page': 1,
                'extended': 1
            })

            if not first_page_data or 'recenttracks' not in first_page_data:
                logger.warning("No recent tracks data returned")
                return []

            tracks_data = first_page_data['recenttracks']
            total_pages = int(tracks_data.get('@attr', {}).get('totalPages', 1))
            logger.info(f"Total pages to fetch: {total_pages}")

            # Parse first page
            all_tracks = []
            tracks = tracks_data.get('track', [])
            if not isinstance(tracks, list):
                tracks = [tracks]

            for track in tracks:
                if '@attr' in track and track['@attr'].get('nowplaying') == 'true':
                    continue
                parsed_track = self._parse_track(track)
                if parsed_track:
                    all_tracks.append(parsed_track)

            # If only one page, return early
            if total_pages == 1:
                logger.info(f"Fetched {len(all_tracks)} tracks from Last.FM")
                return all_tracks

            # Fetch remaining pages sequentially (1 worker instead of parallel)
            # This reduces rate limiting issues and is more reliable
            remaining_pages = list(range(2, total_pages + 1))
            failed_pages = []

            logger.info(f"Fetching {len(remaining_pages)} remaining pages sequentially...")

            for idx, page in enumerate(remaining_pages, 1):
                try:
                    page_tracks = self._fetch_page(from_timestamp, limit, page)
                    if page_tracks:
                        all_tracks.extend(page_tracks)
                        if idx % 5 == 0 or idx == len(remaining_pages):  # Log progress every 5 pages
                            logger.info(f"Progress: {idx + 1}/{total_pages} pages fetched ({len(all_tracks)} tracks so far)")
                    else:
                        # Empty result, might be a failed request
                        logger.warning(f"Page {page} returned no data (will retry later)")
                        failed_pages.append(page)
                except Exception as e:
                    logger.warning(f"Failed to fetch page {page}: {e} (will retry later)")
                    failed_pages.append(page)

            # Retry failed pages with longer delays
            if failed_pages:
                logger.info(f"Retrying {len(failed_pages)} failed pages with extended delays...")
                successful_retries = 0
                for i, page in enumerate(failed_pages, 1):
                    try:
                        # Longer delay for retries (5-10 seconds between attempts)
                        time.sleep(5 + (i % 5))  # Variable delay: 5-10 seconds
                        page_tracks = self._fetch_page(from_timestamp, limit, page)
                        if page_tracks:
                            all_tracks.extend(page_tracks)
                            successful_retries += 1
                            logger.info(f"Retry {i}/{len(failed_pages)}: Successfully fetched page {page} ({len(page_tracks)} tracks)")
                        else:
                            logger.warning(f"Retry {i}/{len(failed_pages)}: Page {page} still returned no data (skipping)")
                    except Exception as e:
                        logger.warning(f"Retry {i}/{len(failed_pages)}: Failed to fetch page {page}: {e} (skipping)")

                logger.info(f"Retry complete: {successful_retries}/{len(failed_pages)} pages recovered")

            # Sort by timestamp (newest first)
            all_tracks.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

            # Log completion with statistics
            expected_min = (total_pages - 1) * limit  # Approximate expected minimum
            logger.info(f"Fetched {len(all_tracks)} total tracks from Last.FM ({len(all_tracks) / max(expected_min, 1) * 100:.1f}% of expected)")
            return all_tracks

        # Try cache first
        cached_tracks = None
        if use_cache:
            cached_tracks = self._load_cache(days)
            if cached_tracks is not None:
                logger.info(
                    "Last.FM cache hit for %s (days=%d, tracks=%d); checking for new scrobbles",
                    self.username,
                    days,
                    len(cached_tracks),
                )

        if cached_tracks:
            latest_ts = max((t.get('timestamp') or 0) for t in cached_tracks)
            if latest_ts <= 0:
                cached_tracks = None  # fallback to full fetch
            else:
                # Fetch only new scrobbles since latest cached, with 1-hour overlap
                delta_from = max(0, latest_ts - 3600)
                new_tracks = _fetch(delta_from)
                if new_tracks:
                    merged = _dedupe_and_sort(new_tracks + cached_tracks)
                    if use_cache:
                        self._save_cache(days, merged)
                        logger.info(
                            "Saved merged Last.FM history to cache for %s (days=%d, tracks=%d)",
                            self.username,
                            days,
                            len(merged),
                        )
                    return merged
                # No new tracks; return cached
                logger.info("No new Last.FM scrobbles found; using cached history")
                return _dedupe_and_sort(cached_tracks)

        # No usable cache; fetch full window
        window_start = int((datetime.now() - timedelta(days=days)).timestamp())
        fresh_tracks = _fetch(window_start)
        if use_cache:
            self._save_cache(days, fresh_tracks)
            logger.info(
                "Saved Last.FM history to cache for %s (days=%d, tracks=%d)",
                self.username,
                days,
                len(fresh_tracks),
            )
        return fresh_tracks

    def _fetch_page(self, from_timestamp: int, limit: int, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a single page of recent tracks (for parallel execution)

        Args:
            from_timestamp: Unix timestamp to fetch from
            limit: Tracks per page
            page: Page number to fetch

        Returns:
            List of parsed tracks from this page
        """
        data = self._make_request('user.getrecenttracks', {
            'from': from_timestamp,
            'limit': limit,
            'page': page,
            'extended': 1
        })

        if not data or 'recenttracks' not in data:
            return []

        tracks_data = data['recenttracks']
        tracks = tracks_data.get('track', [])
        if not isinstance(tracks, list):
            tracks = [tracks]

        parsed_tracks = []
        for track in tracks:
            if '@attr' in track and track['@attr'].get('nowplaying') == 'true':
                continue
            parsed_track = self._parse_track(track)
            if parsed_track:
                parsed_tracks.append(parsed_track)

        return parsed_tracks

    def _parse_track(self, track_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Parse track data from Last.FM API response

        Args:
            track_data: Raw track data from API

        Returns:
            Normalized track dictionary
        """
        try:
            # Extract artist name (can be string or dict)
            artist = track_data.get('artist', {})
            if isinstance(artist, dict):
                artist_name = artist.get('#text', artist.get('name', 'Unknown'))
            else:
                artist_name = str(artist)

            # Extract album name
            album = track_data.get('album', {})
            if isinstance(album, dict):
                album_name = album.get('#text', '')
            else:
                album_name = str(album) if album else ''

            # Get timestamp
            date_info = track_data.get('date', {})
            if isinstance(date_info, dict):
                timestamp = int(date_info.get('uts', 0))
            else:
                timestamp = 0

            return {
                'artist': artist_name.strip(),
                'title': track_data.get('name', '').strip(),
                'album': album_name.strip(),
                'timestamp': timestamp,
                'mbid': track_data.get('mbid', ''),  # MusicBrainz ID
                'url': track_data.get('url', '')
            }
        except Exception as e:
            logger.warning(f"Failed to parse track: {e}")
            return None

    def get_top_tracks(self, period: str = '3month', limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get user's top tracks for a time period

        Args:
            period: Time period (overall, 7day, 1month, 3month, 6month, 12month)
            limit: Number of tracks to return

        Returns:
            List of top tracks with play counts
        """
        logger.info(f"Fetching top {limit} tracks for period: {period}")

        data = self._make_request('user.gettoptracks', {
            'period': period,
            'limit': limit
        })

        if not data or 'toptracks' not in data:
            logger.warning("No top tracks data returned")
            return []

        tracks = data['toptracks'].get('track', [])
        if not isinstance(tracks, list):
            tracks = [tracks]

        top_tracks = []
        for track in tracks:
            artist = track.get('artist', {})
            artist_name = artist.get('name', 'Unknown') if isinstance(artist, dict) else str(artist)

            top_tracks.append({
                'artist': artist_name.strip(),
                'title': track.get('name', '').strip(),
                'play_count': int(track.get('playcount', 0)),
                'mbid': track.get('mbid', ''),
                'url': track.get('url', '')
            })

        logger.info(f"Retrieved {len(top_tracks)} top tracks")
        return top_tracks

    def get_top_artists(self, period: str = '3month', limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get user's top artists for a time period

        Args:
            period: Time period (overall, 7day, 1month, 3month, 6month, 12month)
            limit: Number of artists to return

        Returns:
            List of top artists with play counts
        """
        logger.info(f"Fetching top {limit} artists for period: {period}")

        data = self._make_request('user.gettopartists', {
            'period': period,
            'limit': limit
        })

        if not data or 'topartists' not in data:
            logger.warning("No top artists data returned")
            return []

        artists = data['topartists'].get('artist', [])
        if not isinstance(artists, list):
            artists = [artists]

        top_artists = []
        for artist in artists:
            top_artists.append({
                'name': artist.get('name', '').strip(),
                'play_count': int(artist.get('playcount', 0)),
                'mbid': artist.get('mbid', ''),
                'url': artist.get('url', '')
            })

        logger.info(f"Retrieved {len(top_artists)} top artists")
        return top_artists

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get user profile information

        Returns:
            User info dictionary or None
        """
        logger.info(f"Fetching user info for: {self.username}")

        data = self._make_request('user.getinfo', {})

        if not data or 'user' not in data:
            logger.warning("No user info returned")
            return None

        user = data['user']
        return {
            'username': user.get('name', ''),
            'real_name': user.get('realname', ''),
            'play_count': int(user.get('playcount', 0)),
            'registered': int(user.get('registered', {}).get('unixtime', 0)),
            'url': user.get('url', '')
        }

    def get_artist_tags(self, artist_name: str) -> List[str]:
        """
        Get genre tags for an artist

        Args:
            artist_name: Name of the artist

        Returns:
            List of genre/tag strings (e.g., ["indie rock", "post-punk"])
        """
        data = self._make_request('artist.gettoptags', {
            'artist': artist_name,
            'autocorrect': 1
        })

        if not data or 'toptags' not in data:
            return []

        tags = data['toptags'].get('tag', [])
        if not isinstance(tags, list):
            tags = [tags]

        # Extract tag names, filter out non-genre tags
        genre_tags = []
        excluded = {'seen live', 'favorite', 'favorites', 'favourites', 'albums i own',
                   'love', 'loved', 'beautiful', 'awesome', 'great', 'amazing'}

        for tag in tags[:10]:  # Top 10 tags
            tag_name = tag.get('name', '').lower().strip()
            if tag_name and tag_name not in excluded and len(tag_name) > 2:
                genre_tags.append(tag_name)

        return genre_tags[:5]  # Return top 5 genre tags

    def get_track_tags(self, artist_name: str, track_name: str) -> List[str]:
        """
        Get genre tags for a specific track

        Args:
            artist_name: Name of the artist
            track_name: Name of the track

        Returns:
            List of genre/tag strings
        """
        data = self._make_request('track.gettoptags', {
            'artist': artist_name,
            'track': track_name,
            'autocorrect': 1
        })

        if not data or 'toptags' not in data:
            return []

        tags = data['toptags'].get('tag', [])
        if not isinstance(tags, list):
            tags = [tags]

        # Extract tag names, filter out non-genre tags
        genre_tags = []
        excluded = {'seen live', 'favorite', 'favorites', 'favourites', 'albums i own',
                   'love', 'loved', 'beautiful', 'awesome', 'great', 'amazing'}

        for tag in tags[:10]:  # Top 10 tags
            tag_name = tag.get('name', '').lower().strip()
            if tag_name and tag_name not in excluded and len(tag_name) > 2:
                genre_tags.append(tag_name)

        return genre_tags[:5]  # Return top 5 genre tags

    def get_album_tags(self, artist_name: str, album_name: str) -> List[str]:
        """
        Get genre tags for an album

        Args:
            artist_name: Name of the artist
            album_name: Name of the album

        Returns:
            List of genre/tag strings
        """
        data = self._make_request('album.gettoptags', {
            'artist': artist_name,
            'album': album_name,
            'autocorrect': 1
        })

        if not data or 'toptags' not in data:
            return []

        tags = data['toptags'].get('tag', [])
        if not isinstance(tags, list):
            tags = [tags]

        # Extract tag names, filter out non-genre tags
        genre_tags = []
        excluded = {'seen live', 'favorite', 'favorites', 'favourites', 'albums i own',
                   'love', 'loved', 'beautiful', 'awesome', 'great', 'amazing'}

        for tag in tags[:10]:  # Top 10 tags
            tag_name = tag.get('name', '').lower().strip()
            if tag_name and tag_name not in excluded and len(tag_name) > 2:
                genre_tags.append(tag_name)

        return genre_tags[:5]  # Return top 5 genre tags

    def get_similar_tags(self, tag_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get similar genres/tags to a given tag

        Args:
            tag_name: Name of the genre/tag
            limit: Number of similar tags to return

        Returns:
            List of similar tag dictionaries with name and similarity score
        """
        data = self._make_request('tag.getsimilar', {
            'tag': tag_name,
            'limit': limit
        })

        if not data or 'similartags' not in data:
            return []

        tags = data['similartags'].get('tag', [])
        if not isinstance(tags, list):
            tags = [tags]

        similar_tags = []
        for tag in tags:
            # Last.FM doesn't provide explicit similarity scores in this endpoint
            # But tags are returned in order of similarity
            similar_tags.append({
                'name': tag.get('name', '').lower().strip(),
                'url': tag.get('url', '')
            })

        return similar_tags

    def get_similar_artists(self, artist_name: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get similar artists to a given artist

        Args:
            artist_name: Name of the artist
            limit: Number of similar artists to return

        Returns:
            List of similar artist dictionaries with name and match score
        """
        data = self._make_request('artist.getsimilar', {
            'artist': artist_name,
            'limit': limit
        })

        if not data or 'similarartists' not in data:
            return []

        artists = data['similarartists'].get('artist', [])
        if not isinstance(artists, list):
            artists = [artists]

        similar_artists = []
        for artist in artists:
            similar_artists.append({
                'name': artist.get('name', '').strip(),
                'match': float(artist.get('match', 0.0)),  # Similarity score from Last.FM
                'mbid': artist.get('mbid', '')
            })

        return similar_artists


# Example usage
if __name__ == "__main__":
    logger.info("Last.FM Client module loaded")
