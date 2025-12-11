"""
Last.FM API Client - Fetches listening history and user data
"""
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import time
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

    @retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions=(requests.exceptions.RequestException,))
    def _make_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to Last.FM API

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

        self.rate_limiter.wait()  # Enforce rate limit
        try:
            response = self.session.get(self.BASE_URL, params=request_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Last.FM API request failed: {e}")
            return None

    def get_recent_tracks(self, days: int = 90, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Get recent listening history (with parallel page fetching)

        Args:
            days: Number of days to look back
            limit: Maximum tracks per page (Last.FM max is 200)

        Returns:
            List of track dictionaries with artist, title, album, timestamp
        """
        logger.info(f"Fetching Last.FM history for last {days} days")

        # Calculate timestamp for X days ago
        from_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())

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

        # Fetch remaining pages in parallel (with conservative parallelism)
        remaining_pages = list(range(2, total_pages + 1))
        failed_pages = []

        # Use only 2 workers to avoid overwhelming Last.FM API
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all page fetch tasks
            future_to_page = {
                executor.submit(self._fetch_page, from_timestamp, limit, page): page
                for page in remaining_pages
            }

            # Process completed requests
            completed = 0
            for future in as_completed(future_to_page):
                page = future_to_page[future]
                completed += 1
                try:
                    page_tracks = future.result()
                    if page_tracks:
                        all_tracks.extend(page_tracks)
                        if completed % 50 == 0:  # Log progress every 50 pages
                            logger.info(f"Progress: {completed}/{total_pages} pages fetched")
                    else:
                        # Empty result, might be a failed request
                        failed_pages.append(page)
                except Exception as e:
                    logger.warning(f"Failed to fetch page {page}, will retry later: {e}")
                    failed_pages.append(page)

        # Retry failed pages sequentially with delays
        if failed_pages:
            logger.info(f"Retrying {len(failed_pages)} failed pages sequentially...")
            for i, page in enumerate(failed_pages, 1):
                try:
                    time.sleep(2)  # 2 second delay between retries
                    page_tracks = self._fetch_page(from_timestamp, limit, page)
                    if page_tracks:
                        all_tracks.extend(page_tracks)
                        logger.info(f"Retry {i}/{len(failed_pages)}: Successfully fetched page {page}")
                    else:
                        logger.warning(f"Retry {i}/{len(failed_pages)}: Page {page} returned no data")
                except Exception as e:
                    logger.error(f"Retry {i}/{len(failed_pages)}: Failed to fetch page {page}: {e}")

        # Sort by timestamp (newest first)
        all_tracks.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        logger.info(f"Fetched {len(all_tracks)} total tracks from Last.FM")
        return all_tracks

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
    logging.basicConfig(level=logging.INFO)
    logger.info("Last.FM Client module loaded")
