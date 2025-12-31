"""
MusicBrainz Genre Fetcher
==========================
Fetches genre data from MusicBrainz for all artists.
MusicBrainz provides high-quality, community-curated genre tags.
"""
import requests
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MusicBrainzGenreFetcher:
    """Fetches genres from MusicBrainz"""

    def __init__(self):
        """Initialize fetcher"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PlaylistGenerator/1.0 (https://github.com/yourusername/playlist-generator)'
        })

    def fetch_musicbrainz_artist_genres(self, artist: str) -> List[str]:
        """
        Fetch artist-level genres from MusicBrainz

        Args:
            artist: Artist name

        Returns:
            List of genre tags
        """
        try:
            # Search for artist
            search_url = "https://musicbrainz.org/ws/2/artist/"
            params = {
                'query': f'artist:"{artist}"',
                'fmt': 'json',
                'limit': 1
            }

            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            time.sleep(1.1)  # MusicBrainz rate limit: 1 request per second

            data = response.json()

            if not data.get('artists'):
                logger.debug(f"MusicBrainz: No results for {artist}")
                return []

            artist_mbid = data['artists'][0]['id']

            # Get artist details including tags
            artist_url = f"https://musicbrainz.org/ws/2/artist/{artist_mbid}"
            params = {'fmt': 'json', 'inc': 'tags+genres'}

            response = self.session.get(artist_url, params=params)
            response.raise_for_status()
            time.sleep(1.1)

            artist_data = response.json()

            # Extract genres and tags
            genres = []

            # Get genres
            for genre in artist_data.get('genres', []):
                if genre.get('name'):
                    genres.append(genre['name'].lower())

            # Get tags (user-submitted)
            for tag in artist_data.get('tags', []):
                if tag.get('name') and tag.get('count', 0) >= 3:  # Only tags with 3+ votes
                    tag_name = tag['name'].lower()
                    if tag_name not in genres:
                        genres.append(tag_name)

            if genres:
                logger.info(f"MusicBrainz Artist: Found {len(genres)} genres for {artist}")

            return genres[:10]  # Limit to top 10

        except Exception as e:
            logger.debug(f"MusicBrainz error for {artist}: {e}")
            return []

    def fetch_musicbrainz_release_genres(self, artist: str, album: str) -> List[str]:
        """
        Fetch release (album) level genres from MusicBrainz

        Args:
            artist: Artist name
            album: Album name

        Returns:
            List of genre tags
        """
        try:
            # Search for release-group (album)
            search_url = "https://musicbrainz.org/ws/2/release-group/"
            params = {
                'query': f'artist:"{artist}" AND releasegroup:"{album}"',
                'fmt': 'json',
                'limit': 1
            }

            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            time.sleep(1.1)

            data = response.json()

            if not data.get('release-groups'):
                logger.debug(f"MusicBrainz: No release found for {artist} - {album}")
                return []

            release_mbid = data['release-groups'][0]['id']

            # Get release-group details including tags
            release_url = f"https://musicbrainz.org/ws/2/release-group/{release_mbid}"
            params = {'fmt': 'json', 'inc': 'tags+genres'}

            response = self.session.get(release_url, params=params)
            response.raise_for_status()
            time.sleep(1.1)

            release_data = response.json()

            # Extract genres and tags
            genres = []

            # Get genres
            for genre in release_data.get('genres', []):
                if genre.get('name'):
                    genres.append(genre['name'].lower())

            # Get tags
            for tag in release_data.get('tags', []):
                if tag.get('name') and tag.get('count', 0) >= 2:  # Lower threshold for albums
                    tag_name = tag['name'].lower()
                    if tag_name not in genres:
                        genres.append(tag_name)

            if genres:
                logger.info(f"MusicBrainz Release: Found {len(genres)} genres for {artist} - {album}")

            return genres[:10]

        except Exception as e:
            logger.debug(f"MusicBrainz release error for {artist} - {album}: {e}")
            return []

    # Legacy method for backward compatibility
    def fetch_musicbrainz_genres(self, artist: str, album: Optional[str] = None) -> List[str]:
        """
        Fetch genres from MusicBrainz (legacy method)

        Args:
            artist: Artist name
            album: Album name (optional)

        Returns:
            List of genre tags
        """
        if album:
            return self.fetch_musicbrainz_release_genres(artist, album)
        return self.fetch_musicbrainz_artist_genres(artist)

    def fetch_all_sources(self, artist: str, album: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Fetch genres from MusicBrainz

        Args:
            artist: Artist name
            album: Album name (optional)

        Returns:
            Dictionary mapping source to genre list
        """
        results = {}

        # Fetch from MusicBrainz
        mb_genres = self.fetch_musicbrainz_genres(artist, album)
        if mb_genres:
            results['musicbrainz'] = mb_genres

        return results

    def get_combined_genres(self, artist: str, album: Optional[str] = None,
                          max_genres: int = 10) -> List[str]:
        """
        Get genres from MusicBrainz

        Args:
            artist: Artist name
            album: Album name (optional)
            max_genres: Maximum number of genres to return

        Returns:
            List of genres from MusicBrainz
        """
        return self.fetch_musicbrainz_genres(artist, album)[:max_genres]


# Example usage
if __name__ == "__main__":
    fetcher = MusicBrainzGenreFetcher()

    # Test with an artist
    test_artist = "Minyo Delivery Service"
    logger.info(f"Fetching genres for: {test_artist}")
    logger.info("=" * 60)

    genres = fetcher.fetch_musicbrainz_genres(test_artist)
    logger.info(f"MUSICBRAINZ:")
    logger.info(f"  {', '.join(genres)}")
