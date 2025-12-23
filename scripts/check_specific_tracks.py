"""
Check if specific tracks are being matched by recency filter
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config
from src.lastfm_client import LastFMClient
from src.local_library_client import LocalLibraryClient
from src.string_utils import normalize_match_string
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def _key_for_track(track):
    """Same key logic as in playlist_generator.py (FIXED VERSION)"""
    # Always use artist::title for matching (ignore mbid)
    artist = track.get("artist")
    title = track.get("title")
    if not artist or not title:
        return None
    return f"{normalize_match_string(artist, is_artist=True)}::{normalize_match_string(title)}"

def main():
    # Tracks to check
    check_tracks = [
        {"artist": "Jessica Pratt", "title": "Crossing"},
        {"artist": "Cameron Winter", "title": "$0"},
        {"artist": "Cameron Winter", "title": "Drinking Age"},
    ]

    config = Config()
    api_key = config.get('lastfm', 'api_key')
    username = config.get('lastfm', 'username')
    db_path = config.get('library', 'database_path', default='data/metadata.db')

    lastfm = LastFMClient(api_key, username)
    library = LocalLibraryClient(db_path)

    # Get scrobbles
    history_days = config.get('lastfm', 'history_days', default=90)
    lookback_days = config.recently_played_lookback_days

    logger.info("Fetching Last.fm scrobbles...")
    scrobbles = lastfm.get_recent_tracks(days=history_days, use_cache=True)

    cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())

    logger.info(f"\nRecent scrobbles (last {lookback_days} days):")
    logger.info("=" * 80)

    # Check each track
    for check_track in check_tracks:
        artist = check_track["artist"]
        title = check_track["title"]

        logger.info(f"\nChecking: {artist} - {title}")
        logger.info("-" * 80)

        # Check in Last.fm scrobbles
        matching_scrobbles = []
        for s in scrobbles:
            if s.get("artist", "").lower() == artist.lower() and s.get("title", "").lower() == title.lower():
                ts = s.get("timestamp", 0)
                if ts >= cutoff_timestamp:
                    matching_scrobbles.append(s)

        if matching_scrobbles:
            logger.info(f"✓ Found in Last.fm (recent scrobbles):")
            for s in matching_scrobbles:
                play_date = datetime.fromtimestamp(s.get("timestamp", 0))
                key = _key_for_track(s)
                logger.info(f"    Played: {play_date}")
                logger.info(f"    Last.fm key: {key}")
        else:
            logger.info(f"✗ NOT found in Last.fm recent scrobbles")

        # Check in library
        all_tracks = library.get_all_tracks()
        matching_library = []
        for t in all_tracks:
            if t.get("artist", "").lower() == artist.lower() and t.get("title", "").lower() == title.lower():
                matching_library.append(t)

        if matching_library:
            logger.info(f"✓ Found in library:")
            for t in matching_library:
                key = _key_for_track(t)
                logger.info(f"    track_id: {t.get('rating_key')}")
                logger.info(f"    Library key: {key}")
        else:
            logger.info(f"✗ NOT found in library")

        # Check if keys match
        if matching_scrobbles and matching_library:
            scrobble_key = _key_for_track(matching_scrobbles[0])
            library_key = _key_for_track(matching_library[0])

            if scrobble_key == library_key:
                logger.info(f"✓✓ KEYS MATCH - This track SHOULD be filtered")
            else:
                logger.info(f"✗✗ KEYS DON'T MATCH - This track will NOT be filtered!")
                logger.info(f"    Last.fm key: {scrobble_key}")
                logger.info(f"    Library key: {library_key}")

if __name__ == "__main__":
    main()
