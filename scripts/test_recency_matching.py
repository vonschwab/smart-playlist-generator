"""
Test if Last.fm scrobbles are correctly matching local library tracks
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
    config = Config()

    # Initialize clients
    api_key = config.get('lastfm', 'api_key')
    username = config.get('lastfm', 'username')
    db_path = config.get('library', 'database_path', default='data/metadata.db')

    lastfm = LastFMClient(api_key, username)
    library = LocalLibraryClient(db_path)

    # Get recent scrobbles (last 14 days)
    lookback_days = config.recently_played_lookback_days
    history_days = config.get('lastfm', 'history_days', default=90)

    logger.info(f"Fetching Last.fm scrobbles...")
    scrobbles = lastfm.get_recent_tracks(days=history_days, use_cache=True)

    cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())

    # Build set of recent scrobble keys
    scrobble_keys = set()
    recent_scrobbles = []
    for s in scrobbles:
        ts = s.get("timestamp", 0)
        if ts == 0 or ts < cutoff_timestamp:
            continue
        k = _key_for_track(s)
        if k:
            scrobble_keys.add(k)
            recent_scrobbles.append(s)

    logger.info(f"Found {len(recent_scrobbles)} scrobbles within last {lookback_days} days")
    logger.info(f"Generated {len(scrobble_keys)} unique keys from scrobbles")

    # Get some library tracks
    logger.info(f"\nGetting library tracks...")
    all_tracks = library.get_all_tracks()
    logger.info(f"Library has {len(all_tracks)} tracks")

    # Test matching
    library_keys = {}
    for t in all_tracks:
        k = _key_for_track(t)
        if k:
            library_keys[k] = t

    logger.info(f"Generated {len(library_keys)} unique keys from library")

    # Find matches
    matches = []
    for key in scrobble_keys:
        if key in library_keys:
            scrobble = next(s for s in recent_scrobbles if _key_for_track(s) == key)
            track = library_keys[key]
            matches.append({
                'key': key,
                'scrobble': scrobble,
                'track': track,
            })

    logger.info(f"\n{'='*80}")
    logger.info(f"MATCHING RESULTS:")
    logger.info(f"{'='*80}")
    logger.info(f"Recent scrobbles: {len(recent_scrobbles)}")
    logger.info(f"Scrobble keys: {len(scrobble_keys)}")
    logger.info(f"Library tracks that match recent scrobbles: {len(matches)}")
    logger.info(f"Match rate: {len(matches)/len(scrobble_keys)*100:.1f}%")

    if matches:
        logger.info(f"\nExample matches (first 10):")
        for i, m in enumerate(matches[:10], 1):
            s = m['scrobble']
            t = m['track']
            play_date = datetime.fromtimestamp(s.get('timestamp', 0))
            logger.info(f"\n  {i}. {s.get('artist')} - {s.get('title')}")
            logger.info(f"     Last played: {play_date}")
            logger.info(f"     Library track_id: {t.get('rating_key')}")
            logger.info(f"     Match key: {m['key'][:80]}...")

    # Test if these would be filtered
    logger.info(f"\n{'='*80}")
    logger.info(f"FILTER TEST:")
    logger.info(f"{'='*80}")
    logger.info(f"If you generated a playlist right now:")
    logger.info(f"  - {len(matches)} tracks would be excluded (recently played)")
    logger.info(f"  - {len(all_tracks) - len(matches)} tracks would be available")

    if len(matches) == 0:
        logger.warning(f"\n⚠️  WARNING: No matches found!")
        logger.warning(f"This means the recency filter is NOT working correctly.")
        logger.warning(f"Recently played tracks from Last.fm are NOT being filtered from playlists.")
        logger.warning(f"\nPossible causes:")
        logger.warning(f"  1. normalize_match_string() produces different keys for Last.fm vs library")
        logger.warning(f"  2. Artist/title data format differs between sources")
        logger.warning(f"  3. Library 'rating_key' field is not being populated correctly")

        # Show some examples of non-matching keys
        logger.info(f"\nRecent scrobble keys (first 10):")
        for k in list(scrobble_keys)[:10]:
            logger.info(f"  {k}")

        logger.info(f"\nLibrary keys (first 10):")
        for k in list(library_keys.keys())[:10]:
            logger.info(f"  {k}")

if __name__ == "__main__":
    main()
