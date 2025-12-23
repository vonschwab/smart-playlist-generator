"""
Diagnostic script to check Last.fm recency filtering
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config
from src.lastfm_client import LastFMClient
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    config = Config()

    # Initialize Last.fm client
    api_key = config.get('lastfm', 'api_key')
    username = config.get('lastfm', 'username')

    if not api_key or not username:
        logger.error("Last.fm credentials not found in config")
        return

    lastfm = LastFMClient(api_key, username)

    # Get recent tracks
    history_days = config.get('lastfm', 'history_days', default=90)
    lookback_days = config.recently_played_lookback_days

    logger.info(f"Fetching Last.fm scrobbles (history_days={history_days})")
    scrobbles = lastfm.get_recent_tracks(days=history_days, use_cache=True)

    logger.info(f"Retrieved {len(scrobbles)} scrobbles")

    # Analyze timestamps
    cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
    cutoff_date = datetime.fromtimestamp(cutoff_timestamp)

    logger.info(f"\nRecency filter settings:")
    logger.info(f"  Lookback days: {lookback_days}")
    logger.info(f"  Cutoff date: {cutoff_date}")
    logger.info(f"  Cutoff timestamp: {cutoff_timestamp}")

    # Count scrobbles by category
    recent_with_timestamp = 0
    old_with_timestamp = 0
    missing_timestamp = 0
    zero_timestamp = 0

    recent_examples = []
    old_examples = []
    missing_examples = []

    for s in scrobbles:
        ts = s.get("timestamp", 0)

        if ts == 0:
            zero_timestamp += 1
            if len(missing_examples) < 5:
                missing_examples.append(f"{s.get('artist', 'Unknown')} - {s.get('title', 'Unknown')}")
        elif not ts:
            missing_timestamp += 1
            if len(missing_examples) < 5:
                missing_examples.append(f"{s.get('artist', 'Unknown')} - {s.get('title', 'Unknown')}")
        elif ts < cutoff_timestamp:
            old_with_timestamp += 1
            if len(old_examples) < 5:
                play_date = datetime.fromtimestamp(ts)
                old_examples.append(f"{s.get('artist', 'Unknown')} - {s.get('title', 'Unknown')} ({play_date})")
        else:
            recent_with_timestamp += 1
            if len(recent_examples) < 5:
                play_date = datetime.fromtimestamp(ts)
                recent_examples.append(f"{s.get('artist', 'Unknown')} - {s.get('title', 'Unknown')} ({play_date})")

    logger.info(f"\nScrobble timestamp analysis:")
    logger.info(f"  Recent (within {lookback_days} days): {recent_with_timestamp}")
    logger.info(f"  Old (older than {lookback_days} days): {old_with_timestamp}")
    logger.info(f"  Zero timestamp: {zero_timestamp}")
    logger.info(f"  Missing timestamp: {missing_timestamp}")

    if recent_examples:
        logger.info(f"\nRecent scrobble examples:")
        for ex in recent_examples:
            logger.info(f"    {ex}")

    if old_examples:
        logger.info(f"\nOld scrobble examples:")
        for ex in old_examples:
            logger.info(f"    {ex}")

    if missing_examples:
        logger.info(f"\nMissing/zero timestamp examples (BUG - these would be filtered!):")
        for ex in missing_examples:
            logger.info(f"    {ex}")

    # Show the bug
    if zero_timestamp > 0 or missing_timestamp > 0:
        logger.warning(f"\n⚠️  POTENTIAL BUG DETECTED:")
        logger.warning(f"  {zero_timestamp + missing_timestamp} scrobbles have missing or zero timestamps")
        logger.warning(f"  The current code would INCLUDE these in the filter set")
        logger.warning(f"  This would incorrectly filter out tracks that were never actually played")
        logger.warning(f"\n  Current code logic:")
        logger.warning(f"    if ts and ts < cutoff_timestamp:")
        logger.warning(f"        continue  # Skip old scrobbles")
        logger.warning(f"  Problem: If ts=0, the condition 'if ts' is False, so it doesn't skip")
        logger.warning(f"  Fix: Change to 'if ts == 0 or ts < cutoff_timestamp:'")

if __name__ == "__main__":
    main()
