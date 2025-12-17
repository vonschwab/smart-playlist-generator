import logging
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def fetch_artist_top_tracks(
    username: str,
    api_key: str,
    artist_name: str,
    max_pages: int = 5,
    page_limit: int = 200,
) -> List[Dict[str, object]]:
    """
    Fetch user top tracks and filter to the given artist (case-insensitive).
    Stops after collecting up to 10 tracks or hitting max_pages.
    """
    matches: List[Dict[str, object]] = []
    target_artist = _norm(artist_name)
    for page in range(1, max_pages + 1):
        params = {
            "method": "user.gettoptracks",
            "user": username,
            "api_key": api_key,
            "format": "json",
            "limit": page_limit,
            "page": page,
        }
        try:
            resp = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=10)
        except requests.RequestException as exc:
            logger.warning("Last.fm request failed: %s", exc)
            break

        if resp.status_code != 200:
            logger.warning("Last.fm non-200 response: %s", resp.status_code)
            break

        data = resp.json()
        tracks = data.get("toptracks", {}).get("track", [])
        if not tracks:
            break

        for entry in tracks:
            try:
                name = entry.get("name") or ""
                artist = entry.get("artist", {}).get("name") or ""
                playcount = int(entry.get("playcount", 0))
            except Exception:
                continue

            if _norm(artist) != target_artist:
                continue
            matches.append({"track_name": name, "artist_name": artist, "playcount": playcount})
            if len(matches) >= 10:
                break

        if len(matches) >= 10:
            break

    matches.sort(key=lambda t: t.get("playcount", 0), reverse=True)
    return matches[:10]
