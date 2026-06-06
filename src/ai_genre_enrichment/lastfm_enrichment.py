"""Slim Last.fm tag fetcher for the genre enrichment pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://ws.audioscrobbler.com/2.0/"

# Last.fm occasionally returns 5xx / rate-limits; these are worth a quick retry.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}

LASTFM_NOISE_TAGS = {
    "seen live", "favorite", "favorites", "favourites", "albums i own",
    "love", "loved", "beautiful", "awesome", "great", "amazing",
    "check out", "cool", "nice", "good", "best",
}


def fetch_lastfm_tags(
    *,
    artist: str,
    album: str | None = None,
    api_key: str,
    limit: int = 20,
) -> list[str]:
    """Fetch top tags for an artist (and optionally album) from Last.fm API.

    Calls album.gettoptags if album is provided, then artist.gettoptags.
    Pre-filters Last.fm noise tags and META_TAGS/DROP_TOKENS.

    Returns:
        List of tag name strings, filtered and deduplicated.
    """
    from src.genre.normalize_unified import DROP_TOKENS, META_TAGS

    # Skip junk artist names (e.g. "@", pure punctuation) that can never resolve.
    # Keep non-Latin names (Japanese/Cyrillic, etc.) — str.isalnum() accepts them.
    if not artist or not any(ch.isalnum() for ch in artist):
        return []

    noise = LASTFM_NOISE_TAGS | META_TAGS | DROP_TOKENS
    all_tags: list[str] = []

    if album:
        album_tags = _fetch_toptags(
            "album.gettoptags", api_key, artist=artist, album=album, limit=limit
        )
        all_tags.extend(album_tags)

    artist_tags = _fetch_toptags(
        "artist.gettoptags", api_key, artist=artist, limit=limit
    )
    all_tags.extend(artist_tags)

    seen: set[str] = set()
    filtered: list[str] = []
    for tag in all_tags:
        key = tag.strip().casefold()
        if key and key not in noise and key not in seen and len(key) > 2:
            seen.add(key)
            filtered.append(tag.strip())
    return filtered


def _fetch_toptags(
    method: str, api_key: str, *, limit: int = 20, max_retries: int = 3, **params: str
) -> list[str]:
    """Call a Last.fm gettoptags endpoint and return raw tag names.

    Retries transient failures (5xx / rate-limit / network) with a short
    backoff. On permanent failure logs a one-line warning (no traceback, since
    the caller handles an empty result) and returns an empty list.
    """
    request_params: dict[str, Any] = {
        "method": method,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
        **params,
    }

    data: dict[str, Any] | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(BASE_URL, params=request_params, timeout=15)
            if response.status_code in _TRANSIENT_STATUS:
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)
                    continue
                logger.warning(
                    "Last.fm %s: HTTP %d after %d attempts (giving up)",
                    method, response.status_code, attempt,
                )
                return []
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            logger.warning("Last.fm %s failed after %d attempts: %s", method, attempt, exc)
            return []
        except ValueError as exc:  # malformed JSON body
            logger.warning("Last.fm %s returned invalid JSON: %s", method, exc)
            return []

    if data is None:
        return []

    toptags = data.get("toptags", {})
    tags = toptags.get("tag", [])
    if not isinstance(tags, list):
        tags = [tags] if tags else []

    return [tag.get("name", "") for tag in tags[:limit] if tag.get("name")]
