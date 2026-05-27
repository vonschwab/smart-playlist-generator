"""Slim Last.fm tag fetcher for the genre enrichment pipeline."""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://ws.audioscrobbler.com/2.0/"

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
    method: str, api_key: str, *, limit: int = 20, **params: str
) -> list[str]:
    """Call a Last.fm gettoptags endpoint and return raw tag names."""
    request_params: dict[str, Any] = {
        "method": method,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
        **params,
    }

    try:
        response = requests.get(BASE_URL, params=request_params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Last.fm API request failed for %s", method)
        return []

    toptags = data.get("toptags", {})
    tags = toptags.get("tag", [])
    if not isinstance(tags, list):
        tags = [tags] if tags else []

    return [tag.get("name", "") for tag in tags[:limit] if tag.get("name")]
