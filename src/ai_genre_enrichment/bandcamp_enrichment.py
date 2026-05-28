"""Slim Bandcamp tag fetcher for the genre enrichment pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .source_extraction import fetch_bandcamp_release_tags, is_bandcamp_release_url
from .source_locator import SOURCE_LOCATOR_INSTRUCTIONS, source_locator_response_format

logger = logging.getLogger(__name__)

MIN_LOCATOR_CONFIDENCE = 0.7


def fetch_bandcamp_tags(
    *,
    artist: str,
    album: str | None,
    api_key: str,
    model: str = "gpt-4o-mini",
    fetch_html: Callable[[str], str] | None = None,
) -> list[str]:
    """Fetch Bandcamp tags for a release.

    Uses the AI source locator to find a confirmed Bandcamp release URL,
    then scrapes the tag list from the page.

    Returns:
        List of raw tag strings (deduplicated). Empty if no confirmed Bandcamp URL found.
    """
    locator_response = _locate_bandcamp_url(
        artist=artist, album=album, model=model, api_key=api_key
    )
    url = _pick_bandcamp_url(locator_response)
    if not url:
        return []
    try:
        tags = fetch_bandcamp_release_tags(url, fetch_html=fetch_html)
    except OSError:
        logger.exception("Bandcamp HTML fetch failed for %s", url)
        return []

    seen: set[str] = set()
    filtered: list[str] = []
    for tag in tags:
        key = tag.strip().casefold()
        # skip single-char noise tags
        if key and key not in seen and len(key) > 2:
            seen.add(key)
            filtered.append(tag.strip())
    return filtered


def _pick_bandcamp_url(locator_response: dict[str, Any]) -> str | None:
    candidates = locator_response.get("candidate_sources", []) or []
    bandcamp_candidates = [
        c for c in candidates
        if c.get("source_type") == "bandcamp_release"
        and c.get("identity_status") == "confirmed"
        and (c.get("identity_confidence") or 0) >= MIN_LOCATOR_CONFIDENCE
        and is_bandcamp_release_url(c.get("source_url") or "")
    ]
    if not bandcamp_candidates:
        return None
    bandcamp_candidates.sort(key=lambda c: c.get("identity_confidence") or 0, reverse=True)
    return bandcamp_candidates[0]["source_url"]


def _locate_bandcamp_url(
    *, artist: str, album: str | None, model: str, api_key: str
) -> dict[str, Any]:
    """Call OpenAI source locator to find a Bandcamp URL for the release."""
    from .client import OpenAIEnrichmentClient, _extract_response_json

    client = OpenAIEnrichmentClient(model=model, api_key=api_key)
    prompt = f"artist: {artist}\nalbum: {album or ''}"
    try:
        # Use _call_openai directly — source locator schema differs from the
        # classification schema that client.enrich() validates against.
        raw = client._call_openai(
            prompt,
            source_locator_response_format(),
            instructions=SOURCE_LOCATOR_INSTRUCTIONS,
        )
        return _extract_response_json(raw) or {"candidate_sources": [], "warnings": []}
    except Exception as exc:
        logger.warning(
            "Source locator failed for %s / %s — %s",
            artist,
            album,
            exc,
        )
        return {"candidate_sources": [], "warnings": []}
