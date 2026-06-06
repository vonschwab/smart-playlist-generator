"""Slim Bandcamp tag fetcher for the genre enrichment pipeline."""

from __future__ import annotations

import logging
import time
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
) -> tuple[str | None, list[str], float | None]:
    """Fetch Bandcamp tags for a release.

    Uses the AI source locator to find a confirmed Bandcamp release URL,
    then scrapes the tag list from the page.

    Returns:
        Selected Bandcamp release URL, deduplicated raw tags, and locator
        confidence. The URL is None if no confirmed release URL was found.
    """
    locator_response = _locate_bandcamp_url(
        artist=artist, album=album, model=model, api_key=api_key
    )
    url, confidence = _pick_bandcamp_url(locator_response)
    if not url:
        return None, [], None
    try:
        tags = fetch_bandcamp_release_tags(url, fetch_html=fetch_html)
    except OSError:
        logger.exception("Bandcamp HTML fetch failed for %s", url)
        return url, [], confidence

    seen: set[str] = set()
    filtered: list[str] = []
    for tag in tags:
        key = tag.strip().casefold()
        # skip single-char noise tags
        if key and key not in seen and len(key) > 2:
            seen.add(key)
            filtered.append(tag.strip())
    return url, filtered, confidence


def _pick_bandcamp_url(locator_response: dict[str, Any]) -> tuple[str | None, float | None]:
    candidates = locator_response.get("candidate_sources", []) or []
    bandcamp_candidates = [
        c for c in candidates
        if c.get("source_type") == "bandcamp_release"
        and c.get("identity_status") == "confirmed"
        and (c.get("identity_confidence") or 0) >= MIN_LOCATOR_CONFIDENCE
        and is_bandcamp_release_url(c.get("source_url") or "")
    ]
    if not bandcamp_candidates:
        return None, None
    bandcamp_candidates.sort(key=lambda c: c.get("identity_confidence") or 0, reverse=True)
    selected = bandcamp_candidates[0]
    return selected["source_url"], selected["identity_confidence"]


def _locate_bandcamp_url(
    *, artist: str, album: str | None, model: str, api_key: str, max_retries: int = 3
) -> dict[str, Any]:
    """Call OpenAI source locator to find a Bandcamp URL for the release.

    Retries transient failures with a short backoff. On *persistent* failure it
    re-raises rather than returning empty: callers must distinguish "locator ran
    and found nothing" (a genuine miss, safe to cache) from "the call failed"
    (retryable, must NOT be cached as a miss — otherwise a bad key or outage
    would poison the attempt ledger for the whole library).
    """
    from .client import OpenAIEnrichmentClient, _extract_response_json

    client = OpenAIEnrichmentClient(model=model, api_key=api_key)
    prompt = f"artist: {artist}\nalbum: {album or ''}"
    for attempt in range(1, max_retries + 1):
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
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            logger.warning(
                "Source locator failed for %s / %s after %d attempts — %s",
                artist, album, attempt, exc,
            )
            raise
