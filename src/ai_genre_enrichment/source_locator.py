"""Structured request contract for locating authoritative release sources."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SOURCE_LOCATOR_INSTRUCTIONS = """
Find candidate authoritative release-specific source pages for a local library album.

Return only official artist, official label, official release, Bandcamp release,
label catalog, release notes, press release, or artist/label supplied liner-note pages.

Do not return MusicBrainz, Discogs, Last.fm, Spotify, Apple Music, Qobuz, Tidal,
Wikipedia, Wikidata, generic stores, blogs, reviews, shops, or tag-cloud pages as
authoritative sources. MusicBrainz and Discogs are baseline payload data already
handled deterministically outside this lane.

Prefer Bandcamp release pages when a release-specific Bandcamp URL can be confirmed.
Do not infer genres. This step only locates candidate source pages and summarizes
why each URL appears to identify the requested release.
""".strip()


_SOURCE_LOCATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_sources", "warnings"],
    "properties": {
        "candidate_sources": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_url",
                    "source_type",
                    "source_name",
                    "identity_status",
                    "identity_confidence",
                    "release_specific",
                    "reason",
                ],
                "properties": {
                    "source_url": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": [
                            "official_release",
                            "official_artist",
                            "official_label",
                            "bandcamp_release",
                            "label_catalog",
                            "release_notes",
                            "press_release",
                            "liner_notes",
                        ],
                    },
                    "source_name": {"type": "string"},
                    "identity_status": {"type": "string", "enum": ["confirmed", "probable", "ambiguous"]},
                    "identity_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "release_specific": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def source_locator_response_format() -> dict[str, Any]:
    """Return a strict JSON schema for source-locator Responses API calls."""
    return {
        "type": "json_schema",
        "name": "ai_genre_source_locator",
        "schema": deepcopy(_SOURCE_LOCATOR_SCHEMA),
        "strict": True,
    }
