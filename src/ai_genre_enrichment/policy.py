"""Stable source-policy rules for sidecar genre enrichment."""

from __future__ import annotations

LEGACY_POLICY_VERSION = "legacy-v0"
STABILIZED_POLICY_VERSION = "genre-enrichment-v2"
CANONICAL_BANDCAMP_SOURCE_TYPE = "bandcamp_release"


def canonical_source_type(source_type: str) -> str:
    """Map historical source names to the current read-side contract."""
    return CANONICAL_BANDCAMP_SOURCE_TYPE if source_type == "bandcamp_tags" else source_type


def evidence_basis(source_type: str) -> str:
    """Return the durable enriched_genres basis for a canonicalized source."""
    canonical = canonical_source_type(source_type)
    if canonical == "lastfm_tags":
        return "lastfm_tags"
    if canonical == "local_metadata":
        return "local_metadata"
    return "authoritative_source"


def can_seed_signature(source_type: str) -> bool:
    """Last.fm may corroborate an existing term but cannot create one."""
    return canonical_source_type(source_type) != "lastfm_tags"
