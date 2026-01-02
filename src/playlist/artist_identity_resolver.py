"""
Artist Identity Resolver - Collapse ensemble variants and split collaborations

This module provides identity key resolution for artist diversity constraints.
It collapses ensemble name variants (e.g., "Bill Evans Trio" → "bill evans")
and splits collaboration strings into participant keys.

Usage:
    from src.playlist.artist_identity_resolver import resolve_artist_identity_keys, ArtistIdentityConfig

    config = ArtistIdentityConfig(enabled=True)
    keys = resolve_artist_identity_keys("Bill Evans Trio", config)
    # Returns: {"bill evans"}

    keys = resolve_artist_identity_keys("Bob Brookmeyer & Bill Evans", config)
    # Returns: {"bob brookmeyer", "bill evans"}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Set, List
import logging

from src.string_utils import normalize_artist_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtistIdentityConfig:
    """
    Configuration for artist identity resolution.

    Attributes:
        enabled: Enable identity-based artist matching (default: False for backward compatibility)
        split_delimiters: Collaboration delimiters to split on (applied in order)
        strip_trailing_ensemble_terms: Remove ensemble designators from end of artist names
        trailing_ensemble_terms: List of ensemble terms to strip (supports multi-word)
    """
    enabled: bool = False
    split_delimiters: List[str] = field(default_factory=lambda: [
        ",",           # "Duke Ellington, John Coltrane"
        " & ",         # "Bob Brookmeyer & Bill Evans"
        " and ",       # "Artist A and Artist B"
        " feat. ",     # "Charli XCX feat. MØ"
        " feat ",
        " featuring ", # "Artist featuring Guest"
        " ft. ",
        " ft ",
        " with ",      # "Mount Eerie with Julie Doiron"
        " x ",         # "Artist A x Artist B"
        " + ",         # "Artist A + Artist B"
    ])
    strip_trailing_ensemble_terms: bool = True
    trailing_ensemble_terms: List[str] = field(default_factory=lambda: [
        # Multi-word terms (check first)
        "big band",
        "chamber orchestra",
        "symphony orchestra",
        "string quartet",
        # Single-word terms
        "orchestra",
        "ensemble",
        "trio",
        "quartet",
        "quintet",
        "sextet",
        "septet",
        "octet",
        "nonet",
        "group",
        "band",
    ])


def _normalize_component(text: str) -> str:
    """
    Normalize a single artist name component.

    Steps:
    - Strip whitespace
    - Lowercase
    - Collapse multiple spaces
    - Remove leading "the "

    Returns:
        Normalized string (may be empty)
    """
    if not text:
        return ""

    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^the\s+', '', text)
    text = text.strip()

    return text


def _strip_ensemble_designator(component: str, terms: List[str]) -> str:
    """
    Strip trailing ensemble designator from artist name component.

    Only strips if the term appears at the END of the string (after whitespace).
    Does NOT strip if term appears mid-name (e.g., "Art Ensemble of Chicago").

    Args:
        component: Normalized artist name component
        terms: List of ensemble terms (checked in order, multi-word first)

    Returns:
        Component with trailing ensemble term removed (if found)
    """
    if not component or not terms:
        return component

    # Check each term in order (multi-word terms should be first in the list)
    for term in terms:
        term_lower = term.lower().strip()
        if not term_lower:
            continue

        # Pattern: must be at end, preceded by whitespace
        # Use word boundary to avoid matching partial words
        pattern = r'\s+' + re.escape(term_lower) + r'$'

        if re.search(pattern, component):
            component = re.sub(pattern, '', component).strip()
            # Only strip one term per component
            break

    return component


def resolve_artist_identity_keys(
    artist_str: str,
    cfg: ArtistIdentityConfig,
) -> Set[str]:
    """
    Resolve artist string to a set of identity keys.

    Behavior:
    1. Normalize: lowercase, collapse whitespace, strip punctuation
    2. Split collaborations using configured delimiters
    3. Strip trailing ensemble terms from each component
    4. Return set of identity keys (one per participant)

    Never returns empty set - falls back to normalized original on parse failure.

    Args:
        artist_str: Raw artist string from metadata
        cfg: Identity resolution configuration

    Returns:
        Set of identity keys (always non-empty)

    Examples:
        >>> cfg = ArtistIdentityConfig(enabled=True)
        >>> resolve_artist_identity_keys("Bill Evans", cfg)
        {'bill evans'}
        >>> resolve_artist_identity_keys("Bill Evans Trio", cfg)
        {'bill evans'}
        >>> resolve_artist_identity_keys("Bob Brookmeyer & Bill Evans", cfg)
        {'bob brookmeyer', 'bill evans'}
        >>> resolve_artist_identity_keys("Duke Ellington, John Coltrane", cfg)
        {'duke ellington', 'john coltrane'}
    """
    if not cfg or not cfg.enabled:
        # Feature disabled - use existing normalization
        key = normalize_artist_key(artist_str)
        return {key} if key else {_normalize_component(artist_str or "")}

    if not artist_str:
        return {""}

    original = str(artist_str).strip()
    if not original:
        return {""}

    # Start with the full string
    components = [original]

    # Split on collaboration delimiters (applied in order)
    for delimiter in cfg.split_delimiters:
        new_components = []
        for comp in components:
            # Split on this delimiter (case-insensitive)
            if delimiter.strip():  # Non-empty delimiter
                # For exact string delimiters, use case-insensitive split
                parts = re.split(re.escape(delimiter), comp, flags=re.IGNORECASE)
                new_components.extend(parts)
            else:
                new_components.append(comp)
        components = new_components

    # Normalize each component
    identity_keys: Set[str] = set()

    for component in components:
        # Step 1: Basic normalization
        normalized = _normalize_component(component)
        if not normalized:
            continue

        # Step 2: Strip trailing ensemble designator (if enabled)
        if cfg.strip_trailing_ensemble_terms and cfg.trailing_ensemble_terms:
            normalized = _strip_ensemble_designator(normalized, cfg.trailing_ensemble_terms)

        # Step 3: Final cleanup and add to set
        normalized = normalized.strip()
        if normalized:
            identity_keys.add(normalized)

    # Fallback: if parsing failed completely, use original normalized string
    if not identity_keys:
        fallback = _normalize_component(original)
        identity_keys.add(fallback if fallback else original.lower())

    return identity_keys


def format_identity_keys_for_logging(keys: Set[str], max_keys: int = 3) -> str:
    """
    Format identity keys for logging output.

    Args:
        keys: Set of identity keys
        max_keys: Maximum keys to show before truncating

    Returns:
        Formatted string like "{key1, key2, ...}" or "{key1}" for single key
    """
    if not keys:
        return "{}"

    sorted_keys = sorted(keys)
    if len(sorted_keys) <= max_keys:
        return "{" + ", ".join(sorted_keys) + "}"
    else:
        shown = sorted_keys[:max_keys]
        remaining = len(sorted_keys) - max_keys
        return "{" + ", ".join(shown) + f", ... +{remaining}more" + "}"
