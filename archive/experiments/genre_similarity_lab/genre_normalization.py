"""
Shared genre normalization and filtering utilities for experiments.

This module provides conservative, well-documented helpers to:
- Normalize individual genre tokens.
- Split composite genre strings and normalize pieces.
- Apply filter sets (broad/garbage/meta) and optional canonical whitelist.
- Load filter sets from CSVs and config-provided lists.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def normalize_genre_token(raw: str) -> Optional[str]:
    """
    Normalize a single raw genre string.

    Steps:
    - Strip leading/trailing whitespace.
    - Lowercase.
    - Collapse internal whitespace to a single space.
    - Remove obvious trailing punctuation like commas/semicolons.
    - Replace punctuation variants so that "post bop" and "post-bop" normalize similarly:
        * Convert separators like "-", "/", "\\" to spaces.
        * Normalize '&' to 'and' (e.g., "r&b" -> "r and b").
    - Collapse multiple spaces again.
    - If empty after cleaning, return None.

    This is intentionally conservative; alias collapsing (e.g., "r and b" -> "rnb")
    can be layered on later if desired.
    """
    if raw is None:
        return None
    # Trim and lowercase
    s = raw.strip().lower()
    if not s:
        return None
    # Replace common separators (\ / -) with spaces
    s = re.sub(r"[\\/\\-]", " ", s)
    # Normalize ampersand to "and"
    s = s.replace("&", " and ")
    # Remove trailing punctuation like commas/semicolons
    s = s.rstrip(";,")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    return s


def split_and_normalize(raw: str) -> Set[str]:
    """
    Take a raw genre field that may contain multiple genres and return a set of normalized tokens.

    Splitting rules (conservative):
    - Always split on semicolons ';'.
    - Also split on commas ',' and forward slashes '/' as separators.
      (We accept the risk of splitting composite labels like "folk, world, & country"
       for now; behavior can be tightened later.)
    - Normalize each piece via normalize_genre_token.
    - Drop Nones/empties.
    """
    tokens: Set[str] = set()
    if raw is None:
        return tokens

    # First pass: split on semicolons
    pieces = []
    for part in raw.split(";"):
        pieces.extend(re.split(r"[,/]", part))

    for piece in pieces:
        norm = normalize_genre_token(piece)
        if norm:
            tokens.add(norm)
    return tokens


def normalize_and_filter_genres(
    raw_genres: Iterable[str],
    *,
    broad_set: Optional[Set[str]] = None,
    garbage_set: Optional[Set[str]] = None,
    meta_set: Optional[Set[str]] = None,
    canonical_set: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Given raw genre strings (possibly composite), return a set of normalized, filtered tokens.

    Steps:
    - For each raw genre, split_and_normalize(raw) to get normalized tokens.
    - Union all tokens.
    - Drop tokens in broad_set, garbage_set, meta_set if provided.
    - If canonical_set is provided, keep only tokens present in canonical_set.
    """
    tokens: Set[str] = set()
    for g in raw_genres:
        tokens.update(split_and_normalize(g))

    def _drop(source: Set[str], drops: Optional[Set[str]]) -> Set[str]:
        return source if not drops else {t for t in source if t not in drops}

    tokens = _drop(tokens, broad_set)
    tokens = _drop(tokens, garbage_set)
    tokens = _drop(tokens, meta_set)

    if canonical_set is not None:
        tokens = {t for t in tokens if t in canonical_set}

    return tokens


def _load_genre_csv(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = normalize_genre_token(row.get("genre", ""))
            if norm:
                out.add(norm)
    return out


def load_filter_sets(
    broad_filters: Optional[Iterable[str]] = None,
    garbage_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Load/normalize filter sets for genres.

    Args:
        broad_filters: iterable from config.yaml (already in memory)
        garbage_path: CSV with column 'genre' (optional)
        meta_path: CSV with column 'genre' (optional)

    Returns:
        (broad_set, garbage_set, meta_set) as normalized sets.
    """
    broad_set: Set[str] = set()
    garbage_set: Set[str] = set()
    meta_set: Set[str] = set()

    if broad_filters:
        for g in broad_filters:
            norm = normalize_genre_token(g)
            if norm:
                broad_set.add(norm)

    if garbage_path:
        garbage_set = _load_genre_csv(Path(garbage_path))
    if meta_path:
        meta_set = _load_genre_csv(Path(meta_path))

    logger.info(
        "Loaded filter sets: broad=%d, garbage=%d, meta=%d",
        len(broad_set),
        len(garbage_set),
        len(meta_set),
    )

    return broad_set, garbage_set, meta_set
