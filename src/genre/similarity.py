"""
Genre Similarity Module - Taxonomy v1
======================================
Structured token-role similarity for genre comparison.

Precedence:
1. Exact canonical token match -> 1.0
2. Manual YAML override (if loaded)
3. Structural similarity based on token roles:
   - Same base genres => high similarity
   - Parent-child (punk <-> post-punk) => medium-high
   - Same scene different base (indie rock <-> indie pop) => medium
   - Conflicting bases (folk rock <-> folk metal) => low
   - Modifier only (post-rock <-> post-punk) => very low
4. Low-confidence fallback for unclassified genres
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import yaml

from .vocabulary import (
    BASE_GENRES,
    MODIFIERS,
    SCENE_TAGS,
    WEAK_ADJECTIVES,
    GenreStructure,
    parse_genre_structure,
    get_genre_family,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenreSimilarityResult:
    """Result of genre similarity calculation with metadata."""
    score: Optional[float]  # None = low confidence
    confidence: str  # "high", "medium", "low"
    reason: str
    matched_bases: FrozenSet[str]
    matched_scenes: FrozenSet[str]
    matched_modifiers: FrozenSet[str]

    @property
    def is_low_confidence(self) -> bool:
        return self.score is None or self.confidence == "low"


# Similarity score constants (tunable)
SCORE_EXACT_MATCH = 1.0
SCORE_PARENT_CHILD = 0.80      # punk <-> post-punk
SCORE_SAME_BASE_SAME_SCENE = 0.75  # indie rock <-> alt rock (both scene rock)
SCORE_SIBLINGS = 0.60          # hard rock <-> soft rock (same base, diff modifier)
SCORE_SAME_FAMILY = 0.20       # rock <-> metal (same family - weak signal)
SCORE_SHARED_BASE_CONFLICT = 0.15  # folk rock <-> folk metal (conflicting directions)
SCORE_SAME_SCENE_DIFF_BASE = 0.10  # indie rock <-> indie pop (scene alone = very weak)
SCORE_MODIFIER_ONLY = 0.0      # post-rock <-> post-punk (modifier alone = no signal)
SCORE_UNRELATED = 0.0


# YAML override cache
_yaml_matrix: Optional[Dict[str, Dict[str, float]]] = None
_yaml_path: Optional[str] = None


def load_yaml_overrides(filepath: str) -> None:
    """
    Load manual YAML similarity overrides.
    These take precedence over structural similarity.
    """
    global _yaml_matrix, _yaml_path

    try:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"YAML similarity file not found: {filepath}")
            _yaml_matrix = {}
            _yaml_path = filepath
            return

        with open(path, 'r', encoding='utf-8') as f:
            _yaml_matrix = yaml.safe_load(f) or {}
        _yaml_path = filepath
        logger.info(f"Loaded {len(_yaml_matrix)} genre entries from YAML overrides")
    except Exception as e:
        logger.error(f"Error loading YAML similarity file: {e}")
        _yaml_matrix = {}
        _yaml_path = filepath


def _lookup_yaml_override(genre1: str, genre2: str) -> Optional[float]:
    """Look up similarity from YAML matrix (both directions)."""
    if _yaml_matrix is None:
        return None

    g1, g2 = genre1.lower(), genre2.lower()

    # Check g1 -> g2
    if g1 in _yaml_matrix and g2 in _yaml_matrix[g1]:
        return _yaml_matrix[g1][g2]

    # Check g2 -> g1
    if g2 in _yaml_matrix and g1 in _yaml_matrix[g2]:
        return _yaml_matrix[g2][g1]

    return None


def _is_parent_child(struct1: GenreStructure, struct2: GenreStructure) -> bool:
    """
    Check if one genre is a parent/child of the other.
    e.g., "rock" <-> "indie rock", "punk" <-> "post-punk"

    A genre A is parent of B if:
    - A is a base genre
    - B contains A as a base plus additional modifiers/scenes
    """
    # Check if one is simpler (just a base) and the other is compound
    if struct1.bases and not struct1.modifiers and not struct1.scenes:
        # struct1 is a pure base, check if struct2 contains it
        if struct1.bases & struct2.bases:
            # struct2 has the same base plus something else
            if struct2.modifiers or struct2.scenes:
                return True

    if struct2.bases and not struct2.modifiers and not struct2.scenes:
        # struct2 is a pure base, check if struct1 contains it
        if struct2.bases & struct1.bases:
            if struct1.modifiers or struct1.scenes:
                return True

    return False


def _are_siblings(struct1: GenreStructure, struct2: GenreStructure) -> bool:
    """
    Check if genres are siblings (same base, different modifier).
    e.g., "hard rock" <-> "soft rock"
    """
    # Same bases
    if struct1.bases != struct2.bases:
        return False

    # Both have modifiers
    if not struct1.modifiers or not struct2.modifiers:
        return False

    # Different modifiers (but overlapping is OK if different primary)
    return struct1.modifiers != struct2.modifiers


def _structural_similarity(struct1: GenreStructure, struct2: GenreStructure) -> Tuple[float, str, str]:
    """
    Compute structural similarity between two parsed genres.

    Returns (score, reason, confidence)
    """
    # Exact match on original
    if struct1.original == struct2.original:
        return SCORE_EXACT_MATCH, "exact_match", "high"

    shared_bases = struct1.bases & struct2.bases
    shared_scenes = struct1.scenes & struct2.scenes
    shared_modifiers = struct1.modifiers & struct2.modifiers

    # Case 1: Parent-child relationship
    if _is_parent_child(struct1, struct2):
        return SCORE_PARENT_CHILD, "parent_child", "high"

    # Case 2: Same bases, same scenes (e.g., indie rock <-> alt rock)
    if shared_bases and shared_scenes:
        return SCORE_SAME_BASE_SAME_SCENE, "same_base_same_scene", "high"

    # Case 3: Siblings (same base, different modifiers)
    if _are_siblings(struct1, struct2):
        return SCORE_SIBLINGS, "siblings", "high"

    # Case 4: Same base(s), no scene conflict
    if shared_bases:
        # Check for conflicting bases (e.g., rock + metal in one, rock + jazz in other)
        other_bases_1 = struct1.bases - shared_bases
        other_bases_2 = struct2.bases - shared_bases

        if other_bases_1 or other_bases_2:
            # They share some base but also have conflicting bases
            return SCORE_SHARED_BASE_CONFLICT, "shared_base_conflict", "medium"
        else:
            # Same bases entirely
            return SCORE_SIBLINGS, "same_bases", "high"

    # Case 5: Same scene, different bases (e.g., indie rock <-> indie pop)
    if shared_scenes and struct1.bases and struct2.bases:
        return SCORE_SAME_SCENE_DIFF_BASE, "same_scene_diff_base", "medium"

    # Case 5b: Scene-only genre matches scene + base genre
    # e.g., "indie" <-> "indie rock", "alternative" <-> "alternative rock"
    if shared_scenes:
        # One has scene only, other has scene + base
        s1_scene_only = struct1.scenes and not struct1.bases and not struct1.modifiers
        s2_scene_only = struct2.scenes and not struct2.bases and not struct2.modifiers

        if s1_scene_only or s2_scene_only:
            # This is like parent-child for scene tags
            return SCORE_PARENT_CHILD, "scene_parent_child", "high"

    # Case 6: Same family (rock <-> metal, jazz <-> bebop)
    family1 = get_genre_family(struct1.original)
    family2 = get_genre_family(struct2.original)
    if family1 and family2 and family1 == family2:
        return SCORE_SAME_FAMILY, "same_family", "medium"

    # Case 7: Weak adjective only match (black metal <-> black gospel) -> block
    # Check this BEFORE modifier-only to catch modifiers that are also weak adjectives
    shared_weak = struct1.weak_adjectives & struct2.weak_adjectives
    if shared_weak and not shared_bases and not shared_scenes:
        # If the only shared modifiers are also weak adjectives, block
        modifiers_minus_weak = shared_modifiers - shared_weak
        if not modifiers_minus_weak:
            return SCORE_UNRELATED, "weak_adjective_only", "high"

    # Case 8: Modifier only match (dangerous - post-rock <-> post-punk)
    if shared_modifiers and not shared_bases and not shared_scenes:
        # Explicitly low score - sharing "post" doesn't mean similar
        return SCORE_MODIFIER_ONLY, "modifier_only", "low"

    # Case 9: No meaningful overlap
    # But check if we have enough info to be confident
    if not struct1.bases and not struct2.bases:
        # Both are unclassified
        return None, "both_unclassified", "low"

    return SCORE_UNRELATED, "unrelated", "medium"


@lru_cache(maxsize=10000)
def pairwise_genre_similarity(
    genre1: str,
    genre2: str,
    use_yaml_overrides: bool = True,
) -> GenreSimilarityResult:
    """
    Compute similarity between two canonical genre tokens.

    Precedence:
    1. Exact match -> 1.0
    2. YAML override (if enabled and found)
    3. Structural similarity

    Args:
        genre1: First normalized genre token
        genre2: Second normalized genre token
        use_yaml_overrides: Whether to check YAML for manual overrides

    Returns:
        GenreSimilarityResult with score, confidence, and metadata
    """
    g1, g2 = genre1.lower().strip(), genre2.lower().strip()

    # Exact match
    if g1 == g2:
        return GenreSimilarityResult(
            score=SCORE_EXACT_MATCH,
            confidence="high",
            reason="exact_match",
            matched_bases=frozenset(),
            matched_scenes=frozenset(),
            matched_modifiers=frozenset(),
        )

    # Check YAML override
    if use_yaml_overrides:
        yaml_score = _lookup_yaml_override(g1, g2)
        if yaml_score is not None:
            return GenreSimilarityResult(
                score=yaml_score,
                confidence="high",
                reason="yaml_override",
                matched_bases=frozenset(),
                matched_scenes=frozenset(),
                matched_modifiers=frozenset(),
            )

    # Structural similarity
    struct1 = parse_genre_structure(g1)
    struct2 = parse_genre_structure(g2)

    score, reason, confidence = _structural_similarity(struct1, struct2)

    return GenreSimilarityResult(
        score=score,
        confidence=confidence,
        reason=reason,
        matched_bases=struct1.bases & struct2.bases,
        matched_scenes=struct1.scenes & struct2.scenes,
        matched_modifiers=struct1.modifiers & struct2.modifiers,
    )


def genre_set_similarity(
    genres1: FrozenSet[str],
    genres2: FrozenSet[str],
    method: str = "best_match",
    use_yaml_overrides: bool = True,
) -> Tuple[Optional[float], str]:
    """
    Compute similarity between two sets of genre tokens.

    Methods:
    - "best_match": Maximum pairwise similarity (good for "is there ANY match?")
    - "average": Mean of all pairwise similarities
    - "weighted_best": Best match, penalized if few matches

    Args:
        genres1: First set of genre tokens
        genres2: Second set of genre tokens
        method: Similarity method to use
        use_yaml_overrides: Whether to use YAML overrides (False for pure structural)

    Returns:
        (score, confidence) where score may be None for low confidence
    """
    if not genres1 or not genres2:
        return None, "low"

    # Collect all pairwise similarities
    pairs = []
    low_confidence_count = 0

    for g1 in genres1:
        for g2 in genres2:
            result = pairwise_genre_similarity(g1, g2, use_yaml_overrides=use_yaml_overrides)
            if result.is_low_confidence:
                low_confidence_count += 1
            else:
                pairs.append(result.score if result.score is not None else 0.0)

    if not pairs:
        # All pairs were low confidence
        return None, "low"

    if method == "best_match":
        score = max(pairs)
    elif method == "average":
        score = sum(pairs) / len(pairs)
    elif method == "weighted_best":
        best = max(pairs)
        # Penalize if mostly low-confidence pairs
        total_pairs = len(genres1) * len(genres2)
        confidence_ratio = 1 - (low_confidence_count / total_pairs)
        score = best * (0.5 + 0.5 * confidence_ratio)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Determine overall confidence
    if low_confidence_count > len(pairs):
        confidence = "low"
    elif low_confidence_count > 0:
        confidence = "medium"
    else:
        confidence = "high"

    return score, confidence


def clear_similarity_cache() -> None:
    """Clear the LRU cache for similarity computations."""
    pairwise_genre_similarity.cache_clear()


def get_cache_info() -> dict:
    """Get cache statistics."""
    info = pairwise_genre_similarity.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "maxsize": info.maxsize,
    }
