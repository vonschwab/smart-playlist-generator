"""
Similarity scoring module for playlist generation.

This module computes sonic, genre, and hybrid similarity scores for candidate tracks.

Migrated from src/playlist_generator.py scoring methods (Phase 5).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for similarity scoring."""
    sonic_weight: float = 0.67
    genre_weight: float = 0.33
    min_genre_similarity: float = 0.2
    genre_method: str = "ensemble"
    artist_cap: int = 6


@dataclass(frozen=True)
class ScoringResult:
    """
    Result of scoring operation.

    Attributes:
        scored_tracks: Tracks with similarity_score, hybrid_score, genre_sim fields
        genre_pass: Tracks that passed genre gate
        genre_fail: Tracks that failed genre gate with their scores
        stats: Diagnostic statistics
    """
    scored_tracks: List[Dict[str, Any]]
    genre_pass: List[Dict[str, Any]] = field(default_factory=list)
    genre_fail: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def score_sonic_similarity(
    *,
    candidates: List[Dict[str, Any]],
    similarity_calculator,
) -> List[Dict[str, Any]]:
    """
    Compute sonic similarity scores for candidates.

    Adds 'similarity_score' field to each track.

    Args:
        candidates: Candidate tracks
        similarity_calculator: SimilarityCalculator instance

    Returns:
        Candidates with sonic similarity scores
    """
    # TODO: Implement in Phase 5
    raise NotImplementedError("Will be implemented in Phase 5")


def score_genre_and_hybrid(
    *,
    candidates: List[Dict[str, Any]],
    similarity_calculator,
    genre_method: str = "ensemble",
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], float]]]:
    """
    Compute genre and hybrid scores; partition into pass/fail.

    Adds 'genre_sim' and 'hybrid_score' fields to each track.
    Partitions based on min_genre_similarity gate.

    Args:
        candidates: Candidates with sonic scores already computed
        similarity_calculator: SimilarityCalculator instance
        genre_method: Genre similarity method

    Returns:
        Tuple of (genre_pass, genre_fail) lists
    """
    genre_pass = []
    genre_fail = []

    for track in candidates:
        seed_id = track.get('seed_rating_key')
        track_id = track.get('rating_key')
        if not seed_id or not track_id:
            continue

        # Get genres for seed and candidate
        seed_genres = similarity_calculator._get_combined_genres(seed_id)
        cand_genres = similarity_calculator._get_combined_genres(track_id)

        # Calculate genre similarity
        genre_sim = (
            similarity_calculator.genre_calc.calculate_similarity(
                seed_genres, cand_genres, method=genre_method
            )
            if seed_genres and cand_genres
            else 0.0
        )

        # Calculate hybrid score (combines sonic + genre)
        hybrid = similarity_calculator.calculate_hybrid_similarity(seed_id, track_id)
        if hybrid is None or hybrid <= 0:
            genre_fail.append((track, genre_sim))
            continue

        # Add scores to track
        track['hybrid_score'] = hybrid
        track['genre_sim'] = genre_sim
        genre_pass.append(track)

    return genre_pass, genre_fail


def cap_candidates_by_artist(
    *,
    candidates: List[Dict[str, Any]],
    artist_cap: int,
    limit: int,
    score_key: str = "similarity_score",
) -> List[Dict[str, Any]]:
    """
    Apply artist cap and truncate to limit based on score ordering.

    Ensures no artist has more than artist_cap tracks.

    Args:
        candidates: Candidates to cap
        artist_cap: Maximum tracks per artist
        limit: Maximum total tracks
        score_key: Score field to sort by

    Returns:
        Capped and limited candidates
    """
    sorted_candidates = sorted(candidates, key=lambda t: t.get(score_key, 0), reverse=True)
    capped = []
    artist_counts = Counter()

    for track in sorted_candidates:
        artist = (track.get('artist') or '').lower()
        if artist_counts[artist] >= artist_cap:
            continue
        artist_counts[artist] += 1
        capped.append(track)
        if len(capped) >= limit:
            break

    return capped


def finalize_pool(
    *,
    candidates: List[Dict[str, Any]],
    artist_cap: int,
    pool_target: int,
    score_key: str = "hybrid_score",
) -> List[Dict[str, Any]]:
    """
    Sort by score, enforce artist cap, and trim to pool_target.

    Final step to create candidate pool ready for playlist construction.

    Args:
        candidates: Scored candidates
        artist_cap: Maximum tracks per artist
        pool_target: Target pool size
        score_key: Score field to sort by

    Returns:
        Finalized candidate pool
    """
    final_pool = []
    artist_counts = Counter()

    for track in sorted(candidates, key=lambda t: t.get(score_key, 0), reverse=True):
        artist = (track.get('artist') or '').lower()
        if artist_counts[artist] >= artist_cap:
            continue
        artist_counts[artist] += 1
        final_pool.append(track)
        if len(final_pool) >= pool_target:
            break

    return final_pool
