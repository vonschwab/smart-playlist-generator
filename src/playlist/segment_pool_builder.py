"""
Segment Pool Builder for Pier-Bridge Playlists
==============================================

Extracted from pier_bridge_builder.py (Phase 3.2).

This module builds candidate pools for bridge segments, filtering and scoring
candidates based on similarity to both endpoint piers.

Functions extracted from pier_bridge_builder.py:
- _build_segment_candidate_pool_scored() → SegmentCandidatePoolBuilder.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import math

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.identity_keys import identity_keys_for_index


@dataclass
class SegmentPoolConfig:
    """Configuration for segment candidate pool building."""

    pier_a: int
    """Index of first pier (segment start)."""

    pier_b: int
    """Index of second pier (segment end)."""

    X_full_norm: np.ndarray
    """Full-track similarity matrix (N, D) - L2 normalized."""

    universe_indices: List[int]
    """Base universe of candidate indices to consider."""

    used_track_ids: Set[int]
    """Track indices already used in playlist."""

    bundle: ArtifactBundle
    """Artifact bundle with track metadata."""

    bridge_floor: float
    """Minimum similarity score for bridge candidates."""

    segment_pool_max: int
    """Maximum candidates to return."""

    allowed_set: Optional[Set[int]] = None
    """If provided, only consider indices in this set."""

    internal_connectors: Optional[Set[int]] = None
    """Optional internal connector indices (priority candidates)."""

    internal_connector_cap: int = 0
    """Maximum internal connectors to select."""

    internal_connector_priority: bool = True
    """If True, select internal connectors before external candidates."""

    seed_artist_key: Optional[str] = None
    """Seed artist key for policy enforcement."""

    disallow_pier_artists_in_interiors: bool = False
    """If True, exclude pier artists from interior positions."""

    disallow_seed_artist_in_interiors: bool = False
    """If True, exclude seed artist from interior positions."""

    used_track_keys: Optional[Set[tuple[str, str]]] = None
    """Track keys (artist, title) already used in playlist."""

    seed_track_keys: Optional[Set[tuple[str, str]]] = None
    """Track keys of seed tracks (for diagnostics)."""

    diagnostics: Optional[Dict[str, Any]] = None
    """Optional diagnostics dict to populate."""

    experiment_bridge_scoring_enabled: bool = False
    """Enable experimental bridge scoring for diagnostics/dry runs only."""

    experiment_bridge_min_weight: float = 0.25
    """Weight for min(sim_a, sim_b) in experimental bridge score."""

    experiment_bridge_balance_weight: float = 0.15
    """Weight for balance term in experimental bridge score."""

    # DJ union pooling (testing/opt-in only)
    pool_strategy: str = "segment_scored"
    """Pooling strategy: segment_scored (baseline) or dj_union."""

    pool_k_local: int = 0
    """Top-K local sonic neighbors (S1)."""

    pool_k_toward: int = 0
    """Top-K toward-B neighbors per step (S2)."""

    pool_k_genre: int = 0
    """Top-K genre-waypoint neighbors per step (S3)."""

    pool_k_union_max: int = 0
    """Max union size after dedupe."""

    pool_step_stride: int = 1
    """Stride for per-step targets (reduces compute)."""

    pool_cache_enabled: bool = True
    """Cache per-step top-K computations within a segment."""

    interior_length: int = 0
    """Interior length for the segment (used for step targets)."""

    progress_arc_enabled: bool = False
    """Whether to use arc-shaped progress for toward-B targets."""

    progress_arc_shape: str = "linear"
    """Progress arc shape for toward-B targets."""

    X_genre_norm: Optional[np.ndarray] = None
    """Normalized genre vectors (optional, for S3)."""

    X_genre_norm_idf: Optional[np.ndarray] = None
    """IDF-weighted normalized genre vectors (Phase 2, optional for S3)."""

    genre_targets: Optional[List[np.ndarray]] = None
    """Optional per-step genre targets (same basis as X_genre_norm/X_genre_norm_idf)."""

    pooling_cache: Optional[Dict[str, Any]] = None
    """Optional cache for per-step pooling computations."""

    pool_debug_compare_baseline: bool = False
    """If True, caller may compute baseline overlap for diagnostics."""

    pool_verbose: bool = False
    """Phase 3 fix: If True, log verbose per-step pool breakdown."""


@dataclass
class SegmentPoolResult:
    """Result of segment pool building."""

    candidates: List[int]
    """Candidate track indices for beam search."""

    artist_key_by_idx: Dict[int, str]
    """Mapping from track index to artist identity key."""

    title_key_by_idx: Dict[int, str]
    """Mapping from track index to title key."""

    diagnostics: Dict[str, Any] = field(default_factory=dict)
    """Diagnostics information about pool building."""


class SegmentCandidatePoolBuilder:
    """Builds candidate pool for a single bridge segment.

    Scores candidates based on similarity to both endpoint piers (harmonic mean),
    applies structural filters (used tracks, artist policies, track key collisions),
    gates by bridge floor, and returns top-K candidates.
    """

    def build(self, config: SegmentPoolConfig) -> SegmentPoolResult:
        """Build filtered, scored candidate pool for segment.

        Steps:
        1. Score all candidates by bridge quality (similarity to both piers)
        2. Apply bridge floor gate
        3. Apply artist policies (seed artist, pier artists)
        4. Apply track key deduplication
        5. Return sorted pool with 1-per-artist constraint

        Args:
            config: Configuration for pool building

        Returns:
            SegmentPoolResult with candidates and metadata
        """
        # Handle empty segment_pool_max
        segment_pool_max = int(max(0, config.segment_pool_max))
        if segment_pool_max <= 0:
            return self._empty_result(config, segment_pool_max)

        used_track_keys = config.used_track_keys or set()
        seed_track_keys = config.seed_track_keys or set()

        pool_strategy = str(config.pool_strategy or "segment_scored").strip().lower()
        if pool_strategy not in {"segment_scored", "dj_union"}:
            pool_strategy = "segment_scored"

        universe_indices = list(config.universe_indices)
        dj_diag: Dict[str, Any] = {}
        if pool_strategy == "dj_union":
            union_indices, dj_diag = self._build_dj_union_pool(config)
            if union_indices:
                universe_indices = union_indices

        # Get endpoint artist keys for policy enforcement
        pier_a_artist_key = identity_keys_for_index(config.bundle, config.pier_a).artist_key
        pier_b_artist_key = identity_keys_for_index(config.bundle, config.pier_b).artist_key

        # Phase 1: Structural filtering
        structural_result = self._apply_structural_filters(
            config=config,
            universe_indices=universe_indices,
            pool_strategy=pool_strategy,
            pier_a_artist_key=pier_a_artist_key,
            pier_b_artist_key=pier_b_artist_key,
            used_track_keys=used_track_keys,
            seed_track_keys=seed_track_keys,
        )

        if not structural_result.candidates:
            if dj_diag:
                structural_result.diagnostics.update(dj_diag)
            return self._empty_result_with_diagnostics(
                config,
                segment_pool_max,
                structural_result.diagnostics,
            )

        # Phase 2: Compute bridge scores
        bridge_result = self._compute_bridge_scores(
            config=config,
            candidates=structural_result.candidates,
            diagnostics=structural_result.diagnostics,
        )

        if not bridge_result.passing_candidates:
            return self._empty_result_with_diagnostics(
                config,
                segment_pool_max,
                bridge_result.diagnostics,
            )

        # Phase 3: Handle internal connectors (optional)
        internal_result = self._process_internal_connectors(
            config=config,
            pier_a_artist_key=pier_a_artist_key,
            pier_b_artist_key=pier_b_artist_key,
            used_track_keys=used_track_keys,
            artist_key_by_idx=structural_result.artist_key_by_idx,
            title_key_by_idx=structural_result.title_key_by_idx,
        )

        # Phase 4: Select final candidates with 1-per-artist constraint
        final_result = self._select_final_candidates(
            config=config,
            passing_sorted=bridge_result.passing_sorted,
            bridge_sim=bridge_result.bridge_sim,
            internal_result=internal_result,
            artist_key_by_idx=structural_result.artist_key_by_idx,
            segment_pool_max=segment_pool_max,
        )

        # Combine diagnostics
        diagnostics = {
            **structural_result.diagnostics,
            **bridge_result.diagnostics,
            **internal_result.diagnostics,
            **final_result.diagnostics,
            **dj_diag,
        }

        # Only return mappings for indices in the final candidate list
        artist_key_final = {
            i: structural_result.artist_key_by_idx.get(i, "")
            for i in final_result.candidates
        }
        title_key_final = {
            i: structural_result.title_key_by_idx.get(i, "")
            for i in final_result.candidates
        }

        if config.diagnostics is not None:
            config.diagnostics.update(diagnostics)

        return SegmentPoolResult(
            candidates=final_result.candidates,
            artist_key_by_idx=artist_key_final,
            title_key_by_idx=title_key_final,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _compute_bridge_score(
        sim_a: float,
        sim_b: float,
        config: SegmentPoolConfig,
    ) -> float:
        denom = sim_a + sim_b
        hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
        if not config.experiment_bridge_scoring_enabled:
            return float(hmean)

        min_weight = float(config.experiment_bridge_min_weight)
        min_weight = max(0.0, min(1.0, min_weight))
        balance_weight = float(config.experiment_bridge_balance_weight)
        balance_weight = max(0.0, min(1.0 - min_weight, balance_weight))
        hmean_weight = max(0.0, 1.0 - min_weight - balance_weight)

        min_sim = min(sim_a, sim_b)
        balance = 1.0 - abs(sim_a - sim_b)
        if balance < 0.0:
            balance = 0.0
        elif balance > 1.0:
            balance = 1.0

        return float(
            (hmean_weight * hmean)
            + (min_weight * min_sim)
            + (balance_weight * balance)
        )

    def _empty_result(
        self,
        config: SegmentPoolConfig,
        segment_pool_max: int,
    ) -> SegmentPoolResult:
        """Return empty result when segment_pool_max <= 0."""
        diagnostics = {
            "pool_strategy": str(config.pool_strategy or "segment_scored"),
            "base_universe": int(len(config.universe_indices)),
            "excluded_used_track_ids": 0,
            "excluded_allowed_set": 0,
            "excluded_seed_artist_policy": 0,
            "excluded_pier_artist_policy": 0,
            "excluded_track_key_collision": 0,
            "excluded_track_key_collision_with_piers": 0,
            "eligible_after_structural": 0,
            "below_bridge_floor": 0,
            "pass_bridge_floor": 0,
            "collapsed_by_artist_key": 0,
            "selected_external": 0,
            "internal_connectors_candidates": 0,
            "internal_connectors_pass_gate": 0,
            "internal_connectors_selected": 0,
            "final": 0,
            "segment_pool_max": int(segment_pool_max),
        }

        if config.diagnostics is not None:
            config.diagnostics.update(diagnostics)

        return SegmentPoolResult(
            candidates=[],
            artist_key_by_idx={},
            title_key_by_idx={},
            diagnostics=diagnostics,
        )

    def _empty_result_with_diagnostics(
        self,
        config: SegmentPoolConfig,
        segment_pool_max: int,
        diagnostics: Dict[str, Any],
    ) -> SegmentPoolResult:
        """Return empty result with provided diagnostics."""
        diagnostics.update(
            {
                "below_bridge_floor": 0,
                "pass_bridge_floor": 0,
                "collapsed_by_artist_key": 0,
                "selected_external": 0,
                "internal_connectors_candidates": 0,
                "internal_connectors_pass_gate": 0,
                "internal_connectors_selected": 0,
                "final": 0,
                "segment_pool_max": int(segment_pool_max),
            }
        )

        if config.diagnostics is not None:
            config.diagnostics.update(diagnostics)

        return SegmentPoolResult(
            candidates=[],
            artist_key_by_idx={},
            title_key_by_idx={},
            diagnostics=diagnostics,
        )

    @dataclass
    class _StructuralFilterResult:
        """Result of structural filtering phase."""
        candidates: List[int]
        artist_key_by_idx: Dict[int, str]
        title_key_by_idx: Dict[int, str]
        diagnostics: Dict[str, Any]

    def _apply_structural_filters(
        self,
        config: SegmentPoolConfig,
        universe_indices: List[int],
        pool_strategy: str,
        pier_a_artist_key: str,
        pier_b_artist_key: str,
        used_track_keys: Set[tuple[str, str]],
        seed_track_keys: Set[tuple[str, str]],
    ) -> _StructuralFilterResult:
        """Apply structural filters: used ids, allowed-set, policies, track_key collisions."""
        excluded_used = 0
        excluded_allowed = 0
        excluded_seed_artist = 0
        excluded_pier_artist = 0
        excluded_track_key = 0
        excluded_track_key_with_piers = 0

        artist_key_by_idx: Dict[int, str] = {}
        title_key_by_idx: Dict[int, str] = {}
        structural: List[int] = []

        for idx in universe_indices:
            i = int(idx)

            # Exclude already-used tracks
            if i in config.used_track_ids:
                excluded_used += 1
                continue

            # Exclude tracks not in allowed set
            if config.allowed_set is not None and i not in config.allowed_set:
                excluded_allowed += 1
                continue

            # Get identity keys
            keys = identity_keys_for_index(config.bundle, i)
            ak = keys.artist_key
            tk = keys.title_key
            artist_key_by_idx[i] = ak
            title_key_by_idx[i] = tk

            # Apply seed artist policy
            if (
                config.disallow_seed_artist_in_interiors
                and config.seed_artist_key
                and ak == config.seed_artist_key
            ):
                excluded_seed_artist += 1
                continue

            # Apply pier artist policy
            if config.disallow_pier_artists_in_interiors and ak in {
                pier_a_artist_key,
                pier_b_artist_key,
            }:
                excluded_pier_artist += 1
                continue

            # Apply track key collision check
            if keys.track_key in used_track_keys:
                excluded_track_key += 1
                if keys.track_key in seed_track_keys:
                    excluded_track_key_with_piers += 1
                continue

            structural.append(i)

        diagnostics = {
            "pool_strategy": str(pool_strategy),
            "base_universe": int(len(config.universe_indices)),
            "excluded_used_track_ids": int(excluded_used),
            "excluded_allowed_set": int(excluded_allowed),
            "excluded_seed_artist_policy": int(excluded_seed_artist),
            "excluded_pier_artist_policy": int(excluded_pier_artist),
            "excluded_track_key_collision": int(excluded_track_key),
            "excluded_track_key_collision_with_piers": int(
                excluded_track_key_with_piers
            ),
            "eligible_after_structural": int(len(structural)),
        }

        return self._StructuralFilterResult(
            candidates=structural,
            artist_key_by_idx=artist_key_by_idx,
            title_key_by_idx=title_key_by_idx,
            diagnostics=diagnostics,
        )

    @dataclass
    class _BridgeScoreResult:
        """Result of bridge scoring phase."""
        passing_candidates: List[int]
        passing_sorted: List[int]
        bridge_sim: Dict[int, float]
        diagnostics: Dict[str, Any]

    def _compute_bridge_scores(
        self,
        config: SegmentPoolConfig,
        candidates: List[int],
        diagnostics: Dict[str, Any],
    ) -> _BridgeScoreResult:
        """Compute bridge scores and apply bridge floor gate."""
        # Compute similarity to both piers
        sim_to_a = np.dot(config.X_full_norm, config.X_full_norm[config.pier_a])
        sim_to_b = np.dot(config.X_full_norm, config.X_full_norm[config.pier_b])

        below_bridge_floor = 0
        passing: List[int] = []
        bridge_sim: Dict[int, float] = {}

        for i in candidates:
            sim_a = float(sim_to_a[i])
            sim_b = float(sim_to_b[i])

            # Bridge floor gate: both similarities must pass
            if min(sim_a, sim_b) < float(config.bridge_floor):
                below_bridge_floor += 1
                continue

            bridge_sim[i] = self._compute_bridge_score(sim_a, sim_b, config)
            passing.append(i)

        # Sort by bridge score (descending), then by index (ascending)
        passing_sorted = sorted(
            passing, key=lambda i: (-float(bridge_sim.get(i, 0.0)), int(i))
        )

        diagnostics["below_bridge_floor"] = int(below_bridge_floor)
        diagnostics["pass_bridge_floor"] = int(len(passing))
        diagnostics["bridge_score_mode"] = (
            "experimental"
            if config.experiment_bridge_scoring_enabled
            else "hmean"
        )
        if config.experiment_bridge_scoring_enabled:
            min_weight = max(0.0, min(1.0, float(config.experiment_bridge_min_weight)))
            balance_weight = max(
                0.0,
                min(1.0 - min_weight, float(config.experiment_bridge_balance_weight)),
            )
            diagnostics["bridge_score_weights"] = {
                "hmean": float(max(0.0, 1.0 - min_weight - balance_weight)),
                "min": float(min_weight),
                "balance": float(balance_weight),
            }

        return self._BridgeScoreResult(
            passing_candidates=passing,
            passing_sorted=passing_sorted,
            bridge_sim=bridge_sim,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _step_fraction(step_idx: int, steps: int) -> float:
        if steps <= 0:
            return 0.0
        return float(step_idx + 1) / float(steps + 1)

    @classmethod
    def _progress_target(cls, step_idx: int, steps: int, *, shape: str, enabled: bool) -> float:
        if not enabled:
            return cls._step_fraction(step_idx, steps)
        shape = str(shape or "linear").strip().lower()
        if shape not in {"linear", "arc"}:
            shape = "linear"
        frac = cls._step_fraction(step_idx, steps)
        if shape == "arc":
            return 0.5 - 0.5 * math.cos(math.pi * frac)
        return frac

    def _build_dj_union_pool(
        self,
        config: SegmentPoolConfig,
    ) -> tuple[List[int], Dict[str, Any]]:
        diag: Dict[str, Any] = {
            "dj_pool_strategy": "dj_union",
            "dj_pool_k_local": int(config.pool_k_local),
            "dj_pool_k_toward": int(config.pool_k_toward),
            "dj_pool_k_genre": int(config.pool_k_genre),
            "dj_pool_k_union_max": int(config.pool_k_union_max),
            "dj_pool_step_stride": int(config.pool_step_stride),
        }

        available = [int(i) for i in config.universe_indices if int(i) not in config.used_track_ids]
        if config.allowed_set is not None:
            allowed = set(int(i) for i in config.allowed_set)
            available = [int(i) for i in available if int(i) in allowed]

        diag["dj_pool_available"] = int(len(available))
        if not available:
            return [], diag

        cand_indices = np.array(available, dtype=int)
        cand_mat = config.X_full_norm[cand_indices]
        vec_a = config.X_full_norm[config.pier_a]
        vec_b = config.X_full_norm[config.pier_b]
        sim_a = np.dot(cand_mat, vec_a)
        sim_b = np.dot(cand_mat, vec_b)

        k_local = max(0, int(config.pool_k_local))
        k_toward = max(0, int(config.pool_k_toward))
        k_genre = max(0, int(config.pool_k_genre))
        union_max = max(0, int(config.pool_k_union_max))

        local_indices: List[int] = []
        if k_local > 0:
            k_a = max(1, k_local // 2) if k_local > 1 else 1
            k_b = k_local - k_a
            order_a = np.argsort(-sim_a)[:k_a]
            local_indices.extend([int(cand_indices[i]) for i in order_a])
            if k_b > 0:
                order_b = np.argsort(-sim_b)[:k_b]
                local_indices.extend([int(cand_indices[i]) for i in order_b])

        toward_indices: List[int] = []
        if k_toward > 0 and int(config.interior_length) > 0:
            stride = max(1, int(config.pool_step_stride))
            cache = config.pooling_cache if config.pool_cache_enabled else None
            toward_cache = None
            if cache is not None:
                toward_cache = cache.setdefault("toward", {})
            steps = int(config.interior_length)
            for step in range(0, steps, stride):
                if toward_cache is not None and step in toward_cache:
                    toward_indices.extend(toward_cache[step])
                    continue
                t = self._progress_target(
                    step,
                    steps,
                    shape=str(config.progress_arc_shape),
                    enabled=bool(config.progress_arc_enabled),
                )
                target = (1.0 - t) * vec_a + t * vec_b
                norm = float(np.linalg.norm(target))
                if not math.isfinite(norm) or norm <= 1e-12:
                    continue
                target = target / norm
                sims = np.dot(cand_mat, target)
                order = np.argsort(-sims)[:k_toward]
                picked = [int(cand_indices[i]) for i in order]
                toward_indices.extend(picked)
                if toward_cache is not None:
                    toward_cache[step] = picked

        genre_indices: List[int] = []
        # Task A: Hard diagnostic check for genre pool prerequisites
        import logging
        logger = logging.getLogger(__name__)
        if k_genre > 0:
            has_X = config.X_genre_norm is not None
            has_targets = config.genre_targets is not None
            has_targets_len = len(config.genre_targets) if config.genre_targets else 0
            interior = int(config.interior_length)

            # Warn if prerequisites are missing
            if not has_X:
                logger.warning("[Genre Pool] k_genre=%d but X_genre_norm is None - genre pool disabled", k_genre)
            elif not has_targets or has_targets_len == 0:
                logger.warning("[Genre Pool] k_genre=%d X_genre_norm=present but genre_targets=%s (len=%d) interior=%d - genre pool disabled",
                              k_genre, has_targets, has_targets_len, interior)

        if (
            k_genre > 0
            and config.X_genre_norm is not None
            and config.genre_targets is not None
            and int(config.interior_length) > 0
        ):
            stride = max(1, int(config.pool_step_stride))
            cache = config.pooling_cache if config.pool_cache_enabled else None
            genre_cache = None
            if cache is not None:
                genre_cache = cache.setdefault("genre", {})
            steps = int(config.interior_length)
            # Use IDF-weighted matrix if available (Phase 2)
            X_genre_for_pooling = config.X_genre_norm_idf if config.X_genre_norm_idf is not None else config.X_genre_norm
            cand_genre = X_genre_for_pooling[cand_indices]
            for step in range(0, steps, stride):
                if step >= len(config.genre_targets):
                    break
                if genre_cache is not None and step in genre_cache:
                    genre_indices.extend(genre_cache[step])
                    continue
                target = config.genre_targets[step]
                sims = np.dot(cand_genre, target)
                order = np.argsort(-sims)[:k_genre]
                picked = [int(cand_indices[i]) for i in order]
                genre_indices.extend(picked)
                if genre_cache is not None:
                    genre_cache[step] = picked

        diag["dj_pool_source_local"] = int(len(local_indices))
        diag["dj_pool_source_toward"] = int(len(toward_indices))
        diag["dj_pool_source_genre"] = int(len(genre_indices))

        # Task B: Always-on pool composition summary
        local_set = set(local_indices)
        toward_set = set(toward_indices)
        genre_set = set(genre_indices)
        overlap_local_toward = len(local_set & toward_set)
        overlap_local_genre = len(local_set & genre_set)
        overlap_toward_genre = len(toward_set & genre_set)

        logger.info("  DJ union pool: S1_local=%d S2_toward=%d S3_genre=%d | overlaps: L∩T=%d L∩G=%d T∩G=%d",
                   len(local_indices), len(toward_indices), len(genre_indices),
                   overlap_local_toward, overlap_local_genre, overlap_toward_genre)

        # Phase 3 fix: Verbose logging for debugging genre pool issues
        if hasattr(config, "pool_verbose") and bool(config.pool_verbose):
            # Log comprehensive breakdown (INFO level for visibility)
            logger.info(
                "[DJ Pool Debug] Segment pool breakdown:"
            )
            logger.info(
                "  Raw sizes: S1_local=%d S2_toward=%d S3_genre=%d",
                len(local_indices), len(toward_indices), len(genre_indices)
            )
            logger.info(
                "  Config: k_local=%d k_toward=%d k_genre=%d union_max=%d stride=%d",
                k_local, k_toward, k_genre, union_max, max(1, int(config.pool_step_stride))
            )
            logger.info(
                "  Prerequisites: has_X_genre=%s has_X_genre_idf=%s has_targets=%s interior_len=%d",
                config.X_genre_norm is not None,
                config.X_genre_norm_idf is not None,
                config.genre_targets is not None,
                int(config.interior_length)
            )
            logger.info(
                "  Overlaps: local∩genre=%d toward∩genre=%d local∩toward=%d",
                overlap_local_genre, overlap_toward_genre, overlap_local_toward
            )

            # If genre pool is empty, explain why
            if len(genre_indices) == 0:
                if k_genre <= 0:
                    logger.info("  Genre pool empty: k_genre=%d (disabled)", k_genre)
                elif config.X_genre_norm is None:
                    logger.info("  Genre pool empty: X_genre_norm is None")
                elif config.genre_targets is None or len(config.genre_targets) == 0:
                    logger.info("  Genre pool empty: genre_targets is None or empty (len=%s)",
                               len(config.genre_targets) if config.genre_targets else 0)
                elif int(config.interior_length) <= 0:
                    logger.info("  Genre pool empty: interior_length=%d", int(config.interior_length))
                else:
                    logger.info("  Genre pool empty: unknown reason (all prerequisites met but no candidates returned)")

            # Log which genre matrix is being used
            if config.X_genre_norm_idf is not None:
                logger.info("  Genre pooling matrix: X_genre_norm_idf (IDF-weighted)")
            elif config.X_genre_norm is not None:
                logger.info("  Genre pooling matrix: X_genre_norm (normalized, no IDF)")
            else:
                logger.info("  Genre pooling matrix: None")

        combined_raw = local_indices + toward_indices + genre_indices
        combined = list(dict.fromkeys(combined_raw))
        diag["dj_pool_union_raw"] = int(len(combined_raw))
        diag["dj_pool_union_deduped"] = int(len(combined))

        logger.info("  DJ union dedup: raw=%d → deduped=%d (%.1f%% reduction)",
                   len(combined_raw), len(combined),
                   100.0 * (1.0 - len(combined) / max(1, len(combined_raw))))

        if union_max > 0 and len(combined) > union_max:
            sim_a_all = np.dot(config.X_full_norm, vec_a)
            sim_b_all = np.dot(config.X_full_norm, vec_b)
            scores = []
            for idx in combined:
                denom = float(sim_a_all[idx] + sim_b_all[idx])
                if denom <= 1e-9:
                    score = 0.0
                else:
                    score = (2.0 * float(sim_a_all[idx]) * float(sim_b_all[idx])) / denom
                scores.append(score)
            order = np.argsort(-np.array(scores))[:union_max]
            combined = [combined[int(i)] for i in order]
        diag["dj_pool_union_capped"] = int(len(combined))

        combined_set = set(combined)
        diag["dj_pool_contrib_local"] = int(len([i for i in local_indices if i in combined_set]))
        diag["dj_pool_contrib_toward"] = int(len([i for i in toward_indices if i in combined_set]))
        diag["dj_pool_contrib_genre"] = int(len([i for i in genre_indices if i in combined_set]))

        if config.pooling_cache is not None:
            config.pooling_cache["dj_pool_sources"] = {
                "local": set(int(i) for i in local_indices),
                "toward": set(int(i) for i in toward_indices),
                "genre": set(int(i) for i in genre_indices),
                "union": combined_set,
            }

        return combined, diag

    @dataclass
    class _InternalConnectorResult:
        """Result of internal connector processing."""
        internal_selected: List[int]
        internal_ranked: List[tuple[float, int]]
        diagnostics: Dict[str, Any]

    def _process_internal_connectors(
        self,
        config: SegmentPoolConfig,
        pier_a_artist_key: str,
        pier_b_artist_key: str,
        used_track_keys: Set[tuple[str, str]],
        artist_key_by_idx: Dict[int, str],
        title_key_by_idx: Dict[int, str],
    ) -> _InternalConnectorResult:
        """Process internal connectors (optional priority candidates)."""
        internal_candidates = 0
        internal_pass_gate = 0
        internal_ranked: List[tuple[float, int]] = []

        if not config.internal_connectors:
            return self._InternalConnectorResult(
                internal_selected=[],
                internal_ranked=[],
                diagnostics={
                    "internal_connectors_candidates": 0,
                    "internal_connectors_pass_gate": 0,
                },
            )

        # Compute similarity to both piers
        sim_to_a = np.dot(config.X_full_norm, config.X_full_norm[config.pier_a])
        sim_to_b = np.dot(config.X_full_norm, config.X_full_norm[config.pier_b])

        for idx in config.internal_connectors:
            i = int(idx)

            # Apply same structural filters as external candidates
            if i in config.used_track_ids:
                continue
            if config.allowed_set is not None and i not in config.allowed_set:
                continue

            keys = identity_keys_for_index(config.bundle, i)
            ak = keys.artist_key
            tk = keys.title_key
            artist_key_by_idx[i] = ak
            title_key_by_idx[i] = tk

            if (
                config.disallow_seed_artist_in_interiors
                and config.seed_artist_key
                and ak == config.seed_artist_key
            ):
                continue
            if config.disallow_pier_artists_in_interiors and ak in {
                pier_a_artist_key,
                pier_b_artist_key,
            }:
                continue
            if keys.track_key in used_track_keys:
                continue

            internal_candidates += 1

            # Compute bridge score
            sim_a = float(sim_to_a[i])
            sim_b = float(sim_to_b[i])
            if min(sim_a, sim_b) < float(config.bridge_floor):
                continue

            internal_pass_gate += 1
            score = self._compute_bridge_score(sim_a, sim_b, config)
            internal_ranked.append((float(score), i))

        # Sort internal connectors by score
        internal_ranked.sort(key=lambda t: (-t[0], t[1]))

        return self._InternalConnectorResult(
            internal_selected=[],  # Will be populated in selection phase
            internal_ranked=internal_ranked,
            diagnostics={
                "internal_connectors_candidates": int(internal_candidates),
                "internal_connectors_pass_gate": int(internal_pass_gate),
            },
        )

    @dataclass
    class _FinalSelectionResult:
        """Result of final candidate selection."""
        candidates: List[int]
        diagnostics: Dict[str, Any]

    def _select_final_candidates(
        self,
        config: SegmentPoolConfig,
        passing_sorted: List[int],
        bridge_sim: Dict[int, float],
        internal_result: _InternalConnectorResult,
        artist_key_by_idx: Dict[int, str],
        segment_pool_max: int,
    ) -> _FinalSelectionResult:
        """Select final candidates with 1-per-artist constraint."""
        selected_external: List[int] = []
        internal_selected: List[int] = []
        collapsed_by_artist = 0

        if config.internal_connector_priority:
            # Select internal connectors first, then fill from external candidates
            used_artists: Set[str] = set()

            # Select internal connectors
            cap = (
                int(config.internal_connector_cap)
                if int(config.internal_connector_cap) > 0
                else len(internal_result.internal_ranked)
            )
            for _score, i in internal_result.internal_ranked:
                ak = artist_key_by_idx.get(i) or identity_keys_for_index(
                    config.bundle, i
                ).artist_key
                if ak in used_artists:
                    continue
                internal_selected.append(i)
                used_artists.add(ak)
                if len(internal_selected) >= cap:
                    break

            # Fill with external candidates
            for i in passing_sorted:
                ak = artist_key_by_idx.get(i) or identity_keys_for_index(
                    config.bundle, i
                ).artist_key
                if ak in used_artists:
                    collapsed_by_artist += 1
                    continue
                used_artists.add(ak)
                selected_external.append(i)
                if len(selected_external) >= int(segment_pool_max):
                    break

            combined = list(dict.fromkeys(internal_selected + selected_external))

        else:
            # Select external first, then add internal connectors up to cap
            used_artists = set()

            for i in passing_sorted:
                ak = artist_key_by_idx.get(i) or identity_keys_for_index(
                    config.bundle, i
                ).artist_key
                if ak in used_artists:
                    collapsed_by_artist += 1
                    continue
                used_artists.add(ak)
                selected_external.append(i)
                if len(selected_external) >= int(segment_pool_max):
                    break

            cap = (
                int(config.internal_connector_cap)
                if int(config.internal_connector_cap) > 0
                else len(internal_result.internal_ranked)
            )
            for _score, i in internal_result.internal_ranked:
                ak = artist_key_by_idx.get(i) or identity_keys_for_index(
                    config.bundle, i
                ).artist_key
                if ak in used_artists:
                    continue
                internal_selected.append(i)
                used_artists.add(ak)
                if len(internal_selected) >= cap:
                    break

            combined = list(dict.fromkeys(selected_external + internal_selected))

        diagnostics = {
            "collapsed_by_artist_key": int(collapsed_by_artist),
            "selected_external": int(len(selected_external)),
            "internal_connectors_selected": int(len(internal_selected)),
            "final": int(len(combined)),
        }

        return self._FinalSelectionResult(
            candidates=combined,
            diagnostics=diagnostics,
        )
