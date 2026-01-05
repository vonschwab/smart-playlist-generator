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

        # Get endpoint artist keys for policy enforcement
        pier_a_artist_key = identity_keys_for_index(config.bundle, config.pier_a).artist_key
        pier_b_artist_key = identity_keys_for_index(config.bundle, config.pier_b).artist_key

        # Phase 1: Structural filtering
        structural_result = self._apply_structural_filters(
            config=config,
            pier_a_artist_key=pier_a_artist_key,
            pier_b_artist_key=pier_b_artist_key,
            used_track_keys=used_track_keys,
            seed_track_keys=seed_track_keys,
        )

        if not structural_result.candidates:
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

    def _empty_result(
        self,
        config: SegmentPoolConfig,
        segment_pool_max: int,
    ) -> SegmentPoolResult:
        """Return empty result when segment_pool_max <= 0."""
        diagnostics = {
            "pool_strategy": "segment_scored",
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

        for idx in config.universe_indices:
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
            "pool_strategy": "segment_scored",
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

            # Harmonic mean of similarities
            denom = sim_a + sim_b
            hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
            bridge_sim[i] = float(hmean)
            passing.append(i)

        # Sort by bridge score (descending), then by index (ascending)
        passing_sorted = sorted(
            passing, key=lambda i: (-float(bridge_sim.get(i, 0.0)), int(i))
        )

        diagnostics["below_bridge_floor"] = int(below_bridge_floor)
        diagnostics["pass_bridge_floor"] = int(len(passing))

        return self._BridgeScoreResult(
            passing_candidates=passing,
            passing_sorted=passing_sorted,
            bridge_sim=bridge_sim,
            diagnostics=diagnostics,
        )

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
            denom = sim_a + sim_b
            hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
            internal_ranked.append((float(hmean), i))

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
