from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayeredGenreComponents:
    family_affinity: float
    niche_similarity: float
    facet_alignment: float
    bridge_permission: float
    broad_only_penalty: float
    unexplained_jump_penalty: float
    source_quality: float = 1.0


@dataclass(frozen=True)
class LayeredGenreDecision:
    admitted: bool
    score: float
    components: LayeredGenreComponents
    reason: str


@dataclass(frozen=True)
class LayeredGenreWeights:
    family: float
    leaf: float
    facet: float
    bridge: float
    broad_only: float
    unexplained_jump: float
    admit_threshold: float
    require_leaf_for_broad: bool


@dataclass(frozen=True)
class LayeredTransitionComponents:
    local_family_continuity: float
    local_leaf_continuity: float
    bridge_edge_bonus: float
    facet_continuity: float
    sonic_similarity: float
    transition_quality: float
    unexplained_family_jump_penalty: float
    bridge_evidence_penalty: float


@dataclass(frozen=True)
class LayeredTransitionDecision:
    explained: bool
    score: float
    components: LayeredTransitionComponents
    reason: str


@dataclass(frozen=True)
class LayeredTransitionThresholds:
    sonic_similarity: float
    facet_continuity: float
    transition_quality: float


MODE_WEIGHTS: dict[str, LayeredGenreWeights] = {
    "strict": LayeredGenreWeights(
        family=0.18,
        leaf=0.58,
        facet=0.14,
        bridge=0.05,
        broad_only=0.65,
        unexplained_jump=0.70,
        admit_threshold=0.50,
        require_leaf_for_broad=True,
    ),
    "narrow": LayeredGenreWeights(
        family=0.24,
        leaf=0.46,
        facet=0.16,
        bridge=0.12,
        broad_only=0.50,
        unexplained_jump=0.65,
        admit_threshold=0.42,
        require_leaf_for_broad=True,
    ),
    "dynamic": LayeredGenreWeights(
        family=0.30,
        leaf=0.30,
        facet=0.20,
        bridge=0.24,
        broad_only=0.35,
        unexplained_jump=0.55,
        admit_threshold=0.34,
        require_leaf_for_broad=False,
    ),
    "discover": LayeredGenreWeights(
        family=0.28,
        leaf=0.22,
        facet=0.24,
        bridge=0.26,
        broad_only=0.20,
        unexplained_jump=0.45,
        admit_threshold=0.25,
        require_leaf_for_broad=False,
    ),
    "off": LayeredGenreWeights(
        family=0.0,
        leaf=0.0,
        facet=0.0,
        bridge=0.0,
        broad_only=0.0,
        unexplained_jump=0.0,
        admit_threshold=0.0,
        require_leaf_for_broad=False,
    ),
}

TRANSITION_THRESHOLDS: dict[str, LayeredTransitionThresholds] = {
    "strict": LayeredTransitionThresholds(
        sonic_similarity=0.75,
        facet_continuity=0.50,
        transition_quality=0.75,
    ),
    "narrow": LayeredTransitionThresholds(
        sonic_similarity=0.65,
        facet_continuity=0.35,
        transition_quality=0.65,
    ),
    "dynamic": LayeredTransitionThresholds(
        sonic_similarity=0.50,
        facet_continuity=0.25,
        transition_quality=0.50,
    ),
    "discover": LayeredTransitionThresholds(
        sonic_similarity=0.35,
        facet_continuity=0.15,
        transition_quality=0.35,
    ),
    "off": LayeredTransitionThresholds(
        sonic_similarity=0.0,
        facet_continuity=0.0,
        transition_quality=0.0,
    ),
}


def score_layered_candidate(
    *,
    seed_leaf: np.ndarray,
    candidate_leaf: np.ndarray,
    seed_family: np.ndarray,
    candidate_family: np.ndarray,
    seed_bridge: np.ndarray,
    candidate_bridge: np.ndarray,
    seed_facet: np.ndarray,
    candidate_facet: np.ndarray,
    mode: str,
    source_quality: float = 1.0,
) -> LayeredGenreDecision:
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["dynamic"])
    if mode == "off":
        return LayeredGenreDecision(
            admitted=True,
            score=0.0,
            components=LayeredGenreComponents(
                family_affinity=0.0,
                niche_similarity=0.0,
                facet_alignment=0.0,
                bridge_permission=0.0,
                broad_only_penalty=0.0,
                unexplained_jump_penalty=0.0,
                source_quality=source_quality,
            ),
            reason="genre_off",
        )

    family_affinity = _cosine(seed_family, candidate_family)
    niche_similarity = _cosine(seed_leaf, candidate_leaf)
    facet_alignment = _cosine(seed_facet, candidate_facet)
    bridge_permission = max(
        _binary_overlap(seed_bridge, candidate_leaf),
        _binary_overlap(candidate_bridge, seed_leaf),
    )
    broad_only_penalty = 1.0 if family_affinity > 0.0 and niche_similarity == 0.0 and bridge_permission == 0.0 else 0.0
    unexplained_jump_penalty = 1.0 if family_affinity == 0.0 and niche_similarity == 0.0 and bridge_permission == 0.0 else 0.0
    components = LayeredGenreComponents(
        family_affinity=family_affinity,
        niche_similarity=niche_similarity,
        facet_alignment=facet_alignment,
        bridge_permission=bridge_permission,
        broad_only_penalty=broad_only_penalty,
        unexplained_jump_penalty=unexplained_jump_penalty,
        source_quality=source_quality,
    )
    score = (
        weights.family * family_affinity
        + weights.leaf * niche_similarity
        + weights.facet * facet_alignment
        + weights.bridge * bridge_permission
        - weights.broad_only * broad_only_penalty
        - weights.unexplained_jump * unexplained_jump_penalty
    ) * max(0.0, min(1.0, source_quality))

    admitted, reason = _decision_reason(
        components=components,
        score=score,
        weights=weights,
    )
    return LayeredGenreDecision(
        admitted=admitted,
        score=float(score),
        components=components,
        reason=reason,
    )


def layered_decision_to_diagnostics(decision: LayeredGenreDecision) -> dict[str, object]:
    components = decision.components
    return {
        "admitted": bool(decision.admitted),
        "score": _round_score(decision.score),
        "reason": decision.reason,
        "family_affinity": _round_score(components.family_affinity),
        "niche_similarity": _round_score(components.niche_similarity),
        "facet_alignment": _round_score(components.facet_alignment),
        "bridge_permission": _round_score(components.bridge_permission),
        "broad_only_penalty": _round_score(components.broad_only_penalty),
        "unexplained_jump_penalty": _round_score(components.unexplained_jump_penalty),
        "source_quality": _round_score(components.source_quality),
    }


def score_layered_transition(
    *,
    from_leaf: np.ndarray,
    to_leaf: np.ndarray,
    from_family: np.ndarray,
    to_family: np.ndarray,
    from_bridge: np.ndarray,
    to_bridge: np.ndarray,
    from_facet: np.ndarray,
    to_facet: np.ndarray,
    sonic_similarity: float,
    transition_quality: float,
    mode: str,
) -> LayeredTransitionDecision:
    if mode == "off":
        components = LayeredTransitionComponents(
            local_family_continuity=0.0,
            local_leaf_continuity=0.0,
            bridge_edge_bonus=0.0,
            facet_continuity=0.0,
            sonic_similarity=0.0,
            transition_quality=0.0,
            unexplained_family_jump_penalty=0.0,
            bridge_evidence_penalty=0.0,
        )
        return LayeredTransitionDecision(
            explained=True,
            score=0.0,
            components=components,
            reason="genre_off",
        )

    thresholds = TRANSITION_THRESHOLDS.get(mode, TRANSITION_THRESHOLDS["dynamic"])
    family_continuity = _cosine(from_family, to_family)
    leaf_continuity = _cosine(from_leaf, to_leaf)
    facet_continuity = _cosine(from_facet, to_facet)
    bridge_bonus = max(
        _binary_overlap(from_bridge, to_leaf),
        _binary_overlap(to_bridge, from_leaf),
    )
    unexplained_jump = (
        1.0
        if family_continuity == 0.0 and leaf_continuity == 0.0 and bridge_bonus == 0.0
        else 0.0
    )
    bridge_evidence_penalty = 0.0
    if bridge_bonus > 0.0 and (
        float(sonic_similarity) < thresholds.sonic_similarity
        or facet_continuity < thresholds.facet_continuity
        or float(transition_quality) < thresholds.transition_quality
    ):
        bridge_evidence_penalty = 1.0

    components = LayeredTransitionComponents(
        local_family_continuity=family_continuity,
        local_leaf_continuity=leaf_continuity,
        bridge_edge_bonus=bridge_bonus,
        facet_continuity=facet_continuity,
        sonic_similarity=max(0.0, min(1.0, float(sonic_similarity))),
        transition_quality=max(0.0, min(1.0, float(transition_quality))),
        unexplained_family_jump_penalty=unexplained_jump,
        bridge_evidence_penalty=bridge_evidence_penalty,
    )
    score = (
        0.22 * family_continuity
        + 0.34 * leaf_continuity
        + 0.24 * bridge_bonus
        + 0.18 * facet_continuity
        + 0.04 * components.sonic_similarity
        + 0.04 * components.transition_quality
        - 0.65 * unexplained_jump
        - 0.35 * bridge_evidence_penalty
    )
    explained, reason = _transition_reason(components)
    return LayeredTransitionDecision(
        explained=explained,
        score=float(score),
        components=components,
        reason=reason,
    )


def _decision_reason(
    *,
    components: LayeredGenreComponents,
    score: float,
    weights: LayeredGenreWeights,
) -> tuple[bool, str]:
    if components.unexplained_jump_penalty > 0.0:
        return False, "unexplained_family_jump"
    if components.broad_only_penalty > 0.0 and weights.require_leaf_for_broad:
        return False, "broad_only_without_leaf_support"
    if components.bridge_permission > 0.0 and components.facet_alignment > 0.0:
        return True, "bridge_supported"
    if score >= weights.admit_threshold:
        return True, "layered_score_threshold"
    return False, "below_layered_score_threshold"


def _transition_reason(components: LayeredTransitionComponents) -> tuple[bool, str]:
    if components.unexplained_family_jump_penalty > 0.0:
        return False, "unexplained_family_jump"
    if components.bridge_edge_bonus > 0.0 and components.bridge_evidence_penalty > 0.0:
        return False, "bridge_evidence_insufficient"
    if components.bridge_edge_bonus > 0.0:
        return True, "bridge_supported"
    if components.local_leaf_continuity > 0.0:
        return True, "leaf_continuity"
    if components.local_family_continuity > 0.0 and components.facet_continuity > 0.0:
        return True, "family_facet_continuity"
    return False, "below_layered_transition_threshold"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _binary_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size == 0 or right.size == 0 or left.shape != right.shape:
        return 0.0
    return 1.0 if bool(np.any((left > 0) & (right > 0))) else 0.0


def _round_score(value: float) -> float:
    return round(float(value), 6)
