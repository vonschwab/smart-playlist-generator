import numpy as np


def test_layered_genre_score_rewards_leaf_family_and_facet_alignment():
    from src.playlist.layered_genre_scoring import score_layered_candidate

    decision = score_layered_candidate(
        seed_leaf=np.array([1.0, 0.0], dtype=np.float32),
        candidate_leaf=np.array([0.8, 0.0], dtype=np.float32),
        seed_family=np.array([1.0, 0.0], dtype=np.float32),
        candidate_family=np.array([0.7, 0.0], dtype=np.float32),
        seed_bridge=np.array([0.0, 0.0], dtype=np.float32),
        candidate_bridge=np.array([0.0, 0.0], dtype=np.float32),
        seed_facet=np.array([1.0, 0.0], dtype=np.float32),
        candidate_facet=np.array([0.9, 0.0], dtype=np.float32),
        mode="strict",
    )

    assert decision.components.niche_similarity > 0.99
    assert decision.components.family_affinity > 0.99
    assert decision.components.facet_alignment > 0.99
    assert decision.components.broad_only_penalty == 0.0
    assert decision.score > 0.80
    assert decision.admitted is True


def test_layered_genre_score_penalizes_broad_only_strict_candidate():
    from src.playlist.layered_genre_scoring import score_layered_candidate

    decision = score_layered_candidate(
        seed_leaf=np.array([1.0, 0.0], dtype=np.float32),
        candidate_leaf=np.array([0.0, 0.0], dtype=np.float32),
        seed_family=np.array([1.0, 0.0], dtype=np.float32),
        candidate_family=np.array([1.0, 0.0], dtype=np.float32),
        seed_bridge=np.array([0.0, 0.0], dtype=np.float32),
        candidate_bridge=np.array([0.0, 0.0], dtype=np.float32),
        seed_facet=np.array([0.0, 0.0], dtype=np.float32),
        candidate_facet=np.array([0.0, 0.0], dtype=np.float32),
        mode="strict",
    )

    assert decision.components.family_affinity > 0.99
    assert decision.components.niche_similarity == 0.0
    assert decision.components.broad_only_penalty == 1.0
    assert decision.admitted is False
    assert decision.reason == "broad_only_without_leaf_support"


def test_layered_genre_score_allows_dynamic_bridge_with_facet_support():
    from src.playlist.layered_genre_scoring import score_layered_candidate

    decision = score_layered_candidate(
        seed_leaf=np.array([1.0, 0.0], dtype=np.float32),
        candidate_leaf=np.array([0.0, 1.0], dtype=np.float32),
        seed_family=np.array([1.0], dtype=np.float32),
        candidate_family=np.array([1.0], dtype=np.float32),
        seed_bridge=np.array([0.0, 1.0], dtype=np.float32),
        candidate_bridge=np.array([1.0, 0.0], dtype=np.float32),
        seed_facet=np.array([1.0, 0.0], dtype=np.float32),
        candidate_facet=np.array([0.8, 0.0], dtype=np.float32),
        mode="dynamic",
    )

    assert decision.components.niche_similarity == 0.0
    assert decision.components.bridge_permission > 0.0
    assert decision.components.facet_alignment > 0.99
    assert decision.admitted is True
    assert decision.reason == "bridge_supported"


def test_layered_genre_score_rejects_unexplained_family_jump():
    from src.playlist.layered_genre_scoring import score_layered_candidate

    decision = score_layered_candidate(
        seed_leaf=np.array([1.0, 0.0], dtype=np.float32),
        candidate_leaf=np.array([0.0, 1.0], dtype=np.float32),
        seed_family=np.array([1.0, 0.0], dtype=np.float32),
        candidate_family=np.array([0.0, 1.0], dtype=np.float32),
        seed_bridge=np.array([0.0, 0.0], dtype=np.float32),
        candidate_bridge=np.array([0.0, 0.0], dtype=np.float32),
        seed_facet=np.array([1.0, 0.0], dtype=np.float32),
        candidate_facet=np.array([0.0, 1.0], dtype=np.float32),
        mode="dynamic",
    )

    assert decision.components.family_affinity == 0.0
    assert decision.components.unexplained_jump_penalty == 1.0
    assert decision.admitted is False
    assert decision.reason == "unexplained_family_jump"


def test_layered_genre_decision_to_diagnostics_is_json_stable():
    from src.playlist.layered_genre_scoring import score_layered_candidate, layered_decision_to_diagnostics

    decision = score_layered_candidate(
        seed_leaf=np.array([1.0], dtype=np.float32),
        candidate_leaf=np.array([1.0], dtype=np.float32),
        seed_family=np.array([1.0], dtype=np.float32),
        candidate_family=np.array([1.0], dtype=np.float32),
        seed_bridge=np.array([0.0], dtype=np.float32),
        candidate_bridge=np.array([0.0], dtype=np.float32),
        seed_facet=np.array([0.0], dtype=np.float32),
        candidate_facet=np.array([0.0], dtype=np.float32),
        mode="narrow",
        source_quality=0.75,
    )

    diagnostic = layered_decision_to_diagnostics(decision)

    assert diagnostic == {
        "admitted": True,
        "score": diagnostic["score"],
        "reason": "layered_score_threshold",
        "family_affinity": diagnostic["family_affinity"],
        "niche_similarity": diagnostic["niche_similarity"],
        "facet_alignment": 0.0,
        "bridge_permission": 0.0,
        "broad_only_penalty": 0.0,
        "unexplained_jump_penalty": 0.0,
        "source_quality": 0.75,
    }
    assert diagnostic["score"] > 0.5
    assert diagnostic["family_affinity"] == 1.0
    assert diagnostic["niche_similarity"] == 1.0


def test_layered_transition_score_rewards_leaf_and_facet_continuity():
    from src.playlist.layered_genre_scoring import score_layered_transition

    decision = score_layered_transition(
        from_leaf=np.array([1.0, 0.0], dtype=np.float32),
        to_leaf=np.array([0.9, 0.0], dtype=np.float32),
        from_family=np.array([1.0], dtype=np.float32),
        to_family=np.array([1.0], dtype=np.float32),
        from_bridge=np.array([0.0, 0.0], dtype=np.float32),
        to_bridge=np.array([0.0, 0.0], dtype=np.float32),
        from_facet=np.array([1.0], dtype=np.float32),
        to_facet=np.array([0.8], dtype=np.float32),
        sonic_similarity=0.80,
        transition_quality=0.82,
        mode="narrow",
    )

    assert decision.explained is True
    assert decision.reason == "leaf_continuity"
    assert decision.components.local_leaf_continuity > 0.99
    assert decision.score > 0.70


def test_layered_transition_score_requires_bridge_evidence_quality():
    from src.playlist.layered_genre_scoring import score_layered_transition

    decision = score_layered_transition(
        from_leaf=np.array([1.0, 0.0], dtype=np.float32),
        to_leaf=np.array([0.0, 1.0], dtype=np.float32),
        from_family=np.array([1.0], dtype=np.float32),
        to_family=np.array([1.0], dtype=np.float32),
        from_bridge=np.array([0.0, 1.0], dtype=np.float32),
        to_bridge=np.array([1.0, 0.0], dtype=np.float32),
        from_facet=np.array([1.0], dtype=np.float32),
        to_facet=np.array([1.0], dtype=np.float32),
        sonic_similarity=0.20,
        transition_quality=0.90,
        mode="dynamic",
    )

    assert decision.components.bridge_edge_bonus > 0.0
    assert decision.explained is False
    assert decision.reason == "bridge_evidence_insufficient"


def test_layered_transition_score_explains_supported_bridge():
    from src.playlist.layered_genre_scoring import score_layered_transition

    decision = score_layered_transition(
        from_leaf=np.array([1.0, 0.0], dtype=np.float32),
        to_leaf=np.array([0.0, 1.0], dtype=np.float32),
        from_family=np.array([1.0], dtype=np.float32),
        to_family=np.array([1.0], dtype=np.float32),
        from_bridge=np.array([0.0, 1.0], dtype=np.float32),
        to_bridge=np.array([1.0, 0.0], dtype=np.float32),
        from_facet=np.array([1.0], dtype=np.float32),
        to_facet=np.array([0.9], dtype=np.float32),
        sonic_similarity=0.72,
        transition_quality=0.74,
        mode="dynamic",
    )

    assert decision.explained is True
    assert decision.reason == "bridge_supported"
    assert decision.score > 0.50


def test_layered_transition_score_penalizes_unexplained_jump():
    from src.playlist.layered_genre_scoring import score_layered_transition

    decision = score_layered_transition(
        from_leaf=np.array([1.0, 0.0], dtype=np.float32),
        to_leaf=np.array([0.0, 1.0], dtype=np.float32),
        from_family=np.array([1.0, 0.0], dtype=np.float32),
        to_family=np.array([0.0, 1.0], dtype=np.float32),
        from_bridge=np.array([0.0, 0.0], dtype=np.float32),
        to_bridge=np.array([0.0, 0.0], dtype=np.float32),
        from_facet=np.array([1.0], dtype=np.float32),
        to_facet=np.array([0.0], dtype=np.float32),
        sonic_similarity=0.80,
        transition_quality=0.80,
        mode="dynamic",
    )

    assert decision.explained is False
    assert decision.reason == "unexplained_family_jump"
    assert decision.components.unexplained_family_jump_penalty == 1.0
