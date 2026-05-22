"""Test for min_edge beam selection objective.

Tests the opt-in lexicographic selection that prefers the path with the highest
minimum edge (trans_score_in_beam), breaking ties by total score.
"""
from src.playlist.pier_bridge.beam import _select_best_beam_state


class FakeState:
    """Mock BeamState for testing."""
    def __init__(self, score, edge_scores):
        self.score = score
        self.edge_components = [{"trans_score_in_beam": v} for v in edge_scores]


def test_total_score_objective_picks_higher_total():
    """Default objective picks highest total score."""
    a = FakeState(score=4.0, edge_scores=[1.0, 0.1, 0.9, 2.0])  # min=0.1
    b = FakeState(score=3.5, edge_scores=[0.7, 0.8, 0.9, 1.1])  # min=0.7
    chosen = _select_best_beam_state([a, b], objective="total_score")
    assert chosen is a


def test_min_edge_objective_picks_higher_min():
    """min_edge objective picks path with highest minimum edge."""
    a = FakeState(score=4.0, edge_scores=[1.0, 0.1, 0.9, 2.0])  # min=0.1
    b = FakeState(score=3.5, edge_scores=[0.7, 0.8, 0.9, 1.1])  # min=0.7
    chosen = _select_best_beam_state([a, b], objective="min_edge")
    assert chosen is b


def test_min_edge_ties_broken_by_total_score():
    """When min edges are equal, min_edge objective breaks ties with total score."""
    a = FakeState(score=4.0, edge_scores=[0.5, 0.7, 0.9, 1.9])  # min=0.5 total=4.0
    b = FakeState(score=3.0, edge_scores=[0.5, 0.6, 0.8, 1.1])  # min=0.5 total=3.0
    chosen = _select_best_beam_state([a, b], objective="min_edge")
    assert chosen is a


def test_empty_beam_returns_none():
    """Empty beam returns None."""
    assert _select_best_beam_state([], objective="min_edge") is None


def test_single_state_returns_itself():
    """Single state in beam is returned regardless of objective."""
    a = FakeState(score=2.0, edge_scores=[0.3, 0.4])
    result = _select_best_beam_state([a], objective="min_edge")
    assert result is a


def test_no_edge_components_handled():
    """States with missing/empty edge_components are handled gracefully."""
    a = FakeState(score=2.0, edge_scores=[])  # empty, min=None -> -1e18
    b = FakeState(score=1.5, edge_scores=[0.5, 0.6])  # min=0.5
    chosen = _select_best_beam_state([a, b], objective="min_edge")
    # b should win because min_edge prefers higher min
    assert chosen is b


def test_invalid_objective_defaults_to_total_score():
    """Invalid objective value defaults to total_score behavior."""
    a = FakeState(score=4.0, edge_scores=[1.0, 0.1, 0.9, 2.0])  # min=0.1
    b = FakeState(score=3.5, edge_scores=[0.7, 0.8, 0.9, 1.1])  # min=0.7
    chosen = _select_best_beam_state([a, b], objective="invalid")
    # Should behave like total_score, picking a
    assert chosen is a
