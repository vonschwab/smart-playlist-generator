"""SP2 seed-character anti-collapse scoring — pure-function unit tests.

anti_center: penalize a candidate by how much closer it sits to the local pool center
than to its own piers (the within-bridge sag fix). The "hubness" variant was retired.
"""
from src.playlist.pier_bridge.seed_character import anti_center_penalty


def test_anti_center_penalty_fires_only_when_more_central_than_pier_like():
    assert anti_center_penalty(cand_center_sim=0.7, bridge_score=0.4, strength=0.5) > 0
    assert anti_center_penalty(cand_center_sim=0.3, bridge_score=0.6, strength=0.5) == 0.0
    assert anti_center_penalty(0.9, 0.1, strength=0.0) == 0.0   # strength 0 => inert
