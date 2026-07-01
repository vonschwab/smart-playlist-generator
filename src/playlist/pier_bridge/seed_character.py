"""SP2 seed-character anti-collapse scoring (pure functions, unit-testable).

Genre-blur collapse: bridge interiors drift off the seeds' specific character into the
dense generic neighborhood ([[project_collapse_attack_design]]). The "anti_center"
adjustment demotes such interiors, validated as the winner via the collapse harness
(the "hubness" variant was retired — weak and non-scaling):

  "anti_center" — penalize a candidate by how much closer it sits to the local pool
                  centroid than to its own piers. This is the scoring twin of the
                  within-bridge sag metric (cos(t,center) - cos(t,piers)): it directly
                  demotes interiors that sag toward the average.

Off (strength 0 / mode "off") => the caller leaves the score untouched (byte-identical).
"""
from __future__ import annotations


def anti_center_penalty(cand_center_sim: float, bridge_score: float, strength: float) -> float:
    """SP2-B penalty (>= 0) to SUBTRACT from combined_score: ``strength`` times how
    much closer the candidate sits to the local pool centroid than to its own piers
    (the bridge score). 0 when the candidate is more pier-like than central."""
    return float(strength) * max(0.0, float(cand_center_sim) - float(bridge_score))
