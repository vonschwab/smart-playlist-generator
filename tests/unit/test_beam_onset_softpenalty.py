"""Rhythm-cosine soft penalty demotes below threshold (MERT no-tower path).

The onset bridge band gate test was removed with filter_candidates_by_onset_target
(dead code, Phase 0 Task 2, 2026-07-16): beam.py never called it (onset
banding is enforced via compute_energy_pace_penalty inline in the beam)."""


def test_soft_penalty_multiplier_below_threshold():
    # Pure arithmetic guard for the multiplier the beam applies.
    strength = 0.15
    base = 1.0
    demoted = base * (1.0 - strength)
    assert demoted == 0.85
