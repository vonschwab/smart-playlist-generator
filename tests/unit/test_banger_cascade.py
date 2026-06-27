"""Unit tests for Task 5: _banger_relaxation_steps generator.

Tests the ladder order, popularity-rung cutoffs, and the final step properties.
"""
from __future__ import annotations

from dataclasses import replace
from src.playlist.pipeline.core import _banger_relaxation_steps
from src.playlist.config import CandidatePoolConfig


def _cfg():
    return CandidatePoolConfig(
        similarity_floor=0.0, min_sonic_similarity=0.3, max_pool_size=200,
        target_artists=20, candidates_per_artist=6, seed_artist_bonus=4,
        max_artist_fraction_final=0.2, sonic_admission_percentile=0.6,
    )


def test_ladder_order_sonic_pace_genre_then_popularity_last():
    steps = list(_banger_relaxation_steps(_cfg(), base_genre_gate=0.2, base_cutoff=10))
    labels = [s.label for s in steps]
    # sonic/pace appear before genre; popularity rungs are strictly last
    pop_idx = [i for i, l in enumerate(labels) if l.startswith("popularity")]
    genre_idx = [i for i, l in enumerate(labels) if l.startswith("genre")]
    sonic_pace_idx = [i for i, l in enumerate(labels) if l.startswith(("sonic", "pace"))]
    assert max(sonic_pace_idx) < min(genre_idx)         # sonic/pace before genre
    assert max(genre_idx) < min(pop_idx)                # genre before popularity
    assert pop_idx == list(range(min(pop_idx), len(labels)))  # popularity is the tail


def test_popularity_rungs_loosen_cutoff_then_disable():
    steps = list(_banger_relaxation_steps(_cfg(), base_genre_gate=0.2, base_cutoff=10))
    pop_cutoffs = [s.rank_cutoff for s in steps if s.label.startswith("popularity")]
    assert pop_cutoffs == [25, 50, None]   # top-25, top-50, gate off (last resort)


def test_final_step_disables_all_gates():
    steps = list(_banger_relaxation_steps(_cfg(), base_genre_gate=0.2, base_cutoff=10))
    last = steps[-1]
    assert last.rank_cutoff is None
    assert last.genre_gate is None
    assert last.candidate_cfg.sonic_admission_percentile in (0.0, None)
    assert last.candidate_cfg.min_sonic_similarity in (0.0, None)
