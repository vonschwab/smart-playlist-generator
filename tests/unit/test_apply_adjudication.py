"""Unit tests for apply_adjudication pure helpers."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.research.apply_adjudication import (
    best_results,
    invented_genres,
    split_lanes,
)


def _r(genres, escalate=False):
    return {"genres": [{"term": g} for g in genres], "escalate": escalate}


def test_best_results_prefers_thorough():
    rows = [
        ("a1", "pv-std", _r(["x"])),
        ("a1", "pv-thorough", _r(["x", "y", "z"])),
        ("a2", "pv-std", _r(["q"])),
    ]
    best = best_results(rows, thorough_pv="pv-thorough")
    assert [g["term"] for g in best["a1"]["genres"]] == ["x", "y", "z"]
    assert [g["term"] for g in best["a2"]["genres"]] == ["q"]


def test_split_lanes_separates_escalated():
    best = {"a1": _r(["x"]), "a2": _r(["y"], escalate=True)}
    auto, escalated = split_lanes(best)
    assert set(auto) == {"a1"} and set(escalated) == {"a2"}


def test_invented_genres_are_proposed_minus_prior():
    assert invented_genres(["afrobeat", "funk"], ["funk", "soul"]) == ["afrobeat"]
    assert invented_genres(["funk"], ["funk", "soul"]) == []
