"""Unit tests for run_adjudicator_bulk pure helpers."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.research.run_adjudicator_bulk import exclude_done_album_ids


def test_exclude_done_drops_completed_album_ids_by_id_only():
    targets = [("a1", None), ("a2", "rk2"), ("a3", None)]
    # a2 is already complete — its (key) differs but exclusion is by album_id alone.
    result = exclude_done_album_ids(targets, {"a2"})
    assert result == [("a1", None), ("a3", None)]


def test_exclude_done_empty_done_set_keeps_all():
    targets = [("a1", None), ("a2", None)]
    assert exclude_done_album_ids(targets, set()) == targets


def test_exclude_done_all_done_returns_empty():
    targets = [("a1", None), ("a2", None)]
    assert exclude_done_album_ids(targets, {"a1", "a2"}) == []


def test_exclude_done_accepts_any_iterable():
    targets = [("a1", None), ("a2", None)]
    # done_ids may arrive as a list/generator, not only a set.
    assert exclude_done_album_ids(targets, ["a1"]) == [("a2", None)]
