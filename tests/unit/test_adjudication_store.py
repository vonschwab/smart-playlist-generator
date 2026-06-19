"""Tests for the resumable adjudication checkpoint store (Phase 3)."""
from __future__ import annotations

from src.ai_genre_enrichment.adjudication_store import AdjudicationStore


def test_save_and_is_done(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    assert s.is_done("a1", "pv1", "h1") is False
    s.save(album_id="a1", prompt_version="pv1", input_hash="h1", status="complete",
           response={"genres": []}, tokens={"total_tokens": 100})
    assert s.is_done("a1", "pv1", "h1") is True
    assert s.is_done("a1", "pv1", "h2") is False  # input changed -> re-run
    assert s.is_done("a1", "pv2", "h1") is False  # prompt changed -> re-run


def test_failed_is_not_done_so_it_retries(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    s.save(album_id="a1", prompt_version="pv1", input_hash="h1", status="failed", error="rate limit")
    assert s.is_done("a1", "pv1", "h1") is False


def test_save_upserts(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    s.save(album_id="a1", prompt_version="pv1", input_hash="h1", status="failed", error="x")
    s.save(album_id="a1", prompt_version="pv1", input_hash="h1", status="complete",
           response={"genres": []}, tokens={"total_tokens": 50})
    assert s.is_done("a1", "pv1", "h1") is True
    assert s.stats() == {"complete": 1}


def test_stats_and_total_tokens(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    s.save(album_id="a1", prompt_version="pv", input_hash="h", status="complete", tokens={"total_tokens": 100})
    s.save(album_id="a2", prompt_version="pv", input_hash="h", status="failed", error="e")
    assert s.stats() == {"complete": 1, "failed": 1}
    assert s.total_tokens() == 100


def test_reopen_persists(tmp_path):
    path = tmp_path / "shadow.db"
    s = AdjudicationStore(path)
    s.save(album_id="a1", prompt_version="pv", input_hash="h", status="complete", response={"x": 1})
    s.close()
    s2 = AdjudicationStore(path)  # reopen
    assert s2.is_done("a1", "pv", "h") is True


def test_shallow_album_ids_returns_low_genre_non_escalated(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    _save = lambda aid, genres, esc: s.save(
        album_id=aid, prompt_version="pv1", input_hash="h", status="complete",
        response={"genres": [{"term": g} for g in genres], "escalate": esc},
    )
    _save("a1", ["indie pop"], False)           # 1 genre, not escalated -> SHALLOW
    _save("a2", ["foo", "bar", "baz"], False)   # 3 genres -> not shallow
    _save("a3", ["foo", "bar"], True)           # 2 genres, escalated -> excluded
    _save("a4", [], False)                      # 0 genres -> SHALLOW
    _save("a5", ["x", "y"], False)              # 2 genres, not escalated -> SHALLOW

    result = s.shallow_album_ids("pv1")
    assert set(result) == {"a1", "a4", "a5"}


def test_complete_album_ids_scoped_by_prompt_version_ignoring_input_hash(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    # a1 complete under std; a2 complete under thorough; a3 failed under std.
    s.save(album_id="a1", prompt_version="std", input_hash="h1", status="complete",
           response={"genres": []})
    s.save(album_id="a2", prompt_version="thorough", input_hash="h2", status="complete",
           response={"genres": []})
    s.save(album_id="a3", prompt_version="std", input_hash="h3", status="failed", error="x")
    # a1 also has a thorough row (Phase-4-style overlap).
    s.save(album_id="a1", prompt_version="thorough", input_hash="h4", status="complete",
           response={"genres": []})

    assert s.complete_album_ids("std") == {"a1"}          # a3 failed -> excluded
    assert s.complete_album_ids("thorough") == {"a1", "a2"}
    assert s.complete_album_ids("missing") == set()


def test_shallow_album_ids_only_looks_at_given_prompt_version(tmp_path):
    s = AdjudicationStore(tmp_path / "shadow.db")
    s.save(album_id="a1", prompt_version="pv1", input_hash="h", status="complete",
           response={"genres": [{"term": "foo"}], "escalate": False})
    s.save(album_id="a2", prompt_version="pv2", input_hash="h", status="complete",
           response={"genres": [{"term": "foo"}], "escalate": False})

    assert s.shallow_album_ids("pv1") == ["a1"]  # pv2 album not included
    assert s.shallow_album_ids("pv2") == ["a2"]
