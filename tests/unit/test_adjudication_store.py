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
