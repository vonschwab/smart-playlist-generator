# tests/unit/test_review_queue_logic.py
"""Tests for review-queue scan and decision logic."""
import json

import pytest

from src.ai_genre_enrichment.review_queue import (
    apply_review_decision,
    compute_review_terms,
    scan_review_queue,
)
from src.ai_genre_enrichment.storage import SidecarStore


def _store(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    return store


def _diag_row(term, basis="hybrid_fusion", confidence=0.4):
    return {
        "term": term,
        "confidence": confidence,
        "source_basis": basis,
        "sources": ["lastfm_tags"],
        "reason": "Uncertain evidence.",
    }


def test_compute_review_terms_maps_diagnostic_rows(tmp_path):
    store = _store(tmp_path)
    diag = {"review_terms": [_diag_row("slowcore"), _diag_row("sadcore", basis="layered_taxonomy")]}
    terms = compute_review_terms(
        store, taxonomy=None, release_key="a::b",
        diagnostics_fn=lambda s, *, release_id, taxonomy: diag,
    )
    assert [t["term"] for t in terms] == ["slowcore", "sadcore"]
    assert terms[0]["basis"] == "hybrid_fusion"
    assert terms[1]["basis"] == "layered_taxonomy"
    assert terms[0]["sources"] == ["lastfm_tags"]


def test_compute_review_terms_skips_override_settled(tmp_path):
    store = _store(tmp_path)
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["slowcore"], genres_remove=["sadcore"],
    )
    diag = {"review_terms": [_diag_row("slowcore"), _diag_row("SADCORE"), _diag_row("dronefolk")]}
    terms = compute_review_terms(
        store, taxonomy=None, release_key="a::b",
        diagnostics_fn=lambda s, *, release_id, taxonomy: diag,
    )
    assert [t["term"] for t in terms] == ["dronefolk"]


def test_scan_review_queue_iterates_and_syncs(tmp_path):
    store = _store(tmp_path)
    store.upsert_source_page(
        release_key="a::one", normalized_artist="a", normalized_album="one",
        album_id=None, source_url="lastfm://a/one", source_type="lastfm_tags",
        identity_status="confirmed", identity_confidence=1.0, evidence_summary="x",
    )
    store.upsert_source_page(
        release_key="b::two", normalized_artist="b", normalized_album="two",
        album_id=None, source_url="lastfm://b/two", source_type="lastfm_tags",
        identity_status="confirmed", identity_confidence=1.0, evidence_summary="x",
    )
    diags = {
        "a::one": {"review_terms": [_diag_row("x"), _diag_row("y")]},
        "b::two": {"review_terms": []},
    }
    progress = []
    summary = scan_review_queue(
        store, taxonomy=None,
        diagnostics_fn=lambda s, *, release_id, taxonomy: diags[release_id],
        progress_cb=lambda cur, total, detail: progress.append((cur, total)),
    )
    assert summary["releases_scanned"] == 2
    assert summary["new_terms"] == 2
    assert summary["pending_terms"] == 2
    assert progress == [(1, 2), (2, 2)]


def test_scan_review_queue_cancel_keeps_partial(tmp_path):
    store = _store(tmp_path)
    for key in ("a::one", "b::two"):
        artist, album = key.split("::")
        store.upsert_source_page(
            release_key=key, normalized_artist=artist, normalized_album=album,
            album_id=None, source_url=f"lastfm://{key}", source_type="lastfm_tags",
            identity_status="confirmed", identity_confidence=1.0, evidence_summary="x",
        )

    class Cancelled(Exception):
        pass

    calls = {"n": 0}

    def cancel_cb():
        # First release goes through; cancel before the second.
        if calls["n"] >= 1:
            raise Cancelled()
        calls["n"] += 1

    with pytest.raises(Cancelled):
        scan_review_queue(
            store, taxonomy=None,
            diagnostics_fn=lambda s, *, release_id, taxonomy: {"review_terms": [_diag_row("x")]},
            cancel_cb=cancel_cb,
        )
    # First release's rows were committed before the cancel.
    assert store.get_review_queue_page()["pending_terms"] == 1


def _seed_queue_row(store, release_key="a::b", term="slowcore"):
    artist, album = release_key.split("::")
    store.sync_review_queue_for_release(
        release_key=release_key, normalized_artist=artist, normalized_album=album,
        terms=[{"term": term, "confidence": 0.4, "basis": "hybrid_fusion",
                "sources": ["lastfm_tags"], "reason": "r"}],
    )


def test_apply_decision_accept_merges_override(tmp_path):
    store = _store(tmp_path)
    # Pre-existing override must be preserved (set_user_override replaces).
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["dreampop"], genres_remove=["indie"],
    )
    _seed_queue_row(store)
    result = apply_review_decision(store, release_key="a::b", term="slowcore", decision="accept")
    assert result["status"] == "accepted"
    override = store.get_user_override("a::b")
    assert set(override["genres_add"]) == {"dreampop", "slowcore"}
    assert set(override["genres_remove"]) == {"indie"}
    page = store.get_review_queue_page()
    assert page["pending_terms"] == 0


def test_apply_decision_reject_adds_to_remove(tmp_path):
    store = _store(tmp_path)
    _seed_queue_row(store)
    apply_review_decision(store, release_key="a::b", term="slowcore", decision="reject")
    override = store.get_user_override("a::b")
    assert override["genres_add"] == []
    assert override["genres_remove"] == ["slowcore"]


def test_apply_decision_revert_clears_override_entry(tmp_path):
    store = _store(tmp_path)
    _seed_queue_row(store)
    apply_review_decision(store, release_key="a::b", term="slowcore", decision="accept")
    apply_review_decision(store, release_key="a::b", term="slowcore", decision="revert")
    override = store.get_user_override("a::b")
    assert override["genres_add"] == []
    assert override["genres_remove"] == []
    assert store.get_review_queue_page()["pending_terms"] == 1


def test_apply_decision_unknown_row_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        apply_review_decision(store, release_key="no::pe", term="x", decision="accept")


def test_apply_decision_invalid_decision_raises(tmp_path):
    store = _store(tmp_path)
    _seed_queue_row(store)
    with pytest.raises(ValueError):
        apply_review_decision(store, release_key="a::b", term="slowcore", decision="maybe")
