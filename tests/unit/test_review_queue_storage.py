# tests/unit/test_review_queue_storage.py
"""Tests for the ai_genre_review_queue table and its SidecarStore methods."""
from src.ai_genre_enrichment.storage import SidecarStore


def _store(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    return store


def _term(name, confidence=0.5, basis="hybrid_fusion", sources=None, reason="uncertain"):
    return {
        "term": name,
        "confidence": confidence,
        "basis": basis,
        "sources": sources or ["lastfm_tags"],
        "reason": reason,
    }


def test_sync_inserts_new_pending_rows(tmp_path):
    store = _store(tmp_path)
    counts = store.sync_review_queue_for_release(
        release_key="acetone::york blvd",
        normalized_artist="acetone",
        normalized_album="york blvd",
        terms=[_term("slowcore"), _term("sadcore", basis="layered_taxonomy")],
    )
    assert counts == {"inserted": 2, "updated": 0, "pruned": 0}
    page = store.get_review_queue_page()
    assert page["pending_releases"] == 1
    assert page["pending_terms"] == 2
    rel = page["releases"][0]
    assert rel["release_key"] == "acetone::york blvd"
    assert {t["term"] for t in rel["pending"]} == {"slowcore", "sadcore"}
    assert rel["pending"][0]["sources"] == ["lastfm_tags"]


def test_sync_prunes_stale_pending_keeps_decided(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x"), _term("y")],
    )
    store.set_review_queue_status(release_key="a::b", term="x", status="accepted")
    # Rescan: 'y' no longer appears, 'x' (decided) no longer appears either.
    counts = store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("z")],
    )
    assert counts["pruned"] == 1      # y removed
    assert counts["inserted"] == 1    # z added
    page = store.get_review_queue_page()
    rel = page["releases"][0]
    assert {t["term"] for t in rel["pending"]} == {"z"}
    assert {t["term"] for t in rel["decided"]} == {"x"}  # decided row survives


def test_sync_updates_pending_in_place(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x", confidence=0.3)],
    )
    counts = store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x", confidence=0.7, reason="more evidence")],
    )
    assert counts == {"inserted": 0, "updated": 1, "pruned": 0}
    rel = store.get_review_queue_page()["releases"][0]
    assert rel["pending"][0]["confidence"] == 0.7
    assert rel["pending"][0]["reason"] == "more evidence"


def test_set_status_and_revert(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x")],
    )
    store.set_review_queue_status(release_key="a::b", term="x", status="rejected")
    page = store.get_review_queue_page()
    assert page["pending_terms"] == 0
    assert page["releases"] == []  # fully-decided releases drop off the page
    store.set_review_queue_status(release_key="a::b", term="x", status="pending")
    page = store.get_review_queue_page()
    assert page["pending_terms"] == 1
    assert page["releases"][0]["pending"][0]["status"] == "pending"


def test_page_search_and_ordering(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="acetone::cindy", normalized_artist="acetone",
        normalized_album="cindy", terms=[_term("x")],
    )
    store.sync_review_queue_for_release(
        release_key="low::things we lost", normalized_artist="low",
        normalized_album="things we lost", terms=[_term("x"), _term("y"), _term("z")],
    )
    page = store.get_review_queue_page()
    # Ordered by pending count desc
    assert [r["release_key"] for r in page["releases"]] == [
        "low::things we lost", "acetone::cindy",
    ]
    page = store.get_review_queue_page(search="acet")
    assert [r["release_key"] for r in page["releases"]] == ["acetone::cindy"]
    # Header counts ignore the search filter (they describe the whole queue)
    assert page["pending_releases"] == 2
    assert page["pending_terms"] == 4


def test_list_review_scan_releases(tmp_path):
    store = _store(tmp_path)
    store.upsert_source_page(
        release_key="acetone::cindy", normalized_artist="acetone",
        normalized_album="cindy", album_id=None,
        source_url="lastfm://artist/acetone/album/cindy",
        source_type="lastfm_tags", identity_status="confirmed",
        identity_confidence=1.0, evidence_summary="lastfm",
    )
    releases = store.list_review_scan_releases()
    assert releases == [{
        "release_key": "acetone::cindy",
        "normalized_artist": "acetone",
        "normalized_album": "cindy",
    }]
