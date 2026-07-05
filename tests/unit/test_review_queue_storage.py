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


def test_pending_page_includes_global_decided_totals(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x"), _term("y")],
    )
    store.set_review_queue_status(release_key="a::b", term="x", status="accepted")
    page = store.get_review_queue_page()
    assert page["pending_terms"] == 1
    assert page["decided_releases"] == 1
    assert page["decided_terms"] == 1


def test_completed_page_lists_decided_releases_recent_first(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x"), _term("y")],
    )
    store.sync_review_queue_for_release(
        release_key="c::d", normalized_artist="c", normalized_album="d",
        terms=[_term("p")],
    )
    store.set_review_queue_status(release_key="a::b", term="x", status="accepted")
    store.set_review_queue_status(release_key="c::d", term="p", status="rejected")
    # _now_iso() is second-resolution, so pin distinct decided_at to assert that
    # the page orders by most-recent decision (c::d decided after a::b).
    with store.connect() as conn:
        conn.execute("UPDATE ai_genre_review_queue SET decided_at=? WHERE release_key=? AND term=?",
                     ("2026-06-13T00:00:01+00:00", "a::b", "x"))
        conn.execute("UPDATE ai_genre_review_queue SET decided_at=? WHERE release_key=? AND term=?",
                     ("2026-06-13T00:00:09+00:00", "c::d", "p"))
        conn.commit()
    page = store.get_completed_review_page()
    assert page["decided_releases"] == 2
    assert page["decided_terms"] == 2
    # Most-recently-decided first, so the user's latest work is on top.
    assert [r["release_key"] for r in page["releases"]] == ["c::d", "a::b"]
    ab = next(r for r in page["releases"] if r["release_key"] == "a::b")
    assert {t["term"] for t in ab["decided"]} == {"x"}
    assert ab["decided"][0]["status"] == "accepted"
    # A partially-decided release still surfaces its remaining pending term.
    assert {t["term"] for t in ab["pending"]} == {"y"}


def test_completed_page_excludes_fully_pending_releases(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        terms=[_term("x")],
    )  # never decided
    page = store.get_completed_review_page()
    assert page["releases"] == []
    assert page["decided_releases"] == 0
    assert page["decided_terms"] == 0


def test_completed_page_search_filters_but_counts_are_global(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="acetone::cindy", normalized_artist="acetone",
        normalized_album="cindy", terms=[_term("x")],
    )
    store.sync_review_queue_for_release(
        release_key="low::hey", normalized_artist="low",
        normalized_album="hey", terms=[_term("y")],
    )
    store.set_review_queue_status(release_key="acetone::cindy", term="x", status="accepted")
    store.set_review_queue_status(release_key="low::hey", term="y", status="rejected")
    page = store.get_completed_review_page(search="acet")
    assert [r["release_key"] for r in page["releases"]] == ["acetone::cindy"]
    # Header counts describe the whole queue, not the filtered page.
    assert page["decided_releases"] == 2
    assert page["decided_terms"] == 2


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


def test_review_queue_page_splits_pending_by_basis(tmp_path):
    store = _store(tmp_path)
    store.sync_review_queue_for_release(
        release_key="a::x", normalized_artist="a", normalized_album="x",
        terms=[
            {"term": "shoegaze", "confidence": 0.4, "basis": "hybrid_provisional",
             "sources": ["lastfm_tags"], "reason": "published capped"},
            {"term": "zeuhl", "confidence": 0.6, "basis": "layered_taxonomy",
             "sources": ["discogs"], "reason": "Unknown layered taxonomy term."},
        ],
    )
    page = store.get_review_queue_page()
    assert page["pending_terms"] == 2
    assert page["pending_published_terms"] == 1
    assert page["pending_coverage_terms"] == 1
