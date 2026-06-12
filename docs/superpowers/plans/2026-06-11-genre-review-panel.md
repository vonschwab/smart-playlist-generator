# Genre Review Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the web GUI's "Genre Review" stub tab with a working review surface for hybrid-evidence review terms, backed by a persisted queue in the sidecar DB.

**Architecture:** A new `ai_genre_review_queue` table is populated by a tracked worker scan job (`scan_genre_review`) that runs `build_layered_release_diagnostics` per release. Quick worker commands (`get_genre_review_queue`, `apply_genre_review_decision`) serve reads and decisions over the existing `bridge.command` path; decisions merge into `ai_genre_user_overrides` (which `publish` already applies) and re-bake the release signature. A new `GenreReviewPanel` React component replaces the stub in `AdvancedPanel.tsx`.

**Tech Stack:** Python 3.11 (sqlite3, FastAPI, Pydantic), TypeScript/React/Tailwind, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-06-11-genre-review-panel-design.md`

---

## Background the implementer needs

- **Sidecar DB:** `data/ai_genre_enrichment.db`, wrapped by `src/ai_genre_enrichment/storage.py::SidecarStore`. `initialize()` creates tables via one `executescript` (the `ai_genre_user_overrides` CREATE is at `storage.py:330`). Helper `_now_iso()` at `storage.py:60`.
- **`set_user_override` REPLACES the whole row** (`storage.py:2646`) — decision code must read-merge-write, never write a bare single-term override. It casefolds and sorts internally. `get_user_override(release_key)` (`storage.py:2678`) returns `{"genres_add": [...], "genres_remove": [...], "updated_at": ...}` or `None`. `rebuild_enriched_genres_for_release(release_key)` (`storage.py:1408`) re-bakes `enriched_genres` + the signature; it tolerates releases with no source rows.
- **Review terms** come from `build_layered_release_diagnostics(store, release_id=..., taxonomy=...)` (`src/ai_genre_enrichment/layered_assignment.py:203`), key `"review_terms"`: a list of dicts with keys `term`, `confidence`, `sources` (list), `reason`, and `source_basis` (one of `layered_taxonomy`, `hybrid_provisional`, `hybrid_fusion`). Rows merged via `_merge_decision_row` always carry `source_basis`; treat a missing value as `hybrid_fusion`. Taxonomy loads via `load_default_layered_taxonomy()` from `src.ai_genre_enrichment.layered_taxonomy` (~12 s, load once per scan).
- **Release universe for the scan:** distinct `release_key` in `ai_genre_source_pages` (~4,000 rows; columns `release_key`, `normalized_artist`, `normalized_album`).
- **Worker** (`src/playlist_gui/worker.py`): `SIDECAR_DB_PATH = "data/ai_genre_enrichment.db"` (line 78). Emit helpers: `emit_progress(stage, current, total, detail=None)`, `emit_result(result_type, data)`, `emit_done(cmd, ok, detail=None, cancelled=False, summary=None)`, `emit_error(message, tb=None)`. Cancellation: call module-level `check_cancelled()` at release boundaries; it raises `CancellationError` (worker.py:85); catch it and `emit_done(..., cancelled=True)`. Register handlers in `TRACKED_COMMAND_HANDLERS` (worker.py:2407). Every handler must emit `done` on all paths or the bridge hangs.
- **FastAPI** (`src/playlist_web/app.py`): long jobs use `registry.create(...)` + `await bridge.submit({...})` with `BridgeBusy → 409` (see `/api/tools/analyze`, line 174). Quick commands use `await bridge.command({...})` with `BridgeBusy → 409` and `WorkerCommandError → 422/502` (see `/api/edit_genres`, line 359). `bridge.command` returns the worker's result-event dict.
- **Front-end:** served from `web/dist` — run `npm --prefix web run build` after editing `web/src` or changes are invisible. Worker code changes need a `serve_web.py` restart. WS events arrive via `useWorkerEvents(cb)` from `web/src/lib/ws` (usage pattern: `web/src/components/ToolsPanel.tsx:137`).
- **Tests:** unit tests use `SidecarStore(tmp_path / "sidecar.db")` — never the real DB. Integration tests use `TestClient(create_app(worker_cmd=FAKE))` with `FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]` and poll `/api/jobs/{id}` (pattern: `tests/integration/test_web_tools_api.py`). Playwright specs live in `web/tests/`; `web/playwright.config.ts` builds + serves on 8771 with the fake worker.

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/ai_genre_enrichment/storage.py` | Modify | `ai_genre_review_queue` table + 4 CRUD methods |
| `src/ai_genre_enrichment/review_queue.py` | Create | Scan + decision domain logic (no worker/IPC imports) |
| `src/playlist_gui/worker.py` | Modify | 3 handlers + registration |
| `src/playlist_web/schemas.py` | Modify | `ReviewDecisionRequest` |
| `src/playlist_web/app.py` | Modify | `/api/review/scan`, `/api/review/queue`, `/api/review/decision` |
| `tests/unit/test_review_queue_storage.py` | Create | Table CRUD + sync semantics |
| `tests/unit/test_review_queue_logic.py` | Create | Scan + decision logic |
| `tests/unit/test_worker_review_queue.py` | Create | Worker handler round-trips |
| `tests/fixtures/fake_worker.py` | Modify | 3 fake command branches |
| `tests/integration/test_web_review_api.py` | Create | Endpoint round-trips |
| `web/src/lib/types.ts` | Modify | Review types |
| `web/src/lib/api.ts` | Modify | 3 client methods |
| `web/src/components/GenreReviewPanel.tsx` | Create | The panel |
| `web/src/components/AdvancedPanel.tsx` | Modify | Replace stub with panel |
| `web/tests/review.spec.ts` | Create | Playwright smoke |

---

## Task 1: Storage — queue table + CRUD methods

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Test: `tests/unit/test_review_queue_storage.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_review_queue_storage.py -v`
Expected: FAIL with `AttributeError: 'SidecarStore' object has no attribute 'sync_review_queue_for_release'`

- [ ] **Step 3: Add the table to `initialize()`**

In `src/ai_genre_enrichment/storage.py`, inside the `initialize()` executescript, immediately after the `ai_genre_user_overrides` CREATE block (after line 337), add:

```sql
                CREATE TABLE IF NOT EXISTS ai_genre_review_queue (
                    release_key       TEXT NOT NULL,
                    normalized_artist TEXT NOT NULL,
                    normalized_album  TEXT NOT NULL,
                    term              TEXT NOT NULL,
                    confidence        REAL,
                    basis             TEXT NOT NULL DEFAULT 'hybrid_fusion',
                    sources_json      TEXT NOT NULL DEFAULT '[]',
                    reason            TEXT NOT NULL DEFAULT '',
                    status            TEXT NOT NULL DEFAULT 'pending' CHECK (
                        status IN ('pending', 'accepted', 'rejected')
                    ),
                    scanned_at        TEXT NOT NULL,
                    decided_at        TEXT,
                    PRIMARY KEY (release_key, term)
                );

                CREATE INDEX IF NOT EXISTS idx_review_queue_status
                    ON ai_genre_review_queue (status, release_key);
```

- [ ] **Step 4: Add the four methods to `SidecarStore`**

Add after `get_user_override` / `delete_user_override` (near `storage.py:2700`):

```python
    def list_review_scan_releases(self) -> list[dict[str, Any]]:
        """Distinct releases known to the evidence layer, for the review scan."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT release_key,
                       MIN(normalized_artist) AS normalized_artist,
                       MIN(normalized_album) AS normalized_album
                FROM ai_genre_source_pages
                GROUP BY release_key
                ORDER BY release_key
                """
            )
            return [dict(row) for row in rows]

    def sync_review_queue_for_release(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        terms: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Reconcile pending queue rows for one release against a fresh scan.

        Inserts new terms, refreshes still-pending ones, prunes pending rows
        whose term no longer appears. Rows with a decided status are never
        touched, so rescans cannot resurrect settled questions.
        """
        now = _now_iso()
        term_names = {t["term"] for t in terms}
        inserted = updated = pruned = 0
        with self.connect() as conn:
            existing = {
                row["term"]: row["status"]
                for row in conn.execute(
                    "SELECT term, status FROM ai_genre_review_queue WHERE release_key = ?",
                    (release_key,),
                )
            }
            for term, status in existing.items():
                if status == "pending" and term not in term_names:
                    conn.execute(
                        "DELETE FROM ai_genre_review_queue WHERE release_key = ? AND term = ?",
                        (release_key, term),
                    )
                    pruned += 1
            for t in terms:
                status = existing.get(t["term"])
                if status is None:
                    conn.execute(
                        """
                        INSERT INTO ai_genre_review_queue (
                            release_key, normalized_artist, normalized_album, term,
                            confidence, basis, sources_json, reason, status, scanned_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                        """,
                        (
                            release_key, normalized_artist, normalized_album, t["term"],
                            t.get("confidence"), t.get("basis") or "hybrid_fusion",
                            json.dumps(list(t.get("sources") or [])),
                            t.get("reason") or "", now,
                        ),
                    )
                    inserted += 1
                elif status == "pending":
                    conn.execute(
                        """
                        UPDATE ai_genre_review_queue
                        SET confidence = ?, basis = ?, sources_json = ?, reason = ?, scanned_at = ?
                        WHERE release_key = ? AND term = ? AND status = 'pending'
                        """,
                        (
                            t.get("confidence"), t.get("basis") or "hybrid_fusion",
                            json.dumps(list(t.get("sources") or [])),
                            t.get("reason") or "", now, release_key, t["term"],
                        ),
                    )
                    updated += 1
            conn.commit()
        return {"inserted": inserted, "updated": updated, "pruned": pruned}

    def get_review_queue_page(
        self,
        *,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Releases with pending review terms, most-pending first.

        Header counts (pending_releases / pending_terms) always describe the
        whole queue, not the filtered page.
        """
        with self.connect() as conn:
            counts = conn.execute(
                "SELECT COUNT(DISTINCT release_key) AS pr, COUNT(*) AS pt "
                "FROM ai_genre_review_queue WHERE status = 'pending'"
            ).fetchone()
            where = ""
            params: list[Any] = []
            if search:
                where = "WHERE (normalized_artist LIKE ? OR normalized_album LIKE ?)"
                pattern = f"%{search.strip().casefold()}%"
                params = [pattern, pattern]
            release_rows = list(conn.execute(
                f"""
                SELECT release_key, normalized_artist, normalized_album,
                       SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count
                FROM ai_genre_review_queue
                {where}
                GROUP BY release_key, normalized_artist, normalized_album
                HAVING pending_count > 0
                ORDER BY pending_count DESC, release_key
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ))
            releases = []
            for rel in release_rows:
                terms = [
                    {
                        "term": row["term"],
                        "confidence": row["confidence"],
                        "basis": row["basis"],
                        "sources": json.loads(row["sources_json"]),
                        "reason": row["reason"],
                        "status": row["status"],
                    }
                    for row in conn.execute(
                        "SELECT term, confidence, basis, sources_json, reason, status "
                        "FROM ai_genre_review_queue WHERE release_key = ? "
                        "ORDER BY confidence DESC, term",
                        (rel["release_key"],),
                    )
                ]
                releases.append({
                    "release_key": rel["release_key"],
                    "artist": rel["normalized_artist"],
                    "album": rel["normalized_album"],
                    "pending": [t for t in terms if t["status"] == "pending"],
                    "decided": [t for t in terms if t["status"] != "pending"],
                })
        return {
            "releases": releases,
            "pending_releases": int(counts["pr"]),
            "pending_terms": int(counts["pt"]),
        }

    def set_review_queue_status(
        self, *, release_key: str, term: str, status: str
    ) -> None:
        """Set a queue row's status. 'pending' clears decided_at (revert)."""
        if status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"invalid review queue status: {status}")
        decided_at = None if status == "pending" else _now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE ai_genre_review_queue SET status = ?, decided_at = ? "
                "WHERE release_key = ? AND term = ?",
                (status, decided_at, release_key, term),
            )
            if cur.rowcount == 0:
                raise ValueError(f"no review queue row for {release_key!r} / {term!r}")
            conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_review_queue_storage.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run the existing storage/enrichment tests for regressions**

Run: `pytest tests/unit/test_user_overrides_storage.py tests/unit/test_ai_genre_enrichment.py -q`
Expected: same pass count as before this task.

- [ ] **Step 7: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_review_queue_storage.py
git commit -m "feat(genre-review): add ai_genre_review_queue table and SidecarStore CRUD"
```

---

## Task 2: Domain logic — scan + decisions (`review_queue.py`)

**Files:**
- Create: `src/ai_genre_enrichment/review_queue.py`
- Test: `tests/unit/test_review_queue_logic.py` (create)

Pure domain module: no worker/IPC imports, taxonomy and diagnostics injectable so tests don't need the 12-second taxonomy load.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_review_queue_logic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.review_queue'`

- [ ] **Step 3: Create `src/ai_genre_enrichment/review_queue.py`**

```python
"""Scan and decision logic for the human genre-review queue.

The queue persists hybrid-evidence review terms (uncertain fusion decisions and
taxonomy-unknown terms) per release in ai_genre_review_queue. Decisions are
written through the existing user-override mechanism, which the publish stage
already applies.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .layered_assignment import build_layered_release_diagnostics

DiagnosticsFn = Callable[..., dict[str, Any]]

_DECISION_TO_STATUS = {"accept": "accepted", "reject": "rejected", "revert": "pending"}


def compute_review_terms(
    store: Any,
    *,
    taxonomy: Any,
    release_key: str,
    diagnostics_fn: DiagnosticsFn = build_layered_release_diagnostics,
) -> list[dict[str, Any]]:
    """Review-term rows for one release, excluding override-settled terms."""
    diag = diagnostics_fn(store, release_id=release_key, taxonomy=taxonomy)
    override = store.get_user_override(release_key) or {"genres_add": [], "genres_remove": []}
    settled = {
        g.casefold()
        for g in list(override["genres_add"]) + list(override["genres_remove"])
    }
    terms: list[dict[str, Any]] = []
    for row in diag.get("review_terms") or []:
        term = str(row.get("term") or "").strip()
        if not term or term.casefold() in settled:
            continue
        terms.append({
            "term": term,
            "confidence": float(row.get("confidence") or 0.0),
            "basis": str(row.get("source_basis") or row.get("basis") or "hybrid_fusion"),
            "sources": sorted(row.get("sources") or []),
            "reason": str(row.get("reason") or ""),
        })
    return terms


def scan_review_queue(
    store: Any,
    *,
    taxonomy: Any,
    diagnostics_fn: DiagnosticsFn = build_layered_release_diagnostics,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_cb: Optional[Callable[[], None]] = None,
) -> dict[str, int]:
    """Reconcile the review queue against fresh diagnostics for every release.

    cancel_cb is invoked at release boundaries only; each release's sync is a
    committed transaction, so a cancelled scan keeps its partial results.
    """
    releases = store.list_review_scan_releases()
    total = len(releases)
    summary = {"releases_scanned": 0, "new_terms": 0, "pruned_terms": 0, "pending_terms": 0}
    for i, rel in enumerate(releases):
        if cancel_cb is not None:
            cancel_cb()
        terms = compute_review_terms(
            store, taxonomy=taxonomy, release_key=rel["release_key"],
            diagnostics_fn=diagnostics_fn,
        )
        counts = store.sync_review_queue_for_release(
            release_key=rel["release_key"],
            normalized_artist=rel["normalized_artist"],
            normalized_album=rel["normalized_album"],
            terms=terms,
        )
        summary["releases_scanned"] += 1
        summary["new_terms"] += counts["inserted"]
        summary["pruned_terms"] += counts["pruned"]
        if progress_cb is not None:
            progress_cb(i + 1, total, f"{rel['normalized_artist']} – {rel['normalized_album']}")
    summary["pending_terms"] = store.get_review_queue_page(limit=1)["pending_terms"]
    return summary


def apply_review_decision(
    store: Any,
    *,
    release_key: str,
    term: str,
    decision: str,
) -> dict[str, Any]:
    """Apply accept/reject/revert for one queue row.

    Merges into the release's user override (set_user_override REPLACES the
    row, so we must read-merge-write), re-bakes the enriched signature, and
    updates the queue row status.
    """
    status = _DECISION_TO_STATUS.get(decision)
    if status is None:
        raise ValueError(f"invalid decision: {decision!r} (use accept/reject/revert)")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT normalized_artist, normalized_album FROM ai_genre_review_queue "
            "WHERE release_key = ? AND term = ?",
            (release_key, term),
        ).fetchone()
    if row is None:
        raise ValueError(f"no review queue row for {release_key!r} / {term!r}")

    override = store.get_user_override(release_key) or {"genres_add": [], "genres_remove": []}
    add = {g.casefold() for g in override["genres_add"]}
    remove = {g.casefold() for g in override["genres_remove"]}
    key = term.casefold()
    add.discard(key)
    remove.discard(key)
    if decision == "accept":
        add.add(key)
    elif decision == "reject":
        remove.add(key)

    store.set_user_override(
        release_key=release_key,
        normalized_artist=row["normalized_artist"],
        normalized_album=row["normalized_album"],
        genres_add=sorted(add),
        genres_remove=sorted(remove),
    )
    store.rebuild_enriched_genres_for_release(release_key)
    store.set_review_queue_status(release_key=release_key, term=term, status=status)
    return {"release_key": release_key, "term": term, "decision": decision, "status": status}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_review_queue_logic.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Lint and type-check the new module**

Run: `ruff check src/ai_genre_enrichment/review_queue.py && mypy src/ai_genre_enrichment/review_queue.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ai_genre_enrichment/review_queue.py tests/unit/test_review_queue_logic.py
git commit -m "feat(genre-review): scan and decision logic for the review queue"
```

---

## Task 3: Worker handlers

**Files:**
- Modify: `src/playlist_gui/worker.py` (handlers after `handle_edit_genres` ~line 2368; registration in `TRACKED_COMMAND_HANDLERS` ~line 2407)
- Test: `tests/unit/test_worker_review_queue.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_worker_review_queue.py
"""Worker handler round-trips for the genre review queue commands."""
import json

from src.ai_genre_enrichment.storage import SidecarStore
from src.playlist_gui.worker import (
    handle_apply_genre_review_decision,
    handle_get_genre_review_queue,
)


def _events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]


def _seed(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    store.sync_review_queue_for_release(
        release_key="acetone::cindy", normalized_artist="acetone",
        normalized_album="cindy",
        terms=[{"term": "slowcore", "confidence": 0.4, "basis": "hybrid_fusion",
                "sources": ["lastfm_tags"], "reason": "uncertain"}],
    )
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))
    return store


def test_get_queue_emits_result_and_done(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_get_genre_review_queue({"cmd": "get_genre_review_queue", "request_id": "r1"})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "genre_review_queue"
    assert result["pending_terms"] == 1
    assert result["releases"][0]["release_key"] == "acetone::cindy"
    assert done["ok"] is True


def test_apply_decision_emits_result_and_updates_db(tmp_path, monkeypatch, capsys):
    store = _seed(tmp_path, monkeypatch)
    handle_apply_genre_review_decision({
        "cmd": "apply_genre_review_decision", "request_id": "r2",
        "release_key": "acetone::cindy", "term": "slowcore", "decision": "accept",
    })
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "genre_review_decision"
    assert result["status"] == "accepted"
    assert done["ok"] is True
    assert store.get_user_override("acetone::cindy")["genres_add"] == ["slowcore"]


def test_apply_decision_bad_input_emits_failed_done(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_apply_genre_review_decision({
        "cmd": "apply_genre_review_decision", "request_id": "r3",
        "release_key": "acetone::cindy", "term": "slowcore", "decision": "maybe",
    })
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_worker_review_queue.py -v`
Expected: FAIL with `ImportError: cannot import name 'handle_apply_genre_review_decision'`

- [ ] **Step 3: Add the three handlers**

In `src/playlist_gui/worker.py`, after `handle_edit_genres` (line 2368), add:

```python
def handle_scan_genre_review(cmd_data: Dict[str, Any]) -> None:
    """Scan all releases for hybrid review terms and persist the queue."""
    try:
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        from src.ai_genre_enrichment.review_queue import scan_review_queue
        from src.ai_genre_enrichment.storage import SidecarStore

        store = SidecarStore(SIDECAR_DB_PATH)
        store.initialize()
        emit_log("INFO", "Loading layered taxonomy for review scan...")
        taxonomy = load_default_layered_taxonomy()

        def progress(current: int, total: int, detail: str) -> None:
            emit_progress("scan_genre_review", current, total, detail)

        summary = scan_review_queue(
            store, taxonomy=taxonomy, progress_cb=progress, cancel_cb=check_cancelled,
        )
        emit_result("scan_genre_review", summary)
        emit_done(
            "scan_genre_review", True,
            f"Scanned {summary['releases_scanned']} releases",
            summary=f"{summary['pending_terms']} terms pending review",
        )
    except CancellationError:
        emit_done("scan_genre_review", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("scan_genre_review", False, str(e))


def handle_get_genre_review_queue(cmd_data: Dict[str, Any]) -> None:
    """Return the persisted review queue page (quick read)."""
    try:
        from src.ai_genre_enrichment.storage import SidecarStore

        store = SidecarStore(SIDECAR_DB_PATH)
        store.initialize()
        page = store.get_review_queue_page(
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0),
        )
        emit_result("genre_review_queue", page)
        emit_done("get_genre_review_queue", True, f"{page['pending_terms']} pending")
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("get_genre_review_queue", False, str(e))


def handle_apply_genre_review_decision(cmd_data: Dict[str, Any]) -> None:
    """Apply accept/reject/revert for one review-queue row (quick write)."""
    try:
        from src.ai_genre_enrichment.review_queue import apply_review_decision
        from src.ai_genre_enrichment.storage import SidecarStore

        store = SidecarStore(SIDECAR_DB_PATH)
        store.initialize()
        result = apply_review_decision(
            store,
            release_key=str(cmd_data.get("release_key") or ""),
            term=str(cmd_data.get("term") or ""),
            decision=str(cmd_data.get("decision") or ""),
        )
        emit_result("genre_review_decision", result)
        emit_done("apply_genre_review_decision", True, f"{result['term']}: {result['status']}")
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("apply_genre_review_decision", False, str(e))
```

- [ ] **Step 4: Register the handlers**

In `TRACKED_COMMAND_HANDLERS` (worker.py:2407), after `"edit_genres": handle_edit_genres,` add:

```python
    "scan_genre_review": handle_scan_genre_review,
    "get_genre_review_queue": handle_get_genre_review_queue,
    "apply_genre_review_decision": handle_apply_genre_review_decision,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_worker_review_queue.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/worker.py tests/unit/test_worker_review_queue.py
git commit -m "feat(genre-review): worker commands scan/get/apply for the review queue"
```

---

## Task 4: Schemas + FastAPI endpoints

**Files:**
- Modify: `src/playlist_web/schemas.py` (append)
- Modify: `src/playlist_web/app.py` (imports line 21-33; endpoints after `/api/tools/enrich`, line 204)

- [ ] **Step 1: Add `ReviewDecisionRequest` to `schemas.py`**

Append at the end of `src/playlist_web/schemas.py`:

```python
class ReviewDecisionRequest(BaseModel):
    release_key: str
    term: str
    decision: str  # accept | reject | revert — validated by the worker
```

- [ ] **Step 2: Add the endpoints to `app.py`**

Add `ReviewDecisionRequest` to the `.schemas` import block (alphabetical, after `PlexExportRequest`):

```python
from .schemas import (
    AnalyzeToolRequest,
    BlacklistArtistRequest,
    BlacklistFetchResponse,
    BlacklistRequest,
    EditGenresRequest,
    EnrichToolRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsRequest,
    ReplaceSuggestionsResponse,
    ReviewDecisionRequest,
)
```

After the `tools_enrich` handler (line 204), add:

```python
    @app.post("/api/review/scan")
    async def review_scan() -> dict:
        job_id = registry.create(request_params={"tool": "scan_genre_review"})
        try:
            await bridge.submit({"cmd": "scan_genre_review", "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.get("/api/review/queue")
    async def review_queue(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            result = await bridge.command({
                "cmd": "get_genre_review_queue",
                "search": search,
                "limit": limit,
                "offset": offset,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return result

    @app.post("/api/review/decision")
    async def review_decision(body: ReviewDecisionRequest) -> dict:
        if not body.release_key.strip() or not body.term.strip():
            raise HTTPException(status_code=422, detail="release_key and term are required")
        try:
            result = await bridge.command({
                "cmd": "apply_genre_review_decision",
                "release_key": body.release_key,
                "term": body.term,
                "decision": body.decision,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}
```

- [ ] **Step 3: Run the existing web API tests for regressions**

Run: `pytest tests/integration/test_web_api.py tests/integration/test_web_tools_api.py -q`
Expected: all pass (additive changes only).

- [ ] **Step 4: Commit**

```bash
git add src/playlist_web/schemas.py src/playlist_web/app.py
git commit -m "feat(genre-review): /api/review scan, queue, and decision endpoints"
```

---

## Task 5: Fake worker branches + integration tests

**Files:**
- Modify: `tests/fixtures/fake_worker.py` (before the `else` fallback, line 113)
- Create: `tests/integration/test_web_review_api.py`

- [ ] **Step 1: Add the fake worker branches**

In `tests/fixtures/fake_worker.py`, before the final `else`, add:

```python
        elif name == "scan_genre_review":
            emit({"type": "progress", "stage": "scan_genre_review", "current": 1, "total": 2,
                  "detail": "acetone – cindy", "request_id": rid, "job_id": jid})
            emit({"type": "result", "result_type": "scan_genre_review",
                  "request_id": rid, "job_id": jid,
                  "releases_scanned": 2, "new_terms": 3, "pruned_terms": 0, "pending_terms": 3})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "Scanned 2 releases",
                  "request_id": rid, "job_id": jid})
        elif name == "get_genre_review_queue":
            emit({"type": "result", "result_type": "genre_review_queue",
                  "request_id": rid, "job_id": jid,
                  "releases": [{
                      "release_key": "acetone::cindy", "artist": "acetone", "album": "cindy",
                      "pending": [
                          {"term": "slowcore", "confidence": 0.4, "basis": "hybrid_fusion",
                           "sources": ["lastfm_tags"], "reason": "uncertain", "status": "pending"},
                          {"term": "sadcore", "confidence": 0.3, "basis": "layered_taxonomy",
                           "sources": ["discogs"], "reason": "Unknown layered taxonomy term.",
                           "status": "pending"},
                      ],
                      "decided": [],
                  }],
                  "pending_releases": 1, "pending_terms": 2})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "2 pending",
                  "request_id": rid, "job_id": jid})
        elif name == "apply_genre_review_decision":
            decision = cmd.get("decision", "accept")
            status = {"accept": "accepted", "reject": "rejected", "revert": "pending"}.get(decision)
            if status is None:
                emit({"type": "error", "message": f"invalid decision: {decision}",
                      "request_id": rid, "job_id": jid})
                emit({"type": "done", "cmd": name, "ok": False, "request_id": rid, "job_id": jid})
            else:
                emit({"type": "result", "result_type": "genre_review_decision",
                      "request_id": rid, "job_id": jid,
                      "release_key": cmd.get("release_key"), "term": cmd.get("term"),
                      "decision": decision, "status": status})
                emit({"type": "done", "cmd": name, "ok": True, "detail": status,
                      "request_id": rid, "job_id": jid})
```

- [ ] **Step 2: Create `tests/integration/test_web_review_api.py`**

```python
# tests/integration/test_web_review_api.py
"""Integration tests for /api/review/* endpoints."""
import sys
import time

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def _wait_done(client, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("success", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    return client.get(f"/api/jobs/{job_id}").json()


def test_scan_creates_job_and_result_lands_in_tool_result():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/review/scan")
        assert resp.status_code == 200
        job = _wait_done(client, resp.json()["job_id"])
        assert job["status"] == "success"
        assert job["tool_result"]["result_type"] == "scan_genre_review"
        assert job["tool_result"]["pending_terms"] == 3


def test_queue_returns_releases_and_counts():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/review/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending_terms"] == 2
        assert data["releases"][0]["release_key"] == "acetone::cindy"
        assert len(data["releases"][0]["pending"]) == 2


def test_decision_round_trip():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/review/decision", json={
            "release_key": "acetone::cindy", "term": "slowcore", "decision": "accept",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


def test_decision_invalid_decision_is_422():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/review/decision", json={
            "release_key": "acetone::cindy", "term": "slowcore", "decision": "maybe",
        })
        assert resp.status_code == 422
```

- [ ] **Step 3: Run the new integration tests**

Run: `pytest tests/integration/test_web_review_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Run the fast suite for regressions**

Run: `pytest -m "not slow" -q`
Expected: same pass count as before plus the new tests; no new failures.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/fake_worker.py tests/integration/test_web_review_api.py
git commit -m "test(genre-review): fake worker branches and /api/review integration tests"
```

---

## Task 6: Frontend types + API client

**Files:**
- Modify: `web/src/lib/types.ts` (append)
- Modify: `web/src/lib/api.ts`

- [ ] **Step 1: Append types to `web/src/lib/types.ts`**

```typescript
export interface ReviewTermOut {
  term: string;
  confidence: number | null;
  basis: string;
  sources: string[];
  reason: string;
  status: "pending" | "accepted" | "rejected";
}

export interface ReviewReleaseOut {
  release_key: string;
  artist: string;
  album: string;
  pending: ReviewTermOut[];
  decided: ReviewTermOut[];
}

export interface ReviewQueueResponse {
  releases: ReviewReleaseOut[];
  pending_releases: number;
  pending_terms: number;
}

export interface ReviewDecisionRequest {
  release_key: string;
  term: string;
  decision: "accept" | "reject" | "revert";
}
```

- [ ] **Step 2: Add client methods to `web/src/lib/api.ts`**

Add `ReviewDecisionRequest` and `ReviewQueueResponse` to the type import block, then add to the `api` object after `enrich`:

```typescript
  async reviewScan(): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/review/scan", { method: "POST" }));
  },
  async reviewQueue(search = "", limit = 50, offset = 0): Promise<ReviewQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/review/queue?${params}`));
  },
  async reviewDecision(req: ReviewDecisionRequest): Promise<{ ok: boolean; status: string }> {
    return jsonOrThrow(await fetch("/api/review/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
```

- [ ] **Step 3: Type-check**

Run (from `web/`): `npx tsc --noEmit`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(genre-review): review queue types and API client methods"
```

---

## Task 7: GenreReviewPanel component + AdvancedPanel wiring

**Files:**
- Create: `web/src/components/GenreReviewPanel.tsx`
- Modify: `web/src/components/AdvancedPanel.tsx`

The right panel is narrow — single-column layout. Styling follows the existing semantic tokens used in `AdvancedPanel`/`BlacklistPanel` (`text-muted`, `text-faint`, `bg-panel2`, `border-border`, `text-accent`, `text-danger`, `bg-chip`).

- [ ] **Step 1: Create `web/src/components/GenreReviewPanel.tsx`**

```tsx
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useWorkerEvents } from "../lib/ws";
import type {
  ReviewQueueResponse,
  ReviewReleaseOut,
  ReviewTermOut,
  WsEvent,
} from "../lib/types";

const BASIS_LABEL: Record<string, string> = {
  layered_taxonomy: "unknown term",
  hybrid_provisional: "provisional",
  hybrid_fusion: "uncertain",
};

function TermRow({
  term,
  onDecide,
  focused,
}: {
  term: ReviewTermOut;
  onDecide: (decision: "accept" | "reject") => void;
  focused: boolean;
}) {
  return (
    <div
      className={[
        "px-2 py-1.5 rounded border",
        focused ? "border-accent/50 bg-panel2" : "border-border",
      ].join(" ")}
    >
      <div className="flex items-center gap-2">
        <span className="text-text text-xs font-medium flex-1 truncate">{term.term}</span>
        <span className="text-faint text-[9px] uppercase tracking-wide">
          {BASIS_LABEL[term.basis] ?? term.basis}
        </span>
        {term.confidence != null && (
          <span className="text-faint text-[10px]">{term.confidence.toFixed(2)}</span>
        )}
      </div>
      {term.reason && <div className="text-muted text-[10px] mt-0.5">{term.reason}</div>}
      {term.sources.length > 0 && (
        <div className="text-faint text-[9px] mt-0.5">{term.sources.join(" · ")}</div>
      )}
      <div className="flex gap-1.5 mt-1.5">
        <button
          onClick={() => onDecide("accept")}
          className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold"
        >
          Accept (A)
        </button>
        <button
          onClick={() => onDecide("reject")}
          className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text"
        >
          Reject (R)
        </button>
      </div>
    </div>
  );
}

export function GenreReviewPanel() {
  const [data, setData] = useState<ReviewQueueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [scanJobId, setScanJobId] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState("");

  const load = useCallback(async (q: string) => {
    try {
      const page = await api.reviewQueue(q);
      setData(page);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load(search);
  }, [load, search]);

  useWorkerEvents(
    useCallback(
      (e: WsEvent) => {
        if (!scanJobId || e.job_id !== scanJobId) return;
        if (e.type === "progress") {
          const cur = e["current"] as number | undefined;
          const total = e["total"] as number | undefined;
          const detail = (e["detail"] as string) ?? "";
          setScanProgress(`${cur ?? "?"}/${total ?? "?"} ${detail}`);
        }
        if (e.type === "done") {
          setScanJobId(null);
          setScanProgress("");
          load(search);
        }
      },
      [scanJobId, load, search]
    )
  );

  async function startScan() {
    setError(null);
    try {
      const { job_id } = await api.reviewScan();
      setScanJobId(job_id);
      setScanProgress("starting…");
    } catch (e) {
      setError(String(e));
    }
  }

  const decide = useCallback(
    async (release: ReviewReleaseOut, term: ReviewTermOut, decision: "accept" | "reject" | "revert") => {
      // Optimistic update; reload on error.
      setData((prev) => {
        if (!prev) return prev;
        const releases = prev.releases
          .map((r) => {
            if (r.release_key !== release.release_key) return r;
            if (decision === "revert") {
              const t = r.decided.find((x) => x.term === term.term);
              if (!t) return r;
              return {
                ...r,
                decided: r.decided.filter((x) => x.term !== term.term),
                pending: [...r.pending, { ...t, status: "pending" as const }],
              };
            }
            const t = r.pending.find((x) => x.term === term.term);
            if (!t) return r;
            const status = decision === "accept" ? ("accepted" as const) : ("rejected" as const);
            return {
              ...r,
              pending: r.pending.filter((x) => x.term !== term.term),
              decided: [...r.decided, { ...t, status }],
            };
          })
          .filter((r) => r.pending.length > 0 || r.release_key === release.release_key);
        const delta = decision === "revert" ? 1 : -1;
        return { ...prev, releases, pending_terms: prev.pending_terms + delta };
      });
      try {
        await api.reviewDecision({
          release_key: release.release_key,
          term: term.term,
          decision,
        });
      } catch (e) {
        setError(String(e));
        load(search);
      }
    },
    [load, search]
  );

  const releases = data?.releases ?? [];
  const selected =
    releases.find((r) => r.release_key === selectedKey) ?? releases[0] ?? null;

  function onKeyDown(e: React.KeyboardEvent) {
    if (!selected || selected.pending.length === 0) return;
    const key = e.key.toLowerCase();
    if (key === "a" || key === "r") {
      e.preventDefault();
      decide(selected, selected.pending[0], key === "a" ? "accept" : "reject");
    }
  }

  return (
    <div className="h-full flex flex-col p-3 gap-2 outline-none" tabIndex={0} onKeyDown={onKeyDown}>
      {/* Header */}
      <div className="flex items-center gap-2">
        <div className="text-muted text-xs flex-1">
          {data ? `${data.pending_releases} releases · ${data.pending_terms} terms` : "…"}
        </div>
        {scanJobId ? (
          <span className="text-faint text-[10px] truncate max-w-[160px]">{scanProgress}</span>
        ) : (
          <button
            onClick={startScan}
            className="text-[10px] px-2 py-1 rounded border border-border text-muted hover:text-text"
          >
            Scan
          </button>
        )}
      </div>
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Filter artist / album…"
        className="bg-panel2 border border-border rounded text-[11px] text-text px-2 py-1
                   placeholder:text-faint outline-none"
      />
      {error && <div className="text-danger text-[10px]">{error}</div>}

      {/* Empty state */}
      {data && releases.length === 0 && !scanJobId && (
        <div className="text-faint text-xs p-3">
          Queue is empty. Run a scan to find genre terms needing review.
        </div>
      )}

      {/* Release list */}
      <div className="flex-1 overflow-auto flex flex-col gap-1">
        {releases.map((r) => (
          <div key={r.release_key}>
            <button
              onClick={() =>
                setSelectedKey(selected?.release_key === r.release_key ? null : r.release_key)
              }
              className={[
                "w-full text-left px-2 py-1 rounded flex items-center gap-2",
                selected?.release_key === r.release_key
                  ? "bg-panel2 text-text"
                  : "text-muted hover:text-text",
              ].join(" ")}
            >
              <span className="text-xs flex-1 truncate">
                {r.artist} – {r.album}
              </span>
              <span className="bg-chip text-chipText text-[10px] px-1.5 rounded-full">
                {r.pending.length}
              </span>
            </button>
            {selected?.release_key === r.release_key && (
              <div className="flex flex-col gap-1 mt-1 mb-2 ml-1">
                {r.pending.map((t, i) => (
                  <TermRow
                    key={t.term}
                    term={t}
                    focused={i === 0}
                    onDecide={(d) => decide(r, t, d)}
                  />
                ))}
                {r.pending.length > 1 && (
                  <div className="flex gap-1.5 mt-0.5">
                    <button
                      onClick={() => r.pending.slice().forEach((t) => decide(r, t, "accept"))}
                      className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text"
                    >
                      Accept all
                    </button>
                    <button
                      onClick={() => r.pending.slice().forEach((t) => decide(r, t, "reject"))}
                      className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text"
                    >
                      Reject all
                    </button>
                  </div>
                )}
                {r.decided.length > 0 && (
                  <details className="text-[10px] text-faint">
                    <summary className="cursor-pointer select-none">
                      {r.decided.length} decided
                    </summary>
                    <div className="flex flex-col gap-0.5 mt-1">
                      {r.decided.map((t) => (
                        <div key={t.term} className="flex items-center gap-2 px-2">
                          <span className="flex-1 truncate">{t.term}</span>
                          <span className={t.status === "accepted" ? "text-accent" : "text-danger"}>
                            {t.status}
                          </span>
                          <button
                            onClick={() => decide(r, t, "revert")}
                            className="underline hover:text-text"
                          >
                            revert
                          </button>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire into `AdvancedPanel.tsx`**

Replace the stub. Add the import:

```tsx
import { GenreReviewPanel } from "./GenreReviewPanel";
```

Replace:

```tsx
        {tab === "review" && (
          <div className="p-3 text-xs text-muted">Genre review lands in a later phase.</div>
        )}
```

with:

```tsx
        {tab === "review" && <GenreReviewPanel />}
```

- [ ] **Step 3: Type-check and build**

Run (from `web/`): `npx tsc --noEmit && npm run build`
Expected: 0 errors, build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/GenreReviewPanel.tsx web/src/components/AdvancedPanel.tsx
git commit -m "feat(genre-review): GenreReviewPanel replaces the stub tab"
```

---

## Task 8: Playwright smoke + full verification

**Files:**
- Create: `web/tests/review.spec.ts`

- [ ] **Step 1: Create `web/tests/review.spec.ts`**

```typescript
import { test, expect } from "@playwright/test";

test("Genre Review tab lists the queue and accepting a term updates counts", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /genre review/i }).click();

  // Queue loads from the fake worker
  await expect(page.getByText("1 releases · 2 terms")).toBeVisible();
  await expect(page.getByText("acetone – cindy")).toBeVisible();

  // Expand the release and accept the first term
  await page.getByText("acetone – cindy").click();
  await expect(page.getByText("slowcore")).toBeVisible();
  await page.getByRole("button", { name: /accept \(a\)/i }).first().click();

  // Optimistic update: pending count drops, decided section appears
  await expect(page.getByText("1 releases · 1 terms")).toBeVisible();
  await expect(page.getByText("1 decided")).toBeVisible();
});

test("Scan button kicks off a job and re-enables on completion", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /genre review/i }).click();
  await page.getByRole("button", { name: /^scan$/i }).click();
  // Fake worker completes immediately; Scan button returns
  await expect(page.getByRole("button", { name: /^scan$/i })).toBeVisible({ timeout: 10000 });
});
```

- [ ] **Step 2: Run the Playwright tests**

Run (from `web/`): `npx playwright test tests/review.spec.ts --reporter=list`
Expected: both tests pass.

- [ ] **Step 3: Full fast suite + lint**

```
pytest -m "not slow" -q
ruff check src/ tests/
```

Expected: no new failures, no lint errors.

- [ ] **Step 4: Manual smoke against the real DB**

1. **Back up the sidecar first** (spec §2 backup discipline):
   `Copy-Item data/ai_genre_enrichment.db ("data/ai_genre_enrichment.db.bak_" + (Get-Date -Format "yyyyMMdd_HHmmss"))`
2. Restart `python tools/serve_web.py` (worker code changed) — `web/dist` was rebuilt in Task 7.
3. Open the GUI → right panel → Genre Review tab. Expect the empty state.
4. Click **Scan**. Progress should tick per release (~4k releases, ~9 min — cancelling partway is fine; partial results persist).
5. Confirm releases appear sorted by pending count; accept one term and reject another; confirm counts update and the terms move to "decided".
6. Verify persistence: `python -c "import sqlite3; c=sqlite3.connect('data/ai_genre_enrichment.db'); print(c.execute(\"SELECT status, COUNT(*) FROM ai_genre_review_queue GROUP BY status\").fetchall())"`
7. Verify the override landed: check `ai_genre_user_overrides` for the release you decided.

- [ ] **Step 5: Commit**

```bash
git add web/tests/review.spec.ts
git commit -m "test(genre-review): Playwright smoke for the review tab"
```

---

## Self-Review

**Spec coverage:**
- ✅ §2 table schema + decided-rows-kept — Task 1 (CREATE TABLE, `sync` never touches decided rows, tested)
- ✅ §2 sidecar backup before migration — Task 8 Step 4 (migration is additive `CREATE TABLE IF NOT EXISTS`; backup precedes the first real-DB run)
- ✅ §3 scan: enumerate source_pages, upsert pending, prune stale, skip override-settled, progress + cancel at release boundaries, summary result — Tasks 1–3
- ✅ §4 decisions: merge-not-replace override, rebuild signature, status transitions incl. revert — Task 2 (`apply_review_decision`, tested against pre-existing override)
- ✅ §4 no LLM/external calls — domain module imports only storage + layered_assignment
- ✅ §5 endpoints `/api/review/{scan,queue,decision}` — Task 4
- ✅ §5 GUI: header counts, Scan with WS progress, search, sorted release list, term rows w/ confidence+basis+sources+reason, accept/reject, accept-all/reject-all, decided+revert, A/R keys, empty state, optimistic updates — Task 7
- ✅ §5 known 409 limitation — endpoints return 409 detail strings; panel surfaces them inline as errors
- ✅ §6 every test layer — Tasks 1, 2, 3, 5, 8
- ⚠️ §5 "finishing a release auto-advances": implicit — a fully-decided release drops from the list on next reload and selection falls back to `releases[0]`. Good enough for v1; noted, not a separate mechanism.

**Placeholder scan:** none — all steps carry complete code/commands.

**Type consistency:** command names (`scan_genre_review` / `get_genre_review_queue` / `apply_genre_review_decision`) match across worker registration, endpoints, fake worker, and tests. `result_type` strings (`scan_genre_review`, `genre_review_queue`, `genre_review_decision`) match between handlers, fake worker, and assertions. Storage method names match between Tasks 1–3. TS `ReviewQueueResponse`/`ReviewTermOut` field names match the Python page dict (`releases[].pending[].term/confidence/basis/sources/reason/status`, `pending_releases`, `pending_terms`).

**Caveats for the implementer:**
- `upsert_source_page` kwargs in the tests were verified against the current signature (`storage.py:1054`) — they match as written.
- If `mypy` complains about `review_queue.py`, type the `store` params against `SidecarStore` directly instead of loosening config (`[[tool.mypy.overrides]]` is for clean modules only, per CLAUDE.md).
