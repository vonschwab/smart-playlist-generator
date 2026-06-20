# Escalation Review Panel (SP2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repurpose the web Genre Review panel from tag-grain (`ai_genre_review_queue`) to album-grain (`adjudication_escalations`): per-album accept/edit/reject + an in-panel "Publish decided" button.

**Architecture:** Reuse the panel's shell + bridge→worker→FastAPI→`api.ts` plumbing; replace the term-grain internals and data source. A new read-only `EscalationQueue.list_page` feeds the untracked reader-thread handler; decisions call `EscalationQueue.record_decision`/`revert`; a tracked `publish_decided` job backs up `metadata.db` and runs `publish()`.

**Tech Stack:** Python (FastAPI, the NDJSON worker bridge, SQLite sidecar), React + TypeScript + Tailwind (Studio Dark), pytest + Playwright.

## Global Constraints

- **`publish()` is the only authority writer.** The panel never writes `release_effective_genres` directly; the "Publish decided" button runs `publish()` via a tracked job that first backs up `metadata.db` (timestamped).
- **Untracked read handlers run on the reader thread and must be strictly read-only** (no DDL/writes, no blocking behind a write lock) — the 2026-06-12 review-queue timeout incident. The escalation read path uses `EscalationQueue.list_page` which opens `mode=ro` and never does DDL.
- **Untracked handlers emit events with `request_id` from `cmd_data` and `job_id` explicitly `None`** (so an inline run during a tracked job never corrupts that job).
- **Decisions never invent genres** — unknown edit terms are skipped by the materializer.
- **No live `metadata.db`/sidecar writes in tests** — temp DBs + the fake worker only.
- **Studio Dark, raw Tailwind** per the existing panels; no new UI deps.

## File map

| File | Change |
|---|---|
| `src/ai_genre_enrichment/escalation_queue.py` | add module fn `list_page` (read-only) + method `revert` |
| `src/playlist_gui/worker.py` | add 4 handlers + register (3 untracked, 1 tracked) |
| `src/playlist_web/app.py` | repoint 4 routes; remove `/api/review/scan`; new request model |
| `web/src/lib/types.ts` | add `EscalationOut`/`EscalationQueueResponse`; change `ReviewDecisionRequest` |
| `web/src/lib/api.ts` | change `reviewDecision` body; add `reviewPublish`; drop `reviewScan` |
| `web/src/components/GenreReviewPanel.tsx` | rewrite internals to album-grain |
| `tests/fixtures/fake_worker.py` | new branches for the 4 commands |
| `tests/unit/test_escalation_queue.py` | `list_page` + `revert` tests (extends SP1 file) |
| `tests/unit/test_worker_escalation_review.py` | new — worker-handler round-trips |
| `tests/integration/test_web_review_api.py` | repoint to album-grain endpoints |
| `web/tests/review.spec.ts` | repoint Playwright to album-grain |

## Reference signatures (exist — consume, don't redefine)

- `EscalationQueue(db_path)` with `record_decision(album_id, decision, *, genres=None, sidecar_store, taxonomy, model="review")`, `get(album_id)`, `_mark`, `_row_to_dict`, `close()`. Table `adjudication_escalations` (columns: album_id PK, release_key, artist, album, prior_observed_leaf_json, proposed_genres_json, escalate_reason, dropped_file_tags_json, prompt_version, model, input_hash, status, decision_genres_json, created_at, decided_at).
- `SidecarStore(path).replace_layered_assignments_for_release(*, release_id, artist, album, genre_assignments, facet_assignments)`.
- `src.ai_genre_enrichment.normalization`: `normalize_release_artist`, `normalize_release_name`.
- `src.ai_genre_enrichment.layered_taxonomy.load_default_layered_taxonomy()`.
- `src.genre.genre_publish.publish(metadata_db, sidecar_db, dry_run=False) -> PublishStats`.
- Worker: `SIDECAR_DB_PATH`, `METADATA_DB_PATH` constants; `emit_event(dict)`, `emit_progress(cmd, current, total, detail)`, `emit_result(cmd, dict)`, `emit_done(cmd, ok, detail, summary=...)`, `emit_error(msg, tb=None)`, `check_cancelled()`, `CancellationError`; `TRACKED_COMMAND_HANDLERS`, `UNTRACKED_COMMAND_HANDLERS`.
- Worker-handler test pattern (`tests/unit/test_worker_review_queue.py`): `monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(tmp))`, call `handle_x(cmd_data)`, read events via `capsys` → `[json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]`.
- App route pattern: `await bridge.command({"cmd": ..., **payload}, untracked=True)` (409 `BridgeBusy`, 502 `WorkerCommandError`); tracked: `registry.create(...)` + `await bridge.submit({"cmd":..., "job_id":...})` (409 `BridgeBusy`).

---

## Task 1: `EscalationQueue.list_page` (read-only) + `revert`

**Files:**
- Modify: `src/ai_genre_enrichment/escalation_queue.py`
- Test: `tests/unit/test_escalation_queue.py` (append)

**Interfaces:**
- Produces:
  - `list_page(db_path, *, status="pending", search=None, limit=50, offset=0) -> dict` — module-level fn; opens read-only; returns `{escalations: [row...], pending_albums: int, decided_albums: int}`. `status` is `"pending"` or `"decided"` (the latter = accepted|edited|rejected).
  - `EscalationQueue.revert(album_id, *, sidecar_store) -> None` — re-open to pending + un-materialize.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_escalation_queue.py
from src.ai_genre_enrichment.escalation_queue import list_page


def test_list_page_readonly_returns_pending(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q, album_id="a1", proposed=("shoegaze",))
    page = list_page(tmp_path / "side.db", status="pending")
    assert page["pending_albums"] == 1
    assert page["decided_albums"] == 0
    assert page["escalations"][0]["album_id"] == "a1"
    assert page["escalations"][0]["proposed_genres"][0]["term"] == "shoegaze"


def test_list_page_decided_and_search(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q, album_id="a1", proposed=("shoegaze",))
    q._mark("a1", status="accepted", decision_genres=["shoegaze"])
    _enq(q, album_id="a2", proposed=("dream pop",))
    decided = list_page(tmp_path / "side.db", status="decided")
    assert [e["album_id"] for e in decided["escalations"]] == ["a1"]
    pending = list_page(tmp_path / "side.db", status="pending", search="slowdive")
    assert [e["album_id"] for e in pending["escalations"]] == ["a1", "a2"]  # _enq uses Slowdive/Souvlaki for both


def test_list_page_missing_table_is_empty(tmp_path):
    # a sidecar that exists but has no escalations table yet
    import sqlite3
    (tmp_path / "empty.db").write_bytes(b"")
    sqlite3.connect(tmp_path / "empty.db").close()
    page = list_page(tmp_path / "empty.db", status="pending")
    assert page == {"escalations": [], "pending_albums": 0, "decided_albums": 0}


def test_revert_reopens_and_unmaterializes(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    side = tmp_path / "side.db"
    store = SidecarStore(str(side)); store.initialize()
    q = EscalationQueue(side)
    _enq(q, album_id="a1", proposed=("shoegaze",))
    q.record_decision("a1", "accept", sidecar_store=store, taxonomy=load_default_layered_taxonomy())
    import sqlite3 as _s
    before = _s.connect(side).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki'").fetchone()[0]
    assert before >= 1
    q.revert("a1", sidecar_store=store)
    assert q.get("a1")["status"] == "pending"
    assert q.get("a1")["decided_at"] is None
    after = _s.connect(side).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki'").fetchone()[0]
    assert after == 0
```

(Note: the SP1 `_enq` helper enqueues with `release_key="slowdive::souvlaki", artist="Slowdive", album="Souvlaki"` for every album_id, so the search test matches both.)

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_escalation_queue.py -k "list_page or revert" -q`
Expected: FAIL — `cannot import name 'list_page'` / `EscalationQueue has no attribute 'revert'`.

- [ ] **Step 3: Implement `list_page` + a shared row mapper + `revert`**

At the top of `escalation_queue.py`, add a module-level row mapper and refactor the class to use it:

```python
def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "album_id": row["album_id"], "release_key": row["release_key"],
        "artist": row["artist"], "album": row["album"],
        "prior_observed_leaf": json.loads(row["prior_observed_leaf_json"] or "[]"),
        "proposed_genres": json.loads(row["proposed_genres_json"] or "[]"),
        "escalate_reason": row["escalate_reason"],
        "dropped_file_tags": json.loads(row["dropped_file_tags_json"] or "[]"),
        "prompt_version": row["prompt_version"], "model": row["model"],
        "input_hash": row["input_hash"], "status": row["status"],
        "decision_genres": json.loads(row["decision_genres_json"] or "null"),
        "created_at": row["created_at"], "decided_at": row["decided_at"],
    }
```

Change the class method `_row_to_dict(self, row)` body to `return _row_to_dict(row)`.

Add the read-only page function (module level):

```python
_DECIDED = "('accepted','edited','rejected')"


def list_page(db_path, *, status: str = "pending", search=None, limit: int = 50, offset: int = 0) -> dict:
    """Read-only page of escalations. Opens mode=ro and does NO DDL — safe on the
    worker reader thread (2026-06-12 timeout-incident rule). `status` is 'pending'
    or 'decided' (accepted|edited|rejected)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        where = "status='pending'" if status == "pending" else f"status IN {_DECIDED}"
        params: list = []
        if search:
            where += " AND (lower(artist) LIKE ? OR lower(album) LIKE ?)"
            like = f"%{search.lower()}%"
            params += [like, like]
        rows = conn.execute(
            f"SELECT * FROM adjudication_escalations WHERE {where} "
            "ORDER BY created_at LIMIT ? OFFSET ?",
            (*params, int(limit), int(offset)),
        ).fetchall()
        pending = conn.execute(
            "SELECT COUNT(*) FROM adjudication_escalations WHERE status='pending'").fetchone()[0]
        decided = conn.execute(
            f"SELECT COUNT(*) FROM adjudication_escalations WHERE status IN {_DECIDED}").fetchone()[0]
        return {"escalations": [_row_to_dict(r) for r in rows],
                "pending_albums": int(pending), "decided_albums": int(decided)}
    except sqlite3.OperationalError:
        # table not created yet (no escalations enqueued)
        return {"escalations": [], "pending_albums": 0, "decided_albums": 0}
    finally:
        conn.close()
```

Add the `revert` method to `EscalationQueue` (after `record_decision`):

```python
    def revert(self, album_id: str, *, sidecar_store: Any) -> None:
        from .normalization import normalize_release_artist, normalize_release_name

        row = self.get(album_id)
        if row is None:
            raise KeyError(f"no escalation queued for album_id={album_id!r}")
        release_id = row["release_key"] or (
            f"{normalize_release_artist(row['artist'])}::{normalize_release_name(row['album'])}")
        sidecar_store.replace_layered_assignments_for_release(
            release_id=release_id, artist=row["artist"], album=row["album"],
            genre_assignments=[], facet_assignments=[])
        self._c.execute(
            "UPDATE adjudication_escalations "
            "SET status='pending', decision_genres_json=NULL, decided_at=NULL WHERE album_id=?",
            (album_id,))
        self._c.commit()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_escalation_queue.py -q`
Expected: PASS (SP1 cases + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/escalation_queue.py tests/unit/test_escalation_queue.py
git commit -m "feat(genre): EscalationQueue.list_page (read-only) + revert"
```

---

## Task 2: Worker handlers — queue/completed/decision (untracked)

**Files:**
- Modify: `src/playlist_gui/worker.py`
- Test: `tests/unit/test_worker_escalation_review.py` (create)

**Interfaces:**
- Consumes: `list_page`, `EscalationQueue`, `load_default_layered_taxonomy`, `SidecarStore`, `SIDECAR_DB_PATH`, `emit_event`.
- Produces: handlers `handle_get_escalation_queue`, `handle_get_escalation_completed`,
  `handle_apply_escalation_decision`; registered in `UNTRACKED_COMMAND_HANDLERS` under
  `get_escalation_queue` / `get_escalation_completed` / `apply_escalation_decision`. Result `result_type`s:
  `escalation_queue`, `escalation_completed`, `escalation_decision`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_worker_escalation_review.py
"""Worker handler round-trips for the album-grain escalation review commands."""
import json

from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.storage import SidecarStore
from src.playlist_gui.worker import (
    handle_apply_escalation_decision,
    handle_get_escalation_queue,
)


def _events(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]


def _seed(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar)); store.initialize()
    q = EscalationQueue(sidecar)
    q.enqueue(album_id="a1", release_key="slowdive::souvlaki", artist="Slowdive",
              album="Souvlaki", prior_observed_leaf=["indie rock"],
              proposed_genres=[{"term": "shoegaze", "confidence": 0.9}],
              escalate_reason="sparse", dropped_file_tags=["dream pop"],
              prompt_version="pv", model="sonnet", input_hash="h1")
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))
    return sidecar


def test_get_escalation_queue_emits_result(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_get_escalation_queue({"cmd": "get_escalation_queue", "request_id": "r1"})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "escalation_queue"
    assert result["pending_albums"] == 1
    assert result["escalations"][0]["album_id"] == "a1"
    assert result["escalations"][0]["dropped_file_tags"] == ["dream pop"]
    assert result["job_id"] is None
    assert done["ok"] is True


def test_apply_escalation_decision_accept_materializes(tmp_path, monkeypatch, capsys):
    sidecar = _seed(tmp_path, monkeypatch)
    handle_apply_escalation_decision({
        "cmd": "apply_escalation_decision", "request_id": "r2",
        "album_id": "a1", "decision": "accept",
    })
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    import sqlite3
    n = sqlite3.connect(sidecar).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'").fetchone()[0]
    assert n >= 1
    assert EscalationQueue(sidecar).get("a1")["status"] == "accepted"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_worker_escalation_review.py -q`
Expected: FAIL — `cannot import name 'handle_get_escalation_queue'`.

- [ ] **Step 3: Implement the three handlers (place beside `handle_apply_genre_review_decision`)**

```python
def handle_get_escalation_queue(cmd_data: Dict[str, Any]) -> None:
    """Album-grain escalation queue page. UNTRACKED + read-only (reader thread)."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.escalation_queue import list_page
        page = list_page(
            SIDECAR_DB_PATH, status="pending",
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0))
        emit_event({"type": "result", "result_type": "escalation_queue",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_escalation_queue", "ok": True,
                    "detail": f"{page['pending_albums']} pending",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_escalation_queue", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_get_escalation_completed(cmd_data: Dict[str, Any]) -> None:
    """Decided escalations page. UNTRACKED + read-only."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.escalation_queue import list_page
        page = list_page(
            SIDECAR_DB_PATH, status="decided",
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0))
        emit_event({"type": "result", "result_type": "escalation_completed",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_escalation_completed", "ok": True,
                    "detail": f"{page['decided_albums']} decided",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_escalation_completed", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_apply_escalation_decision(cmd_data: Dict[str, Any]) -> None:
    """Apply accept/edit/reject/revert for one escalation. UNTRACKED (quick write)."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.escalation_queue import EscalationQueue
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        from src.ai_genre_enrichment.storage import SidecarStore

        album_id = str(cmd_data.get("album_id") or "")
        decision = str(cmd_data.get("decision") or "")
        genres = cmd_data.get("genres") or None
        store = SidecarStore(SIDECAR_DB_PATH)
        queue = EscalationQueue(SIDECAR_DB_PATH)
        if decision == "revert":
            queue.revert(album_id, sidecar_store=store)
            status = "pending"
        else:
            taxonomy = load_default_layered_taxonomy()
            queue.record_decision(album_id, decision, genres=genres,
                                  sidecar_store=store, taxonomy=taxonomy)
            status = {"accept": "accepted", "edit": "edited", "reject": "rejected"}[decision]
        queue.close()
        emit_event({"type": "result", "result_type": "escalation_decision",
                    "album_id": album_id, "status": status,
                    "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "apply_escalation_decision", "ok": True,
                    "detail": f"{album_id}: {status}", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "apply_escalation_decision", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})
```

Register in `UNTRACKED_COMMAND_HANDLERS` (beside the existing review entries):

```python
    "get_escalation_queue": handle_get_escalation_queue,
    "get_escalation_completed": handle_get_escalation_completed,
    "apply_escalation_decision": handle_apply_escalation_decision,
```

Note: `apply_escalation_decision` constructs `EscalationQueue(SIDECAR_DB_PATH)` which runs `CREATE TABLE IF NOT EXISTS` (DDL). This is acceptable for the *decision/write* path (it already writes); only the *read* handlers must avoid DDL, which they do via `list_page`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_worker_escalation_review.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/worker.py tests/unit/test_worker_escalation_review.py
git commit -m "feat(web): album-grain escalation queue/decision worker handlers"
```

---

## Task 3: `publish_decided` tracked handler

**Files:**
- Modify: `src/playlist_gui/worker.py`
- Test: `tests/unit/test_worker_escalation_review.py` (append)

**Interfaces:**
- Consumes: `publish`, `SIDECAR_DB_PATH`, `METADATA_DB_PATH`, `emit_progress/result/done/error`.
- Produces: `handle_publish_decided(cmd_data)` registered in `TRACKED_COMMAND_HANDLERS` as `publish_decided`;
  result `result_type='publish_decided'` carrying `PublishStats` fields.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_worker_escalation_review.py
def test_publish_decided_backs_up_and_publishes(tmp_path, monkeypatch, capsys):
    import sqlite3
    from src.playlist_gui import worker as W
    # minimal metadata.db the publisher can open (albums + the tables publish needs are
    # created by publish()'s own schema setup; an empty albums table is enough here).
    meta = tmp_path / "metadata.db"
    c = sqlite3.connect(meta)
    c.executescript("CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);")
    c.commit(); c.close()
    side = tmp_path / "sidecar.db"
    from src.ai_genre_enrichment.storage import SidecarStore
    SidecarStore(str(side)).initialize()
    monkeypatch.setattr(W, "METADATA_DB_PATH", str(meta))
    monkeypatch.setattr(W, "SIDECAR_DB_PATH", str(side))

    W.handle_publish_decided({"cmd": "publish_decided", "request_id": "r3", "job_id": "j3"})
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    # a timestamped backup was created next to metadata.db
    assert any(p.name.startswith("metadata.db.bak.") for p in tmp_path.iterdir())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_worker_escalation_review.py::test_publish_decided_backs_up_and_publishes -q`
Expected: FAIL — `module 'src.playlist_gui.worker' has no attribute 'handle_publish_decided'`.

- [ ] **Step 3: Implement the tracked handler (place beside `handle_scan_genre_review`)**

```python
def handle_publish_decided(cmd_data: Dict[str, Any]) -> None:
    """Back up metadata.db, then publish() the materialized assignments into the authority.

    Tracked job — the button click is the explicit confirmation; the backup is automatic
    (CLAUDE.md metadata.db discipline). publish() is the only authority writer.
    """
    import datetime
    import shutil

    try:
        emit_progress("publish_decided", 0, 2, "backing up metadata.db")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = f"{METADATA_DB_PATH}.bak.{ts}"
        shutil.copy2(METADATA_DB_PATH, bak)
        check_cancelled()
        emit_progress("publish_decided", 1, 2, "publishing")
        from src.genre.genre_publish import publish
        stats = publish(METADATA_DB_PATH, SIDECAR_DB_PATH, dry_run=False)
        result = {
            "graph_albums": stats.graph_albums, "legacy_albums": stats.legacy_albums,
            "total_albums": stats.total_albums, "collisions": stats.collisions,
            "backup": bak,
        }
        emit_result("publish_decided", result)
        emit_done("publish_decided", True, f"Published {stats.graph_albums} graph albums",
                  summary=f"graph={stats.graph_albums} legacy={stats.legacy_albums}")
    except CancellationError:
        emit_done("publish_decided", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("publish_decided", False, str(e))
```

Register in `TRACKED_COMMAND_HANDLERS`:

```python
    "publish_decided": handle_publish_decided,
```

If `METADATA_DB_PATH` is not already a module constant in `worker.py`, define it next to `SIDECAR_DB_PATH` using the same resolution (`resolve_db("metadata.db")` or the existing constant the other handlers use to reach metadata.db). Confirm the existing name before adding a duplicate.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_worker_escalation_review.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/worker.py tests/unit/test_worker_escalation_review.py
git commit -m "feat(web): publish_decided tracked worker job (backup + publish)"
```

---

## Task 4: API endpoints + request model + fake worker

**Files:**
- Modify: `src/playlist_web/app.py`; the request-models module that defines `ReviewDecisionRequest` (imported at `app.py:33`); `tests/fixtures/fake_worker.py`
- Test: `tests/integration/test_web_review_api.py` (repoint)

**Interfaces:**
- Produces: `GET /api/review/queue` → `get_escalation_queue`; `GET /api/review/completed` →
  `get_escalation_completed`; `POST /api/review/decision` (body `EscalationDecisionRequest{album_id, decision,
  genres?}`) → `apply_escalation_decision`; `POST /api/review/publish` → tracked `publish_decided`. `/api/review/scan` removed.

- [ ] **Step 1: Add the new request model**

Find where `ReviewDecisionRequest` is defined (it's imported in `src/playlist_web/app.py:33`). Add beside it:

```python
class EscalationDecisionRequest(BaseModel):
    album_id: str
    decision: str  # accept | edit | reject | revert
    genres: list[str] | None = None
```

- [ ] **Step 2: Write the failing integration test (repoint existing)**

Replace the body of `tests/integration/test_web_review_api.py` with album-grain round-trips (keep the file's existing `create_app`/`FAKE` harness imports):

```python
"""Album-grain escalation review API round-trips against the fake worker."""
from fastapi.testclient import TestClient

from src.playlist_web.app import create_app
from tests.fixtures.fake_worker import FAKE_WORKER_CMD


def _client():
    return TestClient(create_app(worker_cmd=FAKE_WORKER_CMD))


def test_queue_returns_escalations():
    with _client() as c:
        r = c.get("/api/review/queue")
        assert r.status_code == 200
        body = r.json()
        assert "escalations" in body and "pending_albums" in body


def test_decision_round_trip():
    with _client() as c:
        r = c.post("/api/review/decision",
                   json={"album_id": "a1", "decision": "accept"})
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_publish_returns_job_id():
    with _client() as c:
        r = c.post("/api/review/publish")
        assert r.status_code == 200
        assert "job_id" in r.json()
```

(Match `FAKE_WORKER_CMD`/`create_app` to the actual names already used at the top of the existing test file.)

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/integration/test_web_review_api.py -q`
Expected: FAIL — `/api/review/decision` 422 (old model needs `release_key`/`term`) and `/api/review/publish` 404.

- [ ] **Step 4: Repoint the routes in `app.py`**

Replace the four review routes (lines ~267–321) with:

```python
    @app.get("/api/review/queue")
    async def review_queue(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            return await bridge.command({
                "cmd": "get_escalation_queue", "search": search,
                "limit": limit, "offset": offset}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/api/review/completed")
    async def review_completed(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            return await bridge.command({
                "cmd": "get_escalation_completed", "search": search,
                "limit": limit, "offset": offset}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/review/decision")
    async def review_decision(body: EscalationDecisionRequest) -> dict:
        if not body.album_id.strip() or not body.decision.strip():
            raise HTTPException(status_code=422, detail="album_id and decision are required")
        try:
            result = await bridge.command({
                "cmd": "apply_escalation_decision", "album_id": body.album_id,
                "decision": body.decision, "genres": body.genres}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.post("/api/review/publish")
    async def review_publish() -> dict:
        job_id = registry.create(request_params={"tool": "publish_decided"})
        try:
            await bridge.submit({"cmd": "publish_decided", "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}
```

Update the import at `app.py:33` to bring in `EscalationDecisionRequest` (and drop `ReviewDecisionRequest` if now unused). Delete the old `/api/review/scan` route.

- [ ] **Step 5: Update the fake worker branches**

In `tests/fixtures/fake_worker.py`, replace the `get_genre_review_queue` / `get_genre_review_completed` / `apply_genre_review_decision` / `scan_genre_review` branches with:

```python
        elif name == "get_escalation_queue":
            emit({"type": "result", "result_type": "escalation_queue",
                  "escalations": [{"album_id": "a1", "artist": "Slowdive", "album": "Souvlaki",
                                   "prior_observed_leaf": ["indie rock"],
                                   "proposed_genres": [{"term": "shoegaze", "confidence": 0.9}],
                                   "escalate_reason": "sparse", "dropped_file_tags": [],
                                   "status": "pending"}],
                  "pending_albums": 1, "decided_albums": 0, "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "get_escalation_queue", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "get_escalation_completed":
            emit({"type": "result", "result_type": "escalation_completed",
                  "escalations": [], "pending_albums": 1, "decided_albums": 0,
                  "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "get_escalation_completed", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "apply_escalation_decision":
            emit({"type": "result", "result_type": "escalation_decision",
                  "album_id": cmd.get("album_id"), "status": "accepted",
                  "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "apply_escalation_decision", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "publish_decided":
            emit({"type": "result", "result_type": "publish_decided",
                  "graph_albums": 3325, "legacy_albums": 81, "total_albums": 3428,
                  "collisions": 31, "request_id": rid, "job_id": cmd.get("job_id")})
            emit({"type": "done", "cmd": "publish_decided", "ok": True,
                  "request_id": rid, "job_id": cmd.get("job_id")})
```

(Match the exact `emit`/`rid`/`cmd` variable names already used in the fake worker's command loop.)

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/integration/test_web_review_api.py -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add src/playlist_web/app.py tests/fixtures/fake_worker.py tests/integration/test_web_review_api.py
git commit -m "feat(web): repoint /api/review/* to album-grain escalations + publish"
```

---

## Task 5: Frontend types + api client

**Files:**
- Modify: `web/src/lib/types.ts`, `web/src/lib/api.ts`

**Interfaces:**
- Produces: TS types `EscalationOut`, `EscalationQueueResponse`; `api.reviewQueue/reviewCompleted` return
  `EscalationQueueResponse`; `api.reviewDecision({album_id, decision, genres?})`; `api.reviewPublish(): Promise<{job_id: string}>`. `reviewScan` removed.

- [ ] **Step 1: Replace the review types in `types.ts`**

Replace `ReviewTermOut` / `ReviewReleaseOut` / `ReviewQueueResponse` / `CompletedReviewResponse` /
`ReviewDecisionRequest` (lines ~155–192) with:

```typescript
export interface ProposedGenre {
  term: string;
  confidence: number | null;
}

export interface EscalationOut {
  album_id: string;
  artist: string;
  album: string;
  prior_observed_leaf: string[];
  proposed_genres: ProposedGenre[];
  escalate_reason: string;
  dropped_file_tags: string[];
  status: "pending" | "accepted" | "edited" | "rejected";
}

export interface EscalationQueueResponse {
  escalations: EscalationOut[];
  pending_albums: number;
  decided_albums: number;
}

export interface EscalationDecisionRequest {
  album_id: string;
  decision: "accept" | "edit" | "reject" | "revert";
  genres?: string[];
}
```

- [ ] **Step 2: Repoint `api.ts`**

In `web/src/lib/api.ts`: update the import line to the new type names; replace the review methods:

```typescript
  async reviewQueue(search = "", limit = 50, offset = 0): Promise<EscalationQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/review/queue?${params}`));
  },
  async reviewCompleted(search = "", limit = 50, offset = 0): Promise<EscalationQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/review/completed?${params}`));
  },
  async reviewDecision(req: EscalationDecisionRequest): Promise<{ ok: boolean; status: string }> {
    return jsonOrThrow(await fetch("/api/review/decision", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req),
    }));
  },
  async reviewPublish(): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/review/publish", { method: "POST" }));
  },
```

Delete `reviewScan`. (The component rewrite in Task 6 is what stops calling it; this task just removes it from the client.)

- [ ] **Step 3: Type-check (this is the task's test)**

Run: `npx --prefix web tsc --noEmit` (from repo root: `cd web && npx tsc --noEmit`)
Expected: errors ONLY in `GenreReviewPanel.tsx` (it still references the old types) — those are fixed in Task 6. No errors in `api.ts`/`types.ts`. If other files reference the removed types, repoint them.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(web): album-grain escalation types + api client"
```

---

## Task 6: Component rewrite — `GenreReviewPanel.tsx`

**Files:**
- Modify: `web/src/components/GenreReviewPanel.tsx`
- Test: `web/tests/review.spec.ts` (repoint)

**Interfaces:**
- Consumes: `api.reviewQueue/reviewCompleted/reviewDecision/reviewPublish`, `EscalationOut`,
  `EscalationQueueResponse`, the existing `useWorkerEvents`/`useJobReconcile` hooks.

- [ ] **Step 1: Repoint the Playwright spec (failing)**

In `web/tests/review.spec.ts`, point the fake-worker queue response at the album-grain shape (one
`EscalationOut`) and assert: the panel renders the album row (`Slowdive – Souvlaki`); selecting it shows
proposed `shoegaze`; clicking **Accept** decrements the pending count; the **Publish decided** button is
present when `decided_albums > 0`. (Mirror the existing spec's fake-worker wiring; only the payload shape and
assertions change.)

- [ ] **Step 2: Run to verify it fails**

Run: `npm --prefix web run test:e2e -- review.spec.ts` (or the repo's Playwright command)
Expected: FAIL — the panel still renders the term-grain UI / build references removed types.

- [ ] **Step 3: Rewrite the component**

Replace `web/src/components/GenreReviewPanel.tsx` with the album-grain version. Full file:

```tsx
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useJobReconcile } from "../lib/useJobReconcile";
import { useWorkerEvents } from "../lib/ws";
import type { EscalationOut, EscalationQueueResponse, WsEvent } from "../lib/types";

type View = "pending" | "completed";

function chips(items: string[], cls: string) {
  return items.map((t) => (
    <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded-full ${cls}`}>{t}</span>
  ));
}

function AlbumCard({
  esc, onDecide,
}: {
  esc: EscalationOut;
  onDecide: (decision: "accept" | "edit" | "reject", genres?: string[]) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(esc.proposed_genres.map((g) => g.term).join(", "));
  return (
    <div className="flex flex-col gap-1 mt-1 mb-2 ml-1 px-2 py-2 rounded border border-border">
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-faint text-[9px] uppercase tracking-wide">currently</span>
        {esc.prior_observed_leaf.length ? chips(esc.prior_observed_leaf, "bg-panel2 text-muted")
          : <span className="text-faint text-[10px]">—</span>}
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-faint text-[9px] uppercase tracking-wide">proposed</span>
        {esc.proposed_genres.length
          ? esc.proposed_genres.map((g) => (
              <span key={g.term} className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent/20 text-text">
                {g.term}{g.confidence != null ? ` ${g.confidence.toFixed(2)}` : ""}
              </span>))
          : <span className="text-faint text-[10px]">(none resolved — use Edit)</span>}
      </div>
      {esc.escalate_reason && <div className="text-muted text-[10px]">{esc.escalate_reason}</div>}
      {esc.dropped_file_tags.length > 0 && (
        <div className="text-danger text-[10px]">⚠ would drop your file tag: {esc.dropped_file_tags.join(", ")}</div>
      )}
      {editing ? (
        <div className="flex flex-col gap-1 mt-1">
          <input
            autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
            placeholder="genre a, genre b, …"
            className="bg-panel2 border border-border rounded text-[11px] text-text px-2 py-1 outline-none"
          />
          <div className="flex gap-1.5">
            <button
              onClick={() => onDecide("edit", draft.split(",").map((s) => s.trim()).filter(Boolean))}
              className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold"
            >Save edit</button>
            <button onClick={() => setEditing(false)}
              className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Cancel</button>
          </div>
        </div>
      ) : (
        <div className="flex gap-1.5 mt-1">
          <button onClick={() => onDecide("accept")}
            className="text-[10px] px-2 py-0.5 rounded bg-accent text-bg font-semibold">Accept (A)</button>
          <button onClick={() => setEditing(true)}
            className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Edit</button>
          <button onClick={() => onDecide("reject")}
            className="text-[10px] px-2 py-0.5 rounded border border-border text-muted hover:text-text">Reject (R)</button>
        </div>
      )}
    </div>
  );
}

export function GenreReviewPanel() {
  const [view, setView] = useState<View>("pending");
  const [data, setData] = useState<EscalationQueueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [publishJob, setPublishJob] = useState<string | null>(null);
  const [publishMsg, setPublishMsg] = useState("");
  const [sessionCount, setSessionCount] = useState(0);
  const [flash, setFlash] = useState<string | null>(null);

  const load = useCallback(async (q: string, v: View) => {
    try {
      const page = v === "pending" ? await api.reviewQueue(q) : await api.reviewCompleted(q);
      setData(page);
      setError(null);
    } catch (e) { setError(String(e)); }
  }, []);

  useEffect(() => { load(search, view); }, [search, view, load]);
  useEffect(() => { if (!flash) return; const t = setTimeout(() => setFlash(null), 1600); return () => clearTimeout(t); }, [flash]);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (!publishJob || e.job_id !== publishJob) return;
    if (e.type === "progress") {
      const d = ((e as Record<string, unknown>)["detail"] as string) ?? "";
      setPublishMsg(`publishing… ${d}`);
    }
    if (e.type === "done") { setPublishJob(null); setPublishMsg(""); load(search, view); }
  }, [publishJob, search, view, load]));

  useJobReconcile(publishJob, useCallback(() => {
    setPublishJob(null); setPublishMsg(""); load(search, view);
  }, [search, view, load]));

  const decide = useCallback(async (esc: EscalationOut, decision: "accept" | "edit" | "reject", genres?: string[]) => {
    // optimistic: drop from the pending list
    setData((prev) => prev && view === "pending"
      ? { ...prev, escalations: prev.escalations.filter((x) => x.album_id !== esc.album_id),
          pending_albums: Math.max(0, prev.pending_albums - 1), decided_albums: prev.decided_albums + 1 }
      : prev);
    try {
      await api.reviewDecision({ album_id: esc.album_id, decision, genres });
      setSessionCount((n) => n + 1);
      setFlash(`saved ✓ ${esc.artist} – ${esc.album}`);
    } catch (e) { setError(String(e)); load(search, view); }
  }, [view, search, load]);

  async function publishDecided() {
    setError(null);
    try { const { job_id } = await api.reviewPublish(); setPublishJob(job_id); setPublishMsg("starting…"); }
    catch (e) { setError(String(e)); }
  }

  const escalations = data?.escalations ?? [];
  const sel = escalations.find((x) => x.album_id === selected) ?? escalations[0] ?? null;

  function onKeyDown(e: React.KeyboardEvent) {
    if (view !== "pending" || !sel) return;
    const k = e.key.toLowerCase();
    if (k === "a") { e.preventDefault(); decide(sel, "accept"); }
    else if (k === "r") { e.preventDefault(); decide(sel, "reject"); }
    // Edit is via the card's Edit button (it owns the edit-input state).
  }

  const decidedK = data?.decided_albums ?? 0;

  return (
    <div data-testid="review-panel" className="h-full flex flex-col p-3 gap-2 outline-none" tabIndex={0} onKeyDown={onKeyDown}>
      <div className="flex items-center gap-1">
        {(["pending", "completed"] as View[]).map((v) => (
          <button key={v} onClick={() => { setView(v); setSelected(null); }}
            className={["text-[10px] px-2 py-1 rounded border capitalize",
              view === v ? "border-accent/60 bg-panel2 text-text" : "border-border text-muted hover:text-text"].join(" ")}>
            {v}
          </button>
        ))}
        <div className="flex-1" />
        {sessionCount > 0 && <span className="text-accent text-[10px]">✓ {sessionCount} this session</span>}
        {publishJob ? (
          <span className="text-faint text-[10px] truncate max-w-[160px]">{publishMsg}</span>
        ) : decidedK > 0 ? (
          <button onClick={publishDecided}
            className="text-[10px] px-2 py-1 rounded bg-accent text-bg font-semibold">
            Publish decided ({decidedK})
          </button>
        ) : null}
      </div>

      <div className="flex items-center gap-2 min-h-[16px]">
        <div className="text-muted text-xs flex-1">
          {data ? `${data.pending_albums} pending · ${data.decided_albums} decided` : "…"}
        </div>
        {flash && <span className="text-accent text-[10px] truncate max-w-[160px]">{flash}</span>}
      </div>

      <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Filter artist / album…"
        className="bg-panel2 border border-border rounded text-[11px] text-text px-2 py-1 placeholder:text-faint outline-none" />
      {error && <div className="text-danger text-[10px]">{error}</div>}

      {data && escalations.length === 0 && (
        <div className="text-faint text-xs p-3">
          {view === "pending" ? "No escalations pending — all reviewed." : "No decisions yet."}
        </div>
      )}

      <div className="flex-1 overflow-auto flex flex-col gap-1">
        {escalations.map((esc) => (
          <div key={esc.album_id}>
            <button onClick={() => setSelected(sel?.album_id === esc.album_id ? null : esc.album_id)}
              className={["w-full text-left px-2 py-1 rounded flex items-center gap-2",
                sel?.album_id === esc.album_id ? "bg-panel2 text-text" : "text-muted hover:text-text"].join(" ")}>
              <span className="text-xs flex-1 truncate">{esc.artist} – {esc.album}</span>
              {esc.dropped_file_tags.length > 0 && <span className="text-danger text-[10px]">⚠</span>}
              <span className="text-faint text-[10px] capitalize">{view === "completed" ? esc.status : ""}</span>
            </button>
            {sel?.album_id === esc.album_id && view === "pending" && (
              <AlbumCard esc={esc} onDecide={(d, g) => decide(esc, d, g)} />
            )}
            {sel?.album_id === esc.album_id && view === "completed" && (
              <div className="ml-1 mb-2 px-2 py-1 text-[10px] text-muted flex items-center gap-2">
                <span className="flex-1">decided: {(esc.proposed_genres.map((g) => g.term)).join(", ") || "—"}</span>
                <button onClick={async () => {
                  await api.reviewDecision({ album_id: esc.album_id, decision: "revert" });
                  load(search, view);
                }} className="underline hover:text-text">revert</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Type-check + build + Playwright**

Run: `cd web && npx tsc --noEmit` → no errors.
Run: `npm --prefix web run build` → succeeds (stale-dist trap: the dev server serves `web/dist`).
Run the Playwright review spec → PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/GenreReviewPanel.tsx web/tests/review.spec.ts
git commit -m "feat(web): album-grain escalation review panel (accept/edit/reject + publish)"
```

---

## Task 7: Full-suite gate + manual verification note

**Files:**
- None (verification) + a short note in `docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md`.

- [ ] **Step 1: Python suite**

Run: `python -m pytest -q -m "not slow"`
Expected: all pass; quote real counts. Investigate any failure in `test_web_review_api` / `test_worker_*` / `test_escalation_queue`.

- [ ] **Step 2: Lint + types + build**

Run: `python -m ruff check src/ai_genre_enrichment/escalation_queue.py src/playlist_gui/worker.py src/playlist_web/app.py` → clean.
Run: `cd web && npx tsc --noEmit && npm run build` → clean.

- [ ] **Step 3: Manual verification note**

Append to `docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md` a "Reviewing escalations in the GUI" section: open `python tools/serve_web.py`, the **Genre Review** tab now lists pending album escalations; Accept/Edit/Reject per album; **Publish decided** lands them (backs up metadata.db + `publish()`); the artifact rebuild for *generation* remains a separate step. Note the web-gui restart trap (restart `serve_web.py` after a worker edit; rebuild `web/dist` after a front-end edit).

- [ ] **Step 4: Commit**

```bash
git add docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md
git commit -m "docs(web): GUI escalation review operator note"
```

---

## Notes for the executor

- **Do not run the live panel against the real DBs as part of the plan** — tests use temp DBs + the fake worker. The 368 already-queued escalations are exercised only at manual-verification time.
- **The reader-thread read-only rule is load-bearing** (Task 1/2): the queue/completed reads must go through `list_page` (mode=ro, no DDL). Re-introducing DDL on the read path re-opens the 2026-06-12 timeout bug.
- **Confirm `METADATA_DB_PATH`/`SIDECAR_DB_PATH` constant names** in `worker.py` before adding/duplicating (Task 3). If only `SIDECAR_DB_PATH` exists, derive the metadata path the same way the other handlers reach metadata.db.
- **`proposed_genres` may be empty** for some of the already-queued escalations (SP1 canonicalization). The card already handles this ("(none resolved — use Edit)"). A deeper fix (enqueue raw terms) is out of scope here.
