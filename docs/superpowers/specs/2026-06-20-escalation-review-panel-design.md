# Escalation Review Panel (SP2) — Design

**Date:** 2026-06-20
**Branch:** `worktree-phase1-album-adjudicator`
**Status:** design approved, pending spec review
**Predecessors:** SP1 (`docs/superpowers/specs/2026-06-19-analyze-adjudicate-stage-design.md`) shipped the album
adjudicator as analyze stages plus the `EscalationQueue` (`adjudication_escalations` table) that this panel
reads. Supersedes the tag-grain Genre Review panel
(`docs/superpowers/specs/2026-06-11-genre-review-panel-design.md`), whose `ai_genre_review_queue` /
hybrid-fusion source is the now-broken previous enrichment strategy.

## Goal

Repurpose the existing web Genre Review panel — currently a **tag-grain** UI over `ai_genre_review_queue`
(accept/reject individual terms within a release) — into an **album-grain** review over the
`adjudication_escalations` queue: per album, show prior-vs-proposed genres + escalate reason + dropped file
tags, and let the user **accept / edit / reject**, then **publish** the decided albums into the authority from
within the panel. This is the human-review surface the SP1 hold-and-queue policy was built for; ~368
escalations are already queued.

## Decisions locked (during brainstorming)

1. **Decision → live = stage + "Publish decided" button.** Each accept/edit materializes to the sidecar
   instantly; a `Publish decided (N)` button runs `publish()` to land all decided albums into the authority
   (`release_effective_genres`) at once, refreshing the GUI genre cards. Keeps `publish()` the single
   authority writer (no per-album authority write), batches the heavy write.
2. **Approach = reuse the shell, replace the internals, repoint the data layer.** Keep the panel's tab slot,
   Pending/Completed view toggle, search, header counts, "saved ✓" flash + session tally, WS/busy handling,
   optimistic-update pattern, and the bridge→worker→FastAPI→`api.ts` plumbing. Replace the term-grain
   component internals + worker handlers + data source. Not a new parallel panel; not an in-place tweak of the
   two-level term logic.

## Architecture / data flow

```
EscalationQueue (sidecar adjudication_escalations)
  └─ worker handlers (worker.py)
       get_escalation_queue / _completed  (UNTRACKED, read-only reader thread)
       apply_escalation_decision          (UNTRACKED, quick write — record_decision)
       publish_decided                    (TRACKED job — backup metadata.db + publish())
  └─ FastAPI (app.py)  /api/review/{queue,completed,decision,publish}
  └─ api.ts  →  GenreReviewPanel.tsx  (album-grain)
```

`metadata.db` is written only by `publish()`, reached via the tracked `publish_decided` job with an automatic
timestamped backup — no new authority-writer path.

## Backend

### `EscalationQueue.list_page` (read-only) — the key refinement

The existing untracked read handler (`handle_get_genre_review_queue`) runs on the **reader thread and must be
strictly read-only** — its docstring cites the *2026-06-12 review-queue timeout incident*: "must never write
(no initialize/DDL) and never wait behind a scan's write lock." `EscalationQueue.__init__` currently runs
`CREATE TABLE IF NOT EXISTS` (DDL), so it cannot be constructed on the reader thread.

Add to `src/ai_genre_enrichment/escalation_queue.py`:

```python
def list_page(db_path, *, status, search=None, limit=50, offset=0, readonly=True) -> dict
```

- Opens the sidecar **read-only** (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`), no DDL.
- `status='pending'` for the queue view; `status='decided'` returns `accepted|edited|rejected`.
- Optional `search` (substring on artist/album), `limit`/`offset` paging.
- Returns `{ escalations: [...], pending_albums, decided_albums }`, each escalation:
  `{ album_id, artist, album, prior_observed_leaf: [...], proposed_genres: [{term, confidence}],
  escalate_reason, dropped_file_tags: [...], status }`.
- A module-level function (not a method) so the read path never touches the DDL-running constructor.

### `EscalationQueue.revert` (new)

`revert(album_id, *, sidecar_store)` → set the row back to `status='pending'`, clear `decision_genres`/
`decided_at`, and **un-materialize**: `sidecar_store.replace_layered_assignments_for_release(release_id, …,
genre_assignments=[], facet_assignments=[])` so the album is truly held again (its prior authority returns on
the next publish).

### Worker handlers (`worker.py`)

- **`handle_get_escalation_queue`** (UNTRACKED): calls `EscalationQueue.list_page(SIDECAR_DB_PATH,
  status='pending', …)`; emits `{type:result, result_type:'escalation_queue', request_id, job_id:None, **page}`
  then `{type:done, …, job_id:None}` — the explicit `job_id=None` is mandatory (avoids corrupting a running
  tracked job, per the existing handler's docstring).
- **`handle_get_escalation_completed`** (UNTRACKED): same with `status='decided'`,
  `result_type:'escalation_completed'`.
- **`handle_apply_escalation_decision`** (UNTRACKED): payload `album_id`, `decision`
  (`accept|edit|reject|revert`), optional `genres`. accept/edit/reject → `EscalationQueue.record_decision`
  (SP1, materializes via the adjudication materializer); revert → `EscalationQueue.revert`. Emits the standard
  result/done with `job_id=None`.
- **`handle_publish_decided`** (TRACKED): mirrors `handle_scan_genre_review` — back up `metadata.db`
  (`metadata.db.bak.<ts>`), run `publish(metadata_db, sidecar_db)`, `emit_progress` around the phases,
  `emit_result` with `PublishStats`, `emit_done`. Registered in `TRACKED_COMMAND_HANDLERS`.

Register the three untracked handlers in `UNTRACKED_COMMAND_HANDLERS`, `publish_decided` in
`TRACKED_COMMAND_HANDLERS`. Remove the `scan_genre_review` registration **only if** nothing else needs it;
otherwise leave it dead (it is harmless and out of scope).

### API endpoints (`app.py`) — reuse the route paths

- `GET /api/review/queue?search=&limit=&offset=` → `cmd: get_escalation_queue` (untracked). (Returns `status='pending'`.)
- `GET /api/review/completed?search=&limit=&offset=` → `cmd: get_escalation_completed` (untracked).
- `POST /api/review/decision` → `cmd: apply_escalation_decision` (untracked). New body model
  `EscalationDecisionRequest(album_id: str, decision: str, genres: list[str] | None = None)` replacing the
  term-grain `ReviewDecisionRequest`; 422 on missing `album_id`/`decision`.
- `POST /api/review/publish` → `registry.create` + `bridge.submit({cmd: publish_decided, job_id})`; 409 on
  `BridgeBusy`; returns `{job_id}` (progress + `PublishStats` land in `JobOut.tool_result` over WS).
- **Remove** `POST /api/review/scan` and its route.

## UI (`web/src/components/GenreReviewPanel.tsx`)

One-level album list + a single album card (replaces `TermRow` and the nested per-term list):

- **Album list:** each item = one escalation (`artist – album`), with a `⚠` marker when `dropped_file_tags`
  is non-empty. Queue order (`created_at`).
- **Album card (selected):** prior `prior_observed_leaf` as muted chips; proposed `proposed_genres` as accent
  chips with confidence; `escalate_reason` text; `dropped_file_tags` in danger color ("would drop your file
  tag: …"). Actions **Accept (A) / Edit (E) / Reject (R)**.
- **Edit:** reveals a text input pre-filled with the proposed genres comma-separated; submit sends
  `decision:'edit', genres:[…]` (mirrors the CLI `edit a,b,c`).
- **Keyboard:** `A`/`R` decide and advance to the next pending album; `E` opens the edit input.
- **Header:** `N pending · M decided`, session tally + "saved ✓" flash, and **`Publish decided (K)`** where
  `K = decided_albums` (the accepted/edited count from the queue header) → POSTs `/api/review/publish`, shows
  job progress over WS, refreshes on `done` (cards live). The button runs the full idempotent `publish()`
  (lands all materialized assignments, not only this session's) — K is the decided-count hint, not a precise
  unpublished delta. Disabled while any job runs; hidden when `K == 0`.
- **Pending** view = `status='pending'`; **Completed** view = `accepted|edited|rejected`, each with **Revert**
  (re-open to pending + un-materialize).
- Optimistic updates: a decision moves the album pending→decided locally; on API error the row rolls back and
  the error shows inline.
- Empty state: "No escalations pending — all reviewed."

New `api.ts` methods: `reviewDecision` body changes to `{album_id, decision, genres?}`; add `reviewPublish()`.
`web/src/lib/types.ts` gains `EscalationOut` / `EscalationQueueResponse`. The Pending/Completed plumbing and WS
job handling already exist and are reused.

## Error handling

- **Busy/409:** while the publish job (or any tracked job) runs, decision buttons disable and show "worker
  busy — try again when the current job finishes"; the publish job surfaces progress. (Single-flight bridge.)
- **Reader-thread read-only:** the queue/completed reads never write (the `list_page` read-only path); this is
  the load-bearing fix against re-triggering the 2026-06-12 timeout incident.
- **Edit input:** unknown genres are skipped by the materializer (never invented); v1 materializes what
  resolves, no separate validation UI.
- **Publish failure:** the tracked job emits the error; the panel shows it and leaves decisions staged
  (re-runnable).

## Testing

Per the web-gui skill — every layer or it silently no-ops:

- **Unit (store):** `EscalationQueue.list_page` — read-only (opens `mode=ro`, no DDL), status filter
  (`pending` vs `decided`), search, paging, response shape; `revert` — re-opens to pending and clears the
  materialized rows. (`record_decision` is already covered by SP1.)
- **Fake worker** (`tests/fixtures/fake_worker.py`): branches for `get_escalation_queue`,
  `get_escalation_completed`, `apply_escalation_decision`, `publish_decided`.
- **Integration** (`TestClient(create_app(worker_cmd=FAKE))`): round-trips for `/api/review/queue`,
  `/completed`, `/decision`; `/publish` returns a `job_id` and the `PublishStats` land in `tool_result`; 409
  when busy.
- **Playwright** (`web/tests/review.spec.ts`): tab renders the queue from the fake worker; an accept updates
  the counts; the edit flow submits a genre list; the publish button triggers the job.
- **Frontend gates:** `npx tsc --noEmit`, `npm --prefix web run build` (stale-dist trap), and a noted manual
  `serve_web.py` restart for live verification.

## Out of scope

- The artifact rebuild (generation) — `Publish decided` updates the authority/cards only; rebuilding
  `data_matrices_step1.npz` so generation sees the new genres stays a separate, heavier operator step.
- Bulk operations beyond per-album accept/edit/reject (no "accept all" across albums — the file-tag-floor
  cases specifically warrant per-album eyes).
- Taxonomy-aware autocomplete in the edit input (the materializer's skip-unknown behavior is the v1 guard).
- Removing the dead tag-grain `review_queue.py` / `ai_genre_review_queue` machinery (a separate cleanup).

## Open items

- Whether to delete vs leave-dead the `scan_genre_review` worker handler + `review_queue.py` — decided at
  plan time; default is leave-dead (out of scope), remove only if trivially safe.
- Confirm `proposed_genres` richness for escalated albums (SP1 noted canonicalization can empty it for fully
  non-canonical proposals) — if the card shows an empty proposed set, fall back to displaying the raw response
  genres; finalized at plan time.
