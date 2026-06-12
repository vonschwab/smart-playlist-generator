# Genre Review Panel — Design

**Status:** Design approved (brainstorm 2026-06-11), pending spec review
**Fills:** the "Genre Review" stub tab in the web GUI's right panel (`web/src/components/AdvancedPanel.tsx:33`, "Genre review lands in a later phase"), carried since the browser-GUI phase-1 design.

## 1. Problem and queue-source decision

The deleted PySide6 desktop GUI had a Genre Review dock (`review_panel.py`) that classified scraped source tags. The web stub was reserved as its replacement, but the backend has moved twice since:

- **Source-tag classification** is now adjudicated automatically by the Claude enrichment backend (`enrich` stage chunks `review_only` tags through `adjudicate_tags` and caches results). The human keystroke-classification flow is largely obsolete.
- **Escalated release checks** (`get_escalated_queue`, the `review-escalated` CLI) are **dead data**: all 1,312 `needs_review` rows in `ai_genre_release_checks` date from the gpt-4o-mini era, carry zero add/prune suggestion rows (all 3,922 adds / 2,560 prunes sit on auto-applied `complete` checks), and only 15 have `review_only_suggestions` content — junk (e.g. tag `'empty'`). The current pipeline never feeds this queue.

What the **current** architecture actually flags for human judgment is the hybrid-evidence fusion output: `fuse_hybrid_evidence` produces per-release `needs_review` and `provisional` term decisions, and `classify_layered_term` marks taxonomy-unknown terms as `review`. `build_layered_release_diagnostics` aggregates these into `review_terms` per release — computed on demand, counted by `materialize_layered_assignments`, **persisted nowhere**.

**Decision:** the panel reviews hybrid review terms. Decisions persist through the existing `ai_genre_user_overrides` mechanism, which the `publish` stage already applies (`overrides_applied`).

### Measured workload (2026-06-11, read-only sample)

- Release universe: ~3,998 distinct release keys in `ai_genre_source_pages` (498 have enriched signatures; 39 have layered assignments — the graph stages just shipped).
- 300-release sample from signature-bearing releases: **93% have review terms, avg 5.0 terms each, 130 ms/release** to compute.
- Full-library scan ≈ 9 minutes. Conclusion: the queue must be **persisted**, not computed per page-load (CLAUDE.md layer-4 #24: pre-compute heavy work).

## 2. Data model

New sidecar table (in `ai_genre_enrichment.db` — `metadata.db` untouched):

```sql
CREATE TABLE IF NOT EXISTS ai_genre_review_queue (
    release_key       TEXT NOT NULL,
    normalized_artist TEXT NOT NULL,
    normalized_album  TEXT NOT NULL,
    term              TEXT NOT NULL,
    confidence        REAL,
    basis             TEXT NOT NULL,      -- hybrid_provisional | hybrid_fusion | unknown_taxonomy
    sources_json      TEXT NOT NULL DEFAULT '[]',
    reason            TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected
    scanned_at        TEXT NOT NULL,
    decided_at        TEXT,
    PRIMARY KEY (release_key, term)
);
```

Decided rows are **kept** so rescans never resurrect a settled question. The `term` column doubles as the hook for a future global "reject everywhere" layer (deferred — see §7).

One-time timestamped backup of the sidecar DB before the migration runs (per artifact-write discipline).

## 3. Scan command (tracked worker job)

`scan_genre_review` in `src/playlist_gui/worker.py`, registered in `TRACKED_COMMAND_HANDLERS`:

1. Enumerate distinct release keys from `ai_genre_source_pages`.
2. Per release: `build_layered_release_diagnostics(store, release_id, taxonomy)` → `review_terms`.
3. Upsert **pending** rows; never touch rows with `status != 'pending'`.
4. Prune pending rows whose term no longer appears for that release (evidence changed).
5. Skip terms already present in the release's user override (either `genres_add` or `genres_remove`) — those are settled.
6. `emit_progress` per release; honor cancellation at release boundaries only (all writes are per-release upserts, so a cancelled scan keeps its partial results). `emit_result` with summary counts (`releases_scanned`, `pending_terms`, `new_terms`, `pruned_terms`) then `emit_done`.

Later (not v1): the analyze `enrich`/assign stage refreshes queue rows in the same pass it already fuses, making pipeline runs keep the queue fresh for free. v1 is button-triggered scan only.

## 4. Read + decision commands (quick, `bridge.command` pattern)

Both follow the existing `edit_genres` shape (tracked but fast; FastAPI wraps with `bridge.command`, `BridgeBusy` → HTTP 409).

- **`get_genre_review_queue`** — returns pending rows grouped by release, ordered by pending-term count desc, plus header counts (`pending_releases`, `pending_terms`). Optional `search` filter (substring on artist/album) and `limit`/`offset` paging.
- **`apply_genre_review_decision`** — payload: `release_key`, `term`, `decision` (`accept` | `reject` | `revert`).
  - **accept** → merge term into `genres_add` of the release's `ai_genre_user_overrides` row (preserving existing entries), call `rebuild_enriched_genres_for_release(release_key)`, set queue row `accepted` + `decided_at`.
  - **reject** → merge into `genres_remove`, rebuild, set `rejected`.
  - **revert** → remove term from both override sets, rebuild, set row back to `pending` (`decided_at` cleared).

No LLM or external API calls anywhere in the panel — pure local SQLite adjudication of already-generated evidence.

## 5. Web API + GUI

**Endpoints** (`src/playlist_web/app.py`):
- `POST /api/review/scan` → `bridge.submit` the tracked scan job; returns `job_id` (progress over the existing WS pipeline; result lands in `JobOut.tool_result`).
- `GET /api/review/queue?search=&limit=&offset=` → wraps `get_genre_review_queue`.
- `POST /api/review/decision` → wraps `apply_genre_review_decision`.

**Component** `web/src/components/GenreReviewPanel.tsx`, replacing the stub branch in `AdvancedPanel.tsx`. Studio Dark, raw Tailwind per existing panels.

- **Header:** pending counts ("N releases · M terms"), **Scan** button (disabled while any job runs; shows progress label from WS events keyed by `job_id`), search box.
- **Release list:** sorted by pending-term count desc; row = artist – album + pending count badge.
- **Release card** (selected release): one row per pending term — term, confidence, basis chip, source list, reason — with Accept / Reject buttons; per-release **Accept all** / **Reject all**; a collapsed "decided" section with Revert.
- **Keyboard:** `A` / `R` decide the focused term and advance; finishing a release auto-advances to the next.
- **Empty state:** "Queue is empty — run a scan" with the Scan button.
- Decisions optimistic-update the list; on API error the row reverts and the error shows inline.

**Known limitation (documented in-UI):** while a long tracked job runs (scan, enrich, analyze), queue reads *and* decision calls return 409 — the bridge is single-flight. The panel disables its own actions and shows scan progress during its scan; a 409 from elsewhere surfaces as "worker busy — try again when the current job finishes".

## 6. Testing

Per the web-gui skill checklist, every layer or it silently no-ops:

- **Unit (storage):** queue upsert semantics — decided rows survive rescan; pruning removes stale pending rows; override-settled terms are skipped. Decision writes — override merge (not replace), signature rebuild called, status transitions including revert.
- **Fake worker:** branches for `scan_genre_review`, `get_genre_review_queue`, `apply_genre_review_decision`.
- **Integration:** `TestClient(create_app(worker_cmd=FAKE))` round-trips for all three endpoints; scan result lands in `tool_result`; 409 when busy.
- **Playwright:** tab renders queue from fake worker; accept flow updates counts.
- **Frontend:** `npx tsc --noEmit`, `npm --prefix web run build` (stale-dist trap), server restart noted for manual verification.

## 7. Deferred (designed-for, not built)

- **Global term decisions** ("reject `'empty'` everywhere"): additive — a global decision table consulted by scan + fusion; the queue's `term` column and per-term grouping already support it.
- **Pipeline-integrated refresh:** wiring the scan upserts into the analyze `enrich`/assign stage.
- **Stale escalated-check cleanup:** the 1,312 dead `needs_review` rows in `ai_genre_release_checks` are untouched by this feature; bulk-completing them is a separate one-off decision.
