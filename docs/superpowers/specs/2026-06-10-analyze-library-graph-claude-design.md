# Analyze Library → graph pipeline + Claude-Code backend + web Tools panel — design

**Date:** 2026-06-10
**Status:** Approved (brainstorm with Dylan)
**Lanes:** SP5 (web GUI tools), SP2/SP3 integration (sources → enrichment → graph), LLM provider switch

## Problem

1. The web GUI cannot trigger Analyze Library or enrichment — the worker has handlers
   (`handle_analyze_library`, `handle_enrich_genres` in `src/playlist_gui/worker.py`) built for the
   removed PySide6 GUI, but `src/playlist_web/app.py` exposes no endpoint and the React app has no UI.
2. `scripts/analyze_library.py` predates the genre enrichment / layered graph program. Its stages
   (`scan → genres → discogs → sonic → genre-sim → artifacts → genre-embedding → verify`) never touch
   the enrichment sidecar: no Last.fm scrape, no hybrid/layered enrichment into
   `data/ai_genre_enrichment.db`, no publish into `release_effective_genres`. "Analyze Library" does
   not actually make the genre graph current.
3. Dylan cancelled OpenAI; all enrichment LLM call sites (`OpenAIEnrichmentClient`) are dead-ended.
   He has Claude Max. Direct Anthropic API billing is not an option (no budget for API credits), so
   LLM calls must route through Claude Code via the **Claude Agent SDK**, which runs against the Max
   subscription.

## Decisions made during brainstorm

- Enrichment stages become **first-class incremental stages** of `analyze_library.py` (run by
  default, skip work already done), not opt-in flags or a separate tool.
- **All** LLM call sites switch to the Claude-Code backend (option B1: drop-in client swap). No
  dual-provider abstraction (YAGNI — the OpenAI key is gone).
- Classification-shaped work (tag classification, adjudication, model prior) is **stashed and run in
  chunked batches** to minimize Max-subscription usage (amortizes Claude Code's per-session system
  prompt overhead; consecutive chunks in one session keep the prefix cached). Note: Anthropic's
  Message Batches API discount is API-billing only — not applicable here.
- **Bandcamp is dropped from scope.** The locator was the only web-search call site and Dylan judged
  it not worth the cost. The `extract-bandcamp` CLI subcommand and `bandcamp_enrichment.py` stay in
  the repo, unwired from the pipeline. The Claude client therefore needs **no tools at all**.
- GUI scope: a Tools panel with **Analyze Library + Enrich** actions (not the full CLI suite).

## Phase 1 — Claude-Code LLM backend

### New module: `src/ai_genre_enrichment/claude_client.py`

`ClaudeCodeEnrichmentClient` with the same public surface as `OpenAIEnrichmentClient`:

- `enrich(payload, prompt, response_format, *, instructions)` → `EnrichmentResult`
- `request_structured(...)` → `EnrichmentResult`
- `call_structured(prompt, response_format, *, instructions)` — the provider-neutral replacement for
  the current `_call_openai` (which `graph_growth.py` and others call directly).
- Dry-run path identical in shape to the OpenAI client's (token estimates, `status="skipped"`).

Backed by the **Claude Agent SDK** (`claude-agent-sdk` PyPI package), which drives the locally
installed, already-authenticated Claude Code CLI. Configuration:

- **No tools enabled** (`allowed_tools=[]`, permission mode that denies everything) — pure
  prompt → JSON text.
- `instructions` becomes the system prompt; `response_format` (the JSON schema) is rendered into the
  prompt as an explicit output contract, since the Agent SDK has no strict structured-output mode.
- Enforcement = the existing per-schema validators (`validate_ai_response`, per-call validators) +
  one retry that feeds the validation error back, preserved from the OpenAI client.
- JSON extraction: parse the result message text; tolerate code fences.

### Batch mode

`classify_batch(items, *, chunk_size=30)`:

- Each item carries a stable ID (release_key or tag ID). One prompt per chunk lists all items;
  the response contract is a JSON array keyed by item ID.
- Each item's result is validated independently. Invalid/missing items are re-queued; after batch
  retries are exhausted, fall back to per-item `call_structured` for just the failures.
- Chunks within one run execute as consecutive turns of a single SDK session (prefix cache stays
  warm). A new run starts a new session.
- Used by: `classify-tags`, AI tag adjudication (`tag_adjudicator.py`), model prior generation
  (`model_prior.py`). The graph-growth placement call (`graph_growth.py`) stays per-candidate
  (it is inherently sequential: same-batch forward references).

### Provider seam

- Factory `create_enrichment_client(...)` in the enrichment package, keyed off `config.yaml`:

  ```yaml
  ai_genre:
    provider: claude_code        # the only live provider; 'openai' remains constructible for tests
    claude_model: haiku          # Agent SDK model selector; Haiku default (lightest on Max windows)
  ```

- All production call sites that construct `OpenAIEnrichmentClient` directly switch to the factory:
  `scripts/ai_genre_enrich.py` (4 sites), `tag_adjudicator.py`, `model_prior` path,
  `graph_growth.py`. `bandcamp_enrichment.py` is out of scope (unwired).
- `OpenAIEnrichmentClient` keeps its `_call_openai` name internally but gains the
  `call_structured` alias so call sites are provider-neutral. The class is not deleted
  (unit tests exercise its dry-run and validation logic).

### Failure discipline (project gotcha: no silent no-ops)

- Claude Code CLI missing or unauthenticated → `RuntimeError` at client construction with a
  remediation message. Never a silent skip.
- Rate-window exhaustion / SDK errors mid-batch → the current chunk fails, the run aborts with a
  clear error; everything already classified is persisted in the sidecar DB, so re-running resumes
  (ledger/skip-existing semantics already exist).
- `pricing.py`: claude_code path reports token usage only; `estimated_cost_usd=None`
  (subscription usage, not billable per-token).

### Dependencies

- Add `claude-agent-sdk` to the `[ai]`/enrichment optional extra in `pyproject.toml`.

## Phase 2 — Analyze pipeline stages

New default stage order:

```
scan → genres → discogs → lastfm → sonic → enrich → publish → genre-sim → artifacts → genre-embedding → verify
```

All new stages follow the existing contract: fingerprint via `compute_stage_fingerprint`, pending
estimate via `estimate_stage_units`, skip-if-fingerprint-unchanged, respect `--stages`, `--force`,
`--limit`, ProgressLogger wiring, and result dicts feeding the run report.

### `lastfm` stage

- In-process wrap of the `extract-lastfm` subcommand logic: fetch Last.fm top tags for releases that
  have no Last.fm source page in the sidecar (skip-existing via `ai_genre_source_pages` /
  attempt ledger).
- Uses `Config.lastfm_api_key` (already exists). Missing key → stage **errors loudly**
  (like the discogs stage's token requirement), it does not silently skip.
- No LLM. Sets `evidence_dirty` when new source pages are added.

### `enrich` stage

- Collect all releases with un-enriched or changed evidence (no `enriched_genre_signatures` row, or
  new source pages since the signature was built).
- Steps, in-process (not the worker's per-release subprocess pattern):
  1. Deterministic tag extraction from stored source pages (`extract-tags` logic) for pending pages.
  2. **Chunked batch classification** of unclassified tags (Phase 1 `classify_batch`).
  3. Hybrid evidence fusion + model prior per pending release (`hybrid-enrich-one --apply
     --with-model-prior` logic as library calls; model prior generation itself batched).
  4. `graph-build-assignments` to refresh layered graph assignments.
- Writes only to `data/ai_genre_enrichment.db` (sidecar). Result dict reports releases enriched,
  tags classified, chunks used, token usage.

### `publish` stage

- `scripts/publish_genres.py` logic in-process: resolve graph-where-present-else-legacy per album
  into `release_effective_genres` in **metadata.db**, then run the `validate_published_genres`
  checks.
- This is the pipeline's sanctioned metadata.db write path (same class as scan/genres/discogs
  writes); publish is idempotent. Sets `genres_dirty` so `genre-sim` and `artifacts` rebuild.

### Out of scope

- The artifact builder keeps `genre_source="legacy"`. Flipping the engine/artifact to consume the
  graph is the SP4 lane (separate spec, already staged).
- Bandcamp stage: dropped (see decisions).

## Phase 3 — Web GUI Tools panel

### API (`src/playlist_web/app.py` + `schemas.py`)

- `POST /api/tools/analyze` — body: `{stages?: string[], force?: bool, dry_run?: bool}`.
  Creates a job in `JobRegistry`, submits worker cmd `analyze_library` through the bridge.
  409 via `BridgeBusy` if anything is running (single-flight worker is acceptable for a local app).
- `POST /api/tools/enrich` — body: `{scope: "artist"|"release"|"all_unenriched", artist?, album?}`.
  Submits worker cmd `enrich_genres` (handler already supports these scopes).
- Job progress/log/result already flow through the registry + WebSocket hub; verify the analyze
  handler's events carry the `job_id` the registry expects (the handler predates the web bridge —
  align event payloads if needed).
- Cancel: existing `POST /api/jobs/{id}/cancel` path (the analyze handler already honors
  cancellation).

### UI (`web/src/`)

- New **Tools** view in the Shell (tab alongside the existing panels), Studio Dark / shadcn:
  - **Analyze Library card** — stage checkboxes (default: all), Force and Dry-run toggles, Run /
    Cancel buttons, live progress bar + current-stage label from WS events, last-run summary
    (per-stage decision/duration/errors from the run report result event).
  - **Enrich card** — artist autocomplete (existing `/api/autocomplete`), optional album field,
    "Enrich all pending" action; live progress + result summary.
- Disable Run buttons while any job is running (mirrors the 409).

## Testing

- **Phase 1:** unit tests with a fake SDK transport (valid JSON, fenced JSON, invalid-then-valid
  retry, rate-limit error, CLI-missing construction error). Batch chunking, per-item validation,
  re-queue, and per-item fallback tested pure. Existing OpenAI-client tests untouched.
- **Phase 2:** extend `test_analyze_orchestration` patterns — stage order, fingerprints, dirty-flag
  propagation (`evidence_dirty` → enrich runs; enrich/publish → `genres_dirty` → genre-sim/artifacts
  rebuild), loud-failure paths (no Last.fm key, no Claude CLI). Enrich stage against a temp sidecar
  DB with canned source pages and a fake client. **No live metadata.db writes in tests** —
  temp DBs only (data-safety rule).
- **Phase 3:** FastAPI endpoint tests using the fake-worker harness (`PG_WEB_WORKER_CMD`);
  Playwright smoke test for the Tools panel (run dry-run analyze, see progress + completion).

## Build order & shippability

Phase 1 → Phase 2 → Phase 3; each phase lands independently (1: enrichment CLI works on Claude;
2: CLI analyze feeds the graph; 3: GUI exposes it). Full-library sweeps will hit Max rate windows —
abort-and-resume is the designed behavior, not a failure.
