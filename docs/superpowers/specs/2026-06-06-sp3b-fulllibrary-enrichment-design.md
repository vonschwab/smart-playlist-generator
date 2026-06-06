# SP3b — Tiered Full-Library Enrichment + Re-Publish — Design

**Date:** 2026-06-06
**Status:** Approved (design); implementation plan pending
**Program:** Genre enrichment. Depends on **SP3a** (grown graph) and **SP2**
(collected tags). Feeds **SP4** (artifact build).

## Context

After SP2 collects tags library-wide and SP3a grows the taxonomy to cover the
real vocabulary, every album's collected tags can be turned into a graph
signature. SP3b runs that enrichment over the **whole library**, then re-publishes
so `metadata.db.release_effective_genres` flips from `legacy` to `graph`
library-wide.

### Decisions locked during brainstorming

- **Tiered enrichment.** Deterministic graph fusion for every album first (free,
  fast, reproducible); reserve an AI set-level reasoning pass for the
  weak/sparse/ambiguous minority where it changes the answer.
- **Re-publish via SP1.** Reuse the SP1 `publish_genres` pipeline to materialize
  `release_effective_genres`; no new publish logic. The live `metadata.db` write
  is the same gated, backed-up, second-confirmation step as SP1's live run.

## Grounding (current code)

- **Deterministic fusion already exists.** `graph-build-assignments`
  (`scripts/ai_genre_enrich.py::cmd_graph_build_assignments`) iterates discovered
  releases, runs `_fuse_hybrid_for_release` (→ `collect_hybrid_evidence` +
  `fuse_hybrid_evidence`) and `materialize_layered_assignments` into the sidecar
  authority tables (`genre_graph_release_genre_assignments` /
  `…_facet_assignments`). It is idempotent per release (re-running overwrites that
  release's rows). This *is* the deterministic tier — SP3b orchestrates it at
  full-library scale.
- **Lane routing already exists.** `route_release(payload, web_mode)`
  (`src/ai_genre_enrichment/routing.py`) returns a lane:
  - `SKIP_WELL_TAGGED` — 3–8 specific genres, no descriptors, no conflict.
  - `NO_WEB_ADJUDICATION` — local metadata sufficient.
  - `AUTHORITATIVE_SOURCE_ENRICHMENT` — no genres / generic-only / descriptor-only
    / material source conflict / thin identity → **the "weak" set for the AI tier.**
- **AI enrichment pass exists.** The `run` / `run-one` web-enrichment path (with
  the `ai_genre_release_checks` cache + `_should_skip_cached`) adds
  authoritative-source evidence for weak releases; that evidence then flows back
  through the same fusion → materialize path.
- **Publish exists (SP1).** `scripts/publish_genres.py` /
  `src/genre/genre_publish.py` read the sidecar authority and write
  `release_effective_genres` (graph-where-present else legacy). Already live-run
  once on 2026-06-06.

## Components & data flow

```
grown graph (SP3a) + collected tags (SP2)
  → [1 deterministic fusion]  every album → materialize assignments (sidecar)
  → [2 weak detection + AI tier]  weak albums → AI evidence → re-fuse → re-materialize
  → [3 re-publish (SP1)]  sidecar authority → metadata.db.release_effective_genres
  ⇒ whole library on graph genres  (input to SP4)
```

### 1. Full-library deterministic pass — `graph-enrich-library` (tier 1)
Run fusion + materialize over **all** discovered releases against the grown
graph (essentially `graph-build-assignments` with no release filter), wrapped for
full-library scale like the SP2 collection passes: `[i/N]` progress, per-release
try/except (one bad release never kills the run), UTF-8 output, final summary.
Idempotent, so resumable by re-running (each release's assignments are
overwritten). Records per-release outcome (assignment count, whether escalated).

### 2. Weak detection + AI tier — same command, `--ai-weak` (tier 2)
A release is escalated to the AI tier when **either**:
- `route_release` returns `AUTHORITATIVE_SOURCE_ENRICHMENT` (pre-fusion signal), **or**
- the deterministic fusion result is **below an accepted-genre floor**
  (post-fusion signal: fewer than `min_accepted_leaves` accepted leaf genres, or
  only family/generic assignments). This catches albums that *looked* fine but
  fused weakly.

For escalated releases, run the existing AI web-enrichment pass (cached via
`ai_genre_release_checks`, so reruns skip completed AI work — resumable, no
re-spend), then re-fuse + re-materialize. The AI tier is **opt-in via `--ai-weak`
with a `--max-ai N` cap** so the spend is bounded and observable; tier 1 alone is
always safe to run first and inspect the escalation count before paying.

### 3. Re-publish — SP1 `publish_genres` (gated live write)
After assignments are materialized, run `publish_genres` to rebuild
`release_effective_genres`. Validate against a **copy** first
(`metadata.db.worktest`) exactly as SP1 did; the live write requires a fresh
timestamped backup and explicit second confirmation **from the user at that
moment**. Not part of automated execution.

## Tiering criteria (concrete, all tunable)

| Signal | Tier |
|--------|------|
| `route_release` = `SKIP_WELL_TAGGED` or `NO_WEB_ADJUDICATION`, fusion ≥ floor | deterministic only |
| `route_release` = `AUTHORITATIVE_SOURCE_ENRICHMENT` | AI tier (if `--ai-weak`) |
| fusion accepted leaves < `min_accepted_leaves` (default 2) | AI tier (if `--ai-weak`) |

## Scale, cost, resumability

- **Tier 1** is local fusion: free, fast, fully reproducible; re-runnable.
- **Tier 2** is AI web search per *weak* album only, capped by `--max-ai`, cached
  by `ai_genre_release_checks` (resumable, no double-spend). Inspect the
  escalation count from a tier-1 dry run before committing spend.
- **Re-publish** is seconds on this dataset.

## Safety & reversibility

- Tiers 1–2 write only the **sidecar** (rebuildable scratch) — safe to iterate.
- The single `metadata.db` write is the SP1 publish: copy-validate → backup →
  second confirmation → live. Honors the project rule on `metadata.db`.
- Materialize is idempotent; AI work is cached; both passes are resumable.

## Testing

- **Deterministic golden:** a few known albums fuse to the expected graph
  signature against a fixture grown-graph (ATCQ →
  `east_coast_hip_hop`/`jazz_rap`/`boom_bap`; Jobim →
  `bossa_nova`/`latin_jazz`/`mpb`).
- **Tiering:** well-tagged album → deterministic only; no-genre / generic-only /
  conflicting / thin album → escalated; album with fusion below the leaf floor →
  escalated even if `route_release` said skip.
- **`--max-ai` cap** halts escalation after N albums; remaining weak albums stay
  on their deterministic result.
- **Full-library orchestration:** progress/summary counts; one failing release
  doesn't abort the run; re-run is idempotent (same materialized rows).
- **Re-publish (against a copy):** after enrichment, `release_effective_genres`
  shows the formerly-legacy albums now `graph`; SP1's validation script passes.

## Out of scope (SP3b)

- Growing the taxonomy (SP3a — must run first).
- Artifact build / playlist engine consuming the graph (SP4) — SP3b ends at a
  re-published `release_effective_genres`.
- The genre-editing GUI (SP5).
