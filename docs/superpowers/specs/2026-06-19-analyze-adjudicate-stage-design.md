# Analyze pipeline — album adjudicator stage (SP1)

**Date:** 2026-06-19
**Branch:** `worktree-phase1-album-adjudicator`
**Status:** design approved, pending plan
**Predecessors:** Phases 0–4 (gold corpus → `album-adjudicator-v1` contract → scorer/eval gate →
resumable bulk runner → Phase 4 publish to authority). The album adjudicator currently runs as a
research/shadow process (`scripts/research/run_adjudicator_bulk.py` → `data/adjudication_pass1.db`
→ `apply_adjudication.py` / `review_escalated.py` → `publish()`). Roadmap:
`docs/GENRE_RESOURCES_AND_CLAUDE_ADJUDICATION_ROADMAP_2026-06-15.md`.

## Goal

Make the album-grain Sonnet adjudicator a first-class stage of the Analyze Library pipeline
(`scripts/analyze_library.py`), so that running `analyze` produces tightened, adjudicated genre
identities natively — incrementally, resumably, and with a conservative human-in-the-loop hold on
escalations — instead of via standalone research scripts plus a manual apply/publish.

## Scope decomposition

This work splits into two independently shippable sub-projects with a clean interface (the
escalation queue):

- **SP1 (this spec): the `adjudicate` + `apply` pipeline stages + escalation queue (backend).**
  Produces tightened genres end-to-end through the analyze pipeline; escalations are held and
  written to a durable queue, reviewable via the adapted `review_escalated.py` CLI as the interim
  surface.
- **SP2 (separate spec, later): web Genre Review GUI album-grain escalation panel (frontend).**
  Surfaces and clears the queue SP1 produces. Out of scope here.

SP1 → SP2 sequencing: SP1 is the load-bearing core and locks the queue contract; SP2 builds on it.

## Decisions locked (during brainstorming)

1. **Adjudicator role:** *replace* the tag-grain `enrich` stage. The album adjudicator becomes the
   genre-production stage; `enrich` is retired from the default stage order but kept runnable
   on demand (`--stages enrich`) as a legacy fallback. Not deleted.
2. **Escalation policy:** *hold + queue*. Non-escalated albums materialize and publish normally;
   escalated albums are withheld from publish (prior authority preserved) and added to a review
   queue cleared later.
3. **Review surface:** the durable queue is consumed by the web Genre Review GUI (SP2). For SP1, the
   adapted `review_escalated.py` CLI is the interim review surface.
4. **Model/pass structure:** a *single Sonnet pass per album* in the pipeline. The two-tier
   Haiku-standard + Sonnet-thorough path was a cost optimization for the 3,428-album backfill; at
   incremental analyze volume the cost argument vanishes, so the pipeline favors quality + simplicity.
   The two-tier path stays available via the standalone backfill script.

## Architecture — stage order

Current genre path:

```
… mert → enrich → publish → genre-sim → artifacts …
```

New genre path:

```
… mert → adjudicate → apply → publish → genre-sim → artifacts …
```

- `adjudicate` — rate-limited, resumable Sonnet LLM work → raw responses to a checkpoint.
- `apply` — pure deterministic logic → checkpoint → materialize non-escalated to the sidecar,
  enqueue escalated.
- `publish` — unchanged; the hold is enforced upstream by `apply` not materializing escalated albums.
- `enrich` — removed from `STAGE_ORDER_DEFAULT`, still dispatchable via `--stages enrich`.

## `adjudicate` stage (LLM → checkpoint)

- **Targeting (incremental):** the candidate set is **all albums in `metadata.db`** (the
  bulk runner's `library` source), filtered by `AdjudicationStore.is_done(album_id, prompt_version,
  input_hash)`. A new album, or one whose evidence changed (different `input_hash`), is adjudicated;
  everything already complete is skipped. So a normal analyze run after a small scan adjudicates only
  the handful of new/changed albums, while a first run over a fresh library adjudicates everything.
- **Evidence:** built per album (artist/album, track titles, source tags by origin, file tags,
  current observed-leaf) via the promoted `build_evidence`.
- **Call:** one Sonnet structured call per album against the `album-adjudicator-v1` contract, with
  the file-tag floor enforced (drops force-escalate).
- **Checkpoint location:** the `adjudications` table lives **in the sidecar**
  (`data/ai_genre_enrichment.db`), so the whole analyze genre state is in one DB and `apply`/`publish`
  read it without a cross-DB attach. A one-time importer brings the in-flight backfill
  (`data/adjudication_pass1.db`) into the sidecar so the pipeline inherits that work rather than
  re-calling the LLM.
- **Resumability:** on the rate wall the stage returns a `paused` result (the pattern `stage_enrich`
  already uses), preserving the checkpoint; re-running resumes. Per-album commit = zero loss.
- **Fingerprint:** count + hash of albums needing adjudication, so `compute_stage_fingerprint`
  no-ops the stage when nothing is new.

## `apply` stage (checkpoint → sidecar + queue)

- Reads each album's **best result** (thorough over standard when both exist — preserving the
  two-tier backfill data already gathered).
- **Non-escalated** → `materialize_adjudication` (classify terms → observed_leaf + inferred
  parent/family) → `genre_graph_release_genre_assignments`.
- **Escalated** → write to the escalation queue (held; not materialized, so prior authority stays).
- **Idempotent, cheap to re-run:** after a taxonomy-growth pass, re-running `apply` re-materializes
  from the same checkpoint and picks up new canonical mappings — no LLM calls. Its fingerprint
  includes the taxonomy version, so a taxonomy change auto-retriggers `apply` (and only `apply`) on
  the next analyze.

Key property: the expensive LLM work happens once per album, ever; the deterministic apply re-runs
freely as the taxonomy grows.

## Escalation queue — the SP1 ↔ SP2 contract

A single table in the sidecar plus a small store class, designed so the GUI renders a review card
with no recomputation.

**Table `adjudication_escalations` (sidecar):**

| field | purpose |
|---|---|
| `album_id` (PK) | the album |
| `release_key`, `artist`, `album` | display + publish mapping |
| `prior_observed_leaf` (JSON) | current authority (left side of the diff) |
| `proposed_genres` (JSON) | adjudicator's set with confidences (right side) |
| `escalate_reason` (text) | why it escalated |
| `dropped_file_tags` (JSON) | file-tag-floor casualties — the thing most needing review |
| `prompt_version`, `model`, `input_hash` | provenance + change detection |
| `status` | `pending` / `accepted` / `edited` / `rejected` |
| `decision_genres` (JSON) | the corrected set when edited |
| `created_at`, `decided_at` | lifecycle timestamps |

**Lifecycle:**
- `apply` **upserts** a `pending` row when an album escalates. If a row is already *decided* and the
  new proposal is unchanged (same `input_hash`), it is left alone; if the proposal changed
  (re-adjudicated), it re-opens to `pending` so a stale decision never silently sticks.
- The review surface reads `pending` rows and renders prior-vs-proposed + reason + dropped tags
  directly from the row.
- On **accept / edit** → materialize the chosen set (same materializer) → status `accepted`/`edited`,
  `decided_at` stamped. On **reject** → status `rejected`, nothing materialized (prior authority
  preserved).
- The next `publish` picks up the newly materialized assignments.

**Contract surface — `EscalationQueue` store class** (both the CLI and SP2's API call this; the
decision logic is never reimplemented in a UI layer):
- `list_pending()` → rows with everything needed to render.
- `get(album_id)`.
- `record_decision(album_id, decision, genres=None)` → persists status **and** performs the
  materialize for accept/edit.

## Publish interaction

`publish` is unchanged. Because `apply` never writes assignments for escalated albums, they retain
whatever they had:
- A **re-adjudicated** album that now escalates keeps its prior graph assignments until decided.
- A **brand-new** album that escalates has no graph row and falls back to its legacy raw tags.

No active exclusion logic; a regression test asserts an escalated album is never cleared or
overwritten.

## Code organization (promote research → `src`)

The shadow-flow logic currently in `scripts/research/` becomes importable modules; the stages in
`analyze_library.py` stay thin wrappers (targeting + delegating + returning the stage-result dict),
matching how `stage_enrich` already delegates.

| Module | Origin | Role |
|---|---|---|
| `src/ai_genre_enrichment/album_evidence.py` | promote `build_evidence` | per-album evidence |
| `src/ai_genre_enrichment/adjudication_runner.py` | promote bulk-runner core | incremental session loop, checkpoint targeting, pause/resume |
| `src/ai_genre_enrichment/adjudication_store.py` | exists | checkpoint; repoint to sidecar |
| `src/ai_genre_enrichment/adjudication_materializer.py` | exists | reuse as-is |
| `src/ai_genre_enrichment/escalation_queue.py` | new | queue store + `record_decision` |
| `scripts/analyze_library.py` | edit | thin `stage_adjudicate` / `stage_apply`, stage-order + fingerprint + unit-estimate entries |
| `scripts/research/review_escalated.py` | adapt | point at the sidecar `EscalationQueue` (interim review surface) |
| one-time importer | new small script | `adjudication_pass1.db` → sidecar `adjudications` |

## Error handling

Following CLAUDE.md's "a configured knob that can't act is a startup error, not a silent no-op":
- **Transient LLM / rate wall** → `adjudicate` returns `paused`, checkpoint preserved, re-run resumes.
- **Unknown terms** → skipped, never invented; recorded so they feed the taxonomy-growth loop.
- **File-tag-floor drops** → force-escalate → queue; never auto-published.
- **Compound facet strings** (e.g. `"grief-stricken, meditative, confessional"`) → split on commas,
  route atomic terms to the facet table, skip the un-resolvable gracefully. Genres are never affected
  (facets are a separate, secondary lane).
- **Misconfig** (missing model, unreadable sidecar) → raise loudly, never skip silently.

## Testing strategy

Unit tests use temp DBs and an injected fake adjudicator client (the `enrich` tests already inject
`enrich_client`, so the pattern exists):
- `adjudicate`: incremental skip via `is_done`; pause/resume on transient failure.
- `apply`: materializes non-escalated; enqueues escalated *without* materializing; idempotent re-run
  after a taxonomy-version bump picks up new mappings.
- `EscalationQueue`: `list_pending` / `get` / `record_decision` (accept/edit/reject) roundtrip;
  re-open-on-changed-proposal; the materialize side-effect on accept/edit.
- `publish` regression: an escalated album is never cleared.
- importer: `adjudication_pass1.db` → sidecar rows.
- **Integration:** extend `tests/unit/test_analyze_graph_stages.py` (which already covers
  lastfm/enrich/publish end-to-end on a tiny `metadata.db`) with `adjudicate → apply` using the
  injected client.
- No live `metadata.db` writes in tests — temp DBs only.

## Out of scope (SP2 and later)

- The web Genre Review GUI album-grain escalation panel, its API endpoint, and bridge plumbing —
  all build on the `EscalationQueue` contract above.
- Flipping any remaining live config from the legacy enrichment path to the adjudicator beyond the
  stage-order change (the default-order swap is in scope; deeper config cleanup is not).

## Open items

- Exact `input_hash` change-detection for the escalation re-open rule reuses the adjudicator's
  existing `stable_input_hash`; confirmed at plan time.
- The one-time backfill importer is a convenience to inherit in-flight Pass-2 work; if the backfill
  is still running when SP1 lands, run the importer after the backfill completes.
