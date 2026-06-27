# Handoff — Taxonomy Term Adjudication panel (GUI)

**Date:** 2026-06-26
**Status:** Design approved by Dylan. Not yet implemented. This doc is the implementation handoff for the GUI session.
**Author session:** brainstormed + scoped the feature, mapped all integration points (file:line refs below are from that mapping — re-verify before editing, the repo moves).

---

## 0. Read these skills FIRST (they encode hard-won discipline this feature touches)

Invoke them via the Skill tool before writing code — each prevents a class of bug this feature can hit:

- **`web-gui`** — stale `web/dist`, worker-restart, end-to-end-wiring, silently-dropped-result traps. Mandatory for any `web/src` + `playlist_web` + worker change.
- **`taxonomy-growth`** — the `GrowthProposal` → `validate_proposal` → `append_approved_to_taxonomy` API, the **same-batch forward-reference rule**, reject-reason enum, the validator's unsupported kinds (`reject`, `microgenre`), and the known-flaky enrichment-test deselects. The Apply step lives or dies on this.
- **`genre-data-authority`** — which table/store is authoritative for genres (so you don't wire the panel to a stale layer).
- **`playlist-testing`** — only if you touch generation; not central here.

---

## 1. What this feature is

A new **"Taxonomy"** tab in the right-sidebar `AdvancedPanel` (sibling to diagnostics / blacklist / review). It surfaces genre **terms** that show up in the library but aren't in the taxonomy graph, lets the user summon Claude for a verdict (**add** / **alias** / **reject**) with proposed graph relationships, lets the user ratify/edit that verdict, and — in a deliberate **second step** — writes the accepted decisions into `data/layered_genre_taxonomy.yaml`.

It is the **album-review → publish** shape you already have (`GenreReviewPanel`), applied one layer up: at the **vocabulary** level instead of the **album→genre assignment** level. Reuse that panel as the structural template.

**Critical distinction — do NOT conflate with the existing Genre Review panel.** The existing `GenreReviewPanel` (`/api/review/*`, `adjudication_escalations` table) adjudicates *which genres an album gets*. This new panel adjudicates *what genres exist in the taxonomy vocabulary*. Different concern, different queue, different store, different write target (`layered_genre_taxonomy.yaml`, not `metadata.db`). It is a **sibling**, built by mirroring the same patterns.

---

## 2. Locked design decisions (Dylan approved these — do not relitigate)

| # | Decision | Choice |
|---|----------|--------|
| 1 | **Queue source** | **Merged + deduped**: union of adjudicator `taxonomy_gaps` (from `album_adjudicator.canonicalize_proposed`) **and** `graph_growth.gather_growth_candidates()`. Collapse spacing variants (`collapse_variants`). Annotate each term with **affected album/track count** (triage by reach), co-occurring tags, and example releases. |
| 2 | **Write model** | **Two-phase.** Accept records the decision instantly to a staging table (cheap, revertable, survives restart). A separate **"Apply N decisions"** button validates the whole batch, backs up + writes the YAML, reloads the graph. Mirrors `record_decision` → `publish`. |
| 3 | **Activation** | **Graph-only this feature.** Apply writes the taxonomy and reloads; existing albums pick up the change on the next normal adjudicate/publish cycle. Panel must surface "**M albums will re-classify on next publish**" so it's not silently inert — but re-tagging is explicitly **out of scope** here. |
| 4 | **Minor calls** | "Ask Claude" is **per-term on-demand** (not auto-run on the whole queue); optional opt-in "Ask Claude on all un-triaged" with shown cost is a nice-to-have. Panel lives as a **new tab in `AdvancedPanel`**, not in the Tools panel. |

---

## 3. Components

### New modules (backend)
| Module | Purpose |
|---|---|
| `src/ai_genre_enrichment/taxonomy_review_queue.py` | Build the merged/deduped candidate queue with impact annotations. `list_page(status, search, limit, offset)` — mirror `escalation_queue.list_page`. |
| `src/ai_genre_enrichment/taxonomy_term_adjudicator.py` | Claude contract: payload builder + `ADJUDICATOR_INSTRUCTIONS` + response schema + `validate_response()` → returns a `GrowthProposal`. Model on `album_adjudicator.py`. |
| Staging store | New table `taxonomy_term_decisions` in `data/ai_genre_enrichment.db`. Can live in `taxonomy_review_queue.py` or its own `taxonomy_decision_store.py`. Mirror `escalation_queue.py` / `adjudication_store.py`. |

### Existing code to REUSE (do not reinvent)
| Need | Reuse | Ref |
|---|---|---|
| Unknown-term discovery | `graph_growth.gather_growth_candidates()` | `src/ai_genre_enrichment/graph_growth.py:30-82` |
| Spacing-variant dedup | `graph_growth.collapse_variants()` | `graph_growth.py:90-124` |
| Adjudicator gaps | `album_adjudicator.canonicalize_proposed()` (splits proposed → canonical vs gaps) | `src/ai_genre_enrichment/album_adjudicator.py:245-273` |
| Name normalization | `normalize_taxonomy_name` | `src/ai_genre_enrichment/layered_taxonomy.py` |
| The record schema | `graph_growth.GrowthProposal` (+ `validate_proposal`, `append_approved_to_taxonomy`, `write_proposals`) | `graph_growth.py` |
| Claude calls | `provider.create_enrichment_client()` → `claude_client.request_structured()` / `call_structured()` | `provider.py`; `claude_client.py:388-404, 641-668` |
| Adjudicator contract template | `album_adjudicator.ADJUDICATOR_INSTRUCTIONS`, `ADJUDICATOR_RESPONSE_SCHEMA`, `build_adjudicator_payload`, `validate_adjudicator_response` | `album_adjudicator.py:19-103, 130-212` |
| Taxonomy load | `load_default_layered_taxonomy()` (reads fresh, no lru_cache at def — see §6 cache note) | `layered_taxonomy.py:243-247` |

---

## 4. Data model

### Staging table `taxonomy_term_decisions` (in `data/ai_genre_enrichment.db`)
```
term            TEXT PRIMARY KEY   -- normalize_taxonomy_name(raw_term)
raw_term        TEXT
verdict         TEXT               -- 'add' | 'alias' | 'reject'
proposal_json   TEXT               -- the ratified GrowthProposal payload (what gets written)
claude_json     TEXT               -- Claude's original verdict, for audit
human_edited    INTEGER            -- 0/1: did the human change Claude's proposal
status          TEXT               -- 'pending' | 'applied' | 'reverted'
created_at      TEXT
applied_at      TEXT
batch_version   TEXT               -- the taxonomy_version stamped when applied
```
Methods (mirror `EscalationQueue`): `record_decision`, `revert`, `list_pending`, `list_applied`, `mark_applied(batch_version)`.

### Claude verdict → `GrowthProposal` mapping (the contract)
Claude returns a structured verdict; `validate_response()` maps it onto the existing proposal helpers (see `taxonomy-growth` skill for the exact helper signatures):

- **add** → `genre_proposal`: `kind` (umbrella/genre/subgenre/microgenre/family), `role`, `status`, `specificity_score`, `parent_edges[]` (target/edge_type/weight/confidence), `similar_to[]`, `rationale`. A leaf needs **≥1 parent edge** or validation fails.
- **alias** → `alias_proposal`: `canonical_target`, `rationale`. Verify the target exists first; no-op if the alias already exists.
- **reject** → reject record with a `reject_reason` from the enum (`label, artist_name, release_title, place, format, era, user_list, malformed, joke_tag, negative_tag, retail_bucket, source_noise, unknown_noise`).

**Bake the placement guardrails into `ADJUDICATOR_INSTRUCTIONS`** (verbatim from the `taxonomy-growth` skill): umbrellas are low-specificity (~0.24–0.42) with spread parentage; instrument-led terms → facets, not genre leaves; don't collapse distinct genres into co-occurrence aliases (`uk garage` ≠ `garage rock`); specificity ladder umbrella 0.24–0.42 · genre 0.48–0.66 · subgenre 0.62–0.82; broad/noisy → `status: review` not `active`.

**Payload to Claude** must include a *relevant slice of the taxonomy* (candidate parents/aliases by name + co-occurrence, plus family/umbrella anchors) so it places against the real graph — not the whole 760-record file.

---

## 5. Wiring (mirror the GenreReview path exactly)

### API — `src/playlist_web/app.py` (model on the review routes at `:276-319`)
- `GET  /api/taxonomy/queue`     → worker `get_taxonomy_queue` (untracked)
- `GET  /api/taxonomy/completed` → worker `get_taxonomy_completed` (untracked)
- `POST /api/taxonomy/adjudicate`→ worker `adjudicate_taxonomy_term` (untracked; **calls Claude, returns verdict, does NOT persist**)
- `POST /api/taxonomy/decision`  → worker `record_taxonomy_decision` (untracked; accept/edit/reject/revert → staging)
- `POST /api/taxonomy/apply`     → **tracked job** `apply_taxonomy_decisions`

Add request/response schemas to `src/playlist_web/schemas.py` and TS types to `web/src/lib/types.ts` (mirror `EscalationOut` / `EscalationDecisionRequest` at `types.ts:181-203`).

### Worker — `src/playlist_gui/worker.py`
- Untracked handlers (inline, quick) modeled on `handle_get_escalation_queue` (`:2684-2702`), `handle_apply_escalation_decision` (`:2726-2758`).
- **Tracked** `handle_apply_taxonomy_decisions` modeled on `handle_publish_decided` (`:2550-2580`); register in `TRACKED_COMMAND_HANDLERS` (`:2817`). The publish handler is your template for: timestamped backup → do the work → emit progress/result/done with stats.

### Frontend — `web/src/components/TaxonomyReviewPanel.tsx`
- Mirror `web/src/components/GenreReviewPanel.tsx` (queue fetch, pending/completed toggle, search, keyboard shortcuts, decision submit, job-tracking via `useWorkerEvents`).
- Register as a tab in `web/src/components/AdvancedPanel.tsx` (tab buttons at `:26-28`, render switch at `:33`).
- API client methods in `web/src/lib/api.ts` (mirror `reviewQueue/reviewDecision/reviewPublish` at `:125-142`).
- Per-row: term · affected albums/tracks · co-occurring tags · example releases · **"Ask Claude"** button → renders verdict (parent edges + specificity, or alias target, or reject reason) → **Accept / Edit / Override / Reject**. Plus an **"Apply N decisions"** button showing result stats incl. "M albums will re-classify on next publish" and any deferred edges.

---

## 6. The Apply algorithm (the load-bearing part — get this right)

Run inside the tracked worker job. Steps:

1. **Re-read the YAML fresh** from disk (do NOT trust a snapshot taken when decisions were recorded — concurrent worktree sessions may have changed it). Load via `load_layered_taxonomy(DEFAULT_TAXONOMY_PATH)`.
2. **Build `GrowthProposal`s** from `list_pending()`.
3. **Validate the whole batch** with `graph_growth.validate_proposal(taxonomy, p)` per proposal — this is the skill's "isolated-copy test". Expect `N OK, 0 FAIL`. **If any fail, abort the write** and return the failures to the GUI. Skip `reject`/`microgenre` kinds in the preflight loop (validator doesn't support them; ingest handles them fine).
4. **Handle same-batch forward references** (see `taxonomy-growth` skill, the #1 gotcha): if a proposal's parent/`similar_to` target is *another pending new term in this same batch*, topologically order the ingest so parents land first. If a true forward/cyclic edge remains, **trim it** and queue a follow-up edge-upgrade (surface it in the GUI as "deferred"). For the common case (targets are existing canonical genres) this is a no-op.
5. **Timestamped backup** of `layered_genre_taxonomy.yaml` (same discipline as the `metadata.db` backups in `handle_publish_decided`).
6. **Write** via `append_approved_to_taxonomy(path, approved, new_version="0.X.0-gui-YYYYMMDD-grown")`. Signature has **no `dry_run`** — for dry runs ingest into a temp copy.
7. **Reload** so the running process sees the new graph (see cache note below).
8. **`mark_applied(batch_version)`** on the staging rows; compute "M albums will re-classify" by counting albums whose legacy/observed tags match the just-added/aliased terms.
9. Emit stats: added / aliased / rejected counts, deferred edges, backup path, affected-album count.

**Cache note:** `load_default_layered_taxonomy()` reads the file fresh each call (no `@lru_cache` at its definition — `layered_taxonomy.py:243`). BUT verify whether any *long-lived consumer* in the worker holds a `LayeredTaxonomy` singleton/`@lru_cache`; if so, bust it after write. Grep for `lru_cache` / module-level taxonomy globals in the worker and genre runtime before assuming reload is automatic.

**Safety invariants:** validate-before-write; backup-before-write; one Apply = one reviewable YAML diff = one commit (Dylan commits it himself); never touch `metadata.db` in this feature.

---

## 7. Scope boundaries (what this feature deliberately does NOT do)

This came up directly from a real Analyze run. Keep these out of scope:

- **It does not clear the `graph genre_id 'X' has no canonical name` warnings by itself.** Those warnings (from `build_beat3tower_artifacts.py:494`) are raw tags on the ~40 **legacy / un-enriched** albums (verified: all 30 unmapped ids are `assignment_layer='legacy'`). Clearing them needs (a) the term in the taxonomy [this feature], **plus** (b) those legacy albums re-adjudicated/re-published, **plus** (c) the artifact rebuilt. Only (a) is in scope. The panel's "will re-classify on next publish" message is the honest hook to (b)/(c). Note the warning is a *graceful degradation* ("keeping raw id as token"), not a crash.
- **It does not fix album-assignment accuracy** (e.g. the `antonio carlos jobim::wave: missing {'latin_jazz'}` validation miss). That's the existing album-level GenreReview / adjudication lane, not vocabulary.
- **It does not re-tag albums.** Decision 3 above.

---

## 8. Test plan

- **Unit:** queue merge/dedup/impact-count; staging `record_decision`/`revert`/`mark_applied`; adjudicator `validate_response` → `GrowthProposal` mapping; Apply batch-validation + forward-ref sequencing + backup + version bump **against a temp copy of the YAML**; reject-reason enforcement; alias no-op-when-exists.
- **Carry the `taxonomy-growth` trap-catalog deselects** for the known pre-existing failures in `test_ai_genre_hybrid_cli.py` / `test_ai_genre_hybrid_evidence.py` (3+1 deselects, listed in that skill) — they fail on unmodified taxonomy; not your regressions.
- **Run pytest bounded, never piped** (`python -m pytest -q -m "not slow"`, use the tool timeout — see project CLAUDE.md).
- **End-to-end (per `web-gui` skill):** rebuild `web/dist`, restart the worker, exercise the real path (queue → Ask Claude → accept → Apply → confirm a temp-copy YAML gained the record + version bump), look for regressions adjacent to the change. Don't claim done until you've run it.

---

## 9. Suggested build order

1. **Backend queue builder + staging store** (+ unit tests) — no Claude, no GUI yet. Verifiable in isolation.
2. **Claude adjudicator contract** (+ unit test on `validate_response` mapping). Model on `album_adjudicator.py`.
3. **Apply engine** (+ unit test against temp-copy YAML, incl. forward-ref case). This is the riskiest; do it before the GUI.
4. **API routes + worker handlers** (untracked first, then the tracked Apply job).
5. **Frontend panel + tab registration + types.**
6. **End-to-end pass** per §8.

Phases 1–3 are pure backend and fully testable without the GUI — land them first so the GUI session is wiring against a verified core.

---

## 10. Open questions for the GUI session to resolve in-flight (not blockers)

- Exact impact-count semantics: albums vs tracks, and whether to count only legacy-layer occurrences or all observed. (Recommend: distinct albums where the term appears as an observed/legacy tag.)
- Whether "Ask Claude on all un-triaged" batch button ships in v1 or is deferred (Decision 4 makes it optional).
- Where exactly the long-lived taxonomy cache (if any) lives in the worker — resolve via grep during Apply implementation (§6).
