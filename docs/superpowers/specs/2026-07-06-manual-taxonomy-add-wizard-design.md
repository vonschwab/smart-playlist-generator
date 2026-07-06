# Manual taxonomy ADD wizard — design

**Date:** 2026-07-06
**Status:** Approved design (brainstorm complete; implementation plan to follow)
**Goal:** Let a user *without* Claude grow the canonical taxonomy themselves: a governed, inline step-flow in the Taxonomy Review panel that does manually what "Ask Claude" does for the **ADD** verdict — place a new genre/subgenre or facet into the taxonomy with the same structural guardrails — so the taxonomy stays valid and canonical no matter who grows it.

## Context

The Taxonomy Review panel (`web/src/components/TaxonomyReviewPanel.tsx`) triages unknown terms with three actions: **Ask Claude** (LLM proposes add/alias/reject with full placement), **Alias…** (manual, no LLM), and **Reject** (manual, no LLM). The only path that can **add a new record** is Claude-dependent. The taxonomy is a living vocabulary — other users' libraries will contain genres Dylan's doesn't, and new sub-genres keep emerging — so end users need a first-class manual ADD path. The hard part of ADD is placement (kind, parent edges, specificity) and the hard-won governance rules around it (`.claude/skills/taxonomy-growth/SKILL.md`); the wizard's job is to walk a human through those decisions and make invalid output impossible to stage.

### Decisions taken (with Dylan, 2026-07-06)

1. **Governed form, phased.** v1 is a guided, guardrailed form — the user supplies the musical judgment (kind, parents, specificity, aliases); the wizard supplies structure, governance hints, and validation. Deterministic *suggestions* (pre-filled parents/kind from co-occurrence + the similarity graph) are an explicitly deferred phase 2.
2. **Kinds: genre/subgenre + facet.** The genre-vs-facet fork is a core governance rule (instrument-led terms like "jazz piano" are facets, not genre leaves) and the wizard actively offers it. Umbrella and microgenre are deferred.
3. **Inline step flow** in the term card (where Ask Claude / Alias / Reject live), matching the panel's compact style. Not a modal, not a single flat form.

## Architecture — one new endpoint, everything else reused

The wizard builds the **same `TaxonomyProposal` shape** (`web/src/lib/types.ts:224`) that a Claude ADD verdict produces, then submits through the **existing** staging call: `onDecide("add", proposal, /*claude*/ null, /*humanEdited*/ true)` — the exact pattern the manual Alias path already uses. Everything downstream is untouched and already proven: `api.taxonomyDecision` → `taxonomy_decision_store` staging → the "Apply N decisions" job (isolated-copy validation → timestamped YAML backup → version bump → write to `data/layered_genre_taxonomy.yaml`).

**The one new backend piece:** a validation endpoint so the wizard can show authoritative structural errors *before* staging.

- Worker: new **untracked** (inline, low-latency) command `validate_taxonomy_proposal` in `src/playlist_gui/worker.py`, mirroring `get_genre_review_queue`'s readonly shape. It converts the `TaxonomyProposal` JSON to a `GrowthProposal` and returns `{"errors": validate_proposal(taxonomy, gp)}` against the live loaded taxonomy.
- Web: thin `POST /api/taxonomy/validate` route in `src/playlist_web/app.py` + `api.taxonomyValidate(...)` client method, mirroring the existing taxonomy routes.

**Guardrail authority stays single-sourced:** `graph_growth.validate_proposal` (graph_growth.py:389) is the only rule engine. The wizard performs only trivial client-side step gating (can't advance without a parent / facet_type); every structural rule — leaf-needs-parent, facet-has-no-parents, parent-must-exist-and-not-be-a-facet-or-alias, duplicate-name, specificity range, `term_kind_confirm` match — is enforced server-side by the same function the apply path uses. No rule is re-implemented in TypeScript.

## The step flow

A new **"Add manually…"** button appears alongside Ask Claude / Alias… / Reject in the term card's initial action row. It opens a four-step inline flow (state machine inside a new `TaxonomyAddWizard` component; `TermCard` stays thin):

**Step 1 — What is this term?**
- Fork: **Genre / subgenre** (a style of music with a place in the hierarchy) vs **Facet** (a descriptor — mood, instrumentation, era, region…).
- Governance hint shown inline: *"Instrument-led terms ('jazz piano', 'jazz guitar') are usually facets (instrumentation), not genres — unless there's a genuine scene/style tradition beyond the instrument."*
- **Canonical name field**, prefilled with the queue term, editable — Claude's ADD verdicts can rename a raw term to its clean canonical form and the wizard must allow the same (raw `"shoe gaze"` → name `"shoegaze"`). If the final name differs from the original term, the wizard auto-adds the original term to `alias_variants` so the raw spelling still resolves after ingest (the ingest helper already skips a variant identical to the record name).

**Step 2 — Placement** (content depends on the fork)
- *Genre/subgenre:* pick **≥1 parent** via the existing `GenreAutocomplete` (targets constrained to existing canonical genres — forward references and invented targets are unselectable by construction). Per parent, one of exactly two governed edge presets (no free-form weights in v1):
  - **Strong parent** → `is_a`, weight 0.75, confidence 0.85 — "X *is a kind of* parent"
  - **Family context** → `family_context`, weight 0.55, confidence 0.80 — "X belongs in this family's orbit"
  - Optional **similar_to** multi-pick (same autocomplete); these become `bridge_to` edges on ingest.
- *Facet:* pick a **facet_type** from the enum (`mood, texture, instrumentation, production, era, region, function, vocal, scene, format, rhythm`). Parent/similar pickers are not rendered (the validator forbids them on facets).
- Also choose **genre vs subgenre** kind here (leaf granularity), with a one-line hint: subgenre = a recognized style *within* a named parent genre.

**Step 3 — Specificity + aliases**
- Specificity slider showing the governance ladder as labeled bands: *genre 0.48–0.66 · subgenre 0.62–0.82* (defaults: genre 0.55, subgenre 0.70, facet 0.50). Values outside 0–1 impossible; values outside the band allowed but visually flagged.
- Optional **alias spellings** (free-text chips) — variant spellings that should resolve to this record.
- `status` fixed to `active` in v1 (a "mark for review" toggle is unnecessary for a human-authored deliberate add).

**Step 4 — Review + validate + stage**
- Renders the assembled proposal exactly like the Claude path's `VerdictSummary` (kind, parents with edge chips, aliases, specificity).
- Calls `POST /api/taxonomy/validate`; errors render inline (same style as the panel's existing error text). **Stage is disabled until the error list is empty.**
- **Stage decision** → `onDecide("add", proposal, null, true)`. The decision joins the pending queue; the existing **Apply** button writes it with the existing safety train (isolated-copy validation, timestamped backup, version bump). Nothing applies immediately.

Back-navigation between steps preserves entered state; Cancel discards.

### Proposal assembly (exact field mapping)

| Field | Source |
|---|---|
| `name` | Step 1 name field (trimmed, lowercased per taxonomy convention) |
| `kind` | `genre` \| `subgenre` \| `facet` (Steps 1–2) |
| `status` | `"active"` |
| `specificity_score` | Step 3 slider |
| `parent_edges` | Step 2 picks with preset `edge_type`/`weight`/`confidence` (empty for facets) |
| `similar_to` | Step 2 optional picks (empty for facets) |
| `alias_variants` | Step 3 chips + auto-added original term when renamed |
| `term_kind_confirm` | auto: `"facet"` for facets, `"genre"` otherwise (validator requires the match) |
| `facet_type` | Step 2 facet pick (null for genres) |
| `rationale` | optional free-text note field on Step 4 (small, single line; lands in the record's audit trail like Claude's rationale) |

## Why this is structurally simpler than Claude's batch loop

The taxonomy-growth playbook's #1 recurring trap — same-batch forward references requiring trim → PASS-N edge restoration — **cannot occur** here: the wizard places one term at a time and its pickers only offer targets that already exist in the live taxonomy. The deferred-edge queue, batch scripts, and PASS-N machinery are irrelevant to this path.

## Error handling

- Server `validate_proposal` errors are the authoritative gate; the wizard displays them verbatim and blocks staging.
- The validate endpoint failing (worker down, etc.) blocks staging with the error shown — never "stage anyway."
- Duplicate names (term already canonical or aliased) surface as the validator's duplicate error at Step 4; the hint suggests the Alias path instead.
- The apply path's existing failure mode is unchanged: validation failure at apply time writes nothing and reports per-term errors (`applyStats.validation_failures`).

## Testing

- **Backend:** unit tests for the worker command + route — valid genre proposal → `[]`; leaf without parent → error; facet with parent → error; parent that is a facet/alias/nonexistent → error; duplicate name → error; `term_kind_confirm` auto-match. Reuse `graph_growth` test fixtures.
- **Frontend:** component tests for the wizard state machine — genre path emits correct `parent_edges` with preset shapes; facet path emits `facet_type` + no parents; rename auto-adds the original term to `alias_variants`; validation errors block Stage; Stage calls `onDecide` with the exact assembled proposal; Cancel/back preserve/discard state correctly.
- **E2E (integration):** stage a genre through the wizard against a tmp taxonomy copy → run the apply job → assert the record + alias variants landed and the version bumped. Never touches the real YAML in tests; never touches `metadata.db` at all.

## Out of scope (deliberate)

- **Phase 2:** deterministic suggestions (pre-filled kind/parents/specificity from co-occurring known tags + the taxonomy similarity graph).
- Umbrella and microgenre kinds (rare; Claude/curator path covers them).
- Free-form edge weights/types beyond the two presets.
- Any change to Ask Claude, Alias…, Reject, the decision store, or the apply pipeline.
- Any write to `metadata.db` (taxonomy growth touches only the YAML, as ever).

## Open items for the implementation plan

- Exact worker command registration + app route naming (mirror the nearest existing taxonomy endpoints).
- Whether `GenreAutocomplete`'s source list includes review-status genres (acceptable either way — the validator is the gate; but note it in the picker's placeholder if excluded).
- Slider vs numeric input for specificity (match existing GUI control conventions).
