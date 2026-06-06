# SP3a — Graph Growth Pre-Pass — Design

**Date:** 2026-06-06
**Status:** Approved (design); implementation plan pending
**Program:** Genre enrichment (SP1 done; SP2 collection tooling done). This is SP3a.

## Context

The layered genre graph (`data/layered_genre_taxonomy.yaml`, `taxonomy_version
0.2.0-expanded`, 230 records) is the single authority for genre. But 144-ish
canonical genres cannot cover the real library vocabulary. After the SP2 scrapes
(Last.fm + Bandcamp + the existing MusicBrainz/Discogs/file tags), the library's
*collected* tag vocabulary contains many real genres with no home in the graph
yet (`vaporwave`, `jangle pop`, `chillwave`, …), plus noise (`aaron`, `1996`,
`4ad`) and facets (`acoustic`, `instrumental`).

**SP3a grows the graph to cover that vocabulary, with the user approving every
addition, *before* any album enrichment runs.** This is the "living document"
mechanism. It deliberately stops before per-album enrichment (SP3b).

### Why grow before enriching (locked decision)

Album enrichment quality depends on the graph being complete. Enriching an album
against a graph still missing `vaporwave` produces a worse signature and forces
re-enrichment later. Growing first → each album is enriched once, against a
stable taxonomy. We also curate a *deduped vocabulary* (hundreds of distinct
terms ranked by impact), not a per-album stream.

### Decisions locked during brainstorming

- **AI proposes, user approves.** Every new node is the user's call; the AI does
  the tedious "where might this fit" first pass. The user is the final authority.
- **Full placement proposal.** For each candidate the AI proposes the whole
  package — name/slug, kind, parent edges, specificity, similar-to edges —
  grounded in the current taxonomy + the term's evidence. The user edits/rejects.
- **Alias-collapse before proposing.** Obvious variants (`synthwave` /
  `synth wave` / `retrowave`) collapse to one node up front to shrink the review
  pile; the graph already has an alias mechanism.
- **Structural validation on ingest.** Parent targets must exist, no cycles,
  slug must be unique, family consistent — a bad proposal is flagged, never
  silently corrupts the graph.
- **File-based review.** Proposals land in an editable YAML the user reviews
  directly (keep / edit / reject); an ingest command applies the approved rows.
  No dependency on the GUI (SP5). The YAML stays the human source of truth, so
  every growth edit is a reviewable git diff.
- **Candidate threshold:** a term must appear on **≥3 albums** to be
  auto-proposed (tunable). Rarer terms still map if they already fit; they just
  aren't pushed into the review pile as one-offs.

## Grounding (current code)

- **Taxonomy source of truth:** `data/layered_genre_taxonomy.yaml`, a `records`
  list. Each record: `name`, `kind` (family/umbrella/genre/subgenre/microgenre/
  facet/alias/reject), `role`, `status` (active/review/…), `specificity_score`,
  `canonical_target` (aliases), `parent_edges` (list of
  `{target, edge_type, weight, confidence, notes}`), `notes`, etc. `genre_id` is
  **derived by slugifying `name`** (`"east coast hip hop"` → `east_coast_hip_hop`).
  Loaded by `load_layered_taxonomy()` (`src/ai_genre_enrichment/layered_taxonomy.py`)
  and imported into the sidecar `genre_graph_*` tables via
  `SidecarStore.upsert_layered_taxonomy()`.
- **Term classification:** `classify_layered_term(taxonomy, term)`
  (`src/ai_genre_enrichment/layered_assignment.py`) returns
  `LayeredTermClassification(term_kind, canonical_id, …)` with `term_kind` ∈
  {mapped genre kinds, `alias`, `facet`, `reject`, `review`}. **A growth
  candidate is `term_kind == "review"` AND `canonical_id is None`** (genuinely
  unmapped). `review` *with* a `canonical_id` means the term already maps to an
  existing genre that merely has review status — NOT a growth candidate.
- **Collected tags live in** `ai_genre_source_tags` (`normalized_tag`) joined to
  `ai_genre_source_pages` (`release_key`, `source_type`). Album-frequency = count
  of distinct `release_key` per `normalized_tag`.
- **Structural helpers on `LayeredTaxonomy`:** `genre_by_id`, `parents_for_genre`,
  `families_for_genre`, `facet_by_id`, `alias_target_for_name` — used for
  validation and for building proposal context.
- **Precedent:** `graduate-reviewed` / `graduate-ai` already append accepted tags
  into a vocab YAML — same shape of "review file → ingest into YAML" we reuse here
  (but targeting the *layered taxonomy* YAML, not `genre_vocabulary.yaml`).

## Components & data flow

```
collected tags (SP2)
  → [1 gather+classify+rank]  distinct unmapped genres, freq ≥ threshold
  → [2 AI propose placement]  per-candidate full placement + evidence
  → proposal YAML  ──(user edits: keep/edit/reject)──►  reviewed YAML
  → [3 ingest]  validate → append records to layered_genre_taxonomy.yaml
              → bump taxonomy_version → re-import into graph tables
  ⇒ grown, stable taxonomy   (input to SP3b)
```

### 1. Candidate gathering — `graph-propose-growth` (phase A, no AI)
Aggregate distinct `normalized_tag` across `ai_genre_source_tags` with
album-frequency. For each, run `classify_layered_term`:
- `reject` / `facet` / `alias` / already-mapped genre (incl. `review` *with* a
  `canonical_id`) → **drop** (already handled).
- `review` with `canonical_id is None` (genuinely unmapped) and album-frequency ≥
  threshold → **candidate**.
Collapse near-duplicate candidates (normalized edit-distance + shared-token
heuristic) into one representative, recording the variants as alias suggestions.
Output: an ordered candidate list (highest impact first) with evidence per
candidate: album-frequency, top co-occurring tags, up to N example
`artist — album` strings.

### 2. AI placement proposal — `graph-propose-growth` (phase B, AI)
For each candidate, build a prompt with: the candidate term, its evidence, and a
**taxonomy context** (the family/umbrella/genre neighborhood likely relevant —
to keep prompt size bounded, send families + genres whose names share tokens or
co-occur, not the whole graph). The AI returns a strict-schema proposal:
- `name` (human-readable, canonical), `kind` (`genre`|`subgenre`),
  `specificity_score` (0–1), `status` (`active`|`review`),
- `parent_edges`: list of `{target (existing record name), edge_type
  (`family_context`|`is_a`), weight, confidence}`,
- `similar_to`: optional list of existing names (→ `bridge_to`/`scene_adjacent`
  edges),
- `alias_variants`: the collapsed variants to register as aliases,
- `term_kind_confirm` (`genre`|`facet`|`noise`) — lets the AI veto a candidate
  the deterministic classifier mis-bucketed,
- `rationale` (one line).

Web search is **off by default** for placement (music-genre knowledge is usually
in-model); `--web-mode auto` available for tunability. ~hundreds of calls total
(distinct candidates), not per-album — modest cost.

The proposal + evidence is written to an **editable YAML** at
`data/genre_growth/proposals_<date>.yaml`, each entry carrying a
`decision: pending` field.

### 3. Review (human) then ingest — `graph-ingest-growth`
The user edits the proposal YAML: set `decision` to `keep` / `reject`, and edit
any placement fields inline. `graph-ingest-growth` reads entries with
`decision: keep`, then **validates** each before applying:
- slug (`genre_id` from `name`) is unique (not an existing record),
- every `parent_edges[].target` and `similar_to` target exists in the taxonomy,
- adding the node introduces no `is_a`/`family_context` cycle,
- `kind`/`role` consistent (`genre`/`subgenre` → `role: leaf`),
- `specificity_score` ∈ [0,1].
Valid entries are appended to `layered_genre_taxonomy.yaml` as new records
(genre record + alias records for variants + any new edges), `taxonomy_version`
is bumped (e.g. `0.2.0-expanded` → `0.3.0-grown-<date>`), and the taxonomy is
re-imported into the sidecar graph tables via `upsert_layered_taxonomy`.
`--dry-run` prints what would be appended + any validation failures, writes
nothing. Invalid entries are reported and skipped (never partially applied).

## Schema — proposal YAML entry

```yaml
- term: "vaporwave"            # the collected tag (candidate)
  album_frequency: 14
  cooccurring_tags: ["chillwave", "synthwave", "ambient"]
  examples: ["Macintosh Plus — Floral Shoppe", "..."]
  decision: pending            # user edits: keep | reject
  proposal:
    name: "vaporwave"
    kind: subgenre
    status: active
    specificity_score: 0.8
    parent_edges:
      - {target: "electronic", edge_type: family_context, weight: 0.55, confidence: 0.8}
    similar_to: ["chillwave"]
    alias_variants: ["vapor wave", "vapourwave"]
    term_kind_confirm: genre
    rationale: "Internet-era plunderphonic electronic microgenre."
```

## Safety & reversibility

- **Additive + git-versioned.** Growth only appends taxonomy records; every edit
  is a reviewable diff. Reversal = revert the YAML and re-import. No writes to
  `metadata.db` in SP3a (that's SP3b's publish step).
- **Dry-run** on ingest; validation rejects bad entries atomically.
- **User-gated.** Nothing enters the taxonomy without `decision: keep`.

## Testing

- **Candidate selection:** mapped/alias/facet/reject tags excluded; only
  `review`-kind genres above the frequency threshold survive; ranking by
  album-frequency.
- **Alias collapse:** near-duplicate variants merge to one candidate with the
  others as `alias_variants`.
- **Proposal round-trip:** proposal YAML writes and re-reads; `decision: pending`
  default; AI call mocked.
- **Structural validation:** dangling parent target → rejected; cycle → rejected;
  duplicate slug → rejected; out-of-range specificity → rejected; valid entry →
  accepted.
- **Ingest:** approved entry appends a correct genre record (+ alias records),
  bumps `taxonomy_version`, and the taxonomy re-loads via
  `load_layered_taxonomy()` with the new genre present and
  `parents_for_genre`/`families_for_genre` resolving.
- **Real-data smoke (against a copy of the sidecar):** the candidate pass over
  the live collected tags produces a sane candidate count and the top candidates
  are recognizable genres, not noise.

## Out of scope (SP3a)

- Per-album enrichment and re-publish to `metadata.db` (SP3b).
- The genre-editing GUI (SP5) — file-based review is the SP3a surface; the GUI
  later becomes a nicer front-end over the same propose/ingest machinery.
- Changing the artifact build / playlist engine (SP4).
