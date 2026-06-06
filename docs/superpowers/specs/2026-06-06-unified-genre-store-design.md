# Unified Genre Store — Design (Sub-project 1)

**Date:** 2026-06-06
**Status:** Approved (design); implementation plan pending
**Branch:** `codex/layered-genre-graph`

## Context

The layered genre graph produces **layered graph assignments** — the
authoritative genre signature for a release (e.g. `east_coast_hip_hop`,
`bossa_nova`, `indie_alternative`). These currently live only in the enrichment
sidecar (`data/ai_genre_enrichment.db`), keyed by `release_key`
(`"norm_artist::norm_album"`). Playlist generation reads genres from
`metadata.db` (keyed by `album_id` / `track_id`), built with
`genre_source=legacy`. The two stores are disconnected, so enriched genres do
not yet influence playlists.

**Goal (overall, across sub-projects):** layered graph assignments become the
single authority for genre across all playlist-generation features, with legacy
genres as the fallback for releases that have no graph assignment. Consolidate
onto **one runtime database** (`metadata.db`).

**This sub-project (SP1)** establishes the unified *store* and the tooling to
populate it. It deliberately stops before changing how the playlist engine
*consumes* genres (SP2).

### Decisions locked during brainstorming

- **"Enriched genres" = the layered graph assignments**, specifically. The older
  `enriched_genres` table becomes input/provenance, not the answer.
- **Publish/mirror model (not full consolidation).** The AI enrichment pipeline
  keeps the sidecar as a high-churn build-time scratch DB. A **publish** step
  copies only the authoritative layered genres (+ taxonomy + minimal provenance)
  into clean tables in `metadata.db`. `metadata.db` is the single source of truth
  for *reads*; the sidecar is rebuildable cache. This keeps enrichment churn off
  the irreplaceable production DB.
- **Approach A — materialized resolved table.** The publish step builds one
  already-resolved table (`release_effective_genres`): graph genres where a
  release has them, legacy where it doesn't, with a `source` column. Fallback is
  computed once, in one auditable place. Consumers read one table.
- **Release/album grain** for the resolved table (not per-track). Per-track
  fan-out happens at artifact-build time (SP2).

### Sequencing (the four sub-projects)

1. **SP1 — Unified genre store** (this doc): target schema in `metadata.db`,
   publish tooling, album_id linkage, read API. Proven against a copy.
2. **SP2 — Enriched genres drive playlists**: point the artifact build + genre
   resolver at the unified store. Owns the vocabulary-unification problem
   (144-term graph vocab vs the legacy raw vocab in the same feature space).
3. **SP3 — Analyze Library auto-enrichment**: detect sparse albums, enrich
   (AI web `auto`, reusing existing scraped tags), build graph assignments,
   call publish.
4. **SP4 — Analyze Library in the web GUI**: endpoint + React UI.

**SP1 boundary:** ends at *"`metadata.db` has clean, album_id-linked, resolved
authoritative genre tables + a read API + migration/publish tooling, proven
against a copy."* Vocabulary unification and artifact consumption are **SP2**.

## Grounding (audit, 2026-06-06)

`metadata.db` genre tables:
- `album_genres` (19,902): musicbrainz_release, discogs_master, discogs_release
- `artist_genres` (5,928): musicbrainz_artist (+ inherited)
- `track_genres` (27,142): file (ID3)
- `track_effective_genres` (256,178): resolved per-track legacy view
- `genre_canonical_token` (696), `genre_raw_map` (857): normalization maps
- `albums` PK `album_id`; `tracks` carry `album_id`, `norm_artist`

Sidecar (`ai_genre_enrichment.db`) relevant tables:
- `genre_graph_release_genre_assignments` (325 rows / **39 releases**) — authority
- `genre_graph_release_facet_assignments` (3)
- taxonomy: `genre_graph_canonical_genres` (144), `genre_graph_edges` (342),
  `genre_graph_aliases` (40), `genre_graph_canonical_facets` (30),
  `genre_graph_bridge_rules` (8), `genre_graph_rejected_terms` (14)
- `enriched_genre_signatures` (498, **album_id 100% populated**) — the
  `release_key → album_id` bridge
- `enriched_genres` (6,082 / 497 release_keys) — provenance/input
- `ai_genre_source_pages` (946), `ai_genre_source_tags` (7,660),
  `ai_genre_tag_classifications` (7,660) — scraped tags (do not re-scrape)
- `ai_genre_suggestions` (22,863), `ai_genre_release_checks` (4,121)
- `ai_genre_user_overrides` (2,003) — user add/remove, keyed by release_key

**Key facts that shaped the design:**
1. Authority covers only **39 releases** today, but **497** have enriched genres
   and **436** have scraped source tags; `graph-build-assignments` also reads
   MB/Discogs tags from `metadata.db`. So the authority can be populated for most
   albums **with no new AI** — pure fusion of existing evidence.
2. The authority table has **no `album_id`**; the link must be established at
   publish time via `enriched_genre_signatures` (exact) and recomputation from
   `albums` (fallback).
3. `release_id` in the authority table == `release_key`
   (`norm_artist::norm_album`).

## Schema (new tables in `metadata.db`)

All additive. Legacy genre tables are never modified. Reversal = `DROP`.

### (a) Taxonomy (verbatim copy from sidecar)
`genre_graph_canonical_genres`, `genre_graph_canonical_facets`,
`genre_graph_edges`, `genre_graph_aliases`, `genre_graph_bridge_rules`,
`genre_graph_rejected_terms`, plus:

```
genre_graph_taxonomy_meta
  version        TEXT
  fingerprint    TEXT
  published_at   TEXT
  (single row)
```

### (b) Authority (graph assignments + the missing link)
`genre_graph_release_genre_assignments` and `genre_graph_release_facet_assignments`:
sidecar columns **plus** `album_id TEXT` (nullable). User overrides applied.

### (c) Resolved table (heart of Approach A)

```
release_effective_genres
  album_id          TEXT     -- join key to tracks/albums
  release_key       TEXT     -- provenance/debugging
  genre_id          TEXT     -- graph genre_id, or normalized legacy genre
  assignment_layer  TEXT     -- observed_leaf | inferred_parent | inferred_family | legacy
  confidence        REAL
  source            TEXT     -- 'graph' | 'legacy' | 'user'
  PRIMARY KEY (album_id, genre_id, assignment_layer)
  INDEX (album_id)
```

Covers every `album_id` in `albums`; exactly one base source per album
(`graph` if it has assignments, else `legacy`), with user overrides layered on
top regardless of source.

## Publish algorithm (`publish-genres`)

Single command, one transaction against `metadata.db`. Idempotent
(drop-and-rebuild each step → running twice yields identical output).

1. **Copy taxonomy.** Drop & recreate the six taxonomy tables + `taxonomy_meta`.
2. **Resolve `release_key → album_id`:** (1) `enriched_genre_signatures` exact;
   (2) recompute `normalize_release_artist(artist)::normalize_release_name(title)`
   from `albums` for uncovered keys; (3) collisions resolve by sorted `album_id`,
   logged.
3. **Populate authority tables** with resolved `album_id` stamped on each row.
   Unresolved keys kept with `album_id = NULL` (excluded from the resolved table).
4. **Apply user overrides** (`ai_genre_user_overrides`): map names → `genre_id`
   via taxonomy classifier; remove removed; add added as `observed_leaf`,
   `source='user'`, `confidence=1.0`; unmappable adds skipped + logged.
5. **Build `release_effective_genres`** for every `album_id`: graph rows where
   present (`source='graph'`), else legacy genres (`source='legacy'`), then apply
   overrides as a final pass.

   The legacy half mirrors what the artifact's `genre_source=legacy` path reads
   today — `load_genres_for_tracks` in `build_beat3tower_artifacts.py`, a
   per-track weighted blend of `track_genres` (weight 1.0) + `album_genres`
   (0.8) + `artist_genres`, normalized via `genre_raw_map`/`genre_canonical_token`.
   Because the resolved table is album-grain, the legacy half is that same
   construction **aggregated up to `album_id`** (union across the album's tracks,
   max weight per genre). This is a deliberate consequence of the release-grain
   decision; it loses per-track legacy nuance for fallback albums. **SP2 owns the
   call** on whether album-grain legacy is sufficient at artifact-build time or
   whether it reads per-track legacy directly for fallback tracks — SP1 only
   needs to expose a correct, inspectable album-level resolution.

`--dry-run` prints per-source album counts, override hits, and unlinked-release
counts without writing.

## Read API — `src/genre/authority.py`

Single import point for SP2 and all future features; simple indexed SELECTs over
the materialized table:

- `resolved_genres_for_album(conn, album_id) -> list[GenreRow]`
- `resolved_genres_for_track(conn, track_id) -> list[GenreRow]` (joins
  `track → album_id`)
- `genre_source_for_album(conn, album_id) -> 'graph' | 'legacy' | 'none'`
- taxonomy helpers: `parents_for(genre_id)`, `families_for(genre_id)`,
  `is_facet(genre_id)`

Consumers never re-implement the fallback.

## Safety & reversibility

- **Additive only.** Legacy tables untouched; reversal = `DROP` published tables.
- **Develop against a copy.** `metadata.db → metadata.db.worktest`; build and run
  publish against the copy; validate; only then back up the live DB
  (`metadata.db.bak.<ts>`), get explicit second confirmation, and run live.
  Honors the project rule on `metadata.db`.
- **No writes to the live DB during design.**

## Testing

- **Unit** (synthetic fixture DB): graph-where-present / legacy-elsewhere;
  overrides add+remove+win; album_id linkage via signature and recomputed paths;
  idempotency (twice → identical); reversal leaves legacy tables byte-intact.
- **Real-data validation** (against the copy): every `album_id` resolves to
  exactly one base source; no album dropped; collision count sane; spot-checks
  (ATCQ → `east_coast_hip_hop`/`jazz_rap`/`boom_bap`; Jobim →
  `bossa_nova`/`latin_jazz`/`mpb`; a tag-less album → `legacy`).
- Existing suite stays green.

## Out of scope (SP1)

- Changing the artifact build or playlist engine to consume the resolved table
  (SP2).
- Vocabulary unification of graph vs legacy genre spaces (SP2).
- Auto-enrichment of sparse albums and Analyze Library wiring (SP3).
- Web GUI (SP4).
- Retiring/deleting the sidecar (it remains the enrichment scratch DB).
