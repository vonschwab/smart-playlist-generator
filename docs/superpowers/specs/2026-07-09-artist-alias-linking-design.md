# Manual artist-alias linking — design

**Date:** 2026-07-09
**Status:** Approved design, pre-plan
**Author:** Dylan + Claude (brainstorming session)

## Problem

Some artists appear in the library under multiple names that no algorithm can
reconcile, because the strings are semantically unrelated:

- **Spelling / formatting variants of one project** — `Alex G` vs `(Sandy) Alex G`.
- **One person, genuinely distinct projects** — `Smog` / `Bill Callahan`,
  `Mount Eerie` / `The Microphones`, `Los Angeles Police Department` / `Ryan Pollie`.

The identity normalizers (`normalize_artist_name`, `normalize_artist_key`,
`normalize_primary_artist_key`) are **pure string transforms** — they can collapse
`Süss`/`SUSS` or strip `feat.`/`Trio`, but they can never know that `Smog` is
`Bill Callahan`. That knowledge only exists in the user's head, so the mechanism
must be a **manually curated lookup**.

Today this is greenfield: no artist alias / merge / canonical mechanism exists
anywhere in the codebase (the only "alias" code is the genre taxonomy's, which is
unrelated).

## Goals

1. Let the user manually declare two-or-more artist names as related, from the GUI.
2. Two relationship types with distinct playlist behavior (see below).
3. Take effect on the **next generation** with **no** `metadata.db` write and **no**
   artifact rebuild — a pure runtime resolution layer, fully reversible.

## Non-goals (v1)

- No relabeling of tracks. Every track keeps its own stored artist string in output;
  only the *identity used internally* changes. (This is why no "canonical name" is
  needed.)
- No nested groups — a given artist name belongs to **at most one** group. An
  artist that is both an alias *and* a sibling-of-another is out of scope for v1
  (future extension).
- No automatic alias suggestion / detection. Purely manual.
- No effect on non-playlist surfaces (Genre Review grouping, artist stats). Scoped to
  playlist generation identity.

## The two link types

The behavior we settled, per type:

| Behavior | **Alias** (`Alex G` / `(Sandy) Alex G`) | **Sibling** (`Smog` / `Bill Callahan`) |
|---|---|---|
| Min-gap / diversity spacing | **merged** — one artist | separate, **+ repulsion at `min_gap`** between siblings |
| Per-artist track budget / pool cap | **merged** (counts as one) | **own budget each** |
| Seed / pier / Fire catalog | **merged** — seed one → pull both | **own catalog each** (no cross-pollination) |
| Dedup, seed-artist bridge-exclusion | **merged** | own |

**Rationale.** An *alias* is a metadata artifact — the split is accidental, so it is
genuinely one catalog and must be erased everywhere. A *sibling* link is real — two
distinct bodies of work by one human — so each project stays fully independent
(own budget, own seed catalog, own self-spacing), with exactly **one** added rule:
a sibling may not be placed within `min_gap` of its sibling. This is the same shape
as how collaborations are treated (a shared participant affects diversity across
tracks without merging the participants into one catalog).

`min_gap` here is the same per-artist self-gap the beam already uses for repeat
spacing (option B, chosen over "adjacency only") — siblings space like one artist
would, without sharing budget or catalog.

## Data model & storage

A group is `{ type, members[] }`. No canonical/primary name.

```yaml
# data/artist_aliases.yaml   (git-tracked)
version: 1
groups:
  - type: alias
    members: ["Alex G", "(Sandy) Alex G"]
  - type: sibling
    members: ["Smog", "Bill Callahan"]
  - type: sibling
    members: ["Mount Eerie", "The Microphones"]
```

**Storage decision: git-tracked YAML** (`data/artist_aliases.yaml`), chosen over a
sidecar-DB table or a `metadata.db` table:

- Human-readable, diffable, version-controlled; travels to satellites via git.
- Writes **nothing** to `metadata.db` or the NPZ artifact.
- Mirrors exactly how `data/layered_genre_taxonomy.yaml` is edited from the Taxonomy
  panel (GUI writes the YAML through a worker command).
- Absent/empty file ⇒ feature is inert. No feature flag — *presence of definitions
  is the feature*, consistent with "activate fixes; never gate behind a default-off
  flag."

Rejected alternatives: the genre sidecar DB (`ai_genre_enrichment.db`) is
semantically the *genre* store; a `metadata.db` table (à la `blacklist_db.py`) is
the closest structural precedent but touches the sacred DB for no benefit here.

### Validation

- Each artist name may appear in at most one group (reject duplicates across groups).
- A group needs ≥2 members.
- `type` ∈ {`alias`, `sibling`}.
- Validation runs on save (worker command) and on load (warn-and-skip a malformed
  group rather than crashing generation — but a *configured-but-unusable* group
  should warn loudly, never silently no-op).

## Resolver design

New module: `src/playlist/artist_aliases.py`.

- `load_artist_link_map(path=DEFAULT) -> ArtistLinkMap` — parses the YAML, builds the
  lookup structures, **cached** (LRU keyed on path+mtime, or an explicit
  cache-bust hook — see cache invalidation below).
- `ArtistLinkMap` exposes:
  - `resolve_alias(normalized_key: str) -> str` — if `normalized_key` belongs to an
    **alias** group, returns the group's synthetic merged key (e.g.
    `alias_group:<slug>`); otherwise returns the input unchanged. Used at every point
    an artist identity or catalog key is computed.
  - `sibling_group_of(normalized_key: str) -> Optional[str]` — if the key belongs to a
    **sibling** group, returns the shared repulsion id; else `None`. Used **only** by
    the beam's spacing constraint.

**Two-key-space handling (important).** The semantic key space
(`normalize_primary_artist_key`) and the structural key space
(`normalize_artist_key`) produce *different* strings for the same display name
(e.g. `(Sandy) Alex G` → `(sandy) alex g` vs `sandy alex g`). The resolver therefore
registers each member under **both** normalized forms, all pointing at the same
group id. `resolve_alias`/`sibling_group_of` then work regardless of which
normalization produced the incoming key.

## Runtime integration

Two independent mechanisms.

### 1. Alias merge — "resolve normalized key → group key"

Applied wherever an artist identity or catalog key is derived. Because both identity
systems funnel through a few key functions, this is a small number of insertion
points, not a scatter:

- **Semantic identity** (beam/bridge diversity, dedup, seed-artist bridge-exclusion):
  `src/playlist/identity_keys.py::normalize_primary_artist_key` /
  `identity_keys_for_index` (the single chokepoint the whole pier/bridge/beam stack
  reads).
- **Structural identity** (candidate-pool per-artist cap, pool collapse):
  `src/playlist/candidate_pool.py` per-artist grouping (`_normalize_artist_key`,
  the `candidates_per_artist` cap) and
  `src/playlist/pier_bridge/pool.py::collapse_pool_by_artist`.
- **Seed / pier catalog gathering** — the seed-artist track resolution so seeding one
  alias pulls the other's tracks as seed/pier material.
- **Fire / popularity path** — popular-seed expansion resolves through the alias map
  so both names' popular tracks are considered (see `project_popular_seeds_cache`).

Because the alias merged-key is applied *inside* the identity computation, the many
downstream `identity_keys_for_index(...).artist_key` call sites inherit it for free.

### 2. Sibling repulsion — targeted beam constraint

A **hard placement constraint** added to the beam's candidate admission
(`src/playlist/pier_bridge/beam.py`, alongside the existing `min_gap` /
`used_artists` checks): block a candidate whose `sibling_group_of(key)` matches that
of any track already placed within `min_gap` of the target slot. It deliberately
does **not** touch budgets, pools, seed selection, or catalogs — siblings keep
everything independent except spacing. This is enforced hard, like min-gap
(diversity is a hard constraint, per Layer-2 principle 11).

### Cache invalidation

The worker caches the loaded map. Editing the YAML from the GUI must bust that cache
so the next generation sees it — the same pattern as
`handle_apply_taxonomy_decisions` busting the `graph_adapter` LRU. A fresh CLI/harness
process loads the current file anyway.

## GUI

A 5th tab — **"Artist Links"** — in the Advanced rail
(`web/src/components/AdvancedPanel.tsx`), beside Genre Review / Taxonomy, following
the established React-panel → `web/src/lib/api.ts` → FastAPI route
(`src/playlist_web/app.py`) → worker-command (`src/playlist_gui/worker.py`) pattern,
and conforming to `docs/UI_UX_DISCIPLINE.md` (touch targets, tokens/contrast, states,
copy).

New component `web/src/components/ArtistLinksPanel.tsx`:

- Lists existing groups, grouped by type; each group removable/editable.
- **New link** form: choose type (**Alias** / **Same artist**), add ≥2 members via
  **typeahead over the real library artist list** (no free-text — prevents typos and
  guarantees the name actually matches library tracks), save.
- **Single-phase**: save writes the YAML and busts the runtime cache immediately.
  No stage→publish/adjudication step — unlike genre, aliasing is deterministic and
  low-risk, so there is nothing to adjudicate.

### Wiring checklist (from the web-gui skill)

- `web/src/lib/types.ts` — request/response types.
- `web/src/lib/api.ts` — `artistLinksList`, `artistLinksSave`, `artistLinksDelete`
  (or a single `artistLinksSet`), plus an artist-typeahead source (reuse the
  library-artists query if one exists; otherwise a small `GET` endpoint).
- `src/playlist_web/schemas.py` — request models.
- `src/playlist_web/app.py` — routes dispatching to the worker as **untracked**
  (fast read/quick-write) commands.
- `src/playlist_gui/worker.py` — `handle_list_artist_links` /
  `handle_save_artist_links` (read + quick YAML write), registered in
  `UNTRACKED_COMMAND_HANDLERS`; every path emits a terminal `done`.
- `tests/fixtures/fake_worker.py` — a branch for the new command(s).
- **Restart `serve_web.py`** after worker edits; **rebuild `web/dist`** after React
  edits.

## Testing

- **Unit** (`src/playlist/artist_aliases.py`): from a fixture YAML —
  - alias member (in either key space) resolves to the shared merged key;
  - sibling member resolves to `None` for `resolve_alias` but a shared id for
    `sibling_group_of`;
  - duplicate-across-groups and <2-member groups are rejected;
  - malformed group warns and is skipped, not fatal.
- **Generation** (via the `gui_fidelity` harness — multi-pier, production-faithful,
  per the `playlist-testing` skill; never hand-built overrides):
  - **Alias:** seed one name; assert the other name's tracks are eligible as
    seed/pier material and the pair counts as **one** artist for both diversity
    spacing and the per-artist budget.
  - **Sibling:** seed a context where both appear; assert both *can* appear, never
    within `min_gap` of each other, and each retains its **own** budget (i.e. the
    joint count can exceed a single-artist cap).
- **Regression:** an empty/absent `artist_aliases.yaml` leaves generation
  bit-identical to today (golden diff).

## Edge cases

- Empty / missing file → no-op (no groups).
- A member name that matches no library track → harmless (resolves nothing); the GUI
  typeahead makes this unlikely.
- A name in both an alias and a sibling group → rejected by validation (v1
  single-group rule).
- Satellite vs canonical: the YAML is git-tracked config like the taxonomy; edited in
  the canonical GUI (port 8770) and committed, then pulled by satellites. No data-write
  guard implications (no `metadata.db`/artifact write).

## Out of scope / future

- Nested alias-within-sibling resolution.
- Extending links to Genre Review album grouping or history/stats.
- Auto-suggested aliases (e.g. from MusicBrainz artist relationships).

### Known v1 limitations (engine shipped 2026-07-09; candidates for a "Plan 1b")

Sibling repulsion is enforced at the **beam admission gate** — the same layer and
strength as the normal per-artist `min_gap`. The following edge paths were reviewed
and consciously deferred; the alias merge and default-path sibling spacing are correct
and tested without them:

- **Post-beam passes are not sibling-aware.** `tail_dp.py` is fully artist-blind, and
  `_greedy_terminal_path` (the guaranteed-fill fallback) enforces artist-key diversity
  but not sibling groups. So a weak-landing tail re-optimization or an infeasible-segment
  fallback *could* place a sibling pair within `min_gap`. This is the **same** best-effort
  limitation the normal artist `min_gap` already has under tail-DP — not a sibling-specific
  regression. Follow-up: thread `sibling_group_of` into `tail_dp` + `_greedy_terminal_path`.
- **Sibling "own budget" can over-exclude under a hard per-artist cap.** The beam seeds
  `used_sibling_groups_init` from the whole `recent_global_artists` list, which conflates
  the positional `min_gap` boundary with artists that permanently exceeded
  `max_non_seed_tracks_per_artist` (`global_non_seed_artist_counts`). When that cap is
  active (only the GUI "hard one-per-artist" mode, default `None`/off), a sibling can be
  excluded from later segments because its sibling hit *its* cap. The only trigger is a
  mode where extra spacing is benign. Follow-up: seed sibling groups from the positional
  boundary window only.
- **Missing test:** the design's "sibling joint count can exceed a single-artist cap"
  (budget-independence) scenario is not yet covered — the shipped sibling test asserts
  spacing (a strong A/B control), but its fixture has one track per sibling. Add with the
  budget-seeding fix above.
