# Genre chips: graph-canonical, granularity-ordered display

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Surfaces:** Finished playlist table, Staged seed list (NOT the seed-search dropdown)

## Problem

The genre chips in the web UI don't show the "real" genres in a useful order. Today:

- **Finished playlist table** (`web/src/components/TrackTable.tsx`) renders `genres.slice(0, 2)` — the
  enriched genres resolved by the worker (`_resolve_track_genres`, sidecar → metadata fallback), in
  stored signature order.
- **Staged seed list** (`web/src/components/SeedTrackSection.tsx`) renders `t.genres.slice(0, 4)` — and
  those genres are *reused verbatim* from the seed-search dropdown, which comes from the FastAPI SQL
  endpoint `GET /api/tracks/search` (`track_effective_genres` by priority, capped 5).

So the two surfaces draw from different vocabularies, the order is arbitrary (stored/priority order, not
granularity), and noise tags (artist names, retail buckets, malformed) can appear. The user wants to see
the **canonical genres, ordered from most-specific (sub-genre) to broadest**, with noise removed.

The SP3a layered taxonomy graph (`data/layered_genre_taxonomy.yaml`, live via `src/genre/graph_adapter.py`)
provides exactly the needed signal: every node has a `kind` ladder
(`family → umbrella → genre → subgenre → microgenre`) and a numeric `specificity_score`, and
`canonicalize_tag()` maps any raw tag onto a node (or rejects it).

## Goal

For the finished playlist table and the staged seed list, render genre chips that are:

1. **Graph-canonical** — each raw tag canonicalized to its taxonomy node name.
2. **Granularity-ordered** — sorted by `specificity_score` descending (sub-genre → broad).
3. **De-noised** — tags that don't resolve to an active node are dropped.
4. **Capped at 6** with a `+N` overflow pill (hover tooltip lists the remainder).

Out of scope: the seed-search dropdown (per-keystroke endpoint stays untouched), any change to the
genres used for *scoring/steering* (that already reads the graph), any DB writes.

## Architecture

Three pieces, with one shared pure unit.

### 1. Pure ordering unit — `src/genre/granularity.py` (new)

```python
def order_genres_by_granularity(
    raw_tags: list[str],
    adapter: GenreGraphAdapter | None = None,
) -> list[str]:
    """Canonicalize raw genre tags through the SP3a taxonomy and return canonical
    node names ordered most-specific first (sub-genre → broad).

    - canonicalize each tag via adapter.canonicalize_tag(); skip tags that don't
      resolve to an active node (drops noise/rejects/uncovered)
    - look up adapter.node(canonical).specificity_score
    - de-dup by canonical name, keeping the first (most-specific-context) occurrence
    - stable-sort by specificity_score descending; ties preserve input order
    - adapter defaults to the cached load_graph_adapter()
    """
```

Properties:
- **Pure / no I/O** beyond the process-cached adapter. Input is a list of strings, output is a list of
  strings. Independently unit-testable.
- Returns `[]` when nothing canonicalizes. Callers apply the safety fallback (below), not this function.

### 2. Backend wiring — input resolution per surface, ordering shared

**Input policy (both surfaces, consistent):** enriched genres for the release
(`EnrichedGenreResolver.get_enriched_genres(artist, album)`) if present, else metadata effective genres.
This matches the established enriched-as-authority principle and makes the staged list and playlist agree.

- **Playlist table** — `src/playlist_gui/worker.py`, `handle_generate`:
  after the existing `genres = _resolve_track_genres(track, sidecar_db_path=..., fallback=_raw_genres)`,
  apply `order_genres_by_granularity(genres)` before placing it in the formatted track dict
  (currently `"genres": genres` at the `formatted_tracks.append({...})` site). Apply the safety fallback.
  The worker already has graph access; the adapter is process-cached.

- **Staged seed list** — new endpoint `POST /api/tracks/genres` in `src/playlist_web/app.py`:
  - Request: `{ "track_ids": ["...", ...] }`
  - Response: `{ "<track_id>": ["<canonical genre>", ...], ... }` (each list pre-ordered, capped server-side
    is NOT required — frontend caps; send the full ordered list so `+N` can reveal the rest)
  - Per track id: read `(artist, album)` + raw genres from `metadata.db` (read-only, same pattern as the
    search endpoint), resolve input (enriched → metadata), then `order_genres_by_granularity(...)`.
  - Runs in the FastAPI process using the cached `load_graph_adapter()`. Called only when the staged seed
    **set changes**, not per keystroke — so it does not regress search latency.

**Safety fallback (both call sites):** if `order_genres_by_granularity(raw)` returns `[]` but `raw` was
non-empty, fall back to the raw tags unordered. A track never regresses to blank chips because the
taxonomy didn't cover any of its tags.

### 3. Frontend

- **`web/src/components/TrackTable.tsx`**: replace `genres.slice(0, 2)` with a shared chip renderer that
  shows up to 6 chips + a `+N` pill. Genres arrive already ordered; the frontend only slices/renders.
- **`web/src/components/SeedTrackSection.tsx`**: the *staged seed* branch (currently `t.genres.slice(0, 4)`)
  uses the same chip renderer (6 + `+N`). The *search dropdown* branch (`r.genres.slice(0, 2)`) is left
  unchanged.
- **Staged-seed data fetch**: when the staged seed set changes, call `POST /api/tracks/genres` with the
  staged track ids and use the returned canonical genres for those rows (overriding the metadata genres
  carried from the dropdown). Keep the metadata genres as the pre-fetch placeholder so chips don't flicker
  empty.
- **`+N` overflow**: a small pill showing `+N`; hovering shows a tooltip (shadcn `Tooltip` or native
  `title`) listing the remaining canonical genres. No click-popover required.
- **Types** (`web/src/lib/types.ts`): `genres: string[]` is unchanged (already ordered by the backend).
  Add the new API method to `web/src/lib/api.ts`.

## Data flow

```
Playlist:  worker handle_generate
             _resolve_track_genres (enriched → metadata)   [unchanged]
               → order_genres_by_granularity               [new]
               → safety fallback if empty                  [new]
               → track["genres"]  → NDJSON → TrackTable (6 + +N)

Staged:    SeedTrackSection (seed set changes)
             → POST /api/tracks/genres {track_ids}
                 → per id: resolve (enriched → metadata) → order_genres_by_granularity → fallback
             → returned canonical genres replace dropdown-carried genres on staged rows (6 + +N)

Dropdown:  GET /api/tracks/search  [UNCHANGED — metadata genres, slice(0,2)]
```

## Error handling / edge cases

- **No resolvable genres at all** → empty chip list (current behavior; no regression).
- **All tags uncovered by taxonomy** → safety fallback shows raw tags unordered (never blank when raw existed).
- **Canonicalization collapses variants** (e.g. `dream-pop`, `dreampop` → `dreampop`) → de-duped, single chip.
- **Adapter load failure** (missing/corrupt taxonomy YAML) → `order_genres_by_granularity` catches the load
  error, logs it once, and returns the raw tags unchanged (degraded but functional). It never raises into the
  generation or request path — consistent with the project's graceful-fallback rule.
- **`POST /api/tracks/genres` with unknown/empty ids** → returns `{}` / omits unknown ids; frontend keeps
  placeholders.
- **Perf**: adapter is process-cached (`lru_cache`); ordering is O(n log n) over a handful of tags per track —
  negligible. The staged endpoint is called on seed-set change only.

## Testing

- **Unit (`order_genres_by_granularity`)**: ordering by specificity (sub → broad); uncovered/reject tags
  dropped; variant de-dup; ties preserve input order; empty input → empty; adapter-failure degrade path.
  Use a small fixture taxonomy or the real adapter with known nodes.
- **Backend endpoint**: `POST /api/tracks/genres` returns ordered canonical genres for known track ids;
  enriched-vs-metadata input precedence; safety fallback for an all-uncovered track; unknown id handling.
- **Worker path**: a generated playlist's track dict carries ordered canonical genres (extend an existing
  worker/generation test rather than adding a bespoke harness — see the playlist-testing skill's
  faithful-test rule).
- **Frontend**: chip renderer shows ≤6 + `+N`; `+N` tooltip lists the remainder; staged list fetches and
  swaps in canonical genres; dropdown chips unchanged. (Vitest/Playwright per existing web test setup;
  remember the stale-`dist` rebuild trap from the web-gui skill.)

## Decisions (locked during brainstorming)

- Genre source shown = **graph-canonical names** (canonicalize the track's resolved genres).
- Surfaces = **playlist table + staged seed list** only; dropdown untouched.
- Chip count = **6 cap + `+N` overflow** with hover tooltip.
- Input policy = **enriched → metadata fallback**, consistent across both surfaces.
- Uncovered tags = **dropped**; **zero-result safety fallback** shows raw tags so chips never go blank.
- Ordering signal = **`specificity_score`** descending.
