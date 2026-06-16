# Progressive (infinite-scroll) autocomplete — design

- **Date:** 2026-06-15
- **Status:** Approved (brainstorming complete; ready for implementation plan)
- **Scope:** Web GUI artist-mode autocomplete + seed-track search

## Problem

Both library lookups are hard-capped at the first 15 rows, and the user cannot
reach the rest:

- **Artist mode** (`GenerateControls` → `api.autocomplete` → `/api/autocomplete`):
  `SELECT artist_name FROM artists WHERE artist_name LIKE 'q%' ORDER BY artist_name LIMIT 15`.
- **Seed search** (`SeedTrackSection` → `api.searchTracks` → `/api/tracks/search`):
  substring match across title/artist/album, `LIMIT 15`, with an N+1 genre subquery per row.

The dropdowns already scroll (`overflow-auto`); there is simply nothing to scroll
to because only 15 rows are ever fetched. "Look up an artist" returns a truncated
slice with no way to see the others.

### Data grounding (real `data/metadata.db`, read-only)

| | total | broadest 2-char match |
|---|---|---|
| artists (prefix `q%`) | 2,010 | ~174 (`s%`) |
| tracks (substring) | 40,427 | 38,449 (`a`), 36,358 (`s`) |

**Implication / asymmetry:** the full artist list for a prefix is tiny (≤~200 short
strings) and could even be loaded whole. The track substring list can be tens of
thousands, so "load the entire list" is meaningless there — track results must be
delivered in bounded pages with a hard ceiling.

## Decisions (from brainstorming)

1. **Scope:** fix both inputs with one shared mechanism (they share the LIMIT-15 root cause).
2. **Load model:** infinite scroll with **bounded pages** (not load-all). Artists finish
   in a page or two; track search keeps paging on demand and never ships a 35k payload.
3. **Triggers:** **scroll + pause-prefetch.** Scrolling near the bottom loads the next
   page (core). Additionally, if the user stops typing and the query stays stable for
   ~500ms, auto-load **one** extra page so a paused query shows more than the first N
   without scrolling. Further pages require scrolling (so track search never auto-floods).
4. **Architecture:** Approach A — a shared `useInfiniteSearch` hook plus paginated
   endpoints. (Rejected: inline per-component paging, which duplicates the subtle
   debounce/race-guard/scroll/prefetch logic across two components and drifts.)
5. **Frontend testing:** add **vitest** and unit-test the hook's pure logic, plus one
   Playwright e2e for scroll-loads-more.

## Design

### 1. Backend — paginate both endpoints (`src/playlist_web/app.py`)

- Add `offset: int = 0` to `/api/autocomplete` and `/api/tracks/search` (keep `q`, `limit`).
- Detect "more" by fetching `limit + 1` rows: if the extra row is present, `has_more = true`
  and it is trimmed off the returned page.
- **Response shape changes** from a bare list to `{ "items": [...], "has_more": bool }`.
  This is an internal API consumed only by our own client — a clean breaking change.
- **Kill the N+1 in track search:** replace the per-row genre subquery with one batched
  `SELECT track_id, genre FROM track_effective_genres WHERE track_id IN (…) ORDER BY priority`
  for the page's track_ids, grouped in Python (cap 5 each), with the existing `track_genres`
  fallback for tracks with no effective genres. (Audit `[P#1]`; design principle 24.)
- Offset paging is acceptable: artist sets are ≤~200; track paging is soft-capped at ~400
  rendered rows, so the deepest offset scan stays shallow relative to the 36k filtered set.
- Both endpoints keep the read-only SQLite connection (`?mode=ro`) and the existing
  `try/except → empty` graceful fallback.

### 2. Frontend — `useInfiniteSearch` hook (`web/src/lib/useInfiniteSearch.ts`, new)

Generic, UI-agnostic (no DOM access):

```ts
function useInfiniteSearch<T>(opts: {
  fetchPage: (q: string, offset: number, limit: number) => Promise<{ items: T[]; has_more: boolean }>;
  minChars?: number;        // default 2
  firstDebounceMs?: number; // default 200
  prefetchDelayMs?: number; // default 500
  pageSize?: number;        // per-use
  maxItems?: number;        // soft cap; undefined = unbounded
}): {
  query: string;
  setQuery: (q: string) => void;
  items: T[];
  loading: boolean;
  hasMore: boolean;
  loadMore: () => void;     // called by the component's scroll handler
  reset: () => void;        // clear on select / outside-click / close
};
```

Responsibilities:

- **First page:** debounced (`firstDebounceMs`) fetch of offset 0 when `query.length >= minChars`,
  replacing `items`. Below `minChars` → clear.
- **Stale-response guard:** every fetch is tagged with a monotonic request id; a response is
  applied only if it is still the latest for the current query. Prevents out-of-order results
  (the bug-prone part — the reason the logic lives in one tested place).
- **Pause-prefetch:** after page 0 lands for a stable query, start a `prefetchDelayMs` timer;
  if it fires while `hasMore` and the query is unchanged (not reset), call `loadMore()`
  exactly once. Does not loop. (The component hides the dropdown by calling `reset()` on
  select/outside-click, which cancels this timer — the hook stays UI-agnostic.)
- **`loadMore`:** no-op if `loading || !hasMore || (maxItems && items.length >= maxItems)`;
  otherwise fetch `offset = items.length`, append, update `hasMore`.
- **`reset`:** clears query/items, cancels timers, bumps the request id so in-flight responses
  are ignored.
- **Cleanup:** cancel debounce/prefetch timers on query change and unmount.

Scroll detection is **not** in the hook. Each component owns its `<ul>` ref, computes
near-bottom in `onScroll`, and calls `hook.loadMore()`.

### 3. Component integration

- **`GenerateControls`** (artist mode): replace `suggestions` state + its debounced effect
  with `useInfiniteSearch<string>({ fetchPage: api.autocomplete, pageSize: 30 })`. Render
  `items` in the existing `max-h-48 overflow-auto` ul; add `onScroll` → `loadMore`; call
  `reset()` on select. No `maxItems` (artists top out ~200). Active only when `mode === "artist"`.
- **`SeedTrackSection`** (seed search): replace `results` state + its debounced effect with
  `useInfiniteSearch<SeedTrack>({ fetchPage: api.searchTracks, pageSize: 25, maxItems: 400 })`.
  Existing `max-h-60 overflow-auto` ul gets `onScroll` → `loadMore`; `reset()` on add and on
  outside-click. At the cap, show "Showing first 400 — refine your search" instead of paging.
- **`api.ts`:** `autocomplete(q, offset = 0, limit = 30): Promise<{ items: string[]; has_more: boolean }>`
  and `searchTracks(q, offset = 0, limit = 25): Promise<{ items: SeedTrack[]; has_more: boolean }>`.
- **`types.ts`:** a small `Page<T> = { items: T[]; has_more: boolean }` helper (or inline).

### 4. UX & error handling

- Dropdown footer row: "Loading…" while a page is in flight; the refine hint at the cap;
  nothing when `!hasMore`. First-page debounce stays 200ms (typing stays snappy); prefetch at +500ms.
- Fetch failure: the hook catches, keeps existing items, stops paging (graceful degrade,
  matching today's `.catch(() => [])`). No crash, no thrown error to the component.

### 5. Non-goals (explicit)

- **No change to match semantics.** Artist stays prefix-match (`beatles` still won't find
  `The Beatles`); track stays substring. Relevance is a separate effort.
- **No keyboard arrow-key navigation** (both dropdowns are click-only today).
- **No list virtualization.** The 400-row track cap keeps the DOM bounded.

### 6. Testing

- **Backend (pytest):** both endpoints against a small temp-sqlite fixture —
  - correct offset slices and ordering,
  - `has_more` true mid-list / false on the last page,
  - batched genres correct + `track_genres` fallback,
  - empty / sub-`minChars` query → empty page.
- **Frontend (vitest, new tooling):** unit-test the hook's pure logic —
  - stale-response guard ignores an out-of-order earlier response,
  - `loadMore` gating (no-op while loading / past `hasMore` / at `maxItems`),
  - pause-prefetch loads exactly one extra page then stops,
  - `reset` discards in-flight responses.
- **Frontend (Playwright e2e):** type a broad query → assert first page → scroll the dropdown
  → assert additional rows appended. Uses the existing `test:e2e` config.

## Files touched

| File | Change |
|---|---|
| `src/playlist_web/app.py` | `offset` + `has_more` + batched genres on both endpoints |
| `web/src/lib/api.ts` | `autocomplete` / `searchTracks` signatures + return shapes |
| `web/src/lib/types.ts` | `Page<T>` helper |
| `web/src/lib/useInfiniteSearch.ts` | **new** shared hook |
| `web/src/components/GenerateControls.tsx` | consume hook + scroll handler |
| `web/src/components/SeedTrackSection.tsx` | consume hook + scroll handler + soft cap |
| `tests/…` (backend) | new pytest for both endpoints |
| `web/…` (vitest) | vitest setup + hook unit tests |
| `web/package.json`, vite/vitest config | add vitest dev-dep + `test` script |
| Playwright e2e | scroll-loads-more spec |

## Risks / open items

- **Response-shape change** is breaking for any other consumer of these two endpoints; grep
  confirms only `api.ts` consumes them, but verify during implementation.
- **Offset perf** on the track query is bounded by the 400 soft cap; revisit only if a
  deeper cap is ever wanted (keyset pagination would be the upgrade, but it does not fit the
  OR-substring query cleanly).
- **vitest introduction** is the only new tooling; keep config minimal (jsdom env for the hook).
- **Branch placement:** this is a distinct feature from the in-flight Plex-export fix currently
  uncommitted in this worktree. Decide at implementation time whether it gets its own branch/worktree.
