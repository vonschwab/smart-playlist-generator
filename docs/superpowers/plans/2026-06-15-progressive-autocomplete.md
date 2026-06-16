# Progressive (infinite-scroll) Autocomplete — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard 15-row cap on the artist-mode autocomplete and seed-track search with bounded-page infinite scroll (scroll + pause-prefetch), so the user can reach the full list.

**Architecture:** Two paginated read-only endpoints return `{ items, has_more }` (detected via `limit + 1`). One shared, UI-agnostic React hook (`useInfiniteSearch`) owns debounce, page accumulation, stale-response guarding, the pause-prefetch timer, and `loadMore` gating; each component owns its scroll container and calls `loadMore()`. Track search's per-row N+1 genre subquery is replaced by one batched query.

**Tech Stack:** FastAPI + sqlite3 (read-only), React 19 + TypeScript + Vite, vitest + @testing-library/react (new), Playwright e2e.

**Spec:** `docs/superpowers/specs/2026-06-15-progressive-autocomplete-design.md`

**Conventions:**
- Page size: artists 30, tracks 25. Track soft cap `maxItems = 400`.
- Hook defaults: `minChars 2`, `firstDebounceMs 200`, `prefetchDelayMs 500`.
- Run pytest directly with a bounded timeout — never pipe through `tail`/`head` (repo rule).
- After web edits, the served `dist` is stale; the final task rebuilds it.

---

### Task 1: Backend — batched genre helper + paginate `/api/tracks/search`

**Files:**
- Modify: `src/playlist_web/app.py` (add module-level helper near `_resolve_seed_artist_keys` ~line 49; rewrite `track_search` ~lines 250-294)
- Test: `tests/integration/test_search_pagination.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_search_pagination.py`:

```python
import sqlite3
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def _make_db(tmp: Path) -> Path:
    db = tmp / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE artists (artist_name TEXT PRIMARY KEY);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
                             album TEXT, duration_ms INTEGER, file_path TEXT);
        CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT, source TEXT,
                                             priority INTEGER, weight REAL);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        """
    )
    for name in ["Beach House", "Beck", "Beirut", "Bell Orchestre", "Belle & Sebastian"]:
        conn.execute("INSERT INTO artists VALUES (?)", (name,))
    for i in range(5):
        conn.execute(
            "INSERT INTO tracks VALUES (?,?,?,?,?,?)",
            (f"t{i}", f"Song {i}", "Beach House", "Bloom", 200000, f"/m/{i}.flac"),
        )
    # t0: two effective genres (priority order); t1: only a track_genres fallback; t2-4: none
    conn.execute("INSERT INTO track_effective_genres VALUES ('t0','dream pop','x',1,1.0)")
    conn.execute("INSERT INTO track_effective_genres VALUES ('t0','shoegaze','x',2,1.0)")
    conn.execute("INSERT INTO track_genres VALUES ('t1','indie','x',0.9)")
    conn.commit()
    conn.close()
    return db


def _client(monkeypatch, db: Path) -> TestClient:
    import src.playlist_web.app as appmod
    monkeypatch.setattr(appmod, "DB_PATH", db)
    return TestClient(create_app(worker_cmd=FAKE))


def test_track_search_paginates(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            r0 = client.get("/api/tracks/search", params={"q": "beach", "offset": 0, "limit": 3}).json()
            assert [it["track_id"] for it in r0["items"]] == ["t0", "t1", "t2"]
            assert r0["has_more"] is True
            r1 = client.get("/api/tracks/search", params={"q": "beach", "offset": 3, "limit": 3}).json()
            assert [it["track_id"] for it in r1["items"]] == ["t3", "t4"]
            assert r1["has_more"] is False


def test_track_search_batches_genres_with_fallback(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            items = client.get("/api/tracks/search", params={"q": "beach", "limit": 25}).json()["items"]
            by_id = {it["track_id"]: it["genres"] for it in items}
            assert by_id["t0"] == ["dream pop", "shoegaze"]  # effective, priority order
            assert by_id["t1"] == ["indie"]                   # track_genres fallback
            assert by_id["t2"] == []                          # neither


def test_track_search_empty_query(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            assert client.get("/api/tracks/search", params={"q": ""}).json() == {"items": [], "has_more": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_search_pagination.py -q` (timeout 120000)
Expected: FAIL — current endpoint returns a bare list, so `.json()` is a list and `["items"]` raises / assertions fail.

- [ ] **Step 3: Add the batched genre helper**

In `src/playlist_web/app.py`, add at module level (e.g. directly below `_resolve_seed_artist_keys`):

```python
def _genres_for_tracks(
    conn: sqlite3.Connection, track_ids: list[str], per_track: int = 5
) -> dict[str, list[str]]:
    """Batch genre lookup for a page of tracks: effective genres first (priority
    order), then a track_genres fallback for tracks with none. Replaces the old
    per-row N+1 subquery (audit [P#1])."""
    if not track_ids:
        return {}
    out: dict[str, list[str]] = {tid: [] for tid in track_ids}
    ph = ",".join("?" for _ in track_ids)
    for tid, genre in conn.execute(
        f"SELECT track_id, genre FROM track_effective_genres "
        f"WHERE track_id IN ({ph}) ORDER BY priority",
        tuple(track_ids),
    ):
        if len(out[tid]) < per_track:
            out[tid].append(genre)
    missing = [tid for tid, gl in out.items() if not gl]
    if missing:
        ph2 = ",".join("?" for _ in missing)
        for tid, genre in conn.execute(
            f"SELECT track_id, genre FROM track_genres "
            f"WHERE track_id IN ({ph2}) ORDER BY weight DESC",
            tuple(missing),
        ):
            if len(out[tid]) < per_track:
                out[tid].append(genre)
    return out
```

- [ ] **Step 4: Rewrite the `track_search` endpoint**

Replace the entire `@app.get("/api/tracks/search")` handler (current lines ~250-294) with:

```python
    @app.get("/api/tracks/search")
    async def track_search(q: str = "", offset: int = 0, limit: int = 25) -> dict:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": [], "has_more": False}
        pattern = f"%{q.lower()}%"
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    """
                    SELECT t.track_id, t.title, t.artist, t.album, t.duration_ms, t.file_path
                    FROM tracks t
                    WHERE lower(t.title) LIKE ? OR lower(t.artist) LIKE ? OR lower(t.album) LIKE ?
                    ORDER BY t.artist, t.title
                    LIMIT ? OFFSET ?
                    """,
                    (pattern, pattern, pattern, limit + 1, offset),
                ).fetchall()
                has_more = len(rows) > limit
                rows = rows[:limit]
                genres_by_id = _genres_for_tracks(conn, [r[0] for r in rows])
                items = [
                    {
                        "track_id": r[0],
                        "title": r[1] or "Unknown",
                        "artist": r[2] or "Unknown",
                        "album": r[3] or "",
                        "duration_ms": r[4] or 0,
                        "file_path": r[5] or "",
                        "genres": genres_by_id.get(r[0], []),
                    }
                    for r in rows
                ]
                return {"items": items, "has_more": has_more}
            finally:
                conn.close()
        except Exception:
            return {"items": [], "has_more": False}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_search_pagination.py -q` (timeout 120000)
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/app.py tests/integration/test_search_pagination.py
git commit -m "feat(web-search): paginate /api/tracks/search + batch genre lookup"
```

---

### Task 2: Backend — paginate `/api/autocomplete`

**Files:**
- Modify: `src/playlist_web/app.py` (rewrite `autocomplete` ~lines 341-357)
- Test: `tests/integration/test_search_pagination.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_search_pagination.py`:

```python
def test_autocomplete_paginates(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            r0 = client.get("/api/autocomplete", params={"q": "be", "offset": 0, "limit": 2}).json()
            assert r0["items"] == ["Beach House", "Beck"]  # alphabetical
            assert r0["has_more"] is True
            r2 = client.get("/api/autocomplete", params={"q": "be", "offset": 4, "limit": 2}).json()
            assert r2["items"] == ["Belle & Sebastian"]
            assert r2["has_more"] is False


def test_autocomplete_empty_query(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            assert client.get("/api/autocomplete", params={"q": ""}).json() == {"items": [], "has_more": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_search_pagination.py -k autocomplete -q` (timeout 120000)
Expected: FAIL — endpoint returns a bare list.

- [ ] **Step 3: Rewrite the `autocomplete` endpoint**

Replace the entire `@app.get("/api/autocomplete")` handler with:

```python
    @app.get("/api/autocomplete")
    async def autocomplete(q: str = "", offset: int = 0, limit: int = 30) -> dict:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": [], "has_more": False}
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT artist_name FROM artists WHERE artist_name LIKE ? "
                    "ORDER BY artist_name LIMIT ? OFFSET ?",
                    (q + "%", limit + 1, offset),
                ).fetchall()
            finally:
                conn.close()
            has_more = len(rows) > limit
            return {"items": [r[0] for r in rows[:limit]], "has_more": has_more}
        except Exception:
            return {"items": [], "has_more": False}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_search_pagination.py -q` (timeout 120000)
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/app.py tests/integration/test_search_pagination.py
git commit -m "feat(web-search): paginate /api/autocomplete with offset + has_more"
```

---

### Task 3: Frontend — `Page<T>` type + paginated API client

**Files:**
- Modify: `web/src/lib/types.ts` (add `Page<T>`)
- Modify: `web/src/lib/api.ts` (`autocomplete`, `searchTracks` signatures + return shapes)

- [ ] **Step 1: Add the `Page<T>` type**

Append to `web/src/lib/types.ts`:

```ts
export interface Page<T> {
  items: T[];
  has_more: boolean;
}
```

- [ ] **Step 2: Update the API client**

In `web/src/lib/api.ts`, add `Page` to the type import from `./types`, then replace the `autocomplete` and `searchTracks` methods with:

```ts
  async autocomplete(q: string, offset = 0, limit = 30): Promise<Page<string>> {
    const params = new URLSearchParams({ q, offset: String(offset), limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/autocomplete?${params}`));
  },
  async searchTracks(q: string, offset = 0, limit = 25): Promise<Page<SeedTrack>> {
    const params = new URLSearchParams({ q, offset: String(offset), limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/tracks/search?${params}`));
  },
```

- [ ] **Step 3: Typecheck**

Run: `npm --prefix web exec tsc -b` (timeout 120000)
Expected: FAIL — `GenerateControls.tsx` / `SeedTrackSection.tsx` still consume the old return types (a bare array). This is expected; Tasks 6-7 fix the call sites. (If you want a green intermediate commit, do Step 4 now and accept that `npm run build` is red until Task 7.)

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(web-search): Page<T> type + offset/limit on autocomplete & searchTracks"
```

---

### Task 4: Add vitest tooling

**Files:**
- Modify: `web/package.json` (devDeps + `test` script)
- Modify: `web/vite.config.ts` (test block)
- Test: `web/src/lib/smoke.test.ts` (create, then delete after proving the runner)

- [ ] **Step 1: Install dev dependencies**

Run (from repo root):
```bash
npm --prefix web install -D vitest@^3 jsdom @testing-library/react@^16 @testing-library/dom
```
Expected: packages added to `web/package.json` devDependencies.

- [ ] **Step 2: Add the `test` script**

In `web/package.json` `"scripts"`, add:
```json
    "test": "vitest run",
    "test:watch": "vitest",
```

- [ ] **Step 3: Configure vitest in `web/vite.config.ts`**

Change the first import line from `import { defineConfig } from "vite";` to `import { defineConfig } from "vitest/config";` and add a `test` block to the config object:

```ts
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
  },
```

(Keep `plugins`, `resolve`, `server`, `build` as-is.)

- [ ] **Step 4: Prove the runner with a throwaway smoke test**

Create `web/src/lib/smoke.test.ts`:
```ts
import { describe, it, expect } from "vitest";

describe("vitest", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```

Run: `npm --prefix web run test` (timeout 120000)
Expected: PASS (1 test). Then delete the file: `git rm -f web/src/lib/smoke.test.ts` (or remove if untracked).

- [ ] **Step 5: Commit**

```bash
git add web/package.json web/package-lock.json web/vite.config.ts
git commit -m "chore(web): add vitest + jsdom + testing-library"
```

---

### Task 5: `useInfiniteSearch` hook (TDD)

**Files:**
- Create: `web/src/lib/useInfiniteSearch.ts`
- Test: `web/src/lib/useInfiniteSearch.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `web/src/lib/useInfiniteSearch.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useInfiniteSearch } from "./useInfiniteSearch";
import type { Page } from "./types";

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("useInfiniteSearch", () => {
  it("ignores a stale out-of-order response from a previous query", async () => {
    const d1 = deferred<Page<string>>();
    const d2 = deferred<Page<string>>();
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockReturnValueOnce(d1.promise)
      .mockReturnValueOnce(d2.promise);
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 100, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("aa"));
    act(() => { vi.advanceTimersByTime(100); }); // debounce -> fetch1 (query "aa")
    act(() => result.current.setQuery("ab"));
    act(() => { vi.advanceTimersByTime(100); }); // debounce -> fetch2 (query "ab")

    await act(async () => { d2.resolve({ items: ["B1", "B2"], has_more: false }); });
    await act(async () => { d1.resolve({ items: ["A1"], has_more: false }); }); // late + stale

    expect(result.current.items).toEqual(["B1", "B2"]);
  });

  it("loadMore is a no-op while loading and when there is no more", async () => {
    const d1 = deferred<Page<string>>();
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockReturnValueOnce(d1.promise);
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 0, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("aa"));
    act(() => { vi.advanceTimersByTime(0); });
    act(() => result.current.loadMore());          // loading -> ignored
    expect(fetchPage).toHaveBeenCalledTimes(1);

    await act(async () => { d1.resolve({ items: ["A1"], has_more: false }); });
    act(() => result.current.loadMore());          // hasMore false -> ignored
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  it("auto-prefetches exactly one extra page after a pause, then stops", async () => {
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockResolvedValueOnce({ items: ["A1", "A2"], has_more: true })
      .mockResolvedValueOnce({ items: ["A3", "A4"], has_more: true });
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 100, prefetchDelayMs: 500, pageSize: 2 }));

    act(() => result.current.setQuery("aa"));
    await act(async () => { vi.advanceTimersByTime(100); });
    expect(result.current.items).toEqual(["A1", "A2"]);

    await act(async () => { vi.advanceTimersByTime(500); }); // pause-prefetch one page
    expect(result.current.items).toEqual(["A1", "A2", "A3", "A4"]);
    expect(fetchPage).toHaveBeenCalledTimes(2);

    await act(async () => { vi.advanceTimersByTime(2000); }); // no further auto-fetch
    expect(fetchPage).toHaveBeenCalledTimes(2);
  });

  it("reset discards an in-flight response and clears state", async () => {
    const d1 = deferred<Page<string>>();
    const fetchPage = vi.fn<(q: string, o: number, l: number) => Promise<Page<string>>>()
      .mockReturnValueOnce(d1.promise);
    const { result } = renderHook(() =>
      useInfiniteSearch<string>({ fetchPage, firstDebounceMs: 0, prefetchDelayMs: 999999, pageSize: 10 }));

    act(() => result.current.setQuery("aa"));
    act(() => { vi.advanceTimersByTime(0); });
    act(() => result.current.reset());
    await act(async () => { d1.resolve({ items: ["A1"], has_more: true }); });

    expect(result.current.items).toEqual([]);
    expect(result.current.query).toBe("");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm --prefix web run test` (timeout 120000)
Expected: FAIL — `./useInfiniteSearch` module does not exist.

- [ ] **Step 3: Implement the hook**

Create `web/src/lib/useInfiniteSearch.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from "react";
import type { Page } from "./types";

export interface UseInfiniteSearchOptions<T> {
  fetchPage: (q: string, offset: number, limit: number) => Promise<Page<T>>;
  minChars?: number;
  firstDebounceMs?: number;
  prefetchDelayMs?: number;
  pageSize?: number;
  maxItems?: number;
}

export interface UseInfiniteSearchResult<T> {
  query: string;
  setQuery: (q: string) => void;
  items: T[];
  loading: boolean;
  hasMore: boolean;
  loadMore: () => void;
  reset: () => void;
}

export function useInfiniteSearch<T>(opts: UseInfiniteSearchOptions<T>): UseInfiniteSearchResult<T> {
  const {
    fetchPage,
    minChars = 2,
    firstDebounceMs = 200,
    prefetchDelayMs = 500,
    pageSize = 25,
    maxItems,
  } = opts;

  const [query, setQueryState] = useState("");
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(false);

  // Refs mirror state so async callbacks read current values without re-subscribing.
  const reqId = useRef(0);            // monotonic; only the latest fetch may apply
  const itemsRef = useRef<T[]>([]);
  const queryRef = useRef("");
  const hasMoreRef = useRef(false);
  const loadingRef = useRef(false);
  const debounceTimer = useRef<number | undefined>(undefined);
  const prefetchTimer = useRef<number | undefined>(undefined);

  const clearTimers = () => {
    window.clearTimeout(debounceTimer.current);
    window.clearTimeout(prefetchTimer.current);
  };

  const apply = (next: T[], more: boolean) => {
    itemsRef.current = next;
    hasMoreRef.current = more;
    setItems(next);
    setHasMore(more);
  };

  const runFetch = useCallback((q: string, offset: number, isFirst: boolean) => {
    const myId = ++reqId.current;
    loadingRef.current = true;
    setLoading(true);
    fetchPage(q, offset, pageSize)
      .then((page) => {
        if (myId !== reqId.current || q !== queryRef.current) return; // stale
        const merged = isFirst ? page.items : itemsRef.current.concat(page.items);
        apply(merged, page.has_more);
        loadingRef.current = false;
        setLoading(false);
        if (isFirst && page.has_more && (maxItems === undefined || merged.length < maxItems)) {
          window.clearTimeout(prefetchTimer.current);
          prefetchTimer.current = window.setTimeout(() => {
            if (q === queryRef.current && hasMoreRef.current && !loadingRef.current) {
              runFetch(q, itemsRef.current.length, false);
            }
          }, prefetchDelayMs);
        }
      })
      .catch(() => {
        if (myId !== reqId.current) return;
        loadingRef.current = false;
        setLoading(false); // keep existing items; stop paging silently
      });
  }, [fetchPage, pageSize, prefetchDelayMs, maxItems]);

  const setQuery = useCallback((q: string) => {
    queryRef.current = q;
    setQueryState(q);
    clearTimers();
    reqId.current += 1; // invalidate any in-flight fetch for the old query
    if (q.length < minChars) {
      itemsRef.current = [];
      hasMoreRef.current = false;
      loadingRef.current = false;
      setItems([]);
      setHasMore(false);
      setLoading(false);
      return;
    }
    debounceTimer.current = window.setTimeout(() => runFetch(q, 0, true), firstDebounceMs);
  }, [minChars, firstDebounceMs, runFetch]);

  const loadMore = useCallback(() => {
    if (loadingRef.current || !hasMoreRef.current) return;
    if (maxItems !== undefined && itemsRef.current.length >= maxItems) return;
    if (queryRef.current.length < minChars) return;
    runFetch(queryRef.current, itemsRef.current.length, false);
  }, [maxItems, minChars, runFetch]);

  const reset = useCallback(() => {
    reqId.current += 1;
    clearTimers();
    queryRef.current = "";
    itemsRef.current = [];
    hasMoreRef.current = false;
    loadingRef.current = false;
    setQueryState("");
    setItems([]);
    setHasMore(false);
    setLoading(false);
  }, []);

  useEffect(() => () => clearTimers(), []);

  return { query, setQuery, items, loading, hasMore, loadMore, reset };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm --prefix web run test` (timeout 120000)
Expected: PASS (4 tests). If a timer/microtask flake appears, adjust the test's `await act(async …)` wrapping — not the hook.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/useInfiniteSearch.ts web/src/lib/useInfiniteSearch.test.ts
git commit -m "feat(web-search): useInfiniteSearch hook (debounce, prefetch, race-guard)"
```

---

### Task 6: Wire `GenerateControls` (artist mode) to the hook

**Files:**
- Modify: `web/src/components/GenerateControls.tsx`

- [ ] **Step 1: Replace the autocomplete state + effects**

Remove the `suggestions` state and `timer` ref (current lines ~89-90). Add the hook and a selection-suppression ref:

```tsx
  // Autocomplete (artist mode) — bounded-page infinite scroll
  const artistSearch = useInfiniteSearch<string>({ fetchPage: api.autocomplete, pageSize: 30 });
  const suppressSearch = useRef(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
```

Add the import at the top of the file: `import { useInfiniteSearch } from "../lib/useInfiniteSearch";`

Replace the outside-click effect body (`setSuggestions([])` at ~line 121) with `artistSearch.reset();`.

Replace the debounced effect (current lines ~128-135) with a sync effect that drives the hook from `seed`, skipping the cycle right after a selection:

```tsx
  // Drive the artist autocomplete from the seed input. Skip one cycle after a
  // selection so picking a name doesn't immediately re-open the dropdown.
  useEffect(() => {
    if (mode !== "artist") { artistSearch.reset(); return; }
    if (suppressSearch.current) { suppressSearch.current = false; return; }
    artistSearch.setQuery(seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, mode]);
```

- [ ] **Step 2: Update the mode selector reset**

In the mode `<select onChange>` (current line ~173), replace `setSuggestions([])` with `artistSearch.reset()`:

```tsx
            onChange={(e) => { onModeChange(e.target.value as Mode); setSeed(""); artistSearch.reset(); }}
```

- [ ] **Step 3: Replace the dropdown JSX**

Replace the `{suggestions.length > 0 && ( … )}` block (current lines ~196-208) with:

```tsx
              {mode === "artist" && artistSearch.items.length > 0 && (
                <ul
                  ref={listRef}
                  onScroll={() => {
                    const el = listRef.current;
                    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) artistSearch.loadMore();
                  }}
                  className="absolute z-10 mt-1 w-full bg-[#16181d] border border-[#23262d] rounded shadow-xl max-h-48 overflow-auto"
                >
                  {artistSearch.items.map((s) => (
                    <li
                      key={s}
                      onClick={() => { suppressSearch.current = true; setSeed(s); artistSearch.reset(); }}
                      className="px-2.5 py-1.5 text-[11px] text-[#e6e9ec] hover:bg-[#1e2229] cursor-pointer"
                    >
                      {s}
                    </li>
                  ))}
                  {(artistSearch.loading || artistSearch.hasMore) && (
                    <li className="px-2.5 py-1.5 text-[10px] text-[#5b6470]">
                      {artistSearch.loading ? "Loading…" : "Scroll for more"}
                    </li>
                  )}
                </ul>
              )}
```

- [ ] **Step 4: Typecheck**

Run: `npm --prefix web exec tsc -b` (timeout 120000)
Expected: `GenerateControls.tsx` is now type-clean; `SeedTrackSection.tsx` still red (fixed in Task 7).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/GenerateControls.tsx
git commit -m "feat(web-search): artist autocomplete uses infinite-scroll hook"
```

---

### Task 7: Wire `SeedTrackSection` (seed search) to the hook

**Files:**
- Modify: `web/src/components/SeedTrackSection.tsx`

- [ ] **Step 1: Replace query/results state with the hook**

Add import: `import { useInfiniteSearch } from "../lib/useInfiniteSearch";`

Remove the `q`, `results`, and `timer` state/refs and the debounced search effect (current lines ~33-35 and ~63-69). Add:

```tsx
  const search = useInfiniteSearch<SeedTrack>({
    fetchPage: api.searchTracks,
    pageSize: 25,
    maxItems: 400,
  });
  const listRef = useRef<HTMLUListElement>(null);
```

(Keep `dropdownRef`.)

- [ ] **Step 2: Update outside-click + addTrack to use the hook**

In the outside-click effect, replace `setResults([])` with `search.reset();`.

Replace `addTrack`:

```tsx
  function addTrack(track: SeedTrack) {
    if (tracks.some((t) => t.track_id === track.track_id)) return;
    onAdd(track);
    search.reset();
  }
```

- [ ] **Step 3: Update the input + dropdown JSX**

Replace the input's `value`/`onChange` (current lines ~98-99):

```tsx
            value={search.query}
            onChange={(e) => search.setQuery(e.target.value)}
```

Replace the `{results.length > 0 && ( … )}` dropdown block (current lines ~103-124) with:

```tsx
          {search.items.length > 0 && (
            <ul
              ref={listRef}
              onScroll={() => {
                const el = listRef.current;
                if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) search.loadMore();
              }}
              className="absolute z-20 left-3 right-3 mt-1 bg-[#16181d] border border-[#23262d] rounded shadow-xl max-h-60 overflow-auto"
            >
              {search.items.map((r) => {
                const already = tracks.some((t) => t.track_id === r.track_id);
                return (
                  <li
                    key={r.track_id}
                    onClick={() => !already && addTrack(r)}
                    className={`flex items-center gap-2 px-3 py-2 text-xs border-b border-[#1e2128] last:border-b-0 ${
                      already ? "opacity-40 cursor-default" : "hover:bg-[#1e2229] cursor-pointer"
                    }`}
                  >
                    <span className="text-[#e6e9ec] font-medium truncate min-w-0 flex-1">{r.title}</span>
                    <span className="text-[#5b6470] truncate shrink-0">{r.artist}</span>
                    {r.genres.slice(0, 2).map((g) => (
                      <span key={g} className={CHIP}>{g}</span>
                    ))}
                  </li>
                );
              })}
              {search.items.length >= 400 && search.hasMore ? (
                <li className="px-3 py-2 text-[10px] text-[#5b6470]">Showing first 400 — refine your search</li>
              ) : (search.loading || search.hasMore) ? (
                <li className="px-3 py-2 text-[10px] text-[#5b6470]">{search.loading ? "Loading…" : "Scroll for more"}</li>
              ) : null}
            </ul>
          )}
```

- [ ] **Step 4: Full frontend build (typecheck + bundle)**

Run: `npm --prefix web run build` (timeout 300000)
Expected: PASS (`tsc -b` clean, `vite build` writes `dist/`). All call sites now match the paginated API.

- [ ] **Step 5: Run vitest + backend tests (no regressions)**

Run: `npm --prefix web run test` (timeout 120000)
Run: `python -m pytest tests/integration/test_search_pagination.py tests/unit/test_web_schemas.py -q` (timeout 120000)
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/SeedTrackSection.tsx
git commit -m "feat(web-search): seed-track search uses infinite-scroll hook + soft cap"
```

---

### Task 8: Playwright e2e — scroll loads the next page

**Files:**
- Create: `web/tests/autocomplete.spec.ts`

The worktree has no `data/metadata.db`, so this spec mocks `/api/tracks/search` at the network layer (Playwright route interception) — it verifies the component wiring (scroll → `loadMore` → appended rows), independent of the DB.

- [ ] **Step 1: Write the e2e spec**

Create `web/tests/autocomplete.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("seed search appends a second page when the dropdown is scrolled", async ({ page }) => {
  await page.route("**/api/tracks/search**", async (route) => {
    const url = new URL(route.request().url());
    const offset = Number(url.searchParams.get("offset") ?? "0");
    const items = Array.from({ length: 25 }, (_, k) => {
      const i = offset + k;
      return {
        track_id: `t${i}`,
        title: `Song ${i}`,
        artist: "Beach House",
        album: "Bloom",
        duration_ms: 200000,
        file_path: `/m/${i}.flac`,
        genres: ["dream pop"],
      };
    });
    await route.fulfill({ json: { items, has_more: offset === 0 } }); // one extra page
  });

  await page.goto("/");
  await page.getByTestId("seed-search-input").fill("beach");

  await expect(page.getByText("Song 0", { exact: true })).toBeVisible();
  await expect(page.getByText("Song 24", { exact: true })).toBeVisible();
  await expect(page.getByText("Song 25", { exact: true })).toHaveCount(0);

  const list = page.locator("ul.overflow-auto").first();
  await list.evaluate((el) => { el.scrollTop = el.scrollHeight; });

  await expect(page.getByText("Song 40", { exact: true })).toBeVisible();
});
```

- [ ] **Step 2: Run the e2e**

Run: `npm --prefix web run test:e2e -- autocomplete` (timeout 300000)
Expected: PASS. (Playwright builds the app and starts `serve_web.py` on port 8771 per `playwright.config.ts`.) If the first-page debounce makes the first assertion race, add `await page.waitForTimeout(300)` after `.fill`.

- [ ] **Step 3: Commit**

```bash
git add web/tests/autocomplete.spec.ts
git commit -m "test(web-search): e2e scroll-loads-more for seed search"
```

---

### Task 9: Final verification + rebuild served dist

**Files:** none (verification only)

- [ ] **Step 1: Full backend suite (bounded, not slow)**

Run: `python -m pytest -q -m "not slow"` (timeout 600000)
Expected: PASS. Quote the real pass/fail counts.

- [ ] **Step 2: Frontend build + unit tests**

Run: `npm --prefix web run build` (timeout 300000) — expected PASS.
Run: `npm --prefix web run test` (timeout 120000) — expected PASS.

- [ ] **Step 3: Manual exercise (real path)**

Restart the worker/server: `python tools/serve_web.py --no-browser` (note: needs `data/metadata.db`; run from a checkout that has it, or the main checkout, since the worktree lacks the DB).
- Artist mode: type a broad prefix (e.g. `be`), confirm >15 results, scroll to load more, confirm pause loads one extra page.
- Seeds: type a broad query, confirm paging on scroll and the "refine your search" cap at 400.

- [ ] **Step 4: Confirm no stale-state traps**

Per the web-gui rules: `serve_web.py` restarted after the worker/back-end edits, and `web/dist` rebuilt after the front-end edits (Step 2). Verify the dropdown behavior in a fresh browser load.

---

## Self-Review

**Spec coverage:**
- Paginate both endpoints + `has_more` → Tasks 1, 2. ✓
- Batch the N+1 genre query → Task 1 (`_genres_for_tracks`). ✓
- `Page<T>` + API client → Task 3. ✓
- Shared `useInfiniteSearch` (debounce, accumulation, stale-guard, pause-prefetch, loadMore gating, reset) → Task 5. ✓
- Components consume hook + scroll handler; seed soft cap + refine hint → Tasks 6, 7. ✓
- Scroll + pause-prefetch triggers → hook (Task 5) + scroll handlers (Tasks 6-7). ✓
- vitest tooling + hook unit tests + e2e → Tasks 4, 5, 8. ✓
- Backend pytest for both endpoints → Tasks 1, 2. ✓
- Non-goals (no match-semantics change, no keyboard nav, no virtualization) → respected (prefix/substring queries unchanged; no arrow-key code; 400-row cap instead of virtualization). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command + expected result.

**Type consistency:** `Page<T> = { items: T[]; has_more: boolean }` used identically in `types.ts`, `api.ts`, the hook, and tests. `useInfiniteSearch` returns `{ query, setQuery, items, loading, hasMore, loadMore, reset }` — consumed exactly as defined in Tasks 6-7. `fetchPage(q, offset, limit)` matches `api.autocomplete` / `api.searchTracks` signatures. Page sizes (artist 30, track 25) and `maxItems 400` consistent between spec, hook calls, and the soft-cap JSX.

**Known intermediate-red points (intentional):** after Task 3 the build is red until Task 7 wires the last call site; each is called out in-task.
