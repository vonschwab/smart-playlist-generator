# Genre Chips: Graph-Canonical, Granularity-Ordered Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Genre chips in the finished-playlist table and the staged seed list show graph-canonical genre names ordered most-specific → broadest (6 cap + `+N` overflow), instead of arbitrary-order raw tags.

**Architecture:** One new pure module (`src/genre/granularity.py`) canonicalizes raw tags through the cached SP3a taxonomy adapter and sorts by `specificity_score` descending. The worker wraps its existing enriched→metadata genre resolution with it (playlist table); a new `POST /api/tracks/genres` FastAPI endpoint does the same resolution+ordering for staged seeds (called on seed-set change only — the per-keystroke search dropdown is untouched). Frontend gets a shared `GenreChips` component (6 + `+N` with hover tooltip).

**Tech Stack:** Python 3.11 (FastAPI, pydantic, sqlite3, pytest), React 19 + TypeScript (Vite, Playwright e2e).

**Spec:** `docs/superpowers/specs/2026-06-11-genre-chips-granularity-design.md`

---

## Reference: taxonomy facts the tests rely on

`src/genre/graph_adapter.py` — `GenreGraphAdapter`:
- `canonicalize_tag(raw) -> CanonicalizationResult` with `resolution ∈ {"canonical","alias","facet","rejected","unknown"}`; `result.node` is a `GraphNode(name, kind, status, role, specificity_score, is_broad)` set only for canonical/alias.
- `load_graph_adapter()` (no args) loads `data/layered_genre_taxonomy.yaml` through an `lru_cache` — cheap after first call. `load_graph_adapter(path)` loads a fixture taxonomy (used by `tests/unit/test_genre_graph_adapter.py`).

Real production taxonomy values (verified 2026-06-11, taxonomy frozen at `0.12.1-group1-pass9-edge-upgrade` — stable):

| raw tag | resolution | canonical | specificity |
|---|---|---|---|
| `slowcore` | canonical | slowcore | 0.88 |
| `shoegaze` | canonical | shoegaze | 0.86 |
| `dream pop` | canonical | dream pop | 0.78 |
| `indie rock` | canonical | indie rock | 0.55 |
| `rock` | canonical | rock | 0.05 |
| `seen live` | rejected | — | — |
| `lo-fi` | facet | — | — |
| `totally-not-a-genre` | unknown | — | — |

---

### Task 1: Pure ordering module `src/genre/granularity.py`

**Files:**
- Create: `src/genre/granularity.py`
- Create: `tests/unit/test_genre_granularity.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_genre_granularity.py`. The fixture taxonomy copies the pattern from `tests/unit/test_genre_graph_adapter.py` (same record schema):

```python
"""Tests for src/genre/granularity.py — granularity ordering of genre tags.

Display-path helper: canonicalize raw tags through the SP3a taxonomy and order
most-specific first. Pinned against a fixture taxonomy so production growth
doesn't break the tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.genre.graph_adapter import GenreGraphAdapter, load_graph_adapter

FIXTURE = {
    "taxonomy_version": "granularity-test-0.1",
    "records": [
        {"name": "rock", "kind": "family", "status": "active", "specificity_score": 0.05},
        {"name": "indie rock", "kind": "genre", "status": "active", "specificity_score": 0.55},
        {"name": "shoegaze", "kind": "genre", "status": "active", "specificity_score": 0.86},
        {"name": "dream pop", "kind": "genre", "status": "active", "specificity_score": 0.78},
        # Same specificity as dream pop — tie-order test.
        {"name": "noise pop", "kind": "genre", "status": "active", "specificity_score": 0.78},
        # Review-status node: must NOT appear in display output.
        {"name": "chillwave", "kind": "genre", "status": "review", "specificity_score": 0.70},
        {"name": "nu gaze", "kind": "alias", "status": "alias_only", "canonical_target": "shoegaze"},
        {"name": "lo-fi", "kind": "facet", "facet_type": "production", "status": "active"},
        {"name": "seen live", "kind": "reject", "status": "rejected", "reject_reason": "source_noise"},
    ],
}


@pytest.fixture()
def adapter(tmp_path: Path) -> GenreGraphAdapter:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(FIXTURE, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_graph_adapter(path)


def test_orders_most_specific_first(adapter):
    from src.genre.granularity import order_genres_by_granularity

    result = order_genres_by_granularity(["rock", "shoegaze", "indie rock"], adapter=adapter)
    assert result == ["shoegaze", "indie rock", "rock"]


def test_drops_noise_facets_unknown_and_review(adapter):
    from src.genre.granularity import order_genres_by_granularity

    result = order_genres_by_granularity(
        ["seen live", "lo-fi", "no-such-genre", "chillwave", "shoegaze"], adapter=adapter
    )
    assert result == ["shoegaze"]


def test_alias_resolves_and_dedups_with_canonical(adapter):
    from src.genre.granularity import order_genres_by_granularity

    # "nu gaze" is an alias of shoegaze; the canonical name appears once.
    result = order_genres_by_granularity(["nu gaze", "shoegaze", "rock"], adapter=adapter)
    assert result == ["shoegaze", "rock"]


def test_variant_spellings_dedup(adapter):
    from src.genre.granularity import order_genres_by_granularity

    result = order_genres_by_granularity(["Dream-Pop", "dream pop"], adapter=adapter)
    assert result == ["dream pop"]


def test_ties_preserve_input_order(adapter):
    from src.genre.granularity import order_genres_by_granularity

    # dream pop and noise pop share specificity 0.78 — input order wins.
    assert order_genres_by_granularity(["noise pop", "dream pop"], adapter=adapter) == [
        "noise pop", "dream pop",
    ]
    assert order_genres_by_granularity(["dream pop", "noise pop"], adapter=adapter) == [
        "dream pop", "noise pop",
    ]


def test_empty_and_blank_input(adapter):
    from src.genre.granularity import order_genres_by_granularity

    assert order_genres_by_granularity([], adapter=adapter) == []
    assert order_genres_by_granularity(["", "  "], adapter=adapter) == []


def test_nothing_canonicalizes_returns_empty(adapter):
    from src.genre.granularity import order_genres_by_granularity

    assert order_genres_by_granularity(["seen live", "no-such-genre"], adapter=adapter) == []


def test_display_fallback_shows_raw_when_nothing_canonicalizes(adapter):
    from src.genre.granularity import order_genres_for_display

    raw = ["seen live", "no-such-genre"]
    assert order_genres_for_display(raw, adapter=adapter) == raw


def test_display_orders_when_canonicalization_succeeds(adapter):
    from src.genre.granularity import order_genres_for_display

    assert order_genres_for_display(["rock", "shoegaze"], adapter=adapter) == ["shoegaze", "rock"]


def test_adapter_load_failure_degrades_to_raw(monkeypatch):
    """Taxonomy unavailable -> raw tags pass through unchanged, no raise."""
    import src.genre.granularity as granularity

    def _boom():
        raise FileNotFoundError("taxonomy missing")

    monkeypatch.setattr(granularity, "load_graph_adapter", _boom)
    raw = ["shoegaze", "rock"]
    assert granularity.order_genres_by_granularity(raw) == raw
    assert granularity.order_genres_for_display(raw) == raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_genre_granularity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genre.granularity'` (every test errors at the deferred import).

- [ ] **Step 3: Write the implementation**

Create `src/genre/granularity.py`:

```python
"""Granularity ordering of genre tags via the SP3a layered taxonomy.

Display-path helper (spec: docs/superpowers/specs/2026-06-11-genre-chips-granularity-design.md):
canonicalize raw genre tags through the taxonomy graph and order the canonical
names most-specific first (sub-genre -> broad). Used by the GUI worker
(playlist table) and the web staged-seed endpoint. Read-only; never raises
into a generation or request path.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.genre.graph_adapter import GenreGraphAdapter, load_graph_adapter

logger = logging.getLogger(__name__)

# Log the degraded-taxonomy path once per process, not once per track.
_ADAPTER_WARNED = False


def order_genres_by_granularity(
    raw_tags: Sequence[str],
    adapter: Optional[GenreGraphAdapter] = None,
) -> list[str]:
    """Canonical node names for raw tags, most-specific first.

    - canonicalizes each tag (aliases resolve to their canonical genre)
    - drops tags that don't resolve to an ACTIVE canonical node
      (facets, rejects, unknown, review/deprecated nodes)
    - de-dups by canonical name (first occurrence wins)
    - stable-sorts by specificity_score descending; ties keep input order
    - returns [] when nothing canonicalizes
    - never raises: on taxonomy load failure, returns the raw tags unchanged
      (degraded but functional; logged once per process)
    """
    global _ADAPTER_WARNED
    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    if not tags:
        return []
    try:
        if adapter is None:
            adapter = load_graph_adapter()
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for tag in tags:
            result = adapter.canonicalize_tag(tag)
            node = result.node
            if result.resolution not in ("canonical", "alias") or node is None:
                continue
            if node.status != "active":
                continue
            if node.name in seen:
                continue
            seen.add(node.name)
            ranked.append((float(node.specificity_score), node.name))
        # list.sort is stable: equal scores keep input order.
        ranked.sort(key=lambda pair: -pair[0])
        return [name for _, name in ranked]
    except Exception:
        if not _ADAPTER_WARNED:
            logger.warning(
                "Taxonomy unavailable for genre display ordering; showing raw tags",
                exc_info=True,
            )
            _ADAPTER_WARNED = True
        return tags


def order_genres_for_display(
    raw_tags: Sequence[str],
    adapter: Optional[GenreGraphAdapter] = None,
) -> list[str]:
    """order_genres_by_granularity with the display safety fallback.

    When raw tags exist but none canonicalize, return the raw tags unordered —
    a track never regresses to blank chips because the taxonomy didn't cover it.
    """
    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    ordered = order_genres_by_granularity(tags, adapter=adapter)
    if not ordered and tags:
        return tags
    return ordered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_genre_granularity.py -v`
Expected: 10 passed.

- [ ] **Step 5: Lint and type-check the new module**

Run: `ruff check src/genre/granularity.py tests/unit/test_genre_granularity.py && mypy src/genre/granularity.py`
Expected: no errors. (Don't add the module to `[[tool.mypy.overrides]]` — it must type cleanly.)

- [ ] **Step 6: Commit**

```bash
git add src/genre/granularity.py tests/unit/test_genre_granularity.py
git commit -m "feat(genre): granularity ordering helper over the SP3a taxonomy"
```

---

### Task 2: Worker wiring — playlist-table genres ordered

**Files:**
- Modify: `src/playlist_gui/worker.py` (helper next to `_resolve_track_genres` at ~line 606; call site in `handle_generate` at ~line 1323)
- Test: `tests/unit/test_playlist_gui_genre_resolver.py` (extend — existing file tests `_resolve_track_genres` with a tmp sidecar via `SidecarStore`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_playlist_gui_genre_resolver.py`. NOTE: these use the REAL production taxonomy via the cached default adapter (values pinned in the Reference table above). `_seed_sidecar` already exists in this file and inserts release `duster::stratosphere`; add a second seeding helper for ordering-specific genres:

```python
def _seed_sidecar_for_ordering(sidecar_path: Path) -> None:
    """Release whose enriched genres are stored broad-first with one noise tag."""
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "acetone::york blvd",
                "acetone",
                "york blvd",
                None,
                json.dumps({"genres": ["rock", "slowcore", "seen live"], "sources": []}),
                "2026-06-11T00:00:00",
            ),
        )
        conn.commit()


def test_resolve_display_genres_orders_sub_to_broad(tmp_path):
    """Enriched genres come back canonicalized and most-specific first; noise dropped.

    Uses the real taxonomy: slowcore (0.88) > rock (0.05); 'seen live' rejected.
    """
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar_for_ordering(sidecar_path)

    track = {"artist": "Acetone", "album": "York Blvd", "rating_key": "t1"}
    result = worker._resolve_display_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=lambda: [],
    )
    assert result == ["slowcore", "rock"]


def test_resolve_display_genres_raw_fallback_when_uncovered(tmp_path):
    """All-uncovered genres fall back to raw unordered — chips never go blank."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "nonexistent.db"
    track = {"artist": "Unknown", "album": "Album", "rating_key": "t1"}
    raw = ["totally-not-a-genre", "also-not-one"]
    result = worker._resolve_display_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=lambda: list(raw),
    )
    assert result == raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_playlist_gui_genre_resolver.py -v`
Expected: the 3 existing tests PASS; the 2 new tests FAIL with `AttributeError: module 'src.playlist_gui.worker' has no attribute '_resolve_display_genres'`.

- [ ] **Step 3: Add the worker helper**

In `src/playlist_gui/worker.py`, directly below `_resolve_track_genres` (after its `return` at ~line 626), add:

```python
def _resolve_display_genres(
    track: Dict[str, Any],
    *,
    sidecar_db_path: str,
    fallback,
) -> List[str]:
    """Display genres for a track: resolved (enriched -> fallback), then
    canonicalized through the taxonomy and ordered most-specific first.

    order_genres_for_display applies the raw-tags safety fallback and never
    raises (degrades to raw tags if the taxonomy is unavailable).
    """
    from src.genre.granularity import order_genres_for_display

    return order_genres_for_display(
        _resolve_track_genres(track, sidecar_db_path=sidecar_db_path, fallback=fallback)
    )
```

(Local import matches the file's existing style — `_resolve_track_genres` imports `EnrichedGenreResolver` the same way.)

- [ ] **Step 4: Wire the call site in `handle_generate`**

In `src/playlist_gui/worker.py` at ~line 1323, replace:

```python
                genres = _resolve_track_genres(
                    track,
                    sidecar_db_path=SIDECAR_DB_PATH,
                    fallback=_raw_genres,
                )
```

with:

```python
                genres = _resolve_display_genres(
                    track,
                    sidecar_db_path=SIDECAR_DB_PATH,
                    fallback=_raw_genres,
                )
```

(The dict literal further down already reads `"genres": genres` — unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_playlist_gui_genre_resolver.py tests/unit/test_genre_granularity.py -v`
Expected: all pass (5 in the resolver file, 10 in granularity).

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/worker.py tests/unit/test_playlist_gui_genre_resolver.py
git commit -m "feat(worker): playlist-table genres canonicalized and ordered sub->broad"
```

---

### Task 3: Web endpoint `POST /api/tracks/genres`

**Files:**
- Modify: `src/playlist_web/schemas.py` (add request model)
- Modify: `src/playlist_web/app.py` (add `SIDECAR_DB_PATH` constant ~line 40; add endpoint after `track_search`, ~line 247)
- Create: `tests/integration/test_web_track_genres_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_web_track_genres_api.py`. Pattern follows `tests/integration/test_web_tools_api.py` (TestClient + fake worker); DB paths are module globals in `app.py` read at call time, so `monkeypatch.setattr` works:

```python
# tests/integration/test_web_track_genres_api.py
"""Integration tests for POST /api/tracks/genres (staged-seed canonical genres).

Uses the REAL production taxonomy for ordering (slowcore 0.88 > shoegaze 0.86
> rock 0.05; 'seen live' rejected) and tmp metadata/sidecar DBs.
"""
import json
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

import src.playlist_web.app as app_module
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # --- tmp metadata.db ---
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT);
        CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT, priority INTEGER);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, weight REAL);

        INSERT INTO tracks VALUES ('t-enriched', 'Duster', 'Stratosphere');
        INSERT INTO tracks VALUES ('t-metadata', 'Acetone', 'Cindy');
        INSERT INTO tracks VALUES ('t-uncovered', 'Somebody', 'Something');

        -- t-metadata: effective genres stored broad-first + one unknown
        INSERT INTO track_effective_genres VALUES ('t-metadata', 'rock', 1);
        INSERT INTO track_effective_genres VALUES ('t-metadata', 'shoegaze', 2);
        INSERT INTO track_effective_genres VALUES ('t-metadata', 'totally-not-a-genre', 3);

        -- t-uncovered: nothing canonicalizes
        INSERT INTO track_effective_genres VALUES ('t-uncovered', 'seen live', 1);
        INSERT INTO track_effective_genres VALUES ('t-uncovered', 'no-such-genre', 2);
        """
    )
    conn.commit()
    conn.close()

    # --- tmp enrichment sidecar: t-enriched's release, stored broad-first ---
    from src.ai_genre_enrichment.storage import SidecarStore

    sidecar_path = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as sconn:
        sconn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "duster::stratosphere",
                "duster",
                "stratosphere",
                None,
                json.dumps({"genres": ["rock", "slowcore", "seen live"], "sources": []}),
                "2026-06-11T00:00:00",
            ),
        )
        sconn.commit()

    monkeypatch.setattr(app_module, "DB_PATH", db_path)
    monkeypatch.setattr(app_module, "SIDECAR_DB_PATH", sidecar_path)
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as c:
        yield c


def test_enriched_release_ordered_sub_to_broad(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-enriched"]})
    assert resp.status_code == 200
    # slowcore (0.88) before rock (0.05); 'seen live' rejected by the graph.
    assert resp.json() == {"t-enriched": ["slowcore", "rock"]}


def test_metadata_fallback_ordered_and_denoised(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-metadata"]})
    assert resp.status_code == 200
    assert resp.json() == {"t-metadata": ["shoegaze", "rock"]}


def test_uncovered_track_falls_back_to_raw(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-uncovered"]})
    assert resp.status_code == 200
    # Nothing canonicalizes -> raw tags unordered (never blank).
    assert resp.json() == {"t-uncovered": ["seen live", "no-such-genre"]}


def test_unknown_ids_omitted_and_batch_works(client):
    resp = client.post(
        "/api/tracks/genres",
        json={"track_ids": ["t-enriched", "t-metadata", "nope"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"t-enriched", "t-metadata"}


def test_empty_request_returns_empty(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": []})
    assert resp.status_code == 200
    assert resp.json() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_web_track_genres_api.py -v`
Expected: FAIL — `AttributeError: <module 'src.playlist_web.app' ...> does not have the attribute 'SIDECAR_DB_PATH'` (fixture monkeypatch), or 404s if run after a partial edit.

- [ ] **Step 3: Add the request schema**

In `src/playlist_web/schemas.py` (pydantic `BaseModel` classes — match neighbors), add:

```python
class TrackGenresRequest(BaseModel):
    """Batch lookup of display genres for staged seed tracks."""

    track_ids: list[str] = Field(default_factory=list)
```

If `Field` isn't already imported in the file, extend the import: `from pydantic import BaseModel, Field`.

- [ ] **Step 4: Add the constant and endpoint**

In `src/playlist_web/app.py`:

a) Next to `DB_PATH` (~line 40), add:

```python
SIDECAR_DB_PATH = ROOT / "data" / "ai_genre_enrichment.db"
```

b) Add `TrackGenresRequest` to the `from .schemas import (...)` block.

c) Inside `create_app`, immediately after the `track_search` endpoint (after its closing `finally: conn.close()` block, ~line 250), add:

```python
    @app.post("/api/tracks/genres")
    async def track_genres(body: TrackGenresRequest) -> dict[str, list[str]]:
        """Display genres for staged seed tracks: enriched -> metadata fallback,
        canonicalized through the taxonomy, ordered most-specific first.

        Called when the staged seed set changes (NOT per keystroke). Unknown
        track ids are omitted from the response.
        """
        ids = [str(t) for t in body.track_ids if str(t).strip()]
        if not ids or not DB_PATH.exists():
            return {}
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
        from src.genre.granularity import order_genres_for_display

        resolver = EnrichedGenreResolver(SIDECAR_DB_PATH)
        out: dict[str, list[str]] = {}
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                placeholders = ",".join("?" for _ in ids)
                rows = conn.execute(
                    f"SELECT track_id, artist, album FROM tracks WHERE track_id IN ({placeholders})",
                    ids,
                ).fetchall()
                for track_id, artist, album in rows:
                    tid = str(track_id)
                    raw = resolver.get_enriched_genres(artist=artist or "", album=album) or []
                    if not raw:
                        raw = [g[0] for g in conn.execute(
                            "SELECT genre FROM track_effective_genres WHERE track_id = ? ORDER BY priority",
                            (tid,),
                        ).fetchall()]
                    if not raw:
                        raw = [g[0] for g in conn.execute(
                            "SELECT genre FROM track_genres WHERE track_id = ? ORDER BY weight DESC",
                            (tid,),
                        ).fetchall()]
                    out[tid] = order_genres_for_display(raw)
            finally:
                conn.close()
        except sqlite3.Error:
            logger.warning("track_genres lookup failed", exc_info=True)
            return {}
        return out
```

(Sync sqlite inside `async def` matches the existing `track_search` endpoint. No `LIMIT` on the genre queries — the frontend's `+N` overflow needs the full list; ordering caps nothing.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_track_genres_api.py tests/integration/test_web_tools_api.py -v`
Expected: all pass (5 new + 4 existing; the tools file confirms `create_app` wasn't broken).

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/schemas.py src/playlist_web/app.py tests/integration/test_web_track_genres_api.py
git commit -m "feat(web): POST /api/tracks/genres returns canonical genres ordered sub->broad"
```

---

### Task 4: Frontend — GenreChips component + both surfaces

**Files:**
- Create: `web/src/components/GenreChips.tsx`
- Modify: `web/src/components/TrackTable.tsx:74-78` (replace `slice(0, 2)` chips)
- Modify: `web/src/components/SeedTrackSection.tsx` (staged-table chips at lines 152-154 + canonical-genre fetch; dropdown at line 93 UNCHANGED)
- Modify: `web/src/lib/api.ts` (add `trackGenres`)
- Create: `web/tests/genre-chips.spec.ts`

- [ ] **Step 1: Write the failing Playwright test**

Create `web/tests/genre-chips.spec.ts` (pattern: `web/tests/seeds.spec.ts` — localStorage seeding + route interception; the Playwright `webServer` builds `dist` and serves with the fake worker):

```typescript
import { test, expect } from "@playwright/test";

// Staged seeds fetch graph-canonical genres (POST /api/tracks/genres) and render
// them most-specific first: 6 chips + a "+N" overflow pill with a title tooltip.
// The search dropdown is intentionally untouched by this feature.
test("staged seeds show canonical genres with +N overflow", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("pg_mode", JSON.stringify("seeds"));
    localStorage.setItem(
      "pg_seed_tracks",
      JSON.stringify([
        { track_id: "k0", title: "Sundown", artist: "Acetone", album: "Cindy", genres: ["rock"], duration_ms: 200000, file_path: "/0.flac" },
      ]),
    );
  });

  // 7 ordered genres -> 6 chips + "+1"
  await page.route("**/api/tracks/genres", (route) =>
    route.fulfill({
      json: {
        k0: ["slowcore", "sadcore", "dream pop", "noise pop", "indie rock", "alternative rock", "rock"],
      },
    }),
  );

  await page.goto("/");

  // Canonical genres replaced the "rock" placeholder, most-specific first.
  await expect(page.getByText("slowcore")).toBeVisible();
  await expect(page.getByText("alternative rock")).toBeVisible();
  // 7th genre is behind the overflow pill.
  const overflow = page.getByTestId("genre-overflow");
  await expect(overflow).toHaveText("+1");
  await expect(overflow).toHaveAttribute("title", "rock");
});

test("staged seeds keep placeholder genres when the genres API fails", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("pg_mode", JSON.stringify("seeds"));
    localStorage.setItem(
      "pg_seed_tracks",
      JSON.stringify([
        { track_id: "k0", title: "Sundown", artist: "Acetone", album: "Cindy", genres: ["slowcore"], duration_ms: 200000, file_path: "/0.flac" },
      ]),
    );
  });
  await page.route("**/api/tracks/genres", (route) => route.abort());

  await page.goto("/");
  // Fallback: the metadata genres carried from the dropdown still render.
  await expect(page.getByText("slowcore")).toBeVisible();
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `web/`): `npm run test:e2e -- tests/genre-chips.spec.ts`
Expected: FAIL — `genre-overflow` test id doesn't exist (test 1); test 2 may pass already (placeholder rendering is current behavior) — that's fine, it pins the regression guard.

- [ ] **Step 3: Create the GenreChips component**

Create `web/src/components/GenreChips.tsx`:

```tsx
const DEFAULT_CAP = 6;

// Shared chip row for genre lists that arrive pre-ordered (most-specific first)
// from the backend. Shows up to `cap` chips, then a "+N" pill whose title
// tooltip lists the remainder.
export function GenreChips({
  genres,
  chipClass,
  cap = DEFAULT_CAP,
}: {
  genres: string[];
  chipClass: string;
  cap?: number;
}) {
  const shown = genres.slice(0, cap);
  const rest = genres.slice(cap);
  return (
    <>
      {shown.map((g) => (
        <span key={g} className={chipClass}>
          {g}
        </span>
      ))}
      {rest.length > 0 && (
        <span data-testid="genre-overflow" className={chipClass} title={rest.join(", ")}>
          +{rest.length}
        </span>
      )}
    </>
  );
}
```

- [ ] **Step 4: Add the API method**

In `web/src/lib/api.ts`, add to the `api` object (after `searchTracks`):

```typescript
  async trackGenres(trackIds: string[]): Promise<Record<string, string[]>> {
    return jsonOrThrow(await fetch("/api/tracks/genres", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_ids: trackIds }),
    }));
  },
```

- [ ] **Step 5: Wire TrackTable**

In `web/src/components/TrackTable.tsx`:

a) Add the import: `import { GenreChips } from "./GenreChips";`

b) Replace lines 74-78:

```tsx
              {c.row.original.genres.slice(0, 2).map((g) => (
                <span key={g} className="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full">
                  {g}
                </span>
              ))}
```

with:

```tsx
              <GenreChips
                genres={c.row.original.genres}
                chipClass="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full"
              />
```

- [ ] **Step 6: Wire SeedTrackSection (staged table only)**

In `web/src/components/SeedTrackSection.tsx`:

a) Add imports: `import { GenreChips } from "./GenreChips";` (the `api` import already exists).

b) Inside the component, after `const trackOuts = tracks.map(seedToTrackOut);` (line 37), add state + fetch-on-seed-change:

```tsx
  // Graph-canonical genres for staged seeds, fetched when the seed SET changes
  // (not per keystroke — the search dropdown keeps its metadata genres).
  // Until the fetch lands (or if it fails), t.genres is the placeholder.
  const [canonGenres, setCanonGenres] = useState<Record<string, string[]>>({});
  const idsKey = tracks.map((t) => t.track_id).join("|");
  useEffect(() => {
    if (tracks.length === 0) {
      setCanonGenres({});
      return;
    }
    let cancelled = false;
    api
      .trackGenres(tracks.map((t) => t.track_id))
      .then((m) => {
        if (!cancelled) setCanonGenres(m);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);
```

c) Replace the staged-table chips (lines 152-154):

```tsx
                      {t.genres.slice(0, 4).map((g) => (
                        <span key={g} className={CHIP}>{g}</span>
                      ))}
```

with:

```tsx
                      <GenreChips genres={canonGenres[t.track_id] ?? t.genres} chipClass={CHIP} />
```

d) Do NOT touch line 93 (`r.genres.slice(0, 2)` in the dropdown) — out of scope by design.

- [ ] **Step 7: Build and run the Playwright tests**

Run (from `web/`): `npm run build`
Expected: clean `tsc -b && vite build` (this also refreshes `web/dist` — the served GUI is the built dist, not the source).

Run (from `web/`): `npm run test:e2e -- tests/genre-chips.spec.ts tests/seeds.spec.ts`
Expected: all pass (`seeds.spec.ts` confirms the new fetch effect doesn't break generate; its unmocked `/api/tracks/genres` call is swallowed by the `.catch(() => {})`).

- [ ] **Step 8: Commit**

```bash
git add web/src/components/GenreChips.tsx web/src/components/TrackTable.tsx web/src/components/SeedTrackSection.tsx web/src/lib/api.ts web/tests/genre-chips.spec.ts
git commit -m "feat(web-ui): genre chips ordered sub->broad with 6-cap +N overflow"
```

---

### Task 5: Full verification

- [ ] **Step 1: Run the Python suite**

Run: `pytest -m "not slow" -q`
Expected: passes apart from the pre-existing deselect/perma-fail list (see memory: 12 known). The new tests (`test_genre_granularity.py`, `test_web_track_genres_api.py`, extended `test_playlist_gui_genre_resolver.py`) all pass. Investigate any NEW failure before proceeding — do not skip or loosen.

- [ ] **Step 2: Run the full web e2e suite**

Run (from `web/`): `npm run test:e2e`
Expected: all specs pass (generate, interactions, phase3, tools, seeds, genre-chips).

- [ ] **Step 3: Lint + types over changed Python**

Run: `ruff check src/ tests/ && mypy src/genre/granularity.py src/playlist_web/app.py`
Expected: clean.

- [ ] **Step 4: Manual GUI verification (real data)**

1. `python tools/serve_web.py` (rebuild already done in Task 4 — the stale-dist trap; the worker is spawned fresh by the bridge, so worker changes are live too).
2. Seed mode: add 2-3 seeds → staged rows should re-render with canonical genres, most-specific first, ≤6 + `+N` (hover the pill to see the rest). The dropdown chips remain as before.
3. Generate a playlist → result-table chips show canonical genres, most-specific first.
4. Spot-check one track with a busy release (e.g. a shoegaze album): the most niche tag leads, "rock"-tier tags trail or sit behind `+N`.

Report what was actually observed — if anything can't be verified, say so explicitly.

- [ ] **Step 5: Final commit (if any fixups)**

```bash
git status   # confirm only intended files changed
```

If fixups were needed in steps 1-4, commit them with a `fix(...)` message describing the actual issue found.
