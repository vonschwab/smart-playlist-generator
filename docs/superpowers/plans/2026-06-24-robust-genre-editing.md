# Robust Genre Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a GUI per-release genre edit authoritative immediately (write `release_effective_genres`), durable across a full re-publish (durable override + orphan-safe album_id), and able to steer generation after a one-click genre-only artifact re-bake.

**Architecture:** On Save, the worker resolves typed genres to canonical taxonomy `genre_id`s, writes a durable add/remove override (vs the non-user authority base) to `ai_genre_user_overrides`, and surgically materializes `release_effective_genres` for that album via the *same single-album materializer the full publish uses*. Display/export read the authority immediately; a separate one-click "Refresh genres for generation" re-bakes only the `X_genre_*` arrays in the artifact NPZ. A one-time publish fix makes orphaned albums (no `albums` row, e.g. Pet Grief) survive a full re-publish by deriving `album_id` from `tracks`.

**Tech Stack:** Python 3.11, SQLite (metadata.db + ai_genre_enrichment.db sidecar), numpy artifact NPZ, FastAPI + NDJSON worker bridge, React + TypeScript + Vite + Tailwind, pytest, Playwright.

**Spec:** `docs/superpowers/specs/2026-06-24-robust-genre-editing-design.md`

## Global Constraints

- Python **3.11+**. Run `pytest` directly with the tool timeout; never pipe through `tail`/`head`. Fast subset: `python -m pytest -q -m "not slow"`.
- **Never write the real metadata.db / ai_genre_enrichment.db in tests.** Tests build or copy DBs into a `tmp_path`. Never symlink a real SQLite DB into the worktree (dual-WAL corruption).
- The only metadata.db write in this feature is to `release_effective_genres` (derived, regenerable). Never write `tracks`, `albums`, or audio/MERT tables. No per-edit DB backup.
- After editing `web/src/**`, rebuild: `npm --prefix web run build`. After editing `worker.py` (or its imports), the running `serve_web.py` must be restarted to take effect.
- Worker handlers must `emit_done(...)` on every path (success, error, cancel) or the bridge wedges.
- Subagent model policy: spawn search/mechanical agents on `haiku`, implementation agents on `sonnet`; never inherit the session model. Subagents launch in the MAIN checkout — when executing in this worktree, have each subagent `cd` into the worktree path and verify the branch before committing.
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Worktree: `.claude/worktrees/worktree-robust-genre-editing`, branch `worktree-worktree-robust-genre-editing`. `config.yaml` already copied in; `data/` holds only tracked YAMLs (the big DBs/artifacts live in the main checkout — relevant only for the final manual e2e, which runs from the real checkout).

---

## File structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/genre/genre_publish.py` | Modify | Extract `materialize_album_genres`; single-album `legacy_genres_by_album`; orphan-safe `resolve_release_key_to_album_id` + `build_resolved_table` album set. |
| `src/genre/genre_edit.py` | Create | Edit orchestration: term resolution, album_id-from-tracks, diff vs non-user base, write override + surgical materialize. |
| `src/genre/authority.py` | Modify | `canonical_genre_search` (autocomplete) + `display_genre_names_for_album`. |
| `scripts/build_beat3tower_artifacts.py` | Modify | `refresh_genre_matrices` (genre-only NPZ re-bake). |
| `src/playlist_gui/worker.py` | Modify | Rewrite `handle_edit_genres`; add `handle_refresh_genre_artifact`; register it. |
| `src/playlist_web/app.py` | Modify | `/api/genres/search`, `/api/genres/for_album`, `/api/refresh_genre_artifact`; extend `/api/edit_genres` response. |
| `src/playlist_web/schemas.py` | Modify | `JobLaunchResponse` reuse; extend edit response shape (returned as dict). |
| `web/src/lib/types.ts` | Modify | `EditGenresResponse`, `CanonicalGenre`. |
| `web/src/lib/api.ts` | Modify | `genresSearch`, `albumGenres`, `refreshGenreArtifact`; edit return type. |
| `web/src/components/EditGenresDialog.tsx` | Modify | Autocomplete, pending-input commit, warnings, fetch-on-open. |
| `web/src/App.tsx` | Modify | Authoritative `resolved` update + "Refresh genres for generation" button. |
| `tests/unit/test_genre_publish_materializer.py` | Create | Materializer parity + orphan durability. |
| `tests/unit/test_genre_edit.py` | Create | Term resolution, diff, orphan, no-op, remove. |
| `tests/unit/test_refresh_genre_matrices.py` | Create | Genre-only re-bake changes X_genre, preserves sonic. |
| `tests/integration/test_web_genre_editing.py` | Create | edit_genres + refresh through the real worker bridge (asyncio). |
| `tests/fixtures/fake_worker.py` | Modify | Branches for `edit_genres`, `refresh_genre_artifact`. |
| `web/tests/edit-genres.spec.ts` | Create | Playwright: autocomplete + save + refresh. |

---

## Phase 1 — Shared genre layer

### Task 1: Extract `materialize_album_genres` from `build_resolved_table`

**Files:**
- Modify: `src/genre/genre_publish.py:375-424`
- Test: `tests/unit/test_genre_publish_materializer.py`

**Interfaces:**
- Produces: `materialize_album_genres(conn, album_id: str, *, graph_album_ids: set[str], legacy: dict[str, list[tuple[str, float]]], overrides: dict[str, tuple[list[str], set[str]]], album_to_key: dict[str, str]) -> None` — deletes the album's `release_effective_genres` rows then writes graph-or-legacy + override result. Idempotent per album.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_publish_materializer.py
import sqlite3
import pytest
from src.genre import genre_publish


def _schema(conn):
    conn.execute(
        "CREATE TABLE release_effective_genres ("
        "album_id TEXT NOT NULL, release_key TEXT, genre_id TEXT NOT NULL, "
        "assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, "
        "PRIMARY KEY (album_id, genre_id, assignment_layer))"
    )
    conn.execute(
        "CREATE TABLE genre_graph_release_genre_assignments ("
        "album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL)"
    )


def test_materialize_orphan_album_user_only():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _schema(conn)
    # Orphan: no graph rows, no legacy; user adds two genres.
    genre_publish.materialize_album_genres(
        conn, "ALB1",
        graph_album_ids=set(), legacy={},
        overrides={"ALB1": (["dream_pop", "shoegaze"], set())},
        album_to_key={"ALB1": "the radio dept::pet grief"},
    )
    rows = conn.execute(
        "SELECT genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres WHERE album_id='ALB1' ORDER BY genre_id"
    ).fetchall()
    assert [(r["genre_id"], r["assignment_layer"], r["confidence"], r["source"]) for r in rows] == [
        ("dream_pop", "observed_leaf", 1.0, "user"),
        ("shoegaze", "observed_leaf", 1.0, "user"),
    ]


def test_materialize_graph_minus_remove_plus_add():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _schema(conn)
    conn.executemany(
        "INSERT INTO genre_graph_release_genre_assignments VALUES (?,?,?,?)",
        [("ALB2", "cool_jazz", "observed_leaf", 0.9),
         ("ALB2", "jazz", "inferred_family", 0.9)],
    )
    genre_publish.materialize_album_genres(
        conn, "ALB2",
        graph_album_ids={"ALB2"}, legacy={},
        overrides={"ALB2": (["post_bop"], {"cool_jazz"})},
        album_to_key={"ALB2": "k"},
    )
    got = {(r["genre_id"], r["source"]) for r in conn.execute(
        "SELECT genre_id, source FROM release_effective_genres WHERE album_id='ALB2'")}
    assert got == {("jazz", "graph"), ("post_bop", "user")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_publish_materializer.py -v`
Expected: FAIL — `AttributeError: module 'src.genre.genre_publish' has no attribute 'materialize_album_genres'`

- [ ] **Step 3: Add `materialize_album_genres` and call it from `build_resolved_table`**

Insert this function immediately before `build_resolved_table` (currently line 375):

```python
def materialize_album_genres(
    conn,
    album_id: str,
    *,
    graph_album_ids: set[str],
    legacy: dict[str, list[tuple[str, float]]],
    overrides: dict[str, tuple[list[str], set[str]]],
    album_to_key: dict[str, str],
) -> None:
    """Write release_effective_genres rows for one album. Idempotent per album.

    Graph-where-present else legacy, then apply the album's override
    (drop remove_match genre_ids, add user observed_leaf rows). Shared by the
    full publish loop and the single-release edit path so both produce
    identical rows.
    """
    rows: dict[tuple[str, str], tuple[float, str]] = {}
    if album_id in graph_album_ids:
        for genre_id, layer, conf in conn.execute(
            "SELECT genre_id, assignment_layer, confidence "
            "FROM genre_graph_release_genre_assignments WHERE album_id = ?",
            (album_id,),
        ):
            rows[(genre_id, layer)] = (conf, "graph")
    elif album_id in legacy:
        for genre_id, weight in legacy[album_id]:
            rows[(genre_id, "legacy")] = (weight, "legacy")

    if album_id in overrides:
        add_ids, remove_match = overrides[album_id]
        rows = {k: v for k, v in rows.items() if k[0] not in remove_match}
        for gid in add_ids:
            rows[(gid, "observed_leaf")] = (1.0, "user")

    release_key = album_to_key.get(album_id)
    conn.execute("DELETE FROM release_effective_genres WHERE album_id = ?", (album_id,))
    for (genre_id, layer), (conf, source) in rows.items():
        conn.execute(
            "INSERT OR REPLACE INTO release_effective_genres "
            "(album_id, release_key, genre_id, assignment_layer, confidence, source) "
            "VALUES (?,?,?,?,?,?)",
            (album_id, release_key, genre_id, layer, conf, source),
        )
```

Then replace the per-album body of `build_resolved_table` (the `for album_id in all_album_ids:` loop, lines 398-424) with:

```python
    for album_id in all_album_ids:
        materialize_album_genres(
            conn, album_id,
            graph_album_ids=graph_album_ids, legacy=legacy,
            overrides=overrides, album_to_key=album_to_key,
        )
```

(The `conn.execute("DELETE FROM release_effective_genres")` at the top of `build_resolved_table` stays; the per-album DELETE inside the materializer is harmless on an already-empty table.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_publish_materializer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the existing publish tests for regression**

Run: `python -m pytest tests/unit/test_genre_publish.py -q`
Expected: PASS (no regressions — `build_resolved_table` output is unchanged)

- [ ] **Step 6: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish_materializer.py
git commit -m "refactor(genre): extract materialize_album_genres shared by publish + edit"
```

---

### Task 2: Orphan-safe album_id mapping (durability fix)

**Files:**
- Modify: `src/genre/genre_publish.py` (`resolve_release_key_to_album_id` ~216-251; `build_resolved_table` `all_album_ids` ~392-396)
- Modify: `src/genre/genre_publish.py` (`legacy_genres_by_album` ~290-329 — add optional `album_id` filter, consumed in Task 5)
- Test: `tests/unit/test_genre_publish_materializer.py` (append)

**Interfaces:**
- Modifies: `resolve_release_key_to_album_id(conn) -> tuple[dict[str,str], int]` now also maps keys derived from `tracks` for album_ids absent from `albums`.
- Produces: `legacy_genres_by_album(conn, album_id: str | None = None) -> dict[str, list[tuple[str,float]]]` — when `album_id` is given, restrict to that album.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_orphan_album_survives_full_publish(tmp_path):
    """An override on an album with tracks but no `albums` row must publish."""
    import json
    from src.ai_genre_enrichment.storage import SidecarStore
    meta = tmp_path / "metadata.db"
    side = tmp_path / "sidecar.db"

    mconn = sqlite3.connect(meta)
    mconn.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
    )
    # Orphan album: tracks present, NO albums row.
    mconn.execute("INSERT INTO tracks VALUES ('t1','The Radio Dept.','Pet Grief','ORPH1')")
    mconn.commit()
    mconn.close()

    store = SidecarStore(str(side))
    store.initialize()
    store.set_user_override(
        release_key="the radio dept::pet grief",
        normalized_artist="the radio dept", normalized_album="pet grief",
        genres_add=["dream pop"], genres_remove=[],
    )

    stats = genre_publish.publish(str(meta), str(side))
    mconn = sqlite3.connect(meta)
    got = mconn.execute(
        "SELECT genre_id, source FROM release_effective_genres WHERE album_id='ORPH1'"
    ).fetchall()
    mconn.close()
    assert ("dream_pop", "user") in {(g, s) for g, s in got}, f"orphan override lost: {got}"
```

(Assumes `dream pop` resolves to canonical id `dream_pop`; if the test taxonomy differs, assert membership of the resolved id reported by `classify_override_terms`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest "tests/unit/test_genre_publish_materializer.py::test_orphan_album_survives_full_publish" -v`
Expected: FAIL — `release_effective_genres` has no row for `ORPH1` (orphan dropped).

- [ ] **Step 3: Add the tracks-derived mapping step**

In `resolve_release_key_to_album_id`, after the "recompute from albums" block (before `return mapping, collisions` at line 251), add:

```python
    # 3) recompute from tracks for album_ids absent from `albums` (orphans).
    #    Tracks carry the real album_id; an album row may never have been
    #    created (e.g. a double-space artist string). Genre edits on these
    #    must still publish, so derive their release_key -> album_id here.
    for album_id, artist, album in conn.execute(
        "SELECT DISTINCT album_id, artist, album FROM tracks "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        key = f"{normalize_release_artist(artist)}::{normalize_release_name(album)}"
        if not key or key == "::":
            continue
        mapping.setdefault(key, album_id)
```

In `build_resolved_table`, replace the `all_album_ids` query (lines 392-396) with a UNION of `albums` and track-referenced album_ids:

```python
    all_album_ids = [
        r[0] for r in conn.execute(
            "SELECT album_id FROM albums WHERE album_id IS NOT NULL AND album_id != '' "
            "UNION "
            "SELECT DISTINCT album_id FROM tracks WHERE album_id IS NOT NULL AND album_id != ''"
        )
    ]
```

- [ ] **Step 4: Add the optional `album_id` filter to `legacy_genres_by_album`**

Change the signature and the three queries. New signature:

```python
def legacy_genres_by_album(conn, album_id: str | None = None) -> dict[str, list[tuple[str, float]]]:
```

Add a filter clause to each of the three `conn.execute` queries inside it. For the `tracks` query:

```python
    track_sql = (
        "SELECT t.album_id, tg.genre FROM tracks t "
        "JOIN track_genres tg ON tg.track_id = t.track_id "
        "WHERE t.album_id IS NOT NULL AND t.album_id != ''"
    )
    params: tuple = ()
    if album_id is not None:
        track_sql += " AND t.album_id = ?"
        params = (album_id,)
    for aid, genre in conn.execute(track_sql, params):
        add(aid, genre, _WEIGHT_TRACK)
```

Apply the equivalent `AND album_id = ?` / `AND a.album_id = ?` filter to the `album_genres` and `albums JOIN artist_genres` queries (each guarded by `if album_id is not None`). The full-scan behavior (`album_id=None`) is unchanged, so publish is unaffected.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/unit/test_genre_publish_materializer.py tests/unit/test_genre_publish.py -q`
Expected: PASS (orphan test green; publish regression suite still green)

- [ ] **Step 6: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish_materializer.py
git commit -m "fix(genre): orphan-safe album_id mapping so edits on albums w/o albums-row survive publish"
```

---

## Phase 2 — Edit core

### Task 3: `canonical_genre_search` + `display_genre_names_for_album`

**Files:**
- Modify: `src/genre/authority.py`
- Test: `tests/unit/test_genre_edit.py`

**Interfaces:**
- Produces: `canonical_genre_search(conn, query: str, limit: int = 20) -> list[tuple[str, str]]` — `(genre_id, name)` for active canonical genres whose name contains `query` (case-insensitive), most-specific first.
- Produces: `display_genre_names_for_album(conn, album_id: str) -> list[str]` — deduped display names for an album_id (mirrors `display_genre_names_for_track` but by album_id).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_edit.py
import sqlite3
import pytest
from src.genre import authority


def _canon(conn):
    conn.execute(
        "CREATE TABLE genre_graph_canonical_genres ("
        "genre_id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, "
        "specificity_score REAL NOT NULL, status TEXT NOT NULL, taxonomy_version TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?,?,?,?,?,?)",
        [("dream_pop", "Dream Pop", "genre", 0.8, "active", "v1"),
         ("dreamo", "Dreamo", "genre", 0.7, "active", "v1"),
         ("shoegaze", "Shoegaze", "genre", 0.9, "active", "v1"),
         ("old_thing", "Old Dream", "genre", 0.5, "deprecated", "v1")],
    )


def test_canonical_genre_search_matches_active_by_name():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _canon(conn)
    out = authority.canonical_genre_search(conn, "dream", limit=10)
    names = [n for _, n in out]
    assert "Dream Pop" in names and "Dreamo" in names
    assert "Old Dream" not in names  # deprecated excluded
    assert ("shoegaze", "Shoegaze") not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edit.py::test_canonical_genre_search_matches_active_by_name -v`
Expected: FAIL — `AttributeError: ... has no attribute 'canonical_genre_search'`

- [ ] **Step 3: Implement both functions in `src/genre/authority.py`**

Append:

```python
def canonical_genre_search(conn, query: str, limit: int = 20):
    """Active canonical genres whose name contains `query` (case-insensitive)."""
    q = (query or "").strip()
    if not q:
        return []
    rows = conn.execute(
        "SELECT genre_id, name FROM genre_graph_canonical_genres "
        "WHERE status = 'active' AND LOWER(name) LIKE '%' || LOWER(?) || '%' "
        "ORDER BY specificity_score DESC, name ASC LIMIT ?",
        (q, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def display_genre_names_for_album(conn, album_id: str) -> list[str]:
    """Published genres for an album_id as deduped display names."""
    rows = conn.execute(
        "SELECT reg.genre_id, COALESCE(g.name, reg.genre_id) "
        "FROM release_effective_genres reg "
        "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
        "WHERE reg.album_id = ? ORDER BY reg.assignment_layer, reg.genre_id",
        (str(album_id),),
    ).fetchall()
    out: list[str] = []
    seen: set[str] = set()
    for _gid, name in rows:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_edit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/genre/authority.py tests/unit/test_genre_edit.py
git commit -m "feat(genre): canonical_genre_search + display_genre_names_for_album for editing"
```

---

### Task 4: `genre_edit` helpers — term resolution, album_id-from-tracks

**Files:**
- Create: `src/genre/genre_edit.py`
- Test: `tests/unit/test_genre_edit.py` (append)

**Interfaces:**
- Consumes: `genre_publish._term_to_genre_id`, `genre_publish.legacy_genres_by_album`, `authority.resolved_genres_for_album`, `authority.canonical_genre_names`, `ai_genre_enrichment.normalization.make_release_key`.
- Produces:
  - `resolve_terms_to_genre_ids(taxonomy, names: list[str]) -> tuple[dict[str, str], list[str]]` — `(name -> genre_id resolved, unknown_names)`.
  - `album_id_for_release(conn, artist: str, album: str) -> str | None` — exact (artist, album) over `tracks`, else normalized `make_release_key` grouped over tracks; deterministic pick (most tracks, then min id).

- [ ] **Step 1: Write the failing test (append)**

```python
def test_album_id_for_release_exact_and_orphan():
    from src.genre import genre_edit
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT)")
    conn.executemany(
        "INSERT INTO tracks VALUES (?,?,?,?)",
        [("t1", "The  Radio Dept.", "Pet Grief", "ORPH1"),
         ("t2", "The  Radio Dept.", "Pet Grief", "ORPH1"),
         ("t3", "Acetone", "York Blvd.", "A1")],
    )
    assert genre_edit.album_id_for_release(conn, "The  Radio Dept.", "Pet Grief") == "ORPH1"
    # normalized fallback: double-space vs single-space artist still resolves
    assert genre_edit.album_id_for_release(conn, "The Radio Dept.", "Pet Grief") == "ORPH1"
    assert genre_edit.album_id_for_release(conn, "Nobody", "Nothing") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edit.py::test_album_id_for_release_exact_and_orphan -v`
Expected: FAIL — module `src.genre.genre_edit` does not exist.

- [ ] **Step 3: Create `src/genre/genre_edit.py`**

```python
"""User genre edit orchestration: resolve terms, locate album, apply override.

Writes the durable add/remove override (ai_genre_user_overrides) AND the
surgical release_effective_genres rows via the shared publish materializer, so
the edit is authoritative immediately and reproduced byte-for-byte by a later
full publish.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ai_genre_enrichment.normalization import (
    make_release_key,
    normalize_release_artist,
    normalize_release_name,
)
from src.genre import genre_publish
from src.genre.authority import canonical_genre_names, resolved_genres_for_album


def resolve_terms_to_genre_ids(taxonomy, names: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map free-typed names to canonical genre_ids. Unresolved names returned."""
    resolved: dict[str, str] = {}
    unknown: list[str] = []
    for name in names:
        term = (name or "").strip()
        if not term:
            continue
        gid = genre_publish._term_to_genre_id(taxonomy, term)
        if gid:
            resolved[term] = gid
        else:
            unknown.append(term)
    return resolved, unknown


def album_id_for_release(conn, artist: str, album: str) -> str | None:
    """Resolve album_id from the tracks table (orphan-safe).

    Exact (artist, album) first; else normalized release_key grouped over
    tracks, picking the album_id with the most tracks (ties: lexicographically
    smallest) for determinism.
    """
    row = conn.execute(
        "SELECT album_id, COUNT(*) c FROM tracks "
        "WHERE artist = ? AND album = ? AND album_id IS NOT NULL AND album_id != '' "
        "GROUP BY album_id ORDER BY c DESC, album_id ASC LIMIT 1",
        (artist, album),
    ).fetchone()
    if row and row[0]:
        return row[0]

    target_key = make_release_key(artist, album)
    counts: dict[str, int] = {}
    for aid, a, alb in conn.execute(
        "SELECT album_id, artist, album FROM tracks "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        if make_release_key(a, alb) == target_key:
            counts[aid] = counts.get(aid, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _album_in_graph(conn, album_id: str) -> set[str]:
    row = conn.execute(
        "SELECT 1 FROM genre_graph_release_genre_assignments WHERE album_id = ? LIMIT 1",
        (album_id,),
    ).fetchone()
    return {album_id} if row else set()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_edit.py::test_album_id_for_release_exact_and_orphan -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_edit.py tests/unit/test_genre_edit.py
git commit -m "feat(genre): genre_edit term resolution + orphan-safe album_id lookup"
```

---

### Task 5: `apply_user_genre_edit` — diff, override, surgical materialize

**Files:**
- Modify: `src/genre/genre_edit.py`
- Test: `tests/unit/test_genre_edit.py` (append)

**Interfaces:**
- Consumes: Task 4 helpers; `genre_publish.materialize_album_genres`, `genre_publish.legacy_genres_by_album(conn, album_id)`, `genre_publish.classify_override_terms`; `SidecarStore.set_user_override`.
- Produces:
  - `@dataclass EditResult(resolved: list[str], unknown: list[str], added: list[str], removed: list[str], no_change: bool)`
  - `apply_user_genre_edit(meta_conn, sidecar_store, taxonomy, *, artist: str, album: str, target_names: list[str]) -> EditResult` — base for the diff is read **server-side** from `release_effective_genres` rows with `source != 'user'`.

- [ ] **Step 1: Write the failing test (append)**

```python
def _edit_dbs(tmp_path):
    """Build a metadata.db with the tables the edit path reads/writes."""
    meta = sqlite3.connect(tmp_path / "m.db")
    meta.row_factory = sqlite3.Row
    meta.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        "CREATE TABLE genre_graph_release_genre_assignments "
        "(album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL);"
        "CREATE TABLE genre_graph_canonical_genres "
        "(genre_id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, "
        " specificity_score REAL NOT NULL, status TEXT NOT NULL, taxonomy_version TEXT NOT NULL);"
        "CREATE TABLE release_effective_genres "
        "(album_id TEXT NOT NULL, release_key TEXT, genre_id TEXT NOT NULL, "
        " assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, "
        " PRIMARY KEY (album_id, genre_id, assignment_layer));"
    )
    meta.execute("INSERT INTO tracks VALUES ('t1','The  Radio Dept.','Pet Grief','ORPH1')")
    meta.commit()
    return meta


def test_apply_edit_orphan_zero_to_two(tmp_path):
    from src.genre import genre_edit
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()

    # Use real canonical ids the taxonomy resolves; assert via classify.
    add_terms = ["dream pop", "shoegaze"]
    res = genre_edit.apply_user_genre_edit(
        meta, store, taxonomy,
        artist="The  Radio Dept.", album="Pet Grief", target_names=add_terms,
    )
    assert res.no_change is False
    user_rows = meta.execute(
        "SELECT genre_id FROM release_effective_genres "
        "WHERE album_id='ORPH1' AND source='user'"
    ).fetchall()
    assert len(user_rows) == len(res.added) == 2
    # Durable override persisted
    ov = store.get_user_override("the radio dept::pet grief")
    assert ov is not None and len(ov["genres_add"]) == 2


def test_apply_edit_no_op_when_unchanged(tmp_path):
    from src.genre import genre_edit
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["dream pop"])
    # Re-apply identical target → no change, no second override write churn.
    res2 = genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["dream pop"])
    assert res2.no_change is True
    assert res2.added == [] and res2.removed == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edit.py -k apply_edit -v`
Expected: FAIL — `apply_user_genre_edit` not defined.

- [ ] **Step 3: Implement `EditResult` + `apply_user_genre_edit`**

Append to `src/genre/genre_edit.py`:

```python
def _split(term: str) -> set[str]:
    return set(genre_publish._split(term))


@dataclass
class EditResult:
    resolved: list[str]
    unknown: list[str]
    added: list[str]
    removed: list[str]
    no_change: bool


def apply_user_genre_edit(
    meta_conn,
    sidecar_store,
    taxonomy,
    *,
    artist: str,
    album: str,
    target_names: list[str],
) -> EditResult:
    resolved_map, unknown = resolve_terms_to_genre_ids(taxonomy, target_names)
    target_ids = set(resolved_map.values())

    album_id = album_id_for_release(meta_conn, artist, album)
    if album_id is None:
        raise ValueError(f"no album_id for {artist!r} / {album!r}")

    id_to_name = canonical_genre_names(meta_conn)

    def name_of(gid: str) -> str:
        return id_to_name.get(gid, gid)

    base_ids = {
        r.genre_id for r in resolved_genres_for_album(meta_conn, album_id)
        if r.source != "user"
    }
    add_ids = target_ids - base_ids
    remove_ids = base_ids - target_ids
    no_change = not add_ids and not remove_ids

    resolved_names = sorted(name_of(gid) for gid in target_ids)
    if no_change:
        return EditResult(resolved=resolved_names, unknown=unknown,
                          added=[], removed=[], no_change=True)

    add_names = sorted(name_of(gid) for gid in add_ids)
    remove_names = sorted(name_of(gid) for gid in remove_ids)

    # Durable replay instruction (vs the non-user base) — survives full publish.
    sidecar_store.set_user_override(
        release_key=make_release_key(artist, album),
        normalized_artist=normalize_release_artist(artist),
        normalized_album=normalize_release_name(album),
        genres_add=add_names,
        genres_remove=remove_names,
    )

    # Surgical materialize via the SAME path publish uses (parity).
    remove_match: set[str] = set(remove_ids)
    for n in remove_names:
        remove_match |= _split(n)
    overrides = {album_id: (list(add_ids), remove_match)}
    materialize = genre_publish.materialize_album_genres
    materialize(
        meta_conn, album_id,
        graph_album_ids=_album_in_graph(meta_conn, album_id),
        legacy=genre_publish.legacy_genres_by_album(meta_conn, album_id),
        overrides=overrides,
        album_to_key={album_id: make_release_key(artist, album)},
    )
    meta_conn.commit()
    return EditResult(resolved=resolved_names, unknown=unknown,
                      added=add_names, removed=remove_names, no_change=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_edit.py -k apply_edit -v`
Expected: PASS (2 tests). If `load_default_layered_taxonomy` is slow, mark these `@pytest.mark.slow` is NOT needed — it loads a cached YAML.

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_edit.py tests/unit/test_genre_edit.py
git commit -m "feat(genre): apply_user_genre_edit — durable override + surgical authority write"
```

---

### Task 6: Rewrite worker `handle_edit_genres`

**Files:**
- Modify: `src/playlist_gui/worker.py` (`handle_edit_genres` ~2404-2461)
- Test: covered by Task 9 integration (real worker). Add a focused unit test if a DB fixture is convenient.

**Interfaces:**
- Consumes: `genre_edit.apply_user_genre_edit`, `load_default_layered_taxonomy`, `SidecarStore`, `SIDECAR_DB_PATH`.
- Produces: `result` event `{"artist","album","resolved","unknown","added","removed","no_change"}` then `done`.

- [ ] **Step 1: Replace the body of `handle_edit_genres`**

```python
def handle_edit_genres(cmd_data: Dict[str, Any]) -> None:
    """Apply a user genre edit: durable override + surgical authority write.

    Resolves typed genres to canonical taxonomy ids (unknowns reported, not
    saved), then writes both the durable ai_genre_user_overrides diff and the
    release_effective_genres rows for the album via the shared publish
    materializer. base for the diff is read server-side from the authority.
    """
    try:
        artist = (cmd_data.get("artist") or "").strip()
        album = (cmd_data.get("album") or "").strip()
        target_names = [
            str(g).strip() for g in (cmd_data.get("genres") or []) if str(g).strip()
        ]
        if not artist or not album:
            raise ValueError("artist and album are required")

        import sqlite3
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        from src.ai_genre_enrichment.storage import SidecarStore
        from src.genre.genre_edit import apply_user_genre_edit

        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        db_path = config.get("library", {}).get("database_path", "data/metadata.db")

        meta_conn = sqlite3.connect(db_path)
        meta_conn.row_factory = sqlite3.Row
        try:
            store = SidecarStore(SIDECAR_DB_PATH)
            store.initialize()
            taxonomy = load_default_layered_taxonomy()
            result = apply_user_genre_edit(
                meta_conn, store, taxonomy,
                artist=artist, album=album, target_names=target_names,
            )
        finally:
            meta_conn.close()

        emit_result("edit_genres", {
            "artist": artist, "album": album,
            "resolved": result.resolved, "unknown": result.unknown,
            "added": result.added, "removed": result.removed,
            "no_change": result.no_change,
        })
        emit_done("edit_genres", True, "ok")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("edit_genres", False, str(e))
```

- [ ] **Step 2: Restart-free check (compile)**

Run: `python -c "import ast; ast.parse(open('src/playlist_gui/worker.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/playlist_gui/worker.py
git commit -m "feat(worker): edit_genres applies authority correction (resolved/unknown/no_change)"
```

---

## Phase 3 — Artifact re-bake

### Task 7: `refresh_genre_matrices`

**Files:**
- Modify: `scripts/build_beat3tower_artifacts.py`
- Test: `tests/unit/test_refresh_genre_matrices.py`

**Interfaces:**
- Consumes: existing `load_tracks_with_beat3tower`, `load_genres_for_tracks`, `build_genre_matrices`; `GenreArtifactSource`, `make_resolver`, `Config`.
- Produces: `refresh_genre_matrices(artifact_path: str, db_path: str, *, genre_sim_path: str | None, sidecar_db: str, config_path: str) -> dict` — re-bakes only `X_genre_raw`, `X_genre_smoothed`, `genre_vocab`; preserves all other NPZ arrays; returns `{"n_tracks","n_genres"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_refresh_genre_matrices.py
import numpy as np
import sqlite3
import pytest


@pytest.mark.slow
def test_refresh_changes_genre_preserves_sonic(tmp_path, monkeypatch):
    """Re-bake updates X_genre for an edited album and leaves X_sonic intact."""
    # Minimal artifact with two tracks, one belonging to album ALB1.
    art = tmp_path / "art.npz"
    sonic = np.random.RandomState(0).randn(2, 4).astype(np.float32)
    np.savez(
        art,
        X_sonic=sonic, X_sonic_raw=sonic,
        X_genre_raw=np.zeros((2, 1), dtype=np.float32),
        X_genre_smoothed=np.zeros((2, 1), dtype=np.float32),
        genre_vocab=np.array(["placeholder"], dtype=object),
        track_ids=np.array(["t1", "t2"], dtype=object),
        build_config=np.array({"genre_source": "graph"}, dtype=object),
    )
    # DB with authority genres for t1's album.
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, title TEXT, norm_artist TEXT, "
        " album TEXT, album_id TEXT, duration_ms INT);"
        "CREATE TABLE release_effective_genres (album_id TEXT, release_key TEXT, genre_id TEXT, "
        " assignment_layer TEXT, confidence REAL, source TEXT);"
        "CREATE TABLE genre_graph_canonical_genres (genre_id TEXT PRIMARY KEY, name TEXT, kind TEXT, "
        " specificity_score REAL, status TEXT, taxonomy_version TEXT);"
    )
    conn.execute("INSERT INTO tracks VALUES ('t1','A','x','a','Alb','ALB1',1000)")
    conn.execute("INSERT INTO tracks VALUES ('t2','B','y','b','Alb2','ALB2',1000)")
    conn.execute("INSERT INTO release_effective_genres VALUES "
                 "('ALB1','k','dream_pop','observed_leaf',1.0,'user')")
    conn.execute("INSERT INTO genre_graph_canonical_genres VALUES "
                 "('dream_pop','Dream Pop','genre',0.8,'active','v1')")
    conn.commit()
    conn.close()

    # load_tracks_with_beat3tower expects beat3tower features; monkeypatch it to
    # return our two tracks with no features (genre path queries the DB itself).
    import scripts.build_beat3tower_artifacts as bba
    rows = [
        {"track_id": "t1", "artist": "A", "title": "x", "norm_artist": "a",
         "album": "Alb", "album_id": "ALB1", "duration_ms": 1000},
        {"track_id": "t2", "artist": "B", "title": "y", "norm_artist": "b",
         "album": "Alb2", "album_id": "ALB2", "duration_ms": 1000},
    ]
    monkeypatch.setattr(bba, "load_tracks_with_beat3tower", lambda db_path, max_tracks=0: (rows, [{}, {}]))

    out = bba.refresh_genre_matrices(
        str(art), str(db), genre_sim_path=None,
        sidecar_db=str(tmp_path / "s.db"), config_path="config.yaml",
    )
    data = np.load(art, allow_pickle=True)
    vocab = data["genre_vocab"].tolist()
    assert "dream pop" in [v.lower() for v in vocab] or "Dream Pop" in vocab
    # t1 has nonzero genre weight; sonic preserved exactly.
    j = [v.lower() for v in vocab].index("dream pop") if "dream pop" in [v.lower() for v in vocab] else vocab.index("Dream Pop")
    assert data["X_genre_raw"][0, j] > 0
    assert np.array_equal(data["X_sonic_raw"], sonic)
    assert out["n_tracks"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_refresh_genre_matrices.py -v`
Expected: FAIL — `refresh_genre_matrices` not defined.

- [ ] **Step 3: Implement `refresh_genre_matrices`**

Add near `build_genre_matrices` in `scripts/build_beat3tower_artifacts.py`:

```python
def refresh_genre_matrices(
    artifact_path: str,
    db_path: str,
    *,
    genre_sim_path: str | None,
    sidecar_db: str,
    config_path: str,
) -> dict:
    """Re-bake ONLY the genre matrices in an existing artifact from the authority.

    Loads the NPZ, recomputes X_genre_raw/smoothed + genre_vocab for the
    artifact's track order using the same loaders as a full build, and re-saves
    with every other array (sonic/MERT, metadata) written back unchanged.
    """
    from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource, make_resolver
    from src.config_loader import Config

    data = dict(np.load(artifact_path, allow_pickle=True))
    track_ids = [str(t) for t in data["track_ids"].tolist()]

    config_genre_source = (
        Config(config_path).config.get("playlists", {}).get("ds_pipeline", {}).get("genre_source")
    )
    genre_source = GenreArtifactSource.resolve(config_genre_source)
    resolver = make_resolver(genre_source, sidecar_db)

    tracks_meta, _features = load_tracks_with_beat3tower(db_path)
    by_id = {t["track_id"]: t for t in tracks_meta}
    tracks_metadata = [by_id[tid] for tid in track_ids if tid in by_id]

    genre_lists, vocab, _stats = load_genres_for_tracks(
        db_path, track_ids, normalize_genres=True,
        tracks_metadata=tracks_metadata, enriched_resolver=resolver,
        use_graph_genres=(genre_source is GenreArtifactSource.GRAPH),
    )
    X_genre_raw, X_genre_smoothed = build_genre_matrices(genre_lists, vocab, genre_sim_path)

    data["X_genre_raw"] = X_genre_raw
    data["X_genre_smoothed"] = X_genre_smoothed
    data["genre_vocab"] = np.array(vocab, dtype=object)
    np.savez(artifact_path, **data)
    logger.info("Re-baked genre matrices: %d tracks, %d genres", len(track_ids), len(vocab))
    return {"n_tracks": len(track_ids), "n_genres": len(vocab)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_refresh_genre_matrices.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_beat3tower_artifacts.py tests/unit/test_refresh_genre_matrices.py
git commit -m "feat(artifact): refresh_genre_matrices — genre-only NPZ re-bake from authority"
```

---

### Task 8: Worker `handle_refresh_genre_artifact`

**Files:**
- Modify: `src/playlist_gui/worker.py` (new handler + register in `TRACKED_COMMAND_HANDLERS`)
- Test: covered by Task 9 integration; compile check here.

**Interfaces:**
- Produces: command `refresh_genre_artifact`; emits progress, `result` `{"n_tracks","n_genres","backup"}`, `done`.

- [ ] **Step 1: Add the handler (place near `handle_edit_genres`)**

```python
def handle_refresh_genre_artifact(cmd_data: Dict[str, Any]) -> None:
    """Re-bake only the genre matrices in the artifact NPZ from the authority."""
    emit_log("INFO", "Refreshing genre matrices in artifact")
    emit_progress("refresh_genre", 0, 100, "Loading artifact")
    try:
        import shutil
        from datetime import datetime
        from scripts.build_beat3tower_artifacts import refresh_genre_matrices

        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        ds = config.get("playlists", {}).get("ds_pipeline", {})
        artifact_path = ds.get(
            "artifact_path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
        )
        db_path = config.get("library", {}).get("database_path", "data/metadata.db")
        genre_sim_path = ds.get("genre_sim_path") or "data/genre_similarity_graph.npz"

        art = Path(artifact_path)
        if not art.exists():
            raise FileNotFoundError(
                f"artifact not found: {artifact_path} — build artifacts first"
            )
        # Timestamped backup before overwrite (artifact is rebuildable; cheap insurance).
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = art.with_suffix(art.suffix + f".genrebak_{ts}")
        shutil.copy2(art, backup)
        emit_progress("refresh_genre", 30, 100, "Re-baking genre matrices")

        result = refresh_genre_matrices(
            str(art), db_path,
            genre_sim_path=genre_sim_path if Path(genre_sim_path).exists() else None,
            sidecar_db=SIDECAR_DB_PATH, config_path=base_path,
        )
        emit_progress("refresh_genre", 100, 100, "Done")
        emit_result("refresh_genre_artifact", {**result, "backup": str(backup)})
        emit_done("refresh_genre_artifact", True,
                  f"Re-baked {result['n_genres']} genres across {result['n_tracks']} tracks")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("refresh_genre_artifact", False, str(e))
```

- [ ] **Step 2: Register it** in `TRACKED_COMMAND_HANDLERS` (after `"edit_genres": handle_edit_genres,`):

```python
    "refresh_genre_artifact": handle_refresh_genre_artifact,
```

- [ ] **Step 3: Compile check**

Run: `python -c "import ast; ast.parse(open('src/playlist_gui/worker.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/playlist_gui/worker.py
git commit -m "feat(worker): refresh_genre_artifact command (NPZ backup + genre re-bake)"
```

---

## Phase 4 — API

### Task 9: Read endpoints + real-worker integration test

**Files:**
- Modify: `src/playlist_web/app.py` (add `/api/genres/search`, `/api/genres/for_album`)
- Test: `tests/integration/test_web_genre_editing.py`

**Interfaces:**
- Produces: `GET /api/genres/search?q=&limit=` → `{"items":[{"genre_id","name"}]}`; `GET /api/genres/for_album?artist=&album=` → `{"genres":[...]}`.

- [ ] **Step 1: Add the two read endpoints in `create_app`** (near `track_genres`)

```python
    @app.get("/api/genres/search")
    async def genres_search(q: str = "", limit: int = Query(20, ge=1, le=100)) -> dict:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": []}
        from src.genre.authority import canonical_genre_search
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                return {"items": [
                    {"genre_id": gid, "name": name}
                    for gid, name in canonical_genre_search(conn, q, limit)
                ]}
            finally:
                conn.close()
        except sqlite3.Error:
            return {"items": []}

    @app.get("/api/genres/for_album")
    async def genres_for_album(artist: str = "", album: str = "") -> dict:
        if not artist.strip() or not album.strip() or not DB_PATH.exists():
            return {"genres": []}
        from src.genre.authority import display_genre_names_for_album
        from src.genre.genre_edit import album_id_for_release
        from src.genre.granularity import order_genres_for_display
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                album_id = album_id_for_release(conn, artist, album)
                if not album_id:
                    return {"genres": []}
                names = display_genre_names_for_album(conn, album_id)
                return {"genres": order_genres_for_display(names)}
            finally:
                conn.close()
        except sqlite3.Error:
            return {"genres": []}
```

- [ ] **Step 2: Write the real-worker integration test**

```python
# tests/integration/test_web_genre_editing.py
"""Drives the REAL worker subprocess through WorkerBridge via asyncio
(Proactor loop) — NOT TestClient (which can't faithfully read real-worker
stdout on Windows; see the web-gui skill)."""
import asyncio
import shutil
import sqlite3
import sys
from pathlib import Path
import pytest

from src.playlist_web.worker_bridge import WorkerBridge

WORKER_CMD = [sys.executable, "-m", "src.playlist_gui.worker"]


@pytest.mark.integration
@pytest.mark.slow
def test_edit_genres_writes_authority(tmp_path):
    # Copy a tiny metadata.db + sidecar built here; point a temp config at them.
    # (Build the minimal schema the edit path needs — same tables as Task 5.)
    meta = tmp_path / "metadata.db"
    conn = sqlite3.connect(meta)
    conn.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        "CREATE TABLE genre_graph_release_genre_assignments "
        "(album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL);"
        "CREATE TABLE genre_graph_canonical_genres (genre_id TEXT PRIMARY KEY, name TEXT, kind TEXT, "
        " specificity_score REAL, status TEXT, taxonomy_version TEXT);"
        "CREATE TABLE release_effective_genres (album_id TEXT NOT NULL, release_key TEXT, "
        " genre_id TEXT NOT NULL, assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, "
        " source TEXT NOT NULL, PRIMARY KEY (album_id, genre_id, assignment_layer));"
    )
    conn.execute("INSERT INTO tracks VALUES ('t1','The  Radio Dept.','Pet Grief','ORPH1')")
    conn.commit()
    conn.close()

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"library:\n  database_path: {meta.as_posix()}\n"
        "playlists:\n  ds_pipeline:\n    genre_source: graph\n"
    )

    async def run():
        bridge = WorkerBridge(WORKER_CMD)
        await bridge.start()
        try:
            res = await bridge.command({
                "cmd": "edit_genres", "base_config_path": str(cfg),
                "artist": "The  Radio Dept.", "album": "Pet Grief",
                "genres": ["dream pop", "shoegaze"],
            })
            return res
        finally:
            await bridge.stop()

    result = asyncio.run(run())
    assert result.get("no_change") is False
    assert len(result.get("added", [])) == 2
    c = sqlite3.connect(meta)
    rows = c.execute("SELECT genre_id FROM release_effective_genres "
                     "WHERE album_id='ORPH1' AND source='user'").fetchall()
    c.close()
    assert len(rows) == 2
```

(Adjust `WorkerBridge` start/command/stop to the actual method names in `src/playlist_web/worker_bridge.py` — verify against that file before running. SIDECAR_DB_PATH defaults to `data/ai_genre_enrichment.db` relative to cwd; the worker initializes it if absent.)

- [ ] **Step 3: Run the integration test**

Run: `python -m pytest tests/integration/test_web_genre_editing.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/playlist_web/app.py tests/integration/test_web_genre_editing.py
git commit -m "feat(api): genres/search + genres/for_album; real-worker edit_genres integration test"
```

---

### Task 10: `/api/refresh_genre_artifact` + edit response + fake worker

**Files:**
- Modify: `src/playlist_web/app.py` (new endpoint; `/api/edit_genres` already submits — extend response passthrough)
- Modify: `tests/fixtures/fake_worker.py`
- Test: `tests/integration/test_web_api.py` (or a new `test_web_genre_api.py`) via `TestClient`

**Interfaces:**
- Produces: `POST /api/refresh_genre_artifact` → `{"job_id": ...}`; `409` if busy.
- `/api/edit_genres` response: `{"ok": True, "resolved", "unknown", "added", "removed", "no_change"}` (passthrough of the worker result).

- [ ] **Step 1: Add the endpoint in `create_app`**

```python
    @app.post("/api/refresh_genre_artifact")
    async def refresh_genre_artifact() -> dict:
        job_id = registry.create("refresh_genre_artifact")
        try:
            await bridge.submit({"cmd": "refresh_genre_artifact", "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409,
                                detail="A job is in progress — try again when it finishes.")
        return {"job_id": job_id}
```

(Match `registry.create(...)` / `bridge.submit(...)` to the signatures used by the existing tracked endpoints in `app.py`, e.g. the analyze/enrich handlers.)

- [ ] **Step 2: Extend `/api/edit_genres` to return the worker result fields**

The endpoint already does `result = await bridge.command({...})`. Ensure the return is:

```python
        return {"ok": True, **result}
```

so `resolved`/`unknown`/`added`/`removed`/`no_change` pass through.

- [ ] **Step 3: Add fake-worker branches** in `tests/fixtures/fake_worker.py` (before the `else` fallback). For `edit_genres`:

```python
        elif cmd == "edit_genres":
            emit("result", result_type="edit_genres", data={
                "artist": data.get("artist", ""), "album": data.get("album", ""),
                "resolved": data.get("genres", []), "unknown": [],
                "added": data.get("genres", []), "removed": [], "no_change": False,
            })
            emit("done", cmd="edit_genres", success=True)
        elif cmd == "refresh_genre_artifact":
            emit("result", result_type="refresh_genre_artifact",
                 data={"n_tracks": 1, "n_genres": 1, "backup": "x.npz.bak"})
            emit("done", cmd="refresh_genre_artifact", success=True)
```

(Mirror the exact `emit(...)` helper shape already used in `fake_worker.py`.)

- [ ] **Step 4: Write the API test**

```python
# tests/integration/test_web_genre_api.py
import sys
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_edit_genres_passes_through_resolved():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        r = client.post("/api/edit_genres", json={
            "artist": "The Radio Dept.", "album": "Pet Grief",
            "genres": ["dream pop"], "base_genres": []})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["resolved"] == ["dream pop"]
        assert body["no_change"] is False


def test_refresh_genre_artifact_returns_job():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        r = client.post("/api/refresh_genre_artifact")
        assert r.status_code == 200
        assert "job_id" in r.json()
```

- [ ] **Step 5: Run the API tests**

Run: `python -m pytest tests/integration/test_web_genre_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/app.py tests/fixtures/fake_worker.py tests/integration/test_web_genre_api.py
git commit -m "feat(api): refresh_genre_artifact endpoint + edit_genres result passthrough + fakes"
```

---

## Phase 5 — Frontend

### Task 11: Types + API client

**Files:**
- Modify: `web/src/lib/types.ts` (~117-121)
- Modify: `web/src/lib/api.ts`

**Interfaces:**
- Produces: `CanonicalGenre {genre_id; name}`; `EditGenresResponse {ok; resolved; unknown; added; removed; no_change}`; `api.genresSearch`, `api.albumGenres`, `api.refreshGenreArtifact`.

- [ ] **Step 1: Add types** in `web/src/lib/types.ts` after `EditGenresRequest`:

```typescript
export interface CanonicalGenre {
  genre_id: string;
  name: string;
}

export interface EditGenresResponse {
  ok: boolean;
  resolved: string[];
  unknown: string[];
  added: string[];
  removed: string[];
  no_change: boolean;
}
```

- [ ] **Step 2: Update `api.ts`** — change the `editGenres` return type and add three methods (place near `editGenres`):

```typescript
  async editGenres(req: EditGenresRequest): Promise<EditGenresResponse> {
    return jsonOrThrow(await fetch("/api/edit_genres", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async genresSearch(q: string, limit = 20): Promise<{ items: CanonicalGenre[] }> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/genres/search?${params}`));
  },
  async albumGenres(artist: string, album: string): Promise<{ genres: string[] }> {
    const params = new URLSearchParams({ artist, album });
    return jsonOrThrow(await fetch(`/api/genres/for_album?${params}`));
  },
  async refreshGenreArtifact(): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/refresh_genre_artifact", { method: "POST" }));
  },
```

Add `CanonicalGenre, EditGenresResponse` to the `import type { ... } from "./types"` block.

- [ ] **Step 3: Type-check**

Run: `npm --prefix web run build`
Expected: build succeeds (tsc passes).

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(web): genre editing types + api client (search, for_album, refresh)"
```

---

### Task 12: `EditGenresDialog` — autocomplete, pending-input commit, warnings

**Files:**
- Modify: `web/src/components/EditGenresDialog.tsx`

**Interfaces:**
- Consumes: `api.genresSearch`, `api.albumGenres`, `api.editGenres` (→ `EditGenresResponse`).

- [ ] **Step 1: Rewrite the dialog logic**

Key changes (keep the existing layout/classes):
1. On open, fetch current authoritative genres and seed chips:

```tsx
  useEffect(() => {
    if (!props.open) return;
    setInput(""); setErr(null); setUnknown([]);
    setGenres([...props.initialGenres]);
    api.albumGenres(props.artist, props.album)
      .then((r) => { if (r.genres.length) setGenres(r.genres); })
      .catch(() => {});
  }, [props.open, props.artist, props.album, props.initialGenres]);
```

2. Add autocomplete state + debounced search:

```tsx
  const [suggestions, setSuggestions] = useState<CanonicalGenre[]>([]);
  const [unknown, setUnknown] = useState<string[]>([]);
  useEffect(() => {
    const q = input.trim();
    if (!q) { setSuggestions([]); return; }
    const h = setTimeout(() => {
      api.genresSearch(q, 8).then((r) => setSuggestions(r.items)).catch(() => setSuggestions([]));
    }, 150);
    return () => clearTimeout(h);
  }, [input]);
```

3. Commit a chip by name (used by Enter, click-suggestion, and Save-flush):

```tsx
  const addGenre = (raw?: string) => {
    const g = (raw ?? input).trim();
    if (g && !genres.some((x) => x.toLowerCase() === g.toLowerCase())) {
      setGenres([...genres, g]);
    }
    setInput(""); setSuggestions([]);
  };
```

4. **Flush pending input on Save** (kills the empty-override trap), then send the chips:

```tsx
  const save = async () => {
    const pending = input.trim();
    const finalGenres = pending && !genres.some((x) => x.toLowerCase() === pending.toLowerCase())
      ? [...genres, pending] : genres;
    setSaving(true); setErr(null); setUnknown([]);
    try {
      const res = await api.editGenres({
        artist: props.artist, album: props.album,
        genres: finalGenres, base_genres: props.initialGenres,
      });
      if (res.unknown.length) setUnknown(res.unknown);
      props.onSaved(props.artist, props.album, res.resolved);
      if (!res.unknown.length) props.onOpenChange(false);
    } catch (e) { setErr(String(e)); } finally { setSaving(false); }
  };
```

5. Render the suggestion dropdown under the input and an unknown-terms warning:

```tsx
  {suggestions.length > 0 && (
    <div className="mt-1 bg-panel2 border border-border rounded-md max-h-40 overflow-auto">
      {suggestions.map((s) => (
        <div key={s.genre_id} onClick={() => addGenre(s.name)}
             className="px-2.5 py-1 text-[11px] text-text hover:bg-border cursor-pointer">
          {s.name}
        </div>
      ))}
    </div>
  )}
  {unknown.length > 0 && (
    <div className="text-danger text-[11px] mt-2">
      Not in the genre vocabulary (not saved): {unknown.join(", ")}
    </div>
  )}
```

Import `CanonicalGenre` from `../lib/types`. Keep the `onSaved` prop signature `(artist, album, genres)`.

- [ ] **Step 2: Build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/EditGenresDialog.tsx
git commit -m "feat(web): EditGenresDialog autocomplete + pending-input commit + unknown warning"
```

---

### Task 13: `App.tsx` — authoritative update + Refresh button

**Files:**
- Modify: `web/src/App.tsx` (`applyGenreEdit` ~157-164; add a refresh affordance)

- [ ] **Step 1: `applyGenreEdit` already receives the authoritative `resolved` list** from `onSaved` (Task 12 passes `res.resolved`). No change needed beyond confirming it maps tracks by (artist, album). Add a post-save refresh button. Add state + handler:

```tsx
  const [refreshing, setRefreshing] = useState(false);
  const refreshGenres = useCallback(async () => {
    setRefreshing(true);
    try { await api.refreshGenreArtifact(); }
    catch (e) { console.error(e); }
    finally { setRefreshing(false); }
  }, []);
```

- [ ] **Step 2: Render the button** near the playlist actions (e.g. beside export), wired to `refreshGenres`, disabled while `refreshing`:

```tsx
  {playlist && (
    <button onClick={refreshGenres} disabled={refreshing}
      title="Re-bake genre vectors so generation reflects your edits"
      className="border border-border text-muted text-xs px-3 py-1.5 rounded disabled:opacity-50">
      {refreshing ? "Refreshing…" : "Refresh genres for generation"}
    </button>
  )}
```

(WS progress already flows through `useWorkerEvents`; the button is fire-and-forget. If a generation is running, the API returns 409 — surface it via the existing error toast if present.)

- [ ] **Step 3: Build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/App.tsx
git commit -m "feat(web): authoritative genre update + Refresh-genres-for-generation button"
```

---

### Task 14: Playwright e2e (fake worker)

**Files:**
- Create: `web/tests/edit-genres.spec.ts`

- [ ] **Step 1: Write the spec** (mirror existing specs in `web/tests/`; the Playwright config points the worker at the fake via `PG_WEB_WORKER_CMD`):

```typescript
import { test, expect } from "@playwright/test";

test("edit genres: autocomplete, save, refresh", async ({ page }) => {
  await page.goto("/");
  // (Generate or load a playlist per the existing spec helpers, then:)
  // right-click a track row → "Edit genres for album"
  // type into [data-testid=genre-input], pick a suggestion, Save
  // assert the dialog closed and a chip is present
  // click "Refresh genres for generation" and assert no error toast
});
```

(Fill in the navigation/seed steps by copying the pattern from an existing `web/tests/*.spec.ts`; the fake worker returns `resolved = genres`.)

- [ ] **Step 2: Run Playwright**

Run: `npm --prefix web run test:e2e -- edit-genres` (or the project's Playwright script)
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add web/tests/edit-genres.spec.ts
git commit -m "test(web): playwright e2e for genre editing + refresh"
```

---

## Phase 6 — Full verification (from the real checkout)

### Task 15: Build, restart, manual e2e on Pet Grief, full sweep

**Files:** none (verification)

- [ ] **Step 1: Full fast test suite**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS (quote real counts). Then run the slow genre/integration tests explicitly:
`python -m pytest tests/integration/test_web_genre_editing.py tests/unit/test_refresh_genre_matrices.py -q`

- [ ] **Step 2: Lint/types**

Run: `ruff check src/genre src/playlist_gui src/playlist_web scripts/build_beat3tower_artifacts.py` and `mypy src/genre/genre_edit.py src/genre/genre_publish.py`
Expected: clean (or pre-existing-only).

- [ ] **Step 3: Build front-end + restart server**

From the real checkout (DBs/artifact live there): `npm --prefix web run build`, then restart `python tools/serve_web.py`.

- [ ] **Step 4: Manual e2e on Pet Grief**

Generate a playlist containing The Radio Dept. – Pet Grief (or stage it as a seed). Right-click → "Edit genres for album" → add e.g. "dream pop", "shoegaze" via autocomplete → Save. Confirm: chips show immediately; reopen the dialog and the genres persist (read from authority); click "Refresh genres for generation"; regenerate; confirm Pet Grief still shows the genres and they participate in generation. Verify in DB:

```sql
SELECT genre_id, source FROM release_effective_genres WHERE album_id='1e37941b7875c46c';
```
Expected: the user genres present with `source='user'`.

- [ ] **Step 5: Finalize** (per superpowers:finishing-a-development-branch — merge/PR decision with the user).

---

## Self-review

- **Spec coverage:** Authority-correction depth → Tasks 5/6 (surgical write). Durable override → Task 5. Orphan handling → Tasks 2/4. Taxonomy-constrained autocomplete → Tasks 3/9/12. One-click re-bake → Tasks 7/8/13. Empty-override fix → Task 12 (pending-input flush) + Task 5 (no-op). Free-text-drop fix → Task 4 (unknowns reported) + Task 12 (warning). Generation reads authority chain → Task 7. Backup policy (regenerability) → no per-edit DB backup (Task 6); NPZ backup → Task 8. Tests → Tasks 1,2,5,7,9,10,14. All spec sections map to tasks.
- **Placeholder scan:** Frontend tasks reference "copy the existing spec helper" only for Playwright navigation (inherently project-specific); all backend/API code is complete. No TBD/TODO in code steps.
- **Type consistency:** `materialize_album_genres` signature identical in Tasks 1/5. `EditResult` fields (`resolved/unknown/added/removed/no_change`) consistent across Tasks 5/6/10/11/12. `apply_user_genre_edit(meta_conn, sidecar_store, taxonomy, *, artist, album, target_names)` consistent Tasks 5/6. `refresh_genre_matrices(artifact_path, db_path, *, genre_sim_path, sidecar_db, config_path)` consistent Tasks 7/8. API field names (`resolved/unknown/no_change`) consistent Tasks 10/11/12.

## Notes / verify-before-coding

- Confirm `WorkerBridge` method names (`start`/`command`/`submit`/`stop`) against `src/playlist_web/worker_bridge.py` before Task 9.
- Confirm `registry.create(...)` / `bridge.submit(...)` signatures against an existing tracked endpoint in `app.py` before Task 10.
- Confirm `genre_publish._split` and `_WEIGHT_TRACK/_WEIGHT_ALBUM/_WEIGHT_ARTIST` names before Task 2.
- Confirm `make_resolver` / `GenreArtifactSource` import path (`src/ai_genre_enrichment/artifact_modes.py`) and the genre similarity NPZ path (`data/genre_similarity_graph.npz` vs config `genre_sim_path`) before Task 7/8.
- Confirm `tests/fixtures/fake_worker.py` emit helper shape before Task 10.
