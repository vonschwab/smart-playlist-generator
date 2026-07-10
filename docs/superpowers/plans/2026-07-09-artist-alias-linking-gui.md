# Artist-Alias Linking — "Artist Links" GUI Panel Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Artist Links" management sub-tab to the browser GUI so the user creates/edits/removes artist alias & sibling links (writing `data/artist_aliases.yaml`) without hand-editing YAML — driving the engine shipped in Plan 1.

**Architecture:** New sub-tab in the Advanced rail → `ArtistLinksPanel` (React) → `web/src/lib/api.ts` → FastAPI routes (`src/playlist_web/app.py`) → **untracked** NDJSON worker commands (`src/playlist_gui/worker.py`) → read/write `data/artist_aliases.yaml` via helpers in the Plan-1 resolver (`src/playlist/artist_aliases.py`), busting its cache on write. Members are entered via a new `ArtistAutocomplete` over a new `/api/artists/search` (distinct library artists). Single-phase: Save validates + writes immediately, no adjudication step.

**Tech Stack:** React + TS + Vite + Tailwind (frontend), FastAPI + NDJSON WorkerBridge (backend), pytest + vitest.

**Spec:** `docs/superpowers/specs/2026-07-09-artist-alias-linking-design.md` (GUI = Section D). **Depends on** Plan 1 (engine, already shipped on this branch): resolver exposes `resolve_alias`, `sibling_group_of`, `build_artist_link_map(groups)`, `clear_cache()`, `_DEFAULT_ALIAS_PATH`.

## Global Constraints

- **Untracked worker handlers MUST stamp `request_id` (from `cmd_data`) and `job_id=None` on EVERY emitted event**, using `emit_event({...})` directly — NEVER the auto-stamping `emit_result`/`emit_done` helpers (they inherit a running job's ids and corrupt it). Every code path emits a terminal `{"type":"done", ...}`.
- **Route error mapping:** a failing READ maps `WorkerCommandError`→**502**; a failing WRITE/validation maps `WorkerCommandError`→**422**. Both catch `BridgeBusy`→**409**. (Matches `/api/review/*`, `/api/taxonomy/*`.)
- **Add a `tests/fixtures/fake_worker.py` branch for each new command**, before the final `else`, emitting `result` then `done` with `"job_id": None` (untracked convention).
- **YAML writes:** `yaml.safe_dump(data, sort_keys=False, allow_unicode=True)`; timestamped `shutil.copy2` backup before overwriting an existing file; call `artist_aliases.clear_cache()` after writing. No `metadata.db`/artifact writes anywhere (artist search is read-only).
- **UI discipline (`docs/UI_UX_DISCIPLINE.md`):** primary controls ≥ 44×44 px (nothing interactive < 24×24); inputs ≥ 16px font on touch; theme tokens only (no raw hex, no `text-[Npx]` arbitrary type, no `text-faint` on body); every panel has empty / loading / error states; `:focus-visible` rings; the Advanced rail now has **5** tabs (still ≤ 5 — OK). Give interactive elements `data-testid`s.
- **Build/restart:** after any `web/src` edit, `npm --prefix web run build` (emits `web/dist`); after any `worker.py`/`app.py` edit, `serve_web.py` must be restarted for a live check. The web-gui traps (stale `dist`, unrestarted worker, silently-dropped result) apply.
- **Shared canonical checkout — commit explicit paths only.** `git add <paths>` then `git commit --only -- <paths>`; verify `git diff --cached --name-only` first. NEVER `git add -A`/`-u`/`.` or bare `git commit`. The tree has UNRELATED other-session files (`CLAUDE.md`, `data/layered_genre_taxonomy.yaml`, `docs/superpowers/plans/2026-07-04-*`, untracked docs) — do NOT stage, edit, or touch them. Branch: `feat/artist-alias-linking`.
- **Tests bounded, never piped:** `python -m pytest -q <path>` (no `| tail`/`| head`); `npm --prefix web run test`, `npm --prefix web run lint`, and `tsc -b` (run from `web/`).

---

## File Structure

- **Modify** `src/playlist/artist_aliases.py` — add `read_artist_link_groups`, `save_artist_link_groups`, `validate_artist_link_groups` (file I/O + save-time validation).
- **Modify** `src/playlist_gui/worker.py` — `handle_list_artist_links`, `handle_save_artist_links` + register in `UNTRACKED_COMMAND_HANDLERS`.
- **Modify** `tests/fixtures/fake_worker.py` — branches for the two commands.
- **Modify** `src/playlist_web/schemas.py` — `ArtistLinkGroup`, `ArtistLinksSaveRequest`.
- **Modify** `src/playlist_web/app.py` — `GET /api/artists/links`, `POST /api/artists/links/save`, `GET /api/artists/search`.
- **Modify** `src/metadata_client.py` — `search_artists(query, limit)`.
- **Modify** `web/src/lib/types.ts` — link + artist-search interfaces.
- **Modify** `web/src/lib/api.ts` — `artistLinksList`, `artistLinksSave`, `artistsSearch`.
- **Create** `web/src/components/ArtistAutocomplete.tsx` (+ `.test.tsx`) — mirrors `GenreAutocomplete.tsx`.
- **Create** `web/src/components/ArtistLinksPanel.tsx` (+ `.test.tsx`).
- **Modify** `web/src/components/AdvancedPanel.tsx` — 5th tab.
- **Test** `tests/integration/test_artist_links_api.py` — route tests via fake worker.

---

## Task 1: Resolver file-I/O + save-time validation

**Files:**
- Modify: `src/playlist/artist_aliases.py`
- Test: `tests/unit/test_artist_aliases.py` (append)

**Interfaces — Produces:**
- `validate_artist_link_groups(groups: list) -> list[str]` — human-readable errors (empty list = valid).
- `read_artist_link_groups(path: Optional[Path] = None) -> list[dict]` — the raw `groups` list from the YAML (`[]` if absent/malformed).
- `save_artist_link_groups(groups: list, path: Optional[Path] = None) -> None` — validate (raise `ValueError("; ".join(errors))` if invalid), timestamped backup, `yaml.safe_dump`, then `clear_cache()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_artist_aliases.py`)

```python
def test_validate_artist_link_groups_flags_bad_input():
    from src.playlist.artist_aliases import validate_artist_link_groups
    errs = validate_artist_link_groups([
        {"type": "alias", "members": ["Solo"]},          # <2 members
        {"type": "bogus", "members": ["A", "B"]},         # bad type
        {"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]},
        {"type": "sibling", "members": ["Alex G", "X"]},  # Alex G reused across groups
    ])
    assert len(errs) == 3
    assert validate_artist_link_groups([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}]) == []


def test_save_and_read_round_trip(tmp_path):
    from src.playlist.artist_aliases import save_artist_link_groups, read_artist_link_groups
    p = tmp_path / "artist_aliases.yaml"
    groups = [{"type": "sibling", "members": ["Smog", "Bill Callahan"]}]
    save_artist_link_groups(groups, path=p)
    assert read_artist_link_groups(path=p) == groups
    assert read_artist_link_groups(path=tmp_path / "missing.yaml") == []


def test_save_rejects_invalid_and_backs_up(tmp_path):
    import pytest
    from src.playlist.artist_aliases import save_artist_link_groups, read_artist_link_groups
    p = tmp_path / "artist_aliases.yaml"
    save_artist_link_groups([{"type": "alias", "members": ["A", "B"]}], path=p)
    with pytest.raises(ValueError):
        save_artist_link_groups([{"type": "alias", "members": ["OnlyOne"]}], path=p)
    # original file unchanged; a .bak.* backup exists from the second-write attempt? No:
    # invalid input raises BEFORE any write, so no backup and the good file is intact.
    assert read_artist_link_groups(path=p) == [{"type": "alias", "members": ["A", "B"]}]
    # a valid overwrite creates a timestamped backup of the prior file
    save_artist_link_groups([{"type": "sibling", "members": ["X", "Y"]}], path=p)
    assert list(p.parent.glob("artist_aliases.yaml.bak.*"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q tests/unit/test_artist_aliases.py -k "validate_artist_link or round_trip or rejects_invalid"`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Implement the three helpers** in `src/playlist/artist_aliases.py`

Add these imports at the top (module currently imports `logging`, `dataclass`/`field`, `lru_cache`, `Path`, typing, `yaml`):

```python
import datetime
import shutil
```

Add the functions (place after `build_artist_link_map`, before `_cached_load`):

```python
def validate_artist_link_groups(groups: list) -> List[str]:
    """Save-time validation. Returns human-readable error strings ([] == valid).

    Unlike build_artist_link_map (which logs + skips for a resilient load), this
    surfaces problems so the GUI can reject a bad payload.
    """
    norm_struct, norm_sem = _normalizers()
    errors: List[str] = []
    owner: Dict[str, int] = {}
    for gi, group in enumerate(groups or []):
        label = f"group {gi + 1}"
        if not isinstance(group, dict):
            errors.append(f"{label}: not a mapping")
            continue
        gtype = str(group.get("type", "")).strip().lower()
        raw = group.get("members")
        names = [str(m).strip() for m in raw if str(m).strip()] if isinstance(raw, list) else []
        if gtype not in VALID_TYPES:
            errors.append(f"{label}: type must be 'alias' or 'sibling' (got {group.get('type')!r})")
        if len(names) < 2:
            errors.append(f"{label}: needs at least 2 members")
            continue
        for n in names:
            for form in {norm_struct(n), norm_sem(n)}:
                if not form:
                    continue
                prev = owner.get(form)
                if prev is not None and prev != gi:
                    errors.append(f"{label}: '{n}' is already linked in group {prev + 1}")
                owner[form] = gi
    return errors


def read_artist_link_groups(path: Optional[Path] = None) -> List[dict]:
    """Return the raw `groups` list from the YAML ([] if absent/unreadable)."""
    p = Path(path) if path is not None else _DEFAULT_ALIAS_PATH
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("artist_aliases: failed to read %s: %s", p, exc)
        return []
    groups = data.get("groups") if isinstance(data, dict) else None
    return list(groups or [])


def save_artist_link_groups(groups: list, path: Optional[Path] = None) -> None:
    """Validate, back up any existing file, write the YAML, and bust the cache.

    Raises ValueError (joined error messages) if the groups are invalid — nothing
    is written in that case.
    """
    errors = validate_artist_link_groups(groups)
    if errors:
        raise ValueError("; ".join(errors))
    p = Path(path) if path is not None else _DEFAULT_ALIAS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(p, p.with_name(f"{p.name}.bak.{ts}"))
    payload = {"version": 1, "groups": list(groups)}
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    clear_cache()
```

Note: `datetime.datetime.now()` is fine here (this is production code, not a workflow script).

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q tests/unit/test_artist_aliases.py`
Expected: PASS (all, including the new 3).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/playlist/artist_aliases.py && mypy src/playlist/artist_aliases.py
git add src/playlist/artist_aliases.py tests/unit/test_artist_aliases.py
git commit --only -m "feat(artist-links): resolver read/save/validate helpers for the GUI" -- src/playlist/artist_aliases.py tests/unit/test_artist_aliases.py
```

---

## Task 2: Worker command handlers (list + save)

**Files:**
- Modify: `src/playlist_gui/worker.py`
- Modify: `tests/fixtures/fake_worker.py`

**Interfaces:**
- Consumes: `read_artist_link_groups`, `save_artist_link_groups` (Task 1).
- Produces: NDJSON commands `list_artist_links` → `{result_type:"artist_links", groups:[...]}`; `save_artist_links` (payload `groups`) → `{result_type:"artist_links_saved", count:N}` or a `WorkerCommandError`-surfacing `done ok=False` on validation failure.

- [ ] **Step 1: Add fake_worker branches (the test double first)**

In `tests/fixtures/fake_worker.py`, add before the final `else` (~line 209):

```python
        elif name == "list_artist_links":
            emit({"type": "result", "result_type": "artist_links",
                  "groups": [{"type": "sibling", "members": ["Smog", "Bill Callahan"]}],
                  "request_id": rid, "job_id": None})
            emit({"type": "done", "cmd": "list_artist_links", "ok": True,
                  "request_id": rid, "job_id": None})
        elif name == "save_artist_links":
            groups = cmd.get("groups") or []
            # mimic the real validation: a group with <2 members is rejected
            bad = any(len([m for m in (g.get("members") or []) if str(m).strip()]) < 2 for g in groups)
            if bad:
                emit({"type": "error", "message": "group needs at least 2 members",
                      "request_id": rid, "job_id": None})
                emit({"type": "done", "cmd": "save_artist_links", "ok": False,
                      "detail": "invalid", "request_id": rid, "job_id": None})
            else:
                emit({"type": "result", "result_type": "artist_links_saved",
                      "count": len(groups), "request_id": rid, "job_id": None})
                emit({"type": "done", "cmd": "save_artist_links", "ok": True,
                      "request_id": rid, "job_id": None})
```

- [ ] **Step 2: Write the failing route-less handler check** — deferred to Task 3's route tests (the handlers are exercised end-to-end there via the real worker path is heavy; here we rely on Task 3's `test_web_api`-style tests against the FAKE worker, plus a direct import smoke). Add a direct smoke test now in `tests/integration/test_artist_links_api.py` (created in Task 3) — SKIP writing it here; instead confirm the fake branch parses:

Run: `python -c "import ast; ast.parse(open('tests/fixtures/fake_worker.py',encoding='utf-8').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Implement the real handlers** in `src/playlist_gui/worker.py`

Add both functions just after `handle_apply_escalation_decision` (~line 2791) — mirroring its untracked emit pattern exactly:

```python
def handle_list_artist_links(cmd_data: Dict[str, Any]) -> None:
    """Return the current artist link groups from data/artist_aliases.yaml. UNTRACKED (read)."""
    rid = cmd_data.get("request_id")
    try:
        from src.playlist.artist_aliases import read_artist_link_groups
        groups = read_artist_link_groups()
        emit_event({"type": "result", "result_type": "artist_links",
                    "groups": groups, "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "list_artist_links", "ok": True,
                    "detail": f"{len(groups)} group(s)", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "list_artist_links", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_save_artist_links(cmd_data: Dict[str, Any]) -> None:
    """Validate + write artist link groups, bust the resolver cache. UNTRACKED (quick write).

    On invalid input, emits done ok=False with the validation detail so the route
    surfaces it as a 422 (WorkerCommandError)."""
    rid = cmd_data.get("request_id")
    try:
        from src.playlist.artist_aliases import save_artist_link_groups
        groups = cmd_data.get("groups") or []
        try:
            save_artist_link_groups(groups)
        except ValueError as ve:
            emit_event({"type": "error", "message": str(ve), "request_id": rid, "job_id": None})
            emit_event({"type": "done", "cmd": "save_artist_links", "ok": False,
                        "detail": str(ve), "request_id": rid, "job_id": None})
            return
        emit_event({"type": "result", "result_type": "artist_links_saved",
                    "count": len(groups), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "save_artist_links", "ok": True,
                    "detail": f"saved {len(groups)} group(s)", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "save_artist_links", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})
```

Register both in `UNTRACKED_COMMAND_HANDLERS` (the dict at ~line 3147), adding after `handle_apply_escalation_decision`'s entry:

```python
    "list_artist_links": handle_list_artist_links,
    "save_artist_links": handle_save_artist_links,
```

- [ ] **Step 4: Verify worker imports cleanly**

Run: `python -c "import src.playlist_gui.worker as w; assert 'list_artist_links' in w.UNTRACKED_COMMAND_HANDLERS and 'save_artist_links' in w.UNTRACKED_COMMAND_HANDLERS; print('registered')"`
Expected: `registered`.

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/playlist_gui/worker.py tests/fixtures/fake_worker.py
git add src/playlist_gui/worker.py tests/fixtures/fake_worker.py
git commit --only -m "feat(artist-links): untracked worker handlers list/save + fake branches" -- src/playlist_gui/worker.py tests/fixtures/fake_worker.py
```

---

## Task 3: FastAPI schemas + link routes (with route tests)

**Files:**
- Modify: `src/playlist_web/schemas.py`
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_artist_links_api.py` (create)

**Interfaces:**
- `GET /api/artists/links` → `{groups: [...]}`.
- `POST /api/artists/links/save` body `ArtistLinksSaveRequest{groups}` → `{ok: true, count: N}`; validation failure → HTTP 422.

- [ ] **Step 1: Write the failing route tests**

Create `tests/integration/test_artist_links_api.py`:

```python
import sys
import pytest
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


@pytest.mark.integration
def test_list_artist_links_returns_groups():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/artists/links")
        assert resp.status_code == 200
        assert resp.json()["groups"] == [{"type": "sibling", "members": ["Smog", "Bill Callahan"]}]


@pytest.mark.integration
def test_save_artist_links_ok():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/artists/links/save", json={
            "groups": [{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}]})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["count"] == 1


@pytest.mark.integration
def test_save_artist_links_rejects_invalid_with_422():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/artists/links/save", json={
            "groups": [{"type": "alias", "members": ["OnlyOne"]}]})
        assert resp.status_code == 422
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q tests/integration/test_artist_links_api.py`
Expected: FAIL (404 — routes don't exist).

- [ ] **Step 3: Add the schemas**

In `src/playlist_web/schemas.py`, after `TaxonomyValidateRequest` (~line 321):

```python
class ArtistLinkGroup(BaseModel):
    type: str  # "alias" | "sibling" — validated by the worker
    members: list[str]


class ArtistLinksSaveRequest(BaseModel):
    groups: list[ArtistLinkGroup]
```

- [ ] **Step 4: Add the routes**

In `src/playlist_web/app.py`, add `ArtistLinksSaveRequest` to the `from .schemas import (...)` block (lines 21-38). Add these routes in the nested-function block near the taxonomy routes (~after line 396):

```python
    @app.get("/api/artists/links")
    async def artist_links_list() -> dict:
        try:
            return await bridge.command({"cmd": "list_artist_links"}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/artists/links/save")
    async def artist_links_save(body: ArtistLinksSaveRequest) -> dict:
        groups = [g.model_dump() for g in body.groups]
        try:
            result = await bridge.command({"cmd": "save_artist_links", "groups": groups}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest -q tests/integration/test_artist_links_api.py`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/playlist_web/app.py src/playlist_web/schemas.py
git add src/playlist_web/app.py src/playlist_web/schemas.py tests/integration/test_artist_links_api.py
git commit --only -m "feat(artist-links): FastAPI list/save routes + schemas + route tests" -- src/playlist_web/app.py src/playlist_web/schemas.py tests/integration/test_artist_links_api.py
```

---

## Task 4: Artist typeahead backend (`/api/artists/search`)

**Files:**
- Modify: `src/metadata_client.py`
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_artist_links_api.py` (append)

**Interfaces:**
- `GET /api/artists/search?q=<str>&limit=20` → `{items: [<artist name>, ...]}` (distinct library artists matching, case-insensitive).

- [ ] **Step 1: Read the existing pattern to mirror**

READ `src/playlist_web/app.py` lines ~500-517 (`/api/genres/search`) to see exactly how it opens a read-only SQLite connection and resolves the DB path, and READ how `src/metadata_client.py` methods query `tracks`. Mirror that DB-access style precisely (do NOT invent a new connection pattern).

- [ ] **Step 2: Write the failing test** (append to `tests/integration/test_artist_links_api.py`)

```python
@pytest.mark.integration
def test_artists_search_returns_items():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/artists/search", params={"q": "", "limit": 5})
        assert resp.status_code == 200
        assert isinstance(resp.json()["items"], list)
```

(Note: this asserts shape/200 against whatever DB the app resolves; a fuller assertion needs a seeded DB — the route logic is thin and mirrors the proven `/api/genres/search`. If the app's default DB is absent in CI, mark this test to skip when the DB is missing, matching how other DB-dependent route tests guard.)

- [ ] **Step 3: Add the metadata_client method**

In `src/metadata_client.py`, add a method mirroring existing query methods (use the existing connection helper the class uses):

```python
def search_artists(self, query: str = "", limit: int = 20) -> list[str]:
    """Distinct library artist names matching `query` (case-insensitive), alphabetical."""
    like = f"%{query.strip()}%"
    rows = self._conn.execute(
        "SELECT DISTINCT artist FROM tracks WHERE artist LIKE ? COLLATE NOCASE "
        "ORDER BY artist LIMIT ?",
        (like, int(limit)),
    ).fetchall()
    return [str(r[0]) for r in rows if r[0]]
```

(Adapt `self._conn` / cursor usage to the class's actual connection accessor seen in Step 1.)

- [ ] **Step 4: Add the route** in `src/playlist_web/app.py`, mirroring `/api/genres/search`'s DB-access + resolution:

```python
    @app.get("/api/artists/search")
    async def artists_search(q: str = "", limit: int = 20) -> dict:
        # Mirror /api/genres/search: read-only DB access resolved the same way.
        from src.metadata_client import MetadataClient
        from src.config_loader import resolve_database_path, load_config
        db_path = resolve_database_path(load_config(config_path))
        client = MetadataClient(db_path)
        try:
            return {"items": client.search_artists(q, limit)}
        finally:
            client.close()
```

(In Step 1 you confirmed the ACTUAL helpers `/api/genres/search` uses to get `db_path` and open the DB — use those exact ones here; the import names above are the expected ones but VERIFY against genres/search and adjust to match, including how `config_path` is in scope.)

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest -q tests/integration/test_artist_links_api.py`
Expected: PASS (4 tests; the search test 200s or skips-if-no-DB).

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/metadata_client.py src/playlist_web/app.py
git add src/metadata_client.py src/playlist_web/app.py tests/integration/test_artist_links_api.py
git commit --only -m "feat(artist-links): /api/artists/search distinct-artist typeahead backend" -- src/metadata_client.py src/playlist_web/app.py tests/integration/test_artist_links_api.py
```

---

## Task 5: Frontend API client + types

**Files:**
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/lib/api.ts`

**Interfaces — Produces (consumed by Tasks 6-7):**
- Types: `ArtistLinkGroup{ type: "alias"|"sibling"; members: string[] }`, `ArtistLinksListResponse{ groups: ArtistLinkGroup[] }`, `ArtistLinksSaveRequest{ groups: ArtistLinkGroup[] }`, `ArtistSearchResponse{ items: string[] }`.
- api methods: `artistLinksList(): Promise<ArtistLinksListResponse>`, `artistLinksSave(req): Promise<{ok: boolean; count: number}>`, `artistsSearch(q: string, limit?: number): Promise<ArtistSearchResponse>`.

- [ ] **Step 1: Add the types** — in `web/src/lib/types.ts`, at end of file:

```ts
export interface ArtistLinkGroup {
  type: "alias" | "sibling";
  members: string[];
}
export interface ArtistLinksListResponse {
  groups: ArtistLinkGroup[];
}
export interface ArtistLinksSaveRequest {
  groups: ArtistLinkGroup[];
}
export interface ArtistSearchResponse {
  items: string[];
}
```

- [ ] **Step 2: Add the api methods** — in `web/src/lib/api.ts`, add the type names to the `import type {...} from "./types"` block, and insert before the closing `};` of the `api` object (mirroring `reviewQueue`/`reviewDecision` at lines 132-146):

```ts
  async artistLinksList(): Promise<ArtistLinksListResponse> {
    return jsonOrThrow(await fetch("/api/artists/links"));
  },
  async artistLinksSave(req: ArtistLinksSaveRequest): Promise<{ ok: boolean; count: number }> {
    return jsonOrThrow(await fetch("/api/artists/links/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async artistsSearch(q: string, limit = 20): Promise<ArtistSearchResponse> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/artists/search?${params}`));
  },
```

- [ ] **Step 3: Typecheck + lint** (from `web/`)

Run: `cd web && npx tsc -b && npm run lint`
Expected: clean (no new errors).

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit --only -m "feat(artist-links): frontend api client + types" -- web/src/lib/types.ts web/src/lib/api.ts
```

---

## Task 6: `ArtistAutocomplete` component

**Files:**
- Create: `web/src/components/ArtistAutocomplete.tsx`
- Create: `web/src/components/ArtistAutocomplete.test.tsx`

**Interfaces — Produces:** `<ArtistAutocomplete onPick={(name: string) => void} placeholder?: string />`.

- [ ] **Step 1: Read the reference** — READ `web/src/components/GenreAutocomplete.tsx` in full. Build `ArtistAutocomplete` as a close structural copy: a text `<input>` (≥16px font, `data-testid="artist-autocomplete-input"`), 150ms debounced call to `api.artistsSearch(query)`, a suggestions dropdown (each row `data-testid="artist-suggestion"`, min touch height 44px), pick on click or Enter → call `onPick(name)` and clear. Use theme token classes (no raw hex, no arbitrary `text-[Npx]`). Empty query → no dropdown; loading + error states handled gracefully.

- [ ] **Step 2: Write the failing test** — `web/src/components/ArtistAutocomplete.test.tsx`:

```tsx
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";

vi.mock("../lib/api", () => ({
  api: { artistsSearch: vi.fn(async () => ({ items: ["Alex G", "(Sandy) Alex G"] })) },
}));
import { api } from "../lib/api";
import { ArtistAutocomplete } from "./ArtistAutocomplete";

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("ArtistAutocomplete", () => {
  it("queries the api and picks a suggestion", async () => {
    const onPick = vi.fn();
    render(<ArtistAutocomplete onPick={onPick} />);
    fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: "alex" } });
    await waitFor(() => expect(api.artistsSearch).toHaveBeenCalled());
    const opt = await screen.findAllByTestId("artist-suggestion");
    fireEvent.click(opt[0]);
    expect(onPick).toHaveBeenCalledWith("Alex G");
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `npm --prefix web run test -- ArtistAutocomplete`
Expected: FAIL (component missing).

- [ ] **Step 4: Implement `ArtistAutocomplete.tsx`** per Step 1 (mirror GenreAutocomplete's structure, swap `genresSearch`→`artistsSearch`, and emit raw artist-name strings via `onPick`).

- [ ] **Step 5: Run to verify it passes**

Run: `npm --prefix web run test -- ArtistAutocomplete`
Expected: PASS.

- [ ] **Step 6: Typecheck + commit**

```bash
cd web && npx tsc -b && cd ..
git add web/src/components/ArtistAutocomplete.tsx web/src/components/ArtistAutocomplete.test.tsx
git commit --only -m "feat(artist-links): ArtistAutocomplete typeahead component" -- web/src/components/ArtistAutocomplete.tsx web/src/components/ArtistAutocomplete.test.tsx
```

---

## Task 7: `ArtistLinksPanel` + 5th Advanced tab

**Files:**
- Create: `web/src/components/ArtistLinksPanel.tsx`
- Create: `web/src/components/ArtistLinksPanel.test.tsx`
- Modify: `web/src/components/AdvancedPanel.tsx`

**Interfaces — Consumes:** `api.artistLinksList/artistLinksSave` (Task 5), `ArtistAutocomplete` (Task 6).

- [ ] **Step 1: Read the reference** — READ `web/src/components/GenreReviewPanel.tsx` for the state + load-in-`useEffect` + mutate-with-error-reload shape. `ArtistLinksPanel` requirements:
  - On mount, `api.artistLinksList()` → state `groups: ArtistLinkGroup[]`; render **loading**, **error**, and **empty** ("No artist links yet") states.
  - List existing groups, each labeled by type ("Alias" / "Same artist") with its members and a Remove button (`data-testid="remove-group-<i>"`) that drops it from local state.
  - A "New link" form: a type selector (radio/segmented, `data-testid="link-type"` with values `alias`/`sibling`), one or more members added via `<ArtistAutocomplete onPick={...}>` (added chips, `data-testid="member-chip"`, each removable), and an "Add group" button (`data-testid="add-group"`, disabled until type + ≥2 members) that appends to local `groups`.
  - A "Save" button (`data-testid="save-links"`) → `api.artistLinksSave({ groups })`; optimistic is unnecessary (it's a full-list save) — show a saving state, then a `saved ✓` flash on success; on error `setError(String(e))` and reload from server (`api.artistLinksList()`).
  - UI discipline: 44px touch targets, theme tokens, inputs ≥16px, `:focus-visible`, no raw hex/arbitrary type. Copy uses plain language ("Alias — same act, different spelling", "Same artist — different projects").

- [ ] **Step 2: Write the failing test** — `web/src/components/ArtistLinksPanel.test.tsx`:

```tsx
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";

vi.mock("../lib/api", () => ({
  api: {
    artistLinksList: vi.fn(async () => ({ groups: [] })),
    artistLinksSave: vi.fn(async () => ({ ok: true, count: 1 })),
    artistsSearch: vi.fn(async () => ({ items: ["Smog", "Bill Callahan"] })),
  },
}));
import { api } from "../lib/api";
import { ArtistLinksPanel } from "./ArtistLinksPanel";

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("ArtistLinksPanel", () => {
  it("builds a sibling group and saves it", async () => {
    render(<ArtistLinksPanel />);
    await waitFor(() => expect(api.artistLinksList).toHaveBeenCalled());

    fireEvent.click(screen.getByTestId("link-type-sibling"));
    // add two members via the autocomplete
    for (const name of ["Smog", "Bill Callahan"]) {
      fireEvent.change(screen.getByTestId("artist-autocomplete-input"), { target: { value: name } });
      const opt = await screen.findAllByTestId("artist-suggestion");
      fireEvent.click(opt.find((o) => o.textContent === name) ?? opt[0]);
    }
    fireEvent.click(screen.getByTestId("add-group"));
    fireEvent.click(screen.getByTestId("save-links"));

    await waitFor(() => expect(api.artistLinksSave).toHaveBeenCalled());
    const arg = vi.mocked(api.artistLinksSave).mock.calls[0][0];
    expect(arg.groups[0]).toEqual({ type: "sibling", members: ["Smog", "Bill Callahan"] });
  });
});
```

(Adapt the `data-testid`s in the test to whatever you implement — keep them consistent between the component and the test. The assertion that `artistLinksSave` receives `{type:"sibling", members:[...]}` is the load-bearing check.)

- [ ] **Step 3: Run to verify it fails**

Run: `npm --prefix web run test -- ArtistLinksPanel`
Expected: FAIL (component missing).

- [ ] **Step 4: Implement `ArtistLinksPanel.tsx`** per Step 1.

- [ ] **Step 5: Wire the 5th tab** — in `web/src/components/AdvancedPanel.tsx`:
  - Line 6 area: `import { ArtistLinksPanel } from "./ArtistLinksPanel";`
  - Line 8: `type Tab = "diagnostics" | "blacklist" | "review" | "taxonomy" | "links";`
  - After the taxonomy `tabBtn` (line 30): `{tabBtn("links", "Artist Links")}`
  - After the taxonomy mount (line 36): `{tab === "links" && <ArtistLinksPanel />}`

- [ ] **Step 6: Run to verify it passes**

Run: `npm --prefix web run test -- ArtistLinksPanel`
Expected: PASS.

- [ ] **Step 7: Typecheck + commit**

```bash
cd web && npx tsc -b && cd ..
git add web/src/components/ArtistLinksPanel.tsx web/src/components/ArtistLinksPanel.test.tsx web/src/components/AdvancedPanel.tsx
git commit --only -m "feat(artist-links): ArtistLinksPanel + 5th Advanced-rail tab" -- web/src/components/ArtistLinksPanel.tsx web/src/components/ArtistLinksPanel.test.tsx web/src/components/AdvancedPanel.tsx
```

---

## Task 8: Build, mobile-audit, and verification

**Files:** none (build + verify only; may commit a rebuilt `web/dist` if the repo tracks it — check first).

- [ ] **Step 1: Full frontend gate**

Run (from repo root):
`cd web && npm run lint && npx tsc -b && npm run test && npm run build && cd ..`
Expected: lint clean, typecheck clean, all vitest pass, `vite build` emits `web/dist`.

- [ ] **Step 2: Backend regression**

Run: `python -m pytest -q tests/unit/test_artist_aliases.py tests/integration/test_artist_links_api.py tests/unit/test_pier_bridge_smoke_golden.py`
Expected: PASS. Quote real counts.

- [ ] **Step 3: Mobile-audit sweep on the new panel**

Run: `npm --prefix web exec -- playwright test mobile-audit --config web/playwright.config.ts`
Inspect `web/test-results/mobile-audit/` screenshots + `measurements.json` for the Artist Links surface: confirm no touch-target < 44px on primary controls, no micro-type < 12px, no horizontal overflow at 390px. Fix any violation in `ArtistLinksPanel.tsx`/`ArtistAutocomplete.tsx` and re-run. If the sweep can't run in this environment, say so explicitly.

- [ ] **Step 4: web/dist handling**

Run: `git status --short web/dist` — if `web/dist` is tracked and changed, commit it (`git add web/dist && git commit --only -m "build(artist-links): rebuild web/dist" -- web/dist`); if it's gitignored, note that Dylan must rebuild locally. Determine which by `git check-ignore web/dist` first.

- [ ] **Step 5: Hand off live verification**

The panel's true end-to-end (worker restart + real `data/artist_aliases.yaml` write + click-through) requires a running `serve_web.py` on the canonical GUI (port 8770), which is Dylan's manual check. Summarize for him: restart `serve_web.py`, open the Advanced rail → "Artist Links", create one alias and one sibling link, Save, and confirm `data/artist_aliases.yaml` shows the groups (and that a generation then honors them). Do NOT claim the GUI is verified end-to-end from unit/route tests alone.

---

## Self-Review (completed during authoring)

- **Spec coverage (Section D):** 5th tab (Task 7), single-phase save (Tasks 2-3), typeahead-only members from real library artists (Tasks 4/6), wiring checklist — `handle_list_artist_links`/`handle_save_artist_links` in `UNTRACKED_COMMAND_HANDLERS` (Task 2), schemas + routes (Task 3), fake_worker branch (Task 2), rebuild/restart (Task 8). UI discipline enforced (Global Constraints + Task 8 sweep).
- **Untracked-handler contract** (explicit `request_id`/`job_id=None`, terminal `done`) copied verbatim into Task 2; route error mapping (read 502 / write 422) into Task 3.
- **Cache-bust** (`clear_cache()` after write) lives in `save_artist_link_groups` (Task 1), called by the worker (Task 2), so the next generation in the same worker session sees the change.
- **Type consistency:** `ArtistLinkGroup{type,members}` identical across schemas (Task 3), types.ts (Task 5), and component payloads (Task 7); `artistLinksList/artistLinksSave/artistsSearch` signatures match between api.ts (Task 5) and consumers (Tasks 6-7).
- **Known softness (GUI):** component tasks (6-7) specify structure + exact testids + api calls + UI rules and mirror named existing components rather than reproducing unverified JSX — deliberate, since the real components' exact Tailwind classes aren't in the plan; the vitest assertions pin the load-bearing behavior (correct save payload). Task 4's search-route test may skip if the app's DB is absent; the route logic mirrors the proven `/api/genres/search`.
