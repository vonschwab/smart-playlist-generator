# Browser GUI — Phase 2: Track Interactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a generated playlist actionable in the browser — in-app audio playback, right-click context menu, replace track, blacklist, edit genres, and M3U8/Plex export.

**Architecture:** Extend the existing FastAPI adapter (`src/playlist_web/`) with a synchronous `WorkerBridge.command()` for request/response worker commands, plus new routes for audio streaming, replace suggestions, blacklist, edit-genres, and Plex export. All mutations go through existing worker commands (worker is the single write authority). The React app (`web/`) gains a `PlayerContext` + floating `MiniPlayer`, Radix context menus and dialogs, and export controls.

**Tech Stack:** Python 3.11 / FastAPI / pydantic v2 / pytest. React 19 / TypeScript / Vite 6 / Tailwind v4 / TanStack Table v8 / Radix UI / Playwright.

**Spec:** `docs/superpowers/specs/2026-06-04-browser-gui-phase2-design.md`

**Branch:** continue on `browser-gui-phase1` (Phase 1 not yet merged; Phase 2 builds directly on it).

---

## Background the implementer needs

**The worker protocol.** `src/playlist_gui/worker.py` is a subprocess that reads one JSON command per line on stdin and writes JSON events on stdout. Each command carries a `request_id`; events echo it back. Event shapes:
- `{"type":"log","level":"INFO","msg":"...","request_id":...}`
- `{"type":"progress","stage":"...","detail":"...","request_id":...}`
- `{"type":"result","result_type":"<name>", ...payload..., "request_id":...}` — payload keys are merged into the event dict (see `emit_result`)
- `{"type":"error","message":"...","request_id":...}`
- `{"type":"done","cmd":"<name>","ok":true|false,"detail":"...","request_id":...}`

**Worker commands this phase uses (all already implemented + registered in `TRACKED_COMMAND_HANDLERS`):**
- `find_replacement_suggestions` — `{cmd, position, top_k, mode?}`. Uses the worker's in-memory `_LAST_GENERATION_CACHE` (no config needed). Emits `result_type:"replacement_suggestions"` with `{position, mode, candidates:[...], current_track_id, prev_track_id, next_track_id}`. Each candidate dict: `{index, track_id, artist_key, t_prev, t_next, mean_t, rating_key, title, artist, album, duration_ms, file_path, genres}`. **The fit score is `mean_t`.** Rejects pier positions (0 and last) with `ok:false`.
- `blacklist_set` — `{cmd, base_config_path, overrides, track_ids:[...], value:bool}`. Emits `result_type:"blacklist_set"` with `{track_ids, value, updated}`.
- `blacklist_scope_set` — `{cmd, base_config_path, overrides, scope:"album"|"artist", value, artist, enabled:bool}`. For `"album"` it calls `set_album_blacklisted(artist, value, enabled)` — **both `artist` and `value` (album title) required**. For `"artist"` it calls `set_artist_blacklisted(value, enabled)`. Emits `result_type:"blacklist_scope_set"`.
- `edit_genres` — `{cmd, artist, album, genres:[...]}`. Writes a user override to `ai_genre_enrichment.db`. Emits `result_type:"edit_genres"` with `{artist, album, genres, added, removed}`.
- `ping` — `{cmd}`. Emits `result_type:"pong"` then `done ok:true`. Used by Task 1's test.

**Blacklist writes `metadata.db`** (`UPDATE tracks SET is_blacklisted = ?`). This is the same path the PySide6 GUI uses — a bounded, reversible flag toggle. Task 5 takes a backup first (CLAUDE.md data-safety rule).

**The existing bridge** (`src/playlist_web/worker_bridge.py`) is *fire-and-forget*: `submit(cmd)` writes the command and returns immediately; events flow asynchronously through the `on_event` callback. Generate uses this (streamed). Phase 2 adds `command(cmd)` for *synchronous* request/response — it awaits the matching `done` event and returns the captured `result` payload. Both set `_active_request_id`, so they are mutually exclusive (this gives the 409-busy behaviour for free).

**Test patterns.** Backend tests use `fastapi.testclient.TestClient` with a fake worker subprocess (`tests/fixtures/fake_worker.py`, launched via `worker_cmd=[sys.executable, "tests/fixtures/fake_worker.py"]`). `with TestClient(app) as client:` triggers FastAPI startup (which calls `bridge.start()`). pytest is configured with `asyncio_mode = "auto"`. Frontend uses Playwright (`web/tests/`) against a real server + fake worker injected via `PG_WEB_WORKER_CMD`.

**Run all backend tests:** `pytest tests/integration/test_web_api_phase2.py tests/unit/test_web_worker_bridge.py -v`
**Run frontend build/lint:** `npm --prefix web run build` and `npm --prefix web run lint`
**Run Playwright:** `npm --prefix web run test:e2e`

---

## File Structure

### Backend (`src/playlist_web/`)
| File | Status | Responsibility |
|------|--------|---------------|
| `worker_bridge.py` | modify | Add `command()`, `WorkerCommandError`, pending-future tracking |
| `schemas.py` | modify | Add Phase 2 request/response models |
| `audio.py` | create | Range-aware file streaming helper |
| `plex_export.py` | create | PlexExporter wrapper with graceful import/config errors |
| `app.py` | modify | Register 5 new routes |

### Frontend (`web/src/`)
| File | Status | Responsibility |
|------|--------|---------------|
| `lib/types.ts` | modify | `CandidateOut`, request types |
| `lib/api.ts` | modify | `replaceSuggestions`, `blacklist`, `editGenres`, `exportPlex` |
| `contexts/PlayerContext.tsx` | create | Player state + `usePlayer()` |
| `components/MiniPlayer.tsx` | create | Floating audio player |
| `components/TrackTable.tsx` | modify | Play column, kebab, active/blacklist row states |
| `components/TrackContextMenu.tsx` | create | Radix context menu |
| `components/ReplaceDialog.tsx` | create | Replace modal |
| `components/EditGenresDialog.tsx` | create | Edit-genres modal |
| `components/ExportPlexDialog.tsx` | create | Plex export modal |
| `components/QualityStats.tsx` | modify | M3U8 + Plex export buttons |
| `lib/m3u.ts` | create | Client-side M3U8 blob builder |
| `App.tsx` | modify | Wire player provider, dialogs, action handlers |

### Tests
| File | Status |
|------|--------|
| `tests/fixtures/fake_worker.py` | modify (add Phase 2 canned responses) |
| `tests/unit/test_web_worker_bridge.py` | modify (command() tests) |
| `tests/integration/test_web_api_phase2.py` | create |
| `web/tests/interactions.spec.ts` | create |

---

## Task 1: WorkerBridge.command() for synchronous worker requests

**Files:**
- Modify: `src/playlist_web/worker_bridge.py`
- Test: `tests/unit/test_web_worker_bridge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_web_worker_bridge.py`:

```python
import sys
import pytest
from src.playlist_web.worker_bridge import WorkerBridge, BridgeBusy, WorkerCommandError

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


async def _noop(_e):
    pass


async def test_command_returns_result_payload():
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        result = await bridge.command({"cmd": "ping"})
        assert result["result_type"] == "pong"
    finally:
        await bridge.stop()


async def test_command_raises_on_worker_error():
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        with pytest.raises(WorkerCommandError):
            await bridge.command({"cmd": "does_not_exist"})
    finally:
        await bridge.stop()


async def test_command_rejects_when_busy():
    bridge = WorkerBridge(FAKE, on_event=_noop)
    await bridge.start()
    try:
        bridge._active_request_id = "someone-else"  # simulate in-flight request
        with pytest.raises(BridgeBusy):
            await bridge.command({"cmd": "ping"})
    finally:
        bridge._active_request_id = None
        await bridge.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_web_worker_bridge.py::test_command_returns_result_payload -v`
Expected: FAIL with `ImportError: cannot import name 'WorkerCommandError'`

- [ ] **Step 3: Add `WorkerCommandError`, pending tracking, and `command()`**

In `src/playlist_web/worker_bridge.py`, add the exception next to `BridgeBusy`:

```python
class WorkerCommandError(RuntimeError):
    """Raised when a synchronous worker command completes with ok=false."""
```

In `__init__`, add three tracking dicts after `self._active_request_id = None`:

```python
        self._pending: dict[str, asyncio.Future] = {}
        self._results: dict[str, dict] = {}
        self._errors: dict[str, str] = {}
```

Add the `command()` method (after `submit`):

```python
    async def command(self, cmd: dict, timeout: float = 60.0) -> dict:
        """Submit a worker command and await its done event.

        Returns the captured `result` event payload. Raises BridgeBusy if the
        worker is already handling a request, or WorkerCommandError if the
        command completes with ok=false.
        """
        if not self.running:
            raise RuntimeError("Worker not running")
        if self.busy:
            raise BridgeBusy("Worker is busy with another request")
        request_id = str(uuid.uuid4())
        cmd = dict(cmd)
        cmd["request_id"] = request_id
        cmd["protocol_version"] = PROTOCOL_VERSION
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = fut
        self._active_request_id = request_id
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)
            self._results.pop(request_id, None)
            self._errors.pop(request_id, None)
```

In `_read_loop`, replace the existing `done` handling block:

```python
            if event.get("type") == "done" and event.get("request_id") == self._active_request_id:
                self._active_request_id = None
            await self._on_event(event)
```

with:

```python
            etype = event.get("type")
            rid = event.get("request_id")
            if rid in self._pending:
                if etype == "result":
                    self._results[rid] = event
                elif etype == "error":
                    self._errors[rid] = event.get("message", "command failed")
                elif etype == "done":
                    fut = self._pending.get(rid)
                    if fut and not fut.done():
                        if event.get("ok"):
                            fut.set_result(self._results.get(rid, {}))
                        else:
                            msg = self._errors.get(rid) or event.get("detail") or "command failed"
                            fut.set_exception(WorkerCommandError(msg))
            if etype == "done" and rid == self._active_request_id:
                self._active_request_id = None
            await self._on_event(event)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_web_worker_bridge.py -v`
Expected: PASS (all command tests plus existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/worker_bridge.py tests/unit/test_web_worker_bridge.py
git commit -m "feat(web): add WorkerBridge.command() for synchronous worker requests"
```

---

## Task 2: Phase 2 pydantic schemas

**Files:**
- Modify: `src/playlist_web/schemas.py`
- Test: `tests/integration/test_web_api_phase2.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_web_api_phase2.py`:

```python
# tests/integration/test_web_api_phase2.py
import sys

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_phase2_schemas_importable():
    from src.playlist_web.schemas import (
        CandidateOut,
        ReplaceSuggestionsRequest,
        ReplaceSuggestionsResponse,
        BlacklistRequest,
        EditGenresRequest,
        PlexExportRequest,
    )

    cand = CandidateOut(track_id="k1", title="T", artist="A", album="Al", genres=["x"], fit_score=0.7)
    assert cand.fit_score == 0.7

    bl = BlacklistRequest(scope="album", value="Leisure", artist="Marbled Eye")
    assert bl.artist == "Marbled Eye"

    cands = ReplaceSuggestionsResponse.from_worker_candidates(
        position=3,
        raw=[{"rating_key": "k9", "title": "Song", "artist": "Band", "album": "LP",
              "genres": ["slowcore"], "mean_t": 0.66}],
    )
    assert cands.candidates[0].track_id == "k9"
    assert cands.candidates[0].fit_score == 0.66
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase2.py::test_phase2_schemas_importable -v`
Expected: FAIL with `ImportError: cannot import name 'CandidateOut'`

- [ ] **Step 3: Add the schemas**

Append to `src/playlist_web/schemas.py`:

```python
class CandidateOut(BaseModel):
    """A replacement candidate track."""

    track_id: str
    title: str = "Unknown"
    artist: str = "Unknown"
    album: str = ""
    genres: list[str] = Field(default_factory=list)
    fit_score: float = 0.0


class ReplaceSuggestionsRequest(BaseModel):
    # job_id is unused by the worker (it reads _LAST_GENERATION_CACHE) but kept
    # for client correlation and future multi-job support.
    job_id: str = ""
    position: int
    top_k: int = 10


class ReplaceSuggestionsResponse(BaseModel):
    position: int
    candidates: list[CandidateOut] = Field(default_factory=list)

    @classmethod
    def from_worker_candidates(cls, position: int, raw: list[dict]) -> "ReplaceSuggestionsResponse":
        cands = [
            CandidateOut(
                track_id=str(c.get("rating_key") or c.get("track_id") or ""),
                title=c.get("title", "Unknown"),
                artist=c.get("artist", "Unknown"),
                album=c.get("album", ""),
                genres=list(c.get("genres", []) or []),
                fit_score=float(c.get("mean_t", 0.0) or 0.0),
            )
            for c in raw
        ]
        return cls(position=position, candidates=cands)


class BlacklistRequest(BaseModel):
    track_ids: list[str] = Field(default_factory=list)
    scope: Optional[str] = None          # "album" | "artist"
    value: str = ""                      # album title (album scope) or artist name (artist scope)
    artist: str = ""                     # required for album scope
    enabled: bool = True


class EditGenresRequest(BaseModel):
    artist: str
    album: str
    genres: list[str] = Field(default_factory=list)


class PlexExportRequest(BaseModel):
    title: str
    tracks: list[dict] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_web_api_phase2.py::test_phase2_schemas_importable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/schemas.py tests/integration/test_web_api_phase2.py
git commit -m "feat(web): add Phase 2 request/response schemas"
```

---

## Task 3: Audio streaming route with HTTP Range support

**Files:**
- Create: `src/playlist_web/audio.py`
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_web_api_phase2.py`:

```python
import tempfile
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app


def _make_audio_db(tmp: Path) -> Path:
    """Create a tiny metadata.db with one track pointing at a real bytes file."""
    audio = tmp / "song.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 1000)  # 1003 bytes of fake audio
    db = tmp / "metadata.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, file_path TEXT)")
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("k0", str(audio)))
    conn.commit()
    conn.close()
    return db


def test_audio_full_request(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        import src.playlist_web.app as appmod
        monkeypatch.setattr(appmod, "DB_PATH", db)
        with TestClient(create_app(worker_cmd=FAKE)) as client:
            resp = client.get("/api/audio/k0")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/mpeg"
            assert resp.headers["accept-ranges"] == "bytes"
            assert len(resp.content) == 1003


def test_audio_range_request(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        import src.playlist_web.app as appmod
        monkeypatch.setattr(appmod, "DB_PATH", db)
        with TestClient(create_app(worker_cmd=FAKE)) as client:
            resp = client.get("/api/audio/k0", headers={"Range": "bytes=0-99"})
            assert resp.status_code == 206
            assert resp.headers["content-range"] == "bytes 0-99/1003"
            assert len(resp.content) == 100


def test_audio_unknown_track_404(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        import src.playlist_web.app as appmod
        monkeypatch.setattr(appmod, "DB_PATH", db)
        with TestClient(create_app(worker_cmd=FAKE)) as client:
            resp = client.get("/api/audio/nope")
            assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase2.py::test_audio_full_request -v`
Expected: FAIL with 404 (route not registered yet)

- [ ] **Step 3: Create the audio helper**

Create `src/playlist_web/audio.py`:

```python
"""Range-aware local audio file streaming for the browser player."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

_MIME = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
}

_CHUNK = 256 * 1024


def _lookup_path(track_id: str, db_path: Path) -> Optional[str]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT file_path FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _content_type(path: str) -> str:
    return _MIME.get(Path(path).suffix.lower(), "application/octet-stream")


def stream_audio(track_id: str, db_path: Path, request: Request) -> Response:
    """Stream the audio file for track_id, honouring an optional Range header."""
    file_path = _lookup_path(track_id, db_path)
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Track audio not found")

    file_size = os.path.getsize(file_path)
    content_type = _content_type(file_path)
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            file_path,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"},
        )

    # Parse "bytes=START-END"
    try:
        units, _, rng = range_header.partition("=")
        if units.strip() != "bytes":
            raise ValueError
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if start >= file_size or start < 0:
        raise HTTPException(
            status_code=416,
            detail="Range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)
    length = end - start + 1

    def _iter():
        with open(file_path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": content_type,
    }
    return StreamingResponse(_iter(), status_code=206, headers=headers)
```

- [ ] **Step 4: Register the route in `app.py`**

In `src/playlist_web/app.py`, add the import near the top:

```python
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
```

(add `Request` to the existing `from fastapi import ...` line)

and:

```python
from .audio import stream_audio
```

Inside `create_app`, after the `autocomplete` route, add:

```python
    @app.get("/api/audio/{track_id}")
    async def audio(track_id: str, request: Request):
        return stream_audio(track_id, DB_PATH, request)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase2.py -k audio -v`
Expected: PASS (full, range, 404)

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/audio.py src/playlist_web/app.py tests/integration/test_web_api_phase2.py
git commit -m "feat(web): add Range-aware audio streaming route"
```

---

## Task 4: Replace-suggestions route

**Files:**
- Modify: `tests/fixtures/fake_worker.py`, `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase2.py`

- [ ] **Step 1: Extend the fake worker**

In `tests/fixtures/fake_worker.py`, add a branch before the final `else:`:

```python
        elif name == "find_replacement_suggestions":
            pos = cmd.get("position")
            if pos in (0, None):
                emit({"type": "error", "message": "Cannot replace pier track", "request_id": rid, "job_id": jid})
                emit({"type": "done", "cmd": name, "ok": False, "detail": "Cannot replace pier track", "request_id": rid, "job_id": jid})
            else:
                emit({"type": "result", "result_type": "replacement_suggestions", "request_id": rid, "job_id": jid,
                      "position": pos, "mode": "best",
                      "candidates": [
                          {"index": 7, "track_id": "k9", "rating_key": "k9", "artist_key": "band",
                           "title": "Fall Like Rain", "artist": "Acetone", "album": "Cindy",
                           "genres": ["slowcore"], "mean_t": 0.74, "t_prev": 0.75, "t_next": 0.73,
                           "duration_ms": 200000, "file_path": "/9.flac"},
                      ]})
                emit({"type": "done", "cmd": name, "ok": True, "detail": "Found 1", "request_id": rid, "job_id": jid})
```

- [ ] **Step 2: Write the failing test**

Append to `tests/integration/test_web_api_phase2.py`:

```python
def test_replace_suggestions_returns_candidates():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/replace_suggestions", json={"job_id": "j1", "position": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["position"] == 3
        assert body["candidates"][0]["track_id"] == "k9"
        assert body["candidates"][0]["fit_score"] == 0.74


def test_replace_suggestions_pier_rejected():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/replace_suggestions", json={"job_id": "j1", "position": 0})
        assert resp.status_code == 422
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/integration/test_web_api_phase2.py -k replace -v`
Expected: FAIL with 404 (route not registered)

- [ ] **Step 4: Register the route**

In `src/playlist_web/app.py`, update the schema import:

```python
from .schemas import (
    BlacklistRequest,
    EditGenresRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsRequest,
    ReplaceSuggestionsResponse,
)
```

and add `WorkerCommandError` to the bridge import:

```python
from .worker_bridge import BridgeBusy, WorkerBridge, WorkerCommandError
```

Add the route after the audio route:

```python
    @app.post("/api/replace_suggestions")
    async def replace_suggestions(body: ReplaceSuggestionsRequest) -> ReplaceSuggestionsResponse:
        try:
            result = await bridge.command({
                "cmd": "find_replacement_suggestions",
                "position": body.position,
                "top_k": body.top_k,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ReplaceSuggestionsResponse.from_worker_candidates(
            position=result.get("position", body.position),
            raw=result.get("candidates", []),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase2.py -k replace -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/app.py tests/fixtures/fake_worker.py tests/integration/test_web_api_phase2.py
git commit -m "feat(web): add replace-suggestions route"
```

---

## Task 5: Blacklist route (with metadata.db backup precaution)

**Files:**
- Modify: `tests/fixtures/fake_worker.py`, `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase2.py`

- [ ] **Step 1: Back up metadata.db (one-time safety precaution)**

The blacklist write toggles `is_blacklisted` in `data/metadata.db`. Per the CLAUDE.md data-safety rule, back it up first:

```bash
python -c "import shutil, datetime, pathlib; p=pathlib.Path('data/metadata.db'); \
b=p.with_suffix('.db.bak.'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')); \
shutil.copy2(p,b) if p.exists() else print('no metadata.db present, skipping'); \
print('backup:', b if p.exists() else 'n/a')"
```

Expected: prints a `backup: data/metadata.db.bak.<timestamp>` path, or "no metadata.db present, skipping" in a clean checkout. (No commit — backups are gitignored.)

- [ ] **Step 2: Extend the fake worker**

In `tests/fixtures/fake_worker.py`, add before the final `else:`:

```python
        elif name == "blacklist_set":
            tids = cmd.get("track_ids", []) or []
            emit({"type": "result", "result_type": "blacklist_set", "request_id": rid, "job_id": jid,
                  "track_ids": tids, "value": cmd.get("value", True), "updated": len(tids)})
            emit({"type": "done", "cmd": name, "ok": True, "detail": f"Updated {len(tids)}", "request_id": rid, "job_id": jid})
        elif name == "blacklist_scope_set":
            emit({"type": "result", "result_type": "blacklist_scope_set", "request_id": rid, "job_id": jid,
                  "scope": cmd.get("scope"), "value": cmd.get("value"), "enabled": cmd.get("enabled", True), "track_ids": []})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "scope set", "request_id": rid, "job_id": jid})
```

- [ ] **Step 3: Write the failing test**

Append to `tests/integration/test_web_api_phase2.py`:

```python
def test_blacklist_track():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist", json={"track_ids": ["k0", "k1"], "enabled": True})
        assert resp.status_code == 200
        assert resp.json()["updated"] == 2


def test_blacklist_album_scope():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist", json={"scope": "album", "value": "Leisure", "artist": "Marbled Eye"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


def test_blacklist_album_scope_requires_artist():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist", json={"scope": "album", "value": "Leisure"})
        assert resp.status_code == 422
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/integration/test_web_api_phase2.py -k blacklist -v`
Expected: FAIL with 404

- [ ] **Step 5: Register the route**

In `src/playlist_web/app.py`, add after the replace route:

```python
    @app.post("/api/blacklist")
    async def blacklist(body: BlacklistRequest) -> dict:
        if body.scope:
            if body.scope not in ("album", "artist"):
                raise HTTPException(status_code=422, detail="scope must be 'album' or 'artist'")
            if body.scope == "album" and not body.artist:
                raise HTTPException(status_code=422, detail="album scope requires 'artist'")
            cmd = {
                "cmd": "blacklist_scope_set",
                "base_config_path": config_path,
                "overrides": {},
                "scope": body.scope,
                "value": body.value,
                "artist": body.artist,
                "enabled": body.enabled,
            }
        else:
            if not body.track_ids:
                raise HTTPException(status_code=422, detail="track_ids required when no scope given")
            cmd = {
                "cmd": "blacklist_set",
                "base_config_path": config_path,
                "overrides": {},
                "track_ids": body.track_ids,
                "value": body.enabled,
            }
        try:
            result = await bridge.command(cmd)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase2.py -k blacklist -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/playlist_web/app.py tests/fixtures/fake_worker.py tests/integration/test_web_api_phase2.py
git commit -m "feat(web): add blacklist route (track + album/artist scope)"
```

---

## Task 6: Edit-genres route

**Files:**
- Modify: `tests/fixtures/fake_worker.py`, `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase2.py`

- [ ] **Step 1: Extend the fake worker**

In `tests/fixtures/fake_worker.py`, add before the final `else:`:

```python
        elif name == "edit_genres":
            genres = cmd.get("genres", []) or []
            emit({"type": "result", "result_type": "edit_genres", "request_id": rid, "job_id": jid,
                  "artist": cmd.get("artist"), "album": cmd.get("album"),
                  "genres": sorted(genres), "added": sorted(genres), "removed": []})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "ok", "request_id": rid, "job_id": jid})
```

- [ ] **Step 2: Write the failing test**

Append to `tests/integration/test_web_api_phase2.py`:

```python
def test_edit_genres():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/edit_genres", json={
            "artist": "Marbled Eye", "album": "Leisure", "genres": ["post-punk", "dream pop"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "post-punk" in body["genres"]


def test_edit_genres_requires_artist_album():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/edit_genres", json={"artist": "", "album": "", "genres": ["x"]})
        assert resp.status_code == 422
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/integration/test_web_api_phase2.py -k edit_genres -v`
Expected: FAIL with 404

- [ ] **Step 4: Register the route**

In `src/playlist_web/app.py`, add after the blacklist route:

```python
    @app.post("/api/edit_genres")
    async def edit_genres(body: EditGenresRequest) -> dict:
        if not body.artist.strip() or not body.album.strip():
            raise HTTPException(status_code=422, detail="artist and album are required")
        try:
            result = await bridge.command({
                "cmd": "edit_genres",
                "artist": body.artist,
                "album": body.album,
                "genres": body.genres,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase2.py -k edit_genres -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/app.py tests/fixtures/fake_worker.py tests/integration/test_web_api_phase2.py
git commit -m "feat(web): add edit-genres route"
```

---

## Task 7: Plex export route

**Files:**
- Create: `src/playlist_web/plex_export.py`
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_web_api_phase2.py`:

```python
def test_plex_export_unconfigured_returns_503():
    # Pass a config path with no plex section so the result is deterministic
    # regardless of the developer's real config.yaml.
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "config.yaml"
        cfg.write_text("library:\n  database_path: data/metadata.db\n", encoding="utf-8")
        with TestClient(create_app(worker_cmd=FAKE, config_path=str(cfg))) as client:
            resp = client.post("/api/export/plex", json={
                "title": "My Playlist",
                "tracks": [{"rating_key": "k0", "title": "Sundown", "artist": "Acetone", "file_path": "/0.flac"}],
            })
            assert resp.status_code == 503
            assert "plex" in resp.json()["detail"].lower()
```

The helper treats a missing config, missing `plex` section, `plex.enabled: false`, or missing `base_url`/`token` as "not configured".

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase2.py -k plex -v`
Expected: FAIL with 404 (route not registered)

- [ ] **Step 3: Create the Plex export helper**

Create `src/playlist_web/plex_export.py`:

```python
"""Server-side Plex playlist export, wrapping the existing PlexExporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PlexNotConfigured(RuntimeError):
    """Raised when Plex settings are missing from config."""


def _load_plex_config(config_path: str) -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}
    import yaml

    with open(cfg_file, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("plex", {}) or {}


def run_plex_export(title: str, tracks: list[dict], config_path: str) -> str:
    """Export a playlist to Plex. Returns the Plex playlist key.

    Raises PlexNotConfigured if plex.base_url / plex.token are missing,
    or RuntimeError on export failure.
    """
    plex = _load_plex_config(config_path)
    base_url = plex.get("base_url")
    token = plex.get("token")
    if not plex.get("enabled", False) or not base_url or not token:
        raise PlexNotConfigured(
            "Plex is not configured (set plex.enabled: true, plex.base_url and plex.token in config.yaml)."
        )

    from src.plex_exporter import PlexExporter

    exporter = PlexExporter(
        base_url,
        token,
        music_section=plex.get("music_section"),
        verify_ssl=plex.get("verify_ssl", True),
        path_map=plex.get("path_map"),
    )
    plex_tracks: list[dict[str, Any]] = [
        {
            "rating_key": t.get("rating_key") or t.get("track_id"),
            "title": t.get("title", ""),
            "artist": t.get("artist", ""),
            "file_path": t.get("file_path", ""),
        }
        for t in tracks
    ]
    key = exporter.export_playlist(title, plex_tracks)
    if not key:
        raise RuntimeError("Plex export returned no playlist key.")
    return str(key)
```

- [ ] **Step 4: Register the route**

In `src/playlist_web/app.py`, add the import:

```python
from .plex_export import PlexNotConfigured, run_plex_export
```

Add the route after the edit-genres route:

```python
    @app.post("/api/export/plex")
    async def export_plex(body: PlexExportRequest) -> dict:
        try:
            key = run_plex_export(body.title, body.tracks, config_path)
        except PlexNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any export failure to the client
            raise HTTPException(status_code=502, detail=f"Plex export failed: {exc}")
        return {"ok": True, "playlist_key": key}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase2.py -k plex -v`
Expected: PASS

Then run the full backend suite:

Run: `pytest tests/integration/test_web_api_phase2.py tests/unit/test_web_worker_bridge.py -v`
Expected: PASS (all Phase 2 backend tests)

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/plex_export.py src/playlist_web/app.py tests/integration/test_web_api_phase2.py
git commit -m "feat(web): add Plex export route"
```

---

## Task 8: Frontend types, API client, and Radix dependencies

**Files:**
- Modify: `web/src/lib/types.ts`, `web/src/lib/api.ts`, `web/package.json` (via npm)
- Create: `web/src/lib/m3u.ts`

- [ ] **Step 1: Install Radix primitives**

```bash
npm --prefix web install @radix-ui/react-dropdown-menu @radix-ui/react-dialog
```

Expected: `package.json` dependencies gain both packages; no errors.

> **Why dropdown-menu, not context-menu:** `@radix-ui/react-context-menu` only opens natively on right-click of its trigger and cannot be opened programmatically from a kebab button. We use `@radix-ui/react-dropdown-menu` with a cursor-anchored invisible trigger so both right-click and the kebab open the *same* controlled menu (Task 11).

- [ ] **Step 2: Add TypeScript types**

Append to `web/src/lib/types.ts`:

```typescript
export interface CandidateOut {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genres: string[];
  fit_score: number;
}

export interface ReplaceSuggestionsResponse {
  position: number;
  candidates: CandidateOut[];
}

export interface BlacklistRequest {
  track_ids?: string[];
  scope?: "album" | "artist";
  value?: string;
  artist?: string;
  enabled?: boolean;
}

export interface EditGenresRequest {
  artist: string;
  album: string;
  genres: string[];
}

export interface PlexExportRequest {
  title: string;
  tracks: TrackOut[];
}
```

- [ ] **Step 3: Add API client methods**

In `web/src/lib/api.ts`, update the import and add methods inside the `api` object:

```typescript
import type {
  BlacklistRequest,
  EditGenresRequest,
  GenerateRequestBody,
  JobOut,
  PlexExportRequest,
  ReplaceSuggestionsResponse,
} from "./types";
```

Add inside `export const api = { ... }` (after `autocomplete`):

```typescript
  async replaceSuggestions(jobId: string, position: number, topK = 10): Promise<ReplaceSuggestionsResponse> {
    return jsonOrThrow(await fetch("/api/replace_suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, position, top_k: topK }),
    }));
  },
  async blacklist(req: BlacklistRequest): Promise<{ ok: boolean; updated?: number }> {
    return jsonOrThrow(await fetch("/api/blacklist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async editGenres(req: EditGenresRequest): Promise<{ ok: boolean; genres: string[] }> {
    return jsonOrThrow(await fetch("/api/edit_genres", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async exportPlex(req: PlexExportRequest): Promise<{ ok: boolean; playlist_key: string }> {
    return jsonOrThrow(await fetch("/api/export/plex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
```

- [ ] **Step 4: Create the M3U8 builder**

Create `web/src/lib/m3u.ts`:

```typescript
import type { TrackOut } from "./types";

/** Build an M3U8 playlist string from tracks (uses local file_path per track). */
export function buildM3U8(tracks: TrackOut[]): string {
  const lines = ["#EXTM3U"];
  for (const t of tracks) {
    const seconds = Math.round((t.duration_ms ?? 0) / 1000);
    lines.push(`#EXTINF:${seconds},${t.artist} - ${t.title}`);
    lines.push(t.file_path);
  }
  return lines.join("\n") + "\n";
}

/** Trigger a browser download of the playlist as an .m3u8 file. */
export function downloadM3U8(tracks: TrackOut[], filename = "playlist.m3u8"): void {
  const blob = new Blob([buildM3U8(tracks)], { type: "audio/x-mpegurl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
```

- [ ] **Step 5: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds (TypeScript compiles; the new types/methods are referenced in later tasks but must at least type-check now).

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts web/src/lib/m3u.ts web/package.json web/package-lock.json
git commit -m "feat(web): Phase 2 types, API client, M3U8 builder, Radix deps"
```

---

## Task 9: PlayerContext + MiniPlayer

**Files:**
- Create: `web/src/contexts/PlayerContext.tsx`, `web/src/components/MiniPlayer.tsx`
- Modify: `web/src/App.tsx` (mount provider + player)

- [ ] **Step 1: Create the PlayerContext**

Create `web/src/contexts/PlayerContext.tsx`:

```tsx
import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import type { TrackOut } from "../lib/types";

interface PlayerState {
  playlist: TrackOut[];
  currentIndex: number; // -1 = nothing loaded
  playing: boolean;
}

interface PlayerContextValue extends PlayerState {
  load: (playlist: TrackOut[], index: number) => void;
  setPlaying: (p: boolean) => void;
  next: () => void;
  prev: () => void;
  current: TrackOut | null;
}

const PlayerContext = createContext<PlayerContextValue | null>(null);

export function PlayerProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PlayerState>({ playlist: [], currentIndex: -1, playing: false });

  const load = useCallback((playlist: TrackOut[], index: number) => {
    setState({ playlist, currentIndex: index, playing: true });
  }, []);
  const setPlaying = useCallback((p: boolean) => setState((s) => ({ ...s, playing: p })), []);
  const next = useCallback(() => setState((s) => {
    if (s.playlist.length === 0) return s;
    return { ...s, currentIndex: (s.currentIndex + 1) % s.playlist.length, playing: true };
  }), []);
  const prev = useCallback(() => setState((s) => {
    if (s.playlist.length === 0) return s;
    return { ...s, currentIndex: (s.currentIndex - 1 + s.playlist.length) % s.playlist.length, playing: true };
  }), []);

  const current = state.currentIndex >= 0 ? state.playlist[state.currentIndex] ?? null : null;

  return (
    <PlayerContext.Provider value={{ ...state, load, setPlaying, next, prev, current }}>
      {children}
    </PlayerContext.Provider>
  );
}

export function usePlayer(): PlayerContextValue {
  const ctx = useContext(PlayerContext);
  if (!ctx) throw new Error("usePlayer must be used within PlayerProvider");
  return ctx;
}
```

- [ ] **Step 2: Create the MiniPlayer**

Create `web/src/components/MiniPlayer.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { usePlayer } from "../contexts/PlayerContext";

const fmtTime = (s: number) => {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
};

export function MiniPlayer() {
  const { current, playing, setPlaying, next, prev } = usePlayer();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [elapsed, setElapsed] = useState(0);
  const [duration, setDuration] = useState(0);

  // Load source when the current track changes.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !current) return;
    audio.src = `/api/audio/${encodeURIComponent(current.rating_key ?? "")}`;
    audio.load();
    if (playing) audio.play().catch(() => setPlaying(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.rating_key]);

  // React to play/pause state changes.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !current) return;
    if (playing) audio.play().catch(() => setPlaying(false));
    else audio.pause();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  if (!current) return null;

  return (
    <div
      data-testid="mini-player"
      className="fixed bottom-4 right-4 z-50 flex items-center gap-3 bg-panel border border-accent rounded-lg px-4 py-2 shadow-2xl min-w-[300px]"
    >
      <audio
        ref={audioRef}
        onTimeUpdate={(e) => setElapsed(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onEnded={next}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
      <button onClick={prev} className="text-faint hover:text-text text-sm" title="Previous">⏮</button>
      <button
        onClick={() => setPlaying(!playing)}
        className="bg-accent text-bg font-bold text-sm px-2 py-1 rounded"
        title={playing ? "Pause" : "Play"}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <button onClick={next} className="text-faint hover:text-text text-sm" title="Next">⏭</button>
      <div className="flex-1 min-w-0">
        <div className="text-text text-[10px] truncate">{current.title} — {current.artist}</div>
        <div
          className="mt-1 h-0.5 bg-border rounded cursor-pointer"
          onClick={(e) => {
            const audio = audioRef.current;
            if (!audio || !duration) return;
            const rect = e.currentTarget.getBoundingClientRect();
            audio.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
          }}
        >
          <div className="h-full bg-accent rounded" style={{ width: `${duration ? (elapsed / duration) * 100 : 0}%` }} />
        </div>
      </div>
      <span className="text-faint text-[10px] font-mono whitespace-nowrap">
        {fmtTime(elapsed)} / {fmtTime(duration)}
      </span>
    </div>
  );
}
```

- [ ] **Step 3: Mount provider + player in App.tsx**

In `web/src/App.tsx`, add imports:

```tsx
import { PlayerProvider } from "./contexts/PlayerContext";
import { MiniPlayer } from "./components/MiniPlayer";
```

Wrap the returned `<Shell .../>` so the whole app is inside `PlayerProvider` and `MiniPlayer` renders on top. Change the `return (` block to:

```tsx
  return (
    <PlayerProvider>
      <Shell
        topBar={
          <>
            <div className="font-bold text-sm"><span className="text-accent">◆</span> Playlist Generator</div>
            {error && <div className="text-danger text-xs">{error}</div>}
          </>
        }
        jobs={<JobsPanel jobs={jobs} onSelect={(j) => setPlaylist(j.playlist ?? null)} />}
        center={
          <div className="h-full flex flex-col overflow-hidden">
            <GenerateControls onSubmit={submit} busy={busy} />
            <QualityStats metrics={playlist?.metrics} count={playlist?.track_count ?? 0} />
            <div className="flex-1 overflow-auto">
              <TrackTable tracks={playlist?.tracks ?? []} />
            </div>
          </div>
        }
        right={<AdvancedPanel />}
        logs={<LogPanel lines={logs} />}
      />
      <MiniPlayer />
    </PlayerProvider>
  );
```

(The `TrackTable` props will gain handlers in Task 10/11 — leave as-is for now; it still compiles.)

- [ ] **Step 4: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/contexts/PlayerContext.tsx web/src/components/MiniPlayer.tsx web/src/App.tsx
git commit -m "feat(web): PlayerContext + floating MiniPlayer"
```

---

## Task 10: TrackTable — play button, active row, blacklist dim

**Files:**
- Modify: `web/src/components/TrackTable.tsx`

- [ ] **Step 1: Rewrite TrackTable with play column and row states**

Replace the entire contents of `web/src/components/TrackTable.tsx`:

```tsx
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";
import type { TrackOut } from "../lib/types";
import { usePlayer } from "../contexts/PlayerContext";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export interface TrackTableProps {
  tracks: TrackOut[];
  blacklisted?: Set<string>;
  // x/y are viewport coordinates used to anchor the context menu at the cursor.
  onContextAction?: (track: TrackOut, index: number, x: number, y: number) => void;
}

export function TrackTable({ tracks, blacklisted, onContextAction }: TrackTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const player = usePlayer();

  const col = createColumnHelper<TrackOut>();
  const columns = [
    col.display({
      id: "play",
      header: "",
      cell: (c) => {
        const idx = c.row.index;
        const isCurrent = player.current?.rating_key === c.row.original.rating_key;
        return (
          <button
            data-testid="play-btn"
            onClick={(e) => {
              e.stopPropagation();
              if (isCurrent) player.setPlaying(!player.playing);
              else player.load(tracks, idx);
            }}
            className={`text-sm ${isCurrent ? "text-accent" : "text-faint opacity-40 hover:opacity-100"}`}
            title={isCurrent && player.playing ? "Pause" : "Play"}
          >
            {isCurrent && player.playing ? "❚❚" : "▶"}
          </button>
        );
      },
    }),
    col.accessor("position", {
      header: "#",
      cell: (c) => {
        const isCurrent = player.current?.rating_key === c.row.original.rating_key;
        return (
          <span className={`font-mono text-[10px] ${isCurrent ? "text-accent" : "text-faint"}`}>
            {String(c.getValue() + 1).padStart(2, "0")}
          </span>
        );
      },
    }),
    col.accessor("title", {
      header: "Track",
      cell: (c) => {
        const bl = blacklisted?.has(c.row.original.rating_key ?? "");
        return (
          <div>
            <div className={`text-xs ${bl ? "text-text line-through opacity-60" : "text-text"}`}>
              {c.getValue()}
              {bl && (
                <span className="ml-1.5 bg-[#2a1a1a] text-danger text-[9px] px-1.5 py-0.5 rounded-full">
                  blacklisted
                </span>
              )}
              {c.row.original.genres.slice(0, 2).map((g) => (
                <span key={g} className="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full">
                  {g}
                </span>
              ))}
            </div>
            <div className="text-muted text-[10px]">{c.row.original.artist}</div>
          </div>
        );
      },
    }),
    col.accessor("sonic_similarity", {
      header: "T",
      cell: (c) => <span className="font-mono text-accent text-[11px]">{fmt(c.getValue())}</span>,
    }),
    col.display({
      id: "kebab",
      header: "",
      cell: (c) => (
        <button
          data-testid="kebab-btn"
          onClick={(e) => {
            e.stopPropagation();
            onContextAction?.(c.row.original, c.row.index, e.clientX, e.clientY);
          }}
          className="text-muted opacity-0 group-hover:opacity-100 hover:text-text text-sm"
          title="Actions"
        >
          ⋯
        </button>
      ),
    }),
  ];

  const table = useReactTable({
    data: tracks,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (tracks.length === 0) {
    return <div className="p-6 text-center text-muted text-xs">No playlist yet — generate one.</div>;
  }

  return (
    <table className="w-full text-left" data-testid="track-table">
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id} className="border-b border-border">
            {hg.headers.map((h) => (
              <th
                key={h.id}
                onClick={h.column.getToggleSortingHandler()}
                className="px-3 py-2 text-[9px] uppercase tracking-wide text-faint cursor-pointer select-none"
              >
                {flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((r) => {
          const isCurrent = player.current?.rating_key === r.original.rating_key;
          return (
            <tr
              key={r.id}
              onContextMenu={(e) => {
                e.preventDefault();
                onContextAction?.(r.original, r.index, e.clientX, e.clientY);
              }}
              className={`group border-b border-[#181b21] ${
                isCurrent ? "bg-[#15202b]" : "odd:bg-panel2 hover:bg-[#15202b]"
              }`}
            >
              {r.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-2 align-top">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds. (`App.tsx` calls `<TrackTable tracks={...} />`; the new props are optional, so it still type-checks.)

- [ ] **Step 3: Commit**

```bash
git add web/src/components/TrackTable.tsx
git commit -m "feat(web): track table play button, active-row + blacklist states"
```

---

## Task 11: Track context menu (Radix)

**Files:**
- Create: `web/src/components/TrackContextMenu.tsx`
- Modify: `web/src/App.tsx` (wire menu + handlers — minimal, dialogs added in 12–14)

- [ ] **Step 1: Create the context menu component**

This is a controlled dropdown menu anchored at the cursor. `App` owns the open state, the target row, and the cursor position; an invisible `Trigger` is positioned at those coordinates so the menu pops up where the user clicked (both right-click and kebab provide coordinates via Task 10's `onContextAction`).

Create `web/src/components/TrackContextMenu.tsx`:

```tsx
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import type { TrackOut } from "../lib/types";

export interface MenuTarget {
  track: TrackOut;
  index: number;
  isPier: boolean;
}

export interface TrackContextMenuProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  target: MenuTarget | null;
  pos: { x: number; y: number };
  onReplace: (t: MenuTarget) => void;
  onBlacklistTrack: (t: MenuTarget) => void;
  onBlacklistAlbum: (t: MenuTarget) => void;
  onBlacklistArtist: (t: MenuTarget) => void;
  onEditGenres: (t: MenuTarget) => void;
}

const item =
  "px-3 py-1.5 text-xs text-text hover:bg-border rounded cursor-pointer outline-none data-[disabled]:opacity-40 data-[disabled]:cursor-default";

export function TrackContextMenu(props: TrackContextMenuProps) {
  const t = props.target;
  return (
    <DropdownMenu.Root open={props.open} onOpenChange={props.onOpenChange}>
      <DropdownMenu.Trigger asChild>
        <span
          aria-hidden
          style={{ position: "fixed", left: props.pos.x, top: props.pos.y, width: 0, height: 0 }}
        />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="start"
          className="z-50 min-w-[200px] bg-panel border border-border rounded-md shadow-2xl p-1"
        >
          {t && (
            <>
              <DropdownMenu.Item className={item} disabled={t.isPier} onSelect={() => props.onReplace(t)}>
                Replace this track…
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onBlacklistTrack(t)}>
                Blacklist 1 Track(s)
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onBlacklistAlbum(t)}>
                Blacklist Album: {t.track.album}
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onBlacklistArtist(t)}>
                Blacklist Artist: {t.track.artist}
              </DropdownMenu.Item>
              <DropdownMenu.Separator className="h-px bg-border my-1" />
              <DropdownMenu.Item className={item} onSelect={() => props.onEditGenres(t)}>
                Edit genres for album: {t.track.album}
              </DropdownMenu.Item>
            </>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
```

- [ ] **Step 2: Wire the menu and action handlers in App.tsx**

In `web/src/App.tsx`, add imports:

```tsx
import { TrackContextMenu, type MenuTarget } from "./components/TrackContextMenu";
import { api } from "./lib/api";
```

(`api` is already imported — don't duplicate.)

Add state inside `App`:

```tsx
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuTarget, setMenuTarget] = useState<MenuTarget | null>(null);
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [blacklisted, setBlacklisted] = useState<Set<string>>(new Set());
```

Also add the `TrackOut` type import if not already present:

```tsx
import type { GenerateRequestBody, JobOut, PlaylistOut, TrackOut, WsEvent } from "./lib/types";
```

Add handlers inside `App`:

```tsx
  const openMenu = useCallback((track: TrackOut, index: number, x: number, y: number) => {
    const last = (playlist?.tracks.length ?? 0) - 1;
    setMenuTarget({ track, index, isPier: index === 0 || index === last });
    setMenuPos({ x, y });
    setMenuOpen(true);
  }, [playlist]);

  const markBlacklisted = useCallback((ids: string[]) => {
    setBlacklisted((prev) => { const n = new Set(prev); ids.forEach((i) => n.add(i)); return n; });
  }, []);

  const handleBlacklistTrack = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ track_ids: [t.track.rating_key ?? ""], enabled: true });
      markBlacklisted([t.track.rating_key ?? ""]);
    } catch (e) { setError(String(e)); }
  }, [markBlacklisted]);

  const handleBlacklistAlbum = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ scope: "album", value: t.track.album, artist: t.track.artist, enabled: true });
    } catch (e) { setError(String(e)); }
  }, []);

  const handleBlacklistArtist = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ scope: "artist", value: t.track.artist, enabled: true });
    } catch (e) { setError(String(e)); }
  }, []);
```

(Replace and Edit Genres handlers are added in Tasks 12–13; for now stub them to close the menu:)

```tsx
  const handleReplace = useCallback((_t: MenuTarget) => { setMenuOpen(false); }, []);
  const handleEditGenres = useCallback((_t: MenuTarget) => { setMenuOpen(false); }, []);
```

Pass `onContextAction` to `TrackTable` in the center panel:

```tsx
          <div className="flex-1 overflow-auto">
            <TrackTable
              tracks={playlist?.tracks ?? []}
              blacklisted={blacklisted}
              onContextAction={openMenu}
            />
          </div>
```

Render `TrackContextMenu` as a sibling inside the `PlayerProvider` (it anchors itself at `menuPos` via its own positioned trigger — it does **not** wrap the table). Add it right after `<MiniPlayer />`:

```tsx
      <TrackContextMenu
        open={menuOpen}
        onOpenChange={setMenuOpen}
        target={menuTarget}
        pos={menuPos}
        onReplace={handleReplace}
        onBlacklistTrack={handleBlacklistTrack}
        onBlacklistAlbum={handleBlacklistAlbum}
        onBlacklistArtist={handleBlacklistArtist}
        onEditGenres={handleEditGenres}
      />
```

- [ ] **Step 3: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/TrackContextMenu.tsx web/src/App.tsx
git commit -m "feat(web): track context menu with blacklist actions"
```

---

## Task 12: Replace dialog

**Files:**
- Create: `web/src/components/ReplaceDialog.tsx`
- Modify: `web/src/App.tsx` (open dialog from Replace handler; swap track on confirm)

- [ ] **Step 1: Create the ReplaceDialog**

Create `web/src/components/ReplaceDialog.tsx`:

```tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CandidateOut, TrackOut } from "../lib/types";

export interface ReplaceDialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  jobId: string;
  position: number;
  prevTitle: string;
  nextTitle: string;
  onConfirm: (position: number, candidate: CandidateOut) => void;
}

export function ReplaceDialog(props: ReplaceDialogProps) {
  const [candidates, setCandidates] = useState<CandidateOut[]>([]);
  const [selected, setSelected] = useState<number>(-1);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!props.open) return;
    setLoading(true); setErr(null); setSelected(-1); setCandidates([]);
    api.replaceSuggestions(props.jobId, props.position)
      .then((r) => setCandidates(r.candidates))
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [props.open, props.jobId, props.position]);

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="replace-dialog"
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[560px] max-w-[90vw] bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <div>
              <Dialog.Title className="text-text text-sm font-semibold">Replace track</Dialog.Title>
              <div className="text-muted text-[11px] mt-0.5">
                Position {props.position + 1} · between <span className="text-text">{props.prevTitle}</span> and <span className="text-text">{props.nextTitle}</span>
              </div>
            </div>
            <Dialog.Close className="text-faint text-lg leading-none">×</Dialog.Close>
          </div>

          <div className="px-5 py-2 bg-panel2 border-b border-border text-[10px] text-faint">
            {loading ? "Loading candidates…" : `${candidates.length} candidates · ranked by transition fit to neighbors`}
          </div>

          {err && <div className="px-5 py-3 text-danger text-xs">{err}</div>}

          <div className="max-h-[260px] overflow-y-auto">
            <table className="w-full text-xs">
              <tbody>
                {candidates.map((c, i) => (
                  <tr
                    key={c.track_id}
                    data-testid="replace-candidate"
                    onClick={() => setSelected(i)}
                    style={{ opacity: 1 - i * 0.06 }}
                    className={`border-b border-[#181b21] cursor-pointer ${selected === i ? "bg-[#1a2535]" : "hover:bg-[#15202b]"}`}
                  >
                    <td className="px-3 py-2 font-mono text-faint text-[10px] w-8">{String(i + 1).padStart(2, "0")}</td>
                    <td className="px-3 py-2">
                      <div className="text-text">
                        {c.title}
                        {c.genres.slice(0, 2).map((g) => (
                          <span key={g} className="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full">{g}</span>
                        ))}
                      </div>
                      <div className="text-muted text-[10px]">{c.artist} · {c.album}</div>
                    </td>
                    <td className="px-3 py-2 font-mono text-accent">{c.fit_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="px-5 py-3 border-t border-border flex items-center justify-between bg-panel2">
            <div className="text-faint text-[10px]">Click a row to select, then Replace</div>
            <div className="flex gap-2">
              <Dialog.Close className="border border-border text-muted text-xs px-3.5 py-1.5 rounded">Cancel</Dialog.Close>
              <button
                disabled={selected < 0}
                onClick={() => { props.onConfirm(props.position, candidates[selected]); props.onOpenChange(false); }}
                className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50"
              >
                Replace
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
```

- [ ] **Step 2: Wire into App.tsx**

In `web/src/App.tsx`, add import:

```tsx
import { ReplaceDialog } from "./components/ReplaceDialog";
import type { CandidateOut } from "./lib/types";
```

Add state + job-id tracking. Store the current job id when generating (used by the replace route to correlate):

```tsx
  const [jobId, setJobId] = useState<string>("");
  const [replaceOpen, setReplaceOpen] = useState(false);
  const [replacePos, setReplacePos] = useState(0);
```

In `submit`, capture the id:

```tsx
      const { job_id } = await api.generate(body);
      setJobId(job_id);
      refreshJobs();
```

Replace the stub `handleReplace`:

```tsx
  const handleReplace = useCallback((t: MenuTarget) => {
    setMenuOpen(false);
    setReplacePos(t.index);
    setReplaceOpen(true);
  }, []);

  const applyReplacement = useCallback((position: number, cand: CandidateOut) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = [...pl.tracks];
      const old = tracks[position];
      tracks[position] = {
        ...old,
        rating_key: cand.track_id,
        title: cand.title,
        artist: cand.artist,
        album: cand.album,
        genres: cand.genres,
        sonic_similarity: cand.fit_score,
      };
      return { ...pl, tracks };
    });
  }, []);
```

Render the dialog inside the `PlayerProvider`, as a sibling after `<TrackContextMenu .../>`:

```tsx
      <ReplaceDialog
        open={replaceOpen}
        onOpenChange={setReplaceOpen}
        jobId={jobId}
        position={replacePos}
        prevTitle={playlist?.tracks[replacePos - 1]?.title ?? ""}
        nextTitle={playlist?.tracks[replacePos + 1]?.title ?? ""}
        onConfirm={applyReplacement}
      />
```

- [ ] **Step 3: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/ReplaceDialog.tsx web/src/App.tsx
git commit -m "feat(web): replace-track dialog with candidate list"
```

---

## Task 13: Edit-genres dialog

**Files:**
- Create: `web/src/components/EditGenresDialog.tsx`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Create the EditGenresDialog**

Create `web/src/components/EditGenresDialog.tsx`:

```tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";

export interface EditGenresDialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  artist: string;
  album: string;
  initialGenres: string[];
  onSaved: (album: string, genres: string[]) => void;
}

export function EditGenresDialog(props: EditGenresDialogProps) {
  const [genres, setGenres] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (props.open) { setGenres([...props.initialGenres]); setInput(""); setErr(null); }
  }, [props.open, props.initialGenres]);

  const addGenre = () => {
    const g = input.trim();
    if (g && !genres.some((x) => x.toLowerCase() === g.toLowerCase())) setGenres([...genres, g]);
    setInput("");
  };
  const removeGenre = (g: string) => setGenres(genres.filter((x) => x !== g));

  const save = async () => {
    setSaving(true); setErr(null);
    try {
      await api.editGenres({ artist: props.artist, album: props.album, genres });
      props.onSaved(props.album, genres);
      props.onOpenChange(false);
    } catch (e) { setErr(String(e)); } finally { setSaving(false); }
  };

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="edit-genres-dialog"
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[480px] max-w-[90vw] bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <div>
              <Dialog.Title className="text-text text-sm font-semibold">Edit genres</Dialog.Title>
              <div className="text-muted text-[11px] mt-0.5"><span className="text-text">{props.album}</span> · {props.artist}</div>
            </div>
            <Dialog.Close className="text-faint text-lg leading-none">×</Dialog.Close>
          </div>

          <div className="px-5 py-4">
            <div className="text-faint text-[9px] uppercase tracking-wide mb-2">Genres (click × to remove)</div>
            <div className="flex flex-wrap gap-1.5 p-2.5 bg-panel2 border border-border rounded-md min-h-[42px]">
              {genres.map((g) => (
                <span key={g} className="bg-chip text-chipText text-[11px] px-2 py-0.5 rounded-full flex items-center gap-1">
                  {g}
                  <span onClick={() => removeGenre(g)} className="text-faint cursor-pointer">×</span>
                </span>
              ))}
              <input
                data-testid="genre-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addGenre(); } }}
                placeholder="Add genre…"
                className="bg-transparent outline-none text-text text-[11px] min-w-[100px] px-1"
              />
            </div>
            <div className="text-faint text-[9px] mt-1.5">Type a genre and press Enter · applies to all tracks on this album</div>
            {err && <div className="text-danger text-xs mt-2">{err}</div>}
          </div>

          <div className="px-5 py-3 border-t border-border flex items-center justify-between bg-panel2">
            <div className="text-faint text-[10px]">Saves a user override · does not affect source tags</div>
            <div className="flex gap-2">
              <Dialog.Close className="border border-border text-muted text-xs px-3.5 py-1.5 rounded">Cancel</Dialog.Close>
              <button onClick={save} disabled={saving} className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50">
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
```

- [ ] **Step 2: Wire into App.tsx**

In `web/src/App.tsx`, add import:

```tsx
import { EditGenresDialog } from "./components/EditGenresDialog";
```

Add state:

```tsx
  const [editGenresOpen, setEditGenresOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<{ artist: string; album: string; genres: string[] }>({ artist: "", album: "", genres: [] });
```

Replace the stub `handleEditGenres`:

```tsx
  const handleEditGenres = useCallback((t: MenuTarget) => {
    setMenuOpen(false);
    setEditTarget({ artist: t.track.artist, album: t.track.album, genres: t.track.genres });
    setEditGenresOpen(true);
  }, []);

  const applyGenreEdit = useCallback((album: string, genres: string[]) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = pl.tracks.map((tr) => (tr.album === album ? { ...tr, genres } : tr));
      return { ...pl, tracks };
    });
  }, []);
```

Render the dialog after `<ReplaceDialog .../>`:

```tsx
      <EditGenresDialog
        open={editGenresOpen}
        onOpenChange={setEditGenresOpen}
        artist={editTarget.artist}
        album={editTarget.album}
        initialGenres={editTarget.genres}
        onSaved={applyGenreEdit}
      />
```

- [ ] **Step 3: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/EditGenresDialog.tsx web/src/App.tsx
git commit -m "feat(web): edit-genres dialog with tag editor"
```

---

## Task 14: Export controls (M3U8 + Plex)

**Files:**
- Create: `web/src/components/ExportPlexDialog.tsx`
- Modify: `web/src/components/QualityStats.tsx`, `web/src/App.tsx`

- [ ] **Step 1: Create the ExportPlexDialog**

Create `web/src/components/ExportPlexDialog.tsx`:

```tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { TrackOut } from "../lib/types";

export interface ExportPlexDialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  tracks: TrackOut[];
  defaultName: string;
}

export function ExportPlexDialog(props: ExportPlexDialogProps) {
  const [name, setName] = useState("");
  const [status, setStatus] = useState<"idle" | "exporting" | "done" | "error">("idle");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (props.open) { setName(props.defaultName); setStatus("idle"); setMsg(""); }
  }, [props.open, props.defaultName]);

  const doExport = async () => {
    setStatus("exporting"); setMsg("");
    try {
      const r = await api.exportPlex({ title: name, tracks: props.tracks });
      setStatus("done"); setMsg(`Exported to Plex (key ${r.playlist_key}).`);
    } catch (e) { setStatus("error"); setMsg(String(e)); }
  };

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="export-plex-dialog"
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[420px] max-w-[90vw] bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <Dialog.Title className="text-text text-sm font-semibold">Export to Plex</Dialog.Title>
            <Dialog.Close className="text-faint text-lg leading-none">×</Dialog.Close>
          </div>
          <div className="px-5 py-4">
            <div className="text-faint text-[9px] uppercase tracking-wide mb-2">Playlist name</div>
            <input
              data-testid="plex-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-panel2 border border-border rounded text-xs text-text px-2.5 py-1.5"
            />
            {status === "error" && <div className="text-danger text-xs mt-2">{msg}</div>}
            {status === "done" && <div className="text-accent text-xs mt-2">{msg}</div>}
          </div>
          <div className="px-5 py-3 border-t border-border flex justify-end gap-2 bg-panel2">
            <Dialog.Close className="border border-border text-muted text-xs px-3.5 py-1.5 rounded">Close</Dialog.Close>
            <button
              onClick={doExport}
              disabled={status === "exporting" || !name.trim()}
              className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50"
            >
              {status === "exporting" ? "Exporting…" : "Export"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
```

- [ ] **Step 2: Add export buttons to QualityStats**

Replace the contents of `web/src/components/QualityStats.tsx`:

```tsx
import type { MetricsOut, TrackOut } from "../lib/types";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export interface QualityStatsProps {
  metrics?: MetricsOut;
  count: number;
  tracks: TrackOut[];
  onExportM3U8: () => void;
  onExportPlex: () => void;
}

export function QualityStats({ metrics, count, tracks, onExportM3U8, onExportPlex }: QualityStatsProps) {
  if (!metrics || count === 0) return null;
  const disabled = tracks.length === 0;

  const stat = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-faint">{label}</span>
      <span className="font-mono text-accent text-xs">{value}</span>
    </div>
  );

  return (
    <div className="flex items-center gap-5 px-3 py-2 border-b border-border bg-panel2">
      {stat("tracks", String(count))}
      {stat("mean T", fmt(metrics.mean_transition))}
      {stat("min T", fmt(metrics.min_transition))}
      {metrics.p10_transition != null && stat("p10", fmt(metrics.p10_transition))}
      {metrics.p90_transition != null && stat("p90", fmt(metrics.p90_transition))}
      {stat("distinct artists", String(metrics.distinct_artists ?? "—"))}
      <div className="ml-auto flex gap-2">
        <button
          data-testid="export-m3u8"
          onClick={onExportM3U8}
          disabled={disabled}
          className="border border-border text-muted hover:text-text text-[11px] px-2.5 py-1 rounded disabled:opacity-40"
        >
          ↓ M3U8
        </button>
        <button
          data-testid="export-plex"
          onClick={onExportPlex}
          disabled={disabled}
          className="border border-border text-muted hover:text-text text-[11px] px-2.5 py-1 rounded disabled:opacity-40"
        >
          → Plex
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire into App.tsx**

In `web/src/App.tsx`, add imports:

```tsx
import { ExportPlexDialog } from "./components/ExportPlexDialog";
import { downloadM3U8 } from "./lib/m3u";
```

Add state:

```tsx
  const [plexOpen, setPlexOpen] = useState(false);
```

Add a default playlist name helper inside `App`:

```tsx
  const defaultPlexName = useCallback(() => {
    const date = new Date().toISOString().slice(0, 10);
    const seed = playlist?.tracks[0]?.artist ?? "Playlist";
    return `${seed} — ${date}`;
  }, [playlist]);
```

Update the `QualityStats` usage in the center panel:

```tsx
            <QualityStats
              metrics={playlist?.metrics}
              count={playlist?.track_count ?? 0}
              tracks={playlist?.tracks ?? []}
              onExportM3U8={() => playlist && downloadM3U8(playlist.tracks)}
              onExportPlex={() => setPlexOpen(true)}
            />
```

Render the dialog after `<EditGenresDialog .../>`:

```tsx
      <ExportPlexDialog
        open={plexOpen}
        onOpenChange={setPlexOpen}
        tracks={playlist?.tracks ?? []}
        defaultName={defaultPlexName()}
      />
```

- [ ] **Step 4: Verify the build compiles**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 5: Lint**

Run: `npm --prefix web run lint`
Expected: no errors (fix any unused-import/hook-deps warnings the linter flags).

- [ ] **Step 6: Commit**

```bash
git add web/src/components/ExportPlexDialog.tsx web/src/components/QualityStats.tsx web/src/lib/m3u.ts web/src/App.tsx
git commit -m "feat(web): M3U8 + Plex export controls"
```

---

## Task 15: Playwright interaction tests

**Files:**
- Create: `web/tests/interactions.spec.ts`

- [ ] **Step 1: Write the Playwright spec**

The existing `web/playwright.config.ts` builds the app, starts `tools/serve_web.py` with `PG_WEB_WORKER_CMD` pointing at the fake worker, and serves on port 8771. The fake worker now answers replace/blacklist/edit_genres. Audio requests will 404 (the fake DB has no real files) — the player still mounts, which is what we assert.

Create `web/tests/interactions.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

async function generate(page) {
  await page.goto("/");
  await page.getByPlaceholder("Acetone, Mazzy Star").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("play button mounts the mini-player", async ({ page }) => {
  await generate(page);
  await page.getByTestId("play-btn").first().click();
  await expect(page.getByTestId("mini-player")).toBeVisible();
});

test("context menu shows expected items", async ({ page }) => {
  await generate(page);
  // Row 1 (index 1) is non-pier in the 2-track fake playlist? No — both are piers.
  // The fake playlist has 2 tracks; right-click the last row and assert menu items appear.
  await page.getByTestId("track-table").locator("tbody tr").nth(1).click({ button: "right" });
  await expect(page.getByText("Blacklist 1 Track(s)")).toBeVisible();
  await expect(page.getByText(/Edit genres for album/)).toBeVisible();
});

test("blacklist dims the row", async ({ page }) => {
  await generate(page);
  await page.getByTestId("track-table").locator("tbody tr").nth(1).click({ button: "right" });
  await page.getByText("Blacklist 1 Track(s)").click();
  await expect(page.getByText("blacklisted").first()).toBeVisible();
});

test("M3U8 export triggers a download", async ({ page }) => {
  await generate(page);
  const downloadPromise = page.waitForEvent("download");
  await page.getByTestId("export-m3u8").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("playlist.m3u8");
});
```

**Note:** the fake playlist has 2 tracks, so both positions are piers (Replace is disabled). Replace-flow E2E is covered by the backend test (`test_replace_suggestions_returns_candidates`); the Playwright suite covers menu visibility, blacklist dim, player mount, and M3U8 download. Do not assert Replace is clickable here.

- [ ] **Step 2: Run the Playwright tests**

Run: `npm --prefix web run test:e2e`
Expected: 4 tests PASS. (If the dev server port 8771 is busy, stop stale servers first.)

- [ ] **Step 3: Commit**

```bash
git add web/tests/interactions.spec.ts
git commit -m "test(web): Playwright interaction tests for Phase 2"
```

---

## Task 16: Full-stack manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire backend suite**

Run: `pytest tests/integration/test_web_api_phase2.py tests/unit/test_web_worker_bridge.py tests/integration/test_web_api.py -v`
Expected: all PASS.

- [ ] **Step 2: Build the frontend**

Run: `npm --prefix web run build`
Expected: clean build.

- [ ] **Step 3: Launch against the real worker and verify by hand**

Run: `python tools/serve_web.py`

In the browser (http://127.0.0.1:8770):
1. Generate a playlist (e.g. artist "Built To Spill", cohesion narrow / genre narrow / sonic strict / pace dynamic — a known-good config from Phase 1).
2. Click a track's ▶ — the floating player appears bottom-right and **audio plays**. Confirm seek bar advances; click the seek bar to scrub; ⏭/⏮ move tracks.
3. Right-click an interior track → **Replace this track…** → candidates load → select one → Replace → the row updates in place.
4. Right-click a track → **Blacklist 1 Track(s)** → the row dims with a strikethrough + "blacklisted" chip.
5. Right-click a track → **Edit genres for album: …** → add/remove a genre → Save → chips update on all rows for that album.
6. Click **↓ M3U8** → a `.m3u8` file downloads; open it and confirm `#EXTM3U` + file paths.
7. If Plex is configured in `config.yaml`: click **→ Plex** → name → Export → success message. If not configured, confirm the dialog surfaces the 503 "not configured" message gracefully.

Per CLAUDE.md rigor: this manual pass is required before declaring Phase 2 done — the `qtbot` no-op caveat does not apply (these are real browser interactions), but audio playback and Plex export can only be confirmed by exercising them.

- [ ] **Step 4: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "fix(web): Phase 2 verification adjustments"
```

(Skip if nothing changed.)

---

## Notes for the implementer

- **Worker is single-active-request.** While a generate is running, all Phase 2 mutation routes return 409. The UI shows the message inline; it does not crash. This is by design.
- **Replace does not re-run generation.** It swaps one track in local state. Quality stats are not recalculated and continue to show the original generation's metrics (spec §8).
- **Blacklist is in-session visual only on the client.** The server persists the flag; the client tracks a `Set<string>` of blacklisted rating_keys for the current view. Re-generating reflects the persisted blacklist via the engine.
- **Audio uses `rating_key`** as the track id in `/api/audio/{track_id}`. The metadata.db PK column is `track_id`; the playlist's `rating_key` equals the track_id (see worker `formatted_tracks`).
- **Don't widen mypy/ruff ignores.** New Python modules (`audio.py`, `plex_export.py`) should type-check clean.
