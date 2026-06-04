# Browser GUI — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a React browser front-end that generates a playlist end-to-end (seeds + options → generate → ranked track table, quality stats, live logs), served by a FastAPI adapter that drives the existing worker subprocess.

**Architecture:** React (Vite/TS/Tailwind/shadcn) ⟷ FastAPI (HTTP + WebSocket) ⟷ a *new asyncio NDJSON client* that launches `src/playlist_gui/worker.py` and speaks its existing stdio protocol. The Qt `WorkerClient`/`JobManager` are **not** reused (they require a Qt event loop); only the worker process, its NDJSON protocol, and the shared `GeneratePlaylistRequest` dataclass are reused. The PySide6 GUI keeps working untouched.

**Tech Stack:** Python 3.11 / FastAPI / uvicorn / pydantic v2 / pytest · React 18 / TypeScript / Vite / Tailwind / shadcn-ui (Radix) / react-resizable-panels / TanStack Table / Playwright.

**Design reference:** `docs/superpowers/specs/2026-06-04-browser-gui-phase1-design.md`.

**Testing approach:** Backend units/integration use real TDD with a **fake worker** fixture (a tiny NDJSON-emitting script) so tests never touch the real engine or `data/metadata.db`. Frontend presentational components are verified by a **Playwright end-to-end smoke** against a running server + fake worker (per spec §12; per-component unit tests are low value for presentational UI and skipped deliberately).

---

## File Structure

**Backend (new Python package `src/playlist_web/`):**
- `src/playlist_web/__init__.py` — package marker.
- `src/playlist_web/schemas.py` — pydantic request/response/WS models + mapping to `GeneratePlaylistRequest`.
- `src/playlist_web/worker_bridge.py` — asyncio NDJSON client: launch worker, send commands, stream events via async callback; single active request.
- `src/playlist_web/jobs.py` — in-memory job registry (status, result, bounded log buffer, recent-jobs list).
- `src/playlist_web/ws.py` — WebSocket connection hub: register/unregister sockets, broadcast event dicts.
- `src/playlist_web/app.py` — FastAPI app factory: HTTP routes, `/ws`, static serving of `web/dist`, bridge/registry lifecycle.
- `tools/serve_web.py` — launcher: run uvicorn, open browser.

**Backend tests:**
- `tests/fixtures/fake_worker.py` — stub worker emitting canned NDJSON for `ping`/`generate_playlist`.
- `tests/unit/test_web_schemas.py`
- `tests/unit/test_web_worker_bridge.py`
- `tests/unit/test_web_jobs.py`
- `tests/integration/test_web_api.py`

**Frontend (new `web/` Vite app):**
- `web/` scaffold (package.json, vite.config.ts, tsconfig, tailwind.config.ts, index.html).
- `web/src/main.tsx`, `web/src/App.tsx`, `web/src/index.css`.
- `web/src/theme/studio-dark.css` — Studio Dark CSS variables.
- `web/src/lib/types.ts` — TS mirror of pydantic schemas.
- `web/src/lib/api.ts` — fetch client.
- `web/src/lib/ws.ts` — WebSocket client hook.
- `web/src/components/Shell.tsx` — resizable panel layout.
- `web/src/components/GenerateControls.tsx`
- `web/src/components/TrackTable.tsx`
- `web/src/components/QualityStats.tsx`
- `web/src/components/LogPanel.tsx`
- `web/src/components/JobsPanel.tsx`
- `web/src/components/AdvancedPanel.tsx` — Advanced/Genre-Review tab shells (placeholder content; populated in later phases).
- `web/src/components/ui/` — shadcn components as installed.
- `web/tests/generate.spec.ts` — Playwright smoke.

---

## Task 1: Backend package + health endpoint

**Files:**
- Create: `src/playlist_web/__init__.py`
- Create: `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_web_api.py
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app


def test_health_ok():
    client = TestClient(create_app())
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "worker_running" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api.py::test_health_ok -v`
Expected: FAIL — `ModuleNotFoundError: src.playlist_web.app`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/playlist_web/__init__.py
"""FastAPI adapter exposing the playlist worker to a browser front-end."""
```

```python
# src/playlist_web/app.py
from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Playlist Generator Web")

    @app.get("/api/health")
    async def health() -> dict:
        bridge = getattr(app.state, "bridge", None)
        return {
            "status": "ok",
            "worker_running": bool(bridge and bridge.running),
        }

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_web_api.py::test_health_ok -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/__init__.py src/playlist_web/app.py tests/integration/test_web_api.py
git commit -m "feat(web): FastAPI app factory with health endpoint"
```

---

## Task 2: Pydantic schemas + GeneratePlaylistRequest mapping

**Files:**
- Create: `src/playlist_web/schemas.py`
- Test: `tests/unit/test_web_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_web_schemas.py
from src.playlist_web.schemas import (
    GenerateRequestBody,
    PlaylistOut,
    TrackOut,
)


def test_request_body_maps_to_generate_request():
    body = GenerateRequestBody(
        mode="artist", artist="Acetone", tracks=20,
        genre_mode="narrow", sonic_mode="strict", pace_mode="dynamic",
    )
    req = body.to_request()
    assert req.mode == "artist"
    assert req.artist == "Acetone"
    assert req.tracks == 20
    assert req.genre_mode == "narrow"
    assert req.validation_error() is None
    args = req.to_worker_args()
    assert args["mode"] == "artist"
    assert args["artist"] == "Acetone"


def test_request_body_validation_error_surfaces():
    body = GenerateRequestBody(mode="artist", artist="", tracks=10)
    assert body.to_request().validation_error() == "Enter an artist before generating."


def test_playlist_out_parses_worker_result():
    raw = {
        "name": "Generated Playlist",
        "track_count": 1,
        "tracks": [{
            "position": 0, "rating_key": "k1", "artist": "Acetone",
            "title": "Sundown", "album": "Cindy", "duration_ms": 200000,
            "file_path": "/x.flac", "sonic_similarity": 0.91,
            "genre_similarity": 0.8, "genres": ["slowcore"],
        }],
        "metrics": {"mean_transition": 0.88, "min_transition": 0.81, "distinct_artists": 18},
    }
    out = PlaylistOut.from_worker(raw)
    assert out.track_count == 1
    assert out.tracks[0].title == "Sundown"
    assert out.metrics.distinct_artists == 18
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_web_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: src.playlist_web.schemas`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/playlist_web/schemas.py
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.playlist.request_models import GeneratePlaylistRequest


class GenerateRequestBody(BaseModel):
    mode: str = "artist"
    tracks: int = 30
    artist: Optional[str] = None
    genre: Optional[str] = None
    seed_tracks: list[str] = Field(default_factory=list)
    seed_track_ids: list[str] = Field(default_factory=list)
    genre_mode: Optional[str] = None
    sonic_mode: Optional[str] = None
    pace_mode: Optional[str] = None
    include_collaborations: bool = False
    exclude_seed_tracks_from_recency: bool = False

    def to_request(self) -> GeneratePlaylistRequest:
        return GeneratePlaylistRequest(
            mode=self.mode,
            tracks=self.tracks,
            artist=self.artist,
            genre=self.genre,
            seed_tracks=list(self.seed_tracks),
            seed_track_ids=list(self.seed_track_ids),
            genre_mode=self.genre_mode,
            sonic_mode=self.sonic_mode,
            pace_mode=self.pace_mode,
            include_collaborations=self.include_collaborations,
            exclude_seed_tracks_from_recency=self.exclude_seed_tracks_from_recency,
        )


class TrackOut(BaseModel):
    position: int
    rating_key: Optional[str] = None
    artist: str = "Unknown"
    title: str = "Unknown"
    album: str = ""
    duration_ms: int = 0
    file_path: str = ""
    sonic_similarity: Optional[float] = None
    genre_similarity: Optional[float] = None
    genres: list[str] = Field(default_factory=list)


class MetricsOut(BaseModel):
    mean_transition: Optional[float] = None
    min_transition: Optional[float] = None
    p10_transition: Optional[float] = None
    p90_transition: Optional[float] = None
    distinct_artists: Optional[int] = None


class PlaylistOut(BaseModel):
    name: str = "Generated Playlist"
    track_count: int = 0
    tracks: list[TrackOut] = Field(default_factory=list)
    metrics: MetricsOut = Field(default_factory=MetricsOut)

    @classmethod
    def from_worker(cls, raw: dict[str, Any]) -> "PlaylistOut":
        return cls(
            name=raw.get("name", "Generated Playlist"),
            track_count=raw.get("track_count", len(raw.get("tracks", []))),
            tracks=[TrackOut(**{k: t.get(k) for k in TrackOut.model_fields if k in t})
                    for t in raw.get("tracks", [])],
            metrics=MetricsOut(**(raw.get("metrics") or {})),
        )


class JobOut(BaseModel):
    job_id: str
    status: str
    stage: str = ""
    error: Optional[str] = None
    playlist: Optional[PlaylistOut] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_web_schemas.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/schemas.py tests/unit/test_web_schemas.py
git commit -m "feat(web): pydantic schemas mapping to GeneratePlaylistRequest"
```

---

## Task 3: Fake worker fixture

**Files:**
- Create: `tests/fixtures/fake_worker.py`
- Create: `tests/fixtures/__init__.py` (empty, if not present)

This stub mimics `src/playlist_gui/worker.py`'s NDJSON protocol for tests so we never invoke the real engine.

- [ ] **Step 1: Write the fixture (no test yet — it is test infrastructure, exercised by Task 4)**

```python
# tests/fixtures/fake_worker.py
"""Minimal NDJSON worker stub for tests. Reads commands on stdin, emits events on stdout."""
import json
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        rid = cmd.get("request_id")
        jid = cmd.get("job_id")
        name = cmd.get("cmd")
        if name == "ping":
            emit({"type": "result", "result_type": "pong", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": "ping", "ok": True, "request_id": rid, "job_id": jid})
        elif name == "generate_playlist":
            emit({"type": "log", "level": "INFO", "msg": "fake: starting", "request_id": rid, "job_id": jid})
            emit({"type": "progress", "stage": "beam", "current": 50, "total": 100, "detail": "searching", "request_id": rid, "job_id": jid})
            emit({"type": "result", "result_type": "playlist", "request_id": rid, "job_id": jid, "playlist": {
                "name": "Fake Playlist", "track_count": 2,
                "tracks": [
                    {"position": 0, "rating_key": "k0", "artist": "Acetone", "title": "Sundown",
                     "album": "Cindy", "duration_ms": 200000, "file_path": "/0.flac",
                     "sonic_similarity": 0.91, "genre_similarity": 0.8, "genres": ["slowcore"]},
                    {"position": 1, "rating_key": "k1", "artist": "Mazzy Star", "title": "Taxi",
                     "album": "So Tonight", "duration_ms": 210000, "file_path": "/1.flac",
                     "sonic_similarity": 0.87, "genre_similarity": 0.7, "genres": ["dreampop"]},
                ],
                "metrics": {"mean_transition": 0.89, "min_transition": 0.87, "distinct_artists": 2},
            }})
            emit({"type": "progress", "stage": "complete", "current": 100, "total": 100, "detail": "Done", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": "generate_playlist", "ok": True, "detail": "Generated 2 tracks", "request_id": rid, "job_id": jid})
        else:
            emit({"type": "error", "message": f"unknown cmd {name}", "request_id": rid, "job_id": jid})
            emit({"type": "done", "cmd": name or "?", "ok": False, "request_id": rid, "job_id": jid})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs standalone**

Run: `echo '{"cmd":"ping","request_id":"r1"}' | python tests/fixtures/fake_worker.py`
Expected: two JSON lines — a `pong` result and a `done`.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/fake_worker.py tests/fixtures/__init__.py
git commit -m "test(web): NDJSON fake worker fixture"
```

---

## Task 4: Async worker bridge

**Files:**
- Create: `src/playlist_web/worker_bridge.py`
- Test: `tests/unit/test_web_worker_bridge.py`

**Interface (referenced by later tasks):** `WorkerBridge(worker_cmd: list[str], on_event: Callable[[dict], Awaitable[None]])`, `async start()`, `async stop()`, `property running -> bool`, `property busy -> bool`, `async submit(cmd: dict) -> str` (injects `request_id`/`protocol_version`, returns request_id; raises `BridgeBusy` if a request is active).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_web_worker_bridge.py
import asyncio
import sys

import pytest

from src.playlist_web.worker_bridge import WorkerBridge, BridgeBusy

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


@pytest.mark.asyncio
async def test_bridge_streams_events_for_generate():
    events = []
    bridge = WorkerBridge(worker_cmd=FAKE, on_event=lambda e: events.append(e) or asyncio.sleep(0))
    await bridge.start()
    assert bridge.running
    rid = await bridge.submit({"cmd": "generate_playlist", "job_id": "j1",
                               "base_config_path": "config.yaml", "overrides": {}, "args": {"mode": "artist", "artist": "Acetone", "tracks": 2}})
    # drain until a done event for this request arrives
    for _ in range(100):
        if any(e.get("type") == "done" and e.get("request_id") == rid for e in events):
            break
        await asyncio.sleep(0.02)
    await bridge.stop()
    types = [e.get("type") for e in events]
    assert "result" in types and "done" in types
    result = next(e for e in events if e.get("type") == "result")
    assert result["playlist"]["track_count"] == 2
    assert all(e.get("request_id") == rid for e in events)


@pytest.mark.asyncio
async def test_bridge_rejects_concurrent_submit():
    bridge = WorkerBridge(worker_cmd=FAKE, on_event=lambda e: asyncio.sleep(0))
    await bridge.start()
    await bridge.submit({"cmd": "generate_playlist", "job_id": "j1", "args": {}})
    with pytest.raises(BridgeBusy):
        await bridge.submit({"cmd": "generate_playlist", "job_id": "j2", "args": {}})
    await bridge.stop()
```

(Note: `pytest-asyncio` is required. If not already a dev dep, add `pytest-asyncio` to `[project.optional-dependencies].dev` in `pyproject.toml` and configure `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in this step.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_web_worker_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: src.playlist_web.worker_bridge`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/playlist_web/worker_bridge.py
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Awaitable, Callable, Optional

PROTOCOL_VERSION = 1
EventHandler = Callable[[dict], Awaitable[None]]


class BridgeBusy(RuntimeError):
    """Raised when a request is submitted while another is active."""


class WorkerBridge:
    """Asyncio NDJSON client for the playlist worker subprocess."""

    def __init__(self, worker_cmd: list[str], on_event: EventHandler):
        self._cmd = worker_cmd
        self._on_event = on_event
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._active_request_id: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def busy(self) -> bool:
        return self._active_request_id is not None

    async def start(self) -> None:
        if self.running:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def submit(self, cmd: dict) -> str:
        if not self.running:
            raise RuntimeError("Worker not running")
        if self.busy:
            raise BridgeBusy("Worker is busy with another request")
        request_id = str(uuid.uuid4())
        cmd = dict(cmd)
        cmd["request_id"] = request_id
        cmd["protocol_version"] = PROTOCOL_VERSION
        self._active_request_id = request_id
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        return request_id

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                event = {"type": "log", "level": "INFO", "msg": text}
            if event.get("type") == "done" and event.get("request_id") == self._active_request_id:
                self._active_request_id = None
            await self._on_event(event)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_web_worker_bridge.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/worker_bridge.py tests/unit/test_web_worker_bridge.py pyproject.toml
git commit -m "feat(web): asyncio NDJSON worker bridge"
```

---

## Task 5: In-memory job registry

**Files:**
- Create: `src/playlist_web/jobs.py`
- Test: `tests/unit/test_web_jobs.py`

**Interface:** `JobRegistry(max_log_lines=500, max_jobs=50)`; `create() -> str` (job_id); `apply_event(event: dict)` (mutates job by `job_id`); `get(job_id) -> JobOut | None`; `recent() -> list[JobOut]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_web_jobs.py
from src.playlist_web.jobs import JobRegistry


def _gen_events(job_id):
    return [
        {"type": "log", "level": "INFO", "msg": "starting", "job_id": job_id},
        {"type": "progress", "stage": "beam", "current": 50, "total": 100, "detail": "searching", "job_id": job_id},
        {"type": "result", "result_type": "playlist", "job_id": job_id, "playlist": {
            "name": "P", "track_count": 1,
            "tracks": [{"position": 0, "artist": "A", "title": "T", "genres": []}],
            "metrics": {"mean_transition": 0.9, "min_transition": 0.8, "distinct_artists": 1}}},
        {"type": "done", "cmd": "generate_playlist", "ok": True, "detail": "Generated 1 tracks", "job_id": job_id},
    ]


def test_registry_tracks_job_to_success_with_playlist():
    reg = JobRegistry()
    jid = reg.create()
    for e in _gen_events(jid):
        reg.apply_event(e)
    job = reg.get(jid)
    assert job.status == "success"
    assert job.playlist.track_count == 1
    assert job.playlist.metrics.distinct_artists == 1
    assert reg.logs(jid)[0].endswith("starting")


def test_registry_marks_failure_with_error():
    reg = JobRegistry()
    jid = reg.create()
    reg.apply_event({"type": "error", "message": "boom", "job_id": jid})
    reg.apply_event({"type": "done", "cmd": "generate_playlist", "ok": False, "job_id": jid})
    job = reg.get(jid)
    assert job.status == "failed"
    assert job.error == "boom"


def test_recent_is_capped_and_newest_first():
    reg = JobRegistry(max_jobs=2)
    ids = [reg.create() for _ in range(3)]
    recent = reg.recent()
    assert len(recent) == 2
    assert recent[0].job_id == ids[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_web_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: src.playlist_web.jobs`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/playlist_web/jobs.py
from __future__ import annotations

import uuid
from collections import OrderedDict, deque
from typing import Optional

from .schemas import JobOut, PlaylistOut


class _JobState:
    def __init__(self, job_id: str, max_log_lines: int):
        self.job_id = job_id
        self.status = "pending"
        self.stage = ""
        self.error: Optional[str] = None
        self.playlist: Optional[PlaylistOut] = None
        self.logs: deque[str] = deque(maxlen=max_log_lines)


class JobRegistry:
    def __init__(self, max_log_lines: int = 500, max_jobs: int = 50):
        self._jobs: "OrderedDict[str, _JobState]" = OrderedDict()
        self._max_log_lines = max_log_lines
        self._max_jobs = max_jobs

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = _JobState(job_id, self._max_log_lines)
        self._jobs[job_id].status = "running"
        while len(self._jobs) > self._max_jobs:
            self._jobs.popitem(last=False)
        return job_id

    def apply_event(self, event: dict) -> None:
        job = self._jobs.get(event.get("job_id"))
        if not job:
            return
        etype = event.get("type")
        if etype == "log":
            job.logs.append(f"{event.get('level', 'INFO')}: {event.get('msg', '')}")
        elif etype == "progress":
            job.stage = event.get("detail") or event.get("stage") or job.stage
        elif etype == "result" and event.get("result_type") == "playlist":
            job.playlist = PlaylistOut.from_worker(event.get("playlist", {}))
        elif etype == "error":
            job.error = event.get("message", "Unknown error")
        elif etype == "done":
            if event.get("cancelled"):
                job.status = "cancelled"
            elif event.get("ok"):
                job.status = "success"
            else:
                job.status = "failed"
                if event.get("detail") and not job.error:
                    job.error = event["detail"]

    def _to_out(self, job: _JobState) -> JobOut:
        return JobOut(job_id=job.job_id, status=job.status, stage=job.stage,
                      error=job.error, playlist=job.playlist)

    def get(self, job_id: str) -> Optional[JobOut]:
        job = self._jobs.get(job_id)
        return self._to_out(job) if job else None

    def logs(self, job_id: str) -> list[str]:
        job = self._jobs.get(job_id)
        return list(job.logs) if job else []

    def recent(self) -> list[JobOut]:
        return [self._to_out(j) for j in reversed(self._jobs.values())]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_web_jobs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/jobs.py tests/unit/test_web_jobs.py
git commit -m "feat(web): in-memory job registry"
```

---

## Task 6: WebSocket hub

**Files:**
- Create: `src/playlist_web/ws.py`
- Test: `tests/unit/test_web_jobs.py` (append) — hub is tested without a real socket via a fake.

**Interface:** `WsHub()`; `async connect(ws)`; `disconnect(ws)`; `async broadcast(message: dict)`. Sockets only need an async `send_json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_web_ws.py
import pytest
from src.playlist_web.ws import WsHub


class FakeWS:
    def __init__(self):
        self.sent = []
    async def send_json(self, data):
        self.sent.append(data)


@pytest.mark.asyncio
async def test_hub_broadcasts_to_all_and_drops_disconnected():
    hub = WsHub()
    a, b = FakeWS(), FakeWS()
    await hub.connect(a)
    await hub.connect(b)
    await hub.broadcast({"type": "log", "msg": "hi"})
    assert a.sent == [{"type": "log", "msg": "hi"}]
    hub.disconnect(b)
    await hub.broadcast({"type": "done"})
    assert len(a.sent) == 2
    assert len(b.sent) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_web_ws.py -v`
Expected: FAIL — `ModuleNotFoundError: src.playlist_web.ws`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/playlist_web/ws.py
from __future__ import annotations

from typing import Any, Protocol


class _Sendable(Protocol):
    async def send_json(self, data: Any) -> None: ...


class WsHub:
    def __init__(self) -> None:
        self._clients: set[_Sendable] = set()

    async def connect(self, ws: _Sendable) -> None:
        self._clients.add(ws)

    def disconnect(self, ws: _Sendable) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_web_ws.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist_web/ws.py tests/unit/test_web_ws.py
git commit -m "feat(web): websocket broadcast hub"
```

---

## Task 7: Wire app — generate, jobs, autocomplete, websocket, lifecycle

**Files:**
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api.py` (append)

The app owns a `WorkerBridge`, a `JobRegistry`, and a `WsHub`. The bridge's `on_event` routes every event to both the registry (`apply_event`) and the hub (`broadcast`). `create_app(worker_cmd=...)` accepts an override so tests inject the fake worker. Autocomplete queries `data/metadata.db` read-only (artists by prefix); on any DB error it returns `[]` so it never blocks generation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_web_api.py  (append)
import sys
import time
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_generate_runs_to_success_via_fake_worker():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:  # triggers startup (bridge.start)
        resp = client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 2})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        deadline = time.time() + 5
        job = None
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("success", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert job["status"] == "success"
        assert job["playlist"]["track_count"] == 2
        assert job["playlist"]["metrics"]["distinct_artists"] == 2


def test_generate_rejects_invalid_request():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/generate", json={"mode": "artist", "artist": "", "tracks": 5})
        assert resp.status_code == 422
        assert "artist" in resp.json()["detail"].lower()


def test_jobs_list_returns_recent():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 2})
        time.sleep(0.3)
        jobs = client.get("/api/jobs").json()
        assert isinstance(jobs, list) and len(jobs) >= 1


def test_websocket_streams_generation_events():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 2})
            saw_done = False
            for _ in range(50):
                msg = ws.receive_json()
                if msg.get("type") == "done":
                    saw_done = True
                    break
            assert saw_done
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_web_api.py -v`
Expected: FAIL — `create_app()` takes no `worker_cmd`, no `/api/generate` route.

- [ ] **Step 3: Write implementation**

```python
# src/playlist_web/app.py  (replace file)
from __future__ import annotations

import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .jobs import JobRegistry
from .schemas import GenerateRequestBody, JobOut
from .worker_bridge import BridgeBusy, WorkerBridge
from .ws import WsHub

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_CMD = [sys.executable, "-m", "src.playlist_gui.worker"]
DEFAULT_CONFIG = str(ROOT / "config.yaml")
DB_PATH = ROOT / "data" / "metadata.db"
WEB_DIST = ROOT / "web" / "dist"


def create_app(worker_cmd: Optional[list[str]] = None, config_path: str = DEFAULT_CONFIG) -> FastAPI:
    registry = JobRegistry()
    hub = WsHub()

    async def on_event(event: dict) -> None:
        registry.apply_event(event)
        await hub.broadcast(event)

    bridge = WorkerBridge(worker_cmd or DEFAULT_WORKER_CMD, on_event=on_event)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await bridge.start()
        yield
        await bridge.stop()

    app = FastAPI(title="Playlist Generator Web", lifespan=lifespan)
    app.state.bridge = bridge
    app.state.registry = registry

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "worker_running": bridge.running}

    @app.post("/api/generate")
    async def generate(body: GenerateRequestBody) -> dict:
        req = body.to_request()
        err = req.validation_error()
        if err:
            raise HTTPException(status_code=422, detail=err)
        job_id = registry.create()
        try:
            await bridge.submit({
                "cmd": "generate_playlist",
                "job_id": job_id,
                "base_config_path": config_path,
                "overrides": {},
                "args": req.to_worker_args(),
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is already running.")
        return {"job_id": job_id}

    @app.get("/api/jobs")
    async def jobs() -> list[JobOut]:
        return registry.recent()

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str) -> JobOut:
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/jobs/{job_id}/logs")
    async def job_logs(job_id: str) -> dict:
        return {"logs": registry.logs(job_id)}

    @app.get("/api/autocomplete")
    async def autocomplete(q: str = "") -> list[str]:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT artist_name FROM artists WHERE artist_name LIKE ? ORDER BY artist_name LIMIT 15",
                    (q + "%",),
                ).fetchall()
            finally:
                conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive; client may send pings
        except WebSocketDisconnect:
            hub.disconnect(ws)

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(WEB_DIST / "index.html")

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api.py -v`
Expected: PASS (health + 4 new tests).

- [ ] **Step 5: Run the full backend suite**

Run: `pytest tests/unit/test_web_schemas.py tests/unit/test_web_worker_bridge.py tests/unit/test_web_jobs.py tests/unit/test_web_ws.py tests/integration/test_web_api.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/app.py tests/integration/test_web_api.py
git commit -m "feat(web): generate/jobs/autocomplete routes + websocket + lifecycle"
```

---

## Task 8: Launcher

**Files:**
- Create: `tools/serve_web.py`

- [ ] **Step 1: Write the launcher**

```python
# tools/serve_web.py
"""Launch the browser playlist GUI: start FastAPI (which owns the worker) and open the browser."""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import uvicorn
    from src.playlist_web.app import create_app

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    print(f"Playlist Generator (web) → {url}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the launcher against the fake worker (manual)**

Run: `python -c "from src.playlist_web.app import create_app; print('import ok')"`
Expected: prints `import ok` (no import errors). Full run is exercised in Task 14's Playwright smoke.

- [ ] **Step 3: Commit**

```bash
git add tools/serve_web.py
git commit -m "feat(web): serve_web launcher (uvicorn + browser open)"
```

---

## Task 9: Frontend scaffold + Studio Dark theme

**Files:**
- Create: `web/` (Vite React-TS scaffold), `web/tailwind.config.ts`, `web/src/theme/studio-dark.css`, `web/src/index.css`, `web/vite.config.ts`

- [ ] **Step 1: Scaffold the app**

```bash
cd web 2>/dev/null || (npm create vite@latest web -- --template react-ts && cd web)
# From repo root if the above created it:
cd web
npm install
npm install -D tailwindcss postcss autoprefixer @tailwindcss/postcss
npm install react-resizable-panels @tanstack/react-table clsx tailwind-merge
npx tailwindcss init -p
```

- [ ] **Step 2: Configure Vite dev proxy** (so `/api` and `/ws` reach FastAPI during dev)

```typescript
// web/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8770",
      "/ws": { target: "ws://127.0.0.1:8770", ws: true },
    },
  },
  build: { outDir: "dist" },
});
```

- [ ] **Step 3: Tailwind config with Studio Dark tokens**

```typescript
// web/tailwind.config.ts
import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0f1115", panel: "#16181d", panel2: "#13151a",
        border: "#23262d", text: "#e6e9ec", muted: "#8b939d",
        faint: "#5b6470", accent: "#5eead4", warn: "#fbbf24", danger: "#fb7185",
        chip: "#1d2937", chipText: "#7dd3fc",
      },
      fontFamily: {
        sans: ["Spline Sans", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
```

- [ ] **Step 4: Base CSS**

```css
/* web/src/index.css */
@import url("https://fonts.googleapis.com/css2?family=Spline+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; margin: 0; }
body { background: #0f1115; color: #e6e9ec; font-family: "Spline Sans", system-ui, sans-serif; }
```

(If using Tailwind v4, replace the three `@tailwind` lines with `@import "tailwindcss";` and use the `@tailwindcss/postcss` plugin in `postcss.config.js`. The executing agent confirms which Tailwind major version `npm install` resolved and applies the matching directive.)

- [ ] **Step 5: Verify the scaffold builds**

Run: `cd web && npm run build`
Expected: Vite build succeeds, emits `web/dist/`.

- [ ] **Step 6: Commit**

```bash
git add web/ -- ":!web/node_modules"
echo "web/node_modules/" >> .gitignore
echo "web/dist/" >> .gitignore
git add .gitignore
git commit -m "feat(web): Vite React-TS scaffold + Tailwind Studio Dark theme"
```

---

## Task 10: Types + API + WS clients

**Files:**
- Create: `web/src/lib/types.ts`, `web/src/lib/api.ts`, `web/src/lib/ws.ts`

- [ ] **Step 1: TS types (mirror pydantic schemas from Tasks 2 & 5)**

```typescript
// web/src/lib/types.ts
export type Mode = "artist" | "genre" | "seeds" | "history";
export type AxisValue = "strict" | "narrow" | "dynamic" | "discover" | "off";

export interface GenerateRequestBody {
  mode: Mode;
  tracks: number;
  artist?: string;
  genre?: string;
  seed_tracks?: string[];
  seed_track_ids?: string[];
  genre_mode?: AxisValue;
  sonic_mode?: AxisValue;
  pace_mode?: "strict" | "narrow" | "dynamic";
  include_collaborations?: boolean;
}

export interface TrackOut {
  position: number;
  rating_key?: string;
  artist: string;
  title: string;
  album: string;
  duration_ms: number;
  file_path: string;
  sonic_similarity?: number | null;
  genre_similarity?: number | null;
  genres: string[];
}

export interface MetricsOut {
  mean_transition?: number | null;
  min_transition?: number | null;
  p10_transition?: number | null;
  p90_transition?: number | null;
  distinct_artists?: number | null;
}

export interface PlaylistOut {
  name: string;
  track_count: number;
  tracks: TrackOut[];
  metrics: MetricsOut;
}

export interface JobOut {
  job_id: string;
  status: "pending" | "running" | "success" | "failed" | "cancelled";
  stage: string;
  error?: string | null;
  playlist?: PlaylistOut | null;
}

export interface WsEvent {
  type: "log" | "progress" | "result" | "error" | "done";
  job_id?: string;
  [k: string]: unknown;
}
```

- [ ] **Step 2: API client**

```typescript
// web/src/lib/api.ts
import type { GenerateRequestBody, JobOut } from "./types";

async function jsonOrThrow(resp: Response) {
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

export const api = {
  async generate(body: GenerateRequestBody): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }));
  },
  async job(id: string): Promise<JobOut> {
    return jsonOrThrow(await fetch(`/api/jobs/${id}`));
  },
  async jobs(): Promise<JobOut[]> {
    return jsonOrThrow(await fetch("/api/jobs"));
  },
  async autocomplete(q: string): Promise<string[]> {
    return jsonOrThrow(await fetch(`/api/autocomplete?q=${encodeURIComponent(q)}`));
  },
};
```

- [ ] **Step 3: WS hook**

```typescript
// web/src/lib/ws.ts
import { useEffect, useRef } from "react";
import type { WsEvent } from "./types";

export function useWorkerEvents(onEvent: (e: WsEvent) => void) {
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (m) => {
      try { handler.current(JSON.parse(m.data) as WsEvent); } catch { /* ignore */ }
    };
    const ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
    return () => { clearInterval(ping); ws.close(); };
  }, []);
}
```

- [ ] **Step 4: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib
git commit -m "feat(web): typed api + websocket clients"
```

---

## Task 11: Shell layout (resizable panels)

**Files:**
- Create: `web/src/components/Shell.tsx`, `web/src/components/AdvancedPanel.tsx`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Shell with react-resizable-panels**

```tsx
// web/src/components/Shell.tsx
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type { ReactNode } from "react";

const handleV = "w-1.5 bg-bg hover:bg-accent transition-colors data-[resize-handle-state=drag]:bg-accent";
const handleH = "h-1.5 bg-bg hover:bg-accent transition-colors data-[resize-handle-state=drag]:bg-accent";

export function Shell(props: {
  topBar: ReactNode; jobs: ReactNode; center: ReactNode; right: ReactNode; logs: ReactNode;
}) {
  return (
    <div className="h-screen flex flex-col bg-bg text-text">
      <header className="flex items-center justify-between px-4 py-2.5 bg-panel border-b border-border">
        {props.topBar}
      </header>
      <PanelGroup direction="vertical" className="flex-1" autoSaveId="pg-vert">
        <Panel defaultSize={78} minSize={40}>
          <PanelGroup direction="horizontal" autoSaveId="pg-horiz">
            <Panel defaultSize={16} minSize={10} collapsible className="bg-panel border-r border-border">
              {props.jobs}
            </Panel>
            <PanelResizeHandle className={handleV} />
            <Panel defaultSize={62} minSize={30} className="bg-bg overflow-hidden">
              {props.center}
            </Panel>
            <PanelResizeHandle className={handleV} />
            <Panel defaultSize={22} minSize={14} collapsible className="bg-panel border-l border-border">
              {props.right}
            </Panel>
          </PanelGroup>
        </Panel>
        <PanelResizeHandle className={handleH} />
        <Panel defaultSize={22} minSize={8} collapsible className="bg-[#0c0e12] border-t border-border">
          {props.logs}
        </Panel>
      </PanelGroup>
    </div>
  );
}
```

```tsx
// web/src/components/AdvancedPanel.tsx
import { useState } from "react";

export function AdvancedPanel() {
  const [tab, setTab] = useState<"advanced" | "review">("advanced");
  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-1 px-2 pt-2 bg-panel2">
        {(["advanced", "review"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`text-[11px] px-2.5 py-1.5 rounded-t ${tab === t ? "text-accent bg-bg" : "text-muted"}`}>
            {t === "advanced" ? "Advanced" : "Genre Review"}
          </button>
        ))}
      </div>
      <div className="p-3 text-xs text-muted">
        {tab === "advanced"
          ? "Advanced settings land in a later phase."
          : "Genre review lands in a later phase."}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire App with placeholders**

```tsx
// web/src/App.tsx
import { Shell } from "./components/Shell";
import { AdvancedPanel } from "./components/AdvancedPanel";

export default function App() {
  return (
    <Shell
      topBar={<div className="font-bold text-sm"><span className="text-accent">◆</span> Playlist Generator</div>}
      jobs={<div className="p-3 text-xs text-muted">Jobs</div>}
      center={<div className="p-3 text-xs text-muted">Center</div>}
      right={<AdvancedPanel />}
      logs={<div className="p-3 font-mono text-[11px] text-faint">Logs</div>}
    />
  );
}
```

- [ ] **Step 3: Verify build + dev render**

Run: `cd web && npx tsc --noEmit && npm run build`
Expected: type-check clean, build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/Shell.tsx web/src/components/AdvancedPanel.tsx web/src/App.tsx
git commit -m "feat(web): resizable panel shell + advanced tab placeholder"
```

---

## Task 12: Generate controls

**Files:**
- Create: `web/src/components/GenerateControls.tsx`

- [ ] **Step 1: Controls component (mode, seed input + autocomplete, track count, four axes, Generate)**

```tsx
// web/src/components/GenerateControls.tsx
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { AxisValue, GenerateRequestBody, Mode } from "../lib/types";

const AXES: { key: keyof GenerateRequestBody; label: string; values: string[] }[] = [
  { key: "genre_mode", label: "genre", values: ["off", "discover", "dynamic", "narrow", "strict"] },
  { key: "sonic_mode", label: "sonic", values: ["off", "dynamic", "narrow", "strict"] },
  { key: "pace_mode", label: "pace", values: ["dynamic", "narrow", "strict"] },
];

export function GenerateControls(props: { onSubmit: (body: GenerateRequestBody) => void; busy: boolean }) {
  const [mode, setMode] = useState<Mode>("artist");
  const [seed, setSeed] = useState("");
  const [tracks, setTracks] = useState(30);
  const [axes, setAxes] = useState<Record<string, string>>({ genre_mode: "dynamic", sonic_mode: "dynamic", pace_mode: "dynamic" });
  const [cohesion, setCohesion] = useState("dynamic");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const timer = useRef<number>();

  useEffect(() => {
    if (mode !== "artist" || seed.length < 2) { setSuggestions([]); return; }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setSuggestions(await api.autocomplete(seed).catch(() => []));
    }, 180);
  }, [seed, mode]);

  function submit() {
    const body: GenerateRequestBody = {
      mode, tracks,
      artist: mode === "artist" ? seed : undefined,
      genre: mode === "genre" ? seed : undefined,
      seed_tracks: mode === "seeds" ? seed.split(",").map((s) => s.trim()).filter(Boolean) : undefined,
      genre_mode: axes.genre_mode as AxisValue,
      sonic_mode: axes.sonic_mode as AxisValue,
      pace_mode: axes.pace_mode as "strict" | "narrow" | "dynamic",
    };
    props.onSubmit(body);
  }

  return (
    <div className="flex flex-wrap items-center gap-2 px-3 py-2 bg-panel2 border-b border-border">
      <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}
        className="bg-[#0c0e12] border border-border rounded text-xs text-text px-2 py-1.5">
        <option value="artist">artist</option>
        <option value="seeds">seeds</option>
        <option value="genre">genre</option>
        <option value="history">history</option>
      </select>

      <div className="relative flex-1 min-w-[180px]">
        <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="Acetone, Mazzy Star"
          className="w-full bg-[#0c0e12] border border-border rounded text-xs text-text px-2.5 py-1.5" />
        {suggestions.length > 0 && (
          <ul className="absolute z-10 mt-1 w-full bg-panel border border-border rounded shadow-xl max-h-48 overflow-auto">
            {suggestions.map((s) => (
              <li key={s} onClick={() => { setSeed(s); setSuggestions([]); }}
                className="px-2.5 py-1.5 text-xs text-text hover:bg-border cursor-pointer">{s}</li>
            ))}
          </ul>
        )}
      </div>

      <input type="number" min={1} max={200} value={tracks} onChange={(e) => setTracks(Number(e.target.value))}
        className="w-16 bg-[#0c0e12] border border-border rounded text-xs text-text px-2 py-1.5 text-center" />

      <select value={cohesion} onChange={(e) => setCohesion(e.target.value)}
        title="cohesion mode"
        className="bg-[#0c0e12] border border-border rounded text-xs text-muted px-2 py-1.5">
        {["strict", "narrow", "dynamic", "discover"].map((v) => <option key={v} value={v}>cohesion · {v}</option>)}
      </select>

      {AXES.map((a) => (
        <select key={a.key} value={axes[a.key]} onChange={(e) => setAxes({ ...axes, [a.key]: e.target.value })}
          className="bg-[#0c0e12] border border-border rounded text-xs text-muted px-2 py-1.5">
          {a.values.map((v) => <option key={v} value={v}>{a.label} · {v}</option>)}
        </select>
      ))}

      <button onClick={submit} disabled={props.busy}
        className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50">
        {props.busy ? "Generating…" : "▸ Generate"}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/GenerateControls.tsx
git commit -m "feat(web): generate controls with mode, axes, autocomplete"
```

---

## Task 13: Track table + quality stats

**Files:**
- Create: `web/src/components/TrackTable.tsx`, `web/src/components/QualityStats.tsx`

- [ ] **Step 1: Track table (TanStack)**

```tsx
// web/src/components/TrackTable.tsx
import { createColumnHelper, flexRender, getCoreRowModel, getSortedRowModel, useReactTable, type SortingState } from "@tanstack/react-table";
import { useState } from "react";
import type { TrackOut } from "../lib/types";

const col = createColumnHelper<TrackOut>();
const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

const columns = [
  col.accessor("position", { header: "#", cell: (c) => <span className="font-mono text-faint text-[10px]">{String(c.getValue() + 1).padStart(2, "0")}</span> }),
  col.accessor("title", {
    header: "Track",
    cell: (c) => (
      <div>
        <div className="text-text text-xs">
          {c.getValue()}
          {c.row.original.genres.slice(0, 2).map((g) => (
            <span key={g} className="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full">{g}</span>
          ))}
        </div>
        <div className="text-muted text-[10px]">{c.row.original.artist}</div>
      </div>
    ),
  }),
  col.accessor("sonic_similarity", { header: "T", cell: (c) => <span className="font-mono text-accent text-[11px]">{fmt(c.getValue())}</span> }),
];

export function TrackTable({ tracks }: { tracks: TrackOut[] }) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const table = useReactTable({
    data: tracks, columns, state: { sorting }, onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel(),
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
              <th key={h.id} onClick={h.column.getToggleSortingHandler()}
                className="px-3 py-2 text-[9px] uppercase tracking-wide text-faint cursor-pointer select-none">
                {flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((r) => (
          <tr key={r.id} className="border-b border-[#181b21] odd:bg-panel2 hover:bg-[#15202b]">
            {r.getVisibleCells().map((cell) => (
              <td key={cell.id} className="px-3 py-2 align-top">{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Quality stats**

```tsx
// web/src/components/QualityStats.tsx
import type { MetricsOut } from "../lib/types";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export function QualityStats({ metrics, count }: { metrics?: MetricsOut; count: number }) {
  if (!metrics || count === 0) return null;
  const stat = (label: string, value: string) => (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-faint">{label}</span>
      <span className="font-mono text-accent text-xs">{value}</span>
    </div>
  );
  return (
    <div className="flex gap-5 px-3 py-2 border-b border-border bg-panel2">
      {stat("tracks", String(count))}
      {stat("mean T", fmt(metrics.mean_transition))}
      {stat("min T", fmt(metrics.min_transition))}
      {metrics.p10_transition != null && stat("p10", fmt(metrics.p10_transition))}
      {metrics.p90_transition != null && stat("p90", fmt(metrics.p90_transition))}
      {stat("distinct artists", String(metrics.distinct_artists ?? "—"))}
    </div>
  );
}
```

- [ ] **Step 3: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/TrackTable.tsx web/src/components/QualityStats.tsx
git commit -m "feat(web): track table + quality stats"
```

---

## Task 14: Logs + jobs panels and full wiring

**Files:**
- Create: `web/src/components/LogPanel.tsx`, `web/src/components/JobsPanel.tsx`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Log + jobs panels**

```tsx
// web/src/components/LogPanel.tsx
import { useEffect, useRef } from "react";

export function LogPanel({ lines }: { lines: string[] }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView(); }, [lines.length]);
  return (
    <div className="h-full overflow-auto px-3 py-2 font-mono text-[10px] leading-relaxed text-faint" data-testid="log-panel">
      {lines.map((l, i) => (
        <div key={i} className={l.startsWith("ERROR") ? "text-danger" : l.startsWith("WARNING") ? "text-warn" : ""}>{l}</div>
      ))}
      <div ref={end} />
    </div>
  );
}
```

```tsx
// web/src/components/JobsPanel.tsx
import type { JobOut } from "../lib/types";

const dot: Record<string, string> = {
  success: "text-accent", running: "text-warn", failed: "text-danger", cancelled: "text-muted", pending: "text-muted",
};

export function JobsPanel({ jobs, onSelect }: { jobs: JobOut[]; onSelect: (j: JobOut) => void }) {
  return (
    <div className="h-full overflow-auto">
      <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-faint border-b border-border">Jobs</div>
      {jobs.map((j) => (
        <button key={j.job_id} onClick={() => onSelect(j)}
          className="w-full text-left px-3 py-2 border-b border-[#181b21] hover:bg-border">
          <div className="text-[11px] text-text truncate">{j.playlist?.name ?? j.stage ?? "Playlist"}</div>
          <div className={`text-[9px] ${dot[j.status] ?? "text-muted"}`}>{j.status}</div>
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Full App wiring (state, WS, submit)**

```tsx
// web/src/App.tsx  (replace)
import { useCallback, useState } from "react";
import { Shell } from "./components/Shell";
import { AdvancedPanel } from "./components/AdvancedPanel";
import { GenerateControls } from "./components/GenerateControls";
import { TrackTable } from "./components/TrackTable";
import { QualityStats } from "./components/QualityStats";
import { LogPanel } from "./components/LogPanel";
import { JobsPanel } from "./components/JobsPanel";
import { api } from "./lib/api";
import { useWorkerEvents } from "./lib/ws";
import type { GenerateRequestBody, JobOut, PlaylistOut, WsEvent } from "./lib/types";

export default function App() {
  const [busy, setBusy] = useState(false);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [playlist, setPlaylist] = useState<PlaylistOut | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [jobs, setJobs] = useState<JobOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshJobs = useCallback(() => { api.jobs().then(setJobs).catch(() => {}); }, []);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (e.type === "log") setLogs((l) => [...l, `${(e as any).level ?? "INFO"}: ${(e as any).msg ?? ""}`].slice(-500));
    if (e.type === "error") setError(String((e as any).message ?? "error"));
    if (e.type === "done") {
      setBusy(false);
      if (e.job_id) api.job(e.job_id).then((j) => { if (j.playlist) setPlaylist(j.playlist); }).catch(() => {});
      refreshJobs();
    }
  }, [refreshJobs]));

  async function submit(body: GenerateRequestBody) {
    setError(null); setBusy(true); setLogs([]); setPlaylist(null);
    try {
      const { job_id } = await api.generate(body);
      setActiveJob(job_id);
      refreshJobs();
    } catch (err) {
      setError(String(err)); setBusy(false);
    }
  }

  return (
    <Shell
      topBar={
        <>
          <div className="font-bold text-sm"><span className="text-accent">◆</span> Playlist Generator</div>
          {error && <div className="text-danger text-xs">{error}</div>}
        </>
      }
      jobs={<JobsPanel jobs={jobs} onSelect={(j) => { setActiveJob(j.job_id); setPlaylist(j.playlist ?? null); }} />}
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
  );
}
```

- [ ] **Step 3: Build + type-check**

Run: `cd web && npx tsc --noEmit && npm run build`
Expected: clean type-check, successful build emitting `web/dist/`.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/LogPanel.tsx web/src/components/JobsPanel.tsx web/src/App.tsx
git commit -m "feat(web): logs + jobs panels, full generate wiring"
```

---

## Task 15: End-to-end Playwright smoke + verification

**Files:**
- Create: `web/tests/generate.spec.ts`, `web/playwright.config.ts`
- Modify: `web/package.json` (add `@playwright/test` dev dep + `test:e2e` script)

This is the gate that proves the Generate loop works against a real server + the fake worker.

- [ ] **Step 1: Install Playwright**

```bash
cd web && npm install -D @playwright/test && npx playwright install chromium
```

- [ ] **Step 2: Playwright config — build front-end, run FastAPI with the fake worker, serve `web/dist`**

```typescript
// web/playwright.config.ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  use: { baseURL: "http://127.0.0.1:8771" },
  webServer: {
    // Build the SPA, then start FastAPI pointed at the fake worker on port 8771.
    command: "npm run build && cd .. && python tools/serve_web.py --port 8771 --no-browser",
    url: "http://127.0.0.1:8771/api/health",
    timeout: 60000,
    reuseExistingServer: false,
    env: { PG_WEB_WORKER_CMD: "" },
  },
});
```

(Note: `serve_web.py` must honor a `PG_WEB_WORKER_CMD` env var so the smoke test injects the fake worker. Add this in Step 3.)

- [ ] **Step 3: Teach the launcher + app to read the worker cmd from env**

In `tools/serve_web.py`, before `uvicorn.run`, read the override:

```python
import os, shlex
worker_cmd_env = os.environ.get("PG_WEB_WORKER_CMD", "").strip()
worker_cmd = shlex.split(worker_cmd_env) if worker_cmd_env else None
...
uvicorn.run(create_app(worker_cmd=worker_cmd), host=args.host, port=args.port, log_level="info")
```

And set the Playwright env to the fake worker:

```typescript
// web/playwright.config.ts — replace env line
env: { PG_WEB_WORKER_CMD: `${process.platform === "win32" ? "python" : "python3"} tests/fixtures/fake_worker.py` },
```

(The `command` already `cd ..` into repo root before launching, so the relative `tests/fixtures/fake_worker.py` resolves.)

- [ ] **Step 4: Write the smoke test**

```typescript
// web/tests/generate.spec.ts
import { test, expect } from "@playwright/test";

test("generate loop renders tracks, stats, and logs", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Playlist Generator")).toBeVisible();

  // mode defaults to artist; type a seed and generate
  await page.locator('input[placeholder="Acetone, Mazzy Star"]').fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();

  // fake worker returns a 2-track playlist
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 10000 });
  await expect(page.getByText("Sundown")).toBeVisible();
  await expect(page.getByText("Taxi")).toBeVisible();

  // quality stats rendered
  await expect(page.getByText("distinct artists")).toBeVisible();

  // logs streamed over websocket
  await expect(page.getByTestId("log-panel")).toContainText("fake: starting");
});
```

- [ ] **Step 5: Add the script to package.json**

```jsonc
// web/package.json — under "scripts"
"test:e2e": "playwright test"
```

- [ ] **Step 6: Run the smoke test**

Run: `cd web && npm run test:e2e`
Expected: PASS — the test builds the SPA, boots FastAPI with the fake worker, and verifies tracks, stats, and streamed logs render.

- [ ] **Step 7: Manual verification against the REAL worker (per global rigor rule — verify before claiming success)**

Run: `python tools/serve_web.py` then, in the browser, generate a playlist for a real artist in your library.
Expected: a real playlist renders with live logs. Note any gaps for the Phase 1 punch-list (the spec explicitly anticipates surfacing gaps through use). Do NOT claim Phase 1 complete until this real-worker run is observed.

- [ ] **Step 8: Commit**

```bash
git add web/tests web/playwright.config.ts web/package.json tools/serve_web.py
git commit -m "test(web): playwright e2e smoke for the generate loop"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- §4 stack → Tasks 9–10 (Vite/TS/Tailwind/shadcn-ready, react-resizable-panels, TanStack). shadcn primitives are installed on demand in Phase 2 when context menus/dialogs land; Phase 1 needs no shadcn component yet, so none are pre-installed (YAGNI).
- §5 architecture (asyncio NDJSON client, worker reuse, `GeneratePlaylistRequest`, result/metrics shapes) → Tasks 2, 4, 5, 7.
- §6 Studio Dark → Task 9.
- §7 shell (resizable, collapsible, tabs; jobs left / center / right / logs bottom) → Task 11. Slide-over + context-menu patterns are Phase 2/3 (not built here, correctly).
- §8 Generate workspace (modes, autocomplete, track count, four axes, table, stats, logs, read-only jobs) → Tasks 12–14.
- §9 API surface (generate, jobs, jobs/{id}, autocomplete, health, /ws) → Tasks 1, 7.
- §10 genre-mode exposed → Task 12 (genre axis present).
- §12 testing (pytest + fake worker, Playwright, contract types, real-worker verification) → Tasks 3–7, 15.

**Placeholder scan:** No "TBD"/"implement later"; the two Tailwind-version and shadcn-version notes are explicit decision points the executing agent resolves against what `npm install` produced, with the action specified. The Advanced/Genre-Review tab bodies are intentionally placeholder *content* (per spec — populated in later phases), not plan placeholders.

**Type consistency:** `WorkerBridge(worker_cmd, on_event)`, `.submit(cmd)→request_id`, `.running`, `.busy`, `BridgeBusy` consistent across Tasks 4 & 7. `JobRegistry.create/apply_event/get/logs/recent` consistent across Tasks 5 & 7. `create_app(worker_cmd, config_path)` consistent across Tasks 1, 7, 8, 15. TS `JobOut`/`PlaylistOut`/`TrackOut`/`MetricsOut` mirror the pydantic models. Status strings (`pending/running/success/failed/cancelled`) consistent between `JobRegistry` and TS `JobOut`.

**Known follow-ups for Phase 2 (not gaps):** cancellation endpoint (worker supports it via `cancel`), genre-mode correctness (upstream fix), shadcn install, context menus, slide-overs, export.
