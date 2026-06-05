# src/playlist_web/app.py
from __future__ import annotations

import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import stream_audio
from .jobs import JobRegistry
from .schemas import (
    BlacklistRequest,
    EditGenresRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsRequest,
    ReplaceSuggestionsResponse,
)
from .worker_bridge import BridgeBusy, WorkerBridge, WorkerCommandError
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
        overrides: dict = {}
        if body.cohesion_mode:
            overrides = {"playlists": {"cohesion_mode": body.cohesion_mode}}
        try:
            await bridge.submit({
                "cmd": "generate_playlist",
                "job_id": job_id,
                "base_config_path": config_path,
                "overrides": overrides,
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

    @app.get("/api/audio/{track_id}")
    async def audio(track_id: str, request: Request):
        return stream_audio(track_id, DB_PATH, request)

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
