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

from src.playlist_gui.policy import derive_runtime_config
from src.playlist_gui.ui_state import UIStateModel

from .audio import stream_audio
from .jobs import JobRegistry
from .plex_export import PlexNotConfigured, run_plex_export
from .schemas import (
    BlacklistArtistRequest,
    BlacklistFetchResponse,
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
        job_id = registry.create(request_params=body.model_dump())
        ui = UIStateModel(
            mode=body.mode,  # type: ignore[arg-type]
            cohesion_mode=body.cohesion_mode or "dynamic",  # type: ignore[arg-type]
            genre_mode=body.genre_mode or "dynamic",  # type: ignore[arg-type]
            sonic_mode=body.sonic_mode or "dynamic",  # type: ignore[arg-type]
            pace_mode=body.pace_mode or "dynamic",  # type: ignore[arg-type]
            track_count=body.tracks,
            recency_enabled=body.recency_enabled,
            recency_days=body.recency_days,
            recency_plays_threshold=body.recency_plays_threshold,
            artist_spacing=body.artist_spacing,  # type: ignore[arg-type]
            diversity_gamma=body.diversity_gamma,
            artist_diversity_mode=body.artist_diversity_mode,  # type: ignore[arg-type]
            artist_presence=body.artist_presence,  # type: ignore[arg-type]
            artist_variety=body.artist_variety,  # type: ignore[arg-type]
        )
        overrides = derive_runtime_config(ui).overrides
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

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "running":
            raise HTTPException(status_code=409, detail="Job is not running")
        await bridge.cancel()
        return {"ok": True}

    @app.get("/api/tracks/search")
    async def track_search(q: str = "", limit: int = 15) -> list[dict]:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return []
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
                    LIMIT ?
                    """,
                    (pattern, pattern, pattern, limit),
                ).fetchall()
                results = []
                for row in rows:
                    track_id = row[0]
                    genres = [g[0] for g in conn.execute(
                        "SELECT genre FROM track_effective_genres WHERE track_id = ? ORDER BY priority LIMIT 5",
                        (track_id,),
                    ).fetchall()]
                    if not genres:
                        genres = [g[0] for g in conn.execute(
                            "SELECT genre FROM track_genres WHERE track_id = ? ORDER BY weight DESC LIMIT 5",
                            (track_id,),
                        ).fetchall()]
                    results.append({
                        "track_id": track_id,
                        "title": row[1] or "Unknown",
                        "artist": row[2] or "Unknown",
                        "album": row[3] or "",
                        "duration_ms": row[4] or 0,
                        "file_path": row[5] or "",
                        "genres": genres,
                    })
                return results
            finally:
                conn.close()
        except Exception:
            return []

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

    @app.get("/api/blacklist")
    async def get_blacklist() -> BlacklistFetchResponse:
        try:
            result = await bridge.command({
                "cmd": "blacklist_fetch_scopes",
                "base_config_path": config_path,
                "overrides": {},
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return BlacklistFetchResponse.from_worker(result)

    @app.post("/api/blacklist/artist")
    async def blacklist_artist(body: BlacklistArtistRequest) -> dict:
        if not body.artist.strip():
            raise HTTPException(status_code=422, detail="artist is required")
        try:
            result = await bridge.command({
                "cmd": "blacklist_scope_set",
                "base_config_path": config_path,
                "overrides": {},
                "scope": "artist",
                "value": body.artist,
                "artist": body.artist,
                "enabled": True,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

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

    @app.post("/api/export/plex")
    async def export_plex(body: PlexExportRequest) -> dict:
        try:
            key = run_plex_export(body.title, body.tracks, config_path)
        except PlexNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any export failure to the client
            raise HTTPException(status_code=502, detail=f"Plex export failed: {exc}")
        return {"ok": True, "playlist_key": key}

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
