# src/playlist_web/app.py
from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.playlist_gui.policy import derive_runtime_config
from src.playlist_gui.ui_state import UIStateModel

from .audio import stream_audio
from .jobs import JobRegistry
from .plex_export import PlexNotConfigured, run_plex_export
from .schemas import (
    AnalyzeToolRequest,
    BlacklistArtistRequest,
    BlacklistFetchResponse,
    BlacklistRequest,
    EditGenresRequest,
    EnrichToolRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsRequest,
    ReplaceSuggestionsResponse,
    ReviewDecisionRequest,
    TrackGenresRequest,
)
from .worker_bridge import BridgeBusy, WorkerBridge, WorkerCommandError
from .ws import WsHub

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_CMD = [sys.executable, "-m", "src.playlist_gui.worker"]
DEFAULT_CONFIG = str(ROOT / "config.yaml")
DB_PATH = ROOT / "data" / "metadata.db"
SIDECAR_DB_PATH = ROOT / "data" / "ai_genre_enrichment.db"
WEB_DIST = ROOT / "web" / "dist"

logger = logging.getLogger(__name__)


def _resolve_seed_artist_keys(track_ids: list[str]) -> list[str]:
    """Resolve seed track IDs to artist keys for policy evaluation (read-only).

    Policy gates DJ bridging on >= 2 unique seed artists; without this lookup
    it conservatively force-disables DJ bridging on every web request.
    """
    if not track_ids:
        return []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" for _ in track_ids)
            rows = conn.execute(
                "SELECT track_id, COALESCE(NULLIF(artist_key, ''), artist) "
                f"FROM tracks WHERE track_id IN ({placeholders})",
                [str(t) for t in track_ids],
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        logger.warning(
            "Seed artist resolution failed; policy will keep DJ bridging disabled",
            exc_info=True,
        )
        return []
    by_id = {str(r[0]): str(r[1] or "") for r in rows}
    return [by_id.get(str(t), "") for t in track_ids]


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


def create_app(
    worker_cmd: Optional[list[str]] = None,
    config_path: str = DEFAULT_CONFIG,
    seed_artist_resolver: Optional[Callable[[list[str]], list[str]]] = None,
) -> FastAPI:
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
            seed_track_ids=list(body.seed_track_ids),
            recency_enabled=body.recency_enabled,
            recency_days=body.recency_days,
            recency_plays_threshold=body.recency_plays_threshold,
            artist_spacing=body.artist_spacing,  # type: ignore[arg-type]
            diversity_gamma=body.diversity_gamma,
            artist_diversity_mode=body.artist_diversity_mode,  # type: ignore[arg-type]
            artist_presence=body.artist_presence,  # type: ignore[arg-type]
            artist_variety=body.artist_variety,  # type: ignore[arg-type]
        )
        # Resolve seed artists so the policy can evaluate DJ-bridging
        # eligibility (it conservatively disables when keys are missing).
        seed_artist_keys: Optional[list[str]] = None
        if body.mode == "seeds" and body.seed_track_ids:
            resolver = seed_artist_resolver or _resolve_seed_artist_keys
            seed_artist_keys = [k for k in resolver(list(body.seed_track_ids)) if k] or None
        policy = derive_runtime_config(ui, seed_artist_keys=seed_artist_keys)
        for note in policy.notes:
            logger.info("Policy: %s", note)
        overrides = policy.overrides
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

    @app.post("/api/tools/analyze")
    async def tools_analyze(body: AnalyzeToolRequest) -> dict:
        job_id = registry.create(request_params=body.model_dump())
        try:
            await bridge.submit({
                "cmd": "analyze_library",
                "job_id": job_id,
                "base_config_path": config_path,
                "overrides": {},
                "stages": body.stages or None,
                "force": body.force,
                "dry_run": body.dry_run,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.post("/api/tools/enrich")
    async def tools_enrich(body: EnrichToolRequest) -> dict:
        job_id = registry.create(request_params=body.model_dump())
        try:
            await bridge.submit({
                "cmd": "enrich_genres",
                "job_id": job_id,
                "scope": body.scope,
                "artist": body.artist,
                "album": body.album,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.post("/api/review/scan")
    async def review_scan() -> dict:
        job_id = registry.create(request_params={"tool": "scan_genre_review"})
        try:
            await bridge.submit({"cmd": "scan_genre_review", "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.get("/api/review/queue")
    async def review_queue(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            result = await bridge.command({
                "cmd": "get_genre_review_queue",
                "search": search,
                "limit": limit,
                "offset": offset,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return result

    @app.post("/api/review/decision")
    async def review_decision(body: ReviewDecisionRequest) -> dict:
        if not body.release_key.strip() or not body.term.strip():
            raise HTTPException(status_code=422, detail="release_key and term are required")
        try:
            result = await bridge.command({
                "cmd": "apply_genre_review_decision",
                "release_key": body.release_key,
                "term": body.term,
                "decision": body.decision,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.get("/api/tracks/search")
    async def track_search(
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(25, ge=1, le=200),
    ) -> dict:
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

    @app.get("/api/autocomplete")
    async def autocomplete(
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(30, ge=1, le=200),
    ) -> dict:
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
