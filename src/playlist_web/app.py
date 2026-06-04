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
