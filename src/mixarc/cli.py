"""`mixarc` console entry point: start the web app, open the browser.

The repo dev launcher (tools/serve_web.py) delegates here so both paths share
one server-start implementation; only the frontend-build default differs.
"""
from __future__ import annotations

import argparse
import os
import shlex
import threading
import time
import webbrowser
from pathlib import Path

_REPO_WEB = Path(__file__).resolve().parents[2] / "web"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mixarc", description="MixArc — playlist generator for your own library")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--config", default=None, help="explicit path to config.yaml (overrides MIXARC_HOME and defaults)")
    ap.add_argument("--dev", action="store_true",
                    help="repo checkouts only: rebuild web/dist via npm before serving")
    return ap


def run_server(host: str, port: int, open_browser: bool,
               worker_cmd: list[str] | None, config_path: str | None) -> None:
    import uvicorn
    from src.playlist_web.app import create_app

    url = f"http://{host}:{port}/"
    if open_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    print(f"MixArc → {url}")
    kwargs = {} if config_path is None else {"config_path": config_path}
    # timeout_graceful_shutdown: bound uvicorn's wait on the persistent /ws
    # WebSocket so lifespan shutdown always runs and the worker never orphans
    # (worker-orphan incident 2026-06-12).
    uvicorn.run(
        create_app(worker_cmd=worker_cmd, **kwargs),
        host=host, port=port, log_level="info", timeout_graceful_shutdown=5,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dev and _REPO_WEB.exists():
        from tools.serve_web import ensure_frontend_built  # repo checkout only
        ensure_frontend_built()

    worker_cmd_env = os.environ.get("PG_WEB_WORKER_CMD", "").strip()
    worker_cmd = shlex.split(worker_cmd_env) if worker_cmd_env else None
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser,
               worker_cmd=worker_cmd, config_path=args.config)
