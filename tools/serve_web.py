"""Launch the browser playlist GUI: start FastAPI (which owns the worker) and open the browser."""
from __future__ import annotations

import argparse
import os
import shlex
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

    # Allow test harness to inject a fake worker via env var
    worker_cmd_env = os.environ.get("PG_WEB_WORKER_CMD", "").strip()
    worker_cmd = shlex.split(worker_cmd_env) if worker_cmd_env else None

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    print(f"Playlist Generator (web) → {url}")
    # timeout_graceful_shutdown bounds how long uvicorn waits for open connections
    # on Ctrl+C. The browser holds a persistent /ws WebSocket open; without a bound,
    # uvicorn waits indefinitely for it to close, so the lifespan shutdown (which
    # stops the worker) never runs and the worker subprocess orphans. See the
    # worker-orphan incident (2026-06-12).
    uvicorn.run(
        create_app(worker_cmd=worker_cmd),
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
