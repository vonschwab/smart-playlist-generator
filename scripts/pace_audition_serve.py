# scripts/pace_audition_serve.py
"""Local HTTP server for the pace audition. Streams whole audio files with
range support; the page seeks client-side to play A-tail -> hard cut -> B-head.
Serves the BLINDED manifest (no arm/seed) and captures dual scores to YAML.

Usage:
    python scripts/pace_audition_serve.py [--port 8767] [--data-dir docs/run_audits/pace_audition]
"""
from __future__ import annotations

import datetime
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTENT_TYPES = {".flac": "audio/flac", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
                 ".ogg": "audio/ogg", ".wav": "audio/wav"}


def _parse_range_header(header: str, file_size: int) -> tuple[int, int]:
    if not header or not header.startswith("bytes="):
        return (0, file_size - 1)
    parts = header[6:].split("-")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    return (start, min(end, file_size - 1))


def blinded_manifest(manifest: dict) -> dict:
    """Served view: ONLY type + edges (ids). Strips edge_data/playlists/
    provenance/file_paths so no arm or seed reaches the browser."""
    return {"type": manifest.get("type", "pace_edges"), "edges": manifest.get("edges", [])}


def upsert_capture_entry(entries: list, entry: dict) -> None:
    for i, e in enumerate(entries):
        if e.get("edge_id") == entry["edge_id"]:
            entries[i] = entry
            return
    entries.append(entry)


def _append_capture(capture_path: Path, entry: dict) -> None:
    data = yaml.safe_load(capture_path.read_text(encoding="utf-8")) if capture_path.exists() else {}
    data = data or {}
    entries = data.get("entries", [])
    upsert_capture_entry(entries, entry)
    data["entries"] = entries
    capture_path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")


class PaceServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler_class, data_dir: Path, manifest: dict, page_html: str):
        super().__init__(addr, handler_class)
        self.data_dir = data_dir
        self.manifest = manifest
        self.file_paths = manifest.get("file_paths", {})
        self.page_html = page_html


class PaceHandler(BaseHTTPRequestHandler):
    server: PaceServer

    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path == "/":
            body = self.server.page_html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/manifest":
            self._json(blinded_manifest(self.server.manifest))
        elif path == "/api/progress":
            cap = self.server.data_dir / "pace_capture.yaml"
            data = (yaml.safe_load(cap.read_text(encoding="utf-8")) or {}) if cap.exists() else {}
            self._json(data.get("entries", []))
        elif path.startswith("/audio/"):
            tid = path[7:]
            fp = self.server.file_paths.get(tid)
            if not fp:
                self.send_error(404, f"track {tid!r} not in manifest")
                return
            self._serve_audio(fp)
        else:
            self.send_error(404)

    def do_POST(self):
        if unquote(self.path) != "/api/save":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        eid = body.get("edge_id", "")
        if not eid:
            self._json({"ok": False, "error": "missing edge_id"}, 400)
            return
        entry = {
            "edge_id": eid,
            "continuity": body.get("continuity"),
            "smoothness": body.get("smoothness"),
            "notes": body.get("notes", ""),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        _append_capture(self.server.data_dir / "pace_capture.yaml", entry)
        self._json({"ok": True})

    def _serve_audio(self, file_path: str):
        p = Path(file_path)
        if not p.exists():
            self.send_error(404, "audio not on disk")
            return
        size = p.stat().st_size
        rng = self.headers.get("Range", "")
        start, end = _parse_range_header(rng, size)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    import argparse
    import webbrowser

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--data-dir", default="docs/run_audits/pace_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    mpath = data_dir / "pace_manifest.json"
    if not mpath.exists():
        print(f"No manifest at {mpath}. Run: python scripts/pace_audition_build.py")
        sys.exit(1)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    page = (Path(__file__).parent / "pace_audition_page.html").read_text(encoding="utf-8")

    try:
        server = PaceServer(("127.0.0.1", args.port), PaceHandler, data_dir, manifest, page)
    except OSError as e:
        print(f"Could not bind port {args.port} ({e}). Another audition server may be "
              f"running there (sonic=8765, genre=8766). Retry: --port <free-port>.")
        sys.exit(1)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving pace audition at {url} ({len(manifest['edges'])} edges). Ctrl-C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
