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

WINDOW_SEC = 12.0  # seconds of A-tail / B-head served per transition side


def read_window_wav(file_path: str, pos: str, window_sec: float = WINDOW_SEC) -> bytes:
    """Return the tail (pos='tail') or head (pos='head') window of an audio file as
    16-bit PCM WAV bytes.

    The library is mostly FLAC, ~30% of it 24-bit or hi-res (96/192 kHz), which the
    HTML5 <audio> element cannot decode. Transcoding the needed window to 16-bit WAV
    makes every clip universally playable AND sidesteps fragile client-side FLAC
    time-seeking by extracting the window server-side (soundfile seeks by frame)."""
    import io
    import soundfile as sf

    info = sf.info(file_path)
    n = int(float(window_sec) * info.samplerate)
    if pos == "tail":
        start, stop = max(0, info.frames - n), info.frames
    else:  # head
        start, stop = 0, min(info.frames, n)
    data, sr = sf.read(file_path, start=start, stop=stop, dtype="int16")
    bio = io.BytesIO()
    sf.write(bio, data, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


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
            rest = path[len("/audio/"):]
            if "/" not in rest:
                self.send_error(400, "expected /audio/<track_id>/<tail|head>")
                return
            tid, pos = rest.rsplit("/", 1)
            if pos not in ("tail", "head"):
                self.send_error(400, "pos must be 'tail' or 'head'")
                return
            fp = self.server.file_paths.get(tid)
            if not fp:
                self.send_error(404, f"track {tid!r} not in manifest")
                return
            self._serve_window(fp, pos)
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

    def _serve_window(self, file_path: str, pos: str):
        p = Path(file_path)
        if not p.exists():
            self.send_error(404, "audio not on disk")
            return
        try:
            wav = read_window_wav(str(p), pos)
        except Exception as e:  # report to client, keep the server alive
            self.send_error(500, f"transcode failed: {type(e).__name__}: {e}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(wav)


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
