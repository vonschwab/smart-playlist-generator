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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW_SEC = 12.0  # seconds of A-tail / B-head served per transition side
SCAN_SEC = 120.0   # how far in from each end to scan for the musical body
TOP_DB = 40.0      # frames quieter than this below the loudest frame are "silence"


def _active_span(mono, sr, top_db: float = TOP_DB, hop_ms: float = 20.0):
    """Return (start, stop) sample indices of the energetic region of a mono signal,
    trimming leading/trailing near-silence (fades, silent intros/outros). A frame is
    'active' if its RMS is within top_db of the loudest frame."""
    import numpy as np

    n = int(len(mono))
    if n == 0:
        return 0, 0
    hop = max(1, int(sr * hop_ms / 1000.0))
    nframes = (n + hop - 1) // hop
    pad = nframes * hop - n
    if pad:
        mono = np.concatenate([mono, np.zeros(pad, dtype=mono.dtype)])
    frames = mono.reshape(nframes, hop).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return 0, n
    thr = peak * (10.0 ** (-float(top_db) / 20.0))
    active = np.where(rms > thr)[0]
    if len(active) == 0:
        return 0, n
    start = int(active[0]) * hop
    stop = min(n, (int(active[-1]) + 1) * hop)
    return start, stop


def read_window_wav(
    file_path: str, pos: str, window_sec: float = WINDOW_SEC,
    scan_sec: float = SCAN_SEC, top_db: float = TOP_DB,
) -> bytes:
    """Return the tail (pos='tail') or head (pos='head') window of a track as 16-bit
    PCM WAV bytes, taken from the track's MUSICAL BODY (leading/trailing silence,
    fade-ins/outs and silent intros skipped).

    Two problems this solves:
      * The library is mostly FLAC, ~30% 24-bit or hi-res (96/192 kHz), which the
        HTML5 <audio> element cannot decode — so we transcode to 16-bit WAV.
      * Fixed end-windows landed on fade-ins/outs and silent/spoken intros, so the
        listener rated silence instead of the transition. We scan up to scan_sec
        from the relevant end, find the energetic span, and take the window there.
    Extracting server-side also avoids fragile client-side FLAC time-seeking."""
    import io
    import soundfile as sf

    info = sf.info(file_path)
    sr = info.samplerate
    frames = info.frames
    win = int(float(window_sec) * sr)
    scan = int(float(scan_sec) * sr)

    if pos == "head":
        block, _ = sf.read(file_path, start=0, stop=min(frames, scan),
                           dtype="int16", always_2d=True)
        a_start, _a_end = _active_span(block.mean(axis=1), sr, top_db)
        start = a_start
        stop = min(frames, start + win)
    else:  # tail
        rstart = max(0, frames - scan)
        block, _ = sf.read(file_path, start=rstart, stop=frames,
                           dtype="int16", always_2d=True)
        _a_start, a_end = _active_span(block.mean(axis=1), sr, top_db)
        abs_end = rstart + a_end
        start = max(0, abs_end - win)
        stop = abs_end

    clip, _ = sf.read(file_path, start=start, stop=stop, dtype="int16", always_2d=True)
    bio = io.BytesIO()
    sf.write(bio, clip, sr, format="WAV", subtype="PCM_16")
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
