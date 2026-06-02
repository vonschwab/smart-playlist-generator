"""Local HTTP server for the sonic audition harness.

Streams audio with HTTP range support, serves the audition page and blinded
manifests, and appends to per-seed capture YAMLs.

Usage:
    python scripts/sonic_audition_serve.py [--port 8765] [--data-dir docs/run_audits/sonic_audition]

Requires manifests — run sonic_audition_build.py first.
"""
from __future__ import annotations

import datetime
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}


def _parse_range_header(header: str, file_size: int) -> tuple[int, int]:
    """Parse a 'bytes=X-Y' Range header. Returns (start, end) inclusive."""
    if not header or not header.startswith("bytes="):
        return (0, file_size - 1)
    parts = header[6:].split("-")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    return (start, min(end, file_size - 1))


def _append_capture_entry(capture_path: Path, entry: dict) -> None:
    """Append or update one entry in the capture YAML, keyed by track_id."""
    if capture_path.exists():
        data = yaml.safe_load(capture_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    entries: list = data.get("entries", [])
    track_id = entry["track_id"]
    for i, e in enumerate(entries):
        if e.get("track_id") == track_id:
            entries[i] = entry
            break
    else:
        entries.append(entry)
    data["entries"] = entries
    capture_path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )


class AuditionServer(HTTPServer):
    def __init__(
        self,
        addr: tuple,
        handler_class,
        data_dir: Path,
        manifests: Dict[str, dict],
        index: List[dict],
        page_html: str,
    ):
        super().__init__(addr, handler_class)
        self.data_dir = data_dir
        self.manifests = manifests
        self.index = index
        self.page_html = page_html

    def _find_file_path(self, track_id: str) -> Optional[str]:
        for m in self.manifests.values():
            if m.get("seed", {}).get("track_id") == track_id:
                return m["seed"].get("file_path")
            for n in m.get("neighbors", []):
                if n["track_id"] == track_id:
                    return n.get("file_path")
            for pair in m.get("pairs", []):
                for side in ("prev", "next"):
                    if pair[side]["track_id"] == track_id:
                        return pair[side].get("file_path")
        return None


class AuditionHandler(BaseHTTPRequestHandler):
    server: AuditionServer

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _json(self, data, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])

        if path == "/":
            if self.server.index:
                first = self.server.index[0]["slug"]
                self.send_response(302)
                self.send_header("Location", f"/seed/{first}")
                self.end_headers()
            else:
                self.send_error(404, "No manifests found")

        elif path.startswith("/seed/"):
            slug = path[6:]
            if slug not in self.server.manifests:
                self.send_error(404, f"Seed {slug!r} not found")
                return
            html = self.server.page_html.replace(
                "SEED_LIST_PLACEHOLDER", json.dumps(self.server.index)
            )
            self._html(html)

        elif path.startswith("/audio/"):
            track_id = path[7:]
            file_path = self.server._find_file_path(track_id)
            if not file_path:
                self.send_error(404, f"Track {track_id!r} not in any manifest")
                return
            self._serve_audio(file_path)

        elif path.startswith("/api/manifest/"):
            slug = path[14:]
            m = self.server.manifests.get(slug)
            if not m:
                self.send_error(404)
                return
            # Serve blinded manifest: omit space_data
            blinded = {k: v for k, v in m.items() if k != "space_data"}
            self._json(blinded)

        elif path.startswith("/api/progress/"):
            slug = path[14:]
            cap = self.server.data_dir / f"{slug}_capture.yaml"
            if not cap.exists():
                self._json([])
                return
            data = yaml.safe_load(cap.read_text(encoding="utf-8")) or {}
            self._json([
                {
                    "track_id": e["track_id"],
                    "verdict": e.get("verdict", ""),
                    "notes": e.get("notes", ""),
                }
                for e in data.get("entries", [])
            ])

        else:
            self.send_error(404)

    def do_POST(self):
        if unquote(self.path) != "/api/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        seed = body.get("seed", "")
        track_id = body.get("track_id", "")
        if not seed or not track_id:
            self._json({"ok": False, "error": "missing seed or track_id"}, 400)
            return

        m = self.server.manifests.get(seed, {})
        space_data = m.get("space_data", {}).get(track_id, {})

        artist, title = "", ""
        # Check seed track
        if m.get("seed", {}).get("track_id") == track_id:
            artist = m["seed"].get("artist", "")
            title = m["seed"].get("title", "")
        else:
            # Check neighbors
            for n in m.get("neighbors", []):
                if n["track_id"] == track_id:
                    artist, title = n.get("artist", ""), n.get("title", "")
                    break
            else:
                # Check transition pair sides
                for pair in m.get("pairs", []):
                    for side in ("prev", "next"):
                        if pair[side]["track_id"] == track_id:
                            artist = pair[side].get("artist", "")
                            title = pair[side].get("title", "")

        entry = {
            "track_id": track_id,
            "artist": artist,
            "title": title,
            "verdict": body.get("verdict", ""),
            "notes": body.get("notes", ""),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "spaces": space_data,
        }
        _append_capture_entry(self.server.data_dir / f"{seed}_capture.yaml", entry)
        self._json({"ok": True})

    def _serve_audio(self, file_path: str):
        p = Path(file_path)
        if not p.exists():
            self.send_error(404, "Audio file not found on disk")
            return
        content_type = CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream")
        file_size = p.stat().st_size
        range_hdr = self.headers.get("Range", "")
        start, end = _parse_range_header(range_hdr, file_size)
        length = end - start + 1
        self.send_response(206 if range_hdr else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if range_hdr:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
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


def load_manifests(data_dir: Path) -> tuple[Dict[str, dict], List[dict]]:
    """Load all manifests listed in index.json."""
    index_path = data_dir / "index.json"
    if not index_path.exists():
        return {}, []
    index: List[dict] = json.loads(index_path.read_text())
    manifests = {}
    for entry in index:
        slug = entry["slug"]
        p = data_dir / f"{slug}_manifest.json"
        if p.exists():
            manifests[slug] = json.loads(p.read_text())
    return manifests, index


def main() -> None:
    import argparse
    import webbrowser

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data-dir", default="docs/run_audits/sonic_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    if not data_dir.exists():
        print(f"No data directory at {data_dir}. Run: python scripts/sonic_audition_build.py")
        sys.exit(1)

    manifests, index = load_manifests(data_dir)
    if not manifests:
        print(f"No manifests found in {data_dir}. Run: python scripts/sonic_audition_build.py")
        sys.exit(1)

    page_path = Path(__file__).parent / "sonic_audition_page.html"
    if not page_path.exists():
        print(f"Page template not found at {page_path}.")
        sys.exit(1)
    page_html = page_path.read_text(encoding="utf-8")

    server = AuditionServer(
        ("127.0.0.1", args.port), AuditionHandler, data_dir, manifests, index, page_html
    )
    url = f"http://127.0.0.1:{args.port}/"
    seeds = [e.get("artist", e["slug"]) for e in index]
    print(f"Audition server → {url}")
    print(f"Seeds: {seeds}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
