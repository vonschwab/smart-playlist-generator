"""Local HTTP server for the genre-similarity audition harness.

Serves the rating page and a blinded manifest API (provenance hidden), and
appends relatedness verdicts to per-seed capture YAMLs with provenance
re-attached server-side. No audio.

Usage:
    python scripts/genre_audition_serve.py [--port 8766] [--data-dir docs/run_audits/genre_audition]

Requires manifests — run genre_audition_build.py first.
"""
from __future__ import annotations

import datetime
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _append_capture_entry(capture_path: Path, entry: dict) -> None:
    """Append or update one entry in the capture YAML, keyed by name."""
    if capture_path.exists():
        data = yaml.safe_load(capture_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    entries: list = data.get("entries", [])
    name = entry["name"]
    for i, e in enumerate(entries):
        if e.get("name") == name:
            entries[i] = entry
            break
    else:
        entries.append(entry)
    data["entries"] = entries
    capture_path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )


def _blind_manifest(m: dict) -> dict:
    """Return a copy of the manifest with provenance fields stripped."""
    return {k: v for k, v in m.items() if k not in ("space_data", "cooc_token")}


class AuditionServer(HTTPServer):
    def __init__(self, addr, handler_class, data_dir, manifests, index, page_html):
        super().__init__(addr, handler_class)
        self.data_dir = data_dir
        self.manifests = manifests
        self.index = index
        self.page_html = page_html


class AuditionHandler(BaseHTTPRequestHandler):
    server: AuditionServer

    def log_message(self, fmt, *args):
        pass

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
                self.send_response(302)
                self.send_header("Location", f"/seed/{self.server.index[0]['slug']}")
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
        elif path.startswith("/api/manifest/"):
            m = self.server.manifests.get(path[14:])
            if not m:
                self.send_error(404)
                return
            self._json(_blind_manifest(m))
        elif path.startswith("/api/progress/"):
            cap = self.server.data_dir / f"{path[14:]}_capture.yaml"
            if not cap.exists():
                self._json([])
                return
            data = yaml.safe_load(cap.read_text(encoding="utf-8")) or {}
            self._json([
                {"name": e["name"], "verdict": e.get("verdict", ""), "notes": e.get("notes", "")}
                for e in data.get("entries", [])
            ])
        else:
            self.send_error(404)

    def do_POST(self):
        if unquote(self.path) != "/api/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._json({"ok": False, "error": "invalid JSON"}, 400)
            return
        seed = body.get("seed", "")
        name = body.get("name", "")
        if not seed or not name:
            self._json({"ok": False, "error": "missing seed or name"}, 400)
            return
        if seed not in self.server.manifests:
            self._json({"ok": False, "error": "unknown seed"}, 400)
            return
        m = self.server.manifests[seed]
        spaces = m.get("space_data", {}).get(name, {})
        entry = {
            "name": name,
            "verdict": body.get("verdict", ""),
            "notes": body.get("notes", ""),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "spaces": spaces,
        }
        _append_capture_entry(self.server.data_dir / f"{seed}_capture.yaml", entry)
        self._json({"ok": True})


def load_manifests(data_dir: Path):
    index_path = data_dir / "index.json"
    if not index_path.exists():
        return {}, []
    index = json.loads(index_path.read_text())
    manifests = {}
    for entry in index:
        p = data_dir / f"{entry['slug']}_manifest.json"
        if p.exists():
            manifests[entry["slug"]] = json.loads(p.read_text())
    return manifests, index


def main() -> None:
    import argparse
    import webbrowser

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--data-dir", default="docs/run_audits/genre_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    manifests, index = load_manifests(data_dir)
    if not manifests:
        print(f"No manifests in {data_dir}. Run: python scripts/genre_audition_build.py")
        sys.exit(1)

    page_path = Path(__file__).parent / "genre_audition_page.html"
    if not page_path.exists():
        print(f"Page template not found at {page_path}.")
        sys.exit(1)
    page_html = page_path.read_text(encoding="utf-8")

    server = AuditionServer(
        ("127.0.0.1", args.port), AuditionHandler, data_dir, manifests, index, page_html
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Genre audition server → {url}")
    print(f"Seeds: {[e.get('genre', e['slug']) for e in index]}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
