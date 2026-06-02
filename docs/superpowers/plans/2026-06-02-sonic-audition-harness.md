# Sonic Audition Harness (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local audition harness that computes sonic neighbors across 5 spaces, serves a browser UI for listening and rating, and aggregates findings — to validate whether the corrected tower-weighted space sounds better by ear.

**Architecture:** Three independent scripts communicating through files: `sonic_audition_build.py` computes neighbors and writes blinded JSON manifests; `sonic_audition_serve.py` streams audio and serves an audition page; `sonic_audition_analyze.py` reads completed capture YAMLs and writes a findings report. Per-tower matrices (`X_sonic_rhythm/timbre/harmony`) are loaded directly from the npz (they exist on disk but are not in `ArtifactBundle`) alongside the bundle for metadata.

**Tech Stack:** Python 3.11, NumPy, PyYAML, sqlite3 (stdlib), http.server (stdlib), vanilla JS/HTML (no framework). No new pip installs required.

**Spec:** `docs/superpowers/specs/2026-06-01-sonic-neighborhood-validation-design.md`

**Data locations:**
- Artifact: `data/artifacts/beat3tower_32k/data_matrices_step1.npz`
- DB (read-only, file_path lookup): `data/metadata.db` — table `tracks`, columns `track_id TEXT`, `file_path TEXT`
- Per-tower matrices in npz (NOT in ArtifactBundle): `X_sonic_rhythm` (N,9), `X_sonic_timbre` (N,57), `X_sonic_harmony` (N,20)
- File paths are absolute Windows paths (e.g. `E:\MUSIC\...`)
- Output manifests + captures: `docs/run_audits/sonic_audition/` (gitignored; created at runtime)

**Tower layout in 86-dim X_sonic:** rhythm `[0:9]`, timbre `[9:66]`, harmony `[66:86]` (9+57+20)

**5 sonic spaces:**
1. `full_track` — L2-normalised `X_sonic` (the admission-pool sim, tower-weighted after Phase 1)
2. `production_transition` — centered+normalised `X_sonic_end` (query) vs `X_sonic_start` (search) — what the beam scores
3. `rhythm` — L2-normalised `X_sonic_rhythm` (9-dim per-tower)
4. `timbre` — L2-normalised `X_sonic_timbre` (57-dim per-tower)
5. `harmony` — L2-normalised `X_sonic_harmony` (20-dim per-tower)

**17 seeds:** Green-House, Boards of Canada, Autechre, Charli XCX, Bill Evans, Jean-Yves Thibaudet, William Tyler, Elliott Smith, Duster, Real Estate, Slowdive, Sonic Youth, Minor Threat, James Brown, J Dilla, Beyoncé, Grouper

**3 negative-S transition pairs (from production run notes):**
- Torrey → Pixies (was S=-0.15)
- Built to Spill → Beach House (was S=-0.11)
- Melody's Echo Chamber → Peel Dream Magazine (was S=-0.09)

---

## File Structure

| File | Role |
|---|---|
| `scripts/sonic_audition_build.py` | Compute neighbors, write manifests |
| `scripts/sonic_audition_page.html` | Audition UI (read by server at startup) |
| `scripts/sonic_audition_serve.py` | HTTP server: audio streaming + API |
| `scripts/sonic_audition_analyze.py` | Aggregate captures → findings |
| `tests/unit/test_sonic_audition_build.py` | Tests for manifest builder |
| `tests/unit/test_sonic_audition_serve.py` | Tests for server helpers |
| `tests/unit/test_sonic_audition_analyze.py` | Tests for analysis aggregation |

---

### Task 1: Manifest builder

**Files:**
- Create: `scripts/sonic_audition_build.py`
- Test: `tests/unit/test_sonic_audition_build.py`

The manifest builder computes top-15 neighbors per seed per space, unions across spaces, shuffles (blinded), and writes per-seed JSON manifests. Space/rank data is hidden from the neighbor list and stored separately under `space_data` — the server uses this when annotating saved capture entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sonic_audition_build.py
import numpy as np
import pytest
from scripts.sonic_audition_build import (
    compute_spaces,
    find_medoid,
    top_k_for_seed,
    build_seed_manifest,
    _slug,
)


class _Bundle:
    """Minimal bundle-like object for testing."""
    def __init__(self, N=20):
        rng = np.random.default_rng(0)
        self.X_sonic = rng.normal(size=(N, 86)).astype(np.float32)
        self.X_sonic_start = rng.normal(size=(N, 86)).astype(np.float32)
        self.X_sonic_end = rng.normal(size=(N, 86)).astype(np.float32)
        self.track_ids = np.array([f"t{i:03d}" for i in range(N)], dtype=object)
        self.track_artists = np.array(
            ["ArtistA"] * 5 + ["ArtistB"] * 5 + ["ArtistC"] * 10, dtype=object
        )
        self.track_titles = np.array([f"Track{i}" for i in range(N)], dtype=object)


def _make_per_tower(N=20):
    rng = np.random.default_rng(1)
    return {
        "X_sonic_rhythm": rng.normal(size=(N, 9)).astype(np.float32),
        "X_sonic_timbre": rng.normal(size=(N, 57)).astype(np.float32),
        "X_sonic_harmony": rng.normal(size=(N, 20)).astype(np.float32),
    }


def test_compute_spaces_returns_five_spaces():
    b = _Bundle()
    spaces = compute_spaces(b, _make_per_tower())
    assert set(spaces.keys()) == {
        "full_track", "production_transition", "rhythm", "timbre", "harmony"
    }


def test_compute_spaces_query_rows_are_unit_norm():
    b = _Bundle(N=20)
    spaces = compute_spaces(b, _make_per_tower())
    for name, (Xq, _) in spaces.items():
        norms = np.linalg.norm(Xq, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"{name}: query rows not unit-norm"


def test_compute_spaces_transition_uses_end_start():
    b = _Bundle(N=20)
    spaces = compute_spaces(b, _make_per_tower())
    Xq, Xs = spaces["production_transition"]
    # shapes match X_sonic_end and X_sonic_start
    assert Xq.shape == b.X_sonic_end.shape
    assert Xs.shape == b.X_sonic_start.shape


def test_top_k_excludes_same_artist_and_seed():
    b = _Bundle()
    spaces = compute_spaces(b, _make_per_tower())
    artist_a = set(range(5))  # ArtistA = indices 0-4
    neighbors = top_k_for_seed(seed_idx=0, spaces=spaces, exclude_indices=artist_a, k=5)
    for space, pairs in neighbors.items():
        assert len(pairs) == 5, f"{space}: expected 5 neighbors"
        for idx, _ in pairs:
            assert idx not in artist_a, f"{space}: same-artist idx {idx} in neighbors"


def test_build_seed_manifest_blinded_structure():
    b = _Bundle()
    per_tower = _make_per_tower()
    spaces = compute_spaces(b, per_tower)
    file_paths = {f"t{i:03d}": f"/music/t{i}.flac" for i in range(20)}
    manifest = build_seed_manifest("ArtistA", b, spaces, file_paths, k=3)
    assert manifest is not None
    assert manifest["slug"] == "artista"
    # space_data lives at top level, not in individual neighbor entries
    assert "space_data" in manifest
    for n in manifest["neighbors"]:
        assert "spaces" not in n
        assert "track_id" in n
        assert "artist" in n
        assert "file_path" in n


def test_build_seed_manifest_returns_none_for_unknown():
    b = _Bundle()
    spaces = compute_spaces(b, _make_per_tower())
    assert build_seed_manifest("Nobody", b, spaces, {}, k=3) is None


def test_slug():
    assert _slug("Charli XCX") == "charli_xcx"
    assert _slug("J Dilla") == "j_dilla"
    assert _slug("Green-House") == "green_house"
    assert _slug("Beyoncé") == "beyonc"  # unicode stripped by regex
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sonic_audition_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.sonic_audition_build'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/sonic_audition_build.py
"""Build per-seed audition manifests for sonic neighborhood validation (Phase 2).

For each seed artist, finds the medoid track, computes top-K neighbors in 5 sonic
spaces, deduplicates across spaces, shuffles (blinded — space/rank hidden from
the neighbor list), and writes a JSON manifest. Also builds a manifest for
known negative-S transition pairs from production run notes.

Usage:
    python scripts/sonic_audition_build.py
    python scripts/sonic_audition_build.py --seeds "Real Estate" "Grouper"
    python scripts/sonic_audition_build.py --top-k 15

Output: docs/run_audits/sonic_audition/<slug>_manifest.json per seed,
        docs/run_audits/sonic_audition/negative_s_manifest.json,
        docs/run_audits/sonic_audition/index.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SEEDS = [
    "Green-House", "Boards of Canada", "Autechre", "Charli XCX",
    "Bill Evans", "Jean-Yves Thibaudet", "William Tyler", "Elliott Smith",
    "Duster", "Real Estate", "Slowdive", "Sonic Youth", "Minor Threat",
    "James Brown", "J Dilla", "Beyoncé", "Grouper",
]

NEGATIVE_S_PAIRS = [
    ("Torrey", "Pixies"),
    ("Built to Spill", "Beach House"),
    ("Melody's Echo Chamber", "Peel Dream Magazine"),
]


def _l2(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float64)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def compute_spaces(
    bundle,
    per_tower: Dict[str, np.ndarray],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {space_name: (X_query, X_search)} for all 5 sonic spaces.

    For symmetric spaces, X_query == X_search.
    For production_transition, X_query is the centered end-segment and
    X_search is the centered start-segment (matching the beam's scoring).
    """
    # Full-track (admission pool similarity)
    Xf = _l2(bundle.X_sonic.astype(np.float64))

    # Production transition: center independently then normalise
    Xe = bundle.X_sonic_end.astype(np.float64)
    Xs = bundle.X_sonic_start.astype(np.float64)
    Xe_c = _l2(Xe - Xe.mean(axis=0, keepdims=True))
    Xs_c = _l2(Xs - Xs.mean(axis=0, keepdims=True))

    # Per-tower (raw per-tower matrices from npz, L2-normalised)
    Xr = _l2(per_tower["X_sonic_rhythm"].astype(np.float64))
    Xt = _l2(per_tower["X_sonic_timbre"].astype(np.float64))
    Xh = _l2(per_tower["X_sonic_harmony"].astype(np.float64))

    return {
        "full_track": (Xf, Xf),
        "production_transition": (Xe_c, Xs_c),
        "rhythm": (Xr, Xr),
        "timbre": (Xt, Xt),
        "harmony": (Xh, Xh),
    }


def find_medoid(Xq: np.ndarray, indices: np.ndarray) -> int:
    """Return the index (in the full matrix) of the track closest to the artist centroid."""
    sub = Xq[indices]
    centroid = sub.mean(axis=0)
    cn = centroid / (np.linalg.norm(centroid) + 1e-12)
    return int(indices[int(np.argmax(sub @ cn))])


def top_k_for_seed(
    seed_idx: int,
    spaces: Dict[str, Tuple[np.ndarray, np.ndarray]],
    exclude_indices: set,
    k: int,
) -> Dict[str, List[Tuple[int, float]]]:
    """Return {space: [(track_idx, cosine), ...]} top-k per space, excluding exclude_indices."""
    N = next(iter(spaces.values()))[0].shape[0]
    exclude = set(exclude_indices) | {seed_idx}
    mask = np.ones(N, dtype=bool)
    for idx in exclude:
        if 0 <= idx < N:
            mask[idx] = False
    valid = np.where(mask)[0]

    result = {}
    for space, (Xq, Xs) in spaces.items():
        sims = Xq[seed_idx] @ Xs[valid].T
        top = np.argsort(-sims)[:k]
        result[space] = [(int(valid[i]), float(sims[i])) for i in top]
    return result


def build_seed_manifest(
    artist: str,
    bundle,
    spaces: Dict[str, Tuple[np.ndarray, np.ndarray]],
    file_paths: Dict[str, str],
    k: int = 15,
) -> Optional[dict]:
    """Build the blinded manifest for one seed artist. Returns None if artist not found."""
    artists = np.array([str(a) for a in bundle.track_artists])
    artist_idx = np.where(np.char.lower(artists) == artist.lower())[0]
    if len(artist_idx) == 0:
        return None

    Xq_full = spaces["full_track"][0]
    seed_idx = find_medoid(Xq_full, artist_idx)
    seed_tid = str(bundle.track_ids[seed_idx])
    seed_title = str(bundle.track_titles[seed_idx]) if bundle.track_titles is not None else "?"

    exclude = {int(i) for i in artist_idx}
    per_space = top_k_for_seed(seed_idx, spaces, exclude, k)

    # Union across spaces, keyed by track index
    seen: Dict[int, Dict[str, dict]] = {}
    for space, neighbors in per_space.items():
        for rank, (idx, cos) in enumerate(neighbors):
            if idx not in seen:
                seen[idx] = {}
            seen[idx][space] = {"rank": rank + 1, "cosine": round(float(cos), 4)}

    # Shuffle for blind presentation (deterministic per artist so re-runs are stable)
    shuffled = list(seen.keys())
    rng = np.random.default_rng(abs(hash(artist)) % (2 ** 32))
    rng.shuffle(shuffled)

    neighbors = []
    space_data = {}
    for idx in shuffled:
        tid = str(bundle.track_ids[idx])
        neighbors.append({
            "track_id": tid,
            "artist": str(bundle.track_artists[idx]),
            "title": str(bundle.track_titles[idx]) if bundle.track_titles is not None else "?",
            "file_path": file_paths.get(tid, ""),
        })
        space_data[tid] = seen[idx]

    return {
        "slug": _slug(artist),
        "type": "seed",
        "seed": {
            "artist": artist,
            "track_id": seed_tid,
            "title": seed_title,
            "file_path": file_paths.get(seed_tid, ""),
        },
        "neighbors": neighbors,
        "space_data": space_data,
    }


def build_negative_s_manifest(
    pairs: List[Tuple[str, str]],
    bundle,
    spaces: Dict[str, Tuple[np.ndarray, np.ndarray]],
    file_paths: Dict[str, str],
) -> dict:
    """Build a manifest for known negative-S transition pairs."""
    artists = np.array([str(a) for a in bundle.track_artists])
    Xq_full = spaces["full_track"][0]
    Xe_c, Xs_c = spaces["production_transition"]

    pair_entries = []
    for prev_artist, next_artist in pairs:
        prev_arr = np.where(np.char.lower(artists) == prev_artist.lower())[0]
        next_arr = np.where(np.char.lower(artists) == next_artist.lower())[0]
        if len(prev_arr) == 0 or len(next_arr) == 0:
            print(f"  SKIP pair {prev_artist!r}→{next_artist!r}: artist(s) not found")
            continue
        prev_idx = find_medoid(Xq_full, prev_arr)
        next_idx = find_medoid(Xq_full, next_arr)
        S = float(Xe_c[prev_idx] @ Xs_c[next_idx])
        prev_tid = str(bundle.track_ids[prev_idx])
        next_tid = str(bundle.track_ids[next_idx])
        pair_entries.append({
            "label": f"{prev_artist} → {next_artist}",
            "S": round(S, 4),
            "prev": {
                "track_id": prev_tid,
                "artist": prev_artist,
                "title": str(bundle.track_titles[prev_idx]) if bundle.track_titles is not None else "?",
                "file_path": file_paths.get(prev_tid, ""),
            },
            "next": {
                "track_id": next_tid,
                "artist": next_artist,
                "title": str(bundle.track_titles[next_idx]) if bundle.track_titles is not None else "?",
                "file_path": file_paths.get(next_tid, ""),
            },
        })

    return {
        "slug": "negative_s",
        "type": "transition_pairs",
        "pairs": pair_entries,
    }


def lookup_file_paths(track_ids: List[str], db_path: str) -> Dict[str, str]:
    """Read-only lookup of file_path by track_id."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    result = {}
    batch = 900  # SQLite variable limit
    for i in range(0, len(track_ids), batch):
        chunk = track_ids[i : i + batch]
        ph = ",".join(["?"] * len(chunk))
        rows = con.execute(
            f"SELECT track_id, file_path FROM tracks WHERE track_id IN ({ph})", chunk
        ).fetchall()
        result.update({str(r[0]): str(r[1]) for r in rows})
    con.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument(
        "--artifact",
        default="data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    ap.add_argument("--db", default="data/metadata.db")
    ap.add_argument("--out-dir", default="docs/run_audits/sonic_audition")
    args = ap.parse_args()

    from src.features.artifacts import load_artifact_bundle

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(args.artifact)

    # Load per-tower matrices from npz (not exposed by ArtifactBundle)
    npz = np.load(bundle.artifact_path, allow_pickle=True)
    per_tower = {
        "X_sonic_rhythm": npz["X_sonic_rhythm"],
        "X_sonic_timbre": npz["X_sonic_timbre"],
        "X_sonic_harmony": npz["X_sonic_harmony"],
    }

    print(f"Computing 5 sonic spaces over {len(bundle.track_ids)} tracks...")
    spaces = compute_spaces(bundle, per_tower)

    all_tids = [str(t) for t in bundle.track_ids]
    print("Looking up file paths...")
    file_paths = lookup_file_paths(all_tids, args.db)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for artist in args.seeds:
        manifest = build_seed_manifest(artist, bundle, spaces, file_paths, k=args.top_k)
        if manifest is None:
            print(f"  SKIP {artist!r} (not found in bundle)")
            continue
        slug = manifest["slug"]
        p = out_dir / f"{slug}_manifest.json"
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        n = len(manifest["neighbors"])
        index.append({"slug": slug, "artist": artist})
        print(f"  OK   {artist!r} → {slug}_manifest.json ({n} neighbors)")

    neg = build_negative_s_manifest(NEGATIVE_S_PAIRS, bundle, spaces, file_paths)
    (out_dir / "negative_s_manifest.json").write_text(
        json.dumps(neg, indent=2), encoding="utf-8"
    )
    index.append({"slug": "negative_s", "artist": "Negative-S Pairs", "type": "transition_pairs"})
    print(f"  OK   negative_s → negative_s_manifest.json ({len(neg['pairs'])} pairs)")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nDone. {len(index)} manifests in {out_dir}")
    print("Next: python scripts/sonic_audition_serve.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sonic_audition_build.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/sonic_audition_build.py tests/unit/test_sonic_audition_build.py
git commit -m "feat(audition): manifest builder — 5-space neighbor computation, blinded shuffle"
```

---

### Task 2: Audition page HTML

**Files:**
- Create: `scripts/sonic_audition_page.html`

No pytest tests — this is a static file served by the server. Visual verification happens when running the server in Task 3.

- [ ] **Step 1: Create the page**

```html
<!-- scripts/sonic_audition_page.html -->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sonic Audition</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Courier New', monospace; background: #1a1a1a; color: #e0e0e0; min-height: 100vh; }
.top-bar { background: #111; border-bottom: 1px solid #333; padding: 10px 16px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 10; flex-wrap: wrap; }
.top-bar label { font-size: 13px; color: #aaa; }
.top-bar select { background: #222; color: #e0e0e0; border: 1px solid #555; padding: 4px 8px; font-family: inherit; font-size: 13px; border-radius: 3px; cursor: pointer; }
.nav a { background: #222; color: #aaa; text-decoration: none; padding: 4px 10px; border: 1px solid #444; border-radius: 3px; font-size: 12px; margin-right: 4px; }
.nav a:hover { background: #333; color: #e0e0e0; }
.progress-bar { background: #222; padding: 8px 16px; font-size: 12px; color: #888; border-bottom: 1px solid #2a2a2a; }
.progress-bar span { color: #4caf50; font-weight: bold; }
.seed-section { background: #111; border-bottom: 2px solid #333; padding: 16px; }
.seed-section h2 { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.seed-section .track-name { font-size: 17px; color: #fff; margin-bottom: 10px; }
audio { width: 100%; height: 36px; }
.main { max-width: 860px; margin: 0 auto; padding: 16px; }
.neighbor { background: #222; border: 1px solid #333; border-left: 4px solid #333; border-radius: 4px; padding: 12px; margin-bottom: 10px; transition: border-color 0.2s; }
.neighbor.match { border-left-color: #4caf50; }
.neighbor.close { border-left-color: #8bc34a; }
.neighbor.off   { border-left-color: #ff9800; }
.neighbor.wrong { border-left-color: #f44336; }
.neighbor-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.neighbor-header .meta { font-size: 13px; color: #ccc; }
.neighbor-header .num { color: #555; font-size: 11px; margin-right: 6px; }
.status { font-size: 11px; color: #4caf50; min-width: 60px; text-align: right; }
.status.err { color: #f44336; }
.verdicts { display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap; }
.vbtn { display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; color: #bbb; padding: 3px 9px; border: 1px solid #444; border-radius: 3px; }
.vbtn:hover { background: #2a2a2a; }
.vbtn input { accent-color: #4caf50; }
textarea { width: 100%; background: #1a1a1a; color: #ddd; border: 1px solid #444; border-radius: 3px; padding: 6px; font-family: inherit; font-size: 12px; resize: vertical; min-height: 48px; margin-top: 4px; }
textarea:focus { outline: none; border-color: #666; }
.pair-card { background: #222; border: 1px solid #333; border-radius: 4px; padding: 14px; margin-bottom: 14px; }
.pair-label { font-size: 13px; color: #aaa; margin-bottom: 6px; }
.pair-s { font-size: 11px; color: #ff9800; margin-bottom: 10px; }
.pair-tracks { display: grid; grid-template-columns: 1fr 24px 1fr; gap: 8px; align-items: start; margin-bottom: 10px; }
.pair-track-label { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.pair-track-name { font-size: 12px; color: #ccc; margin-bottom: 5px; }
.arrow { font-size: 18px; color: #555; text-align: center; padding-top: 20px; }
</style>
</head>
<body>

<div class="top-bar">
  <label>Seed: <select id="seed-select" onchange="location='/seed/'+this.value"></select></label>
  <div class="nav" id="nav"></div>
</div>
<div class="progress-bar" id="progress-bar">Loading…</div>
<div class="seed-section" id="seed-section">
  <h2>Reference Seed</h2>
  <div class="track-name" id="seed-name">—</div>
  <audio id="seed-audio" controls preload="none"></audio>
</div>
<div class="main" id="main"></div>

<script>
const SLUG = location.pathname.split('/').pop();
const ALL_SEEDS = SEED_LIST_PLACEHOLDER;
let manifest = null, progress = {};

async function init() {
  const sel = document.getElementById('seed-select');
  ALL_SEEDS.forEach(s => {
    const o = document.createElement('option');
    o.value = s.slug; o.textContent = s.artist || s.slug; o.selected = (s.slug === SLUG);
    sel.appendChild(o);
  });
  const cur = ALL_SEEDS.findIndex(s => s.slug === SLUG);
  const nav = document.getElementById('nav');
  if (cur > 0) nav.innerHTML += `<a href="/seed/${ALL_SEEDS[cur-1].slug}">← prev</a>`;
  if (cur < ALL_SEEDS.length-1) nav.innerHTML += `<a href="/seed/${ALL_SEEDS[cur+1].slug}">next →</a>`;

  const [mR, pR] = await Promise.all([
    fetch(`/api/manifest/${SLUG}`), fetch(`/api/progress/${SLUG}`)
  ]);
  manifest = await mR.json();
  (await pR.json()).forEach(e => { progress[e.track_id] = e; });

  if (manifest.type === 'transition_pairs') renderPairs();
  else renderNeighbors();
}

function renderNeighbors() {
  document.getElementById('seed-name').textContent =
    `${manifest.seed.artist} — ${manifest.seed.title}`;
  document.getElementById('seed-audio').src = `/audio/${manifest.seed.track_id}`;
  const main = document.getElementById('main');
  manifest.neighbors.forEach((n, i) => {
    const s = progress[n.track_id] || {};
    main.appendChild(buildCard(n, i, s));
  });
  updateProgress();
}

function buildCard(n, i, saved) {
  const div = document.createElement('div');
  div.className = 'neighbor' + (saved.verdict ? ` ${saved.verdict}` : '');
  div.id = `card-${n.track_id}`;
  div.innerHTML = `
    <div class="neighbor-header">
      <span class="meta"><span class="num">#${i+1}</span>${n.artist} — ${n.title}</span>
      <span class="status" id="st-${n.track_id}"></span>
    </div>
    <audio controls preload="none" src="/audio/${n.track_id}"></audio>
    <div class="verdicts">
      ${['match','close','off','wrong'].map(v => `
        <label class="vbtn"><input type="radio" name="v-${n.track_id}" value="${v}"
          ${saved.verdict===v?'checked':''} onchange="saveN('${n.track_id}')"> ${v}</label>
      `).join('')}
    </div>
    <textarea id="nt-${n.track_id}" placeholder="Notes…"
      onblur="saveN('${n.track_id}')">${saved.notes||''}</textarea>`;
  return div;
}

async function saveN(tid) {
  const v = document.querySelector(`input[name="v-${tid}"]:checked`)?.value||'';
  const notes = document.getElementById(`nt-${tid}`).value;
  const st = document.getElementById(`st-${tid}`);
  try {
    await post({seed:SLUG,track_id:tid,verdict:v,notes});
    st.textContent='saved ✓'; st.className='status';
    progress[tid]={track_id:tid,verdict:v,notes};
    document.getElementById(`card-${tid}`).className='neighbor'+(v?` ${v}`:'');
    updateProgress();
    setTimeout(()=>{ st.textContent=''; },3000);
  } catch(e){ st.textContent='error'; st.className='status err'; }
}

function renderPairs() {
  document.getElementById('seed-section').style.display='none';
  const main = document.getElementById('main');
  const h = document.createElement('h2');
  h.style.cssText='font-size:12px;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px';
  h.textContent='Negative-S Transition Pairs — Do these transitions work by ear?';
  main.appendChild(h);
  manifest.pairs.forEach((pair, i) => {
    const s = progress[`pair_${i}`]||{};
    const div = document.createElement('div');
    div.className='pair-card'; div.id=`card-pair_${i}`;
    div.innerHTML=`
      <div class="pair-label">${pair.label}</div>
      <div class="pair-s">Transition S = ${pair.S} (negative in original space)</div>
      <div class="pair-tracks">
        <div>
          <div class="pair-track-label">Prev</div>
          <div class="pair-track-name">${pair.prev.artist} — ${pair.prev.title}</div>
          <audio controls preload="none" src="/audio/${pair.prev.track_id}"></audio>
        </div>
        <div class="arrow">→</div>
        <div>
          <div class="pair-track-label">Next</div>
          <div class="pair-track-name">${pair.next.artist} — ${pair.next.title}</div>
          <audio controls preload="none" src="/audio/${pair.next.track_id}"></audio>
        </div>
      </div>
      <div class="verdicts">
        ${['yes','maybe','no'].map(v=>`
          <label class="vbtn"><input type="radio" name="vp-${i}" value="${v}"
            ${s.verdict===v?'checked':''} onchange="saveP(${i})"> ${v}</label>
        `).join('')}
      </div>
      <textarea id="pnt-${i}" placeholder="Notes…"
        onblur="saveP(${i})">${s.notes||''}</textarea>`;
    main.appendChild(div);
  });
  updateProgress();
}

async function saveP(i) {
  const v = document.querySelector(`input[name="vp-${i}"]:checked`)?.value||'';
  const notes = document.getElementById(`pnt-${i}`).value;
  await post({seed:SLUG,track_id:`pair_${i}`,verdict:v,notes}).catch(()=>{});
  progress[`pair_${i}`]={track_id:`pair_${i}`,verdict:v,notes};
  updateProgress();
}

async function post(body) {
  const r = await fetch('/api/save',{
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  });
  if (!r.ok) throw new Error('HTTP '+r.status);
}

function updateProgress() {
  if (!manifest) return;
  const total = manifest.type==='transition_pairs' ? manifest.pairs.length : manifest.neighbors.length;
  const done = Object.values(progress).filter(e=>e.verdict).length;
  document.getElementById('progress-bar').innerHTML = `<span>${done}</span> / ${total} reviewed`;
}

init();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the file was written**

Run: `python -c "from pathlib import Path; p=Path('scripts/sonic_audition_page.html'); print(p.stat().st_size, 'bytes')"`
Expected: prints a positive byte count (file exists).

- [ ] **Step 3: Commit**

```bash
git add scripts/sonic_audition_page.html
git commit -m "feat(audition): audition page — blinded neighbor list, verdict radio, auto-save"
```

---

### Task 3: Audio streaming server

**Files:**
- Create: `scripts/sonic_audition_serve.py`
- Test: `tests/unit/test_sonic_audition_serve.py`

The server reads manifests from the output directory, serves audio with HTTP range support (required for audio seeking), exposes a blinded manifest API (space_data excluded), and appends to per-seed capture YAMLs on POST /api/save.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sonic_audition_serve.py
import yaml

from scripts.sonic_audition_serve import _parse_range_header, _append_capture_entry


def test_range_full():
    start, end = _parse_range_header("bytes=0-999", 1000)
    assert start == 0 and end == 999


def test_range_from_offset():
    start, end = _parse_range_header("bytes=500-", 1000)
    assert start == 500 and end == 999


def test_range_empty_header():
    start, end = _parse_range_header("", 1000)
    assert start == 0 and end == 999


def test_range_clamps_to_file_size():
    start, end = _parse_range_header("bytes=0-9999", 100)
    assert end == 99


def test_append_creates_file(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"track_id": "t1", "verdict": "match", "notes": "great"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["track_id"] == "t1"


def test_append_updates_existing(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"track_id": "t1", "verdict": "match", "notes": "a"})
    _append_capture_entry(p, {"track_id": "t1", "verdict": "close", "notes": "b"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["verdict"] == "close"


def test_append_adds_new(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"track_id": "t1", "verdict": "match", "notes": ""})
    _append_capture_entry(p, {"track_id": "t2", "verdict": "off", "notes": "nope"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sonic_audition_serve.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# scripts/sonic_audition_serve.py
"""Local HTTP server for the sonic audition harness.

Streams audio with HTTP range support, serves the audition page and blinded
manifests, and appends to per-seed capture YAMLs.

Usage:
    python scripts/sonic_audition_serve.py [--port 8765] [--data-dir docs/run_audits/sonic_audition]

Requires manifests to exist — run sonic_audition_build.py first.
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
    capture_path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")


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
                {"track_id": e["track_id"], "verdict": e.get("verdict", ""), "notes": e.get("notes", "")}
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

        # Resolve artist/title from manifest
        artist, title = "", ""
        for n in m.get("neighbors", []):
            if n["track_id"] == track_id:
                artist, title = n.get("artist", ""), n.get("title", "")
                break

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
    """Load all manifests listed in index.json. Returns (manifests, index)."""
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
        print(f"Page template not found at {page_path}. Ensure Task 2 is complete.")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sonic_audition_serve.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Run the full fast suite to confirm no regressions**

Run: `pytest -m "not slow" -q`
Expected: all pass (same count as before)

- [ ] **Step 6: Commit**

```bash
git add scripts/sonic_audition_serve.py tests/unit/test_sonic_audition_serve.py
git commit -m "feat(audition): HTTP server — range-capable audio, blinded manifest API, YAML capture"
```

---

### Task 4: Analysis script

**Files:**
- Create: `scripts/sonic_audition_analyze.py`
- Test: `tests/unit/test_sonic_audition_analyze.py`

Reads all `*_capture.yaml` files, groups verdicts by sonic space, computes Pearson r between cosine and verdict score, reports on negative-S pairs, writes `findings.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sonic_audition_analyze.py
from scripts.sonic_audition_analyze import aggregate_by_space, cosine_verdict_correlation

VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


def _entry(verdict, spaces):
    return {
        "verdict": verdict, "notes": "",
        "spaces": {s: {"rank": 1, "cosine": 0.4} for s in spaces},
    }


def test_aggregate_counts_by_space():
    entries = [
        _entry("match", ["timbre", "full_track"]),
        _entry("off", ["timbre"]),
        _entry("close", ["rhythm"]),
    ]
    result = aggregate_by_space(entries)
    assert result["timbre"]["match"] == 1
    assert result["timbre"]["off"] == 1
    assert result["full_track"]["match"] == 1
    assert result["rhythm"]["close"] == 1


def test_aggregate_ignores_empty_verdict():
    entries = [_entry("", ["timbre"])]
    result = aggregate_by_space(entries)
    assert "timbre" not in result or all(v == 0 for v in result.get("timbre", {}).values())


def test_cosine_verdict_correlation_pairs():
    entries = [
        {"verdict": "match", "spaces": {"timbre": {"cosine": 0.8, "rank": 1}}},
        {"verdict": "wrong", "spaces": {"timbre": {"cosine": 0.1, "rank": 10}}},
    ]
    rows = cosine_verdict_correlation(entries)
    assert len(rows) == 2
    assert any(r["cosine"] == 0.8 and r["score"] == 3 for r in rows)
    assert any(r["cosine"] == 0.1 and r["score"] == 0 for r in rows)


def test_cosine_verdict_skips_missing_spaces():
    entries = [{"verdict": "match", "spaces": None}]
    rows = cosine_verdict_correlation(entries)
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sonic_audition_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# scripts/sonic_audition_analyze.py
"""Aggregate sonic audition capture files and write findings.

Reads all *_capture.yaml files in the data directory, computes per-space
verdict distributions and cosine-vs-verdict correlation, reports on
negative-S pairs, and writes findings.md.

Usage:
    python scripts/sonic_audition_analyze.py [--data-dir docs/run_audits/sonic_audition]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERDICT_ORDER = ["match", "close", "off", "wrong"]
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}
SPACES = ["full_track", "production_transition", "rhythm", "timbre", "harmony"]


def load_captures(data_dir: Path) -> List[dict]:
    """Return all entries from every *_capture.yaml in data_dir."""
    entries = []
    for p in sorted(data_dir.glob("*_capture.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        slug = p.stem.replace("_capture", "")
        for e in data.get("entries", []):
            e.setdefault("seed", slug)
            entries.append(e)
    return entries


def aggregate_by_space(entries: List[dict]) -> Dict[str, Dict[str, int]]:
    """Return {space: {verdict: count}} for entries with a verdict and space data."""
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in entries:
        verdict = e.get("verdict", "")
        if not verdict:
            continue
        for space in (e.get("spaces") or {}):
            result[space][verdict] += 1
    return {k: dict(v) for k, v in result.items()}


def cosine_verdict_correlation(entries: List[dict]) -> List[dict]:
    """Return [{space, cosine, score}, ...] for all rated entries with cosine data."""
    rows = []
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict not in VERDICT_SCORE:
            continue
        score = VERDICT_SCORE[verdict]
        for space, meta in (e.get("spaces") or {}).items():
            if meta and "cosine" in meta:
                rows.append({"space": space, "cosine": meta["cosine"], "score": score})
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="docs/run_audits/sonic_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    entries = load_captures(data_dir)
    if not entries:
        print(f"No capture entries found in {data_dir}. Complete the audition first.")
        return

    rated = [e for e in entries if e.get("verdict")]
    by_space = aggregate_by_space(entries)
    corr_rows = cosine_verdict_correlation(entries)

    lines = [
        "# Sonic Audition — Phase 2 Findings",
        "",
        f"Total entries: {len(entries)}  |  Rated: {len(rated)}",
        "",
        "## Verdict Distribution by Space",
        "",
        "| Space | match | close | off | wrong | total |",
        "|---|---|---|---|---|---|",
    ]
    for space in SPACES:
        counts = by_space.get(space, {})
        total = sum(counts.values())
        if total == 0:
            continue
        row = [space] + [str(counts.get(v, 0)) for v in VERDICT_ORDER] + [str(total)]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Cosine ↔ Verdict Correlation (Pearson r)", ""]
    space_groups: Dict[str, list] = defaultdict(list)
    for r in corr_rows:
        space_groups[r["space"]].append((r["cosine"], r["score"]))

    for space in SPACES:
        pairs = space_groups.get(space, [])
        if len(pairs) < 3:
            continue
        cosines = np.array([p[0] for p in pairs])
        scores = np.array([p[1] for p in pairs])
        r_val = float(np.corrcoef(cosines, scores)[0, 1])
        lines.append(
            f"- **{space}**: r={r_val:.3f} "
            f"({len(pairs)} rated, cosine [{cosines.min():.3f}, {cosines.max():.3f}])"
        )

    lines += ["", "## Negative-S Transition Pairs", ""]
    neg = [e for e in entries if e.get("seed") == "negative_s"]
    if neg:
        for e in neg:
            lines.append(
                f"- `{e['track_id']}`: verdict=**{e.get('verdict','—')}**"
                + (f" — {e['notes']}" if e.get("notes") else "")
            )
    else:
        lines.append("*(no negative-S entries yet)*")

    lines += ["", "## Notable Notes", ""]
    for e in sorted(rated, key=lambda x: VERDICT_SCORE.get(x.get("verdict",""),0)):
        if e.get("notes"):
            lines.append(
                f"- **{e.get('seed','')}** | {e.get('artist','')} — "
                f"{e.get('title','')} | {e.get('verdict','')} | {e['notes']}"
            )

    out = data_dir / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"\nVerdicts by space:")
    for space, counts in sorted(by_space.items()):
        print(f"  {space}: {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sonic_audition_analyze.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Run the full fast suite**

Run: `pytest -m "not slow" -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add scripts/sonic_audition_analyze.py tests/unit/test_sonic_audition_analyze.py
git commit -m "feat(audition): analysis script — per-space verdict aggregation, cosine correlation"
```

---

### Task 5: End-to-end smoke run

Run the full pipeline to confirm all three scripts work together: build manifests → start server → verify it loads in the browser.

- [ ] **Step 1: Build manifests for a quick subset (3 seeds)**

Run:
```bash
python scripts/sonic_audition_build.py --seeds "Real Estate" "Grouper" "Bill Evans" --top-k 5
```
Expected output:
```
Computing 5 sonic spaces over 39887 tracks...
Looking up file paths...
  OK   'Real Estate' → real_estate_manifest.json (N neighbors)
  OK   'Grouper' → grouper_manifest.json (N neighbors)
  OK   'Bill Evans' → bill_evans_manifest.json (N neighbors)
  OK   negative_s → negative_s_manifest.json (N pairs)

Done. 4 manifests in docs/run_audits/sonic_audition
```
`N` will be 5 or fewer (deduped across 5 spaces).

- [ ] **Step 2: Verify manifest structure**

Run:
```bash
python -c "
import json
from pathlib import Path
m = json.loads(Path('docs/run_audits/sonic_audition/real_estate_manifest.json').read_text())
print('slug:', m['slug'])
print('type:', m['type'])
print('seed:', m['seed']['artist'], '—', m['seed']['title'])
print('neighbors:', len(m['neighbors']))
print('space_data keys:', list(m['space_data'].values())[:1])
n = m['neighbors'][0]
print('first neighbor keys:', list(n.keys()))
print('space_data NOT in neighbor:', 'spaces' not in n)
"
```
Expected: `space_data NOT in neighbor: True` — the blind is intact.

- [ ] **Step 3: Start server and verify it launches**

Run in a separate terminal (or with `&` in bash):
```bash
python scripts/sonic_audition_serve.py --port 8765
```
Expected: prints `Audition server → http://127.0.0.1:8765/` and opens a browser tab.

Then confirm the manifest API works:
```bash
python -c "
import urllib.request, json
data = json.loads(urllib.request.urlopen('http://127.0.0.1:8765/api/manifest/real_estate').read())
print('type:', data['type'])
print('neighbors:', len(data['neighbors']))
print('space_data absent:', 'space_data' not in data)
"
```
Expected: `space_data absent: True`

Stop the server with Ctrl+C.

- [ ] **Step 4: Build the full 17-seed manifests**

Run (this takes ~30-60s to compute all 5 spaces over 40k tracks):
```bash
python scripts/sonic_audition_build.py
```
Expected: 17 seed manifests + negative_s manifest, all printed OK.

- [ ] **Step 5: Commit**

```bash
git add docs/run_audits/sonic_audition/.gitkeep 2>/dev/null; true
git commit -m "feat(audition): Phase 2 harness complete — build/serve/analyze pipeline"
```

(The `docs/run_audits/` dir is gitignored; the commit message just marks the milestone.)

---

## Self-Review

**Spec coverage:**
- Part A (neighbor computation, 5 spaces, top-15, blinded merge) → Task 1 ✅
- Part B (local server, audio range, blinded manifest API, POST /save, seed selector, verdict+notes, auto-save, codec guard — FLAC/MP3/AAC are browser-playable, .m4a flagged by 404 if it's ALAC — functionally covered via `CONTENT_TYPES`) → Tasks 2+3 ✅
- Part C (analysis/findings) → Task 4 ✅
- Negative-S pairs as dedicated audition set → Task 1 (`build_negative_s_manifest`) + Task 2 (`renderPairs()`) ✅
- Capture file format (`seed_capture.yaml`, append-only, resumable) → Task 3 (`_append_capture_entry`) ✅
- Progress restore on page load (`/api/progress/<slug>`) → Tasks 2+3 ✅

**Placeholder scan:** No TBD/TODO in any code block. The `_slug("Beyoncé")` test note is a real edge case (unicode stripped → "beyonc") — implementer should verify this matches the actual artist name in the bundle. If "Beyoncé" doesn't match by lowercased artist string, the manifest builder returns None and prints SKIP — that's correct handling.

**Type consistency:** `compute_spaces(bundle, per_tower)` used with two args throughout Tasks 1 and 5. `_append_capture_entry(path, entry)` consistent across Tasks 3 and tests. `aggregate_by_space(entries)` / `cosine_verdict_correlation(entries)` consistent across Task 4 implementation and tests.
