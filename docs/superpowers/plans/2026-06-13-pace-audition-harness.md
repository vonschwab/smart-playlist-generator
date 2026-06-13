# Pace Audition Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a blind, transition-level perceptual audition (build/serve/analyze trio + page) that validates whether the shipped `pace_mode: narrow` caps are audibly doing their job without over-tightening.

**Architecture:** Three standalone scripts under `scripts/` mirroring the existing sonic/genre audition trio, plus a static HTML page. `pace_audition_build.py` generates 4 arms (narrow/dynamic/off + synthesized decoy edges) per seed via the production fidelity harness, extracts interior bridge edges, blinds them, and writes a manifest. `pace_audition_serve.py` (forked from `sonic_audition_serve.py`) streams audio windows and captures dual scores. `pace_audition_analyze.py` un-blinds and reports distributions. All output goes under `docs/run_audits/pace_audition/` (gitignored); pure functions are unit-tested with synthetic data, the artifact-dependent generation is verified by one real run.

**Tech Stack:** Python 3.11, NumPy, PyYAML, sqlite3 (stdlib), http.server (stdlib), vanilla JS/HTML. No new pip installs. Reuses `tests/support/gui_fidelity.generate_like_gui`, `src.playlist.bpm_axis.bpm_log_distance`, `src.playlist.bpm_loader.load_bpm_arrays`, `src.features.artifacts.load_artifact_bundle`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/pace_audition_build.py` | Pure functions (edge metrics, interior-edge extraction, sampling, decoy synthesis, blinding, manifest assembly) + `main()` generation glue. |
| `scripts/pace_audition_serve.py` | Forked server: `/audio/<track_id>` range streaming, blinded manifest API, dual-score capture. |
| `scripts/pace_audition_page.html` | Edge-player UI: play A-tail→hard-cut→B-head, two 1-5 sliders, notes, auto-save. |
| `scripts/pace_audition_analyze.py` | Pure functions (distributions, contrasts, confound, discrimination, onset-variance) + `main()` + `findings.md` writer. |
| `tests/unit/test_pace_audition_build.py` | Tests for edge metrics, extraction, sampling, decoy, blinding (incl. no-arm-leak). |
| `tests/unit/test_pace_audition_serve.py` | Tests for blinded-manifest stripping, capture upsert, range parse. |
| `tests/unit/test_pace_audition_analyze.py` | Tests for distribution, contrast, confound, discrimination, onset-variance. |

**Manifest schema** (`pace_manifest.json`) — fixed contract used by all three scripts:

```python
{
  "type": "pace_edges",
  "provenance": {
    "artifact_path": str, "artifact_mtime": float, "artifact_fingerprint": str,
    "random_seed": int,
    "fixed_modes": {"cohesion_mode": "dynamic", "genre_mode": "narrow", "sonic_mode": "narrow"},
    "arms": {"narrow": {...pace cfg...}, "dynamic": {...}, "off": {...}},
    "seeds": {"green-house": {"piers": [tid, ...], "regime": "ambient"}, ...},
  },
  "playlists": [   # for the structural onset-variance check (full playlist, arm-labelled)
    {"seed": "green-house", "arm": "narrow", "track_ids": [...], "onset_seq": [float|None, ...]},
    ...
  ],
  "edges": [       # BLINDED served view: ids only, no arm/seed/artist
    {"edge_id": "e0001", "a": "tidA", "b": "tidB"}, ...
  ],
  "edge_data": {   # SERVER-SIDE ONLY (stripped before serving)
    "e0001": {"arm": "narrow", "seed": "green-house", "regime": "ambient",
              "a": {"track_id": "tidA", "artist": "...", "title": "...", "onset": 2.1, "bpm": 90.0},
              "b": {"track_id": "tidB", "artist": "...", "title": "...", "onset": 2.0, "bpm": 92.0},
              "onset_log_dist": 0.07, "bpm_log_dist": 0.03, "genre_cos": 0.72}, ...
  },
  "file_paths": {"tidA": "E:\\...", "tidB": "..."}  # SERVER-SIDE for audio routing
}
```

Capture (`pace_capture.yaml`): `entries` list, each `{edge_id, continuity, smoothness, notes, saved_at}`.

---

## Task 1: Build — edge metrics + interior-edge extraction

**Files:**
- Create: `scripts/pace_audition_build.py`
- Test: `tests/unit/test_pace_audition_build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pace_audition_build.py
import numpy as np

from scripts.pace_audition_build import (
    genre_cosine,
    edge_metrics,
    extract_interior_edges,
)


def test_genre_cosine_is_l2_normalized_dot():
    u = np.array([3.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    assert genre_cosine(u, v) == 1.0
    w = np.array([0.0, 5.0, 0.0])
    assert abs(genre_cosine(u, w)) < 1e-9


def test_genre_cosine_zero_vector_is_zero():
    assert genre_cosine(np.zeros(4), np.ones(4)) == 0.0


def test_edge_metrics_uses_log2_distance_and_genre_cos():
    m = edge_metrics(
        a_onset=2.0, b_onset=4.0, a_bpm=90.0, b_bpm=90.0,
        a_genre=np.array([1.0, 0.0]), b_genre=np.array([1.0, 0.0]),
    )
    assert abs(m["onset_log_dist"] - 1.0) < 1e-9   # 4/2 = one octave
    assert abs(m["bpm_log_dist"] - 0.0) < 1e-9
    assert abs(m["genre_cos"] - 1.0) < 1e-9


def test_extract_interior_edges_excludes_pier_adjacent():
    # positions:        0(pier) 1 2 3 4(pier) 5
    track_ids = ["p0", "a", "b", "c", "p1", "d"]
    piers = {"p0", "p1"}
    # interior edges = consecutive pairs with NEITHER endpoint a pier
    # (1,2)=a-b ok, (2,3)=b-c ok, (3,4)=c-p1 excluded, (4,5)=p1-d excluded, (0,1)=p0-a excluded
    assert extract_interior_edges(track_ids, piers) == [(1, 2), (2, 3)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pace_audition_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pace_audition_build'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pace_audition_build.py
"""Build the blinded pace-audition manifest.

Generates narrow/dynamic/off playlists per seed via the production fidelity
harness, extracts interior bridge edges, synthesizes pace-bad decoy edges,
blinds everything, and writes pace_manifest.json + index.json under
docs/run_audits/pace_audition/. Read-only against metadata.db and the artifact.

Usage:
    python scripts/pace_audition_build.py
    python scripts/pace_audition_build.py --seeds "Green-House" "J Dilla"
    python scripts/pace_audition_build.py --edges-per-arm 5 --decoy-per-seed 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.playlist.bpm_axis import bpm_log_distance  # noqa: E402


def genre_cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def edge_metrics(
    *, a_onset: float, b_onset: float, a_bpm: float, b_bpm: float,
    a_genre: np.ndarray, b_genre: np.ndarray,
) -> Dict[str, float]:
    return {
        "onset_log_dist": float(bpm_log_distance(a_onset, b_onset)),
        "bpm_log_dist": float(bpm_log_distance(a_bpm, b_bpm)),
        "genre_cos": genre_cosine(a_genre, b_genre),
    }


def extract_interior_edges(
    track_ids: Sequence[str], pier_ids: set
) -> List[Tuple[int, int]]:
    """Consecutive (i, i+1) index pairs where NEITHER endpoint is a pier."""
    out: List[Tuple[int, int]] = []
    for i in range(len(track_ids) - 1):
        if str(track_ids[i]) in pier_ids or str(track_ids[i + 1]) in pier_ids:
            continue
        out.append((i, i + 1))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_pace_audition_build.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/pace_audition_build.py tests/unit/test_pace_audition_build.py
git commit -m "feat(pace-audition): edge metrics + interior-edge extraction"
```

---

## Task 2: Build — seeded edge sampling + decoy synthesis

**Files:**
- Modify: `scripts/pace_audition_build.py`
- Test: `tests/unit/test_pace_audition_build.py`

- [ ] **Step 1: Write the failing test (append)**

```python
from scripts.pace_audition_build import sample_edges, synthesize_decoy_edges


def test_sample_edges_deterministic_and_bounded():
    edges = [(i, i + 1) for i in range(10)]
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = sample_edges(edges, k=3, rng=rng1)
    b = sample_edges(edges, k=3, rng=rng2)
    assert a == b              # same seed → same sample
    assert len(a) == 3
    assert all(e in edges for e in a)


def test_sample_edges_returns_all_when_fewer_than_k():
    edges = [(0, 1), (1, 2)]
    out = sample_edges(edges, k=5, rng=np.random.default_rng(0))
    assert sorted(out) == sorted(edges)


def test_synthesize_decoy_prefers_pace_distant_genre_close_pairs():
    # 3 tracks: t0,t1 close pace & same genre; t0,t2 far pace (octave) same genre.
    tids = ["t0", "t1", "t2"]
    onset = {"t0": 2.0, "t1": 2.1, "t2": 8.0}     # t0-t2 log dist = 2.0 (>1.0)
    bpm = {"t0": 90.0, "t1": 91.0, "t2": 90.0}
    genre = {"t0": np.array([1.0, 0.0]), "t1": np.array([1.0, 0.0]),
             "t2": np.array([1.0, 0.0])}           # all identical genre
    decoys = synthesize_decoy_edges(
        tids, onset=onset, bpm=bpm, genre_vecs=genre,
        k=1, rng=np.random.default_rng(0), min_onset_dist=1.0,
    )
    assert len(decoys) == 1
    a, b = decoys[0]
    assert {a, b} == {"t0", "t2"}                  # the only pace-distant pair
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pace_audition_build.py -q`
Expected: FAIL — `ImportError: cannot import name 'sample_edges'`

- [ ] **Step 3: Implement (append to `scripts/pace_audition_build.py`)**

```python
def sample_edges(
    edges: List[Tuple[int, int]], k: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Seeded sample of up to k edges (all of them if fewer than k)."""
    if len(edges) <= k:
        return list(edges)
    idx = rng.choice(len(edges), size=k, replace=False)
    return [edges[int(i)] for i in sorted(idx)]


def synthesize_decoy_edges(
    context_tids: Sequence[str],
    *,
    onset: Dict[str, float],
    bpm: Dict[str, float],
    genre_vecs: Dict[str, np.ndarray],
    k: int,
    rng: np.random.Generator,
    min_onset_dist: float = 1.0,
) -> List[Tuple[str, str]]:
    """Pairs that are pace-distant (onset log-dist > min) but genre-close
    (genre cos >= median over qualifying pairs). Falls back to the 25th
    percentile genre floor if fewer than k qualify. Returns (a, b) track-id pairs."""
    tids = [str(t) for t in context_tids]
    cand: List[Tuple[str, str, float]] = []  # (a, b, genre_cos)
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            if not np.isfinite(float(bpm_log_distance(onset.get(a, np.nan), onset.get(b, np.nan)))):
                continue
            od = float(bpm_log_distance(onset.get(a, np.nan), onset.get(b, np.nan)))
            if od <= float(min_onset_dist):
                continue
            gc = genre_cosine(genre_vecs.get(a, np.zeros(1)), genre_vecs.get(b, np.zeros(1)))
            cand.append((a, b, gc))
    if not cand:
        return []
    gcs = np.array([c[2] for c in cand])
    floor = float(np.median(gcs))
    qualifying = [(a, b) for (a, b, gc) in cand if gc >= floor]
    if len(qualifying) < k:
        floor = float(np.percentile(gcs, 25))
        qualifying = [(a, b) for (a, b, gc) in cand if gc >= floor]
    if len(qualifying) <= k:
        return qualifying
    idx = rng.choice(len(qualifying), size=k, replace=False)
    return [qualifying[int(i)] for i in sorted(idx)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_pace_audition_build.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/pace_audition_build.py tests/unit/test_pace_audition_build.py
git commit -m "feat(pace-audition): seeded edge sampling + decoy synthesis"
```

---

## Task 3: Build — blinding + manifest assembly (no-arm-leak guard)

**Files:**
- Modify: `scripts/pace_audition_build.py`
- Test: `tests/unit/test_pace_audition_build.py`

- [ ] **Step 1: Write the failing test (append)**

```python
from scripts.pace_audition_build import blind_and_shuffle


def _edge_record(arm, seed, a, b):
    return {
        "arm": arm, "seed": seed, "regime": "ambient",
        "a": {"track_id": a, "artist": "AA", "title": "TA", "onset": 2.0, "bpm": 90.0},
        "b": {"track_id": b, "artist": "BB", "title": "TB", "onset": 2.0, "bpm": 90.0},
        "onset_log_dist": 0.0, "bpm_log_dist": 0.0, "genre_cos": 1.0,
    }


def test_blind_and_shuffle_strips_arm_from_served_edges():
    records = [
        _edge_record("narrow", "s", "a1", "b1"),
        _edge_record("decoy", "s", "a2", "b2"),
    ]
    edges, edge_data = blind_and_shuffle(records, rng=np.random.default_rng(0))
    # served edges expose ONLY edge_id + the two track ids
    for e in edges:
        assert set(e.keys()) == {"edge_id", "a", "b"}
    # the arm lives only in the server-side edge_data, keyed by edge_id
    assert set(edge_data.keys()) == {e["edge_id"] for e in edges}
    arms = {edge_data[e["edge_id"]]["arm"] for e in edges}
    assert arms == {"narrow", "decoy"}


def test_blind_and_shuffle_is_deterministic():
    records = [_edge_record("off", "s", f"a{i}", f"b{i}") for i in range(6)]
    e1, _ = blind_and_shuffle(records, rng=np.random.default_rng(1))
    e2, _ = blind_and_shuffle(records, rng=np.random.default_rng(1))
    assert [e["edge_id"] for e in e1] == [e["edge_id"] for e in e2]
    assert [e["a"] for e in e1] == [e["a"] for e in e2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pace_audition_build.py -q`
Expected: FAIL — `ImportError: cannot import name 'blind_and_shuffle'`

- [ ] **Step 3: Implement (append to `scripts/pace_audition_build.py`)**

```python
def blind_and_shuffle(
    records: List[dict], rng: np.random.Generator
) -> Tuple[List[dict], Dict[str, dict]]:
    """Assign stable edge_ids, shuffle order, and split each record into a
    BLINDED served view (edge_id + two track ids only) and a server-side
    edge_data entry (arm/seed/metrics). The arm NEVER appears in the served view."""
    order = list(range(len(records)))
    rng.shuffle(order)
    edges: List[dict] = []
    edge_data: Dict[str, dict] = {}
    for new_pos, orig_i in enumerate(order):
        rec = records[orig_i]
        eid = f"e{new_pos + 1:04d}"
        edges.append({"edge_id": eid, "a": rec["a"]["track_id"], "b": rec["b"]["track_id"]})
        edge_data[eid] = rec
    return edges, edge_data
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_pace_audition_build.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/pace_audition_build.py tests/unit/test_pace_audition_build.py
git commit -m "feat(pace-audition): blinding/shuffle with no-arm-leak guard"
```

---

## Task 4: Build — `main()` generation glue + provenance + real run

**Files:**
- Modify: `scripts/pace_audition_build.py`

No new unit test (this is artifact + generation glue; verified by a real run). The pure functions it calls are already covered.

- [ ] **Step 1: Add the DB/file-path + onset loaders and `main()` (append)**

```python
def lookup_file_paths(track_ids: List[str], db_path: str) -> Dict[str, str]:
    """Read-only lookup of file_path by track_id (mirrors sonic_audition_build)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    result: Dict[str, str] = {}
    for i in range(0, len(track_ids), 900):
        chunk = track_ids[i : i + 900]
        ph = ",".join(["?"] * len(chunk))
        rows = con.execute(
            f"SELECT track_id, file_path FROM tracks WHERE track_id IN ({ph})", chunk
        ).fetchall()
        result.update({str(r[0]): str(r[1]) for r in rows})
    con.close()
    return result


def _artifact_fingerprint(X: np.ndarray) -> str:
    """Hash a few rows so analyze can assert all arms used the same artifact."""
    sample = np.ascontiguousarray(X[:: max(1, X.shape[0] // 50)][:50])
    return hashlib.sha256(sample.tobytes()).hexdigest()[:16]


AMBIENT = ["Green-House", "Hiroshi Yoshimura", "Brian Eno"]
RHYTHMIC = ["J Dilla", "De La Soul", "Beastie Boys"]
REAL_ARMS = ["narrow", "dynamic", "off"]
FIXED_MODES = {"cohesion_mode": "dynamic", "genre_mode": "narrow", "sonic_mode": "narrow"}


def _artist_piers(bundle, artist: str, max_piers: int = 5) -> List[str]:
    artists = np.array([str(a) for a in bundle.track_artists])
    idx = np.where(np.char.lower(artists) == artist.lower())[0]
    return [str(bundle.track_ids[i]) for i in idx[:max_piers]]


def main() -> None:
    from src.features.artifacts import load_artifact_bundle
    from src.playlist.bpm_loader import load_bpm_arrays
    from tests.support.gui_fidelity import generate_like_gui, resolved_artifact_path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", default=AMBIENT + RHYTHMIC)
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--edges-per-arm", type=int, default=5)
    ap.add_argument("--decoy-per-seed", type=int, default=5)
    ap.add_argument("--db", default="data/metadata.db")
    ap.add_argument("--out-dir", default="docs/run_audits/pace_audition")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    art_path = resolved_artifact_path()
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(art_path)
    tids_all = [str(t) for t in bundle.track_ids]
    tid_to_idx = bundle.track_id_to_index

    bpm_arrays = load_bpm_arrays(bundle.track_ids, db_path=args.db)
    onset_arr = bpm_arrays["onset_rate"]
    bpm_arr = bpm_arrays["perceptual_bpm"]
    Xg = bundle.X_genre_smoothed

    def onset_of(tid): return float(onset_arr[tid_to_idx[tid]])
    def bpm_of(tid): return float(bpm_arr[tid_to_idx[tid]])
    def genre_of(tid): return Xg[tid_to_idx[tid]]

    rng = np.random.default_rng(args.random_seed)
    regime_of = {**{a: "ambient" for a in AMBIENT}, **{a: "rhythmic" for a in RHYTHMIC}}

    records: List[dict] = []
    playlists: List[dict] = []
    seeds_prov: Dict[str, dict] = {}
    index_seeds: List[str] = []

    for artist in args.seeds:
        piers = _artist_piers(bundle, artist)
        if len(piers) < 4:
            print(f"  SKIP {artist!r}: only {len(piers)} piers in artifact")
            continue
        slug = artist.lower().replace(" ", "-").replace("/", "-")
        seeds_prov[slug] = {"piers": piers, "regime": regime_of.get(artist, "?")}
        index_seeds.append(artist)
        pier_set = set(piers)
        context: set = set()

        for arm in REAL_ARMS:
            res = generate_like_gui(
                seeds=piers, pace_mode=arm, length=args.length,
                random_seed=args.random_seed, **FIXED_MODES,
            )
            tids = [str(t) for t in res.track_ids]
            context.update(tids)
            onset_seq = [onset_of(t) if t in tid_to_idx else None for t in tids]
            playlists.append({"seed": slug, "arm": arm, "track_ids": tids, "onset_seq": onset_seq})

            interior = extract_interior_edges(tids, pier_set)
            for (i, j) in sample_edges(interior, args.edges_per_arm, rng):
                a, b = tids[i], tids[j]
                m = edge_metrics(
                    a_onset=onset_of(a), b_onset=onset_of(b),
                    a_bpm=bpm_of(a), b_bpm=bpm_of(b),
                    a_genre=genre_of(a), b_genre=genre_of(b),
                )
                records.append({
                    "arm": arm, "seed": slug, "regime": regime_of.get(artist, "?"),
                    "a": {"track_id": a, "artist": str(bundle.track_artists[tid_to_idx[a]]),
                          "title": str(bundle.track_titles[tid_to_idx[a]]),
                          "onset": onset_of(a), "bpm": bpm_of(a)},
                    "b": {"track_id": b, "artist": str(bundle.track_artists[tid_to_idx[b]]),
                          "title": str(bundle.track_titles[tid_to_idx[b]]),
                          "onset": onset_of(b), "bpm": bpm_of(b)},
                    **m,
                })

        # Decoy edges from this seed's context set
        ctx = [t for t in context if t in tid_to_idx]
        decoys = synthesize_decoy_edges(
            ctx,
            onset={t: onset_of(t) for t in ctx},
            bpm={t: bpm_of(t) for t in ctx},
            genre_vecs={t: genre_of(t) for t in ctx},
            k=args.decoy_per_seed, rng=rng, min_onset_dist=1.0,
        )
        for (a, b) in decoys:
            m = edge_metrics(
                a_onset=onset_of(a), b_onset=onset_of(b),
                a_bpm=bpm_of(a), b_bpm=bpm_of(b),
                a_genre=genre_of(a), b_genre=genre_of(b),
            )
            records.append({
                "arm": "decoy", "seed": slug, "regime": regime_of.get(artist, "?"),
                "a": {"track_id": a, "artist": str(bundle.track_artists[tid_to_idx[a]]),
                      "title": str(bundle.track_titles[tid_to_idx[a]]),
                      "onset": onset_of(a), "bpm": bpm_of(a)},
                "b": {"track_id": b, "artist": str(bundle.track_artists[tid_to_idx[b]]),
                      "title": str(bundle.track_titles[tid_to_idx[b]]),
                      "onset": onset_of(b), "bpm": bpm_of(b)},
                **m,
            })
        print(f"  OK   {artist!r}: {len(interior)} interior edges available")

    edges, edge_data = blind_and_shuffle(records, rng)
    needed = {e["a"] for e in edges} | {e["b"] for e in edges}
    file_paths = lookup_file_paths(sorted(needed), args.db)

    manifest = {
        "type": "pace_edges",
        "provenance": {
            "artifact_path": art_path,
            "artifact_mtime": Path(art_path).stat().st_mtime,
            "artifact_fingerprint": _artifact_fingerprint(bundle.X_sonic),
            "random_seed": args.random_seed,
            "fixed_modes": FIXED_MODES,
            "seeds": seeds_prov,
        },
        "playlists": playlists,
        "edges": edges,
        "edge_data": edge_data,
        "file_paths": file_paths,
    }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pace_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "index.json").write_text(
        json.dumps([{"slug": "pace", "label": "Pace Edges"}], indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(edges)} edges ({len(playlists)} playlists) → {out_dir}/pace_manifest.json")
    print("Next: python scripts/pace_audition_serve.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the build for real (artifact-dependent; ~15-20 min for 18 generations)**

Run: `python scripts/pace_audition_build.py`
Expected: per-seed `OK` lines, final `Done. ~120 edges (18 playlists) → .../pace_manifest.json`. No writes outside `docs/run_audits/pace_audition/`.

- [ ] **Step 3: Sanity-check the manifest is blinded and provenant**

Run:
```bash
python -c "import json; m=json.load(open('docs/run_audits/pace_audition/pace_manifest.json')); \
print('edges', len(m['edges'])); \
print('served keys', sorted(m['edges'][0].keys())); \
print('arms', sorted({d['arm'] for d in m['edge_data'].values()})); \
print('fp', m['provenance']['artifact_fingerprint'])"
```
Expected: `served keys ['a', 'b', 'edge_id']` (no arm), `arms ['decoy', 'dynamic', 'narrow', 'off']`.

- [ ] **Step 4: Commit (code only — output dir is gitignored)**

```bash
git add scripts/pace_audition_build.py
git commit -m "feat(pace-audition): build main() — generate arms, blind, write manifest"
```

---

## Task 5: Serve — forked server with blinded manifest + dual-score capture

**Files:**
- Create: `scripts/pace_audition_serve.py`
- Test: `tests/unit/test_pace_audition_serve.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pace_audition_serve.py
from scripts.pace_audition_serve import blinded_manifest, upsert_capture_entry


def test_blinded_manifest_strips_all_server_side_keys():
    manifest = {
        "type": "pace_edges",
        "provenance": {"arms": {"narrow": {}}},
        "playlists": [{"seed": "s", "arm": "narrow"}],
        "edges": [{"edge_id": "e0001", "a": "ta", "b": "tb"}],
        "edge_data": {"e0001": {"arm": "narrow"}},
        "file_paths": {"ta": "x"},
    }
    served = blinded_manifest(manifest)
    assert set(served.keys()) == {"type", "edges"}
    assert "narrow" not in json_dumps(served)


def json_dumps(o):
    import json
    return json.dumps(o)


def test_upsert_capture_entry_dedupes_by_edge_id():
    entries = [{"edge_id": "e1", "continuity": 3, "smoothness": 3}]
    upsert_capture_entry(entries, {"edge_id": "e1", "continuity": 5, "smoothness": 4})
    upsert_capture_entry(entries, {"edge_id": "e2", "continuity": 2, "smoothness": 2})
    assert len(entries) == 2
    assert entries[0]["continuity"] == 5  # overwritten in place
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pace_audition_serve.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pace_audition_serve'`

- [ ] **Step 3: Implement** (fork `sonic_audition_serve.py`; reuse `_parse_range_header` verbatim, replace the audio lookup, manifest API, and capture)

```python
# scripts/pace_audition_serve.py
"""Local HTTP server for the pace audition. Streams whole audio files with
range support; the page seeks client-side to play A-tail -> hard cut -> B-head.
Serves the BLINDED manifest (no arm/seed) and captures dual scores to YAML.

Usage:
    python scripts/pace_audition_serve.py [--port 8766] [--data-dir docs/run_audits/pace_audition]
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
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--data-dir", default="docs/run_audits/pace_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    mpath = data_dir / "pace_manifest.json"
    if not mpath.exists():
        print(f"No manifest at {mpath}. Run: python scripts/pace_audition_build.py")
        sys.exit(1)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    page = (Path(__file__).parent / "pace_audition_page.html").read_text(encoding="utf-8")

    server = PaceServer(("127.0.0.1", args.port), PaceHandler, data_dir, manifest, page)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving pace audition at {url} ({len(manifest['edges'])} edges). Ctrl-C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_pace_audition_serve.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/pace_audition_serve.py tests/unit/test_pace_audition_serve.py
git commit -m "feat(pace-audition): forked serve with blinded manifest + dual-score capture"
```

---

## Task 6: Page — edge-player UI

**Files:**
- Create: `scripts/pace_audition_page.html`

No unit test (static asset; verified by loading it against the served manifest in Task 9).

- [ ] **Step 1: Create the page**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pace Audition</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Courier New', monospace; background: #1a1a1a; color: #e0e0e0; }
.bar { background: #111; border-bottom: 1px solid #333; padding: 10px 16px; position: sticky; top: 0; }
.bar span { color: #4caf50; font-weight: bold; }
.main { max-width: 720px; margin: 0 auto; padding: 16px; }
.edge { background: #222; border: 1px solid #333; border-radius: 4px; padding: 14px; margin-bottom: 12px; }
.edge h3 { font-size: 13px; color: #aaa; margin-bottom: 10px; }
button.play { background: #2a2a2a; color: #e0e0e0; border: 1px solid #555; border-radius: 3px;
  padding: 8px 16px; font-family: inherit; font-size: 13px; cursor: pointer; }
button.play:hover { background: #333; }
.row { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
.row label { font-size: 12px; color: #bbb; width: 150px; }
input[type=range] { flex: 1; accent-color: #4caf50; }
.val { width: 24px; text-align: center; color: #4caf50; }
textarea { width: 100%; background: #1a1a1a; color: #ddd; border: 1px solid #444; border-radius: 3px;
  padding: 6px; font-family: inherit; font-size: 12px; min-height: 40px; margin-top: 6px; }
.status { font-size: 11px; color: #4caf50; min-height: 14px; }
.status.err { color: #f44336; }
</style>
</head>
<body>
<div class="bar"><span id="done">0</span> / <span id="total">0</span> rated</div>
<div class="main" id="main"></div>
<script>
const TAIL = 12, HEAD = 12;       // seconds of A-tail and B-head
let manifest = null, progress = {};

async function init() {
  const [m, p] = await Promise.all([fetch('/api/manifest'), fetch('/api/progress')]);
  manifest = await m.json();
  (await p.json()).forEach(e => { progress[e.edge_id] = e; });
  document.getElementById('total').textContent = manifest.edges.length;
  const main = document.getElementById('main');
  manifest.edges.forEach((e, i) => main.appendChild(card(e, i)));
  updateDone();
}

function card(e, i) {
  const s = progress[e.edge_id] || {};
  const div = document.createElement('div');
  div.className = 'edge'; div.id = 'c-' + e.edge_id;
  div.innerHTML = `
    <h3>Transition #${i + 1}</h3>
    <button class="play" onclick="playEdge('${e.a}','${e.b}',this)">▶ Play transition</button>
    <div class="row"><label>Energy/tempo continuity</label>
      <input type="range" min="1" max="5" value="${s.continuity || 3}"
        oninput="document.getElementById('cv-${e.edge_id}').textContent=this.value"
        onchange="save('${e.edge_id}')" id="cont-${e.edge_id}">
      <span class="val" id="cv-${e.edge_id}">${s.continuity || 3}</span></div>
    <div class="row"><label>Overall smoothness</label>
      <input type="range" min="1" max="5" value="${s.smoothness || 3}"
        oninput="document.getElementById('sv-${e.edge_id}').textContent=this.value"
        onchange="save('${e.edge_id}')" id="smooth-${e.edge_id}">
      <span class="val" id="sv-${e.edge_id}">${s.smoothness || 3}</span></div>
    <textarea id="nt-${e.edge_id}" placeholder="Notes…" onblur="save('${e.edge_id}')">${s.notes || ''}</textarea>
    <div class="status" id="st-${e.edge_id}"></div>`;
  return div;
}

function playEdge(aTid, bTid, btn) {
  btn.disabled = true;
  const a = new Audio('/audio/' + aTid);
  a.addEventListener('loadedmetadata', () => {
    a.currentTime = Math.max(0, (a.duration || TAIL) - TAIL);
    a.play();
  });
  a.addEventListener('timeupdate', () => {
    if (a.currentTime >= (a.duration - 0.15)) toB();
  });
  let switched = false;
  function toB() {
    if (switched) return; switched = true;
    a.pause();
    const b = new Audio('/audio/' + bTid);
    b.addEventListener('loadedmetadata', () => b.play());
    b.addEventListener('timeupdate', () => {
      if (b.currentTime >= HEAD) { b.pause(); btn.disabled = false; }
    });
    b.addEventListener('ended', () => { btn.disabled = false; });
  }
  a.addEventListener('ended', toB);
}

async function save(eid) {
  const continuity = +document.getElementById('cont-' + eid).value;
  const smoothness = +document.getElementById('smooth-' + eid).value;
  const notes = document.getElementById('nt-' + eid).value;
  const st = document.getElementById('st-' + eid);
  try {
    const r = await fetch('/api/save', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ edge_id: eid, continuity, smoothness, notes }) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    progress[eid] = { edge_id: eid, continuity, smoothness, notes };
    st.textContent = 'saved ✓'; st.className = 'status';
    updateDone(); setTimeout(() => { st.textContent = ''; }, 2500);
  } catch (e) { st.textContent = 'error'; st.className = 'status err'; }
}

function updateDone() {
  document.getElementById('done').textContent =
    Object.values(progress).filter(e => e.continuity != null).length;
}
init();
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add scripts/pace_audition_page.html
git commit -m "feat(pace-audition): edge-player page (A-tail -> hard cut -> B-head, dual sliders)"
```

---

## Task 7: Analyze — distributions, contrasts, confound, discrimination

**Files:**
- Create: `scripts/pace_audition_analyze.py`
- Test: `tests/unit/test_pace_audition_analyze.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pace_audition_analyze.py
from scripts.pace_audition_analyze import (
    distribution,
    join_scores,
    per_arm,
    discrimination_ok,
    confound_flag,
)


def test_distribution_reports_min_p10_p50_p90_n():
    d = distribution([1, 2, 3, 4, 5])
    assert d["n"] == 5
    assert d["min"] == 1
    assert d["p50"] == 3


def test_distribution_handles_empty():
    d = distribution([])
    assert d["n"] == 0
    assert d["p50"] is None


def test_join_scores_attaches_arm_and_regime():
    edge_data = {"e1": {"arm": "narrow", "seed": "s", "regime": "ambient"}}
    captures = [{"edge_id": "e1", "continuity": 4, "smoothness": 3}]
    joined = join_scores(captures, edge_data)
    assert joined[0]["arm"] == "narrow"
    assert joined[0]["regime"] == "ambient"
    assert joined[0]["continuity"] == 4


def test_per_arm_groups_continuity():
    joined = [
        {"arm": "narrow", "continuity": 5, "smoothness": 4, "regime": "ambient"},
        {"arm": "decoy", "continuity": 1, "smoothness": 1, "regime": "ambient"},
    ]
    out = per_arm(joined, "continuity")
    assert out["narrow"]["p50"] == 5
    assert out["decoy"]["p50"] == 1


def test_discrimination_ok_requires_decoy_lowest():
    good = {"narrow": {"p50": 4}, "dynamic": {"p50": 4}, "off": {"p50": 3}, "decoy": {"p50": 1}}
    bad = {"narrow": {"p50": 3}, "dynamic": {"p50": 3}, "off": {"p50": 3}, "decoy": {"p50": 3}}
    assert discrimination_ok(good) is True
    assert discrimination_ok(bad) is False


def test_confound_flag_true_when_continuity_gain_exceeds_smoothness_gain():
    # narrow beats dynamic more on continuity than on smoothness -> pace-specific
    joined = [
        {"arm": "narrow", "continuity": 5, "smoothness": 4},
        {"arm": "dynamic", "continuity": 2, "smoothness": 3.5},
    ]
    res = confound_flag(joined)
    assert res["pace_specific"] is True
    assert res["continuity_gain"] > res["smoothness_gain"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pace_audition_analyze.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pace_audition_analyze'`

- [ ] **Step 3: Implement**

```python
# scripts/pace_audition_analyze.py
"""Un-blind the pace audition captures and write findings.

Reads pace_capture.yaml + pace_manifest.json (server-side edge_data), reports
per-arm continuity/smoothness distributions sliced by regime, runs the
discrimination check (decoy lowest), the narrow-vs-dynamic-vs-off contrasts,
the pace-specificity confound check, and the structural onset-variance check.

Usage:
    python scripts/pace_audition_analyze.py [--data-dir docs/run_audits/pace_audition]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ["narrow", "dynamic", "off", "decoy"]


def distribution(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "p10": None, "p50": None, "p90": None}
    arr = np.array(vals)
    return {
        "n": len(vals),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def join_scores(captures: List[dict], edge_data: Dict[str, dict]) -> List[dict]:
    joined = []
    for c in captures:
        meta = edge_data.get(c.get("edge_id", ""))
        if not meta:
            continue
        joined.append({
            "edge_id": c["edge_id"],
            "arm": meta["arm"], "seed": meta["seed"], "regime": meta.get("regime", "?"),
            "continuity": c.get("continuity"), "smoothness": c.get("smoothness"),
            "onset_log_dist": meta.get("onset_log_dist"), "notes": c.get("notes", ""),
        })
    return joined


def per_arm(joined: List[dict], metric: str, regime: Optional[str] = None) -> Dict[str, dict]:
    buckets: Dict[str, list] = defaultdict(list)
    for j in joined:
        if regime and j["regime"] != regime:
            continue
        if j.get(metric) is not None:
            buckets[j["arm"]].append(j[metric])
    return {arm: distribution(buckets.get(arm, [])) for arm in ARMS}


def discrimination_ok(continuity_by_arm: Dict[str, dict]) -> bool:
    decoy = continuity_by_arm.get("decoy", {}).get("p50")
    if decoy is None:
        return False
    reals = [continuity_by_arm.get(a, {}).get("p50") for a in ("narrow", "dynamic", "off")]
    reals = [r for r in reals if r is not None]
    return bool(reals) and all(decoy < r for r in reals)


def _mean(joined, arm, metric):
    vals = [j[metric] for j in joined if j["arm"] == arm and j.get(metric) is not None]
    return float(np.mean(vals)) if vals else None


def confound_flag(joined: List[dict]) -> dict:
    cn, cd = _mean(joined, "narrow", "continuity"), _mean(joined, "dynamic", "continuity")
    sn, sd = _mean(joined, "narrow", "smoothness"), _mean(joined, "dynamic", "smoothness")
    if None in (cn, cd, sn, sd):
        return {"pace_specific": None, "continuity_gain": None, "smoothness_gain": None}
    cg, sg = cn - cd, sn - sd
    return {"pace_specific": bool(cg > sg), "continuity_gain": cg, "smoothness_gain": sg}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_pace_audition_analyze.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/pace_audition_analyze.py tests/unit/test_pace_audition_analyze.py
git commit -m "feat(pace-audition): analyze distributions, contrasts, confound, discrimination"
```

---

## Task 8: Analyze — onset-variance check, provenance guard, findings writer + main

**Files:**
- Modify: `scripts/pace_audition_analyze.py`
- Test: `tests/unit/test_pace_audition_analyze.py`

- [ ] **Step 1: Write the failing test (append)**

```python
from scripts.pace_audition_analyze import onset_variance_by_arm


def test_onset_variance_by_arm_lower_for_flatter_playlist():
    playlists = [
        {"seed": "s", "arm": "narrow", "onset_seq": [2.0, 2.0, 2.1, 2.0]},
        {"seed": "s", "arm": "off", "onset_seq": [1.0, 6.0, 0.5, 8.0]},
    ]
    out = onset_variance_by_arm(playlists)
    assert out["narrow"] < out["off"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pace_audition_analyze.py::test_onset_variance_by_arm_lower_for_flatter_playlist -q`
Expected: FAIL — `ImportError: cannot import name 'onset_variance_by_arm'`

- [ ] **Step 3: Implement (append to `scripts/pace_audition_analyze.py`)**

```python
def onset_variance_by_arm(playlists: List[dict]) -> Dict[str, float]:
    """Mean within-playlist onset-rate variance per arm (structural monotony
    proxy). Lower = flatter pace profile = monotony risk."""
    buckets: Dict[str, list] = defaultdict(list)
    for pl in playlists:
        seq = [v for v in pl.get("onset_seq", []) if v is not None]
        if len(seq) >= 2:
            buckets[pl["arm"]].append(float(np.var(seq)))
    return {arm: float(np.mean(v)) for arm, v in buckets.items() if v}


def load_capture(data_dir: Path) -> List[dict]:
    cap = data_dir / "pace_capture.yaml"
    if not cap.exists():
        return []
    data = yaml.safe_load(cap.read_text(encoding="utf-8")) or {}
    return data.get("entries", [])


def _fmt(d: dict) -> str:
    if d["n"] == 0:
        return "—"
    return f"n={d['n']} min={d['min']:.1f} p10={d['p10']:.1f} p50={d['p50']:.1f} p90={d['p90']:.1f}"


def write_findings(data_dir: Path, joined: List[dict], playlists: List[dict]) -> Path:
    cont_all = per_arm(joined, "continuity")
    smooth_all = per_arm(joined, "smoothness")
    disc = discrimination_ok(cont_all)
    conf = confound_flag(joined)
    ovar = onset_variance_by_arm(playlists)

    lines = [
        "# Pace Audition — Findings", "",
        f"Total rated edges: {len([j for j in joined if j.get('continuity') is not None])}", "",
        "## Continuity by arm (1-5)", "",
        "| Arm | overall | ambient | rhythmic |", "|---|---|---|---|",
    ]
    for arm in ARMS:
        amb = per_arm(joined, "continuity", "ambient")[arm]
        rhy = per_arm(joined, "continuity", "rhythmic")[arm]
        lines.append(f"| {arm} | {_fmt(cont_all[arm])} | {_fmt(amb)} | {_fmt(rhy)} |")

    lines += ["", "## Smoothness by arm (1-5)", "", "| Arm | overall |", "|---|---|"]
    for arm in ARMS:
        lines.append(f"| {arm} | {_fmt(smooth_all[arm])} |")

    lines += [
        "", "## Verdict", "",
        f"- **Discrimination check** (decoy rated worst on continuity): "
        f"{'PASS' if disc else 'FAIL — ratings not trustworthy'}",
    ]
    if conf["pace_specific"] is not None:
        lines.append(
            f"- **Pace-specific** (narrow continuity gain {conf['continuity_gain']:+.2f} "
            f"vs smoothness gain {conf['smoothness_gain']:+.2f}): "
            f"{'YES — win is pace, not incidental' if conf['pace_specific'] else 'NO — gain may be incidental'}"
        )
    lines += ["", "## Structural monotony (onset variance per arm; lower=flatter)", ""]
    for arm in ARMS:
        if arm in ovar:
            lines.append(f"- {arm}: {ovar[arm]:.3f}")
    lines += [
        "", "## Notes", "",
        *[f"- [{j['arm']}] {j['notes']}" for j in joined if j.get("notes")],
        "", "## Caveats",
        "- Single listener, one library; directional not conclusive. N stated per cell above.",
        "- Onset-variance is a structural proxy for monotony, not a perceptual verdict.",
    ]
    out = data_dir / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="docs/run_audits/pace_audition")
    args = ap.parse_args()
    data_dir = ROOT / args.data_dir

    mpath = data_dir / "pace_manifest.json"
    if not mpath.exists():
        print(f"No manifest at {mpath}. Run the build first.")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    captures = load_capture(data_dir)
    if not captures:
        print("No captures yet. Complete the audition in the browser first.")
        return

    joined = join_scores(captures, manifest["edge_data"])
    out = write_findings(data_dir, joined, manifest.get("playlists", []))
    print(f"Wrote {out}  ({len(joined)} rated edges joined)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the new test (and full file) passes**

Run: `python -m pytest tests/unit/test_pace_audition_analyze.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/pace_audition_analyze.py tests/unit/test_pace_audition_analyze.py
git commit -m "feat(pace-audition): onset-variance check + findings writer + analyze main"
```

---

## Task 9: End-to-end smoke + full unit run

**Files:** none (verification only)

- [ ] **Step 1: Confirm output dir is gitignored (no audit artifacts get committed)**

Run: `git status --porcelain docs/run_audits/pace_audition/`
Expected: empty output (the dir is under the existing `docs/run_audits/` gitignore — confirmed by prior sessions where `findings.md` could not be added).

- [ ] **Step 2: Smoke the serve + page against the real manifest**

Run: `python scripts/pace_audition_serve.py --port 8766`
Then in the browser: confirm edges render as "Transition #N", "▶ Play transition" plays A-tail then hard-cuts to B-head, both sliders + notes auto-save (status shows `saved ✓`), and the top counter increments. Ctrl-C to stop.
Verify no arm leak: `curl -s http://127.0.0.1:8766/api/manifest | python -c "import sys,json; m=json.load(sys.stdin); assert set(m)=={'type','edges'}; assert set(m['edges'][0])=={'edge_id','a','b'}; print('blinded OK')"`
Expected: `blinded OK`.

- [ ] **Step 3: Rate a few edges, then run analyze**

Run: `python scripts/pace_audition_analyze.py`
Expected: `Wrote .../findings.md (N rated edges joined)`; open `findings.md` and confirm the continuity/smoothness tables, discrimination line, pace-specific line, and onset-variance section render.

- [ ] **Step 4: Full unit suite stays green (only the pre-existing UMAP failure)**

Run: `python -m pytest tests/unit -q -m "not slow" -p no:cacheprovider`
Expected: all pass except the pre-existing `test_calibrate_mert_transform.py::test_output_npz_has_all_transform_keys` (UMAP `eigsh`, unrelated).

- [ ] **Step 5: Final note**

The harness is complete. Running the full audition (rating all ~120 edges and producing the real `findings.md`) is the *use* of the harness, a separate analyst session — not part of this build. No final commit needed beyond Tasks 1-8 (verification only).

---

## Self-Review

**Spec coverage:**
- 4 arms (narrow/dynamic/off/decoy) → Task 4 (`REAL_ARMS` + decoy in Task 2). ✓
- Transition-level interior bridge edges → Task 1 `extract_interior_edges`. ✓
- generate_like_gui fidelity, fixed non-pace modes → Task 4 `FIXED_MODES`. ✓
- Decoy: pace-distant (>1.0) + genre ≥ median, 25th-pct fallback → Task 2. ✓
- Blinding, no-arm-leak → Task 3 + Task 5 `blinded_manifest` (tested both sides). ✓
- A-tail→hard-cut→B-head playback → Task 6 page. ✓
- Dual scoring (continuity + smoothness) → Task 5 capture + Task 6 sliders. ✓
- Distributions not means, regime split, discrimination, contrasts, confound → Task 7. ✓
- Structural onset-variance monotony proxy → Task 8. ✓
- Provenance (artifact path/mtime/fingerprint) → Task 4 manifest. ✓
- Output only under docs/run_audits/pace_audition/, read-only DB/artifact → Tasks 4/9. ✓
- N stated, caveats → Task 8 findings. ✓

**Placeholder scan:** No TBD/TODO/"handle errors" — every code step is complete and runnable.

**Type consistency:** manifest keys (`edges`/`edge_data`/`playlists`/`file_paths`/`provenance`) are identical across build (Task 4 writer), serve (Task 5 `blinded_manifest`/`file_paths`), and analyze (Task 8 `edge_data`/`playlists` readers). Edge record shape (`arm`/`seed`/`regime`/`a`/`b`/`onset_log_dist`/...) defined in Task 1-4 and consumed unchanged in Task 7 `join_scores`. Capture entry (`edge_id`/`continuity`/`smoothness`/`notes`) written in Task 5, read in Task 7-8. Function names (`genre_cosine`, `edge_metrics`, `extract_interior_edges`, `sample_edges`, `synthesize_decoy_edges`, `blind_and_shuffle`, `blinded_manifest`, `upsert_capture_entry`, `distribution`, `join_scores`, `per_arm`, `discrimination_ok`, `confound_flag`, `onset_variance_by_arm`) consistent between tests and implementations. ✓
