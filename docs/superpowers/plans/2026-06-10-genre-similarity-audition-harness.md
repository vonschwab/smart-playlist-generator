# Genre Similarity Audition Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local blind-audition harness that, for a set of seed genres, pools neighbors from the graph-derived and legacy co-occurrence similarity matrices plus random decoys, serves a browser rating UI, and aggregates relatedness verdicts sliced by provenance — to validate whether the graph matrix is better than co-occurrence and discriminative at all.

**Architecture:** Three file-communicating scripts mirroring the sonic audition harness, minus audio. `genre_audition_build.py` computes blinded per-seed JSON manifests; `genre_audition_serve.py` serves the page and captures verdicts to YAML; `genre_audition_analyze.py` reads captures and writes a findings report. Provenance (graph / cooccurrence / decoy), per-source rank, and similarity are hidden from the client and re-attached server-side at capture time.

**Tech Stack:** Python 3.11, NumPy, PyYAML, sqlite3 (stdlib), http.server (stdlib), vanilla JS/HTML. No new pip installs.

**Spec:** `docs/superpowers/specs/2026-06-09-genre-similarity-audition-design.md`

**Data locations:**
- Graph matrix (under test): `data/genre_similarity_graph.npz` — keys `genre_vocab` (408 canonical names), `S`, `stats`.
- Co-occurrence matrix (incumbent): `data/artifacts/beat3tower_32k/genre_similarity_matrix.npz` — keys `genre_vocab` (801 raw tokens incl. `__EMPTY__`), `cooc`, `S`, `stats`.
- Vocab bridge: `src/genre/graph_adapter.py` → `load_graph_adapter()`, `adapter.canonicalize_tag(token)`.
- Normalizer: `src.ai_genre_enrichment.layered_taxonomy.normalize_taxonomy_name`.
- Example artists (read-only): `data/metadata.db` — `track_genres(track_id, genre)`, `tracks(track_id, artist)`.
- Output manifests + captures: `docs/run_audits/genre_audition/` (gitignored; created at runtime).

**Shared constants** (defined in each script that needs them):
```python
VERDICTS = ["same", "related", "loose", "unrelated"]
VERDICT_SCORE = {"same": 3, "related": 2, "loose": 1, "unrelated": 0}
PROVENANCES = ["graph", "cooccurrence", "decoy"]
DEFAULT_SEEDS = [
    "rock", "electronic", "pop", "jazz",
    "indie rock", "house", "ambient", "post-punk",
    "slowcore", "shoegaze", "witch house", "drone",
]
```

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/genre_audition_build.py` | Pure compute (matrix load, top-K rows, cooc resolution, decoy sampling, card union) + manifest assembly + DB artist lookup + CLI |
| `scripts/genre_audition_page.html` | Static rating UI (served by the server) |
| `scripts/genre_audition_serve.py` | HTTP server: blinded manifest/progress APIs + YAML capture |
| `scripts/genre_audition_analyze.py` | Provenance-sliced aggregation → `findings.md` |
| `tests/unit/test_genre_audition_build.py` | Tests for compute + union + blinding |
| `tests/unit/test_genre_audition_serve.py` | Tests for capture append/update |
| `tests/unit/test_genre_audition_analyze.py` | Tests for provenance aggregation |

---

### Task 1: Build — pure compute functions

**Files:**
- Create: `scripts/genre_audition_build.py`
- Test: `tests/unit/test_genre_audition_build.py`

The compute layer takes plain vocab lists, NumPy matrices, and injected `canonicalize`/`normalize` callables so it is testable without loading any real artifact or DB. `canonicalize(token) -> Optional[str]` returns the canonical genre name or `None`; `normalize(name) -> str` lowercases/strips for dedupe keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_audition_build.py
import numpy as np
import pytest

from scripts.genre_audition_build import (
    top_k_row,
    resolve_cooc_index,
    sample_decoys,
    union_cards,
    build_seed_manifest,
    _slug,
)


def _norm(s):
    return s.strip().lower()


# canonicalize: clean tokens map to themselves; "funk / soul" is unmappable junk
_CANON = {"indie rock": "indie rock", "shoegaze": "shoegaze", "dream pop": "dream pop"}


def _canon(token):
    return _CANON.get(_norm(token))


def test_top_k_row_excludes_self_and_returns_sorted():
    vocab = ["a", "b", "c", "d"]
    S = np.array([
        [1.0, 0.9, 0.1, 0.5],
        [0.9, 1.0, 0.2, 0.3],
        [0.1, 0.2, 1.0, 0.4],
        [0.5, 0.3, 0.4, 1.0],
    ])
    out = top_k_row(S, vocab, seed_idx=0, k=2)
    assert out == [("b", 0.9), ("d", 0.5)]


def test_top_k_row_honors_exclude():
    vocab = ["a", "b", "c"]
    S = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.1], [0.8, 0.1, 1.0]])
    out = top_k_row(S, vocab, seed_idx=0, k=2, exclude={"b"})
    assert out == [("c", 0.8)]


def test_resolve_cooc_index_exact_match():
    cooc_vocab = ["rock", "indie rock", "jazz"]
    token, idx = resolve_cooc_index("indie rock", cooc_vocab, _canon)
    assert token == "indie rock" and idx == 1


def test_resolve_cooc_index_via_canonicalization():
    # canonical seed "dream pop" is absent verbatim but a token canonicalizes to it
    cooc_vocab = ["rock", "dreampop", "jazz"]
    canon = lambda t: "dream pop" if _norm(t) == "dreampop" else None
    token, idx = resolve_cooc_index("dream pop", cooc_vocab, canon)
    assert token == "dreampop" and idx == 1


def test_resolve_cooc_index_no_match():
    token, idx = resolve_cooc_index("nonexistent", ["rock", "jazz"], _canon)
    assert token is None and idx is None


def test_sample_decoys_from_low_tail_disjoint():
    vocab = ["seed", "near", "mid", "far1", "far2", "far3"]
    # seed row: high to near, ~0 to far*
    S = np.zeros((6, 6))
    np.fill_diagonal(S, 1.0)
    S[0, 1] = S[1, 0] = 0.8
    S[0, 2] = S[2, 0] = 0.4
    rng = np.random.default_rng(0)
    decoys = sample_decoys(S, vocab, seed_idx=0, n=2, exclude={"near"}, rng=rng)
    assert len(decoys) == 2
    assert "seed" not in decoys and "near" not in decoys
    assert all(d in {"far1", "far2", "far3", "mid"} for d in decoys)


def test_union_merges_provenances_on_canonical_key():
    graph_n = [("indie rock", 0.82), ("shoegaze", 0.5)]
    cooc_n = [("indie rock", 0.41), ("funk / soul", 0.3)]  # "funk / soul" -> None
    decoys = ["polka"]
    cards = union_cards(graph_n, cooc_n, decoys, _canon, _norm)
    # indie rock card carries BOTH graph and cooccurrence
    ir = next(c for c in cards if c["name"] == "indie rock")
    assert set(ir["spaces"]) == {"graph", "cooccurrence"}
    assert ir["spaces"]["graph"]["rank"] == 1
    assert ir["spaces"]["cooccurrence"]["rank"] == 1
    # unmappable cooc token stays its own raw card
    junk = next(c for c in cards if c["name"] == "funk / soul")
    assert set(junk["spaces"]) == {"cooccurrence"}
    # decoy card
    polka = next(c for c in cards if c["name"] == "polka")
    assert set(polka["spaces"]) == {"decoy"}


def test_build_seed_manifest_blinded_structure():
    graph_vocab = ["indie rock", "shoegaze", "dream pop", "polka", "techno"]
    cooc_vocab = ["indie rock", "dream pop", "funk / soul"]
    gS = np.eye(5)
    gS[0, 1] = gS[1, 0] = 0.7   # indie rock ~ shoegaze
    gS[0, 2] = gS[2, 0] = 0.6   # indie rock ~ dream pop
    cS = np.eye(3)
    cS[0, 1] = cS[1, 0] = 0.4
    cS[0, 2] = cS[2, 0] = 0.5

    def artist_fn(name):
        return ["ArtistX", "ArtistY"]

    m = build_seed_manifest(
        seed="indie rock",
        graph=(graph_vocab, gS),
        cooc=(cooc_vocab, cS),
        canonicalize=_canon,
        normalize=_norm,
        artist_fn=artist_fn,
        k=3,
        n_decoy=1,
        rng_seed=7,
    )
    assert m is not None
    assert m["slug"] == "indie_rock"
    assert m["seed"]["genre"] == "indie rock"
    assert m["seed"]["artists"] == ["ArtistX", "ArtistY"]
    # blinded: neighbors carry NO provenance/sim/rank
    for n in m["neighbors"]:
        assert set(n.keys()) == {"name", "artists"}
    # space_data lives separately, keyed by name
    assert "space_data" in m
    assert all("graph" in v or "cooccurrence" in v or "decoy" in v
               for v in m["space_data"].values())


def test_build_seed_manifest_unknown_seed_returns_none():
    m = build_seed_manifest(
        seed="not a genre",
        graph=(["indie rock"], np.eye(1)),
        cooc=(["indie rock"], np.eye(1)),
        canonicalize=_canon,
        normalize=_norm,
        artist_fn=lambda n: [],
        k=3,
        n_decoy=1,
        rng_seed=1,
    )
    assert m is None


def test_slug():
    assert _slug("indie rock") == "indie_rock"
    assert _slug("witch house") == "witch_house"
    assert _slug("funk / soul") == "funk_soul"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_audition_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.genre_audition_build'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/genre_audition_build.py
"""Build per-seed blind-audition manifests for genre-similarity validation.

For each seed genre, pulls top-K neighbors from the graph-derived matrix
(under test) and the legacy co-occurrence matrix (incumbent), adds random
decoys from the low-similarity tail (negative control), unions them into
cards (one card per genre, carrying every provenance that proposed it),
attaches example artists from the library, shuffles (blinded — provenance
hidden), and writes a JSON manifest.

Usage:
    python scripts/genre_audition_build.py
    python scripts/genre_audition_build.py --seeds "shoegaze" "rock"
    python scripts/genre_audition_build.py --top-k 10 --decoys 3

Output: docs/run_audits/genre_audition/<slug>_manifest.json per seed,
        docs/run_audits/genre_audition/index.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERDICTS = ["same", "related", "loose", "unrelated"]
PROVENANCES = ["graph", "cooccurrence", "decoy"]
DEFAULT_SEEDS = [
    "rock", "electronic", "pop", "jazz",
    "indie rock", "house", "ambient", "post-punk",
    "slowcore", "shoegaze", "witch house", "drone",
]

GRAPH_NPZ = "data/genre_similarity_graph.npz"
COOC_NPZ = "data/artifacts/beat3tower_32k/genre_similarity_matrix.npz"
COOC_SENTINEL = "__EMPTY__"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_matrix(path: str | Path) -> Tuple[List[str], np.ndarray]:
    """Load an {genre_vocab, S} similarity NPZ. Returns (vocab, S)."""
    data = np.load(path, allow_pickle=True)
    vocab = [str(v) for v in data["genre_vocab"]]
    S = np.asarray(data["S"], dtype=np.float64)
    return vocab, S


def top_k_row(
    S: np.ndarray,
    vocab: List[str],
    seed_idx: int,
    k: int,
    exclude: frozenset | set = frozenset(),
) -> List[Tuple[str, float]]:
    """Top-k neighbors of seed_idx by descending similarity, excluding self,
    the sentinel, and any name in `exclude`."""
    row = S[seed_idx]
    order = np.argsort(-row)
    out: List[Tuple[str, float]] = []
    excl = {e.strip().lower() for e in exclude}
    for j in order:
        j = int(j)
        if j == seed_idx:
            continue
        name = vocab[j]
        if name == COOC_SENTINEL or name.strip().lower() in excl:
            continue
        out.append((name, round(float(row[j]), 4)))
        if len(out) >= k:
            break
    return out


def resolve_cooc_index(
    seed_canonical: str,
    cooc_vocab: List[str],
    canonicalize: Callable[[str], Optional[str]],
) -> Tuple[Optional[str], Optional[int]]:
    """Find the co-occurrence row for a canonical seed name.

    Exact (case-insensitive) match first; else the co-occurrence token that
    canonicalizes to this seed with the most library mass — approximated here
    by the first such token (cooc_vocab is frequency-ordered)."""
    target = seed_canonical.strip().lower()
    for i, tok in enumerate(cooc_vocab):
        if tok.strip().lower() == target:
            return tok, i
    for i, tok in enumerate(cooc_vocab):
        canon = canonicalize(tok)
        if canon is not None and canon.strip().lower() == target:
            return tok, i
    return None, None


def sample_decoys(
    S: np.ndarray,
    vocab: List[str],
    seed_idx: int,
    n: int,
    exclude: set,
    rng: np.random.Generator,
) -> List[str]:
    """Sample n decoys from the low-similarity half of the seed's row,
    excluding self and `exclude` names. Deterministic given rng."""
    row = S[seed_idx]
    order = np.argsort(row)  # ascending similarity (lowest first)
    excl = {e.strip().lower() for e in exclude}
    pool: List[str] = []
    half = max(1, len(order) // 2)
    for j in order[:half]:
        j = int(j)
        if j == seed_idx:
            continue
        name = vocab[j]
        if name == COOC_SENTINEL or name.strip().lower() in excl:
            continue
        pool.append(name)
    if not pool:
        return []
    rng.shuffle(pool)
    return pool[:n]


def union_cards(
    graph_n: List[Tuple[str, float]],
    cooc_n: List[Tuple[str, float]],
    decoys: List[str],
    canonicalize: Callable[[str], Optional[str]],
    normalize: Callable[[str], str],
) -> List[dict]:
    """Merge the three provenance lists into cards keyed by a canonical name.

    A genre proposed by several sources becomes one card carrying every
    provenance (with per-source rank+sim). Co-occurrence tokens that
    canonicalize to a known genre adopt the clean canonical display name and
    dedupe against graph/decoy cards; unmappable tokens keep their raw display.
    Returns a list of {name, spaces} dicts in stable insertion order."""
    cards: Dict[str, dict] = {}

    def _card(key: str, display: str) -> dict:
        if key not in cards:
            cards[key] = {"name": display, "spaces": {}}
        return cards[key]

    for rank, (name, sim) in enumerate(graph_n, start=1):
        c = _card(normalize(name), name)
        c["spaces"]["graph"] = {"rank": rank, "sim": sim}

    for rank, (tok, sim) in enumerate(cooc_n, start=1):
        canon = canonicalize(tok)
        if canon is not None:
            key, display = normalize(canon), canon
        else:
            key, display = "raw:" + normalize(tok), tok
        c = _card(key, display)
        c["spaces"]["cooccurrence"] = {"rank": rank, "sim": sim}

    for name in decoys:
        c = _card(normalize(name), name)
        c["spaces"].setdefault("decoy", {})

    return list(cards.values())


def build_seed_manifest(
    seed: str,
    graph: Tuple[List[str], np.ndarray],
    cooc: Tuple[List[str], np.ndarray],
    canonicalize: Callable[[str], Optional[str]],
    normalize: Callable[[str], str],
    artist_fn: Callable[[str], List[str]],
    k: int = 10,
    n_decoy: int = 3,
    rng_seed: Optional[int] = None,
) -> Optional[dict]:
    """Build the blinded manifest for one seed genre. Returns None if the seed
    is not in the graph vocabulary."""
    graph_vocab, gS = graph
    cooc_vocab, cS = cooc
    gindex = {v.strip().lower(): i for i, v in enumerate(graph_vocab)}
    seed_idx = gindex.get(seed.strip().lower())
    if seed_idx is None:
        return None

    graph_n = top_k_row(gS, graph_vocab, seed_idx, k, exclude={seed})

    cooc_token, cooc_idx = resolve_cooc_index(seed, cooc_vocab, canonicalize)
    if cooc_idx is not None:
        cooc_n = top_k_row(cS, cooc_vocab, cooc_idx, k, exclude={seed})
    else:
        cooc_n = []

    # Exclude everything already proposed (by canonical name) from the decoy pool.
    exclude = {seed}
    exclude.update(name for name, _ in graph_n)
    for tok, _ in cooc_n:
        canon = canonicalize(tok)
        exclude.add(canon if canon is not None else tok)
    rng = np.random.default_rng(rng_seed if rng_seed is not None else abs(hash(seed)) % (2**32))
    decoys = sample_decoys(gS, graph_vocab, seed_idx, n_decoy, exclude, rng)

    cards = union_cards(graph_n, cooc_n, decoys, canonicalize, normalize)

    # Blind shuffle (deterministic per seed)
    srng = np.random.default_rng((rng_seed or 0) ^ (abs(hash(seed)) % (2**32)))
    srng.shuffle(cards)

    neighbors = []
    space_data = {}
    for c in cards:
        neighbors.append({"name": c["name"], "artists": artist_fn(c["name"])})
        space_data[c["name"]] = c["spaces"]

    return {
        "slug": _slug(seed),
        "seed": {"genre": seed, "artists": artist_fn(seed)},
        "cooc_token": cooc_token,
        "neighbors": neighbors,
        "space_data": space_data,
    }


def representative_artists(con: sqlite3.Connection, genre: str, k: int = 3) -> List[str]:
    """Top-k library artists for a genre token, by track count. Read-only."""
    rows = con.execute(
        """
        SELECT t.artist, COUNT(*) AS c
        FROM track_genres tg JOIN tracks t ON t.track_id = tg.track_id
        WHERE tg.genre = ? AND t.artist IS NOT NULL AND t.artist != ''
        GROUP BY t.artist ORDER BY c DESC LIMIT ?
        """,
        (genre, k),
    ).fetchall()
    return [str(r[0]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--decoys", type=int, default=3)
    ap.add_argument("--graph", default=GRAPH_NPZ)
    ap.add_argument("--cooc", default=COOC_NPZ)
    ap.add_argument("--db", default="data/metadata.db")
    ap.add_argument("--out-dir", default="docs/run_audits/genre_audition")
    args = ap.parse_args()

    from src.genre.graph_adapter import load_graph_adapter

    adapter = load_graph_adapter()

    def canonicalize(token: str) -> Optional[str]:
        r = adapter.canonicalize_tag(token)
        return r.canonical if r.resolution in ("canonical", "alias") else None

    from src.ai_genre_enrichment.layered_taxonomy import normalize_taxonomy_name

    graph = load_matrix(ROOT / args.graph)
    cooc = load_matrix(ROOT / args.cooc)

    con = sqlite3.connect(f"file:{ROOT / args.db}?mode=ro", uri=True)
    artist_cache: Dict[str, List[str]] = {}

    def artist_fn(name: str) -> List[str]:
        if name not in artist_cache:
            artist_cache[name] = representative_artists(con, name)
        return artist_cache[name]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for seed in args.seeds:
        m = build_seed_manifest(
            seed, graph, cooc, canonicalize, normalize_taxonomy_name,
            artist_fn, k=args.top_k, n_decoy=args.decoys, rng_seed=1234,
        )
        if m is None:
            print(f"  SKIP {seed!r} (not in graph vocabulary)")
            continue
        (out_dir / f"{m['slug']}_manifest.json").write_text(
            json.dumps(m, indent=2), encoding="utf-8"
        )
        index.append({"slug": m["slug"], "genre": seed})
        cooc_note = "" if m["cooc_token"] else " [no cooc row]"
        print(f"  OK   {seed!r} -> {m['slug']}_manifest.json "
              f"({len(m['neighbors'])} cards){cooc_note}")

    con.close()
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nDone. {len(index)} manifests in {out_dir}")
    print("Next: python scripts/genre_audition_serve.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_audition_build.py -v`
Expected: 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/genre_audition_build.py tests/unit/test_genre_audition_build.py
git commit -m "feat(genre-audition): manifest builder — graph+cooc+decoy pooling, blinded union"
```

---

### Task 2: Audition page HTML

**Files:**
- Create: `scripts/genre_audition_page.html`

No pytest test — static file served by the server, visually verified in Task 4 / Task 5.

- [ ] **Step 1: Create the page**

```html
<!-- scripts/genre_audition_page.html -->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Genre Similarity Audition</title>
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
.seed-section .genre-name { font-size: 20px; color: #fff; margin-bottom: 6px; }
.artists { font-size: 12px; color: #888; }
.main { max-width: 760px; margin: 0 auto; padding: 16px; }
.card { background: #222; border: 1px solid #333; border-left: 4px solid #333; border-radius: 4px; padding: 12px; margin-bottom: 10px; }
.card.same { border-left-color: #4caf50; }
.card.related { border-left-color: #8bc34a; }
.card.loose { border-left-color: #ff9800; }
.card.unrelated { border-left-color: #f44336; }
.card-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
.card-header .name { font-size: 15px; color: #fff; }
.card-header .num { color: #555; font-size: 11px; margin-right: 6px; }
.status { font-size: 11px; color: #4caf50; min-width: 60px; text-align: right; }
.status.err { color: #f44336; }
.verdicts { display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap; }
.vbtn { display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 12px; color: #bbb; padding: 3px 9px; border: 1px solid #444; border-radius: 3px; }
.vbtn:hover { background: #2a2a2a; }
.vbtn input { accent-color: #4caf50; }
textarea { width: 100%; background: #1a1a1a; color: #ddd; border: 1px solid #444; border-radius: 3px; padding: 6px; font-family: inherit; font-size: 12px; resize: vertical; min-height: 40px; margin-top: 4px; }
textarea:focus { outline: none; border-color: #666; }
</style>
</head>
<body>

<div class="top-bar">
  <label>Seed: <select id="seed-select" onchange="location='/seed/'+this.value"></select></label>
  <div class="nav" id="nav"></div>
</div>
<div class="progress-bar" id="progress-bar">Loading…</div>
<div class="seed-section">
  <h2>Seed Genre — rate each candidate's relatedness to this</h2>
  <div class="genre-name" id="seed-name">—</div>
  <div class="artists" id="seed-artists"></div>
</div>
<div class="main" id="main"></div>

<script>
const SLUG = location.pathname.split('/').pop();
const ALL_SEEDS = SEED_LIST_PLACEHOLDER;
const VERDICTS = ["same", "related", "loose", "unrelated"];
let manifest = null, progress = {};

function artistLine(arr) {
  return (arr && arr.length) ? arr.join(', ') : '(no example artists)';
}

async function init() {
  const sel = document.getElementById('seed-select');
  ALL_SEEDS.forEach(s => {
    const o = document.createElement('option');
    o.value = s.slug; o.textContent = s.genre || s.slug; o.selected = (s.slug === SLUG);
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
  (await pR.json()).forEach(e => { progress[e.name] = e; });

  document.getElementById('seed-name').textContent = manifest.seed.genre;
  document.getElementById('seed-artists').textContent = artistLine(manifest.seed.artists);
  const main = document.getElementById('main');
  manifest.neighbors.forEach((n, i) => main.appendChild(buildCard(n, i, progress[n.name] || {})));
  updateProgress();
}

function buildCard(n, i, saved) {
  const div = document.createElement('div');
  div.className = 'card' + (saved.verdict ? ` ${saved.verdict}` : '');
  div.id = `card-${i}`;
  div.dataset.name = n.name;
  div.innerHTML = `
    <div class="card-header">
      <span class="name"><span class="num">#${i+1}</span>${n.name}</span>
      <span class="status" id="st-${i}"></span>
    </div>
    <div class="artists">${artistLine(n.artists)}</div>
    <div class="verdicts">
      ${VERDICTS.map(v => `
        <label class="vbtn"><input type="radio" name="v-${i}" value="${v}"
          ${saved.verdict===v?'checked':''} onchange="saveN(${i})"> ${v}</label>
      `).join('')}
    </div>
    <textarea id="nt-${i}" placeholder="Notes…"
      onblur="saveN(${i})">${saved.notes||''}</textarea>`;
  return div;
}

async function saveN(i) {
  const name = document.getElementById(`card-${i}`).dataset.name;
  const v = document.querySelector(`input[name="v-${i}"]:checked`)?.value || '';
  const notes = document.getElementById(`nt-${i}`).value;
  const st = document.getElementById(`st-${i}`);
  try {
    await post({ seed: SLUG, name, verdict: v, notes });
    st.textContent = 'saved ✓'; st.className = 'status';
    progress[name] = { name, verdict: v, notes };
    document.getElementById(`card-${i}`).className = 'card' + (v ? ` ${v}` : '');
    updateProgress();
    setTimeout(() => { st.textContent = ''; }, 3000);
  } catch (e) { st.textContent = 'error'; st.className = 'status err'; }
}

async function post(body) {
  const r = await fetch('/api/save', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error('HTTP ' + r.status);
}

function updateProgress() {
  if (!manifest) return;
  const total = manifest.neighbors.length;
  const done = Object.values(progress).filter(e => e.verdict).length;
  document.getElementById('progress-bar').innerHTML = `<span>${done}</span> / ${total} reviewed`;
}

init();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the file was written**

Run: `python -c "from pathlib import Path; p=Path('scripts/genre_audition_page.html'); print(p.stat().st_size, 'bytes')"`
Expected: prints a positive byte count.

- [ ] **Step 3: Commit**

```bash
git add scripts/genre_audition_page.html
git commit -m "feat(genre-audition): rating page — blinded candidate cards, verdict radio, auto-save"
```

---

### Task 3: HTTP server + capture

**Files:**
- Create: `scripts/genre_audition_serve.py`
- Test: `tests/unit/test_genre_audition_serve.py`

The server loads manifests, serves the page and a **blinded** manifest API (strips `space_data` and `cooc_token`), exposes progress, and appends verdicts to per-seed capture YAMLs with provenance re-attached server-side.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_audition_serve.py
import yaml

from scripts.genre_audition_serve import _append_capture_entry, _blind_manifest


def test_append_creates_file(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"name": "shoegaze", "verdict": "same", "notes": "x"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["name"] == "shoegaze"


def test_append_updates_existing(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"name": "shoegaze", "verdict": "same", "notes": "a"})
    _append_capture_entry(p, {"name": "shoegaze", "verdict": "loose", "notes": "b"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["verdict"] == "loose"


def test_append_adds_new(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"name": "a", "verdict": "same", "notes": ""})
    _append_capture_entry(p, {"name": "b", "verdict": "loose", "notes": ""})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 2


def test_blind_manifest_strips_hidden_fields():
    m = {
        "slug": "x", "seed": {"genre": "x", "artists": []},
        "cooc_token": "x raw", "neighbors": [{"name": "n", "artists": []}],
        "space_data": {"n": {"graph": {"rank": 1, "sim": 0.5}}},
    }
    blinded = _blind_manifest(m)
    assert "space_data" not in blinded
    assert "cooc_token" not in blinded
    assert blinded["neighbors"] == [{"name": "n", "artists": []}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_audition_serve.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# scripts/genre_audition_serve.py
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
from typing import Dict, List
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
        body = json.loads(self.rfile.read(length))
        seed = body.get("seed", "")
        name = body.get("name", "")
        if not seed or not name:
            self._json({"ok": False, "error": "missing seed or name"}, 400)
            return
        m = self.server.manifests.get(seed, {})
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_audition_serve.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Run the fast suite to confirm no regressions**

Run: `pytest -m "not slow" -q`
Expected: all pass (same count as before this plan, plus the new genre-audition tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/genre_audition_serve.py tests/unit/test_genre_audition_serve.py
git commit -m "feat(genre-audition): HTTP server — blinded manifest API, YAML verdict capture"
```

---

### Task 4: Analysis script

**Files:**
- Create: `scripts/genre_audition_analyze.py`
- Test: `tests/unit/test_genre_audition_analyze.py`

Reads all `*_capture.yaml`, expands each rated entry by provenance, and writes `findings.md` with verdict distribution, mean score per provenance, the graph-vs-cooccurrence and graph-vs-decoy contrasts, sim↔verdict correlation, and callout lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_audition_analyze.py
from scripts.genre_audition_analyze import (
    aggregate_by_provenance,
    mean_score_by_provenance,
    sim_verdict_rows,
)


def _entry(verdict, spaces):
    return {"verdict": verdict, "notes": "", "spaces": spaces}


def test_aggregate_expands_by_provenance():
    entries = [
        _entry("same", {"graph": {"rank": 1, "sim": 0.8}, "cooccurrence": {"rank": 2, "sim": 0.4}}),
        _entry("unrelated", {"decoy": {}}),
    ]
    agg = aggregate_by_provenance(entries)
    assert agg["graph"]["same"] == 1
    assert agg["cooccurrence"]["same"] == 1   # same entry counts for both sources
    assert agg["decoy"]["unrelated"] == 1


def test_aggregate_skips_empty_verdict():
    agg = aggregate_by_provenance([_entry("", {"graph": {"rank": 1, "sim": 0.5}})])
    assert agg.get("graph", {}) == {}


def test_mean_score_by_provenance():
    entries = [
        _entry("same", {"graph": {"rank": 1, "sim": 0.8}}),       # 3
        _entry("loose", {"graph": {"rank": 2, "sim": 0.3}}),      # 1
        _entry("unrelated", {"decoy": {}}),                       # 0
    ]
    means = mean_score_by_provenance(entries)
    assert means["graph"] == 2.0
    assert means["decoy"] == 0.0


def test_sim_verdict_rows_skips_decoy_and_missing_sim():
    entries = [
        _entry("same", {"graph": {"rank": 1, "sim": 0.8}}),
        _entry("unrelated", {"decoy": {}}),
    ]
    rows = sim_verdict_rows(entries)
    assert rows == [{"provenance": "graph", "sim": 0.8, "score": 3}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_audition_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# scripts/genre_audition_analyze.py
"""Aggregate genre-audition captures and write findings.

Reads all *_capture.yaml in the data directory, expands each rated entry by
provenance (an entry proposed by both graph and co-occurrence counts for
both), and writes findings.md: verdict distribution and mean score per
provenance, graph-vs-cooccurrence and graph-vs-decoy contrasts, sim-vs-verdict
correlation, and callout lists for bad graph edges and missed co-occurrence
neighbors.

Usage:
    python scripts/genre_audition_analyze.py [--data-dir docs/run_audits/genre_audition]
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

VERDICT_ORDER = ["same", "related", "loose", "unrelated"]
VERDICT_SCORE = {"same": 3, "related": 2, "loose": 1, "unrelated": 0}
PROVENANCES = ["graph", "cooccurrence", "decoy"]


def load_captures(data_dir: Path) -> List[dict]:
    """Return all entries from every *_capture.yaml, tagged with their seed."""
    entries = []
    for p in sorted(data_dir.glob("*_capture.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        seed = p.stem.replace("_capture", "")
        for e in data.get("entries", []):
            e.setdefault("seed", seed)
            entries.append(e)
    return entries


def aggregate_by_provenance(entries: List[dict]) -> Dict[str, Dict[str, int]]:
    """{provenance: {verdict: count}} over rated entries, expanded by provenance."""
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in entries:
        verdict = e.get("verdict", "")
        if not verdict:
            continue
        for prov in (e.get("spaces") or {}):
            result[prov][verdict] += 1
    return {k: dict(v) for k, v in result.items()}


def mean_score_by_provenance(entries: List[dict]) -> Dict[str, float]:
    """{provenance: mean verdict score} over rated entries, expanded by provenance."""
    totals: Dict[str, list] = defaultdict(list)
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict not in VERDICT_SCORE:
            continue
        score = VERDICT_SCORE[verdict]
        for prov in (e.get("spaces") or {}):
            totals[prov].append(score)
    return {k: round(float(np.mean(v)), 3) for k, v in totals.items() if v}


def sim_verdict_rows(entries: List[dict]) -> List[dict]:
    """[{provenance, sim, score}] for graph/cooccurrence entries with a sim."""
    rows = []
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict not in VERDICT_SCORE:
            continue
        score = VERDICT_SCORE[verdict]
        for prov, meta in (e.get("spaces") or {}).items():
            if meta and "sim" in meta:
                rows.append({"provenance": prov, "sim": meta["sim"], "score": score})
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="docs/run_audits/genre_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    entries = load_captures(data_dir)
    if not entries:
        print(f"No capture entries in {data_dir}. Complete the audition first.")
        return

    rated = [e for e in entries if e.get("verdict")]
    by_prov = aggregate_by_provenance(entries)
    means = mean_score_by_provenance(entries)
    corr_rows = sim_verdict_rows(entries)

    lines = [
        "# Genre Similarity Audition — Findings",
        "",
        f"Total entries: {len(entries)}  |  Rated: {len(rated)}",
        "",
        "## Verdict Distribution by Provenance",
        "",
        "| Provenance | same | related | loose | unrelated | total | mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for prov in PROVENANCES:
        counts = by_prov.get(prov, {})
        total = sum(counts.values())
        if total == 0:
            continue
        row = [prov] + [str(counts.get(v, 0)) for v in VERDICT_ORDER]
        row += [str(total), f"{means.get(prov, float('nan')):.3f}"]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Headline Contrasts", ""]
    g, c, d = means.get("graph"), means.get("cooccurrence"), means.get("decoy")
    if g is not None and c is not None:
        verdict = "graph WINS" if g > c else ("co-occurrence WINS" if c > g else "TIE")
        lines.append(f"- **Graph vs co-occurrence (Q1):** graph={g:.3f} vs cooc={c:.3f} → **{verdict}** (Δ={g-c:+.3f})")
    if g is not None and d is not None:
        lines.append(f"- **Graph vs decoy (Q2):** graph={g:.3f} vs decoy={d:.3f} → gap={g-d:+.3f} "
                     f"({'discriminative' if g - d > 0.5 else 'WEAK — investigate'})")

    lines += ["", "## Similarity ↔ Verdict Correlation (Pearson r)", ""]
    groups: Dict[str, list] = defaultdict(list)
    for r in corr_rows:
        groups[r["provenance"]].append((r["sim"], r["score"]))
    for prov in ("graph", "cooccurrence"):
        pairs = groups.get(prov, [])
        if len(pairs) < 3:
            lines.append(f"- **{prov}**: too few rated pairs ({len(pairs)})")
            continue
        sims = np.array([p[0] for p in pairs])
        scores = np.array([p[1] for p in pairs])
        if sims.std() == 0 or scores.std() == 0:
            lines.append(f"- **{prov}**: r undefined (no variance), {len(pairs)} pairs")
            continue
        r_val = float(np.corrcoef(sims, scores)[0, 1])
        lines.append(f"- **{prov}**: r={r_val:.3f} ({len(pairs)} pairs, "
                     f"sim [{sims.min():.3f}, {sims.max():.3f}])")

    lines += ["", "## Graph Neighbors Rated `unrelated` (candidate bad edges → SP3a)", ""]
    bad = [e for e in rated if e.get("verdict") == "unrelated" and "graph" in (e.get("spaces") or {})]
    if bad:
        for e in bad:
            lines.append(f"- **{e.get('seed','')}** → `{e['name']}`"
                         + (f" — {e['notes']}" if e.get("notes") else ""))
    else:
        lines.append("*(none)*")

    lines += ["", "## Co-occurrence-only Neighbors Rated `same`/`related` (candidate gaps)", ""]
    gaps = [
        e for e in rated
        if e.get("verdict") in ("same", "related")
        and "cooccurrence" in (e.get("spaces") or {})
        and "graph" not in (e.get("spaces") or {})
    ]
    if gaps:
        for e in gaps:
            lines.append(f"- **{e.get('seed','')}** → `{e['name']}` ({e['verdict']})"
                         + (f" — {e['notes']}" if e.get("notes") else ""))
    else:
        lines.append("*(none)*")

    lines += ["", "## Notable Notes", ""]
    for e in sorted(rated, key=lambda x: VERDICT_SCORE.get(x.get("verdict", ""), 0)):
        if e.get("notes"):
            lines.append(f"- **{e.get('seed','')}** → {e['name']} | {e.get('verdict','')} | {e['notes']}")

    out = data_dir / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print("Means by provenance:", means)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_audition_analyze.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Run the fast suite**

Run: `pytest -m "not slow" -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/genre_audition_analyze.py tests/unit/test_genre_audition_analyze.py
git commit -m "feat(genre-audition): analysis — provenance-sliced verdicts, graph-vs-cooc/decoy contrasts"
```

---

### Task 5: End-to-end smoke run

Confirm the three scripts work together on real data.

- [ ] **Step 1: Build a 3-seed subset**

Run:
```bash
python scripts/genre_audition_build.py --seeds "shoegaze" "rock" "jazz" --top-k 8 --decoys 3
```
Expected: three `OK` lines (with card counts) and `index.json` written; no traceback. If any seed prints `SKIP`, note it — the canonical name is absent from the graph vocab and the seed list should be adjusted.

- [ ] **Step 2: Verify manifest is blinded and well-formed**

Run:
```bash
python -c "
import json
from pathlib import Path
m = json.loads(Path('docs/run_audits/genre_audition/shoegaze_manifest.json').read_text())
print('slug:', m['slug'], '| seed:', m['seed']['genre'], '| seed artists:', m['seed']['artists'][:3])
print('cooc_token:', m['cooc_token'])
print('cards:', len(m['neighbors']))
n = m['neighbors'][0]
print('neighbor keys:', sorted(n.keys()))
print('provenance hidden from neighbors:', set(n.keys()) == {'name', 'artists'})
print('space_data present:', 'space_data' in m)
provs = set()
for v in m['space_data'].values():
    provs |= set(v.keys())
print('provenances seen:', provs)
"
```
Expected: `provenance hidden from neighbors: True`, `space_data present: True`, and `provenances seen:` includes at least `graph` (and normally `cooccurrence` and `decoy`).

- [ ] **Step 3: Start the server and confirm the blinded API**

Start in the background:
```bash
python scripts/genre_audition_serve.py --port 8766
```
Then in another shell:
```bash
python -c "
import urllib.request, json
d = json.loads(urllib.request.urlopen('http://127.0.0.1:8766/api/manifest/shoegaze').read())
print('space_data absent:', 'space_data' not in d)
print('cooc_token absent:', 'cooc_token' not in d)
print('cards:', len(d['neighbors']))
"
```
Expected: `space_data absent: True`, `cooc_token absent: True`. Visually confirm the page renders seed + cards with verdict radios in the opened browser tab, rate one card, and check the status shows `saved ✓`. Stop the server with Ctrl+C.

- [ ] **Step 4: Confirm capture + analysis round-trip**

After rating at least one card in Step 3:
```bash
python scripts/genre_audition_analyze.py
```
Expected: prints `Wrote .../findings.md` and `Means by provenance: {...}`. Open `docs/run_audits/genre_audition/findings.md` and confirm the verdict-distribution table and headline contrasts render.

- [ ] **Step 5: Build the full 12-seed set**

Run:
```bash
python scripts/genre_audition_build.py
```
Expected: up to 12 `OK`/`SKIP` lines + `index.json`. The harness is ready for a full rating pass.

- [ ] **Step 6: Commit the milestone**

```bash
git commit --allow-empty -m "feat(genre-audition): harness complete — build/serve/analyze verified end-to-end"
```
(The `docs/run_audits/` dir is gitignored; this empty commit just marks the milestone.)

---

## Self-Review

**Spec coverage:**
- Single blind rating pass pooling graph + cooccurrence + decoy → Task 1 (`build_seed_manifest`, `union_cards`) ✅
- Vocab bridge via `canonicalize_tag` for seed/cooc resolution → Task 1 (`resolve_cooc_index`, injected `canonicalize`) ✅
- Co-occurrence shown raw when unmappable, deduped+cleaned when canonicalizable → Task 1 (`union_cards`) ✅
- Decoys from low-similarity tail, disjoint → Task 1 (`sample_decoys`) ✅
- Example artists read-only from metadata.db → Task 1 (`representative_artists`) ✅
- Blinded manifest (provenance/sim/rank/cooc_token hidden) → Task 1 (`space_data` separate) + Task 3 (`_blind_manifest`) ✅
- Rating page, 4-way verdict, auto-save, no audio → Task 2 ✅
- Capture YAML append/update keyed by name, provenance re-attached server-side → Task 3 (`_append_capture_entry`, `do_POST`) ✅
- Analysis: distribution + mean per provenance, Q1/Q2 contrasts, sim↔verdict correlation, bad-edge + gap callouts → Task 4 ✅
- Stratified 12 seeds, `--seeds` override, SKIP on miss → Task 1 (`DEFAULT_SEEDS`, `main`) + Task 5 ✅
- Edge cases (no cooc row, no artists, tiny decoy tail, sentinel) → Task 1 (`build_seed_manifest`, `sample_decoys`, `top_k_row` sentinel filter) ✅

**Placeholder scan:** No TBD/TODO; every code step contains full code; every command has expected output.

**Type consistency:** `top_k_row(S, vocab, seed_idx, k, exclude)` → `list[(name, sim)]` used consistently in Task 1. `union_cards(...)` → `list[{name, spaces}]`. `build_seed_manifest(...)` returns manifest with `neighbors=[{name, artists}]` + `space_data={name: {prov: {rank, sim}}}` — matched by Task 3 (`_blind_manifest`, `do_POST` reads `space_data[name]`) and Task 4 (`aggregate_by_provenance`/`mean_score_by_provenance`/`sim_verdict_rows` read `e["spaces"][prov]["sim"]`). Capture entry keyed by `name` consistently across Task 3 and Task 4. Verdict vocabulary (`same/related/loose/unrelated`) and `VERDICT_SCORE` identical across page (Task 2), build (Task 1), and analyze (Task 4).

**Note for implementer:** `representative_artists` matches `track_genres.genre` exactly. Canonical graph/decoy names that are also raw library tokens (the common case) resolve; canonical-only names with no matching raw token return `[]` and render "(no example artists)" — acceptable and honest. Do not add a reverse canonical→token expansion unless a full run shows too many empty artist lists.
