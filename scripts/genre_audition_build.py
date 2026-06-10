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
