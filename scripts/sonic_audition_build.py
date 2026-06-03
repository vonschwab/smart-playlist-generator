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


def _zscore(M: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Z-score each column using the mean/std of `rows` only (the valid pool)."""
    M = M.astype(np.float64)
    sub = M[rows]
    mu, sd = sub.mean(axis=0), sub.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (M - mu) / sd


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
    Xf = _l2(bundle.X_sonic.astype(np.float64))

    Xe = bundle.X_sonic_end.astype(np.float64)
    Xs = bundle.X_sonic_start.astype(np.float64)
    Xe_c = _l2(Xe - Xe.mean(axis=0, keepdims=True))
    Xs_c = _l2(Xs - Xs.mean(axis=0, keepdims=True))

    Xt = _l2(per_tower["X_sonic_timbre"].astype(np.float64))
    Xh = _l2(per_tower["X_sonic_harmony"].astype(np.float64))

    return {
        "full_track": (Xf, Xf),
        "production_transition": (Xe_c, Xs_c),
        "timbre": (Xt, Xt),
        "harmony": (Xh, Xh),
    }


def load_2dftm_sidecar(
    sidecar_path: str, bundle
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the harmony 2DFTM sidecar, aligned to bundle.track_ids row order.

    Returns (X_2dftm: (N, D) float64 with zero rows for tracks not yet extracted,
    valid_mask: (N,) bool marking which rows actually have features).
    """
    z = np.load(sidecar_path, allow_pickle=True)
    feat_by_tid = {str(t): z["features"][i] for i, t in enumerate(z["track_ids"])}
    dim = z["features"].shape[1]
    N = len(bundle.track_ids)
    X = np.zeros((N, dim), dtype=np.float64)
    valid = np.zeros(N, dtype=bool)
    for i, tid in enumerate(bundle.track_ids):
        v = feat_by_tid.get(str(tid))
        if v is not None:
            X[i] = v
            valid[i] = True
    return X, valid


def compute_headtohead_spaces(
    bundle, per_tower: Dict[str, np.ndarray], X_2dftm: np.ndarray, valid_mask: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Two blinded harmony spaces for the legacy-vs-2DFTM A/B.

    harmony_legacy = the shipped chroma-median tower; harmony_2dftm = key-invariant
    2DFTM, z-scored over the valid pool then L2 (matching the probe pipeline).
    """
    Xleg = _l2(per_tower["X_sonic_harmony"].astype(np.float64))
    rows = np.where(valid_mask)[0]
    X2 = _l2(_zscore(X_2dftm, rows))
    return {
        "harmony_legacy": (Xleg, Xleg),
        "harmony_2dftm": (X2, X2),
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
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    """Return {space: [(track_idx, cosine), ...]} top-k per space, excluding exclude_indices.

    If valid_mask is given, the candidate pool is restricted to those rows for ALL
    spaces — so head-to-head spaces compete over an identical candidate set.
    """
    N = next(iter(spaces.values()))[0].shape[0]
    exclude = set(exclude_indices) | {seed_idx}
    mask = np.ones(N, dtype=bool)
    if valid_mask is not None:
        mask &= valid_mask
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
    valid_mask: Optional[np.ndarray] = None,
    medoid_vectors: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Build the blinded manifest for one seed artist. Returns None if artist not found.

    medoid_vectors picks the representative seed track (defaults to the full_track
    space); valid_mask restricts the neighbor candidate pool (head-to-head mode).
    """
    artists = np.array([str(a) for a in bundle.track_artists])
    artist_idx = np.where(np.char.lower(artists) == artist.lower())[0]
    if len(artist_idx) == 0:
        return None

    Xq_full = medoid_vectors if medoid_vectors is not None else spaces["full_track"][0]
    seed_idx = find_medoid(Xq_full, artist_idx)
    seed_tid = str(bundle.track_ids[seed_idx])
    seed_title = str(bundle.track_titles[seed_idx]) if bundle.track_titles is not None else "?"

    exclude = {int(i) for i in artist_idx}
    per_space = top_k_for_seed(seed_idx, spaces, exclude, k, valid_mask=valid_mask)

    seen: Dict[int, Dict[str, dict]] = {}
    for space, neighbors in per_space.items():
        for rank, (idx, cos) in enumerate(neighbors):
            if idx not in seen:
                seen[idx] = {}
            seen[idx][space] = {"rank": rank + 1, "cosine": round(float(cos), 4)}

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
    batch = 900
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
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--head-to-head",
        action="store_true",
        help="Build blinded harmony_legacy vs harmony_2dftm A/B manifests",
    )
    ap.add_argument(
        "--sidecar",
        default="data/artifacts/beat3tower_32k/harmony_2dftm_sidecar.npz",
        help="2DFTM harmony sidecar npz (head-to-head mode)",
    )
    args = ap.parse_args()

    from src.features.artifacts import load_artifact_bundle

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(args.artifact)

    npz = np.load(bundle.artifact_path, allow_pickle=True)
    per_tower = {
        "X_sonic_timbre": npz["X_sonic_timbre"],
        "X_sonic_harmony": npz["X_sonic_harmony"],
    }

    valid_mask = None
    medoid_vectors = None
    if args.head_to_head:
        sidecar_path = ROOT / args.sidecar
        if not sidecar_path.exists():
            print(f"No sidecar at {sidecar_path}. Run extract_harmony_2dftm_sidecar.py first.")
            sys.exit(1)
        X_2dftm, valid_mask = load_2dftm_sidecar(str(sidecar_path), bundle)
        n_valid = int(valid_mask.sum())
        print(f"Head-to-head mode: 2DFTM sidecar has {n_valid}/{len(valid_mask)} tracks.")
        spaces = compute_headtohead_spaces(bundle, per_tower, X_2dftm, valid_mask)
        medoid_vectors = _l2(bundle.X_sonic.astype(np.float64))  # representative seed pick
    else:
        print(f"Computing 4 sonic spaces over {len(bundle.track_ids)} tracks...")
        spaces = compute_spaces(bundle, per_tower)

    all_tids = [str(t) for t in bundle.track_ids]
    print("Looking up file paths...")
    file_paths = lookup_file_paths(all_tids, args.db)

    default_out = (
        "docs/run_audits/sonic_audition_h2h"
        if args.head_to_head
        else "docs/run_audits/sonic_audition"
    )
    out_dir = ROOT / (args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for artist in args.seeds:
        manifest = build_seed_manifest(
            artist, bundle, spaces, file_paths, k=args.top_k,
            valid_mask=valid_mask, medoid_vectors=medoid_vectors,
        )
        if manifest is None:
            print(f"  SKIP {artist!r} (not found in bundle)")
            continue
        slug = manifest["slug"]
        p = out_dir / f"{slug}_manifest.json"
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        n = len(manifest["neighbors"])
        index.append({"slug": slug, "artist": artist})
        print(f"  OK   {artist!r} → {slug}_manifest.json ({n} neighbors)")

    if not args.head_to_head:
        neg = build_negative_s_manifest(NEGATIVE_S_PAIRS, bundle, spaces, file_paths)
        (out_dir / "negative_s_manifest.json").write_text(
            json.dumps(neg, indent=2), encoding="utf-8"
        )
        index.append({"slug": "negative_s", "artist": "Negative-S Pairs", "type": "transition_pairs"})
        print(f"  OK   negative_s → negative_s_manifest.json ({len(neg['pairs'])} pairs)")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nDone. {len(index)} manifests in {out_dir}")
    serve_hint = f" --data-dir {out_dir.relative_to(ROOT)}" if args.head_to_head else ""
    print(f"Next: python scripts/sonic_audition_serve.py{serve_hint}")


if __name__ == "__main__":
    main()
