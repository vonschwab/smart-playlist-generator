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
    skipped: List[str] = []

    def _make_record(arm: str, seed: str, regime: str, a: str, b: str) -> dict:
        m = edge_metrics(
            a_onset=onset_of(a), b_onset=onset_of(b),
            a_bpm=bpm_of(a), b_bpm=bpm_of(b),
            a_genre=genre_of(a), b_genre=genre_of(b),
        )
        return {
            "arm": arm, "seed": seed, "regime": regime,
            "a": {"track_id": a, "artist": str(bundle.track_artists[tid_to_idx[a]]),
                  "title": str(bundle.track_titles[tid_to_idx[a]]),
                  "onset": onset_of(a), "bpm": bpm_of(a)},
            "b": {"track_id": b, "artist": str(bundle.track_artists[tid_to_idx[b]]),
                  "title": str(bundle.track_titles[tid_to_idx[b]]),
                  "onset": onset_of(b), "bpm": bpm_of(b)},
            **m,
        }

    for artist in args.seeds:
        piers = _artist_piers(bundle, artist)
        if len(piers) < 4:
            print(f"  SKIP {artist!r}: only {len(piers)} piers in artifact")
            skipped.append(artist)
            continue
        slug = artist.lower().replace(" ", "-").replace("/", "-")
        regime = regime_of.get(artist, "?")
        pier_set = set(piers)
        context: set = set()
        seed_records: List[dict] = []
        seed_playlists: List[dict] = []

        # Generate all 3 real arms for this seed. A single infeasible arm must NOT
        # crash the whole build (an audition over 18 generations has to tolerate the
        # odd seed whose narrow neighbourhood dead-ends). Accumulate per-seed and
        # commit all-or-nothing so arms stay balanced per seed.
        try:
            for arm in REAL_ARMS:
                res = generate_like_gui(
                    seeds=piers, pace_mode=arm, length=args.length,
                    random_seed=args.random_seed, **FIXED_MODES,
                )
                tids = [str(t) for t in res.track_ids]
                context.update(tids)
                onset_seq = [onset_of(t) if t in tid_to_idx else None for t in tids]
                seed_playlists.append(
                    {"seed": slug, "arm": arm, "track_ids": tids, "onset_seq": onset_seq}
                )
                interior = extract_interior_edges(tids, pier_set)
                for (i, j) in sample_edges(interior, args.edges_per_arm, rng):
                    seed_records.append(_make_record(arm, slug, regime, tids[i], tids[j]))
        except Exception as e:
            print(f"  SKIP {artist!r}: generation infeasible ({type(e).__name__}: {str(e)[:140]})")
            skipped.append(artist)
            continue

        ctx = [t for t in context if t in tid_to_idx]
        decoys = synthesize_decoy_edges(
            ctx,
            onset={t: onset_of(t) for t in ctx},
            bpm={t: bpm_of(t) for t in ctx},
            genre_vecs={t: genre_of(t) for t in ctx},
            k=args.decoy_per_seed, rng=rng, min_onset_dist=1.0,
        )
        for (a, b) in decoys:
            seed_records.append(_make_record("decoy", slug, regime, a, b))

        # Commit this seed only now that all arms + decoys succeeded.
        records.extend(seed_records)
        playlists.extend(seed_playlists)
        seeds_prov[slug] = {"piers": piers, "regime": regime}
        index_seeds.append(artist)
        print(f"  OK   {artist!r}: {len(seed_playlists)} playlists, {len(seed_records)} edges")

    if not records:
        print("No seeds produced edges — aborting (check generation feasibility).")
        sys.exit(1)
    if skipped:
        print(f"  Skipped {len(skipped)} seed(s): {', '.join(skipped)}")
    _surv = [regime_of.get(a, '?') for a in index_seeds]
    print(f"  Surviving seeds: {len(index_seeds)} "
          f"({_surv.count('ambient')} ambient, {_surv.count('rhythmic')} rhythmic)")

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
