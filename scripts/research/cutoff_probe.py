"""Where exactly is The Embassy excluded from The Radio Dept's artist-style pool?

Two real cutoffs (from artist_style.py):
  - SONIC clusters: top-2000 per medoid by max MERT-sim (build_balanced_candidate_pool)
  - GENRE-neighbor pool: top-1500 by SMOOTHED (graph) genre_sim, min_sim 0.25, conf 0.50
    (build_genre_neighbor_candidate_pool)

Measures Embassy's rank/sim in BOTH spaces vs the cutoffs, with RD's actual playlist
members as the reference. Answers: is it the sonic top-K, the genre cap/threshold, or
does the GRAPH itself fail to rate Embassy<->RD as close (user's expectation)?
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # worktree (CODE_ROOT)
from src.playlist.artist_style import _compute_genre_similarity  # noqa: E402

a = np.load(ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz", allow_pickle=True)
ids = [str(x) for x in a["track_ids"]]
artists = [str(x) for x in a["track_artists"]]
mert = np.asarray(a["X_sonic_mert"], np.float32)
graw = np.asarray(a["X_genre_raw"], np.float32)
gsmooth = np.asarray(a["X_genre_smoothed"], np.float32) if "X_genre_smoothed" in a.files else None
N = len(ids)
print(f"N={N}  smoothed_genre={'yes' if gsmooth is not None else 'NO'}")


def norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


mn = norm(mert)


def idxs(name):
    return [i for i, ar in enumerate(artists) if ar.strip().lower() == name.strip().lower()]


def ranks(score, exclude):
    s = score.copy()
    s[exclude] = -1e9
    order = np.argsort(-s)
    r = np.empty(N, dtype=int)
    r[order] = np.arange(N)
    return r


rd = idxs("The Radio Dept.")
emb = idxs("The Embassy")
print(f"RD tracks={len(rd)}  Embassy tracks={len(emb)}")

# ---- SONIC: max MERT-sim to RD tracks; rank vs the top-2000-per-cluster cutoff ----
sonic = np.max(mn @ mn[rd].T, axis=1)
srank = ranks(sonic, rd)
print("\n=== SONIC (max MERT-sim to RD tracks) — cluster cutoff = top ~2000/medoid ===")
print(f"  {'artist':22s} {'bestRank':>8s} {'maxSim':>6s}")
for nm in ["The Embassy", "Beach House", "Slowdive", "Cocteau Twins", "Wild Nothing", "Tennis"]:
    B = idxs(nm)
    if not B:
        print(f"  {nm:22s} (not found)")
        continue
    br = int(min(srank[b] for b in B))
    print(f"  {nm:22s} {br:>8d} {float(np.max(sonic[B])):>6.3f}{'   <-- EMBASSY' if nm=='The Embassy' else ''}")

# ---- GENRE: smoothed (graph) + raw; sim + rank vs min_sim 0.25 and top-1500 cap ----
for label, M in [("SMOOTHED(graph)", gsmooth), ("RAW(tags)", graw)]:
    if M is None:
        continue
    seed = np.max(M[rd], axis=0)
    gsim = _compute_genre_similarity(seed, M, method="ensemble")
    grank = ranks(gsim, rd)
    print(f"\n=== GENRE {label} (genre_sim to RD seed) — pool cutoff = top-1500, min_sim 0.25 ===")
    print(f"  {'artist':22s} {'genreSim':>8s} {'bestRank':>8s} {'inPool?':>8s}")
    for nm in ["The Embassy", "Beach House", "Slowdive", "Wild Nothing", "Tennis"]:
        B = idxs(nm)
        if not B:
            print(f"  {nm:22s} (not found)")
            continue
        gs = float(np.max(gsim[B]))
        br = int(min(grank[b] for b in B))
        in_pool = "yes" if (gs >= 0.25 and br < 1500) else "NO"
        print(f"  {nm:22s} {gs:>8.3f} {br:>8d} {in_pool:>8s}{'   <-- EMBASSY' if nm=='The Embassy' else ''}")
