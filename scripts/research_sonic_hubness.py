#!/usr/bin/env python3
"""
Sonic hubness / noisy-attractor research
========================================

Tests the hypothesis that playlists "devolve into punk/shoegaze/noise" because
noisy tracks act as HUBS in the sonic embedding (high similarity to many seeds,
so the beam repeatedly pulls them in). Isolates which tower (rhythm/timbre/
harmony) drives it. Read-only; reproducible.

Sections:
  A. Library base rates (noisy vs calm genre tags).
  B. Neighbor enrichment: for NON-noisy seeds, what fraction of nearest sonic
     neighbors are noisy, per space (whitened full / rhythm / timbre / harmony /
     weighted-centered transition)? Enrichment >> 1 = noisy hubs.
  C. Hubness: k-occurrence skew + the actual top hub tracks and their genres.
  D. Timbre anisotropy: dominant PC, and whether it encodes "noisiness".

Usage:  python scripts/research_sonic_hubness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
RNG = np.random.default_rng(42)
TOWER_W = {"rhythm": 0.20, "timbre": 0.50, "harmony": 0.30}

NOISY = ["punk", "hardcore", "noise", "shoegaze", "no wave", "grindcore",
         "screamo", "garage", "post-hardcore", "math rock", "industrial", "thrash"]
CALM = ["folk", "ambient", "slowcore", "dream pop", "singer-songwriter",
        "americana", "chamber", "acoustic", "new age", "soft rock", "lo-fi"]


def l2(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def tag_mask(raw, vocab, substrs):
    cols = [j for j, g in enumerate(vocab) if any(s in g for s in substrs)]
    matched = sorted({vocab[j] for j in cols})
    mask = (raw[:, cols] > 0).any(axis=1) if cols else np.zeros(raw.shape[0], bool)
    return mask, matched


def neighbor_enrichment(M, query_idx, target_mask, base_rate, k=20):
    """Mean fraction of each query's top-k neighbors that are in target_mask."""
    Mn = l2(M)
    Q = Mn[query_idx]
    sims = Q @ Mn.T                      # (q, N)
    sims[np.arange(len(query_idx)), query_idx] = -2
    # top-k neighbors per query
    part = np.argpartition(-sims, k, axis=1)[:, :k]
    frac = target_mask[part].mean(axis=1)   # fraction noisy among each query's neighbors
    return float(frac.mean()), float(frac.mean() / base_rate)


def centered_weighted_sim(towers, query_idx, target_mask, base_rate, k=20):
    """Transition-style similarity: weighted sum of per-tower CENTERED cosine."""
    N = towers["timbre"].shape[0]
    acc = np.zeros((len(query_idx), N))
    for name, M in towers.items():
        C = l2(M)
        C = C - C.mean(axis=0, keepdims=True)
        C = l2(C)
        acc += TOWER_W[name] * (C[query_idx] @ C.T)
    acc[np.arange(len(query_idx)), query_idx] = -2
    part = np.argpartition(-acc, k, axis=1)[:, :k]
    frac = target_mask[part].mean(axis=1)
    return float(frac.mean()), float(frac.mean() / base_rate)


def hubness(M, n_queries=4000, k=20):
    """k-occurrence: how often each track appears in others' top-k. Returns counts."""
    Mn = l2(M)
    N = Mn.shape[0]
    q = RNG.choice(N, n_queries, replace=False)
    counts = np.zeros(N, dtype=np.int32)
    bs = 500
    for i in range(0, len(q), bs):
        qi = q[i:i + bs]
        sims = Mn[qi] @ Mn.T
        sims[np.arange(len(qi)), qi] = -2
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        for row in part:
            counts[row] += 1
    return counts, n_queries


def main():
    d = np.load(ART, allow_pickle=True)
    raw = d["X_genre_raw"]
    vocab = [str(g) for g in d["genre_vocab"]]
    art = [str(a) for a in d["track_artists"]]
    tit = [str(t) for t in d["track_titles"]]
    towers = {n: d[f"X_sonic_{n}"].astype(np.float64) for n in ["rhythm", "timbre", "harmony"]}
    whit = d["X_sonic_robust_whiten"].astype(np.float64)   # pool-admission space
    full = d["X_sonic"].astype(np.float64)                 # L2 concat
    N = raw.shape[0]

    # ---- A. base rates ----
    print("=" * 78)
    print("A. LIBRARY COMPOSITION")
    print("=" * 78)
    noisy_mask, noisy_tok = tag_mask(raw, vocab, NOISY)
    calm_mask, calm_tok = tag_mask(raw, vocab, CALM)
    br_noisy = float(noisy_mask.mean())
    br_calm = float(calm_mask.mean())
    print(f"  N={N}")
    print(f"  noisy base rate = {br_noisy:.3f} ({int(noisy_mask.sum())} tracks)  tokens={noisy_tok[:12]}")
    print(f"  calm  base rate = {br_calm:.3f} ({int(calm_mask.sum())} tracks)  tokens={calm_tok[:12]}")

    # NON-noisy, NON-calm queries (typical seeds that shouldn't devolve)
    neutral = np.where(~noisy_mask & (raw > 0).any(axis=1))[0]
    qidx = RNG.choice(neutral, min(2000, len(neutral)), replace=False)
    print(f"  queries: {len(qidx)} non-noisy tracks")

    # ---- B. neighbor enrichment ----
    print("=" * 78)
    print("B. NEIGHBOR ENRICHMENT for non-noisy seeds  (frac of top-20 neighbors that are NOISY)")
    print(f"   base rate noisy = {br_noisy:.3f}; enrichment = frac/base. >1 means noisy over-represented.")
    print("=" * 78)
    spaces = {
        "whitened full (pool admit)": whit,
        "L2 concat full": full,
        "rhythm tower (w=0.20)": towers["rhythm"],
        "timbre tower (w=0.50)": towers["timbre"],
        "harmony tower (w=0.30)": towers["harmony"],
    }
    print(f"  {'space':30} {'noisy-frac':>11} {'enrichment':>11}")
    for name, M in spaces.items():
        frac, enr = neighbor_enrichment(M, qidx, noisy_mask, br_noisy)
        print(f"  {name:30} {frac:>11.3f} {enr:>11.2f}x")
    frac_t, enr_t = centered_weighted_sim(towers, qidx, noisy_mask, br_noisy)
    print(f"  {'weighted CENTERED (transition)':30} {frac_t:>11.3f} {enr_t:>11.2f}x")
    # calm contrast in timbre
    fc, ec = neighbor_enrichment(towers["timbre"], qidx, calm_mask, br_calm)
    print(f"\n  [contrast] timbre-tower neighbors that are CALM: frac={fc:.3f} enrichment={ec:.2f}x "
          f"(base {br_calm:.3f})")

    # ---- C. hubness ----
    print("=" * 78)
    print("C. HUBNESS  (k-occurrence: how often a track is in others' top-20)")
    print("=" * 78)
    for sp_name, M in [("whitened full", whit), ("timbre tower", towers["timbre"])]:
        counts, nq = hubness(M)
        exp = nq * 20 / N                       # expected count if uniform
        skew = float(counts.std() / max(counts.mean(), 1e-9))
        hub_noisy = float(noisy_mask[counts >= np.percentile(counts, 99)].mean())
        print(f"\n  [{sp_name}] expected count/track={exp:.1f}  max={counts.max()}  skew(std/mean)={skew:.2f}")
        print(f"    among TOP-1% hubs: noisy fraction = {hub_noisy:.3f}  (base {br_noisy:.3f})")
        top = np.argsort(-counts)[:10]
        print(f"    top-10 hub tracks (count | genres | artist - title):")
        for i in top:
            js = np.where(raw[i] > 0)[0]
            js = js[np.argsort(-raw[i, js])][:3]
            g = ",".join(vocab[j] for j in js)
            print(f"      {counts[i]:>5}  {g:32} {art[i][:18]} - {tit[i][:24]}")

    # ---- D. timbre anisotropy / noisiness axis ----
    print("=" * 78)
    print("D. TIMBRE ANISOTROPY  (is the dominant timbre axis = 'noisiness'?)")
    print("=" * 78)
    T = l2(towers["timbre"])
    mu = T.mean(axis=0)
    print(f"  ||mean timbre vector|| = {np.linalg.norm(mu):.3f}  (0=isotropic, 1=collapsed)")
    Tc = T - mu
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    print(f"  timbre singular values S[0:4] = {np.round(S[:4],1)}  S0/S1={S[0]/S[1]:.2f}")
    pc1 = Tc @ Vt[0]
    # does PC1 separate noisy from non-noisy?
    m_noisy = float(pc1[noisy_mask].mean()); m_other = float(pc1[~noisy_mask].mean())
    sd = float(pc1.std())
    print(f"  PC1 projection: noisy mean={m_noisy:+.3f}  non-noisy mean={m_other:+.3f}  "
          f"separation={abs(m_noisy-m_other)/sd:.2f} sd")
    # which raw axis does mean align with (informational)
    print(f"  mean-vector dominant dims (|mu| top5): {np.argsort(-np.abs(mu))[:5].tolist()}")
    print()


if __name__ == "__main__":
    main()
