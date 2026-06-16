#!/usr/bin/env python3
"""
Sonic TRANSITION-metric attractor test
=======================================

Follow-up to research_sonic_hubness.py, which refuted the global-hub hypothesis
using full-track similarity. The beam actually scores END-of-A -> START-of-B
segment transitions (centered cosine, tower-weighted). This script tests whether
that edge metric over-selects noisy/shoegaze/punk tracks as continuations of
non-noisy seeds, and whether there is a directional energy "drift".

Read-only; reproducible.  python scripts/research_sonic_transition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
RNG = np.random.default_rng(42)
TOWER_W = {"rhythm": 0.20, "timbre": 0.50, "harmony": 0.30}

SUBGENRES = {
    "shoegaze": ["shoegaze"],
    "punk": ["punk", "hardcore"],
    "noise": ["noise rock", "noise pop", "no wave", "noise "],
    "any-noisy": ["punk", "hardcore", "noise", "shoegaze", "no wave", "grindcore", "screamo"],
}


def l2(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def centered(M):
    M = l2(M.astype(np.float64))
    return l2(M - M.mean(axis=0, keepdims=True))


def mask_for(raw, vocab, subs):
    cols = [j for j, g in enumerate(vocab) if any(s in g for s in subs)]
    return (raw[:, cols] > 0).any(axis=1) if cols else np.zeros(raw.shape[0], bool)


def trans_sim(end_towers, start_towers, qidx):
    """Weighted centered cosine: END[q] -> START[all].  Returns (q, N)."""
    N = start_towers["timbre"].shape[0]
    acc = np.zeros((len(qidx), N))
    for name in TOWER_W:
        E = end_towers[name][qidx]
        S = start_towers[name]
        acc += TOWER_W[name] * (E @ S.T)
    return acc


def full_sim(towers, qidx):
    N = towers["timbre"].shape[0]
    acc = np.zeros((len(qidx), N))
    for name in TOWER_W:
        C = towers[name]
        acc += TOWER_W[name] * (C[qidx] @ C.T)
    return acc


def enrich(sims, qidx, mask, base, k=20):
    s = sims.copy()
    s[np.arange(len(qidx)), qidx] = -9
    part = np.argpartition(-s, k, axis=1)[:, :k]
    frac = mask[part].mean(axis=1).mean()
    return float(frac), float(frac / max(base, 1e-9))


def main():
    d = np.load(ART, allow_pickle=True)
    raw = d["X_genre_raw"]
    vocab = [str(g) for g in d["genre_vocab"]]
    full = {n: centered(d[f"X_sonic_{n}"]) for n in TOWER_W}
    end = {n: centered(d[f"X_sonic_{n}_end"]) for n in TOWER_W}
    start = {n: centered(d[f"X_sonic_{n}_start"]) for n in TOWER_W}
    N = raw.shape[0]

    masks = {k: mask_for(raw, vocab, s) for k, s in SUBGENRES.items()}
    bases = {k: float(m.mean()) for k, m in masks.items()}
    any_noisy = masks["any-noisy"]
    neutral = np.where(~any_noisy & (raw > 0).any(axis=1))[0]
    qidx = RNG.choice(neutral, min(2000, len(neutral)), replace=False)

    print("Base rates (fraction of library):")
    for k in SUBGENRES:
        print(f"  {k:10} {bases[k]:.3f}")
    print(f"\nQueries: {len(qidx)} non-noisy seeds. Enrichment = neighbor-frac / base. >1 = over-selected.\n")

    print(f"  {'subgenre':10} | {'FULL-track':>22} | {'TRANSITION end->start':>24}")
    print(f"  {'':10} | {'frac':>10} {'enrich':>10} | {'frac':>11} {'enrich':>11}")
    fsim = full_sim(full, qidx)
    tsim = trans_sim(end, start, qidx)
    for k in SUBGENRES:
        ff, fe = enrich(fsim, qidx, masks[k], bases[k])
        tf, te = enrich(tsim, qidx, masks[k], bases[k])
        print(f"  {k:10} | {ff:>10.3f} {fe:>9.2f}x | {tf:>11.3f} {te:>10.2f}x")

    # ---- directional drift: is the BEST transition target noisier than the source? ----
    print("\nDIRECTIONAL DRIFT (does the single best transition-continuation skew noisy?)")
    s = tsim.copy()
    s[np.arange(len(qidx)), qidx] = -9
    best = s.argmax(axis=1)
    print(f"  top-1 transition continuation of non-noisy seeds is any-noisy: "
          f"{any_noisy[best].mean():.3f}  (base {bases['any-noisy']:.3f})")

    # ---- shoegaze as START-segment universality ----
    print("\nSTART-SEGMENT UNIVERSALITY (which genres are 'easy continuations' for ALL seeds?)")
    qall = RNG.choice(N, 2000, replace=False)
    ts_all = trans_sim(end, start, qall)
    ts_all[np.arange(len(qall)), qall] = -9
    part = np.argpartition(-ts_all, 20, axis=1)[:, :20]
    for k in SUBGENRES:
        f = masks[k][part].mean()
        print(f"  {k:10} as transition target (any seed): {f:.3f}  enrich {f/max(bases[k],1e-9):.2f}x")
    print()


if __name__ == "__main__":
    main()
