#!/usr/bin/env python3
"""
Genre Similarity Research Diagnostic
====================================

Empirical investigation of WHY the dense PMI-SVD genre embedding discriminates
poorly (admits ~56% of the library at the narrow floor; scores a genre-less
track at 0.79 vs an unrelated seed).

Read-only. Retrains PMI-SVD in-memory from the artifact's X_genre_raw so it is
independent of the (stale) shipped sidecar. No DB writes, no artifact writes.

Sections:
  A. Anisotropy of the SHIPPED X_genre_dense (the failing artifact).
  B. Mechanism: retrain PMI-SVD, show top singular direction dominance.
  C. Fix test: all-but-the-top (mean-center + drop top-k PCs) re-projection.
  D. Ground-truth separation (AUC): "shares >=2 specific genres" vs "shares 0".
  E. Reference-seed top-K neighbor tables for qualitative sanity.

Usage:
    python scripts/research_genre_similarity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.utils.extmath import randomized_svd

ARTIFACT = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
RNG = np.random.default_rng(42)

REFERENCE = {
    "charli_xcx": "065933d8e2e0db664ec57af1511b662b",
    "acetone": "6d5696a29b57a765f945ed1be2d6dfee",
    "pharoah_sanders": "a3bb6db554f1d2bae1a8b998ebf53925",
    "beach_boys": "fc302980e8359e5bab53e7a2f45fc61b",
}


def hr(c="-", n=78):
    print(c * n)


def avg_pairwise_cosine(M, n_pairs=40000):
    """Mean cosine over random track pairs (M rows assumed L2-normalized)."""
    N = M.shape[0]
    i = RNG.integers(0, N, n_pairs)
    j = RNG.integers(0, N, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    sims = np.einsum("ij,ij->i", M[i], M[j])
    return sims


def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


def train_ppmi_svd(X, dim=64, smoothing=1.0):
    """Replicates src/genre/pmi_svd.train_pmi_svd but returns RAW U,S too."""
    X = np.asarray(X, dtype=np.float64)
    V = X.shape[1]
    cooc = X.T @ X
    total = cooc.sum()
    denom = total + smoothing * (V * V)
    P = (cooc + smoothing) / denom
    pm = P.sum(axis=1)
    outer = np.outer(pm, pm)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(P / np.maximum(outer, 1e-300))
    ppmi = np.maximum(pmi, 0.0)
    U, S, _ = randomized_svd(ppmi, n_components=dim, random_state=42)
    emb = U * np.sqrt(S)              # (V, dim) GloVe-style, pre-normalization
    return emb, S


def project(X_raw, genre_emb):
    proj = X_raw @ genre_emb
    has = (X_raw > 0).any(axis=1)
    proj = l2norm(proj)
    proj[~has] = 0.0
    return proj.astype(np.float64)


def all_but_the_top(genre_emb, k):
    """Mu & Viswanath: subtract mean, remove top-k principal components."""
    mu = genre_emb.mean(axis=0, keepdims=True)
    centered = genre_emb - mu
    if k > 0:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top = Vt[:k]                       # (k, dim)
        centered = centered - (centered @ top.T) @ top
    return centered


def auc(pos, neg):
    """Mann-Whitney AUC: P(pos > neg)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    d = np.load(ARTIFACT, allow_pickle=True)
    X_raw = d["X_genre_raw"].astype(np.float64)
    vocab = [str(g) for g in d["genre_vocab"]]
    track_ids = [str(t) for t in d["track_ids"]]
    tid2i = {t: i for i, t in enumerate(track_ids)}
    N, V = X_raw.shape
    support = (X_raw > 0).sum(axis=0)
    nnz = (X_raw > 0).sum(axis=1)
    print(f"Artifact: N={N} tracks, V={V} genres, median genres/track={np.median(nnz):.0f}")

    # ---- A. Shipped embedding anisotropy ----
    hr("=")
    print("A. SHIPPED X_genre_dense anisotropy (the failing artifact)")
    hr("=")
    sc_path = ARTIFACT.parent / f"{ARTIFACT.stem}_genre_emb_dim64.npz"
    if sc_path.exists():
        sc = np.load(sc_path, allow_pickle=True)
        Xd = sc["X_genre_dense"].astype(np.float64)
        has = np.linalg.norm(Xd, axis=1) > 1e-9
        Xd_g = Xd[has]
        mu_norm = np.linalg.norm(Xd_g.mean(axis=0))
        sims = avg_pairwise_cosine(Xd_g)
        print(f"  tracks with genre vector : {has.sum()}")
        print(f"  ||mean track vector||    : {mu_norm:.3f}   (0=isotropic, 1=fully collapsed)")
        print(f"  random-pair cosine       : mean={sims.mean():.3f} p10={np.percentile(sims,10):.3f} "
              f"p50={np.percentile(sims,50):.3f} p90={np.percentile(sims,90):.3f}")
        print("  -> If mean-pair cosine >> 0, the floor cannot separate related from unrelated.")
    else:
        print("  (shipped sidecar not found; skipping)")

    # ---- B. Mechanism: retrain, singular spectrum ----
    hr("=")
    print("B. MECHANISM — retrain PMI-SVD on current artifact")
    hr("=")
    emb_raw, S = train_ppmi_svd(X_raw, dim=64)
    emb_raw_n = l2norm(emb_raw)
    # First singular direction sign profile
    U1 = emb_raw[:, 0]
    print(f"  singular values S[0:5] = {np.round(S[:5],2)}")
    print(f"  S[0]/S[1] ratio        = {S[0]/S[1]:.2f}  (dominant first axis if >>1)")
    print(f"  dim-0 of genre emb: {100*(U1>0).mean():.0f}% positive  "
          f"(Perron-Frobenius: ~all-positive => shared direction)")
    g_mu_norm = np.linalg.norm(emb_raw_n.mean(axis=0))
    print(f"  ||mean genre vector|| (L2-normed emb) = {g_mu_norm:.3f}")

    # ---- C. Fix test: all-but-the-top re-projection ----
    hr("=")
    print("C. FIX TEST — all-but-the-top (mean-center + drop top-k PCs), re-project tracks")
    hr("=")
    variants = {"current (raw PMI-SVD)": l2norm(emb_raw)}
    for k in (1, 2, 3):
        variants[f"all-but-top-{k}"] = l2norm(all_but_the_top(emb_raw, k))

    proj_cache = {}
    yuji = tid2i.get("dcfd2c974b15618546c8326a8ce9844d")
    pains = tid2i.get("c5ecd829c72537096005489f1d56c1d3")
    print(f"  {'variant':24} {'mean-pair-cos':>13} {'p90-pair':>9} {'Yuji vs Pains':>14}")
    for name, ge in variants.items():
        P = project(X_raw, ge)
        proj_cache[name] = P
        gmask = np.linalg.norm(P, axis=1) > 1e-9
        sims = avg_pairwise_cosine(P[gmask])
        yp = float(P[yuji] @ P[pains]) if (yuji is not None and pains is not None) else float("nan")
        print(f"  {name:24} {sims.mean():>13.3f} {np.percentile(sims,90):>9.3f} {yp:>14.3f}")
    print("  -> Goal: mean-pair-cos near 0, and Yuji-vs-Pains LOW (they share no real genre).")

    # ---- Sparse baselines ----
    from src.playlist.genre_idf import compute_genre_idf
    idf = compute_genre_idf(X_genre_raw=X_raw, power=1.0, norm="max1")
    sparse_raw = l2norm(X_raw.copy())
    sparse_idf = l2norm(X_raw * idf[None, :])

    # Candidate production fix: all-but-top-2 embedding + IDF-weighted projection
    # (hub genres like 'rock' contribute little to a track's dense position).
    X_idf = X_raw * idf[None, :]
    proj_cache["abt-2 + IDF-proj"] = project(X_idf, all_but_the_top(emb_raw, 2))

    # ---- D. Ground-truth separation AUC ----
    hr("=")
    print("D. GROUND-TRUTH SEPARATION (AUC: shares >=2 specific genres vs shares 0)")
    hr("=")
    # "specific" = genre support below the library median support among present genres
    present_support = support[support > 0]
    spec_thresh = np.percentile(present_support, 50)
    is_specific = (support > 0) & (support <= spec_thresh)
    Xspec = ((X_raw > 0) & is_specific[None, :]).astype(np.float32)  # (N,V)
    Xany = (X_raw > 0).astype(np.float32)

    # Per-anchor sampling: random pairs almost never hit the extremes (hub genres
    # create overlap everywhere), so for each anchor we deterministically draw one
    # positive (>=2 shared specific) and one negative (0 shared any) partner.
    have = np.where(nnz > 0)[0]
    anchors = RNG.choice(have, 4000, replace=False)
    ai_p, bi_p, ai_n, bi_n = [], [], [], []
    for s in anchors:
        spec_counts = Xspec @ Xspec[s]          # (N,) shared specific genres
        any_counts = Xany @ Xany[s]             # (N,) shared any genres
        pos_idx = np.where(spec_counts >= 2)[0]
        pos_idx = pos_idx[pos_idx != s]
        neg_idx = np.where(any_counts == 0)[0]  # genre vector exists but 0 overlap
        neg_idx = neg_idx[nnz[neg_idx] > 0]
        if len(pos_idx) and len(neg_idx):
            ai_p.append(s)
            bi_p.append(RNG.choice(pos_idx))
            ai_n.append(s)
            bi_n.append(RNG.choice(neg_idx))
    ai_p = np.array(ai_p)
    bi_p = np.array(bi_p)
    ai_n = np.array(ai_n)
    bi_n = np.array(bi_n)
    m = len(ai_p)
    print(f"  positive pairs (>=2 shared specific): {m}    negative pairs (0 shared): {len(ai_n)}")
    hr()
    print(f"  {'method':30} {'AUC':>7} {'pos-sim':>9} {'neg-sim':>9} {'gap':>7}")
    methods = {
        "sparse raw cosine": sparse_raw,
        "sparse IDF cosine": sparse_idf,
        "dense current (raw PMI-SVD)": proj_cache["current (raw PMI-SVD)"],
        "dense all-but-top-2": proj_cache["all-but-top-2"],
        "dense abt-2 + IDF-proj": proj_cache["abt-2 + IDF-proj"],
    }
    # also report the random-pair median so we can see where a floor would sit
    for name, M in methods.items():
        ps = np.einsum("ij,ij->i", M[ai_p], M[bi_p])
        ns = np.einsum("ij,ij->i", M[ai_n], M[bi_n])
        gmask = np.linalg.norm(M, axis=1) > 1e-9
        rnd = avg_pairwise_cosine(M[gmask])
        a_uc = auc(ps, ns)
        print(f"  {name:30} {a_uc:>7.3f} {ps.mean():>9.3f} {ns.mean():>9.3f} {ps.mean()-ns.mean():>7.3f}"
              f"   rnd-p50={np.percentile(rnd,50):.2f} rnd-p90={np.percentile(rnd,90):.2f}")
    print("  -> AUC=1.0 perfect separation; 0.5 = useless. gap = dynamic range available to a floor.")
    print("  -> rnd-p50/p90 = where random (often hub-sharing) pairs sit; a floor must clear these.")

    # ---- E. Reference neighbor tables ----
    hr("=")
    print("E. TOP-6 NEIGHBORS for reference seeds (genre tags shown)")
    hr("=")
    artists = [str(x) for x in d["track_artists"]] if "track_artists" in d else ["?"] * N
    titles = [str(x) for x in d["track_titles"]] if "track_titles" in d else ["?"] * N

    def tags(i, top=4):
        js = np.where(X_raw[i] > 0)[0]
        js = js[np.argsort(-X_raw[i, js])][:top]
        return ", ".join(vocab[j] for j in js) or "(none)"

    cmp_methods = {
        "sparse IDF": sparse_idf,
        "dense current": proj_cache["current (raw PMI-SVD)"],
        "abt2+IDFproj": proj_cache["abt-2 + IDF-proj"],
    }
    artist_arr = np.array([a.lower() for a in artists])
    for seed_name, stid in REFERENCE.items():
        si = tid2i.get(stid)
        if si is None:
            continue
        seed_artist = artist_arr[si]
        print(f"\n  SEED [{seed_name}] {artists[si][:22]} — {titles[si][:26]}  | {tags(si)}")
        for mname, M in cmp_methods.items():
            sims = M @ M[si]
            sims[artist_arr == seed_artist] = -1  # cross-artist neighbors only
            top = np.argsort(-sims)[:4]
            print(f"    {mname:14}: " + " | ".join(
                f"{artists[t][:15]}~{tags(t,2)}({sims[t]:.2f})" for t in top))

    print()


if __name__ == "__main__":
    main()
