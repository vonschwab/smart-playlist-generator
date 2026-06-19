"""Pass-1 scoring orchestrator for the pace-representation eval."""
from __future__ import annotations

import math
import os

import numpy as np

from scripts.research.pace_eval_features import CANDIDATES, candidate_vector
from scripts.research.pace_eval_metrics import (
    auc_pos_below_neg,
    distribution,
    weighted_euclidean,
)


def _pair_distances(name, pairs_list, corpus_index, zs, zt) -> np.ndarray:
    out = []
    for a, b in pairs_list:
        ia, ib = corpus_index.get(a), corpus_index.get(b)
        if ia is None or ib is None:
            continue
        va = candidate_vector(name, ia, zs, zt)
        vb = candidate_vector(name, ib, zs, zt)
        out.append(weighted_euclidean(va, vb))
    return np.asarray(out, dtype=float)


def score_candidates(corpus_index, pairs, zscored_scalars, ztower) -> dict:
    results = {}
    for name in CANDIDATES:
        adj = _pair_distances(name, pairs["adjacent"], corpus_index, zscored_scalars, ztower)
        adj_grad = _pair_distances(name, pairs["adjacent_gradient"], corpus_index, zscored_scalars, ztower)
        non = _pair_distances(name, pairs["non_adjacent_same_album"], corpus_index, zscored_scalars, ztower)
        rnd = _pair_distances(name, pairs["random_cross"], corpus_index, zscored_scalars, ztower)
        results[name] = {
            "auc_adj_vs_random": auc_pos_below_neg(adj, rnd),
            "auc_adj_vs_nonadj": auc_pos_below_neg(adj_grad, non),
            "adjacent": distribution(adj),
            "adjacent_gradient": distribution(adj_grad),
            "non_adjacent_same_album": distribution(non),
            "random_cross": distribution(rnd),
        }
    return results


def run_pass1(*, db_path, artifact_path, sidecar_path, out_dir, seed=13) -> dict:
    import sqlite3

    from scripts.research.pace_eval_corpus import build_pairs, resolve_corpus, write_corpus_tsv
    from scripts.research.pace_eval_features import load_raw_features, zscore_features

    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        corpus_tracks, counts = resolve_corpus(conn)
    finally:
        conn.close()
    if not corpus_tracks:
        raise RuntimeError(f"Corpus resolved empty — check db_path and CORPUS LIKE patterns. counts={counts}")
    write_corpus_tsv(os.path.join(out_dir, "corpus.tsv"), corpus_tracks)

    # z-score over ALL artifact tracks (library-wide), then index the corpus.
    art = np.load(artifact_path, allow_pickle=True)
    all_ids = [str(t) for t in art["track_ids"]]
    index, raw_scalars, raw_tower = load_raw_features(
        all_ids, db_path=db_path, artifact_path=artifact_path, sidecar_path=sidecar_path)
    zs, zt = zscore_features(raw_scalars, raw_tower)

    corpus_index = {t.track_id: index[t.track_id] for t in corpus_tracks if t.track_id in index}
    pairs = build_pairs(corpus_tracks, seed=seed)
    results = score_candidates(corpus_index, pairs, zs, zt)

    def _sort_key(kv):
        v = kv[1]["auc_adj_vs_random"]
        return -v if (v is not None and math.isfinite(v)) else 1.0

    # results_pass1.tsv
    with open(os.path.join(out_dir, "results_pass1.tsv"), "w", encoding="utf-8") as f:
        f.write("candidate\tauc_adj_vs_random\tauc_adj_vs_nonadj\t"
                "adj_p50\tadj_gradient_p50\tnonadj_p50\trandom_p50\t"
                "adj_n\tadj_gradient_n\tnonadj_n\trandom_n\n")
        for name, r in sorted(results.items(), key=_sort_key):
            f.write(f"{name}\t{r['auc_adj_vs_random']:.4f}\t{r['auc_adj_vs_nonadj']:.4f}\t"
                    f"{r['adjacent']['p50']:.4f}\t{r['adjacent_gradient']['p50']:.4f}\t"
                    f"{r['non_adjacent_same_album']['p50']:.4f}\t"
                    f"{r['random_cross']['p50']:.4f}\t{r['adjacent']['n']}\t"
                    f"{r['adjacent_gradient']['n']}\t"
                    f"{r['non_adjacent_same_album']['n']}\t{r['random_cross']['n']}\n")

    return {"counts": counts, "n_corpus": len(corpus_tracks),
            "pairs": {k: len(v) for k, v in pairs.items()}, "results": results}
