"""Analyze the Essentia head probe: which heads add pace signal, and redundancy.

Scores each candidate's pace-adjacency AUC (reusing the pace-eval corpus/pairs/metrics)
and reports redundancy: per-track |corr| vs arousal & danceability, and pairwise
distance-correlation vs MERT (high = MERT already captures it). Corpus-scoped z-score
(the probe only has the new heads for the 158 corpus tracks), stated in the output.
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np

ROOT = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3"
WT = f"{ROOT}/.claude/worktrees/pace-energy-steering"
sys.path.insert(0, WT)
from scripts.research.pace_eval_corpus import build_pairs, resolve_corpus  # noqa: E402
from scripts.research.pace_eval_metrics import auc_pos_below_neg, zscore_params, apply_zscore  # noqa: E402

PROBE = f"{ROOT}/docs/run_audits/pace_axis_eval/head_probe.tsv"
MERT_SIDECAR = f"{ROOT}/data/artifacts/beat3tower_32k/mert_sidecar.npz"
HEADS = ["arousal", "danceability", "aggressive", "relaxed", "electronic", "acoustic", "instrumental"]


def _load_probe():
    rows = [ln.rstrip("\n").split("\t") for ln in open(PROBE, encoding="utf-8")]
    hdr = rows[0]
    out = {}
    for r in rows[1:]:
        rec = dict(zip(hdr, r))
        if rec.get("arousal") == "ERR":
            continue
        out[rec["track_id"]] = {h: float(rec[h]) for h in HEADS}
    return out


def _mert_for(track_ids):
    z = np.load(MERT_SIDECAR, allow_pickle=True)
    art_ids = [str(t) for t in z["track_ids"]]
    pos = {t: i for i, t in enumerate(art_ids)}
    mert = np.asarray(z["emb_mid"], dtype=float)  # mid-segment 768-dim MERT
    out = {}
    for t in track_ids:
        j = pos.get(t)
        if j is not None:
            v = mert[j]
            n = np.linalg.norm(v)
            out[t] = v / n if n > 0 else v  # unit-norm for cosine distance
    return out


def main():
    con = sqlite3.connect(f"file:{ROOT}/data/metadata.db?mode=ro", uri=True)
    tracks, _ = resolve_corpus(con)
    con.close()
    probe = _load_probe()
    tracks = [t for t in tracks if t.track_id in probe]
    ids = [t.track_id for t in tracks]
    idx = {t: i for i, t in enumerate(ids)}
    mert = _mert_for(ids)

    # z-scored head matrix (corpus-scoped)
    raw = {h: np.array([probe[t][h] for t in ids]) for h in HEADS}
    zs = {}
    for h in HEADS:
        m, s = zscore_params(raw[h])
        zs[h] = apply_zscore(raw[h], m, s)

    pairs = build_pairs(tracks, seed=13)

    def cand_vec(t, keys):
        return np.array([zs[k][idx[t]] for k in keys])

    def auc_for(keys):
        def dists(plist):
            out = []
            for a, b in plist:
                if a in idx and b in idx:
                    out.append(float(np.linalg.norm(cand_vec(a, keys) - cand_vec(b, keys))))
            return np.array(out)
        adj = dists(pairs["adjacent"])
        adjg = dists(pairs["adjacent_gradient"])
        non = dists(pairs["non_adjacent_same_album"])
        rnd = dists(pairs["random_cross"])
        return auc_pos_below_neg(adj, rnd), auc_pos_below_neg(adjg, non)

    # MERT pairwise distance correlation (redundancy vs sonic) over random_cross
    def mert_redundancy(keys):
        hd, md = [], []
        for a, b in pairs["random_cross"]:
            if a in idx and b in idx and a in mert and b in mert:
                hd.append(float(np.linalg.norm(cand_vec(a, keys) - cand_vec(b, keys))))
                md.append(float(np.linalg.norm(mert[a] - mert[b])))  # euclidean on unit-norm ≈ cosine dist
        if len(hd) < 3:
            return float("nan")
        return float(np.corrcoef(hd, md)[0, 1])

    ar, da = zs["arousal"], zs["danceability"]
    print("HEAD PROBE — N=%d corpus tracks; pairs adj=%d adj_grad=%d nonadj=%d random=%d"
          % (len(ids), len(pairs["adjacent"]), len(pairs["adjacent_gradient"]),
             len(pairs["non_adjacent_same_album"]), len(pairs["random_cross"])))
    print("(z-score corpus-scoped; AUC higher=better; |corr| vs arousal/dance per-track; MERTred=pairdist corr, high=redundant w/ sonic)\n")
    print(f"{'head':14}{'auc_coarse':>11}{'auc_fine':>10}{'|r|arousal':>11}{'|r|dance':>10}{'MERTred':>9}")
    for h in HEADS:
        ac, af = auc_for([h])
        ra = abs(float(np.corrcoef(zs[h], ar)[0, 1]))
        rd = abs(float(np.corrcoef(zs[h], da)[0, 1]))
        mr = mert_redundancy([h])
        print(f"{h:14}{ac:11.3f}{af:10.3f}{ra:11.2f}{rd:10.2f}{mr:9.2f}")

    print("\n=== combinations (does a head ADD over energy_pair?) ===")
    combos = {
        "energy_pair[ar,da]": ["arousal", "danceability"],
        "+aggressive": ["arousal", "danceability", "aggressive"],
        "+relaxed": ["arousal", "danceability", "relaxed"],
        "+aggr+relax": ["arousal", "danceability", "aggressive", "relaxed"],
        "ar+aggressive": ["arousal", "aggressive"],
        "+instrumental": ["arousal", "danceability", "instrumental"],
    }
    print(f"{'candidate':22}{'auc_coarse':>11}{'auc_fine':>10}")
    for name, keys in combos.items():
        ac, af = auc_for(keys)
        print(f"{name:22}{ac:11.3f}{af:10.3f}")


if __name__ == "__main__":
    main()
