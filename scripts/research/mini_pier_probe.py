"""SP3 prototype: what track would the mini-pier selection pick between two piers?

Compares v1 (max_min_sim = the central blur) vs v2 (feasibility-gated + RELATIVE
anti-center: among the top-K most-between candidates, subtract closeness to the LOCAL
between-region center, so the waypoint stays on the A->B line without settling into
the average; local center => piers that sit in dense space aren't over-penalized).

Runs three case sets: dance-punk consecutive piers, gold TWINS (soundalike pairs),
and SAME-ARTIST pairs. Excludes BOTH pier artists' catalogs (as the real beam does).
MuQ space; whole-library pool (real beam uses the gated segment pool). Prints titles.

  python scripts/research/mini_pier_probe.py [lambda] [K_feas]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
ART = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
MAN = Path(__file__).resolve().parent / "gold_pairs_manifest.json"
# MARGIN = the relative smoothness floor: a waypoint must have min-sim within this of
# the BEST available min-sim (adapts to close vs cross-niche piers). Among those
# genuinely-smooth candidates, pick the least-central (anti-center). Small margin =
# stay smooth (near the blur); larger = allow more character at some smoothness cost.
MARGIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
K_FEAS = int(sys.argv[2]) if len(sys.argv) > 2 else 150

SAME_ARTIST = ["Beach House", "Slowdive", "Black Sabbath", "Bill Evans", "The Strokes"]


def main() -> None:
    a = np.load(ART, allow_pickle=True)
    art = np.array([" ".join(str(x).split()) for x in a["track_artists"]])
    tit = np.array([str(t) for t in a["track_titles"]])
    X = np.asarray(a["X_sonic_muq"], np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    artl = np.array([s.strip().lower() for s in art])

    def name(i):
        return f"{art[i]} - {tit[i]}"

    def probe(label, A, B):
        simA, simB = X @ X[A], X @ X[B]
        minsim = np.minimum(simA, simB)
        # exclude both pier artists' whole catalogs + the piers themselves
        mask = (artl == artl[A]) | (artl == artl[B])
        minsim[mask] = -9.0
        # blur direction of the between-region (broad, stable)
        broad = np.argpartition(-minsim, K_FEAS)[:K_FEAS]
        center = X[broad].mean(0)
        center = center / (np.linalg.norm(center) + 1e-12)
        # SMOOTHNESS FLOOR (relative): only candidates within MARGIN of the best min-sim
        best = float(minsim[broad].max())
        feas = np.where(minsim >= best - MARGIN)[0]
        v1 = int(broad[int(np.argmax(minsim[broad]))])         # pure max_min_sim (blur)
        order = feas[np.argsort(X[feas] @ center)]             # least-central-first, smooth only
        print(f"\n=== {label} ===")
        print(f"  A: {name(A)}")
        print(f"  B: {name(B)}")
        print(f"  v1 blur:  {name(v1)}  [min {minsim[v1]:.2f} cent {float(X[v1] @ center):.2f}]  "
              f"(best_min={best:.2f}, floor={best - MARGIN:.2f}, {len(feas)} smooth cands)")
        for j in order[:5]:
            print(f"  v2 pick:  {name(j)}  [min {minsim[j]:.2f} cent {float(X[j] @ center):.2f}]")

    print(f"MuQ | smoothness margin={MARGIN} | K_feas={K_FEAS} | pier-artists excluded")

    print("\n################  GOLD TWINS  ################")
    recs = [r for r in json.loads(MAN.read_text(encoding="utf-8"))
            if r.get("verified_in_library") and "NEAR_NEGATIVE" not in r.get("status", "")]
    for r in recs[:8]:
        A, B = int(r["left_idx"]), int(r["right_idx"])
        probe(f"TWIN [{r.get('lane','?')}] {art[A]} ~ {art[B]}", A, B)

    print("\n################  SAME-ARTIST PAIRS  ################")
    for a_name in SAME_ARTIST:
        idx = np.where(artl == a_name.lower())[0]
        if len(idx) >= 2:
            probe(f"SAME-ARTIST {a_name}", int(idx[0]), int(idx[1]))


if __name__ == "__main__":
    main()
