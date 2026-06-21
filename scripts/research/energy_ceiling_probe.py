"""Measure the energy-reservoir CEILING in real gated runs.

Captures each segment's actual candidate pool (what the beam sees) by patching
the pool builder, then for each interior step asks:
  - POOL-ON-ARC: how many pool candidates already sit within `band` of the
    step's energy-arc target? (if >0, the soft penalty already had room)
  - ADDABLE: how many universe tracks within `band` of the target ALSO pass
    bridge_floor (the proposed reservoir's cohesion guard) and are NOT in the
    pool? (= what the reservoir could add)

Three-way verdict per step:
  HAD_ROOM   pool already has on-arc options  -> reservoir redundant there
  RESERVOIR  pool has none but addable>0      -> reservoir would help
  NO_SUPPORT neither                          -> library can't support pace there
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.playlist.pier_bridge_builder as pbb  # noqa: E402
from src.playlist.energy_loader import load_energy_matrix  # noqa: E402
from tests.support.gui_fidelity import generate_like_gui  # noqa: E402

SEEDS = [
    "49f8bba75408d4e0e0e000d1dc708add",
    "b587eb56fa1e173138152bf09565eb80",
    "f28fd5cebac845cf64fee59d5ac3b3aa",
]
SIDECAR = "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
BAND = 0.5  # z-std: "on the arc target"

_orig = pbb._build_segment_candidate_pool_scored
SEGMENTS = []
CTX = {}


def _capture(*args, **kw):
    res = _orig(*args, **kw)
    candidates = res[0]
    SEGMENTS.append({
        "pier_a": int(kw["pier_a"]), "pier_b": int(kw["pier_b"]),
        "universe": list(kw["universe_indices"]),
        "bridge_floor": float(kw["bridge_floor"]),
        "interior_length": int(kw.get("interior_length", 0) or 0),
        "candidates": [int(c) for c in candidates],
    })
    if "bundle" not in CTX:
        CTX["bundle"] = kw["bundle"]
        CTX["X"] = kw["X_full_norm"]
    return res


def main():
    pbb._build_segment_candidate_pool_scored = _capture
    try:
        generate_like_gui(seeds=SEEDS, length=20,
                          genre_mode="dynamic", sonic_mode="dynamic",
                          pace_mode="dynamic", cohesion_mode="dynamic")
    finally:
        pbb._build_segment_candidate_pool_scored = _orig

    bundle = CTX["bundle"]
    X = CTX["X"]
    e = load_energy_matrix(bundle.track_ids, sidecar_path=SIDECAR,
                           features=("arousal_p50",)).reshape(-1)

    verdict = Counter()
    pool_onarc_counts, addable_counts, pool_spreads = [], [], []
    # de-dup segments (relaxation re-calls the pool for the same segment); keep the
    # largest pool per (pier_a,pier_b) as the representative.
    best = {}
    for s in SEGMENTS:
        k = (s["pier_a"], s["pier_b"])
        if k not in best or len(s["candidates"]) > len(best[k]["candidates"]):
            best[k] = s

    for s in best.values():
        a, b = s["pier_a"], s["pier_b"]
        ea, eb = e[a], e[b]
        if not (np.isfinite(ea) and np.isfinite(eb)):
            continue
        cand = np.array(s["candidates"], dtype=int)
        uni = np.array(s["universe"], dtype=int)
        L = max(1, s["interior_length"])
        # precompute sonic sim to both piers for the universe
        sa = X[uni] @ X[a]
        sb = X[uni] @ X[b]
        pass_floor = np.minimum(sa, sb) >= s["bridge_floor"]
        cand_set = set(cand.tolist())
        e_cand = e[cand]
        e_cand = e_cand[np.isfinite(e_cand)]
        if len(e_cand):
            pool_spreads.append(float(np.std(e_cand)))
        for step in range(1, L):
            t = step / L
            target = ea + (eb - ea) * t
            # pool candidates on-arc
            on_pool = int(np.sum(np.abs(e[cand] - target) <= BAND))
            pool_onarc_counts.append(on_pool)
            # universe addable: near target, pass floor, not already in pool
            near = np.abs(e[uni] - target) <= BAND
            addable_mask = near & pass_floor & np.array([u not in cand_set for u in uni])
            addable = int(np.sum(addable_mask))
            addable_counts.append(addable)
            if on_pool > 0:
                verdict["HAD_ROOM"] += 1
            elif addable > 0:
                verdict["RESERVOIR"] += 1
            else:
                verdict["NO_SUPPORT"] += 1

    total = sum(verdict.values())
    print(f"\nSegments analyzed: {len(best)}  | interior steps: {total}  | band=±{BAND} z")
    print(f"pool energy spread (std) per segment: "
          f"{np.round(pool_spreads, 2).tolist() if pool_spreads else 'n/a'}")
    print(f"pool-on-arc per step:  mean={np.mean(pool_onarc_counts):.1f} "
          f"median={np.median(pool_onarc_counts):.0f} max={max(pool_onarc_counts)}")
    print(f"reservoir-addable per step: mean={np.mean(addable_counts):.1f} "
          f"median={np.median(addable_counts):.0f} max={max(addable_counts)}")
    print("\nPER-STEP VERDICT:")
    for k in ("HAD_ROOM", "RESERVOIR", "NO_SUPPORT"):
        print(f"  {k:11s} {verdict[k]:4d}  ({100*verdict[k]/total:.0f}%)")
    print("\nINTERPRETATION:")
    print("  HAD_ROOM high  -> soft penalty already had options; reservoir redundant (& soft term may be mis-weighted)")
    print("  RESERVOIR high -> reservoir would add real on-arc, sonically-connected options")
    print("  NO_SUPPORT high-> library lacks on-arc+cohesive tracks; pace is subordinate by construction here")


if __name__ == "__main__":
    main()
