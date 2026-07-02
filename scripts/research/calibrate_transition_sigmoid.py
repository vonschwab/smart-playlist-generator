"""Derive the calibrated-sigmoid transition params + sweep the T-consuming floors.

Read-only. Reads the live artifact (sonic matrices are stable across genre edits)
and imports the PRODUCTION calibrated rescale so the recommended params and the
floor sweep reflect exactly what the beam will compute.

Outputs:
  - the realistic centered end->start cosine band (p1/p50/p99),
  - recommended center / scale / gain (maps p1->~0.05, p99->~0.95),
  - the resulting blended T distribution over realistic pools,
  - a floor sweep: fraction of realistic candidate edges rejected at each
    candidate transition_floor (so floors can be set at a chosen percentile).

Usage:  python scripts/research/calibrate_transition_sigmoid.py [VARIANT]

VARIANT defaults to whatever the artifact's X_sonic_variant stamp declares
(muq post-SP-B). Pass an explicit variant to calibrate a different/future
sonic space without editing this script.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Import the production calibrated rescale from THIS checkout (worktree code).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.playlist.transition_metrics import (  # noqa: E402
    _calibrate_transition_cos,
    build_transition_metric_context,
)

# Artifact lives in the main checkout (worktree data/ has no artifacts); sonic
# matrices are unaffected by the genre session's concurrent edits.
ART = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz")
W = (0.70, 0.15, 0.15)        # end->start, mid->mid, full->full
POOL = 2000                    # realistic candidate pool per destination
N_DESTS = 60
TARGET_LO, TARGET_HI = 0.05, 0.95   # where p1/p99 of the band should land


def _stamped_variant(npz) -> str:
    """The artifact's own X_sonic_variant stamp (muq post-SP-B) — NOT a hardcoded
    default. A missing stamp is an error, never a silent fallback (the
    configured-knob-must-act rule): pass the variant argument explicitly."""
    if "X_sonic_variant" in npz:
        v = npz["X_sonic_variant"]
        return str(v.item() if hasattr(v, "item") else v)
    raise ValueError(
        "artifact has no X_sonic_variant stamp — pass the variant argument "
        "explicitly (python scripts/research/calibrate_transition_sigmoid.py <variant>)"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "variant", nargs="?", default=None,
        help="Sonic variant key (X_sonic_<variant>) to calibrate against. "
             "Defaults to the artifact's stamped X_sonic_variant (muq post-SP-B); "
             "this is the seam for calibrating a future variant without editing "
             "this script.",
    )
    return p.parse_args()


args = parse_args()
a = np.load(ART, allow_pickle=True)
VARIANT = args.variant or _stamped_variant(a)
print(f"Calibrating variant={VARIANT}")

artists = np.array([str(x).strip().lower() for x in a["track_artists"]])
X = np.asarray(a[f"X_sonic_{VARIANT}"], np.float32)
mst = np.asarray(a[f"X_sonic_{VARIANT}_start"], np.float32)
mmd = np.asarray(a[f"X_sonic_{VARIANT}_mid"], np.float32)
men = np.asarray(a[f"X_sonic_{VARIANT}_end"], np.float32)

ctx = build_transition_metric_context(
    X_sonic=X, X_start=mst, X_mid=mmd, X_end=men,
    center_transitions=True,
    weight_end_start=W[0], weight_mid_mid=W[1], weight_full_full=W[2],
)

rng = np.random.default_rng(7)
valid = np.linalg.norm(X, axis=1) > 1e-9
dests = rng.choice(np.where(valid)[0], N_DESTS, replace=False)


def components(dst):
    return (ctx.X_end @ ctx.X_start[dst], ctx.X_mid @ ctx.X_mid[dst], ctx.X_full @ ctx.X_full[dst])


# ---- collect the realistic centered end->start band + the per-pool cosines ----
pool_es = []
pools = []
for dst in dests:
    es, mm, ff = components(dst)
    braw = W[0] * es + W[1] * mm + W[2] * ff
    braw[dst] = -9
    braw[artists == artists[dst]] = -9   # seed-artist exclusion parity
    pool = np.argpartition(-braw, POOL)[:POOL]
    pools.append((es[pool], mm[pool], ff[pool]))
    pool_es.append(es[pool])
pool_es = np.concatenate(pool_es)
p1, p50, p99 = np.percentile(pool_es, [1, 50, 99])

# ---- recommended params: center at band midpoint, gain/scale spreads p1..p99 ----
center = float((p1 + p99) / 2.0)
# logit(TARGET_HI) - logit(TARGET_LO) = k*(p99 - p1)  ->  k = 2*ln(19)/(p99-p1) for [0.05,0.95]
span = math.log(TARGET_HI / (1 - TARGET_HI)) - math.log(TARGET_LO / (1 - TARGET_LO))
k = float(span / max(p99 - p1, 1e-6))
scale = float(1.0 / k)        # so gain=1.0, z = (x-center)/scale
gain = 1.0

print("=" * 64)
print(f"CALIBRATION [{VARIANT}] — realistic centered end->start cosine band")
print("=" * 64)
print(f"  pool: top-{POOL} per dest x {N_DESTS} dests = {len(pool_es)} edges")
print(f"  band p1/p50/p99 = {p1:.3f} / {p50:.3f} / {p99:.3f}   (min {pool_es.min():.3f}, max {pool_es.max():.3f})")
print()
print("  RECOMMENDED transition_calibration:")
print(f"    center = {center:.4f}")
print(f"    scale  = {scale:.4f}")
print(f"    gain   = {gain:.4f}        (gain/scale = {k:.2f})")
print(f"    -> r(p1)={_calibrate_transition_cos(p1, center=center, scale=scale, gain=gain):.3f}  "
      f"r(p50)={_calibrate_transition_cos(p50, center=center, scale=scale, gain=gain):.3f}  "
      f"r(p99)={_calibrate_transition_cos(p99, center=center, scale=scale, gain=gain):.3f}")


def blended_T(es, mm, ff):
    r = lambda x: np.array([_calibrate_transition_cos(float(v), center=center, scale=scale, gain=gain) for v in x])
    return W[0] * r(es) + W[1] * r(mm) + W[2] * r(ff)


allT = np.concatenate([blended_T(es, mm, ff) for (es, mm, ff) in pools])
print()
print("  NEW blended T distribution over realistic pools:")
for q in (1, 10, 25, 50, 75, 90, 99):
    print(f"    p{q:<2d} = {np.percentile(allT, q):.3f}")

print()
print("=" * 64)
print("FLOOR SWEEP — fraction of realistic candidate edges rejected")
print("=" * 64)
print(f"  {'floor':>6s}  {'rejected':>9s}")
for floor in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    print(f"  {floor:>6.2f}  {float((allT < floor).mean()):>8.1%}")
print()
print("  Pick transition_floor to reject roughly the bottom decile of realistic")
print("  edges (start ~p10); set bridge_floor similarly. Confirm via probe + audition.")
