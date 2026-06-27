"""Did the CENTERED transition score promote the weak 'Song of Baron' -> 'Heaven's
on Fire' edge? Rank every candidate as a predecessor of Heaven's, RAW vs CENTERED,
using the real build_transition_metric_context / scoring math. If Song-of-Baron's
rank jumps up under centering, the centering manufactured the edge. Read-only.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.playlist.transition_metrics import build_transition_metric_context

A = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz")
a = np.load(A, allow_pickle=True)
artists = np.array([str(x).strip().lower() for x in a["track_artists"]])
titles = np.array([str(x) for x in a["track_titles"]])
mert = np.asarray(a["X_sonic_mert"], np.float32)
mst = np.asarray(a["X_sonic_mert_start"], np.float32)
mmd = np.asarray(a["X_sonic_mert_mid"], np.float32)
men = np.asarray(a["X_sonic_mert_end"], np.float32)


def ctx(center):
    return build_transition_metric_context(
        X_sonic=mert, X_start=mst, X_mid=mmd, X_end=men,
        center_transitions=center, transition_weights=(0.2, 0.5, 0.3),
        sonic_variant="mert", weight_end_start=0.7, weight_mid_mid=0.15, weight_full_full=0.15,
    )


cR, cC = ctx(False), ctx(True)


def vec_T(c, dst):
    es = c.X_end @ c.X_start[dst]
    mm = c.X_mid @ c.X_mid[dst]
    ff = c.X_full @ c.X_full[dst]
    if c.center_transitions:
        r = lambda x: np.clip((x + 1.0) / 2.0, 0.0, 1.0)
        return 0.7 * r(es) + 0.15 * r(mm) + 0.15 * r(ff)
    return 0.7 * es + 0.15 * mm + 0.15 * ff


def find(asub, tsub=""):
    for i in range(len(artists)):
        if asub in artists[i] and tsub.lower() in titles[i].lower():
            return i
    return None


hof = find("the radio dept.", "heaven")
sob = find("yuji nomi", "baron")
TR, TC = vec_T(cR, hof), vec_T(cC, hof)
TR[hof] = -9; TC[hof] = -9  # exclude self


def rank(arr, i):
    return int(np.sum(arr > arr[i]))


print(f"destination = Heaven's on Fire (idx {hof})\n")
print(f"{'predecessor':34s} {'rawT':>6s} {'rawRank':>8s}   {'centT':>6s} {'centRank':>9s}")
cands = [("Yuji - The Song of Baron", sob),
         ("Slowdive (best raw→Heaven)", int(np.argmax(np.where(artists == "slowdive", TR, -9)))),
         ("Beach House (best raw→Heaven)", int(np.argmax(np.where(artists == "beach house", TR, -9)))),
         ("RD self-neighbour (best raw)", int(np.argmax(np.where(artists == "the radio dept.", TR, -9))))]
for lbl, i in cands:
    print(f"{lbl:34s} {TR[i]:6.3f} {rank(TR, i):8d}   {TC[i]:6.3f} {rank(TC, i):9d}")

print("\n--- distribution shift (how centering compresses the field) ---")
print(f"  raw T  : p50={np.percentile(TR[TR>-1],50):.3f}  p90={np.percentile(TR[TR>-1],90):.3f}  max={TR.max():.3f}")
print(f"  cent T : p50={np.percentile(TC[TC>-1],50):.3f}  p90={np.percentile(TC[TC>-1],90):.3f}  max={TC.max():.3f}")
print(f"\n  Song-of-Baron percentile among all predecessors:  "
      f"raw={100*(1-rank(TR,sob)/len(TR)):.1f}%   centered={100*(1-rank(TC,sob)/len(TC)):.1f}%")
