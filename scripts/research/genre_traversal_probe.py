"""Probe: why don't The Radio Dept. and The Embassy co-occur?

Hypothesis (user): they are sonically similar but the genre space (tags) separates
them — Radio Dept is tagged shoegaze, Embassy is not — so the genre gate / genre-
neighbor pool excludes each from the other's neighborhood at pool construction.

Evidence gathered: sonic (MERT) vs genre (raw + dense PMI-SVD) cross-similarity,
the actual top tags, and a sanity baseline (how the RD<->Emb sonic sim compares to
RD's sim to bands that DID make its playlist).
"""
import numpy as np
from pathlib import Path

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
a = np.load(ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz", allow_pickle=True)
print("artifact keys:", list(a.files))

ids = [str(x) for x in a["track_ids"]]
artists = [str(x) for x in a["track_artists"]] if "track_artists" in a.files else None
mert = np.asarray(a["X_sonic_mert"], np.float32)
graw = np.asarray(a["X_genre_raw"], np.float32)
vocab = [str(x) for x in a["genre_vocab"]]
gdense = np.asarray(a["X_genre_dense"], np.float32) if "X_genre_dense" in a.files else None
print(f"N={len(ids)} mert={mert.shape} graw={graw.shape} dense={None if gdense is None else gdense.shape}")


def norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


mn, gn = norm(mert), norm(graw)
dn = norm(gdense) if gdense is not None else None


def idxs(name):
    return [i for i, ar in enumerate(artists) if ar.strip().lower() == name.strip().lower()]


def cross(Mn, A, B):
    return float(np.mean(Mn[A] @ Mn[B].T)) if A and B else float("nan")


def toptags(A, k=12):
    v = graw[A].sum(0)
    order = np.argsort(-v)
    return [(vocab[i], round(float(v[i]), 2)) for i in order[:k] if v[i] > 0]


rd = idxs("The Radio Dept.")
emb = idxs("The Embassy")
print(f"\nRadio Dept tracks={len(rd)}  Embassy tracks={len(emb)}")
print("Radio Dept top tags:", toptags(rd))
print("Embassy    top tags:", toptags(emb))

print("\n=== RD <-> Embassy cross-similarity ===")
print(f"  SONIC (MERT):   {cross(mn, rd, emb):.3f}")
print(f"  GENRE raw:      {cross(gn, rd, emb):.3f}")
if dn is not None:
    print(f"  GENRE dense:    {cross(dn, rd, emb):.3f}   (gate floors were RD=0.66, Emb=0.456)")

# Baseline: how does RD<->Emb sonic compare to RD<->(bands that DID make RD's list)?
print("\n=== Baseline: RD sonic sim to bands that DID make its playlist vs to Embassy ===")
for name in ["Beach House", "Slowdive", "Cocteau Twins", "Ride", "Beach Fossils", "The Embassy"]:
    B = idxs(name)
    if not B:
        print(f"  {name:22s} (not found by exact artist match)")
        continue
    s = cross(mn, rd, B)
    g = cross(dn, rd, B) if dn is not None else float("nan")
    print(f"  {name:22s} sonic={s:.3f}  dense_genre={g:.3f}")

# Embassy's view: its sonic sim to bands that made ITS list vs to Radio Dept
print("\n=== Baseline: Embassy sonic sim to bands that DID make its playlist vs to RD ===")
for name in ["Tennis", "Wild Nothing", "Hoops", "Club 8", "The Radio Dept."]:
    B = idxs(name)
    if not B:
        print(f"  {name:22s} (not found)")
        continue
    s = cross(mn, emb, B)
    g = cross(dn, emb, B) if dn is not None else float("nan")
    print(f"  {name:22s} sonic={s:.3f}  dense_genre={g:.3f}")
