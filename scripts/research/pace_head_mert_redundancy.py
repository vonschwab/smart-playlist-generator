"""Does each Essentia head add info BEYOND MERT?

Leave-one-out k-NN in MERT (emb_mid, cosine) space: predict each head as the mean
of its MERT-nearest neighbors, correlate predicted vs actual. HIGH corr = MERT
already captures the head (redundant, don't add); LOW corr = independent (adds).
Uses the existing 158-track probe + mert_sidecar (no new extraction).
"""
import numpy as np

ROOT = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3"
PROBE = f"{ROOT}/docs/run_audits/pace_axis_eval/head_probe.tsv"
MERT = f"{ROOT}/data/artifacts/beat3tower_32k/mert_sidecar.npz"
HEADS = ["arousal", "danceability", "aggressive", "relaxed", "electronic", "acoustic", "instrumental"]
K = 10

rows = [ln.rstrip("\n").split("\t") for ln in open(PROBE, encoding="utf-8")]
hdr = rows[0]
recs = [dict(zip(hdr, r)) for r in rows[1:] if r[1] != "ERR"]
ids = [r["track_id"] for r in recs]
H = np.array([[float(r[h]) for h in HEADS] for r in recs])  # (n, heads)

z = np.load(MERT, allow_pickle=True)
mert_ids = [str(t) for t in z["track_ids"]]
pos = {t: i for i, t in enumerate(mert_ids)}
emb = np.asarray(z["emb_mid"], dtype=float)
keep = [i for i, t in enumerate(ids) if t in pos]
ids = [ids[i] for i in keep]
H = H[keep]
M = np.array([emb[pos[t]] for t in ids])
M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)  # unit-norm -> cosine via dot
n = len(ids)

sim = M @ M.T
np.fill_diagonal(sim, -np.inf)  # exclude self
nbr = np.argsort(-sim, axis=1)[:, :K]  # top-K MERT neighbors

print(f"MERT-redundancy via leave-one-out k-NN (k={K}, cosine), N={n} corpus tracks")
print("corr(predicted_from_MERT_neighbors, actual): HIGH=MERT captures it (redundant), LOW=adds\n")
print(f"{'head':14}{'kNN_corr':>9}{'verdict':>26}")
for hi, h in enumerate(HEADS):
    pred = np.array([H[nbr[i], hi].mean() for i in range(n)])
    r = float(np.corrcoef(pred, H[:, hi])[0, 1])
    if r >= 0.7:
        v = "REDUNDANT w/ MERT"
    elif r >= 0.5:
        v = "partly redundant"
    else:
        v = "INDEPENDENT (adds)"
    print(f"{h:14}{r:9.2f}{v:>26}")
