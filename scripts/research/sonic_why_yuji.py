"""Why was a Ghibli cue EVER a sonic neighbor of 'Heaven's on Fire'? Two parts:
(1) POOL: which Yuji track is MERT-close to RD, and what interpretable qualities
    (tempo/onset/energy/duration) does MERT think they share?
(2) EDGE: the actual placed edge Song-of-Baron -> Heaven's is raw-WEAK — show the
    raw per-clip cosine the beam centered. Read-only.
"""
import io
import zipfile
import numpy as np
from pathlib import Path

A = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz")
with zipfile.ZipFile(A) as z:
    keys = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
print("feature-ish artifact keys:",
      [k for k in keys if any(t in k.lower() for t in ("bpm", "onset", "tempo", "energy", "dur", "loud", "arous"))])

a = np.load(A, allow_pickle=True)
artists = np.array([str(x).strip().lower() for x in a["track_artists"]])
titles = np.array([str(x) for x in a["track_titles"]])


def col(*names):
    for n in names:
        if n in a.files:
            return np.asarray(a[n]).reshape(len(artists), -1)[:, 0]
    return None


bpm = col("bpm", "tempo", "track_bpm")
onset = col("onset_rate", "onset", "onsets")
dur = col("durations_ms", "duration_ms", "durations")
for variant in ("X_sonic_mert", "X_sonic_mert_start", "X_sonic_mert_mid", "X_sonic_mert_end"):
    pass
mert = np.asarray(a["X_sonic_mert"], np.float32)
mstart = np.asarray(a["X_sonic_mert_start"], np.float32) if "X_sonic_mert_start" in a.files else mert
mend = np.asarray(a["X_sonic_mert_end"], np.float32) if "X_sonic_mert_end" in a.files else mert


def nrm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


mn, mns, mne = nrm(mert), nrm(mstart), nrm(mend)
rd = np.where(artists == "the radio dept.")[0]


def find(asub, tsub=""):
    for i in range(len(artists)):
        if asub in artists[i] and tsub.lower() in titles[i].lower():
            return i
    return None


sob = find("yuji nomi", "baron")
hof = find("the radio dept.", "heaven")
# Yuji's BEST MERT match to RD (the pool-admission track) + which RD track
yuji = np.where(np.char.find(artists, "yuji nomi") >= 0)[0]
pairs = [(float(mn[i] @ mn[r]), int(i), int(r)) for i in yuji for r in rd]
bestsim, byuji, brd = max(pairs)

print("\n(1) POOL — Yuji's strongest MERT tie to RD:")
print(f"    '{titles[byuji]}' (Yuji)  ~  '{titles[brd]}' (RD)   whole-track MERT cos = {bestsim:.3f}")

print("\n(2) EDGE — the cue that was actually placed next to Heaven's on Fire:")
print(f"    whole-track  cos(Song-of-Baron, Heaven's) = {float(mn[sob] @ mn[hof]):.3f}")
print(f"    beam edge    cos(end[Song-of-Baron], start[Heaven's]) = {float(mne[sob] @ mns[hof]):.3f}")

print("\nInterpretable qualities (what MERT may be latching onto):")
hdr = f"  {'track':40s}"
if bpm is not None: hdr += f"{'bpm':>7s}"
if onset is not None: hdr += f"{'onset':>8s}"
if dur is not None: hdr += f"{'dur_s':>7s}"
print(hdr)
for lbl, i in [("Yuji - " + titles[byuji], byuji), ("Yuji - The Song of Baron", sob),
               ("RD - " + titles[brd], brd), ("RD - Heaven's on Fire", hof),
               ("Slowdive (best→RD)", int(max((float(mn[s] @ mn[r]), int(s)) for s in np.where(artists == "slowdive")[0] for r in rd)[1]))]:
    line = f"  {lbl[:40]:40s}"
    if bpm is not None: line += f"{bpm[i]:7.1f}"
    if onset is not None: line += f"{onset[i]:8.3f}"
    if dur is not None: line += f"{dur[i] / 1000:7.0f}"
    print(line)
