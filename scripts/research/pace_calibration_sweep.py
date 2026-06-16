"""Full-pool onset/BPM band calibration. Read-only. Writes docs/run_audits/pace_retune/.

Pre-flight:
  - Full pool against beat3tower_32k artifact (~40k tracks).
  - No writes to production artifact paths.
  - Distributions not means; ambient AND rhythmic seeds reported.
"""
import json
from pathlib import Path

import numpy as np

from src.features.artifacts import load_artifact_bundle
from src.playlist.bpm_loader import load_bpm_arrays
from src.playlist.bpm_axis import bpm_log_distance

ART = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
DB = "data/metadata.db"
OUT = Path("docs/run_audits/pace_retune")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading artifact…")
bundle = load_artifact_bundle(ART)
print(f"  {len(bundle.track_ids)} tracks")

print("Loading BPM + onset from DB…")
arrs = load_bpm_arrays(bundle.track_ids, db_path=DB)
onset = arrs["onset_rate"]
bpm = arrs["perceptual_bpm"]
artists = np.array([str(a) for a in bundle.track_artists])

print(f"  BPM coverage:   {int(np.sum(~np.isnan(bpm)))}/{len(bpm)}")
print(f"  Onset coverage: {int(np.sum(~np.isnan(onset)))}/{len(onset)}")

CAPS = [0.30, 0.40, 0.50, 0.60, 0.75, 0.85, 1.00, 1.35]
AMBIENT = ["Green-House", "Hiroshi Yoshimura", "Stars of the Lid", "Brian Eno"]
RHYTHMIC = ["J Dilla", "De La Soul", "Beastie Boys", "Kendrick Lamar"]


def pass_rate(metric_arr: np.ndarray, seed_idx: np.ndarray, cap: float) -> float:
    """Mean fraction of library within cap log-distance of any seed track."""
    seed_idx = seed_idx[~np.isnan(metric_arr[seed_idx])]
    if len(seed_idx) == 0:
        return float("nan")
    rates = []
    for s in seed_idx:
        d = bpm_log_distance(metric_arr, float(metric_arr[s]))
        rates.append(float(np.nanmean(d <= cap)))
    return float(np.mean(rates))


rng = np.random.default_rng(0)
rand_idx = rng.choice(len(onset), 200, replace=False)

report = {
    "N": int(len(onset)),
    "onset_coverage": float(np.mean(~np.isnan(onset))),
    "bpm_coverage": float(np.mean(~np.isnan(bpm))),
    "onset_percentiles_library": {
        "p10": float(np.nanpercentile(onset, 10)),
        "p50": float(np.nanpercentile(onset, 50)),
        "p90": float(np.nanpercentile(onset, 90)),
    },
    "bpm_percentiles_library": {
        "p10": float(np.nanpercentile(bpm, 10)),
        "p50": float(np.nanpercentile(bpm, 50)),
        "p90": float(np.nanpercentile(bpm, 90)),
    },
}

for label, arr in (("onset", onset), ("bpm", bpm)):
    report[label] = {}
    for cap in CAPS:
        row: dict = {"library_random200": pass_rate(arr, rand_idx, cap)}
        for artist in AMBIENT + RHYTHMIC:
            idx = np.where(artists == artist)[0]
            row[artist] = pass_rate(arr, idx, cap) if len(idx) else None
        report[label][f"cap_{cap}"] = row

json_out = OUT / "index.json"
json_out.write_text(json.dumps(report, indent=2))
print(f"\nResults written to {json_out}")
print(json.dumps(report, indent=2))
