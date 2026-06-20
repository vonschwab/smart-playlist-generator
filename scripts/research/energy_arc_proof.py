"""Proper A/B: does the ARC-BAND term actually shape the arousal curve?

Holds pace_mode constant (dynamic) and flips ONLY the energy terms via the
config pier_bridge overrides. Tests the arc-band (the shaping lever), not just
the step-cap. Reads back the realized arousal_p50 curve for each playlist and
reports arc deviation + max adjacent whiplash.

No global INFO logging (that inflated earlier timings).
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.support.gui_fidelity import generate_like_gui  # noqa: E402

SEEDS = [
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler
]
STOCK = "config.yaml"
TMP = "config.energy_arc.tmp.yaml"
SIDECAR = "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"

# Strict-like energy values chosen to BITE (arc is the shaping lever).
ENERGY_ON = {
    "energy_step_cap": 0.25,
    "energy_step_strength": 0.4,
    "energy_arc_band": 0.25,
    "energy_arc_strength": 0.5,
}


def _z_arousal_map():
    z = np.load(SIDECAR, allow_pickle=True)
    ids = list(z["track_ids"])
    a = np.asarray(z["arousal_p50"], dtype=float)
    finite = np.isfinite(a)
    mu, sd = a[finite].mean(), a[finite].std()
    zr = (a - mu) / (sd if sd > 0 else 1.0)
    return {tid: float(zr[i]) for i, tid in enumerate(ids)}


def _make_config(energy: dict | None) -> str:
    with open(STOCK, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pb = cfg.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    for k in ("energy_step_cap", "energy_step_strength", "energy_arc_band", "energy_arc_strength"):
        pb[k] = (energy or {}).get(k, 0.0)
    with open(TMP, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return TMP


def _curve(ids, amap):
    return [round(amap.get(t, float("nan")), 2) for t in ids]


def _metrics(curve):
    arr = np.array([c for c in curve if c == c])  # drop NaN
    if len(arr) < 2:
        return None
    steps = np.abs(np.diff(arr))
    # arc deviation = RMS distance from the straight line first->last
    line = np.linspace(arr[0], arr[-1], len(arr))
    arc_dev = float(np.sqrt(np.mean((arr - line) ** 2)))
    return {"max_step": float(steps.max()), "mean_step": float(steps.mean()), "arc_dev": arc_dev}


def _run(label, cfg_path, amap):
    t0 = time.time()
    res = generate_like_gui(seeds=SEEDS, config_path=cfg_path, pace_mode="dynamic", length=20)
    dt = time.time() - t0
    ids = list(res.track_ids)
    curve = _curve(ids, amap)
    m = _metrics(curve)
    print(f"\n{label}  ({dt:.0f}s, {len(ids)} tracks)")
    print(f"  arousal curve: {curve}")
    print(f"  metrics: {m}")
    return ids, curve


def main():
    amap = _z_arousal_map()
    off_ids, _ = _run("ENERGY OFF", _make_config(None), amap)
    on_ids, _ = _run("ENERGY ON (arc engaged)", _make_config(ENERGY_ON), amap)
    Path(TMP).unlink(missing_ok=True)
    diff = sum(1 for a, b in zip(off_ids, on_ids) if a != b)
    print(f"\n=> playlists differ at {diff}/{min(len(off_ids), len(on_ids))} positions")


if __name__ == "__main__":
    main()
