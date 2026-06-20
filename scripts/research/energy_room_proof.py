"""Does energy bite when the OTHER gates are loosened (pool has room)?

Same A/B but genre_mode=off, sonic_mode=off so hard genre/sonic floors don't
pre-determine the path. If energy-on now changes the playlist, the earlier
0/20 was 'hard gates left energy no room', not a discard bug.
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.support.gui_fidelity import generate_like_gui  # noqa: E402

SEEDS = [
    "49f8bba75408d4e0e0e000d1dc708add",
    "b587eb56fa1e173138152bf09565eb80",
    "f28fd5cebac845cf64fee59d5ac3b3aa",
]
TMP = "config.energy_room.tmp.yaml"
SIDECAR = "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
ON = {"energy_step_cap": 0.1, "energy_step_strength": 10.0,
      "energy_arc_band": 0.1, "energy_arc_strength": 10.0}


def _z_arousal_map():
    z = np.load(SIDECAR, allow_pickle=True)
    ids = list(z["track_ids"])
    a = np.asarray(z["arousal_p50"], dtype=float)
    f = np.isfinite(a)
    zr = (a - a[f].mean()) / (a[f].std() or 1.0)
    return {t: float(zr[i]) for i, t in enumerate(ids)}


def _cfg(energy):
    with open("config.yaml", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    pb = c.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    for k in ("energy_step_cap", "energy_step_strength", "energy_arc_band", "energy_arc_strength"):
        pb[k] = (energy or {}).get(k, 0.0)
    with open(TMP, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f)
    return TMP


def _run(label, energy, amap):
    t0 = time.time()
    res = generate_like_gui(
        seeds=SEEDS, config_path=_cfg(energy), length=12,
        genre_mode="off", sonic_mode="off", pace_mode="dynamic", cohesion_mode="dynamic",
    )
    ids = list(res.track_ids)
    curve = [round(amap.get(t, float("nan")), 2) for t in ids]
    print(f"{label} ({time.time()-t0:.0f}s): {curve}")
    return ids


def main():
    amap = _z_arousal_map()
    off = _run("OFF ", None, amap)
    on = _run("ON  ", ON, amap)
    Path(TMP).unlink(missing_ok=True)
    diff = sum(1 for a, b in zip(off, on) if a != b)
    print(f"=> differ at {diff}/{min(len(off), len(on))} positions")
    print("VERDICT:", "energy bites when pools have room → earlier 0/20 was hard-gate constraint"
          if diff else "STILL inert with loose pools → real discard bug, keep digging")


if __name__ == "__main__":
    main()
