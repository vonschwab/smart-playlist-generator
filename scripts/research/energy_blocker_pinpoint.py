"""Which gate blocks energy: sonic or genre?

Two arms, each an energy off-vs-on A/B at length 12 with strong energy:
  ARM 1  genre=dynamic, sonic=off   -> if energy BITES, SONIC was the blocker
                                        (energy works with genre still on) -> user's
                                        'outrank sonic, keep genre' is ACHIEVABLE
  ARM 2  genre=off, sonic=dynamic   -> if energy BITES, GENRE was the blocker

If energy is inert in ARM 1 (genre on, sonic off), genre alone blocks energy and
'outrank sonic only' will NOT be enough.
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
TMP = "config.blocker.tmp.yaml"
SIDECAR = "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
ON = {"energy_step_cap": 0.1, "energy_step_strength": 10.0,
      "energy_arc_band": 0.1, "energy_arc_strength": 10.0}


def _z():
    z = np.load(SIDECAR, allow_pickle=True)
    ids = list(z["track_ids"]); a = np.asarray(z["arousal_p50"], float)
    f = np.isfinite(a); zr = (a - a[f].mean()) / (a[f].std() or 1.0)
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


def _run(energy, genre_mode, sonic_mode, amap):
    res = generate_like_gui(seeds=SEEDS, config_path=_cfg(energy), length=12,
                            genre_mode=genre_mode, sonic_mode=sonic_mode,
                            pace_mode="dynamic", cohesion_mode="dynamic")
    ids = list(res.track_ids)
    return ids, [round(amap.get(t, float("nan")), 2) for t in ids]


def arm(name, genre_mode, sonic_mode, amap, blocker_if_bites):
    off_ids, off_c = _run(None, genre_mode, sonic_mode, amap)
    on_ids, on_c = _run(ON, genre_mode, sonic_mode, amap)
    diff = sum(1 for a, b in zip(off_ids, on_ids) if a != b)
    Path(TMP).unlink(missing_ok=True)
    print(f"\n{name} (genre={genre_mode}, sonic={sonic_mode})")
    print(f"  OFF: {off_c}")
    print(f"  ON : {on_c}")
    print(f"  diff {diff}/{min(len(off_ids), len(on_ids))} -> "
          f"{'energy BITES: ' + blocker_if_bites + ' was the blocker' if diff else 'energy INERT: this gate is NOT the blocker'}")
    return diff


def main():
    amap = _z()
    t0 = time.time()
    arm("ARM 1", "dynamic", "off", amap, "SONIC")
    arm("ARM 2", "off", "dynamic", amap, "GENRE")
    print(f"\n(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
