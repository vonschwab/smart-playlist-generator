"""Decisive diagnostic: is compute_energy_pace_penalty actually called during a
real generation, and does it return non-zero penalties?

Monkeypatches the function to count calls / non-zero returns / max penalty,
then runs ONE energy-on generation through the GUI-fidelity chain.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.playlist.pier_bridge.pace_gate as pace_gate  # noqa: E402
from tests.support.gui_fidelity import generate_like_gui  # noqa: E402

SEEDS = [
    "49f8bba75408d4e0e0e000d1dc708add",
    "b587eb56fa1e173138152bf09565eb80",
    "f28fd5cebac845cf64fee59d5ac3b3aa",
]
TMP = "config.energy_instr.tmp.yaml"

stats = {"calls": 0, "nonzero": 0, "max": 0.0, "matrix_none": 0}
_orig = pace_gate.compute_energy_pace_penalty


def _wrapped(energy_matrix, **kw):
    stats["calls"] += 1
    if energy_matrix is None:
        stats["matrix_none"] += 1
    p = _orig(energy_matrix, **kw)
    if p > 0:
        stats["nonzero"] += 1
        stats["max"] = max(stats["max"], p)
    return p


def main():
    pace_gate.compute_energy_pace_penalty = _wrapped
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pb = cfg.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    pb.update({"energy_step_cap": 0.25, "energy_step_strength": 0.4,
               "energy_arc_band": 0.25, "energy_arc_strength": 0.5})
    with open(TMP, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    try:
        res = generate_like_gui(seeds=SEEDS, config_path=TMP, pace_mode="dynamic", length=20)
        print(f"tracks={len(res.track_ids)}")
    finally:
        Path(TMP).unlink(missing_ok=True)
    print("ENERGY PENALTY STATS:", stats)
    if stats["calls"] == 0:
        print("VERDICT: penalty function NEVER CALLED — energy_matrix not threaded to the beam path used")
    elif stats["matrix_none"] == stats["calls"]:
        print("VERDICT: called but energy_matrix is None every time — load/thread gap")
    elif stats["nonzero"] == 0:
        print("VERDICT: called with matrix but ALWAYS returns 0 — cfg strengths 0 in beam, or NaN rows")
    else:
        print(f"VERDICT: penalty IS firing (max={stats['max']:.3f}) — bug is downstream of the penalty")


if __name__ == "__main__":
    main()
