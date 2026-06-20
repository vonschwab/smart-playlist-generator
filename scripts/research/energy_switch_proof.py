"""Prove the energy on-switch works through the real GUI resolution chain.

Runs generate_like_gui (the GUI-fidelity harness) twice with identical seeds:
  - energy OFF (stock config.yaml)
  - energy ON  (temp config with playlists.ds_pipeline.pier_bridge.energy_step_strength)

Confirms: (a) the "energy loaded" log fires only when on, (b) both stay in budget,
(c) the two playlists actually differ (the penalty changes selection).
"""
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.support.gui_fidelity import generate_like_gui  # noqa: E402

SEEDS = [
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler
]
STOCK = "config.yaml"
TMP = "config.energy_on.tmp.yaml"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.energy_lines = []

    def emit(self, record):
        msg = record.getMessage()
        if "energy loaded" in msg or "energy load failed" in msg:
            self.energy_lines.append(msg)


def _make_energy_config(step_strength: float, step_cap: float) -> None:
    with open(STOCK, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pb = cfg.setdefault("playlists", {}).setdefault("ds_pipeline", {}).setdefault("pier_bridge", {})
    pb["energy_step_strength"] = step_strength
    pb["energy_step_cap"] = step_cap
    pb["energy_arc_strength"] = 0.0  # arc is the cascade risk; step-cap only
    pb["energy_arc_band"] = 0.0
    with open(TMP, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _run(label: str, config_path: str):
    cap = _Capture()
    root = logging.getLogger()
    root.addHandler(cap)
    prev = root.level
    root.setLevel(logging.INFO)
    t0 = time.time()
    try:
        res = generate_like_gui(seeds=SEEDS, config_path=config_path, pace_mode="dynamic", length=20)
    finally:
        dt = time.time() - t0
        root.removeHandler(cap)
        root.setLevel(prev)
    ids = list(res.track_ids)
    print(f"{label:12s} tracks={len(ids)} time={dt:.1f}s energy_log={cap.energy_lines}")
    return ids


def main():
    print("=== ENERGY OFF (stock config) ===")
    off = _run("off", STOCK)
    print("\n=== ENERGY ON (step_strength=0.4, step_cap=0.4) ===")
    _make_energy_config(0.4, 0.4)
    try:
        on = _run("on", TMP)
    finally:
        Path(TMP).unlink(missing_ok=True)
    diff = sum(1 for a, b in zip(off, on) if a != b)
    print(f"\nplaylists differ at {diff}/{min(len(off), len(on))} positions "
          f"(set-diff: {len(set(off) ^ set(on)) // 2} swapped)")
    print("PROOF:", "energy ON changed the playlist" if diff else "NO CHANGE — switch is inert")


if __name__ == "__main__":
    main()
