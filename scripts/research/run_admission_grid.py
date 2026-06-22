"""In-process calibration grid for adaptive candidate-pool admission.

Loads the artifact ONCE (lru_cache amortizes across cells) and iterates a tight
grid of (mode, niche, percentile combo). Streams one JSON object per cell to
grid_results.jsonl so progress is durable/visible. Far cheaper than a workflow
that would pay the ~280MB MERT artifact load per process.

Per (mode, niche): a `baseline` cell (no percentile → current absolute-floor
behavior) anchors the worst-edge eval-gate; candidate cells are compared to it.
"""
import sys
import os
import json
import time

ROOT = r"C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/.claude/worktrees/v6-canonical-wiring"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from scripts.research.adaptive_admission_eval import measure_cell, SEED_CORPUS  # noqa: E402

OUT = os.path.join(ROOT, "docs/run_audits/adaptive_admission/grid_results.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

MIN_POOL = {"strict": 12, "narrow": 16, "dynamic": 20}
NICHES = ["hyperpop", "jazz", "metal"]
# (label, sonic_pct, genre_pct)
COMBOS = [("baseline", None, None), ("medium", 0.70, 0.80), ("loose", 0.55, 0.65)]
CALIB_MODES = ["strict", "narrow"]

cells = []
for mode in CALIB_MODES:
    for niche in NICHES:
        for (label, sp, gp) in COMBOS:
            mp = None if label == "baseline" else MIN_POOL[mode]
            cells.append((mode, niche, label, sp, gp, mp))
# dynamic confirmation (already healthy in the verified cell): baseline + preset
for niche in ["hyperpop", "jazz"]:
    cells.append(("dynamic", niche, "baseline", None, None, None))
    cells.append(("dynamic", niche, "preset", 0.40, 0.85, MIN_POOL["dynamic"]))

print(f"GRID: {len(cells)} cells", flush=True)
open(OUT, "w").close()
for i, (mode, niche, label, sp, gp, mp) in enumerate(cells):
    seeds = SEED_CORPUS[niche]
    t0 = time.time()
    try:
        res = measure_cell(mode, seeds, sp, gp, mp)
        res.update({"niche": niche, "label": label})
        status = "ok"
    except Exception as e:  # one bad cell must not kill the grid
        res = {
            "mode": mode, "niche": niche, "label": label,
            "sonic_pct": sp, "genre_pct": gp, "min_pool": mp,
            "error": repr(e)[:400],
        }
        status = "ERROR"
    res["cell_wall_s"] = round(time.time() - t0, 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(
        f"[{i + 1}/{len(cells)}] {mode}/{niche}/{label} sp={sp} gp={gp} -> {status} "
        f"admitted={res.get('admitted')} distinct={res.get('distinct_artists')} "
        f"worst={res.get('worst_edge_sonic')} wall={res.get('wall_time_s', res.get('cell_wall_s'))}s",
        flush=True,
    )

print("GRID DONE", flush=True)
