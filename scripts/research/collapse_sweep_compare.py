"""One-shot collapse comparison across several saved runs (SP2 strength sweep).

Prints, per run tag, the cross-seed CI + within-bridge sag (MuQ space) for both
clusters plus the quality floor (seed_sim / worst_edge), so a strength sweep reads
as a single table: does pushing A/B harder scale the per-face win while quality holds?

  python scripts/research/collapse_sweep_compare.py muq_gen_r3 muq_A_hub muq_A05 muq_A08 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.research.collapse_rescore import (  # noqa: E402
    collapse_loads,
    global_blur,
    load_art,
    sag_loads,
)

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")


def _quality(run):
    out = {}
    for cn, cd in run["clusters"].items():
        ss, we = [], []
        for s in cd["seeds"]:
            for r in cd["repeats"]:
                v = r["playlists"][s]
                if v["seed_sim"] is not None:
                    ss.append(v["seed_sim"])
                if v["worst_edge"] is not None:
                    we.append(v["worst_edge"])
        out[cn] = (float(np.mean(ss)) if ss else float("nan"),
                   float(np.mean(we)) if we else float("nan"))
    return out


def main() -> None:
    tags = sys.argv[1:]
    id2idx, artists, M = load_art()
    _ = global_blur(M)  # warm
    hdr = (f"{'tag':16s} | {'CI dp':>6s} {'CI el':>6s} | {'sag dp':>7s} {'sag el':>7s} | "
           f"{'dp seed/edge':>13s} | {'el seed/edge':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for tag in tags:
        p = ROOT / f"docs/run_audits/collapse/run_{tag}.json"
        if not p.exists():
            print(f"{tag:16s} | (missing)")
            continue
        run = json.loads(p.read_text(encoding="utf-8"))
        ci = collapse_loads(run, "muq", id2idx, artists, M)
        sg = sag_loads(run, "muq", id2idx, artists, M)
        q = _quality(run)
        cdp, cel = ci["dreampop_haze"]["load"], ci["electronic"]["load"]
        sdp = sg["dreampop_haze"]["norm"] * 100
        sel = sg["electronic"]["norm"] * 100
        qdp, qel = q["dreampop_haze"], q["electronic"]
        print(f"{tag:16s} | {cdp:6.3f} {cel:6.3f} | {sdp:6.0f}% {sel:6.0f}% | "
              f"{qdp[0]:.2f} / {qdp[1]:.2f}   | {qel[0]:.2f} / {qel[1]:.2f}")


if __name__ == "__main__":
    main()
