# scripts/pace_audition_analyze.py
"""Un-blind the pace audition captures and write findings.

Reads pace_capture.yaml + pace_manifest.json (server-side edge_data), reports
per-arm continuity/smoothness distributions sliced by regime, runs the
discrimination check (decoy lowest), the narrow-vs-dynamic-vs-off contrasts,
the pace-specificity confound check, and the structural onset-variance check.

Usage:
    python scripts/pace_audition_analyze.py [--data-dir docs/run_audits/pace_audition]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ["narrow", "dynamic", "off", "decoy"]


def distribution(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "p10": None, "p50": None, "p90": None}
    arr = np.array(vals)
    return {
        "n": len(vals),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def join_scores(captures: List[dict], edge_data: Dict[str, dict]) -> List[dict]:
    joined = []
    for c in captures:
        meta = edge_data.get(c.get("edge_id", ""))
        if not meta:
            continue
        joined.append({
            "edge_id": c["edge_id"],
            "arm": meta["arm"], "seed": meta["seed"], "regime": meta.get("regime", "?"),
            "continuity": c.get("continuity"), "smoothness": c.get("smoothness"),
            "onset_log_dist": meta.get("onset_log_dist"), "notes": c.get("notes", ""),
        })
    return joined


def per_arm(joined: List[dict], metric: str, regime: Optional[str] = None) -> Dict[str, dict]:
    buckets: Dict[str, list] = defaultdict(list)
    for j in joined:
        if regime and j["regime"] != regime:
            continue
        if j.get(metric) is not None:
            buckets[j["arm"]].append(j[metric])
    return {arm: distribution(buckets.get(arm, [])) for arm in ARMS}


def discrimination_ok(continuity_by_arm: Dict[str, dict]) -> bool:
    decoy = continuity_by_arm.get("decoy", {}).get("p50")
    if decoy is None:
        return False
    reals = [continuity_by_arm.get(a, {}).get("p50") for a in ("narrow", "dynamic", "off")]
    reals = [r for r in reals if r is not None]
    return bool(reals) and all(decoy < r for r in reals)


def _mean(joined, arm, metric):
    vals = [j[metric] for j in joined if j["arm"] == arm and j.get(metric) is not None]
    return float(np.mean(vals)) if vals else None


def confound_flag(joined: List[dict]) -> dict:
    cn, cd = _mean(joined, "narrow", "continuity"), _mean(joined, "dynamic", "continuity")
    sn, sd = _mean(joined, "narrow", "smoothness"), _mean(joined, "dynamic", "smoothness")
    if None in (cn, cd, sn, sd):
        return {"pace_specific": None, "continuity_gain": None, "smoothness_gain": None}
    cg, sg = cn - cd, sn - sd
    return {"pace_specific": bool(cg > sg), "continuity_gain": cg, "smoothness_gain": sg}
