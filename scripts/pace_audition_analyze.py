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


def onset_variance_by_arm(playlists: List[dict]) -> Dict[str, float]:
    """Mean within-playlist onset-rate variance per arm (structural monotony
    proxy). Lower = flatter pace profile = monotony risk."""
    buckets: Dict[str, list] = defaultdict(list)
    for pl in playlists:
        seq = [v for v in pl.get("onset_seq", []) if v is not None]
        if len(seq) >= 2:
            buckets[pl["arm"]].append(float(np.var(seq)))
    return {arm: float(np.mean(v)) for arm, v in buckets.items() if v}


def load_capture(data_dir: Path) -> List[dict]:
    cap = data_dir / "pace_capture.yaml"
    if not cap.exists():
        return []
    data = yaml.safe_load(cap.read_text(encoding="utf-8")) or {}
    return data.get("entries", [])


def _fmt(d: dict) -> str:
    if d["n"] == 0:
        return "—"
    return f"n={d['n']} min={d['min']:.1f} p10={d['p10']:.1f} p50={d['p50']:.1f} p90={d['p90']:.1f}"


def write_findings(data_dir: Path, joined: List[dict], playlists: List[dict]) -> Path:
    cont_all = per_arm(joined, "continuity")
    smooth_all = per_arm(joined, "smoothness")
    disc = discrimination_ok(cont_all)
    conf = confound_flag(joined)
    ovar = onset_variance_by_arm(playlists)

    lines = [
        "# Pace Audition — Findings", "",
        f"Total rated edges: {len([j for j in joined if j.get('continuity') is not None])}", "",
        "## Continuity by arm (1-5)", "",
        "| Arm | overall | ambient | rhythmic |", "|---|---|---|---|",
    ]
    for arm in ARMS:
        amb = per_arm(joined, "continuity", "ambient")[arm]
        rhy = per_arm(joined, "continuity", "rhythmic")[arm]
        lines.append(f"| {arm} | {_fmt(cont_all[arm])} | {_fmt(amb)} | {_fmt(rhy)} |")

    lines += ["", "## Smoothness by arm (1-5)", "", "| Arm | overall |", "|---|---|"]
    for arm in ARMS:
        lines.append(f"| {arm} | {_fmt(smooth_all[arm])} |")

    lines += [
        "", "## Verdict", "",
        f"- **Discrimination check** (decoy rated worst on continuity): "
        f"{'PASS' if disc else 'FAIL — ratings not trustworthy'}",
    ]
    if conf["pace_specific"] is not None:
        lines.append(
            f"- **Pace-specific** (narrow continuity gain {conf['continuity_gain']:+.2f} "
            f"vs smoothness gain {conf['smoothness_gain']:+.2f}): "
            f"{'YES — win is pace, not incidental' if conf['pace_specific'] else 'NO — gain may be incidental'}"
        )
    lines += ["", "## Structural monotony (onset variance per arm; lower=flatter)", ""]
    for arm in ARMS:
        if arm in ovar:
            lines.append(f"- {arm}: {ovar[arm]:.3f}")
    lines += [
        "", "## Notes", "",
        *[f"- [{j['arm']}] {j['notes']}" for j in joined if j.get("notes")],
        "", "## Caveats",
        "- Single listener, one library; directional not conclusive. N stated per cell above.",
        "- Onset-variance is a structural proxy for monotony, not a perceptual verdict.",
    ]
    out = data_dir / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="docs/run_audits/pace_audition")
    args = ap.parse_args()
    data_dir = ROOT / args.data_dir

    mpath = data_dir / "pace_manifest.json"
    if not mpath.exists():
        print(f"No manifest at {mpath}. Run the build first.")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    captures = load_capture(data_dir)
    if not captures:
        print("No captures yet. Complete the audition in the browser first.")
        return

    joined = join_scores(captures, manifest["edge_data"])
    out = write_findings(data_dir, joined, manifest.get("playlists", []))
    print(f"Wrote {out}  ({len(joined)} rated edges joined)")


if __name__ == "__main__":
    main()
