"""Cross-seed collapse harness (SP1) — the automatic Collapse Index number.

Generates each corpus seed's playlist (faithfully, through the real policy layer
— reusing slider_differentiation_eval's generation machinery, no duplication),
splits interiors from piers, and computes the Collapse Index: are different-niche
seeds' playlists MORE alike than the seeds themselves? Lower CI = less collapse.

Guards (quality floor): a CI win counts only if each playlist keeps its
seed-similarity and worst edge above a baseline-relative floor. The first
(baseline) run establishes the floor; later --override runs are checked against it.

  # baseline (current default config) on both clusters, 3 repeats:
  python scripts/research/collapse_eval.py --tag baseline

  # smoke one cluster, one repeat:
  python scripts/research/collapse_eval.py --clusters dreampop_haze --repeats 1 --tag smoke

  # score an SP2 scoring fix against the baseline floor:
  python scripts/research/collapse_eval.py --tag mutualprox \
      --override '{"playlists":{"ds_pipeline":{"pier_bridge":{"...":...}}}}' \
      --baseline-run docs/run_audits/collapse/run_baseline.json

Output: docs/run_audits/collapse/run_<tag>.json + findings_<tag>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.research.collapse_metric import (  # noqa: E402
    centroid,
    cluster_ci,
    collapse_contribution,
    interior_jaccard,
    mean_pairwise_cos,
    pace_pair_ci,
    passes_floor,
    seed_similarity,
    unit_rows,
)
from scripts.research.slider_differentiation_eval import (  # noqa: E402
    ARTIFACT,
    DB,
    art,
    run_artist_cell,
)

# ---- corpus -----------------------------------------------------------------
CLUSTERS: Dict[str, List[str]] = {
    # different niches that all sag into the same hazy mid-tempo reverb-pop
    "dreampop_haze": ["Real Estate", "Slowdive", "Codeine", "Beach House"],
    # different electronic niches that all sag toward generic four-on-the-floor
    "electronic": ["Boards of Canada", "Autechre", "Aphex Twin", "Four Tet"],
}


# --- pace contour (arousal + log-z onset), library-wide for parity with the generator ---
_PACE: dict[str, Any] = {}


def pace_contour_all() -> np.ndarray:
    """(N, 2) [arousal_z, log_z_onset] for every artifact track, library-z-scored,
    aligned to artifact row order — the SAME signals the generator steers (parity)."""
    if not _PACE:
        from src.playlist.energy_loader import load_energy_matrix
        from src.playlist.bpm_loader import load_bpm_arrays
        from src.playlist.pace_contour import build_contour
        A = art()
        ids = [t for t, _ in sorted(A["id2idx"].items(), key=lambda kv: kv[1])]
        ids_arr = np.array(ids, dtype=object)
        sidecar = str(ROOT / "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz")
        arousal_z = load_energy_matrix(ids, sidecar_path=sidecar, features=("arousal_p50",))[:, 0]
        onset = load_bpm_arrays(ids_arr, db_path=str(DB))["onset_rate"]
        _PACE["contour"] = build_contour(arousal_z, onset)
    return _PACE["contour"]


# ---- artifact-backed seed/interior geometry ---------------------------------
def _seed_indices(seed: str) -> List[int]:
    A = art()
    sl = seed.strip().lower()
    return [i for i, a in enumerate(A["artists"]) if a.strip().lower() == sl]


def seed_catalog_centroid(seed: str) -> np.ndarray:
    """Unit centroid of the seed artist's whole catalog in MERT — the niche anchor."""
    A = art()
    idx = _seed_indices(seed)
    return centroid(unit_rows(np.asarray(A["mert"][idx], dtype=np.float64)))


def split_interior(track_ids: List[str], seed: str) -> Tuple[List[str], np.ndarray]:
    """Interior = playlist tracks NOT by the seed artist (piers are the seed's own).

    Returns (interior_track_ids, interior_unit_vectors) aligned row-for-row.
    """
    A = art()
    sl = seed.strip().lower()
    interior_ids = [
        t for t in track_ids
        if t in A["id2idx"] and A["artists"][A["id2idx"][t]].strip().lower() != sl
    ]
    idx = [A["id2idx"][t] for t in interior_ids]
    vecs = unit_rows(np.asarray(A["mert"][idx], dtype=np.float64)) if idx else np.zeros((0, 1))
    return interior_ids, vecs


def _std(vals: List[float]) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _collapse_stats(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse-sensitive aggregates for one repeat's per-pair CI list.

    The cluster *mean* CI is misleading — collapse is pair-specific (one niche pair
    converges while others diverge), so the mean washes it out. collapse_load (mean
    of the positive CIs) is 0 when nothing collapses and grows with severity,
    undiluted by diverging pairs; max_ci is the single worst pair (north-star #5).
    """
    cis = [p["ci"] for p in pairs]
    if not cis:
        return {"mean_ci": 0.0, "max_ci": 0.0, "max_pair": None,
                "collapse_load": 0.0, "n_collapsing": 0, "n_pairs": 0, "collapsing_pairs": []}
    worst = max(pairs, key=lambda p: p["ci"])
    return {
        "mean_ci": float(np.mean(cis)),
        "max_ci": float(worst["ci"]),
        "max_pair": [worst["i"], worst["j"]],
        "collapse_load": float(np.mean([max(0.0, c) for c in cis])),
        "n_collapsing": int(sum(c > 0 for c in cis)),
        "n_pairs": len(cis),
        "collapsing_pairs": [[p["i"], p["j"], round(p["ci"], 4)] for p in pairs if p["ci"] > 0],
    }


# ---- one repeat of one cluster ----------------------------------------------
def run_cluster_repeat(
    seeds: List[str],
    length: int,
    override: dict | None,
    seed_centroids: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Generate every seed once, split interiors, compute CI + per-seed floor metrics."""
    playlists: Dict[str, Any] = {}
    interiors: Dict[str, np.ndarray] = {}
    for seed in seeds:
        cell = run_artist_cell(seed, "dynamic", "dynamic", "dynamic", "dynamic", length, extra_ov=override)
        track_ids = cell.get("track_ids", [])
        interior_ids, vecs = split_interior(track_ids, seed)
        interiors[seed] = vecs
        own = seed_centroids[seed]
        others = [seed_centroids[s] for s in seeds if s != seed]
        c_vals = [
            {"track_id": tid, "c": round(collapse_contribution(vecs[r], own, others), 4)}
            for r, tid in enumerate(interior_ids)
        ]
        # worst edge: prefer the calibrated reporter edge; fall back to raw MERT min-adjacent.
        worst = cell.get("min_transition")
        if worst is None:
            worst = cell.get("sonic_worst")
        playlists[seed] = {
            "err": cell.get("err"),
            "wall": cell.get("wall"),
            "n_total": cell.get("n"),
            "n_interior": len(interior_ids),
            "interior_ids": interior_ids,
            "seed_sim": round(seed_similarity(vecs, own), 4) if len(vecs) else None,
            "worst_edge": worst,
            "sonic_worst": cell.get("sonic_worst"),
            "c": c_vals,
        }
    ci = cluster_ci(interiors, seed_centroids)
    # secondary diagnostics per cross-seed pair
    jac, mpc = [], []
    for p in ci["pairs"]:
        i, j = p["i"], p["j"]
        jac.append({"i": i, "j": j, "jaccard": round(interior_jaccard(
            playlists[i]["interior_ids"], playlists[j]["interior_ids"]), 4)})
        mpc.append({"i": i, "j": j, "mean_pairwise_cos": round(mean_pairwise_cos(
            interiors[i], interiors[j]), 4)})
    # pace mirror (parity): per cross-seed pair, distance-based CI on the same
    # [arousal, log-z onset] contour the generator steers. NaN rows (missing
    # arousal/onset) are dropped before averaging.
    contour = pace_contour_all()
    A_ = art()

    def _finite(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.float64)
        return M[np.all(np.isfinite(M), axis=1)] if M.size else M

    seed_rows = {s: _finite(contour[_seed_indices(s)]) for s in seed_centroids}
    pace_pairs = []
    for p in ci["pairs"]:
        i, j = p["i"], p["j"]
        ii = _finite(contour[[A_["id2idx"][t] for t in playlists[i]["interior_ids"] if t in A_["id2idx"]]])
        ij = _finite(contour[[A_["id2idx"][t] for t in playlists[j]["interior_ids"] if t in A_["id2idx"]]])
        si, sj = seed_rows[i], seed_rows[j]
        if min(len(si), len(sj), len(ii), len(ij)) == 0:
            continue
        comb = pace_pair_ci(ii, ij, si, sj)
        aro = pace_pair_ci(ii[:, :1], ij[:, :1], si[:, :1], sj[:, :1])
        ons = pace_pair_ci(ii[:, 1:2], ij[:, 1:2], si[:, 1:2], sj[:, 1:2])
        pace_pairs.append({
            "i": i, "j": j,
            "pace_ci": round(comb["ci"], 4),
            "arousal_ci": round(aro["ci"], 4),
            "onset_ci": round(ons["ci"], 4),
            "disagree": bool((p["ci"] > 0) != (comb["ci"] > 0)),
        })
    return {"ci": ci, "jaccard_pairs": jac, "mean_pairwise_pairs": mpc, "playlists": playlists,
            "pace_pairs": pace_pairs}


def seed_distance_table(seeds: List[str], seed_centroids: Dict[str, np.ndarray]) -> List[dict]:
    """Pairwise seed-centroid cosine — the distinctness the cluster *claims*."""
    from itertools import combinations
    out = []
    for i, j in combinations(seeds, 2):
        cos = float(np.dot(seed_centroids[i], seed_centroids[j]))
        out.append({"i": i, "j": j, "seed_cos": round(cos, 4)})
    return out


def run_cluster(name: str, seeds: List[str], length: int, repeats: int, override: dict | None) -> Dict[str, Any]:
    seed_centroids = {s: seed_catalog_centroid(s) for s in seeds}
    reps = [run_cluster_repeat(seeds, length, override, seed_centroids) for _ in range(repeats)]
    per_repeat = [_collapse_stats(r["ci"]["pairs"]) for r in reps]
    load = [s["collapse_load"] for s in per_repeat]
    mx = [s["max_ci"] for s in per_repeat]
    mn = [s["mean_ci"] for s in per_repeat]
    return {
        "seeds": seeds,
        "seed_distances": seed_distance_table(seeds, seed_centroids),
        "repeats": reps,
        "summary": {
            "collapse_load": round(float(np.mean(load)), 4),
            "std_collapse_load": round(_std(load), 4),
            "max_ci": round(float(np.mean(mx)), 4),
            "std_max_ci": round(_std(mx), 4),
            "mean_ci": round(float(np.mean(mn)), 4),
            "std_ci": round(_std(mn), 4),
            "n_collapsing_mean": round(float(np.mean([s["n_collapsing"] for s in per_repeat])), 2),
            "n_pairs": per_repeat[0]["n_pairs"] if per_repeat else 0,
            "per_repeat": per_repeat,
        },
    }


# ---- floors -----------------------------------------------------------------
def baseline_floor_table(run: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per-seed baseline seed_sim / worst_edge (mean over repeats), for floor checks."""
    floors: Dict[str, Dict[str, float]] = {}
    for cname, cdata in run["clusters"].items():
        for seed in cdata["seeds"]:
            sims = [r["playlists"][seed]["seed_sim"] for r in cdata["repeats"]
                    if r["playlists"][seed]["seed_sim"] is not None]
            edges = [r["playlists"][seed]["worst_edge"] for r in cdata["repeats"]
                     if r["playlists"][seed]["worst_edge"] is not None]
            floors[seed] = {
                "seed_sim": round(float(np.mean(sims)), 4) if sims else None,
                "worst_edge": round(float(np.mean(edges)), 4) if edges else None,
            }
    return floors


def check_floors(run: Dict[str, Any], baseline_floors: Dict[str, Dict[str, float]],
                 delta_seed: float, delta_edge: float) -> Dict[str, Any]:
    """Per-seed pass/fail vs baseline floors. A cluster fails if ANY seed trips a floor."""
    result: Dict[str, Any] = {"per_seed": {}, "cluster_pass": {}}
    for cname, cdata in run["clusters"].items():
        cluster_ok = True
        for seed in cdata["seeds"]:
            sims = [r["playlists"][seed]["seed_sim"] for r in cdata["repeats"]
                    if r["playlists"][seed]["seed_sim"] is not None]
            edges = [r["playlists"][seed]["worst_edge"] for r in cdata["repeats"]
                     if r["playlists"][seed]["worst_edge"] is not None]
            cur_sim = float(np.mean(sims)) if sims else None
            cur_edge = float(np.mean(edges)) if edges else None
            base = baseline_floors.get(seed, {})
            sim_ok = (cur_sim is None or base.get("seed_sim") is None
                      or passes_floor(cur_sim, base["seed_sim"], delta_seed))
            edge_ok = (cur_edge is None or base.get("worst_edge") is None
                       or passes_floor(cur_edge, base["worst_edge"], delta_edge))
            result["per_seed"][seed] = {
                "seed_sim": round(cur_sim, 4) if cur_sim is not None else None,
                "base_seed_sim": base.get("seed_sim"),
                "seed_sim_ok": sim_ok,
                "worst_edge": round(cur_edge, 4) if cur_edge is not None else None,
                "base_worst_edge": base.get("worst_edge"),
                "worst_edge_ok": edge_ok,
            }
            cluster_ok = cluster_ok and sim_ok and edge_ok
        result["cluster_pass"][cname] = cluster_ok
    return result


# ---- reporting --------------------------------------------------------------
def write_findings(run: Dict[str, Any], floor_check: Dict[str, Any] | None, out: Path) -> None:
    L: List[str] = [
        "# Cross-Seed Collapse — Findings",
        "",
        f"config: **{run['config_label']}**  |  length {run['length']}  |  repeats {run['repeats']}  "
        f"|  artifact variant `{run['provenance']['variant']}`",
        "",
        "**Collapse Index** `CI = S_play - S_seed` (cosine). CI > 0 ⇒ playlists more alike "
        "than the seeds ⇒ collapse. **Lower CI = less collapse.**",
        "",
    ]
    for cname, cdata in run["clusters"].items():
        s = cdata["summary"]
        cp = s["per_repeat"][0]["collapsing_pairs"] if s.get("per_repeat") else []
        cp_s = ", ".join(f"{i}↔{j} ({ci:+.3f})" for i, j, ci in cp) or "none"
        L += [f"## {cname}", "",
              f"seeds: {', '.join(cdata['seeds'])}", "",
              f"**collapse load = {s['collapse_load']:.4f} ± {s['std_collapse_load']:.4f}**  "
              f"(mean of positive per-pair CI; 0 = nothing collapses)",
              f"worst pair CI = {s['max_ci']:+.4f} ± {s['std_max_ci']:.4f}  |  "
              f"mean CI = {s['mean_ci']:+.4f} ± {s['std_ci']:.4f}  |  "
              f"collapsing pairs/repeat = {s['n_collapsing_mean']}/{s['n_pairs']}",
              f"collapsing pairs (repeat 0): {cp_s}", ""]
        L += ["seed distinctness (pairwise seed-centroid cosine — lower = more distinct niches):", ""]
        for d in cdata["seed_distances"]:
            L.append(f"- {d['i']} ↔ {d['j']}: {d['seed_cos']:.3f}")
        # per-pair CI from repeat 0 (representative)
        L += ["", "per-pair (repeat 0): CI = S_play − S_seed | jaccard | mean-pairwise-cos", ""]
        r0 = cdata["repeats"][0]
        _pp = r0.get("pace_pairs", [])
        if _pp:
            L.append(f"mean pace_ci = {np.mean([x['pace_ci'] for x in _pp]):+.4f}  "
                     f"(arousal {np.mean([x['arousal_ci'] for x in _pp]):+.4f}, "
                     f"onset {np.mean([x['onset_ci'] for x in _pp]):+.4f})")
        jac = {(p["i"], p["j"]): p["jaccard"] for p in r0["jaccard_pairs"]}
        mpc = {(p["i"], p["j"]): p["mean_pairwise_cos"] for p in r0["mean_pairwise_pairs"]}
        pp = {(x["i"], x["j"]): x for x in r0.get("pace_pairs", [])}
        for p in r0["ci"]["pairs"]:
            key = (p["i"], p["j"])
            pj = pp.get(key)
            pace_str = (f" | pace {pj['pace_ci']:+.3f} (a {pj['arousal_ci']:+.3f} o {pj['onset_ci']:+.3f})"
                        + (" ⚑disagree" if pj["disagree"] else "")) if pj else ""
            L.append(f"- {p['i']} ↔ {p['j']}: CI {p['ci']:+.3f} "
                     f"(S_play {p['s_play']:.3f} − S_seed {p['s_seed']:.3f}) | "
                     f"jac {jac.get(key, 0):.3f} | mpc {mpc.get(key, 0):.3f}{pace_str}")
        # per-seed quality
        L += ["", "per-seed quality (mean over repeats): seed_sim | worst_edge | n_interior | wall", ""]
        for seed in cdata["seeds"]:
            sims = [r["playlists"][seed]["seed_sim"] for r in cdata["repeats"] if r["playlists"][seed]["seed_sim"] is not None]
            edges = [r["playlists"][seed]["worst_edge"] for r in cdata["repeats"] if r["playlists"][seed]["worst_edge"] is not None]
            ni = [r["playlists"][seed]["n_interior"] for r in cdata["repeats"]]
            wl = [r["playlists"][seed]["wall"] for r in cdata["repeats"] if r["playlists"][seed]["wall"] is not None]
            err = next((r["playlists"][seed]["err"] for r in cdata["repeats"] if r["playlists"][seed]["err"]), None)
            sim_s = f"{np.mean(sims):.3f}" if sims else "—"
            edge_s = f"{np.mean(edges):.3f}" if edges else "—"
            extra = f"  ⚠ ERR {err}" if err else ""
            L.append(f"- {seed}: seed_sim {sim_s} | worst_edge {edge_s} | n_int {int(np.mean(ni)) if ni else 0} | wall {np.mean(wl):.1f}s{extra}")
        if floor_check:
            ok = floor_check["cluster_pass"].get(cname)
            L += ["", f"**quality floor: {'PASS — all seeds held' if ok else 'FAIL — a seed tripped a floor'}**", ""]
            for seed in cdata["seeds"]:
                ps = floor_check["per_seed"].get(seed, {})
                if not ps:
                    continue
                flags = []
                if ps.get("seed_sim_ok") is False:
                    flags.append(f"seed_sim {ps['seed_sim']} < base {ps['base_seed_sim']}")
                if ps.get("worst_edge_ok") is False:
                    flags.append(f"worst_edge {ps['worst_edge']} < base {ps['base_worst_edge']}")
                if flags:
                    L.append(f"  - ✗ {seed}: {'; '.join(flags)}")
        L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---- main -------------------------------------------------------------------
def _load_override(spec: str | None) -> dict | None:
    if not spec:
        return None
    p = ROOT / spec
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    pw = CODE_ROOT / spec
    if pw.exists():
        return json.loads(pw.read_text(encoding="utf-8"))
    return json.loads(spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clusters", default=",".join(CLUSTERS))
    ap.add_argument("--seeds", default="", help="comma list to restrict seeds within the chosen clusters")
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--override", default=None, help="JSON string or path to an SP2 config override")
    ap.add_argument("--baseline-run", default=None, help="run_*.json whose seed_sim/worst_edge define the floor")
    ap.add_argument("--delta-seed", type=float, default=0.05)
    ap.add_argument("--delta-edge", type=float, default=0.05)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out-dir", default="docs/run_audits/collapse")
    args = ap.parse_args()

    # provenance pre-flight (evaluation-methodology)
    bc = np.load(ARTIFACT, allow_pickle=True)
    variant = str(bc["X_sonic_variant"]) if "X_sonic_variant" in bc.files else "?"
    print(f"PRE-FLIGHT: artifact X_sonic_variant={variant!r}")
    if variant != "mert":
        print(f"ABORT: live sonic variant is {variant!r}, not 'mert' — re-fold before measuring collapse.")
        sys.exit(1)

    override = _load_override(args.override)
    chosen = [c.strip() for c in args.clusters.split(",") if c.strip()]
    seed_filter = {s.strip().lower() for s in args.seeds.split(",") if s.strip()}

    run: Dict[str, Any] = {
        "config_label": args.tag,
        "length": args.length,
        "repeats": args.repeats,
        "override": override,
        "provenance": {"variant": variant},
        "clusters": {},
    }
    for cname in chosen:
        if cname not in CLUSTERS:
            print(f"  SKIP unknown cluster {cname!r}")
            continue
        seeds = [s for s in CLUSTERS[cname] if not seed_filter or s.lower() in seed_filter]
        if len(seeds) < 2:
            print(f"  SKIP {cname}: <2 seeds after filter (need pairs)")
            continue
        print(f"\n=== cluster {cname}: {seeds} (repeats={args.repeats}) ===")
        run["clusters"][cname] = run_cluster(cname, seeds, args.length, args.repeats, override)
        sm = run["clusters"][cname]["summary"]
        print(f"  collapse load = {sm['collapse_load']:.4f} ± {sm['std_collapse_load']:.4f} | "
              f"worst pair CI = {sm['max_ci']:+.4f} | collapsing {sm['n_collapsing_mean']}/{sm['n_pairs']}")

    # floors
    floor_check = None
    if args.baseline_run:
        base = json.loads((ROOT / args.baseline_run).read_text(encoding="utf-8")
                          if (ROOT / args.baseline_run).exists()
                          else Path(args.baseline_run).read_text(encoding="utf-8"))
        floor_check = check_floors(run, baseline_floor_table(base), args.delta_seed, args.delta_edge)
        run["floor_check"] = floor_check
    else:
        run["baseline_floors"] = baseline_floor_table(run)  # this run defines the floor

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"run_{args.tag}.json"
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    find_path = out_dir / f"findings_{args.tag}.md"
    write_findings(run, floor_check, find_path)
    print(f"\nWrote {run_path}")
    print(f"Wrote {find_path}")
    if floor_check:
        for cname, ok in floor_check["cluster_pass"].items():
            print(f"  floor[{cname}]: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
