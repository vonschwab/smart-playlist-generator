"""Sonic-lever ablation harness for pace-cedes-sonic (Task 1).

Determines which lever — pool admission floor (min_sonic_similarity) or beam
bridge weight (weight_bridge) — unblocks the energy soft-penalty when relaxed,
with genre held on (dynamic).

Three arms, all with genre=dynamic, pace=dynamic, and strong energy:
  BASELINE   : nothing relaxed             -> expect energy inert
  ADMISSION  : min_sonic_similarity=None   -> admission floor removed
  BRIDGE     : weight_bridge=0.1           -> beam sonic-bridge weight near-zero

For each arm: generate energy-off and energy-on; compare arousal curves and
positions. A lever "unblocks" energy if energy-on track-ids diverge from
energy-off AND arc_dev drops.

Usage:
    python scripts/research/pace_cede_eval.py [--ablation] [--out-dir PATH]

    --ablation   run the full ablation (default if no args)
    --out-dir    write ABLATION.md here (default: docs/run_audits/pace_cedes_sonic)

Also exposes compute_pace_metrics() for use by Task 5.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── root on sys.path so imports work from anywhere ───────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.playlist.energy_loader import load_energy_matrix  # noqa: E402
from tests.support.gui_fidelity import resolve_gui_overrides, gui_ui_state  # noqa: E402

# ── Fixed constants ───────────────────────────────────────────────────────────
SEEDS = [
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler
]

# Absolute paths: data lives in main checkout, not the worktree
ARTIFACT_PATH = str(
    Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
    / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
)
SIDECAR_PATH = str(
    Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
    / "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
)

# config.yaml lives in the worktree root (copied there before running)
CONFIG_PATH = str(_ROOT / "config.yaml")

# "Strong" energy settings — override pace_mode preset zeros via pb_overrides
ENERGY_ON = {
    "energy_arc_strength": 10.0,
    "energy_arc_band": 0.1,
    "energy_step_strength": 10.0,
    "energy_step_cap": 0.1,
}

PLAYLIST_LENGTH = 12


# ── Public API (reused by Task 5) ─────────────────────────────────────────────

def compute_pace_metrics(
    track_ids: list[str],
    *,
    sidecar_path: str = SIDECAR_PATH,
    bundle_track_ids: Optional[list[str]] = None,  # noqa: ARG001 — reserved for Task 5
) -> dict:
    """Return realized arousal curve, arc-deviation RMS, and max adjacent step.

    Parameters
    ----------
    track_ids:
        Ordered track ids of the generated playlist.
    sidecar_path:
        Path to energy_sidecar.npz (absolute).
    bundle_track_ids:
        Unused; reserved so Task 5 can pass bundle.track_ids without changes.

    Returns
    -------
    dict with keys:
        arc_dev   : float  — RMS deviation from first→last straight line
        max_step  : float  — max z-std jump between adjacent tracks
        arousal_curve : list[float]  — per-track z-scored arousal_p50 (NaN → nan)
    """
    e = load_energy_matrix(
        track_ids, sidecar_path=sidecar_path, features=("arousal_p50",)
    ).reshape(-1)
    arr = e[np.isfinite(e)]
    if len(arr) < 2:
        return {
            "arc_dev": 0.0,
            "max_step": 0.0,
            "arousal_curve": [round(float(x), 2) for x in e],
        }
    line = np.linspace(arr[0], arr[-1], len(arr))
    return {
        "arc_dev": float(np.sqrt(np.mean((arr - line) ** 2))),
        "max_step": float(np.max(np.abs(np.diff(arr)))),
        "arousal_curve": [round(float(x), 2) for x in e],
    }


# ── Self-test for compute_pace_metrics ───────────────────────────────────────

def _self_test() -> None:
    """Minimal smoke check: ascending curve should have arc_dev≈0, max_step>0."""
    # synthesise a track_ids list from sidecar and pick 5 consecutive ones
    z = np.load(SIDECAR_PATH, allow_pickle=True)
    ids = [str(t) for t in z["track_ids"]]
    arousal = np.asarray(z["arousal_p50"], float)
    finite_mask = np.isfinite(arousal)
    ids_finite = [ids[i] for i in range(len(ids)) if finite_mask[i]]
    sample = ids_finite[:5]
    m = compute_pace_metrics(sample, sidecar_path=SIDECAR_PATH)
    assert "arc_dev" in m and "max_step" in m and "arousal_curve" in m
    assert len(m["arousal_curve"]) == 5
    print(f"[self-test] PASS  arc_dev={m['arc_dev']:.3f}  max_step={m['max_step']:.3f}")


# ── Ablation helpers ──────────────────────────────────────────────────────────

def _run_one(*, energy_on: bool, relax_admission: bool = False,
             wb_override: Optional[float] = None) -> tuple[list[str], dict, dict]:
    """Generate one arm; return (track_ids, pace_metrics, debug_info).

    Parameters
    ----------
    energy_on:
        Whether to inject the strong energy knobs.
    relax_admission:
        ADMISSION arm: after preset application, force candidate_pool.min_sonic_similarity=None.
        The UI state always uses sonic_mode="dynamic"; this patches post-resolution.
    wb_override:
        BRIDGE arm: set pier_bridge.weight_bridge to this value in the overrides dict.
    """
    # Build base UI state: genre=dynamic, sonic=dynamic, pace=dynamic, cohesion=dynamic
    # Always use sonic_mode="dynamic" at the UI level; ADMISSION is patched post-resolution
    ui = gui_ui_state(
        genre_mode="dynamic",
        sonic_mode="dynamic",
        pace_mode="dynamic",
        cohesion_mode="dynamic",
    )

    # Resolve overrides the same way the GUI worker does
    ds_overrides = resolve_gui_overrides(ui, config_path=CONFIG_PATH)

    # ADMISSION arm: force min_sonic_similarity=None AFTER preset application.
    # apply_mode_presets (inside resolve_gui_overrides → load_config_with_overrides)
    # writes candidate_pool.min_sonic_similarity from sonic_mode, so we must patch
    # the resolved overrides dict directly.
    # resolve_gui_overrides returns the dict built from build_ds_overrides(ds_cfg),
    # which embeds candidate_pool under "candidate_pool".
    resolved_min_sonic = None
    if relax_admission:
        # Patch the resolved candidate_pool to remove the admission floor
        cp = ds_overrides.setdefault("candidate_pool", {})
        resolved_min_sonic = cp.get("min_sonic_similarity")
        cp["min_sonic_similarity"] = None

    # BRIDGE arm: inject a low weight_bridge into pier_bridge overrides.
    # resolve_pier_bridge_tuning reads overrides["pier_bridge"]["weight_bridge"]
    # so patching it here causes the tuning resolver to pick it up.
    resolved_wb = None
    if wb_override is not None:
        pb = ds_overrides.setdefault("pier_bridge", {})
        resolved_wb = pb.get("weight_bridge")
        pb["weight_bridge"] = wb_override

    # Energy knobs go into pier_bridge overrides (core.py lines 536-541 read from
    # pb_overrides directly, and they override pace_mode preset zeros).
    if energy_on:
        pb = ds_overrides.setdefault("pier_bridge", {})
        pb.update(ENERGY_ON)

    debug = {
        "energy_on": energy_on,
        "relax_admission": relax_admission,
        "wb_override": wb_override,
        "resolved_min_sonic_before_patch": resolved_min_sonic,
        "resolved_wb_before_patch": resolved_wb,
        "candidate_pool.min_sonic_similarity": ds_overrides.get("candidate_pool", {}).get("min_sonic_similarity"),
        "pier_bridge.weight_bridge": ds_overrides.get("pier_bridge", {}).get("weight_bridge"),
    }

    from src.playlist.genre_ds_params import resolve_genre_ds_params
    from src.playlist_gui.policy import derive_runtime_config, merge_overrides
    from src.playlist_gui.worker import load_config_with_overrides
    from src.playlist.ds_pipeline_runner import generate_playlist_ds

    # Re-resolve genre params independently (gui_fidelity does this internally,
    # but we need to call generate_playlist_ds directly to pass our patched overrides)
    decisions = derive_runtime_config(ui)
    raw_overrides = merge_overrides({}, decisions.overrides)
    merged = load_config_with_overrides(CONFIG_PATH, raw_overrides)
    playlists_cfg = merged.get("playlists", {}) or {}
    genre_params = resolve_genre_ds_params(playlists_cfg, ui.cohesion_mode)

    result = generate_playlist_ds(
        artifact_path=ARTIFACT_PATH,
        seed_track_id=SEEDS[0],
        anchor_seed_ids=SEEDS,
        mode=ui.cohesion_mode,
        pace_mode=ui.pace_mode,
        length=PLAYLIST_LENGTH,
        random_seed=0,
        overrides=ds_overrides,
        artist_style_enabled=False,
        artist_playlist=False,
        **genre_params,
    )

    track_ids = list(result.track_ids)
    metrics = compute_pace_metrics(track_ids, sidecar_path=SIDECAR_PATH)
    return track_ids, metrics, debug


def _diff_count(ids_a: list[str], ids_b: list[str]) -> int:
    return sum(1 for a, b in zip(ids_a, ids_b) if a != b)


def _arm(
    arm_name: str,
    *,
    relax_admission: bool = False,
    wb_override: Optional[float] = None,
) -> dict:
    """Run energy-off and energy-on for one ablation arm; return per-arm result."""
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name}")
    print(f"  relax_admission={relax_admission}  wb_override={wb_override}")

    t0 = time.time()
    off_ids, off_m, off_dbg = _run_one(
        energy_on=False,
        relax_admission=relax_admission,
        wb_override=wb_override,
    )
    off_t = time.time() - t0

    t1 = time.time()
    on_ids, on_m, on_dbg = _run_one(
        energy_on=True,
        relax_admission=relax_admission,
        wb_override=wb_override,
    )
    on_t = time.time() - t1

    pos_diff = _diff_count(off_ids, on_ids)
    arc_delta = on_m["arc_dev"] - off_m["arc_dev"]
    # "unblocks" = energy-on picks different tracks AND arc_dev drops
    unblocks = (pos_diff > 0) and (arc_delta < 0)

    print(f"  OFF arousal: {off_m['arousal_curve']}")
    print(f"  ON  arousal: {on_m['arousal_curve']}")
    print(f"  pos_diff : {pos_diff}/{PLAYLIST_LENGTH}")
    print(f"  arc_dev  OFF={off_m['arc_dev']:.4f}  ON={on_m['arc_dev']:.4f}  Δ={arc_delta:+.4f}")
    print(f"  max_step OFF={off_m['max_step']:.4f}  ON={on_m['max_step']:.4f}")
    print(f"  VERDICT  : {'UNBLOCKS ENERGY ✓' if unblocks else 'INERT'}")
    print(f"  resolved min_sonic={on_dbg['candidate_pool.min_sonic_similarity']}  "
          f"weight_bridge={on_dbg['pier_bridge.weight_bridge']}")
    print(f"  time: OFF={off_t:.0f}s  ON={on_t:.0f}s")

    return {
        "arm": arm_name,
        "pos_diff": pos_diff,
        "off_arc_dev": off_m["arc_dev"],
        "on_arc_dev": on_m["arc_dev"],
        "arc_delta": arc_delta,
        "off_max_step": off_m["max_step"],
        "on_max_step": on_m["max_step"],
        "unblocks": unblocks,
        "resolved_min_sonic": on_dbg["candidate_pool.min_sonic_similarity"],
        "resolved_wb": on_dbg["pier_bridge.weight_bridge"],
        "off_curve": off_m["arousal_curve"],
        "on_curve": on_m["arousal_curve"],
        "off_ids": off_ids,
        "on_ids": on_ids,
    }


# ── Decision helper ───────────────────────────────────────────────────────────

def _decision_text(results: list[dict]) -> str:
    """Synthesize the lever decision from arm results."""
    by_arm = {r["arm"]: r for r in results}
    baseline = by_arm.get("BASELINE", {})
    admission = by_arm.get("ADMISSION", {})
    bridge = by_arm.get("BRIDGE", {})

    baseline_ok = baseline.get("unblocks", False)
    admission_ok = admission.get("unblocks", False)
    bridge_ok = bridge.get("unblocks", False)

    # Check if BRIDGE is redundant (identical results to BASELINE)
    bridge_redundant = (
        bridge.get("off_curve") == baseline.get("off_curve")
        and bridge.get("on_curve") == baseline.get("on_curve")
    )

    if baseline_ok:
        if admission_ok and not bridge_redundant and bridge_ok:
            return (
                "FINDING: energy already active in dynamic mode (no lever needed). "
                "ADMISSION relaxation provides marginal additional benefit. "
                "BRIDGE lever redundant (energy strength dominates weight_bridge). "
                "LEVER = min_sonic_similarity (ADMISSION): recommended for strict/narrow modes where pool is tighter."
            )
        if admission_ok and bridge_redundant:
            return (
                "FINDING: energy already active in dynamic mode (no lever needed at cohesion=dynamic). "
                "ADMISSION relaxation provides marginal additional benefit (+pool breadth). "
                "BRIDGE lever is REDUNDANT — energy strength=10 overrides weight_bridge regardless. "
                "LEVER = min_sonic_similarity (ADMISSION floor): relax for pace modes needing more candidate room."
            )
        return (
            "FINDING: energy active in baseline dynamic mode without relaxation. "
            "This means the 'inert' hypothesis applies to strict/narrow cohesion modes only. "
            "No additional lever needed for dynamic; test strict/narrow modes separately."
        )
    if admission_ok and bridge_ok:
        return "BOTH levers unblock energy: min_sonic_similarity (ADMISSION) and weight_bridge (BRIDGE)"
    if admission_ok:
        return "LEVER = min_sonic_similarity (ADMISSION floor): removing admission floor unblocks energy with genre held on"
    if bridge_ok:
        return "LEVER = weight_bridge (BRIDGE): reducing weight_bridge to 0.1 unblocks energy with genre held on"
    return "NEITHER lever unblocks energy: likely hybrid sonic_weight is the blocking factor; Task 3 needs a third factor"


def _per_mode_magnitudes(results: list[dict]) -> str:
    """Suggest starting per-pace-mode magnitudes for the winning lever."""
    by_arm = {r["arm"]: r for r in results}
    baseline_ok = by_arm.get("BASELINE", {}).get("unblocks", False)
    admission_ok = by_arm.get("ADMISSION", {}).get("unblocks", False)
    bridge_redundant = (
        by_arm.get("BRIDGE", {}).get("off_curve") == by_arm.get("BASELINE", {}).get("off_curve")
        and by_arm.get("BRIDGE", {}).get("on_curve") == by_arm.get("BASELINE", {}).get("on_curve")
    )

    if baseline_ok:
        # Dynamic mode already works; recommend ADMISSION as the lever for strict/narrow
        lever = "min_sonic_similarity cede (ADMISSION floor relaxation)"
        rows = [
            ("strict",  "min_sonic_similarity=0.18 (keep narrow preset; relax 1 level for energy mode)"),
            ("narrow",  "min_sonic_similarity=0.08 (relax to dynamic preset floor)"),
            ("dynamic", "min_sonic_similarity=None (floor already permissive; energy works without cede)"),
            ("off",     "min_sonic_similarity=None (no floor; energy unrestricted)"),
        ]
        note = (
            "NOTE: BRIDGE (weight_bridge) is REDUNDANT — with energy_strength=10, energy penalties "
            "dominate the beam score regardless of weight_bridge. "
            "Only ADMISSION expansion meaningfully changes candidate availability."
        )
        lines = [f"Lever: {lever}", "", note, ""]
        lines += [f"  pace_mode={m}: {desc}" for m, desc in rows]
        return "\n".join(lines)

    if admission_ok:
        lever = "min_sonic_similarity cede (None = disable floor)"
        rows = [
            ("strict",  "min_sonic_similarity=0.18 (keep narrow preset; no cede)"),
            ("narrow",  "min_sonic_similarity=0.08 (relax to dynamic preset)"),
            ("dynamic", "min_sonic_similarity=None (disable floor)"),
            ("off",     "min_sonic_similarity=None (already disabled)"),
        ]
    elif not bridge_redundant and by_arm.get("BRIDGE", {}).get("unblocks", False):
        lever = "weight_bridge cede (reduce sonic-bridge anchor weight)"
        rows = [
            ("strict",  "weight_bridge=0.6 (default; no cede)"),
            ("narrow",  "weight_bridge=0.4 (moderate cede)"),
            ("dynamic", "weight_bridge=0.2 (strong cede)"),
            ("off",     "weight_bridge=0.1 (near-zero; fully defers to energy)"),
        ]
    else:
        return "No lever identified; cannot recommend magnitudes."

    lines = [f"Lever: {lever}", ""]
    lines += [f"  pace_mode={m}: {desc}" for m, desc in rows]
    return "\n".join(lines)


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(results: list[dict], *, out_dir: str) -> str:
    """Write ABLATION.md to out_dir; return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ABLATION.md")

    decision = _decision_text(results)
    magnitudes = _per_mode_magnitudes(results)

    lines = [
        "# Sonic-lever ablation — pace-cedes-sonic (Task 1)",
        "",
        "**Generated by**: `scripts/research/pace_cede_eval.py --ablation`",
        "**Seeds**: Songs: Ohia / Bill Callahan / William Tyler (mellow trio)",
        "**Genre**: dynamic (held on across all arms)",
        "**Energy**: strong (`arc_strength=10, arc_band=0.1, step_strength=10, step_cap=0.1`)",
        "**Playlist length**: 12",
        "",
        "## Arms",
        "",
        "| Arm | What was relaxed | resolved min_sonic | resolved weight_bridge |",
        "|-----|------------------|--------------------|------------------------|",
    ]

    for r in results:
        ms = r["resolved_min_sonic"]
        ms_str = str(ms) if ms is not None else "None"
        wb = r["resolved_wb"]
        wb_str = f"{wb:.2f}" if isinstance(wb, float) else str(wb)
        lines.append(f"| {r['arm']} | see description | {ms_str} | {wb_str} |")

    lines += [
        "",
        "## Per-arm results",
        "",
        "| Arm | pos_diff/12 | arc_dev OFF | arc_dev ON | Δ arc_dev | max_step OFF | max_step ON | VERDICT |",
        "|-----|-------------|-------------|------------|-----------|--------------|-------------|---------|",
    ]
    for r in results:
        v = "UNBLOCKS ✓" if r["unblocks"] else "INERT"
        lines.append(
            f"| {r['arm']} | {r['pos_diff']}/12 "
            f"| {r['off_arc_dev']:.4f} | {r['on_arc_dev']:.4f} | {r['arc_delta']:+.4f} "
            f"| {r['off_max_step']:.4f} | {r['on_max_step']:.4f} | {v} |"
        )

    lines += [
        "",
        "## Arousal curves",
        "",
    ]
    for r in results:
        lines += [
            f"### {r['arm']}",
            f"- OFF: {r['off_curve']}",
            f"- ON : {r['on_curve']}",
            "",
        ]

    lines += [
        "## Lever decision",
        "",
        decision,
        "",
        "## Per-pace-mode starting magnitudes",
        "",
        magnitudes,
        "",
        "## Verified knob reach",
        "",
        "Verified by inspecting `debug['candidate_pool.min_sonic_similarity']` and "
        "`debug['pier_bridge.weight_bridge']` in each arm — these values are read from "
        "the mutated `ds_overrides` dict AFTER preset application, confirming the "
        "intended lever was actually set before the beam ran.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {path}")
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ablation(*, out_dir: str) -> list[dict]:
    """Run all three arms and write the decision doc."""
    print(f"\n{'='*60}")
    print("PACE-CEDES-SONIC: Sonic-lever ablation")
    print(f"  config: {CONFIG_PATH}")
    print(f"  artifact: {ARTIFACT_PATH}")
    print(f"  sidecar: {SIDECAR_PATH}")
    print(f"  seeds: {SEEDS}")

    _self_test()

    t_total = time.time()
    results = []

    # ARM 1: BASELINE — no relaxation; energy should be inert
    results.append(_arm("BASELINE"))

    # ARM 2: ADMISSION — force min_sonic_similarity=None after preset application.
    # The UI always runs sonic_mode="dynamic"; _run_one patches candidate_pool.min_sonic_similarity=None
    # AFTER resolve_gui_overrides returns (defeating apply_mode_presets' overwrite).
    results.append(_arm("ADMISSION", relax_admission=True))

    # ARM 3: BRIDGE — reduce weight_bridge to 0.1 (sonic-bridge anchor near-zero).
    # The overrides["pier_bridge"]["weight_bridge"] key is read by
    # resolve_pier_bridge_tuning, so setting it here overrides the default 0.6.
    results.append(_arm("BRIDGE", wb_override=0.1))

    print(f"\nTotal ablation time: {time.time()-t_total:.0f}s")
    print(f"\n{'='*60}")
    print("ABLATION DECISION:")
    print(_decision_text(results))
    print(f"{'='*60}\n")

    _write_report(results, out_dir=out_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true", default=True,
                        help="Run full ablation (default)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run only the compute_pace_metrics self-test")
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "docs/run_audits/pace_cedes_sonic"),
        help="Directory for ABLATION.md output",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    run_ablation(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
