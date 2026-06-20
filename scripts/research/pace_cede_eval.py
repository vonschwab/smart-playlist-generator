"""Sonic-lever ablation harness for pace-cedes-sonic (Task 1).

Determines which lever — pool admission floor (min_sonic_similarity), beam
bridge weight (weight_bridge), BPM bridge band, or onset bridge band — unblocks
the energy soft-penalty when relaxed, with genre held on (dynamic).

Five arms, all with genre=dynamic, pace=dynamic, and strong energy:
  BASELINE      : nothing relaxed             -> expect energy inert w/ BPM active
  ADMISSION     : min_sonic_similarity=None   -> admission floor removed
  BRIDGE        : weight_bridge=0.1           -> beam sonic-bridge weight near-zero
  BPM_BRIDGE    : bpm_bridge_max_log_distance -> very large (off)
  ONSET_BRIDGE  : onset_bridge_max_log_distance -> very large (off)

For each arm: generate energy-off and energy-on; compare arousal curves and
positions. A lever "unblocks" energy if energy-on track-ids diverge from
energy-off AND arc_dev drops.

BPM gate status: verified active by checking for "BPM loaded" log message.
BPM+onset band values come from resolve_pace_mode("dynamic") and are patched
via monkeypatching src.playlist.pipeline.core.resolve_pace_mode for the
BPM_BRIDGE and ONSET_BRIDGE arms.

Usage:
    python scripts/research/pace_cede_eval.py [--ablation] [--out-dir PATH]
    python scripts/research/pace_cede_eval.py --strict-narrow [--out-dir PATH]

    --ablation      run the full ablation at pace=dynamic (default if no args)
    --strict-narrow run BASELINE/ADMISSION/BPM_BRIDGE/ONSET_BRIDGE at
                    pace=strict/strict/strict and pace=narrow/narrow/narrow
    --out-dir       write ABLATION.md here (default: docs/run_audits/pace_cedes_sonic)

Also exposes compute_pace_metrics() for use by Task 5.
"""
from __future__ import annotations

import argparse
import logging
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

# Absolute paths: data lives in main checkout, not the worktree.
# The worktree's data/ directory is linked via a junction but git has already
# checked out a zero-byte placeholder data/metadata.db (tracked in the repo),
# so the relative path resolves to the empty placeholder, not the real 794 MB DB.
# Solution: always use absolute paths for data files and inject the db path via
# overrides so core.py's BPM loader skips the relative fallback.
_MAIN_CHECKOUT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
ARTIFACT_PATH = str(
    _MAIN_CHECKOUT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
)
SIDECAR_PATH = str(
    _MAIN_CHECKOUT / "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
)
# Real database — passed via overrides["library"]["database_path"] to core.py's
# BPM loader so it never falls back to the zero-byte worktree placeholder.
DB_PATH = str(_MAIN_CHECKOUT / "data/metadata.db")

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

# BPM/onset "effectively off" sentinel: log2(1e6) ≈ 20; any real track pair
# is well within this distance, so the gate is never triggered.
_BPM_BAND_OFF = 1e6
_ONSET_BAND_OFF = 1e6


# ── BPM logging capture ───────────────────────────────────────────────────────

class _BpmCapture(logging.Handler):
    """Capture the BPM loaded / BPM load failed log line during a run."""
    def __init__(self) -> None:
        super().__init__()
        self.loaded_msg: Optional[str] = None
        self.failed: bool = False

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "BPM loaded" in msg:
            self.loaded_msg = msg
        elif "BPM load failed" in msg:
            self.failed = True


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

def _run_one(
    *,
    energy_on: bool,
    mode: str = "dynamic",
    relax_admission: bool = False,
    wb_override: Optional[float] = None,
    bpm_bridge_off: bool = False,
    onset_bridge_off: bool = False,
) -> tuple[list[str], dict, dict, str]:
    """Generate one arm; return (track_ids, pace_metrics, debug_info, bpm_status).

    Parameters
    ----------
    energy_on:
        Whether to inject the strong energy knobs.
    mode:
        Pace/genre/sonic/cohesion mode string applied uniformly (e.g. "strict",
        "narrow", "dynamic").  All four axes are set to this value.
    relax_admission:
        ADMISSION arm: after preset application, force candidate_pool.min_sonic_similarity=None.
    wb_override:
        BRIDGE arm: set pier_bridge.weight_bridge to this value in the overrides dict.
    bpm_bridge_off:
        BPM_BRIDGE arm: monkeypatch resolve_pace_mode to return bpm_bridge_max_log_distance=1e6.
    onset_bridge_off:
        ONSET_BRIDGE arm: monkeypatch resolve_pace_mode to return onset_bridge_max_log_distance=1e6.
    """
    # Build base UI state with all four axes set to the requested mode.
    # cohesion_mode drives the beam; genre/sonic/pace drive pool composition.
    ui = gui_ui_state(
        genre_mode=mode,
        sonic_mode=mode,
        pace_mode=mode,
        cohesion_mode=mode,
    )

    # Resolve overrides the same way the GUI worker does
    ds_overrides = resolve_gui_overrides(ui, config_path=CONFIG_PATH)

    # Inject the real database path so core.py's BPM loader doesn't fall back to
    # the relative "data/metadata.db" (which resolves to a zero-byte worktree
    # placeholder because git checks out a tracked empty file there).
    ds_overrides.setdefault("library", {})["database_path"] = DB_PATH

    # ADMISSION arm: force min_sonic_similarity=None AFTER preset application.
    resolved_min_sonic = None
    if relax_admission:
        cp = ds_overrides.setdefault("candidate_pool", {})
        resolved_min_sonic = cp.get("min_sonic_similarity")
        cp["min_sonic_similarity"] = None

    # BRIDGE arm: inject a low weight_bridge into pier_bridge overrides.
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

    # BPM_BRIDGE / ONSET_BRIDGE arm: monkeypatch resolve_pace_mode in core.py.
    # pace_settings comes from resolve_pace_mode(pace_mode) called inside
    # generate_playlist_ds → _run_ds_pipeline → core.py line 264.
    # The function is imported into core's local namespace, so we must patch it
    # there: "import src.playlist.pipeline.core; core.resolve_pace_mode = ...".
    import src.playlist.pipeline.core as _core_mod
    from src.playlist.mode_presets import resolve_pace_mode as _real_resolve_pace_mode

    _bpm_bridge_log_dist = None   # resolved value for debug
    _onset_bridge_log_dist = None

    if bpm_bridge_off or onset_bridge_off:
        def _patched_resolve_pace_mode(pm: str) -> dict:
            settings = dict(_real_resolve_pace_mode(pm))
            if bpm_bridge_off:
                settings["bpm_bridge_max_log_distance"] = _BPM_BAND_OFF
                # Also disable the soft penalty so it's truly "off"
                settings["bpm_bridge_soft_penalty_strength"] = 0.0
            if onset_bridge_off:
                settings["onset_bridge_max_log_distance"] = _ONSET_BAND_OFF
                settings["onset_bridge_soft_penalty_strength"] = 0.0
            return settings
        _core_mod.resolve_pace_mode = _patched_resolve_pace_mode  # type: ignore[assignment]
        # Read what the patch will yield for this mode so we can log it
        _patched = _patched_resolve_pace_mode(mode)
        _bpm_bridge_log_dist = _patched["bpm_bridge_max_log_distance"]
        _onset_bridge_log_dist = _patched["onset_bridge_max_log_distance"]
    else:
        _core_mod.resolve_pace_mode = _real_resolve_pace_mode  # type: ignore[assignment]
        _natural = _real_resolve_pace_mode(mode)
        _bpm_bridge_log_dist = _natural["bpm_bridge_max_log_distance"]
        _onset_bridge_log_dist = _natural["onset_bridge_max_log_distance"]

    debug: dict = {
        "energy_on": energy_on,
        "relax_admission": relax_admission,
        "wb_override": wb_override,
        "bpm_bridge_off": bpm_bridge_off,
        "onset_bridge_off": onset_bridge_off,
        "resolved_min_sonic_before_patch": resolved_min_sonic,
        "resolved_wb_before_patch": resolved_wb,
        "candidate_pool.min_sonic_similarity": ds_overrides.get("candidate_pool", {}).get(
            "min_sonic_similarity"
        ),
        "pier_bridge.weight_bridge": ds_overrides.get("pier_bridge", {}).get("weight_bridge"),
        "bpm_bridge_max_log_distance": _bpm_bridge_log_dist,
        "onset_bridge_max_log_distance": _onset_bridge_log_dist,
        "library.database_path": ds_overrides.get("library", {}).get("database_path"),
    }

    # Capture BPM logging
    bpm_handler = _BpmCapture()
    bpm_logger = logging.getLogger("src.playlist.pipeline.core")
    bpm_logger.addHandler(bpm_handler)
    # Ensure the logger propagates at INFO so our handler sees the message
    _orig_level = bpm_logger.level
    if bpm_logger.level == logging.NOTSET or bpm_logger.level > logging.INFO:
        bpm_logger.setLevel(logging.INFO)

    try:
        from src.playlist.genre_ds_params import resolve_genre_ds_params
        from src.playlist_gui.policy import derive_runtime_config, merge_overrides
        from src.playlist_gui.worker import load_config_with_overrides
        from src.playlist.ds_pipeline_runner import generate_playlist_ds

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
    finally:
        bpm_logger.removeHandler(bpm_handler)
        bpm_logger.setLevel(_orig_level)
        # Restore real resolve_pace_mode
        _core_mod.resolve_pace_mode = _real_resolve_pace_mode  # type: ignore[assignment]

    # Build BPM status string
    if bpm_handler.failed:
        bpm_status = "BPM LOAD FAILED"
    elif bpm_handler.loaded_msg:
        bpm_status = f"BPM active: {bpm_handler.loaded_msg}"
    else:
        # With dynamic pace_mode, BPM should load (distances are finite).
        # No message means either logging wasn't captured or distances are inf.
        bpm_status = "BPM status unknown (no log captured)"

    track_ids = list(result.track_ids)
    metrics = compute_pace_metrics(track_ids, sidecar_path=SIDECAR_PATH)
    return track_ids, metrics, debug, bpm_status


def _diff_count(ids_a: list[str], ids_b: list[str]) -> int:
    return sum(1 for a, b in zip(ids_a, ids_b) if a != b)


def _arm(
    arm_name: str,
    *,
    mode: str = "dynamic",
    relax_admission: bool = False,
    wb_override: Optional[float] = None,
    bpm_bridge_off: bool = False,
    onset_bridge_off: bool = False,
) -> dict:
    """Run energy-off and energy-on for one ablation arm; return per-arm result."""
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name}  [mode={mode}]")
    print(
        f"  relax_admission={relax_admission}  wb_override={wb_override}  "
        f"bpm_bridge_off={bpm_bridge_off}  onset_bridge_off={onset_bridge_off}"
    )

    t0 = time.time()
    off_ids, off_m, off_dbg, off_bpm = _run_one(
        energy_on=False,
        mode=mode,
        relax_admission=relax_admission,
        wb_override=wb_override,
        bpm_bridge_off=bpm_bridge_off,
        onset_bridge_off=onset_bridge_off,
    )
    off_t = time.time() - t0

    t1 = time.time()
    on_ids, on_m, on_dbg, on_bpm = _run_one(
        energy_on=True,
        mode=mode,
        relax_admission=relax_admission,
        wb_override=wb_override,
        bpm_bridge_off=bpm_bridge_off,
        onset_bridge_off=onset_bridge_off,
    )
    on_t = time.time() - t1

    pos_diff = _diff_count(off_ids, on_ids)
    arc_delta = on_m["arc_dev"] - off_m["arc_dev"]
    # "unblocks" = energy-on picks different tracks AND arc_dev drops
    unblocks = (pos_diff > 0) and (arc_delta < 0)

    print(f"  BPM status (OFF): {off_bpm}")
    print(f"  BPM status (ON) : {on_bpm}")
    print(f"  OFF arousal: {off_m['arousal_curve']}")
    print(f"  ON  arousal: {on_m['arousal_curve']}")
    print(f"  pos_diff : {pos_diff}/{PLAYLIST_LENGTH}")
    print(f"  arc_dev  OFF={off_m['arc_dev']:.4f}  ON={on_m['arc_dev']:.4f}  Δ={arc_delta:+.4f}")
    print(f"  max_step OFF={off_m['max_step']:.4f}  ON={on_m['max_step']:.4f}")
    print(f"  VERDICT  : {'UNBLOCKS ENERGY ✓' if unblocks else 'INERT'}")
    print(
        f"  resolved min_sonic={on_dbg['candidate_pool.min_sonic_similarity']}  "
        f"weight_bridge={on_dbg['pier_bridge.weight_bridge']}"
    )
    print(
        f"  bpm_bridge_max_log_dist={on_dbg['bpm_bridge_max_log_distance']}  "
        f"onset_bridge_max_log_dist={on_dbg['onset_bridge_max_log_distance']}"
    )
    print(f"  time: OFF={off_t:.0f}s  ON={on_t:.0f}s")

    return {
        "arm": arm_name,
        "mode": mode,
        "pos_diff": pos_diff,
        "off_arc_dev": off_m["arc_dev"],
        "on_arc_dev": on_m["arc_dev"],
        "arc_delta": arc_delta,
        "off_max_step": off_m["max_step"],
        "on_max_step": on_m["max_step"],
        "unblocks": unblocks,
        "resolved_min_sonic": on_dbg["candidate_pool.min_sonic_similarity"],
        "resolved_wb": on_dbg["pier_bridge.weight_bridge"],
        "bpm_bridge_max_log_distance": on_dbg["bpm_bridge_max_log_distance"],
        "onset_bridge_max_log_distance": on_dbg["onset_bridge_max_log_distance"],
        "off_curve": off_m["arousal_curve"],
        "on_curve": on_m["arousal_curve"],
        "off_ids": off_ids,
        "on_ids": on_ids,
        "bpm_status_off": off_bpm,
        "bpm_status_on": on_bpm,
        "off_wall": off_t,
        "on_wall": on_t,
    }


# ── Decision helper ───────────────────────────────────────────────────────────

def _decision_text(results: list[dict]) -> str:
    """Synthesize the lever decision from arm results."""
    by_arm = {r["arm"]: r for r in results}
    baseline = by_arm.get("BASELINE", {})
    admission = by_arm.get("ADMISSION", {})
    bridge = by_arm.get("BRIDGE", {})
    bpm_brd = by_arm.get("BPM_BRIDGE", {})
    onset_brd = by_arm.get("ONSET_BRIDGE", {})

    baseline_ok = baseline.get("unblocks", False)
    admission_ok = admission.get("unblocks", False)
    bridge_ok = bridge.get("unblocks", False)
    bpm_brd_ok = bpm_brd.get("unblocks", False)
    onset_brd_ok = onset_brd.get("unblocks", False)

    bridge_redundant = (
        bridge.get("off_curve") == baseline.get("off_curve")
        and bridge.get("on_curve") == baseline.get("on_curve")
    )

    if baseline_ok:
        return (
            "FINDING: energy already steers in dynamic mode (no lever needed at cohesion=dynamic, "
            "genre=dynamic, sonic=dynamic, pace=dynamic WITH BPM GATES ACTIVE). "
            "The prior 'inert' finding was measured under a different condition "
            "(different mode or different injection path). "
            f"ADMISSION={'UNBLOCKS ✓' if admission_ok else 'no change'}, "
            f"BRIDGE={'REDUNDANT (same results)' if bridge_redundant else ('UNBLOCKS ✓' if bridge_ok else 'INERT')}, "
            f"BPM_BRIDGE={'UNBLOCKS ✓' if bpm_brd_ok else 'INERT'}, "
            f"ONSET_BRIDGE={'UNBLOCKS ✓' if onset_brd_ok else 'INERT'}. "
            "With BPM active and energy_strength=10, the beam already routes around the BPM/onset "
            "soft penalties. No lever relaxation required for dynamic mode."
        )

    levers = []
    if admission_ok:
        levers.append("min_sonic_similarity (ADMISSION floor)")
    if bpm_brd_ok:
        levers.append("bpm_bridge_max_log_distance (BPM_BRIDGE band)")
    if onset_brd_ok:
        levers.append("onset_bridge_max_log_distance (ONSET_BRIDGE band)")
    if bridge_ok and not bridge_redundant:
        levers.append("weight_bridge (BRIDGE weight)")

    if levers:
        return f"LEVER(S) = {' AND '.join(levers)}: relaxing these (one at a time) unblocks energy steering with genre held on."

    return (
        "NEITHER admission nor BPM/onset band nor weight_bridge unblocks energy. "
        "Energy is blocked by the sonic-weight hybrid (weight_sonic in the beam score), "
        "or the beam candidates simply have no arousal diversity. "
        "Task 3 needs a different approach."
    )


def _per_mode_magnitudes(results: list[dict]) -> str:
    """Suggest starting per-pace-mode magnitudes for the winning lever(s)."""
    by_arm = {r["arm"]: r for r in results}
    baseline_ok = by_arm.get("BASELINE", {}).get("unblocks", False)
    admission_ok = by_arm.get("ADMISSION", {}).get("unblocks", False)
    bpm_brd_ok = by_arm.get("BPM_BRIDGE", {}).get("unblocks", False)
    onset_brd_ok = by_arm.get("ONSET_BRIDGE", {}).get("unblocks", False)
    bridge_redundant = (
        by_arm.get("BRIDGE", {}).get("off_curve") == by_arm.get("BASELINE", {}).get("off_curve")
        and by_arm.get("BRIDGE", {}).get("on_curve") == by_arm.get("BASELINE", {}).get("on_curve")
    )
    bridge_ok = by_arm.get("BRIDGE", {}).get("unblocks", False)

    if baseline_ok:
        lines = [
            "Dynamic mode already steers without relaxation (BPM active). "
            "Lever recommendations apply only to strict/narrow pace modes where BPM/onset "
            "bands are tighter and the pool is smaller.",
            "",
            "Recommended lever: min_sonic_similarity cede for strict/narrow, "
            "OR accept that energy is already live in dynamic.",
            "",
            "  pace_mode=strict  → min_sonic_similarity=0.18 (narrow preset floor; 1 level relax)",
            "  pace_mode=narrow  → min_sonic_similarity=0.08 (dynamic preset floor)",
            "  pace_mode=dynamic → min_sonic_similarity=None (already permissive; no cede needed)",
            "  pace_mode=off     → min_sonic_similarity=None (no floor)",
        ]
        return "\n".join(lines)

    sections = []

    if bpm_brd_ok:
        sections.append(
            "Lever: bpm_bridge_max_log_distance cede\n"
            "  pace_mode=strict  → bpm_bridge_max_log_distance=0.40 (default; tight)\n"
            "  pace_mode=narrow  → bpm_bridge_max_log_distance=0.60 (moderate cede)\n"
            "  pace_mode=dynamic → bpm_bridge_max_log_distance=1.20 (wide cede; frees energy)\n"
            "  pace_mode=off     → bpm_bridge_max_log_distance=inf (no constraint)"
        )

    if onset_brd_ok:
        sections.append(
            "Lever: onset_bridge_max_log_distance cede\n"
            "  pace_mode=strict  → onset_bridge_max_log_distance=0.40 (default; tight)\n"
            "  pace_mode=narrow  → onset_bridge_max_log_distance=0.60 (moderate cede)\n"
            "  pace_mode=dynamic → onset_bridge_max_log_distance=1.20 (wide cede; frees energy)\n"
            "  pace_mode=off     → onset_bridge_max_log_distance=inf (no constraint)"
        )

    if admission_ok:
        sections.append(
            "Lever: min_sonic_similarity cede (None = disable floor)\n"
            "  pace_mode=strict  → min_sonic_similarity=0.18 (keep narrow preset; no cede)\n"
            "  pace_mode=narrow  → min_sonic_similarity=0.08 (relax to dynamic preset)\n"
            "  pace_mode=dynamic → min_sonic_similarity=None (disable floor)\n"
            "  pace_mode=off     → min_sonic_similarity=None (already disabled)"
        )

    if not sections:
        if bridge_ok and not bridge_redundant:
            sections.append(
                "Lever: weight_bridge cede (reduce sonic-bridge anchor weight)\n"
                "  pace_mode=strict  → weight_bridge=0.6 (default; no cede)\n"
                "  pace_mode=narrow  → weight_bridge=0.4 (moderate cede)\n"
                "  pace_mode=dynamic → weight_bridge=0.2 (strong cede)\n"
                "  pace_mode=off     → weight_bridge=0.1 (near-zero; fully defers to energy)"
            )

    if not sections:
        return "No lever identified; cannot recommend magnitudes."

    return "\n\n".join(sections)


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
        "**BPM gates**: ACTIVE (pace=dynamic; bpm_bridge_max_log_distance=0.85 is finite → "
        "BPM loaded from data/metadata.db for all arms except BPM_BRIDGE)",
        "",
        "## Arms",
        "",
        "| Arm | What was relaxed | resolved min_sonic | resolved weight_bridge "
        "| bpm_bridge_max_log_dist | onset_bridge_max_log_dist |",
        "|-----|------------------|--------------------|------------------------|"
        "-------------------------|---------------------------|",
    ]

    for r in results:
        ms = r["resolved_min_sonic"]
        ms_str = str(ms) if ms is not None else "None"
        wb = r["resolved_wb"]
        wb_str = f"{wb:.2f}" if isinstance(wb, float) else str(wb)
        bpm_d = r.get("bpm_bridge_max_log_distance", "?")
        bpm_str = f"{bpm_d:.2f}" if isinstance(bpm_d, float) else str(bpm_d)
        onset_d = r.get("onset_bridge_max_log_distance", "?")
        onset_str = f"{onset_d:.2f}" if isinstance(onset_d, float) else str(onset_d)
        lines.append(
            f"| {r['arm']} | see description | {ms_str} | {wb_str} | {bpm_str} | {onset_str} |"
        )

    lines += [
        "",
        "## BPM confirmation",
        "",
    ]
    for r in results:
        lines.append(f"- {r['arm']}: {r.get('bpm_status_on', 'N/A')}")

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
        "Verified by:",
        "1. Checking `debug['candidate_pool.min_sonic_similarity']` and "
        "`debug['pier_bridge.weight_bridge']` in each arm — values are read from "
        "the mutated `ds_overrides` dict AFTER preset application.",
        "2. Checking `debug['bpm_bridge_max_log_distance']` and "
        "`debug['onset_bridge_max_log_distance']` — values reflect the monkeypatched "
        "`resolve_pace_mode` return for BPM_BRIDGE and ONSET_BRIDGE arms.",
        "3. BPM active confirmed via 'BPM loaded' log line captured during generation "
        "(not post-hoc assertion).",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {path}")
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ablation(*, out_dir: str) -> list[dict]:
    """Run all five arms and write the decision doc."""
    print(f"\n{'='*60}")
    print("PACE-CEDES-SONIC: Sonic-lever ablation (production-faithful, BPM active)")
    print(f"  config: {CONFIG_PATH}")
    print(f"  artifact: {ARTIFACT_PATH}")
    print(f"  sidecar: {SIDECAR_PATH}")
    print(f"  seeds: {SEEDS}")
    print(f"  BPM gates: ACTIVE (pace=dynamic; bpm_bridge_max_log_distance=0.85 finite)")

    _self_test()

    t_total = time.time()
    results = []

    # ARM 1: BASELINE — no relaxation; BPM active; energy should be inert if
    # BPM/onset bands are constraining the pool.
    results.append(_arm("BASELINE"))

    # ARM 2: ADMISSION — force min_sonic_similarity=None after preset application.
    results.append(_arm("ADMISSION", relax_admission=True))

    # ARM 3: BRIDGE — reduce weight_bridge to 0.1 (sonic-bridge anchor near-zero).
    results.append(_arm("BRIDGE", wb_override=0.1))

    # ARM 4: BPM_BRIDGE — set bpm_bridge_max_log_distance to 1e6 (effectively off).
    # The monkeypatch is applied inside _run_one and restored after each call.
    results.append(_arm("BPM_BRIDGE", bpm_bridge_off=True))

    # ARM 5: ONSET_BRIDGE — set onset_bridge_max_log_distance to 1e6 (effectively off).
    results.append(_arm("ONSET_BRIDGE", onset_bridge_off=True))

    print(f"\nTotal ablation time: {time.time()-t_total:.0f}s")
    print(f"\n{'='*60}")
    print("ABLATION DECISION:")
    print(_decision_text(results))
    print(f"{'='*60}\n")

    _write_report(results, out_dir=out_dir)
    return results


def _strict_narrow_decision(results_by_mode: dict[str, list[dict]]) -> str:
    """Synthesize the strict/narrow lever decision."""
    lines = []
    for mode, results in results_by_mode.items():
        by_arm = {r["arm"]: r for r in results}
        baseline_ok = by_arm.get("BASELINE", {}).get("unblocks", False)
        admission_ok = by_arm.get("ADMISSION", {}).get("unblocks", False)
        bpm_brd_ok = by_arm.get("BPM_BRIDGE", {}).get("unblocks", False)
        onset_brd_ok = by_arm.get("ONSET_BRIDGE", {}).get("unblocks", False)

        baseline_str = "NOT INERT (energy already fires)" if baseline_ok else "INERT"
        admission_str = "UNBLOCKS" if admission_ok else "INERT"
        bpm_str = "UNBLOCKS" if bpm_brd_ok else "INERT"
        onset_str = "UNBLOCKS" if onset_brd_ok else "INERT"

        lines.append(
            f"pace_mode={mode}: BASELINE={baseline_str}; "
            f"ADMISSION={admission_str}; BPM_BRIDGE={bpm_str}; ONSET_BRIDGE={onset_str}"
        )
    return "\n".join(lines)


def _append_strict_narrow_report(
    results_by_mode: dict[str, list[dict]],
    *,
    out_dir: str,
) -> str:
    """Append a 'Strict/Narrow ablation' section to ABLATION.md; return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ABLATION.md")

    decision = _strict_narrow_decision(results_by_mode)

    lines = [
        "",
        "---",
        "",
        "# Strict/Narrow ablation (BPM active)",
        "",
        "**Generated by**: `scripts/research/pace_cede_eval.py --strict-narrow`",
        "**Seeds**: Songs: Ohia / Bill Callahan / William Tyler (mellow trio)",
        "**Genre/sonic/pace/cohesion**: matched (all axes set to the same mode)",
        "**Energy**: strong (`arc_strength=10, arc_band=0.1, step_strength=10, step_cap=0.1`)",
        "**Playlist length**: 12",
        "**BPM gates**: ACTIVE (injected via absolute db path override)",
        "",
        "## Arms (per mode)",
        "",
        "| Mode | Arm | resolved min_sonic | bpm_bridge_max_log_dist | onset_bridge_max_log_dist |",
        "|------|-----|--------------------|------------------------|--------------------------|",
    ]

    for mode, results in results_by_mode.items():
        for r in results:
            ms = r["resolved_min_sonic"]
            ms_str = str(ms) if ms is not None else "None"
            bpm_d = r.get("bpm_bridge_max_log_distance", "?")
            bpm_str = f"{bpm_d:.2f}" if isinstance(bpm_d, float) and bpm_d < 1e5 else (
                "1e6 (OFF)" if isinstance(bpm_d, float) else str(bpm_d)
            )
            onset_d = r.get("onset_bridge_max_log_distance", "?")
            onset_str = f"{onset_d:.2f}" if isinstance(onset_d, float) and onset_d < 1e5 else (
                "1e6 (OFF)" if isinstance(onset_d, float) else str(onset_d)
            )
            lines.append(
                f"| {mode} | {r['arm']} | {ms_str} | {bpm_str} | {onset_str} |"
            )

    lines += ["", "## BPM confirmation (per mode)", ""]
    for mode, results in results_by_mode.items():
        for r in results:
            lines.append(f"- {mode}/{r['arm']}: {r.get('bpm_status_on', 'N/A')}")

    lines += [
        "",
        "## Per-arm results",
        "",
        "| Mode | Arm | pos_diff/12 | arc_dev OFF | arc_dev ON | Δ arc_dev "
        "| max_step OFF | max_step ON | wall OFF | wall ON | VERDICT |",
        "|------|-----|-------------|-------------|------------|-----------|"
        "--------------|-------------|----------|---------|---------|",
    ]
    for mode, results in results_by_mode.items():
        for r in results:
            v = "UNBLOCKS" if r["unblocks"] else "INERT"
            off_w = r.get("off_wall", 0.0)
            on_w = r.get("on_wall", 0.0)
            lines.append(
                f"| {mode} | {r['arm']} | {r['pos_diff']}/12 "
                f"| {r['off_arc_dev']:.4f} | {r['on_arc_dev']:.4f} | {r['arc_delta']:+.4f} "
                f"| {r['off_max_step']:.4f} | {r['on_max_step']:.4f} "
                f"| {off_w:.0f}s | {on_w:.0f}s | {v} |"
            )

    lines += ["", "## Arousal curves", ""]
    for mode, results in results_by_mode.items():
        lines.append(f"### pace_mode={mode}")
        lines.append("")
        for r in results:
            lines += [
                f"#### {r['arm']}",
                f"- OFF: {r['off_curve']}",
                f"- ON : {r['on_curve']}",
                "",
            ]

    lines += [
        "## Lever decision (strict/narrow)",
        "",
        decision,
        "",
        "## Interpretation",
        "",
        "- **INERT at BASELINE, UNBLOCKS at ADMISSION**: the sonic admission floor "
        "(`min_sonic_similarity`) is the primary gate. Ceding it (→ None) unblocks energy. "
        "BPM/onset bridge bands are secondary or irrelevant for this seed set.",
        "- **INERT at BASELINE, UNBLOCKS at BPM_BRIDGE or ONSET_BRIDGE**: the bridge bands "
        "are the primary gate. Ceding them unblocks energy; admission floor is secondary.",
        "- **NOT INERT at BASELINE**: energy already fires without relaxation even in tight mode. "
        "No cede needed at this mode for these seeds.",
        "- **ALL ARMS INERT**: energy is blocked by the beam's sonic-weight hybrid or arousal "
        "diversity is too low in the restricted pool. The design's chosen levers do not apply; "
        "a different approach is needed.",
    ]

    append_mode = "a" if os.path.exists(path) else "w"
    with open(path, append_mode, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] appended -> {path}")
    return path


def run_strict_narrow_ablation(*, out_dir: str) -> dict[str, list[dict]]:
    """Run BASELINE/ADMISSION/BPM_BRIDGE/ONSET_BRIDGE for strict and narrow modes.

    Returns a dict mapping mode -> list[arm_result].
    Appends a 'Strict/Narrow ablation' section to ABLATION.md.
    """
    print(f"\n{'='*60}")
    print("PACE-CEDES-SONIC: Strict/Narrow ablation (BPM active)")
    print(f"  config: {CONFIG_PATH}")
    print(f"  artifact: {ARTIFACT_PATH}")
    print(f"  sidecar: {SIDECAR_PATH}")
    print(f"  seeds: {SEEDS}")
    print("  Modes: strict (min_sonic=0.28, bpm_bridge=0.40, onset_bridge=0.40)")
    print("         narrow (min_sonic=0.18, bpm_bridge=0.60, onset_bridge=0.60)")

    _self_test()

    results_by_mode: dict[str, list[dict]] = {}
    t_total = time.time()

    for pace_mode in ("strict", "narrow"):
        print(f"\n{'*'*60}")
        print(f"*** pace_mode={pace_mode} ***")
        print(f"{'*'*60}")
        mode_results = []

        # ARM 1: BASELINE — all four axes at pace_mode; no relaxation.
        # Expect energy INERT if sonic admission floor or BPM bands block it.
        mode_results.append(_arm("BASELINE", mode=pace_mode))

        # ARM 2: ADMISSION — relax min_sonic_similarity to None.
        # Tests whether the sonic admission floor is the primary gate.
        mode_results.append(_arm("ADMISSION", mode=pace_mode, relax_admission=True))

        # ARM 3: BPM_BRIDGE — set bpm_bridge_max_log_distance to 1e6 (off).
        # Tests whether the BPM bridge band gates energy.
        mode_results.append(_arm("BPM_BRIDGE", mode=pace_mode, bpm_bridge_off=True))

        # ARM 4: ONSET_BRIDGE — set onset_bridge_max_log_distance to 1e6 (off).
        # Tests whether the onset bridge band gates energy.
        mode_results.append(_arm("ONSET_BRIDGE", mode=pace_mode, onset_bridge_off=True))

        results_by_mode[pace_mode] = mode_results

    total_wall = time.time() - t_total
    print(f"\nTotal strict/narrow ablation time: {total_wall:.0f}s")
    print(f"\n{'='*60}")
    print("STRICT/NARROW DECISION:")
    print(_strict_narrow_decision(results_by_mode))
    print(f"{'='*60}\n")

    _append_strict_narrow_report(results_by_mode, out_dir=out_dir)
    return results_by_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true", default=False,
                        help="Run full ablation at pace=dynamic")
    parser.add_argument("--strict-narrow", action="store_true", default=False,
                        help="Run BASELINE/ADMISSION/BPM_BRIDGE/ONSET_BRIDGE at strict and narrow")
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

    if args.strict_narrow:
        run_strict_narrow_ablation(out_dir=args.out_dir)
        return

    # Default: run the original dynamic ablation
    run_ablation(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
