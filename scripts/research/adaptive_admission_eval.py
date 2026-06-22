"""Adaptive-admission calibration harness (Task 5).

Measures ONE (mode, seeds, sonic_pct, genre_pct, min_pool) cell and emits JSON.
Designed as the primitive a calibration Workflow fans out over a grid.

Override key path (confirmed from src/playlist/pipeline/core.py):
  overrides["pier_bridge"]["sonic_admission_percentile_{mode}"]
  overrides["pier_bridge"]["genre_admission_percentile_{mode}"]
  overrides["pier_bridge"]["min_pool_size_{mode}"]

  Mode-specific key takes priority over the base key (no-mode suffix).

Usage (CLI):
    python scripts/research/adaptive_admission_eval.py \\
        --mode narrow \\
        --seeds id1,id2,id3 \\
        --sonic-pct 0.6 \\
        --genre-pct 0.9 \\
        --min-pool 16

    python scripts/research/adaptive_admission_eval.py \\
        --mode narrow --niche hyperpop \\
        --sonic-pct 0.6 --genre-pct 0.9 --min-pool 16

All human log noise goes to stderr.  The single JSON result dict goes to stdout.

Emitted JSON keys:
    mode, sonic_pct, genre_pct, min_pool, admitted, distinct_artists,
    worst_edge_sonic, mean_edge_sonic, wall_time_s, bpm_loaded_ok, variant
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── repo root on sys.path so imports work from any cwd ───────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Absolute data paths — always point to main checkout ──────────────────────
_MAIN = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
ARTIFACT_PATH = str(_MAIN / "data/artifacts/beat3tower_32k/data_matrices_step1.npz")
DB_PATH = str(_MAIN / "data/metadata.db")
ENERGY_SIDECAR_PATH = str(
    _MAIN / "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
)
CONFIG_PATH = str(_ROOT / "config.yaml")

# ── Diverse seed corpus — multi-pier sets across ≥7 niches ───────────────────
# All track_ids verified against data/metadata.db (2026-06-21).
# 2-3 tracks per niche; artists chosen for distinctness within the niche.
SEED_CORPUS: dict[str, list[str]] = {
    "hyperpop": [
        "065933d8e2e0db664ec57af1511b662b",  # Charli XCX
        "4637b6d6b70e473818f58a474c6b0df4",  # Charli XCX
        "1dd3b8dbe12308e8db3b0f9b32124148",  # Caroline Polachek
    ],
    "metal": [
        "b7e5ea405f334893ee264f12ab308385",  # Metallica
        "d49ca10ff05db63ce7016dcffcd2e616",  # Black Sabbath
        "7325c318631a9ef072d3b96addeeb468",  # Mastodon
    ],
    "jazz": [
        "6d603651f47ae0d389e1fda1e3d4c171",  # Miles Davis
        "5afd7485a4446cf960603e76a51c70de",  # John Coltrane
        "ab3f750afa7a912ba3cb790bdaf4a559",  # Bill Evans
    ],
    "ambient": [
        "7615779f05003be157b54438486fc55c",  # Brian Eno
        "ff1dd673e37de21bf5fd58a46ee3e101",  # Stars of the Lid
        "6ef808a1ade6bdd2e725f08c85061af1",  # William Basinski
    ],
    "hip_hop": [
        "80524195bbc80f504dfe21264249976d",  # Kendrick Lamar
        "c1d1bdefe50dd84f55a70a0a018be477",  # Run the Jewels
        "d45e7c7e74ed1e8d960ba6f61430ee84",  # De La Soul
    ],
    "folk": [
        "57b98ac330eb456ffcfb488a52755961",  # Nick Drake
        "ebf9bbdeb7d4ba5279b1ac99232ec8bf",  # Fleet Foxes
    ],
    "mainstream_pop": [
        "75f252f3698b07e62132681ad3d51491",  # Ariana Grande
        "e0ce0e96d467ac1201dab61510cbb0cf",  # Dua Lipa
    ],
    # Known-fast 3-seed set (americana/folk) from test_gui_fidelity_regressions.py.
    # Reliably completes in ~45s at dynamic mode. Recommended for quick harness checks.
    "americana": [
        "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler
        "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan
        "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia
    ],
}


# ── BPM log capture ───────────────────────────────────────────────────────────

class _BpmCapture(logging.Handler):
    """Capture BPM-loaded / BPM-failed log lines during a generation run."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded_msg: Optional[str] = None
        self.n_loaded: int = 0
        self.n_total: int = 0
        self.failed: bool = False

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "BPM loaded:" in msg:
            self.loaded_msg = msg
            # parse "BPM loaded: N/M tracks have data"
            try:
                parts = msg.split("BPM loaded:")[-1].strip().split("/")
                self.n_loaded = int(parts[0].strip())
                self.n_total = int(parts[1].strip().split()[0])
            except Exception:
                pass
        elif "BPM load failed" in msg:
            self.failed = True


# ── Variant capture ───────────────────────────────────────────────────────────

class _VariantCapture(logging.Handler):
    """Capture 'Using precomputed sonic variant' log line."""

    def __init__(self) -> None:
        super().__init__()
        self.variant: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Using precomputed sonic variant" in msg:
            # "Using precomputed sonic variant 'mert' from artifact key X_sonic_mert"
            try:
                self.variant = msg.split("'")[1]
            except Exception:
                pass


# ── Admitted count capture ────────────────────────────────────────────────────

class _PoolCapture(logging.Handler):
    """Capture 'Candidate pool: mode=... admitted=N' log line."""

    def __init__(self) -> None:
        super().__init__()
        self.admitted: Optional[int] = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Candidate pool:" in msg and "admitted=" in msg:
            try:
                for part in msg.split():
                    if part.startswith("admitted="):
                        self.admitted = int(part.split("=")[1])
                        break
            except Exception:
                pass


# ── Worst-edge sonic metric ───────────────────────────────────────────────────

def worst_edge_sonic(
    track_ids: list[str],
    bundle: Any,
) -> tuple[float, float]:
    """Return (min_cosine, mean_cosine) over adjacent pairs in the playlist.

    Uses the bundle's active X_sonic (MERT) and track_id_to_index.
    Vectors are L2-normalised before cosine computation (same as pipeline).
    Returns (nan, nan) if fewer than 2 tracks or all indices missing.
    """
    idx_map = bundle.track_id_to_index
    X = bundle.X_sonic  # (N, D) — already the active MERT variant

    # Collect row indices for ordered track_ids
    rows: list[np.ndarray] = []
    for tid in track_ids:
        i = idx_map.get(str(tid))
        if i is not None:
            rows.append(X[i])

    if len(rows) < 2:
        return float("nan"), float("nan")

    mat = np.array(rows, dtype=np.float32)

    # L2-normalise rows (pipeline always works in normalised space)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms

    # Adjacent cosine similarities
    cosines = np.sum(mat[:-1] * mat[1:], axis=1)
    return float(np.min(cosines)), float(np.mean(cosines))


# ── Core measurement cell ─────────────────────────────────────────────────────

def measure_cell(
    mode: str,
    seed_ids: list[str],
    sonic_pct: Optional[float],
    genre_pct: Optional[float],
    min_pool: Optional[int],
) -> dict[str, Any]:
    """Run one calibration cell and return a metrics dict.

    Parameters
    ----------
    mode:
        cohesion/genre/sonic/pace mode string (e.g. "narrow").
    seed_ids:
        Multi-pier seeds (list of bundle track_ids).  Length ≥ 2 recommended.
    sonic_pct:
        sonic_admission_percentile value (0–1), or None to leave unset.
    genre_pct:
        genre_admission_percentile value (0–1), or None to leave unset.
    min_pool:
        min_pool_size backstop, or None to leave unset.

    Returns
    -------
    dict with keys:
        mode, sonic_pct, genre_pct, min_pool,
        admitted, distinct_artists,
        worst_edge_sonic, mean_edge_sonic,
        wall_time_s, bpm_loaded_ok, variant
    """
    from tests.support.gui_fidelity import resolve_gui_overrides, gui_ui_state
    from src.playlist.genre_ds_params import resolve_genre_ds_params
    from src.playlist_gui.policy import derive_runtime_config, merge_overrides
    from src.playlist_gui.worker import load_config_with_overrides
    from src.playlist.ds_pipeline_runner import generate_playlist_ds
    from src.features.artifacts import load_artifact_bundle

    # ── Build UI state: all four axes = mode ─────────────────────────────────
    ui = gui_ui_state(
        cohesion_mode=mode,
        genre_mode=mode,
        sonic_mode=mode,
        pace_mode=mode,
    )

    # ── Resolve overrides (GUI-faithful chain) ────────────────────────────────
    ds_overrides = resolve_gui_overrides(ui, config_path=CONFIG_PATH)

    # ── Inject real DB path so BPM loader skips the zero-byte worktree placeholder
    ds_overrides.setdefault("library", {})["database_path"] = DB_PATH

    # ── Inject adaptive-admission knobs into pier_bridge overrides ────────────
    # Key path: overrides["pier_bridge"]["sonic_admission_percentile_{mode}"]
    # (mode-specific key takes priority; confirmed in core.py lines 420-428)
    pb = ds_overrides.setdefault("pier_bridge", {})
    if sonic_pct is not None:
        pb[f"sonic_admission_percentile_{mode}"] = float(sonic_pct)
    if genre_pct is not None:
        pb[f"genre_admission_percentile_{mode}"] = float(genre_pct)
    if min_pool is not None:
        pb[f"min_pool_size_{mode}"] = int(min_pool)

    # ── Resolve genre params (gate + hybrid weights) ──────────────────────────
    decisions = derive_runtime_config(ui)
    raw_overrides = merge_overrides({}, decisions.overrides)
    merged = load_config_with_overrides(CONFIG_PATH, raw_overrides)
    playlists_cfg = merged.get("playlists", {}) or {}
    genre_params = resolve_genre_ds_params(playlists_cfg, ui.cohesion_mode)

    # ── Attach log handlers before generation ────────────────────────────────
    bpm_capture = _BpmCapture()
    variant_capture = _VariantCapture()
    pool_capture = _PoolCapture()

    core_logger = logging.getLogger("src.playlist.pipeline.core")
    artifact_logger = logging.getLogger("src.features.artifacts")
    pool_logger = logging.getLogger("src.playlist.candidate_pool")

    for handler in (bpm_capture, pool_capture):
        core_logger.addHandler(handler)
        if core_logger.level == logging.NOTSET or core_logger.level > logging.INFO:
            core_logger.setLevel(logging.INFO)

    artifact_logger.addHandler(variant_capture)
    _orig_art_level = artifact_logger.level
    if artifact_logger.level == logging.NOTSET or artifact_logger.level > logging.INFO:
        artifact_logger.setLevel(logging.INFO)

    pool_logger.addHandler(pool_capture)
    _orig_pool_level = pool_logger.level
    if pool_logger.level == logging.NOTSET or pool_logger.level > logging.INFO:
        pool_logger.setLevel(logging.INFO)

    t0 = time.time()
    try:
        result = generate_playlist_ds(
            artifact_path=ARTIFACT_PATH,
            seed_track_id=seed_ids[0],
            anchor_seed_ids=seed_ids,
            mode=mode,
            pace_mode=mode,
            length=30,
            random_seed=42,
            overrides=ds_overrides,
            artist_style_enabled=False,
            artist_playlist=False,
            **genre_params,
        )
    finally:
        for handler in (bpm_capture, pool_capture):
            core_logger.removeHandler(handler)
        artifact_logger.removeHandler(variant_capture)
        artifact_logger.setLevel(_orig_art_level)
        pool_logger.removeHandler(pool_capture)
        pool_logger.setLevel(_orig_pool_level)

    wall_time_s = time.time() - t0
    track_ids: list[str] = list(result.track_ids)

    # ── Compute worst/mean edge sonic using bundle ────────────────────────────
    bundle = load_artifact_bundle(ARTIFACT_PATH)
    w_sonic, m_sonic = worst_edge_sonic(track_ids, bundle)

    # ── Extract admitted count from log (fallback to pool stats) ─────────────
    admitted: Optional[int] = pool_capture.admitted
    if admitted is None:
        # try from result stats
        try:
            admitted = int(
                result.playlist_stats.get("candidate_pool", {}).get("admitted_count", 0)
            )
        except Exception:
            admitted = None

    # ── Distinct artist count ─────────────────────────────────────────────────
    distinct_artists: int = int(result.metrics.get("distinct_artists") or 0)
    if not distinct_artists:
        artist_counts = result.metrics.get("artist_counts") or {}
        distinct_artists = len(artist_counts)

    # ── BPM loaded? ───────────────────────────────────────────────────────────
    bpm_loaded_ok: bool = (
        not bpm_capture.failed
        and bpm_capture.loaded_msg is not None
        and bpm_capture.n_loaded > 0
    )

    # ── Detected sonic variant ────────────────────────────────────────────────
    variant: Optional[str] = variant_capture.variant

    return {
        "mode": mode,
        "sonic_pct": sonic_pct,
        "genre_pct": genre_pct,
        "min_pool": min_pool,
        "admitted": admitted,
        "distinct_artists": distinct_artists,
        "worst_edge_sonic": round(w_sonic, 4) if w_sonic == w_sonic else None,
        "mean_edge_sonic": round(m_sonic, 4) if m_sonic == m_sonic else None,
        "wall_time_s": round(wall_time_s, 1),
        "bpm_loaded_ok": bpm_loaded_ok,
        "variant": variant,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure one adaptive-admission calibration cell; emit JSON to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use explicit seed IDs
  python scripts/research/adaptive_admission_eval.py \\
      --mode narrow \\
      --seeds 065933d8e2e0db664ec57af1511b662b,4637b6d6b70e473818f58a474c6b0df4,1dd3b8dbe12308e8db3b0f9b32124148 \\
      --sonic-pct 0.6 --genre-pct 0.9 --min-pool 16

  # Use a named niche from SEED_CORPUS
  python scripts/research/adaptive_admission_eval.py \\
      --mode narrow --niche hyperpop \\
      --sonic-pct 0.6 --genre-pct 0.9 --min-pool 16

  # List available niches
  python scripts/research/adaptive_admission_eval.py --list-niches

Available niches: """ + ", ".join(SEED_CORPUS.keys()),
    )
    p.add_argument("--mode", default="narrow",
                   choices=["strict", "narrow", "dynamic", "discover", "off"],
                   help="cohesion/genre/sonic/pace mode (default: narrow)")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated track_ids to use as piers")
    p.add_argument("--niche", default=None, choices=list(SEED_CORPUS.keys()),
                   help="Use seeds from SEED_CORPUS for this niche")
    p.add_argument("--sonic-pct", type=float, default=None,
                   help="sonic_admission_percentile (0–1); None = unset")
    p.add_argument("--genre-pct", type=float, default=None,
                   help="genre_admission_percentile (0–1); None = unset")
    p.add_argument("--min-pool", type=int, default=None,
                   help="min_pool_size backstop; None = unset")
    p.add_argument("--list-niches", action="store_true",
                   help="Print available niches and exit")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_niches:
        for niche, ids in SEED_CORPUS.items():
            print(f"{niche}: {len(ids)} seeds", file=sys.stderr)
        sys.exit(0)

    # Resolve seed IDs
    seed_ids: list[str]
    if args.seeds:
        seed_ids = [s.strip() for s in args.seeds.split(",") if s.strip()]
    elif args.niche:
        seed_ids = SEED_CORPUS[args.niche]
    else:
        parser.error("Provide --seeds or --niche")

    if len(seed_ids) < 2:
        parser.error("Need at least 2 seed track_ids for multi-pier generation")

    # Route all library logging to stderr so stdout stays clean JSON
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # But keep pipeline core at INFO so we capture BPM/variant/pool lines
    for name in (
        "src.playlist.pipeline.core",
        "src.features.artifacts",
        "src.playlist.candidate_pool",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)

    print(
        f"[adaptive_admission_eval] mode={args.mode} seeds={len(seed_ids)} "
        f"sonic_pct={args.sonic_pct} genre_pct={args.genre_pct} min_pool={args.min_pool}",
        file=sys.stderr,
    )

    cell = measure_cell(
        mode=args.mode,
        seed_ids=seed_ids,
        sonic_pct=args.sonic_pct,
        genre_pct=args.genre_pct,
        min_pool=args.min_pool,
    )

    # Validate key invariants
    if not cell["bpm_loaded_ok"]:
        print("[WARN] bpm_loaded_ok=False — check DB path or pace_mode BPM gates", file=sys.stderr)
    if cell["variant"] != "mert":
        print(f"[WARN] variant={cell['variant']!r} (expected 'mert') — check artifact", file=sys.stderr)
    if (cell["admitted"] or 0) == 0:
        print("[WARN] admitted=0 — thresholds may be too tight", file=sys.stderr)

    # Emit single JSON object to stdout
    print(json.dumps(cell, indent=2))


if __name__ == "__main__":
    main()
