"""Slider-differentiation harness.

Answers, per axis (genre / sonic / pace / cohesion), three distinct questions on
diverse seeds (artist-mode AND seeds-mode):
  1. differentiation  - does sweeping the slider MOVE the playlist? (track overlap)
  2. direction        - does it move the RIGHT way? (the axis's own signal)
  3. quality floor     - does the worst LIVE transition (calibrated T, parsed from
                         the DS-success log line) stay sane? (no cratering)
                         Raw MuQ adjacent cosine is kept as a secondary signal.

Behavior-change (1,2) is separate from quality (3): a dead slider hides behind a
good-anyway playlist; a differentiating slider can still produce garbage.

FAITHFULNESS: modes are applied through the real policy layer
(derive_runtime_config -> merge_overrides -> load_config_with_overrides), exactly
as the worker does. Setting playlists.*_mode strings directly does NOT translate
them into knobs — the validation slice proved that (gates identical across modes).

Provenance guard (evaluation-methodology pre-flight): assert artifact
X_sonic_variant=='muq' once at startup, and BPM loaded per cell.
(MuQ caveat: quiet/near-silent tracks collapse to ~one vector — raw cosine means
can be inflated by such pairs; the live T + overlap metrics are primary.)

  python scripts/research/slider_differentiation_eval.py --artist Codeine --axis genre
  python scripts/research/slider_differentiation_eval.py --grid artist
"""
from __future__ import annotations
import argparse
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")  # DATA anchor (artifact/DB/config live here)
CODE_ROOT = Path(__file__).resolve().parents[2]  # CODE anchor: THIS script's repo root (worktree) — test this tree's src, not MAIN's
ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
DB = ROOT / "data/metadata.db"
ENERGY = ROOT / "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
CONFIG = ROOT / "config.yaml"
LOGF = ROOT / "_sde.log"

# Per-axis mode values (ui_state.py Literals): pace has no 'discover';
# cohesion has no 'off'. 'dynamic' is the shared baseline for overlap.
AXIS_MODES = {
    "genre": ["strict", "narrow", "dynamic", "discover", "off"],
    "sonic": ["strict", "narrow", "dynamic", "discover", "off"],
    "pace": ["strict", "narrow", "dynamic", "off"],
    "cohesion": ["strict", "narrow", "dynamic", "discover"],
}
ARTIST_CORPUS = ["Bill Evans Trio", "Codeine", "Herbie Hancock", "Yo La Tengo", "Deerhunter", "Modest Mouse"]
MAIN_OV = {
    "library": {"database_path": str(DB)},
    "playlists": {"ds_pipeline": {"artifact_path": str(ARTIFACT)}},
    "energy": {"sidecar_path": str(ENERGY)},
}

# ---- artifact (loaded once for metrics) -------------------------------------
_ART: dict[str, Any] = {}


def art() -> dict[str, Any]:
    if not _ART:
        a = np.load(ARTIFACT, allow_pickle=True)
        ids = [str(t) for t in a["track_ids"]]
        _ART["id2idx"] = {t: i for i, t in enumerate(ids)}
        _ART["sonic"] = np.asarray(a["X_sonic_muq"], dtype=np.float32)
        _ART["artists"] = [str(x) for x in a["track_artists"]] if "track_artists" in a.files else None
        _ART["genre"] = np.asarray(a["X_genre_raw"], dtype=np.float32)
        _ART["vocab"] = [str(x) for x in a["genre_vocab"]]
        # BPM/onset live in the DB post-SP-B (no bpm_array in the artifact).
        # Reuse the production loader so the direction metric matches the gates.
        sys.path.insert(0, str(CODE_ROOT))
        from src.playlist.bpm_loader import load_bpm_arrays
        arrs = load_bpm_arrays(np.asarray(ids, dtype=object), db_path=str(DB))
        _ART["bpm"] = arrs["perceptual_bpm"]
        _ART["onset"] = arrs["onset_rate"]
    return _ART


def _norm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def playlist_metrics(track_ids: list[str]) -> dict[str, Any]:
    A = art()
    idx = [A["id2idx"][t] for t in track_ids if t in A["id2idx"]]
    out: dict[str, Any] = {"n": len(idx)}
    if len(idx) >= 2:
        M = _norm(np.asarray(A["sonic"][idx], dtype=np.float64))
        adj = np.einsum("ij,ij->i", M[:-1], M[1:])
        out["sonic_mean"] = round(float(adj.mean()), 3)
        out["sonic_worst"] = round(float(adj.min()), 3)
    if A["bpm"] is not None and idx:
        b = A["bpm"][idx]
        b = b[np.isfinite(b) & (b > 0)]
        if b.size:
            out["bpm_mean"] = round(float(b.mean()), 1)
            out["bpm_std"] = round(float(b.std()), 1)
    if A.get("onset") is not None and idx:
        o = A["onset"][idx]
        o = o[np.isfinite(o)]
        if o.size:
            out["onset_std"] = round(float(o.std()), 2)
    if A["artists"] is not None and idx:
        out["distinct_artists"] = len({A["artists"][i] for i in idx})
    if idx:
        gp = np.asarray(A["genre"][idx], dtype=np.float64).sum(axis=0)
        top = np.argsort(gp)[::-1][:5]
        out["genre_top"] = [A["vocab"][j] for j in top if gp[j] > 0]
    return out


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return round(len(sa & sb) / len(sa | sb), 3)


# ---- faithful generation via the policy layer -------------------------------
_IMP: dict[str, Any] = {}


def _imp() -> dict[str, Any]:
    if not _IMP:
        sys.path.insert(0, str(CODE_ROOT))
        from src.playlist_gui.ui_state import UIStateModel
        from src.playlist_gui.policy import derive_runtime_config, merge_overrides
        from src.playlist_gui.worker import load_config_with_overrides
        from src.local_library_client import LocalLibraryClient
        from src.playlist_generator import PlaylistGenerator
        from src.track_matcher import TrackMatcher
        from src.metadata_client import MetadataClient
        _IMP.update(
            UIStateModel=UIStateModel, derive_runtime_config=derive_runtime_config,
            merge_overrides=merge_overrides, load_config_with_overrides=load_config_with_overrides,
            LocalLibraryClient=LocalLibraryClient, PlaylistGenerator=PlaylistGenerator,
            TrackMatcher=TrackMatcher, MetadataClient=MetadataClient,
        )
    return _IMP


def policy_config(genre_m: str, sonic_m: str, pace_m: str, cohesion_m: str, roam: dict | None = None, extra_ov: dict | None = None) -> dict:
    """Translate the four slider modes into a merged config exactly as the worker
    does (this is the step the broken first draft skipped).

    ``roam`` (optional) injects the Phase-1 roam-corridor override at
    playlists.ds_pipeline.pier_bridge.roam, which build_ds_overrides surfaces as
    overrides["pier_bridge"]["roam"] -> apply_pier_bridge_overrides -> PierBridgeConfig.
    """
    I = _imp()
    ui = replace(
        I["UIStateModel"](mode="seeds"),
        cohesion_mode=cohesion_m, genre_mode=genre_m, sonic_mode=sonic_m,
        pace_mode=pace_m, artist_spacing="strong",
    )
    decisions = I["derive_runtime_config"](ui, seed_artist_keys=None)
    ov = I["merge_overrides"]({}, decisions.overrides)
    ov = I["merge_overrides"](ov, MAIN_OV)  # MAIN data paths win
    if roam:
        ov = I["merge_overrides"](ov, {"playlists": {"ds_pipeline": {"pier_bridge": {"roam": roam}}}})
    if extra_ov:
        ov = I["merge_overrides"](ov, extra_ov)
    return I["load_config_with_overrides"](str(CONFIG), ov)


def build_generator(merged_cfg: dict):
    I = _imp()

    class MC:
        def __init__(s, c):
            s.config = c
            s.config_path = str(CONFIG)

        def get(s, section, key=None, default=None):
            if section not in s.config:
                return default
            return s.config.get(section, default) if key is None else s.config[section].get(key, default)

        @property
        def library_database_path(s):
            return s.config["library"]["database_path"]

        @property
        def library_music_directory(s):
            return s.config["library"].get("music_directory", "E:/MUSIC")

        @property
        def lastfm_api_key(s):
            import os
            return os.getenv("LASTFM_API_KEY") or s.config.get("lastfm", {}).get("api_key", "")

        @property
        def lastfm_username(s):
            import os
            return os.getenv("LASTFM_USERNAME") or s.config.get("lastfm", {}).get("username", "")

    mc = MC(merged_cfg)
    lib = I["LocalLibraryClient"](db_path=mc.library_database_path)
    matcher = I["TrackMatcher"](lib, library_id=None, db_path=mc.library_database_path)
    try:
        meta = I["MetadataClient"](mc.library_database_path)
    except Exception:
        meta = None
    return I["PlaylistGenerator"](lib, mc, lastfm_client=None, track_matcher=matcher, metadata_client=meta)


def _run_and_parse(cfg_fn, gen_call) -> dict[str, Any]:
    """Build the merged config via `cfg_fn()` INSIDE the log-capture window (the
    mode-preset echoes fire in Config.__init__), build a generator, run
    `gen_call(g)` (artist or seeds mode), and parse the standard cell metrics."""
    open(LOGF, "w").close()
    fh = logging.FileHandler(LOGF, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(fh)
    root.setLevel(logging.INFO)
    t0 = time.time()
    err = None
    track_ids: list[str] = []
    try:
        g = build_generator(cfg_fn())
        res = gen_call(g)
        tracks = res.get("tracks", []) if isinstance(res, dict) else []
        track_ids = [str(t.get("rating_key") or t.get("id") or t.get("track_id") or "") for t in tracks]
        track_ids = [t for t in track_ids if t]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    wall = round(time.time() - t0, 1)
    root.removeHandler(fh)
    fh.close()
    log = LOGF.read_text(encoding="utf-8", errors="ignore")

    def grp(pat, cast=str, default=None):
        mm = re.search(pat, log)
        return cast(mm.group(1)) if mm else default

    m = re.search(r"BPM loaded: (\d+)/(\d+)", log)
    out = {
        "wall": wall, "err": err, "bpm_ok": bool(m) and int(m.group(1)) > 0, "track_ids": track_ids,
        "r_genre": grp(r"Genre mode '(\w+)'"),
        "r_sonic": grp(r"Sonic mode '(\w+)'"),
        "r_pace": grp(r"Pace mode:? '?(\w+)"),
        "r_cohesion": grp(r"Running pipeline with mode=(\w+)"),
        "bridge_floor": grp(r"attempt 1: bridge_floor=([0-9.]+)", float),
        "admitted": grp(r"Candidate pool:.*?admitted=(\d+)", int),
        "rej_genre": grp(r"Candidate pool:.*?rejected_genre=(\d+)", int),
        "sonic_floor": grp(r"effective sonic_floor=([\d.]+)", float),
        "genre_pct": grp(r"Genre admission percentile.*?p=([\d.]+)", float),
        "min_genre_sim": grp(r"min_genre_sim=([\d.]+)", float),
    }
    # Roam diagnostics (Phase-1): per-segment "Roam[seg N]: sonic detour mean=.. max=.."
    _rl = re.findall(r"Roam\[seg \d+\]: sonic detour mean=([\d.]+) max=([\d.]+)", log)
    out["roam_segs"] = len(_rl)
    out["roam_detour_mean"] = round(float(np.mean([float(x[0]) for x in _rl])), 3) if _rl else None
    # Worst live edge (calibrated T): "DS pipeline success ... min_transition=0.240"
    # (also matches the older '"min_transition": 0.240' JSON form).
    out["min_transition"] = grp(r'min_transition[="\s:]+([0-9.]+)', float)
    out["mean_transition"] = grp(r'mean_transition[="\s:]+([0-9.]+)', float)
    _A = art()
    out["artists"] = (
        sorted({_A["artists"][_A["id2idx"][t]].strip().lower() for t in track_ids if t in _A["id2idx"]})
        if _A.get("artists") else []
    )
    if not err:
        out.update(playlist_metrics(track_ids))
    return out


def run_artist_cell(artist: str, genre_m: str, sonic_m: str, pace_m: str, cohesion_m: str, length: int = 30, roam: dict | None = None, extra_ov: dict | None = None) -> dict[str, Any]:
    return _run_and_parse(
        lambda: policy_config(genre_m, sonic_m, pace_m, cohesion_m, roam=roam, extra_ov=extra_ov),
        lambda g: g.create_playlist_for_artist(
            artist, track_count=length, dynamic=(cohesion_m == "dynamic"), cohesion_mode_override=cohesion_m))


def run_seed_cell(seed_ids: list[str], length: int = 30, roam: dict | None = None, extra_ov: dict | None = None) -> dict[str, Any]:
    """Diverse-seed (seeds-mode) generation from explicit track IDs."""
    disp = [f"seed{i}" for i in range(len(seed_ids))]
    return _run_and_parse(
        lambda: policy_config("dynamic", "dynamic", "dynamic", "dynamic", roam=roam, extra_ov=extra_ov),
        lambda g: g.create_playlist_from_seed_tracks(
            seed_tracks=disp, track_count=length, dynamic=True,
            cohesion_mode_override="dynamic", seed_track_ids=list(seed_ids)))


def sweep_artist_axis(artist: str, axis: str, length: int = 30) -> None:
    modes = AXIS_MODES[axis]
    print(f"\n=== ARTIST-MODE  {artist}  | sweep {axis} (others=dynamic) ===")
    cells: dict[str, dict[str, Any]] = {}
    for m in modes:
        gm = m if axis == "genre" else "dynamic"
        sm = m if axis == "sonic" else "dynamic"
        pm = m if axis == "pace" else "dynamic"
        cm = m if axis == "cohesion" else "dynamic"
        cells[m] = run_artist_cell(artist, gm, sm, pm, cm, length)
    base = cells["dynamic"].get("track_ids", [])
    print("  -- resolved (did the mode move the gates?) --")
    print(f"  {'mode':8s} {'g/s/p/c resolved':26s} {'admitted':>8s} {'rejGenre':>8s} {'sonicFloor':>10s} {'brFloor':>7s} {'minGenSim':>9s}")
    for m in modes:
        c = cells[m]
        if c.get("err"):
            continue
        res = f"{c.get('r_genre')}/{c.get('r_sonic')}/{c.get('r_pace')}/{c.get('r_cohesion')}"
        print(f"  {m:8s} {res:26s} {str(c.get('admitted')):>8s} {str(c.get('rej_genre')):>8s} {str(c.get('sonic_floor')):>10s} {str(c.get('bridge_floor')):>7s} {str(c.get('min_genre_sim')):>9s}")
    print("  -- outcome (did the playlist change, the right way, without cratering?) --")
    print(f"  {'mode':8s} {'n':>3s} {'overlapVdyn':>11s} {'minT':>6s} {'sonicMean':>9s} {'sonicWorst':>10s} {'bpmStd':>6s} {'onsetStd':>8s} {'distArt':>7s} {'wall':>5s} bpm | genre_top")
    for m in modes:
        c = cells[m]
        if c.get("err"):
            print(f"  {m:8s} ERR {c['err'][:80]}")
            continue
        ov = jaccard(c.get("track_ids", []), base)
        mt = c.get("min_transition")
        mt = f"{mt:.3f}" if isinstance(mt, float) else str(mt)
        print(f"  {m:8s} {c.get('n',0):>3d} {ov:>11.3f} {mt:>6} {str(c.get('sonic_mean')):>9} {str(c.get('sonic_worst')):>10} {str(c.get('bpm_std')):>6} {str(c.get('onset_std')):>8} {c.get('distinct_artists',0):>7d} {c['wall']:>5} {'ok' if c['bpm_ok'] else 'NOBPM':3s} | {','.join(c.get('genre_top',[])[:4])}")


def sweep_artist_roam(artist: str, length: int = 30) -> None:
    """Roam Corridors proof-of-life: baseline (roam off) vs a sonic-corridor width
    sweep (genre/energy widths 0). Does opening the sonic corridor MOVE the playlist
    (overlap drops) while the worst edge (minT) holds? Are the Roam[seg] diagnostics
    firing? All other dials = dynamic; modes routed through the real policy layer.
    """
    print(f"\n=== ARTIST-MODE ROAM  {artist}  | sonic-corridor sweep (width_genre=width_energy=0) ===")
    cells: dict[str, dict[str, Any]] = {}
    cells["off"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length, roam=None)
    for w in (0.5, 1.0, 2.0):
        roam = {"enabled": True, "width_sonic": float(w), "width_genre": 0.0, "width_energy": 0.0}
        cells[f"w{w}"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length, roam=roam)
    # Same corridors WITH the min-bottleneck worst-edge guard: does it recover minT?
    for w in (0.5, 1.0):
        roam = {"enabled": True, "width_sonic": float(w), "width_genre": 0.0, "width_energy": 0.0,
                "worst_edge_minimax": True}
        cells[f"w{w}+mm"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length, roam=roam)
    base = cells["off"].get("track_ids", [])
    print(f"  {'cell':6s} {'n':>3s} {'overlapVoff':>11s} {'sonicMean':>9s} {'sonicWorst':>10s} {'minT':>5s} {'roamSegs':>8s} {'detourMu':>8s} {'wall':>5s} bpm")
    for k, c in cells.items():
        if c.get("err"):
            print(f"  {k:6s} ERR {c['err'][:90]}")
            continue
        ov = jaccard(c.get("track_ids", []), base)
        print(f"  {k:6s} {c.get('n', 0):>3d} {ov:>11.3f} {str(c.get('sonic_mean')):>9} {str(c.get('sonic_worst')):>10} {str(c.get('min_transition')):>5} {c.get('roam_segs', 0):>8d} {str(c.get('roam_detour_mean')):>8} {c['wall']:>5} {'ok' if c['bpm_ok'] else 'NOBPM'}")


def sweep_genre_broad(artist: str, targets: list[str], length: int = 30) -> None:
    """Genre is squishy: does relaxing the genre gate + the sonic corridor let
    sonically-valid but differently-tagged neighbors (The Radio Dept <-> The
    Embassy) co-occur while the worst edge holds? Compares off / roam+minimax
    (genre gate on) / roam+minimax with the genre gate OFF (broad, sonic-led pool).
    """
    roam = {"enabled": True, "width_sonic": 1.0, "width_genre": 0.0, "width_energy": 0.0,
            "worst_edge_minimax": True}
    # Genre-broad: disable the DENSE per-seed genre gate (genre_admission_percentile)
    # + the beam genre floors. Sonic-broad: also drop the sonic admission percentile.
    # core.py reads the MODE-specific key (genre_admission_percentile_<mode>) first,
    # so the base key alone is ignored — set both. mode is dynamic here.
    gate_off = {"playlists": {"ds_pipeline": {"pier_bridge": {
        "genre_admission_percentile": 0.0, "genre_admission_percentile_dynamic": 0.0,
        "genre_arc_floor": 0.0, "genre_pair_floor": 0.0,
    }}, "genre_similarity": {"enabled": False}}}
    sonic_off = {"playlists": {"ds_pipeline": {"pier_bridge": {
        "genre_admission_percentile": 0.0, "genre_admission_percentile_dynamic": 0.0,
        "genre_arc_floor": 0.0, "genre_pair_floor": 0.0,
        "sonic_admission_percentile": 0.0, "sonic_admission_percentile_dynamic": 0.0,
    }}, "genre_similarity": {"enabled": False}}}
    cells: dict[str, dict[str, Any]] = {}
    cells["off"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length)
    cells["roam+mm"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length, roam=roam)
    cells["roam+genreBroad"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length, roam=roam, extra_ov=gate_off)
    cells["roam+sonicBroad"] = run_artist_cell(artist, "dynamic", "dynamic", "dynamic", "dynamic", length, roam=roam, extra_ov=sonic_off)
    base = cells["off"].get("track_ids", [])
    tnorm = [t.strip().lower() for t in targets]
    print(f"\n=== GENRE-BROAD PROBE  {artist}  | targets={targets} ===")
    print(f"  {'cell':16s} {'n':>3s} {'ovVoff':>7s} {'sonicWorst':>10s} {'minT':>5s} {'rejG':>5s} {'adm':>5s} {'wall':>5s} | targets_present")
    for k, c in cells.items():
        if c.get("err"):
            print(f"  {k:16s} ERR {c['err'][:80]}")
            continue
        ov = jaccard(c.get("track_ids", []), base)
        arts = set(c.get("artists", []))
        present = [t for t, tn in zip(targets, tnorm) if tn in arts]
        print(f"  {k:16s} {c.get('n', 0):>3d} {ov:>7.3f} {str(c.get('sonic_worst')):>10} {str(c.get('min_transition')):>5} {str(c.get('rej_genre')):>5} {str(c.get('admitted')):>5} {c['wall']:>5} | {present}")


ROAM_MM = {"enabled": True, "width_sonic": 1.0, "width_genre": 0.0, "width_energy": 0.0, "worst_edge_minimax": True}


def _gate_ov(dense_pct: float, *, genre_sim: bool = True, sonic_pct: float | None = None) -> dict:
    pb = {"genre_admission_percentile": float(dense_pct), "genre_admission_percentile_dynamic": float(dense_pct)}
    if sonic_pct is not None:
        pb["sonic_admission_percentile"] = float(sonic_pct)
        pb["sonic_admission_percentile_dynamic"] = float(sonic_pct)
    ov: dict = {"playlists": {"ds_pipeline": {"pier_bridge": pb}}}
    if not genre_sim:
        ov["playlists"]["genre_similarity"] = {"enabled": False}
    return ov


def first_track_for_artist(name: str) -> str | None:
    A = art()
    if not A.get("artists"):
        return None
    nl = name.strip().lower()
    idx2id = {i: t for t, i in A["id2idx"].items()}
    for i, ar in enumerate(A["artists"]):
        if ar.strip().lower() == nl:
            return idx2id.get(i)
    return None


def sweep_gates(label: str, run_fn, length: int = 30) -> None:
    """Cohesion/variety trade-off across gate settings, for one playlist type.
    run_fn(roam, extra_ov) -> cell. Sonic North Star is constant; we vary how hard
    genre gates the pool: default dense gate -> light -> dense-off(graph kept) -> no-genre.
    """
    settings = [
        ("off", None, None),
        ("roam+mm", ROAM_MM, None),                                  # default dense gate (~0.85)
        ("+lightGate", ROAM_MM, _gate_ov(0.5)),                      # looser dense gate
        ("+noDense(graph)", ROAM_MM, _gate_ov(0.0)),                 # dense off; graph + sonic kept
        ("+noGenre", ROAM_MM, _gate_ov(0.0, genre_sim=False, sonic_pct=0.0)),  # over-broad reference
    ]
    cells = {name: run_fn(roam, ov) for name, roam, ov in settings}
    base = cells["off"].get("track_ids", [])
    print(f"\n=== GATE SWEEP  {label}  (length {length}) ===")
    print(f"  {'setting':16s} {'n':>3s} {'ovOff':>6s} {'sonMean':>7s} {'sonWorst':>8s} {'minT':>6s} {'distArt':>7s} {'pool':>5s} {'wall':>5s}")
    for name, c in cells.items():
        if c.get("err"):
            print(f"  {name:16s} ERR {c['err'][:70]}")
            continue
        ov = jaccard(c.get("track_ids", []), base)
        mt = c.get("min_transition")
        mt = f"{mt:.3f}" if isinstance(mt, float) else str(mt)
        print(f"  {name:16s} {c.get('n', 0):>3d} {ov:>6.3f} {str(c.get('sonic_mean')):>7} {str(c.get('sonic_worst')):>8} {mt:>6} {c.get('distinct_artists', 0):>7d} {str(c.get('admitted')):>5} {c['wall']:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="Codeine")
    ap.add_argument("--axis", default="genre", choices=["genre", "sonic", "pace", "cohesion"])
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--grid", choices=["artist"], default=None)
    ap.add_argument("--roam", action="store_true", help="Roam-corridor sonic-width sweep (proof-of-life)")
    ap.add_argument("--genre-broad", action="store_true", help="genre-gate-off broad-pool probe")
    ap.add_argument("--targets", default="", help="comma-separated target artists to check for presence")
    ap.add_argument("--gate-sweep", action="store_true", help="sweep gate settings (cohesion/variety trade-off)")
    ap.add_argument("--seed-artists", default="", help="comma-separated artists -> 1 track each, for a diverse-seed sweep")
    args = ap.parse_args()
    bc = np.load(ARTIFACT, allow_pickle=True)
    variant = str(bc["X_sonic_variant"]) if "X_sonic_variant" in bc.files else "?"
    gsrc = bc["build_config"].item().get("genre_source") if "build_config" in bc.files else "?"
    print(f"PRE-FLIGHT: artifact X_sonic_variant={variant!r} genre_source={gsrc!r}")
    if variant != "muq":
        print(f"ABORT: live sonic variant is {variant!r}, not 'muq' — re-fold before calibrating.")
        return
    if args.gate_sweep:
        if args.seed_artists:
            arts = [a.strip() for a in args.seed_artists.split(",") if a.strip()]
            ids = [tid for a in arts if (tid := first_track_for_artist(a))]
            print(f"Diverse seeds: {len(ids)}/{len(arts)} resolved from {arts}")
            sweep_gates(f"SEEDS[{','.join(arts)}]", lambda roam, ov: run_seed_cell(ids, args.length, roam=roam, extra_ov=ov), args.length)
        else:
            sweep_gates(f"ARTIST[{args.artist}]", lambda roam, ov: run_artist_cell(args.artist, "dynamic", "dynamic", "dynamic", "dynamic", args.length, roam=roam, extra_ov=ov), args.length)
    elif args.genre_broad:
        sweep_genre_broad(args.artist, [t.strip() for t in args.targets.split(",") if t.strip()], args.length)
    elif args.roam:
        sweep_artist_roam(args.artist, args.length)
    elif args.grid == "artist":
        for a in ARTIST_CORPUS:
            for ax in ("genre", "sonic", "pace", "cohesion"):
                sweep_artist_axis(a, ax, args.length)
    else:
        sweep_artist_axis(args.artist, args.axis, args.length)


if __name__ == "__main__":
    main()
