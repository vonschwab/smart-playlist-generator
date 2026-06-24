"""Slider-differentiation harness.

Answers, per axis (genre / sonic / pace), three distinct questions on diverse
seeds (artist-mode AND seeds-mode):
  1. differentiation  - does sweeping the slider MOVE the playlist? (track overlap)
  2. direction        - does it move the RIGHT way? (the axis's own signal)
  3. quality floor     - does the worst MERT edge stay sane? (no cratering)

Behavior-change (1,2) is separate from quality (3): a dead slider hides behind a
good-anyway playlist; a differentiating slider can still produce garbage.

FAITHFULNESS: modes are applied through the real policy layer
(derive_runtime_config -> merge_overrides -> load_config_with_overrides), exactly
as the worker does. Setting playlists.*_mode strings directly does NOT translate
them into knobs — the validation slice proved that (gates identical across modes).

Provenance guard (evaluation-methodology pre-flight): assert build_config
X_sonic_variant=='mert' once at startup, and BPM loaded per cell.

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

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
DB = ROOT / "data/metadata.db"
ENERGY = ROOT / "data/artifacts/beat3tower_32k/energy/energy_sidecar.npz"
CONFIG = ROOT / "config.yaml"
LOGF = ROOT / "_sde.log"

MODES = ["strict", "narrow", "dynamic", "off"]
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
        _ART["mert"] = np.asarray(a["X_sonic_mert"], dtype=np.float32)
        _ART["bpm"] = np.asarray(a["bpm_array"], dtype=np.float64) if "bpm_array" in a.files else None
        _ART["artists"] = [str(x) for x in a["track_artists"]] if "track_artists" in a.files else None
        _ART["genre"] = np.asarray(a["X_genre_raw"], dtype=np.float32)
        _ART["vocab"] = [str(x) for x in a["genre_vocab"]]
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
        M = _norm(np.asarray(A["mert"][idx], dtype=np.float64))
        adj = np.einsum("ij,ij->i", M[:-1], M[1:])
        out["sonic_mean"] = round(float(adj.mean()), 3)
        out["sonic_worst"] = round(float(adj.min()), 3)
    if A["bpm"] is not None and idx:
        b = A["bpm"][idx]
        b = b[np.isfinite(b) & (b > 0)]
        if b.size:
            out["bpm_mean"] = round(float(b.mean()), 1)
            out["bpm_std"] = round(float(b.std()), 1)
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
        sys.path.insert(0, str(ROOT))
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


def policy_config(genre_m: str, sonic_m: str, pace_m: str, cohesion_m: str) -> dict:
    """Translate the four slider modes into a merged config exactly as the worker
    does (this is the step the broken first draft skipped)."""
    I = _imp()
    ui = replace(
        I["UIStateModel"](mode="seeds"),
        cohesion_mode=cohesion_m, genre_mode=genre_m, sonic_mode=sonic_m,
        pace_mode=pace_m, artist_spacing="strong",
    )
    decisions = I["derive_runtime_config"](ui, seed_artist_keys=None)
    ov = I["merge_overrides"]({}, decisions.overrides)
    ov = I["merge_overrides"](ov, MAIN_OV)  # MAIN data paths win
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


def run_artist_cell(artist: str, genre_m: str, sonic_m: str, pace_m: str, cohesion_m: str, length: int = 30) -> dict[str, Any]:
    merged = policy_config(genre_m, sonic_m, pace_m, cohesion_m)
    g = build_generator(merged)
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
        res = g.create_playlist_for_artist(
            artist, track_count=length, dynamic=(cohesion_m == "dynamic"), cohesion_mode_override=cohesion_m
        )
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
        "r_genre": grp(r"Genre mode: (\w+)"),
        "r_sonic": grp(r"Sonic mode: (\w+)"),
        "r_pace": grp(r"Pace mode:? '?(\w+)"),
        "r_cohesion": grp(r"Cohesion mode: (\w+)"),
        "admitted": grp(r"Candidate pool:.*?admitted=(\d+)", int),
        "rej_genre": grp(r"Candidate pool:.*?rejected_genre=(\d+)", int),
        "sonic_floor": grp(r"effective sonic_floor=([\d.]+)", float),
        "genre_pct": grp(r"Genre admission percentile.*?p=([\d.]+)", float),
        "min_genre_sim": grp(r"min_genre_sim=([\d.]+)", float),
    }
    if not err:
        out.update(playlist_metrics(track_ids))
    return out


def sweep_artist_axis(artist: str, axis: str, length: int = 30) -> None:
    print(f"\n=== ARTIST-MODE  {artist}  | sweep {axis} (others=dynamic) ===")
    cells: dict[str, dict[str, Any]] = {}
    for m in MODES:
        gm = m if axis == "genre" else "dynamic"
        sm = m if axis == "sonic" else "dynamic"
        pm = m if axis == "pace" else "dynamic"
        cells[m] = run_artist_cell(artist, gm, sm, pm, "dynamic", length)
    base = cells["dynamic"].get("track_ids", [])
    print("  -- resolved (did the mode move the gates?) --")
    print(f"  {'mode':8s} {'g/s/p/c resolved':22s} {'admitted':>8s} {'rejGenre':>8s} {'sonicFloor':>10s} {'genrePct':>8s} {'minGenSim':>9s}")
    for m in MODES:
        c = cells[m]
        if c.get("err"):
            continue
        res = f"{c.get('r_genre')}/{c.get('r_sonic')}/{c.get('r_pace')}/{c.get('r_cohesion')}"
        print(f"  {m:8s} {res:22s} {str(c.get('admitted')):>8s} {str(c.get('rej_genre')):>8s} {str(c.get('sonic_floor')):>10s} {str(c.get('genre_pct')):>8s} {str(c.get('min_genre_sim')):>9s}")
    print("  -- outcome (did the playlist change, the right way, without cratering?) --")
    print(f"  {'mode':8s} {'n':>3s} {'overlapVdyn':>11s} {'sonicMean':>9s} {'sonicWorst':>10s} {'bpmStd':>6s} {'distArt':>7s} {'wall':>5s} bpm | genre_top")
    for m in MODES:
        c = cells[m]
        if c.get("err"):
            print(f"  {m:8s} ERR {c['err'][:80]}")
            continue
        ov = jaccard(c.get("track_ids", []), base)
        print(f"  {m:8s} {c.get('n',0):>3d} {ov:>11.3f} {str(c.get('sonic_mean')):>9} {str(c.get('sonic_worst')):>10} {str(c.get('bpm_std')):>6} {c.get('distinct_artists',0):>7d} {c['wall']:>5} {'ok' if c['bpm_ok'] else 'NOBPM':3s} | {','.join(c.get('genre_top',[])[:4])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="Codeine")
    ap.add_argument("--axis", default="genre", choices=["genre", "sonic", "pace"])
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--grid", choices=["artist"], default=None)
    args = ap.parse_args()
    bc = np.load(ARTIFACT, allow_pickle=True)
    variant = str(bc["X_sonic_variant"]) if "X_sonic_variant" in bc.files else "?"
    gsrc = bc["build_config"].item().get("genre_source") if "build_config" in bc.files else "?"
    print(f"PRE-FLIGHT: artifact X_sonic_variant={variant!r} genre_source={gsrc!r}")
    if variant != "mert":
        print(f"ABORT: live sonic variant is {variant!r}, not 'mert' — re-fold before calibrating.")
        return
    if args.grid == "artist":
        for a in ARTIST_CORPUS:
            for ax in ("genre", "sonic", "pace"):
                sweep_artist_axis(a, ax, args.length)
    else:
        sweep_artist_axis(args.artist, args.axis, args.length)


if __name__ == "__main__":
    main()
